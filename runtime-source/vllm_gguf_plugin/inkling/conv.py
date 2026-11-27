# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 short convolutions on vLLM conv states: one short-conv (MambaBase) state per layer visit, channels =
K | V | attention output | FFN output (llama.cpp n_sconv_channels), n_conv - 1 history columns.

Each conv is `(x + conv(x) [+ dynamic taps]) / sqrt((2 if dynamic else 1) n_conv + 1)` (ref.convolve). The batch
follows vLLM's mamba order (decode requests first, then prefills); prefills without an initial state start from
zeros.  Decode-only batches: one Triton kernel per conv (triton_conv.py); other batches: one batched torch pass.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch
from torch import nn

from .ref import convolve

# "batched": decode-only batches in one Triton kernel per conv (inkling/triton_conv.py), others batched in torch;
# "torch": batched torch for every batch; "loop": the per-request reference path
_CONV_MODE = os.getenv("KAILLM_INKLING_CONV", "batched")
BATCHED = _CONV_MODE in ("batched", "torch")
TRITON_DECODE = _CONV_MODE == "batched"


class InklingConvState(nn.Module):
    def __init__(self, channels: int, n_conv: int, prefix: str, model_config, cache_config):
        super().__init__()
        from vllm.config import get_current_vllm_config
        self.channels, self.n_conv = channels, n_conv
        self.prefix, self.model_config, self.cache_config = prefix, model_config, cache_config
        self.kv_cache = (torch.tensor([]),)
        config = get_current_vllm_config()
        self.num_spec_tokens = config.num_speculative_tokens
        ctx = config.compilation_config.static_forward_context
        if prefix in ctx:
            raise ValueError(f"Duplicate layer name: {prefix}")
        ctx[prefix] = self

    def get_state_shape(self):
        from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
        return MambaStateShapeCalculator.short_conv_state_shape(tp_world_size=1, intermediate_size=self.channels,
                                                               conv_kernel=self.n_conv, num_spec=self.num_spec_tokens)

    def get_state_dtype(self):
        from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator
        return MambaStateDtypeCalculator.short_conv_state_dtype(self.model_config.dtype,
                                                               self.cache_config.mamba_cache_dtype)

    @property
    def mamba_type(self):
        from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
        return MambaAttentionBackendEnum.SHORT_CONV

    def get_attn_backend(self):
        if os.getenv("KAILLM_INKLING_FULL_GRAPHS") == "1" or os.getenv("KAILLM_BENCHMARK_GRAPH_DIR"):
            from .conv_graph import conv_graph_backend
            return conv_graph_backend()
        return super().get_attn_backend()

    def metadata(self):
        from vllm.forward_context import get_forward_context
        md = get_forward_context().attn_metadata
        return None if md is None else (md.get(self.prefix) if isinstance(md, dict) else md)

    def states(self) -> torch.Tensor:
        """[slots, channels, n_conv - 1] view of the conv state."""
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
        st = self.kv_cache[0]
        return st if is_conv_state_dim_first() else st.transpose(-1, -2)


_CONV_CLS = None


def make_conv_state(*args, **kwargs) -> InklingConvState:
    global _CONV_CLS
    if _CONV_CLS is None:
        from vllm.model_executor.layers.mamba.abstract import MambaBase
        _CONV_CLS = type("InklingConvStateLayer", (InklingConvState, MambaBase), {})
    return _CONV_CLS(*args, **kwargs)


def request_spans(md):
    """[(token start, token end, state index, has initial state)] in batch order (decodes first)."""
    spans = []
    nd = int(md.num_decodes)
    if nd:
        idx_d = md.state_indices_tensor_d.reshape(nd, -1)[:, 0].tolist()
        qd = md.query_start_loc_d.tolist() if md.query_start_loc_d is not None else list(range(nd + 1))
        for i in range(nd):
            spans.append((qd[i], qd[i + 1], idx_d[i], True))
    if md.num_prefills:
        base = int(md.num_decode_tokens)
        qp = md.query_start_loc_p.tolist()
        idx_p = md.state_indices_tensor_p.reshape(md.num_prefills, -1)[:, 0].tolist()
        init = md.has_initial_states_p.tolist() if md.has_initial_states_p is not None else [False] * md.num_prefills
        for j in range(md.num_prefills):
            spans.append((base + qp[j], base + qp[j + 1], idx_p[j], bool(init[j])))
    return spans


def batch_plan(md, T: int, n_conv: int, device):
    """Per-batch gather plan (device tensors, no host sync), cached on the metadata object: request of each token,
    window indices into [history (R x (n_conv - 1) rows) ; chunk (T rows)] and the rows of the new history."""
    plan = getattr(md, "_inkling_conv_plan", None)
    if plan is not None and plan[0] == (T, n_conv):
        return plan[1]
    nd, npf = int(md.num_decodes), int(md.num_prefills)
    ndt = int(md.num_decode_tokens)
    H1 = n_conv - 1
    starts, lens, sidx, init = [], [], [], []
    if nd:
        qd = md.query_start_loc_d
        if qd is None:
            starts.append(torch.arange(nd, device=device))
            lens.append(torch.ones(nd, dtype=torch.long, device=device))
        else:
            starts.append(qd[:nd].long())
            lens.append(torch.diff(qd[:nd + 1]).long())
        sidx.append(md.state_indices_tensor_d.reshape(nd, -1)[:, 0].long())
        init.append(torch.ones(nd, dtype=torch.bool, device=device))
    if npf:
        qp = md.query_start_loc_p.long()
        starts.append(ndt + qp[:-1])
        lens.append(qp[1:] - qp[:-1])
        sidx.append(md.state_indices_tensor_p.reshape(npf, -1)[:, 0].long())
        hi = md.has_initial_states_p
        init.append(hi.bool() if hi is not None else torch.zeros(npf, dtype=torch.bool, device=device))
    start, qlen, sidx, init = (torch.cat(v) for v in (starts, lens, sidx, init))
    R = start.shape[0]
    req = torch.repeat_interleave(torch.arange(R, device=device), qlen, output_size=T)
    t = torch.arange(T, device=device)
    lp = t - start[req]
    k = torch.arange(n_conv, device=device)
    time = lp[:, None] - H1 + k[None, :]                                   # [T, n_conv]
    win = torch.where(time >= 0, R * H1 + t[:, None] - H1 + k[None, :], req[:, None] * H1 + H1 + time)
    j = torch.arange(H1, device=device)
    ntime = qlen[:, None] - H1 + j[None, :]                                # [R, n_conv - 1]
    new = torch.where(ntime >= 0, R * H1 + start[:, None] + ntime, torch.arange(R, device=device)[:, None] * H1 + H1 + ntime)
    plan = dict(R=R, sidx=sidx, init=init, win=win, new=new, start=start, lengths=qlen)
    md._inkling_conv_plan = ((T, n_conv), plan)
    return plan


def conv_step_batched(state: InklingConvState, x: torch.Tensor, weight: torch.Tensor, offset: int, md,
                      dynamic: torch.Tensor | None = None, n_heads: int = 1, out_dtype=None) -> torch.Tensor:
    """All requests of the batch in one pass (ref.convolve semantics)."""
    import math
    T = int(md.num_decode_tokens) + int(md.num_prefill_tokens)
    C = x.shape[1]
    n_conv = state.n_conv
    H1 = n_conv - 1
    plan = batch_plan(md, T, n_conv, x.device)
    st = state.states()
    storage = st[plan["sidx"], offset:offset + C, :].transpose(1, 2)
    extended = storage.shape[1] > H1
    if extended:
        # The first query in each decode is always committed. The other queries
        # are proposals; vLLM supplies the accepted prefix on the following step.
        history_start = torch.zeros(plan["R"], device=x.device, dtype=torch.long)
        accepted = md.num_accepted_tokens
        if md.num_decodes and accepted is not None:
            history_start[:md.num_decodes] = (accepted[:md.num_decodes].long() - 1).clamp_min(0)
        rows = torch.arange(plan["R"], device=x.device)[:, None]
        hist = storage[rows, history_start[:, None] + torch.arange(H1, device=x.device)].float()
    else:
        hist = storage.float()
    hist = hist * plan["init"][:, None, None]
    xs = x[:T].float()
    buf = torch.cat([hist.reshape(-1, C), xs], dim=0)
    win = buf[plan["win"]]                                                       # [T, n_conv, C]
    if extended and md.num_decode_tokens and st.dtype != torch.float32:
        # A serial decode stores each prior input at state precision, including
        # past tokens from the current verify block. The current input stays FP32.
        old_taps = win[:md.num_decode_tokens, :-1]
        old_taps.copy_(old_taps.to(st.dtype).float())
    out = (win * weight.float().t()[None]).sum(1)
    if dynamic is not None:
        taps = dynamic[:T].float().view(T, n_heads, n_conv)                      # [T, h, lag]
        lagged = win.flip(1).view(T, n_conv, n_heads, C // n_heads)              # [T, lag, h, c]: x[t - lag]
        out = out + (lagged * taps.transpose(1, 2)[..., None]).sum(1).reshape(T, C)
    scale = 1.0 / math.sqrt((2 if dynamic is not None else 1) * n_conv + 1)
    res = torch.empty_like(x, dtype=out_dtype or x.dtype)
    res[:T] = ((xs + out) * scale).to(res.dtype)
    if x.shape[0] > T:
        res[T:] = 0
    if extended:
        # For a verify chunk store history after the anchor plus all draft inputs.
        # Prefill has already committed its entire chunk. Put that history at zero.
        columns = torch.arange(storage.shape[1], device=x.device)[None, :]
        consumed = torch.where(rows < md.num_decodes, 1, plan["lengths"][:, None])
        relative = columns + consumed - H1
        indices = torch.where(relative < 0, rows * H1 + H1 + relative,
                              plan["R"] * H1 + plan["start"][:, None] + relative)
        valid = relative < plan["lengths"][:, None]
        new = buf[torch.where(valid, indices, 0)]
        new = torch.where(valid[..., None], new, 0)
    else:
        new = buf[plan["new"]]
    st[plan["sidx"], offset:offset + C, :] = new.transpose(1, 2).to(st.dtype)
    return res


def conv_step(state: InklingConvState, x: torch.Tensor, weight: torch.Tensor, offset: int,
              dynamic: torch.Tensor | None = None, n_heads: int = 1, out_dtype=None) -> torch.Tensor:
    """x [T, C] (any float dtype; float32 math): one conv of a slot on channels [offset, offset + C) of its state;
    returns [T, C] in out_dtype (default x.dtype)."""
    T, C = x.shape
    odt = out_dtype or x.dtype
    md = state.metadata()
    if md is None:                                       # profile run: one sequence from zeros
        hist = torch.zeros(state.n_conv - 1, C, dtype=torch.float32, device=x.device)
        return convolve(x, weight, hist, dynamic, n_heads)[0].to(odt)
    from .conv_graph import ConvGraphMetadata, conv_mixed
    if isinstance(md, ConvGraphMetadata):
        if state.num_spec_tokens:
            from .conv_graph import conv_spec_mixed
            return conv_spec_mixed(state, x, weight, offset, md, dynamic, n_heads, odt)
        if x.shape[0] == 1:
            from .triton_conv import conv_decode
            slots = md.slots()
            return conv_decode(state.states(), x, weight, offset, slots, dynamic, n_heads, odt,
                               seq_lens=md.seq_lens, offsets=md.offsets)
        return conv_mixed(state, x, weight, offset, md, dynamic, n_heads, odt)
    if state.states().shape[-1] > state.n_conv - 1:
        if TRITON_DECODE and x.is_cuda and md.num_decodes:
            from .triton_conv import conv_verify
            if not md.num_prefills:
                return conv_verify(state.states(), x, weight, offset, md, dynamic, n_heads, odt)
            # The scheduler packs decode requests first. Use their ordinary target
            # arithmetic even when a new request shares this verification step.
            split = int(md.num_decode_tokens)
            decode = conv_verify(state.states(), x[:split], weight, offset, md,
                                 dynamic[:split] if dynamic is not None else None, n_heads, odt)
            prefill = getattr(md, '_inkling_prefill_only', None)
            if prefill is None:
                prefill = SimpleNamespace(num_decodes=0, num_decode_tokens=0, num_prefills=md.num_prefills,
                                          num_prefill_tokens=md.num_prefill_tokens, query_start_loc_p=md.query_start_loc_p,
                                          state_indices_tensor_p=md.state_indices_tensor_p, has_initial_states_p=md.has_initial_states_p,
                                          num_accepted_tokens=None)
                md._inkling_prefill_only = prefill
            rest = conv_step_batched(state, x[split:], weight, offset, prefill,
                                     dynamic[split:] if dynamic is not None else None, n_heads, odt)
            return torch.cat((decode, rest))
        return conv_step_batched(state, x, weight, offset, md, dynamic, n_heads, odt)
    if BATCHED:
        nd = int(md.num_decodes)
        if (TRITON_DECODE and x.is_cuda and not int(md.num_prefills) and nd == x.shape[0]
                and int(md.num_decode_tokens) == nd):
            from .triton_conv import conv_decode
            sidx = md.state_indices_tensor_d.reshape(nd, -1)[:, 0]
            return conv_decode(state.states(), x, weight, offset, sidx, dynamic, n_heads, odt)
        return conv_step_batched(state, x, weight, offset, md, dynamic, n_heads, odt)
    st = state.states()
    out = torch.empty_like(x, dtype=odt)
    for a, b, idx, init in request_spans(md):
        if b <= a:
            continue
        cur = st[idx, offset:offset + C, :]                # [C, n_conv - 1], oldest first
        hist = cur.t().float() if init else torch.zeros(state.n_conv - 1, C, dtype=torch.float32, device=x.device)
        y, new = convolve(x[a:b], weight, hist, None if dynamic is None else dynamic[a:b], n_heads)
        out[a:b] = y.to(odt)
        st[idx, offset:offset + C, :] = new.t().to(st.dtype)
    return out
