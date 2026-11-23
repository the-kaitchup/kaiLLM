# SPDX-License-Identifier: Apache-2.0
"""Hy4 flash attention for kaiLLM: one sparse MLA backend for every layer.

As in Tencent's vLLM fork, every attention layer attends to a per-token list of token positions
over the paged MLA latent cache (kv_lora + rope per token): sliding-window layers pass their causal
window, indexed layers the compressed indexer's picks (all previous tokens below
`candidate_topk`). Learnable per-head sinks join the softmax as one extra logit.

Two implementations, selected by KAILLM_HY4_SPARSE_MLA (auto | torch | triton):
* torch: gather + softmax in PyTorch (reference; standard kernels only);
* triton: vLLM's Triton bf16 sparse MLA kernel (vllm/v1/attention/ops/xpu_mla_sparse.py), with the
  sinks applied from its natural-log LSE: out * sigmoid(lse - sink). `auto` uses triton on sm >= 9.

The pure functions at the top (indices, reference attention, compressed-indexer selection) carry
the semantics and are tested on CPU against transformers PR 50903.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch
from torch import nn

IMPL = os.environ.get("KAILLM_HY4_SPARSE_MLA", "auto")


# ----------------------------------------------------------------------------------------------
# pure functions (CPU-testable)

def window_indices(positions: torch.Tensor, window: int, width: int) -> torch.Tensor:
    """Causal window [pos - window + 1, pos] per token, -1 padded to `width`."""
    cols = torch.arange(width, device=positions.device)
    idx = positions.view(-1, 1) - window + 1 + cols
    live = (cols < window) & (idx >= 0)
    return torch.where(live, idx, -1).to(torch.int32)


def dense_indices(positions: torch.Tensor, width: int) -> torch.Tensor:
    """All previous tokens [0, pos] (indexed layers below candidate_topk), -1 padded."""
    cols = torch.arange(width, device=positions.device)
    return torch.where(cols <= positions.view(-1, 1), cols, -1).to(torch.int32)


def to_global_rows(topk: torch.Tensor, req_id: torch.Tensor, block_table: torch.Tensor,
                   block_size: int, stride_rows: int) -> torch.Tensor:
    """Per-request token positions -> rows of the flat paged cache (-1 kept)."""
    pos = topk.long()
    valid = pos >= 0
    pos = pos.clamp_min(0)
    col = (pos // block_size).clamp_max(block_table.shape[1] - 1)
    blk = block_table[req_id.long().view(-1, 1), col].long()
    rows = blk * stride_rows + pos % block_size
    return torch.where(valid, rows, -1).to(torch.int32)


def reference_sparse_mla(q: torch.Tensor, kv_rows: torch.Tensor, rows: torch.Tensor, scale: float,
                         d_v: int, sinks: torch.Tensor | None, chunk: int = 64) -> torch.Tensor:
    """q [T, H, D], kv_rows [R, D], rows [T, W] (-1 = none), sinks [H] -> [T, H, d_v]."""
    T, H, _ = q.shape
    out = q.new_empty(T, H, d_v)
    for s in range(0, T, chunk):
        idx = rows[s:s + chunk].long()
        valid = idx >= 0
        kv = kv_rows[idx.clamp_min(0)].float()                       # [t, W, D]
        scores = torch.einsum("thd,twd->thw", q[s:s + chunk].float(), kv) * scale
        scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
        if sinks is not None:
            sink = sinks.float().view(1, H, 1).expand(scores.shape[0], H, 1)
            scores = torch.cat([scores, sink], dim=-1)
        p = torch.nan_to_num(scores.softmax(-1))
        if sinks is not None:
            p = p[..., :-1]
        out[s:s + chunk] = torch.einsum("thw,twd->thd", p, kv[..., :d_v]).to(q.dtype)
    return out


def triton_sparse_mla(q, kv_rows, rows, scale, d_v, rope_dim, sinks):
    from vllm.v1.attention.ops.xpu_mla_sparse import triton_bf16_mla_sparse_interface
    T = q.shape[0]
    out, _, lse = triton_bf16_mla_sparse_interface(
        q.contiguous(), kv_rows.view(-1, 1, kv_rows.shape[-1]), rows.view(T, 1, -1).contiguous(),
        sm_scale=scale, d_v=d_v, block_dpe=rope_dim)
    if sinks is not None:
        out = out * torch.sigmoid(lse.float() - sinks.float().view(1, -1)).unsqueeze(-1).to(out.dtype)
    return out


def rope_interleaved(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """GPT-J style (pairs 2i, 2i+1) rotary embedding on the last dim of x [..., d]."""
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    ang = positions.float().view(*positions.shape, 1) * inv
    while ang.dim() < x.dim():
        ang = ang.unsqueeze(-2)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
    return out.to(x.dtype)


def rms(x: torch.Tensor, w: torch.Tensor | None, eps: float) -> torch.Tensor:
    y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (y * w.float() if w is not None else y).to(x.dtype)


@dataclass
class IndexerWeights:
    query_projection: torch.Tensor | None  # [H_i * D_i, q_lora]; None when select_groups gets q
    head_weight_projection: torch.Tensor  # [H_i, hidden]
    token_key_norm: torch.Tensor          # [D_i]
    key_value_projection: torch.Tensor    # [D_i, D_i] fp32
    key_gate_projection: torch.Tensor     # [D_i, D_i] fp32
    intra_group_ape: torch.Tensor         # [ratio, D_i] fp32
    key_norm: torch.Tensor                # [D_i] fp32
    num_heads: int
    head_dim: int
    rope_dim: int
    ratio: int
    group_topk: int
    theta: float
    eps: float


def pool_groups(token_keys: torch.Tensor, w: IndexerWeights) -> torch.Tensor:
    """[N, D_i] cached token keys (already normed) -> [N // ratio, D_i] group keys."""
    g = token_keys.shape[0] // w.ratio
    grouped = token_keys[: g * w.ratio].float().view(g, w.ratio, w.head_dim)
    values = grouped @ w.key_value_projection.float().t()
    gates = grouped @ w.key_gate_projection.float().t() + w.intra_group_ape.float()
    pooled = (values * gates.softmax(dim=1)).sum(dim=1)
    return rms(pooled, w.key_norm, w.eps).to(token_keys.dtype)


def topk_groups(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the k best groups per row; exact ties (ReLU zeros are common with few indexer
    heads) go to the later group, so the pick does not depend on the batch or chunk layout."""
    G = logits.shape[-1]
    order = torch.sort(logits.flip(-1), dim=-1, descending=True, stable=True).indices[..., :k]
    return G - 1 - order


def select_groups(hidden: torch.Tensor, q_resid: torch.Tensor, positions: torch.Tensor,
                  group_keys: torch.Tensor, w: IndexerWeights, q: torch.Tensor | None = None) -> torch.Tensor:
    """One request: token indices per query [T, group_topk * ratio] (-1 padded), transformers
    semantics: the (group_topk - 1) best complete groups before the query's own group, plus it."""
    T = hidden.shape[0]
    if q is None:   # q: the indexer query projection's output [T, H_i * D_i], when the caller ran the layer
        q = q_resid.float() @ w.query_projection.float().t()
    q = q.float().view(T, w.num_heads, w.head_dim)
    r = w.head_dim - w.rope_dim  # rope on the trailing rope_dim channels (transformers layout)
    q = torch.cat([q[..., :r], rope_interleaved(q[..., r:], positions.view(T, 1).expand(T, w.num_heads), w.theta)],
                  dim=-1)
    G = group_keys.shape[0]
    gpos = torch.arange(G, device=hidden.device) * w.ratio
    k = torch.cat([group_keys[:, :r].float(), rope_interleaved(group_keys[:, r:].float(), gpos, w.theta)], dim=-1)
    hw = (hidden.float() @ w.head_weight_projection.float().t()) * (w.num_heads ** -0.5 * w.head_dim ** -0.5)
    scores = torch.relu(torch.einsum("thd,gd->thg", q, k))
    logits = torch.einsum("th,thg->tg", hw, scores)
    own = (positions.view(-1) // w.ratio).long()
    logits = logits.masked_fill(torch.arange(G, device=hidden.device)[None, :] >= own[:, None], float("-inf"))
    k_sel = min(w.group_topk - 1, G)
    picked = topk_groups(logits, k_sel) if k_sel > 0 else own.new_empty(T, 0)
    valid_pick = torch.gather(logits, 1, picked) > float("-inf") if k_sel > 0 else torch.zeros_like(picked, dtype=torch.bool)
    groups = torch.cat([torch.where(valid_pick, picked, -1), own[:, None]], dim=-1)   # [T, <= group_topk]
    offs = torch.arange(w.ratio, device=hidden.device)
    tok = groups[..., None] * w.ratio + offs
    tok = torch.where((groups[..., None] >= 0) & (tok <= positions.view(-1, 1, 1)), tok, -1).flatten(1)
    width = w.group_topk * w.ratio
    out = torch.full((T, width), -1, dtype=torch.int32, device=hidden.device)
    out[:, : tok.shape[1]] = tok.to(torch.int32)
    return out


# ----------------------------------------------------------------------------------------------
# vLLM backend (imports deferred so the functions above stay importable without vLLM)

def _backend_classes():
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.sparse_mla_attention import SharedTopkIndicesBuffer
    from vllm.utils.torch_utils import np_to_pinned_tensor
    from vllm.v1.attention.backend import (AttentionBackend, AttentionCGSupport, AttentionLayer,
                                           AttentionMetadata, AttentionMetadataBuilder,
                                           CommonAttentionMetadata, MLAAttentionImpl)
    from vllm.v1.attention.backends.mla.sparse_utils import align_blocks_to_rows, flat_kv_row_view
    from vllm.v1.kv_cache_interface import AttentionSpec, is_quantized_kv_cache

    @dataclass
    class Hy4SparseMLAMetadata(AttentionMetadata):
        num_reqs: int
        max_query_len: int
        max_seq_len: int
        num_actual_tokens: int
        query_start_loc: torch.Tensor
        slot_mapping: torch.Tensor
        block_table: torch.Tensor
        req_id_per_token: torch.Tensor
        seq_lens: torch.Tensor
        block_size: int = 1
        topk_tokens: int = 2048
        # every token goes through forward_mqa (see the XPU sparse backend for why)
        num_decodes: int = 0
        num_prefills: int = 0
        num_decode_tokens: int = 0

    class Hy4SparseMLAMetadataBuilder(AttentionMetadataBuilder[Hy4SparseMLAMetadata]):
        _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

        def __init__(self, kv_cache_spec: AttentionSpec, layer_names, vllm_config: VllmConfig, device):
            self.kv_cache_spec = kv_cache_spec
            cfg = vllm_config.model_config.hf_text_config
            self.topk_tokens = int(getattr(cfg, "compressed_indexer_candidate_topk", 2048))
            self.req_id_buf = torch.zeros(vllm_config.scheduler_config.max_num_batched_tokens,
                                          dtype=torch.int32, device=device)

        def build(self, common_prefix_len, common_attn_metadata: CommonAttentionMetadata, fast_build=False):
            m = common_attn_metadata
            starts = np.asarray(m.query_start_loc_cpu, dtype=np.int32)
            req = np.repeat(np.arange(len(starts) - 1, dtype=np.int32), np.diff(starts))
            self.req_id_buf.fill_(0)
            self.req_id_buf[: req.shape[0]].copy_(np_to_pinned_tensor(req), non_blocking=True)
            return Hy4SparseMLAMetadata(
                num_reqs=m.num_reqs, max_query_len=m.max_query_len, max_seq_len=m.max_seq_len,
                num_actual_tokens=m.num_actual_tokens, query_start_loc=m.query_start_loc,
                slot_mapping=m.slot_mapping, block_table=m.block_table_tensor,
                req_id_per_token=self.req_id_buf[: m.num_actual_tokens], seq_lens=m.seq_lens,
                block_size=self.kv_cache_spec.block_size, topk_tokens=self.topk_tokens,
                num_decodes=m.num_reqs, num_prefills=0, num_decode_tokens=m.num_actual_tokens)

    class Hy4SparseMLAImpl(MLAAttentionImpl[Hy4SparseMLAMetadata], SharedTopkIndicesBuffer):
        is_sparse = True
        # every token takes the top-k MQA path: no dense-MHA prefill (sinks + per-token index lists)
        supports_dense_mha_prefill = False

        def __init__(self, num_heads, head_size, scale, num_kv_heads, alibi_slopes, sliding_window,
                     kv_cache_dtype, logits_soft_cap, attn_type, kv_sharing_target_layer_name,
                     topk_indices_buffer=None, indexer=None, hy4_sinks=None, **mla_args):
            self.num_heads, self.head_size, self.scale = num_heads, head_size, float(scale)
            self.num_kv_heads, self.kv_cache_dtype = num_kv_heads, kv_cache_dtype
            self.kv_lora_rank = mla_args["kv_lora_rank"]
            self.rope_dim = mla_args.get("qk_rope_head_dim", head_size - self.kv_lora_rank)
            self.sinks = hy4_sinks
            self.init_topk_indices_buffer(indexer, topk_indices_buffer)

        def record_logical_topk_ready(self) -> None:
            """No index groups here: each layer writes its own index list before attention."""

        def _use_triton(self, q, width: int) -> bool:
            # vLLM's Triton sparse-MLA kernel: d_v = 512 (the real model's latent), width % 16 == 0
            ok = (q.is_cuda and self.kv_lora_rank == 512 and width % 16 == 0
                  and q.dtype in (torch.bfloat16, torch.float16))
            if IMPL == "triton":
                if not ok:
                    raise ValueError("KAILLM_HY4_SPARSE_MLA=triton needs CUDA, BF16/FP16 activations, "
                                     "kv_lora_rank 512 and an index width that is a multiple of 16")
                return True
            if IMPL == "torch":
                return False
            return ok and torch.cuda.get_device_capability(q.device)[0] >= 9

        def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata: Hy4SparseMLAMetadata, layer: AttentionLayer):
            if is_quantized_kv_cache(self.kv_cache_dtype):
                raise NotImplementedError("Hy4 sparse MLA: FP8 KV cache not supported yet")
            if isinstance(q, tuple):
                q = torch.cat(q, dim=-1)
            T = q.shape[0]
            topk = self.topk_indices_buffer[:T]
            kv_rows, stride_rows = flat_kv_row_view(kv_c_and_k_pe_cache, attn_metadata.block_size)
            rows = to_global_rows(topk, attn_metadata.req_id_per_token, attn_metadata.block_table,
                                  attn_metadata.block_size, stride_rows)
            sinks = self.sinks if self.sinks is not None else getattr(layer, "hy4_sinks", None)
            if self._use_triton(q, rows.shape[-1]):
                out = triton_sparse_mla(q, kv_rows, rows, self.scale, self.kv_lora_rank, self.rope_dim, sinks)
            else:
                out = reference_sparse_mla(q, kv_rows.view(-1, kv_rows.shape[-1]), rows, self.scale,
                                           self.kv_lora_rank, sinks)
            return out[:, : self.num_heads, :], None

    class Hy4SparseMLABackend(AttentionBackend):
        supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16, torch.float32]
        supported_kv_cache_dtypes: ClassVar[list] = ["auto", "float16", "bfloat16"]

        @staticmethod
        def get_name() -> str:
            return "HY4_SPARSE_MLA"

        @classmethod
        def customize_spec(cls, spec):
            return align_blocks_to_rows(spec)

        @staticmethod
        def get_metadata_cls():
            return Hy4SparseMLAMetadata

        @staticmethod
        def get_builder_cls():
            return Hy4SparseMLAMetadataBuilder

        @staticmethod
        def get_impl_cls():
            return Hy4SparseMLAImpl

        @classmethod
        def is_mla(cls) -> bool:
            return True

        @classmethod
        def is_sparse(cls) -> bool:
            return True

        @classmethod
        def get_supported_head_sizes(cls) -> list[int]:
            return []

        @classmethod
        def supports_head_size(cls, head_size: int) -> bool:
            return True

        @classmethod
        def supports_compute_capability(cls, capability) -> bool:
            return True

    return Hy4SparseMLABackend


_BACKEND = None


def hy4_sparse_mla_backend():
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = _backend_classes()
    return _BACKEND


class Hy4WindowIndexer(nn.Module):
    """Sliding layers: the causal window as the sparse index list."""

    def __init__(self, window: int, topk_indices_buffer: torch.Tensor):
        super().__init__()
        self.window = window
        self.topk_indices_buffer = topk_indices_buffer
        self.topk_tokens = topk_indices_buffer.shape[1]

    def forward(self, hidden_states, q_c, positions, rope=None):
        T = hidden_states.shape[0]
        self.topk_indices_buffer[:T].copy_(window_indices(positions[:T], self.window, self.topk_tokens))
        return self.topk_indices_buffer[:T]
