# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 Mini (`inkling2_loop`) reference in PyTorch: one sequence, dense weights, llama.cpp semantics.

Mirrors llama.cpp master 29b52d3 src/models/inkling2.cpp + inkling2-loop.cpp op for op (the parity target of the
kaiLLM port); the vLLM model module reuses these functions. Weights use llama.cpp's GGUF names and numpy axis
order (a GGUF tensor with ne = [n_in, n_out] is [n_out, n_in] here: y = x @ W.T).

    prelude layers [0, start) -> `passes` x core [start, end) (each pass restarts from the prelude output; passes
    after the first blend messages from the previous pass into the q / k / v / ffn inputs) -> coda [end, L)
    -> attention-residual mix -> norm -> head -> softcap -> unpadded vocabulary
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

CTRL = ("q", "k", "v", "ffn")
DT = torch.float32               # compute dtype (tests may set torch.float64 with set_dtype)


def set_dtype(dtype: torch.dtype) -> None:
    global DT
    DT = dtype


@dataclass
class InklingHParams:
    n_layer: int
    n_embd: int
    n_head: int
    n_head_kv: int
    head_dim: int
    n_expert: int
    n_expert_used: int
    n_ff_exp: int
    vocab_padded: int
    vocab: int
    swa: list                       # per physical layer: True = sliding window
    n_swa: int
    n_rel: int
    rel_extent: int
    rel_clip: float
    qk_clip: float
    log_floor: float
    log_alpha: float
    rms_eps: float
    n_conv: int
    res_block: int
    situ_beta: float
    situ_lb: float
    w_scale: float
    softcap: float
    loop_start: int
    loop_end: int
    passes: int
    gate_rank: int

    @property
    def kv_dim(self) -> int:
        return self.head_dim * self.n_head_kv

    @property
    def conv_channels(self) -> int:
        """Per cache slot: K, V, attention output, FFN output (llama.cpp n_sconv_channels)."""
        return 2 * self.kv_dim + 2 * self.n_embd

    def slot(self, layer: int, pass_index: int) -> int:
        """llama.cpp cache slot of a layer visit (core passes > 0 go after the physical layers)."""
        if pass_index == 0 or not (self.loop_start <= layer < self.loop_end):
            return layer
        return self.n_layer + (pass_index - 1) * (self.loop_end - self.loop_start) + (layer - self.loop_start)

    @property
    def n_slots(self) -> int:
        return self.n_layer + (self.passes - 1) * (self.loop_end - self.loop_start)


def hparams_from_gguf_kv(kv: dict, prefix: str = "inkling2_loop") -> InklingHParams:
    g = lambda k, d=None: kv.get(f"{prefix}.{k}", d)          # noqa: E731
    heads = g("attention.head_count")
    heads_kv = g("attention.head_count_kv")
    heads = heads[0] if isinstance(heads, (list, tuple)) else heads
    heads_kv = heads_kv[0] if isinstance(heads_kv, (list, tuple)) else heads_kv
    return InklingHParams(
        n_layer=int(g("block_count")), n_embd=int(g("embedding_length")), n_head=int(heads),
        n_head_kv=int(heads_kv), head_dim=int(g("attention.key_length")), n_expert=int(g("expert_count")),
        n_expert_used=int(g("expert_used_count")), n_ff_exp=int(g("expert_feed_forward_length")),
        vocab_padded=int(g("vocab_size")), vocab=int(g("vocab_size_unpadded")),
        swa=[bool(x) for x in g("attention.sliding_window_pattern")], n_swa=int(g("attention.sliding_window")),
        n_rel=int(g("attention.relative_length")), rel_extent=int(g("attention.relative_extent")),
        rel_clip=float(g("attention.relative_clip")), qk_clip=float(g("attention.qk_clip")),
        log_floor=float(g("attention.log_scale_floor")), log_alpha=float(g("attention.log_scale_alpha")),
        rms_eps=float(g("attention.layer_norm_rms_epsilon")), n_conv=int(g("shortconv.l_cache")),
        res_block=int(g("attn_res.block_size")), situ_beta=float(g("activation.situ_beta")),
        situ_lb=float(g("activation.situ_linear_beta", 0.0)), w_scale=float(g("expert_weights_scale", 0.0)),
        softcap=float(g("final_logit_softcapping", 0.0)), loop_start=int(g("loop.start")),
        loop_end=int(g("loop.end")), passes=int(g("loop.count")), gate_rank=int(g("loop.gate_rank")))


# ----------------------------------------------------------------------------------------------------------------
# ops (float tensors, any dtype; computed in float32 like llama.cpp's CPU backend)

def rms(x: torch.Tensor, w: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    x = x.to(DT)
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y * w.to(DT) if w is not None else y


def clip_rows(x: torch.Tensor, scale: torch.Tensor | None, threshold: float) -> torch.Tensor:
    """llama.cpp clip(): optional per-channel scale, then rows (last axis) scaled down to an L2 norm <= threshold."""
    if scale is not None:
        x = x * scale
    norm = x.pow(2).sum(-1, keepdim=True).sqrt() + 1e-6
    return x * (threshold / norm).clamp(0.0, 1.0)


def mix(blocks: list, cur: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Attention over depth: softmax over the RMS-normalized states (blocks + cur) with learned queries.
    cur [T, D], query [nq, D] -> [T, nq, D]."""
    states = torch.stack(list(blocks) + [cur], dim=1).to(DT)            # [T, S, D]
    states = rms(states, None, 1e-6)
    scores = torch.einsum("tsd,qd->tqs", states, query.to(DT)) / math.sqrt(states.shape[-1])
    return torch.einsum("tqs,tsd->tqd", scores.softmax(-1), states)


def convolve(x: torch.Tensor, weight: torch.Tensor, history: torch.Tensor, dynamic: torch.Tensor | None = None,
             n_heads: int = 1):
    """Causal short conv with a residual: (x + conv(x) [+ dynamic taps]) / sqrt((2 if dynamic else 1) n_conv + 1).
    x [T, C]; weight [C, n_conv] (tap n_conv - 1 = current token); history [n_conv - 1, C] (oldest first);
    dynamic [T, n_heads * n_conv] (per head and token, tap lag l multiplies x[t - l]).  Returns (out, new history)."""
    n_conv = weight.shape[-1]
    T, C = x.shape
    full = torch.cat([history.to(DT), x.to(DT)], dim=0)                 # [n_conv - 1 + T, C]
    win = full.unfold(0, n_conv, 1)                                       # [T, C, n_conv]: win[t, c, k] = x[t - (n_conv-1) + k]
    out = (win * weight.to(DT)[None]).sum(-1)
    if dynamic is not None:
        taps = dynamic.to(DT).view(T, n_heads, n_conv)                   # [T, h, lag]
        lagged = win.flip(-1).view(T, n_heads, C // n_heads, n_conv)      # [T, h, c, lag]: x[t - lag]
        out = out + (lagged * taps[:, :, None, :]).sum(-1).reshape(T, C)
    scale = 1.0 / math.sqrt((2 if dynamic is not None else 1) * n_conv + 1)
    return (x.to(DT) + out) * scale, full[-(n_conv - 1):].clone()


def control(cur: torch.Tensor, message: torch.Tensor, down, gain, forget) -> torch.Tensor:
    """(1 - forget) cur + gain message, gain / forget = tanh(W (down [rms(cur); rms(message)]))."""
    g = torch.cat([rms(cur, None, 1e-6), rms(message, None, 1e-6)], dim=-1) @ down.to(DT).t()
    ga = torch.tanh(g @ gain.to(DT).t())
    fo = torch.tanh(g @ forget.to(DT).t())
    return cur.to(DT) - fo * cur.to(DT) + ga * message.to(DT)


def log_tau(positions: torch.Tensor, floor: float, alpha: float) -> torch.Tensor:
    """1 + alpha log(max(1, (pos + 1) / floor)), multiplying q and r of the full-attention layers."""
    t = ((positions.to(DT) + 1.0) / floor).clamp_min(1.0)
    return 1.0 + alpha * torch.log(t)


def attention(q, k, v, rel, bias, sink, q_pos, k_pos, scale: float, window: int | None, extent: int):
    """q [T, H, D], k / v [S, Hkv, D], rel [T, H, n_rel], bias [extent, n_rel] (numpy order of the layer's
    attn_rel_bias, GGUF dims [n_rel, extent]), sink [T, H].
    llama.cpp folds the relative bias into the mask (added after the scale): softmax(scale q.k + rel.bias[:, min(d,
    extent)] - sink + mask) with an extra zero sink column; bias is zero at distances >= extent.  Returns [T, H, D]."""
    T, H, D = q.shape
    S, Hkv, _ = k.shape
    g = H // Hkv
    kk = k.to(DT).repeat_interleave(g, dim=1)                            # [S, H, D]
    vv = v.to(DT).repeat_interleave(g, dim=1)
    dist = q_pos.view(T, 1) - k_pos.view(1, S)                            # [T, S]
    visible = dist >= 0
    if window is not None:
        visible &= dist < window
    idx = torch.where((dist >= 0) & (dist < extent), dist, torch.full_like(dist, extent))
    table = torch.cat([bias.to(DT).t(), bias.new_zeros(bias.shape[1], 1).to(DT)], dim=1)  # [n_rel, extent + 1]
    proj = torch.einsum("thr,re->the", rel.to(DT), table) - sink.to(DT)[..., None]      # [T, H, extent + 1]
    kq_b = torch.gather(proj, 2, idx.view(T, 1, S).expand(T, H, S))                       # [T, H, S]
    logits = torch.einsum("thd,shd->ths", q.to(DT), kk) * scale + kq_b
    logits = logits.masked_fill(~visible.view(T, 1, S), float("-inf"))
    full = torch.cat([logits, torch.zeros(T, H, 1, dtype=DT, device=logits.device)], dim=-1)
    p = full.softmax(-1)[..., :S]
    return torch.einsum("ths,shd->thd", p, vv)


def situ(gate: torch.Tensor, up: torch.Tensor, beta: float, lb: float) -> torch.Tensor:
    act = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if lb > 0:
        up = lb * torch.tanh(up / lb)
    return act * up


def route(x: torch.Tensor, gate_inp: torch.Tensor, probs_bias: torch.Tensor | None, top_k: int, w_scale: float):
    """Sigmoid router: selection by probs + bias, weights = probs normalized (+ 1e-20) x w_scale.
    Returns (ids [T, k], weights [T, k])."""
    probs = torch.sigmoid(x.to(DT) @ gate_inp.to(DT).t())
    sel = probs + probs_bias.to(DT) if probs_bias is not None else probs
    ids = sel.topk(top_k, dim=-1).indices
    w = probs.gather(1, ids)
    w = w / (w.sum(-1, keepdim=True) + 1e-20)
    if w_scale not in (0.0, 1.0):
        w = w * w_scale
    return ids, w


def moe(x, ids, weights, gate_up, down, hp: InklingHParams, gate_up_scale=None, down_scale=None):
    """gate_up [E, 2F, D] (rows [0, F) gate), down [E, D, F]; optional per-expert scales [E]."""
    T = x.shape[0]
    out = torch.zeros(T, hp.n_embd, dtype=DT)
    F = hp.n_ff_exp
    for t in range(T):
        for j in range(ids.shape[1]):
            e = int(ids[t, j])
            gu = gate_up[e].to(DT) @ x[t].to(DT)
            if gate_up_scale is not None:
                gu = gu * float(gate_up_scale[e])
            a = situ(gu[:F], gu[F:], hp.situ_beta, hp.situ_lb)
            y = down[e].to(DT) @ a
            if down_scale is not None:
                y = y * float(down_scale[e])
            out[t] += weights[t, j] * y
    return out


# ----------------------------------------------------------------------------------------------------------------
# whole model (one sequence from position 0; for parity tests)

@dataclass
class SlotState:
    keys: list = field(default_factory=list)
    values: list = field(default_factory=list)
    positions: list = field(default_factory=list)
    conv: torch.Tensor | None = None                 # [n_conv - 1, channels]


class InklingReference:
    def __init__(self, hp: InklingHParams, weights: dict):
        self.hp, self.w = hp, weights
        self.slots = [SlotState() for _ in range(hp.n_slots)]
        self.trace = None            # dict: intermediates named like llama.cpp's graph callbacks ("Qcur-<slot>")

    def _t(self, name, slot, value):
        if self.trace is not None:
            self.trace[f"{name}-{slot}"] = value.detach().clone()

    def reset(self):
        self.slots = [SlotState() for _ in range(self.hp.n_slots)]

    def W(self, name):
        return self.w[name]

    def has(self, name):
        return name in self.w

    def layer(self, il: int, ip: int, slot: int, cur, blocks, positions, tau, messages=None):
        """llama.cpp build_layer(il = slot, ip = physical layer)."""
        hp, p = self.hp, f"blk.{ip}."
        s = self.slots[slot]
        T = cur.shape[0]
        if s.conv is None:
            s.conv = torch.zeros(hp.n_conv - 1, hp.conv_channels, dtype=DT)
        conv = s.conv
        kvd, D = hp.kv_dim, hp.n_embd
        swa = hp.swa[ip]
        inp_sa = cur
        inputs = [cur, cur, cur]
        if self.has(p + "attn_res_query.weight"):
            q = self.W(p + "attn_res_query.weight").view(-1, D)
            mixed = mix(blocks, cur, q)
            inputs = [mixed[:, 0 if swa else j] for j in range(3)]
        if messages is not None:
            inputs = [control(inputs[j], messages[:, j], self.W(f"{p}loop_{CTRL[j]}_ctrl_down.weight"),
                              self.W(f"{p}loop_{CTRL[j]}_ctrl_gain.weight"),
                              self.W(f"{p}loop_{CTRL[j]}_ctrl_forget.weight")) for j in range(3)]
        if swa:
            w = self.W(p + "attn_norm.weight")
            inputs = [rms(x, w, hp.rms_eps) for x in inputs] if messages is not None else [rms(inputs[0], w, hp.rms_eps)] * 3
        else:
            inputs = [rms(inputs[j], self.W(p + f"attn_{c}_inp_norm.weight"), hp.rms_eps) for j, c in enumerate("qkv")]
        lin = lambda name, x: x.to(DT) @ self.W(p + name).to(DT).t()     # noqa: E731
        Q = lin("attn_q.weight", inputs[0])
        Kp = lin("attn_k.weight", inputs[1])
        Vp = lin("attn_v.weight", inputs[2])
        dk = None if swa else lin("attn_k_conv_proj.weight", inputs[1])
        dv = None if swa else lin("attn_v_conv_proj.weight", inputs[2])
        K, hk = convolve(Kp, self.W(p + "attn_k_conv.weight"), conv[:, :kvd], dk, hp.n_head_kv)
        V, hv = convolve(Vp, self.W(p + "attn_v_conv.weight"), conv[:, kvd:2 * kvd], dv, hp.n_head_kv)
        R = lin("attn_rel.weight", inputs[0])
        Q = clip_rows(Q.view(T, hp.n_head, hp.head_dim), self.W(p + "attn_q_clip.weight").view(hp.n_head, hp.head_dim),
                      hp.qk_clip)
        K = clip_rows(K.view(T, hp.n_head_kv, hp.head_dim), None, hp.qk_clip)
        V = rms(V.view(T, hp.n_head_kv, hp.head_dim), None, hp.rms_eps)
        R = clip_rows(R.view(T, hp.n_head, hp.n_rel), self.W(p + "attn_rel_clip.weight").view(hp.n_head, hp.n_rel),
                      hp.rel_clip)
        if not swa:
            Q = Q * tau.view(T, 1, 1)
            R = R * tau.view(T, 1, 1)
        sinks = lin("attn_sink_proj.weight", inputs[0])                    # [T, H]
        for name, value in (("Qcur", Q), ("Kcur", K), ("Vcur", V), ("Rcur", R), ("sinks", sinks)):
            self._t(name, slot, value)
        s.keys.append(K)
        s.values.append(V)
        s.positions.append(positions)
        Kall, Vall, Pall = torch.cat(s.keys), torch.cat(s.values), torch.cat(s.positions)
        att = attention(Q, Kall, Vall, R, self.W(p + "attn_rel_bias.weight"), sinks, positions, Pall,
                        1.0 / math.sqrt(hp.head_dim), hp.n_swa if swa else None, hp.n_swa if swa else hp.rel_extent)
        att = att.reshape(T, -1) * torch.sigmoid(lin("attn_gate.weight", inputs[0]))
        out, ho = convolve(lin("attn_output.weight", att), self.W(p + "attn_post_conv.weight"),
                           conv[:, 2 * kvd:2 * kvd + D])
        self._t("attn_out", slot, out)
        if ip % hp.res_block == 0:
            blocks.append(inp_sa)
            cur = out
        else:
            cur = inp_sa.to(DT) + out
        ffn_inp = cur
        x = mix(blocks, cur, self.W(p + "ffn_res_query.weight").view(1, D))[:, 0]
        if messages is not None:
            x = control(x, messages[:, 3], self.W(f"{p}loop_ffn_ctrl_down.weight"),
                        self.W(f"{p}loop_ffn_ctrl_gain.weight"), self.W(f"{p}loop_ffn_ctrl_forget.weight"))
        x = rms(x, self.W(p + "ffn_norm.weight"), hp.rms_eps)
        ids, wts = route(x, self.W(p + "ffn_gate_inp.weight"), self.w.get(p + "exp_probs_b.bias"),
                         hp.n_expert_used, hp.w_scale)
        y = moe(x, ids, wts, self.W(p + "ffn_gate_up_exps.weight"), self.W(p + "ffn_down_exps.weight"), hp,
                self.w.get(p + "ffn_gate_up_exps.scale"), self.w.get(p + "ffn_down_exps.scale"))
        self._t("ffn_moe_topk", slot, ids)
        self._t("ffn_moe_out", slot, y)
        y, hf = convolve(y, self.W(p + "ffn_post_conv.weight"), conv[:, 2 * kvd + D:])
        s.conv = torch.cat([hk, hv, ho, hf], dim=1)
        self._t("ffn_inp", slot, ffn_inp)
        self._t("ffn_out", slot, y)
        self._t("l_out", slot, ffn_inp + y)
        return ffn_inp + y

    def forward(self, tokens: torch.Tensor, start: int = 0) -> torch.Tensor:
        """Logits [T, vocab] for tokens at positions start.. (call in order: chunks = prefill = decode)."""
        hp = self.hp
        T = tokens.shape[0]
        positions = torch.arange(start, start + T)
        tau = log_tau(positions, hp.log_floor, hp.log_alpha)
        cur = self.W("token_embd.weight")[tokens.long()].to(DT)
        cur = rms(cur, self.W("token_embd_norm.weight"), hp.rms_eps)
        blocks: list = []
        for il in range(hp.loop_start):
            cur = self.layer(il, il, il, cur, blocks, positions, tau)
        prelude, prelude_blocks = cur, list(blocks)
        for p in range(hp.passes):
            messages = None
            if p > 0:
                bank = blocks[len(prelude_blocks) + 1:]
                bank_out = cur
                cur, blocks = prelude, list(prelude_blocks)
            for ip in range(hp.loop_start, hp.loop_end):
                if p > 0:
                    messages = mix(bank, bank_out, self.W(f"blk.{ip}.loop_msg_query.weight").view(4, hp.n_embd))
                cur = self.layer(hp.slot(ip, p), ip, hp.slot(ip, p), cur, blocks, positions, tau, messages)
        for il in range(hp.loop_end, hp.n_layer):
            cur = self.layer(il, il, il, cur, blocks, positions, tau)
        x = mix(blocks, cur, self.W("output_res_query.weight").view(1, hp.n_embd))[:, 0]
        x = rms(x, self.W("output_norm.weight"), hp.rms_eps)
        logits = x @ self.W("output.weight").to(DT).t()
        if hp.softcap:
            logits = hp.softcap * torch.tanh(logits / hp.softcap)
        return logits[:, :hp.vocab]
