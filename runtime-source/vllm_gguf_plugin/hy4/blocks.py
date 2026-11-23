# SPDX-License-Identifier: Apache-2.0
"""Hy4 flash building blocks as pure functions (semantics of transformers PR 50903).

Used by the vLLM model (vllm-overrides/.../models/hy_v4_flash.py) and by the CPU checks. Tensors
follow PyTorch's [out, in] weight convention.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def unweighted_rsqrt(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float | None) -> torch.Tensor:
    """RMSNorm; eps=None means torch.nn.RMSNorm's default (finfo(dtype).eps)."""
    xf = x.float()
    e = torch.finfo(xf.dtype).eps if eps is None else eps
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + e)
    if weight is not None:
        y = y * weight.float()
    return y.to(x.dtype)


# ---------------------------------------------------------------------------------- hyper-connections

def hc_pre(streams: torch.Tensor, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor,
           eps_rms: float, hc_eps: float, magnitude: float) -> tuple[torch.Tensor, torch.Tensor]:
    """streams [T, n, H] -> (post [T, n] fp32, collapsed input [T, H] in streams' dtype)."""
    n = streams.shape[1]
    flat = streams.flatten(1).float()
    mixes = (flat @ fn.float().t()) * unweighted_rsqrt(flat, eps_rms)
    pre_b, post_b = base.float().split(n, dim=-1)
    pre_s, post_s = scale.float().unbind(0)
    pre_l, post_l = mixes.split(n, dim=-1)
    pre = torch.sigmoid(pre_l * pre_s + pre_b) + hc_eps
    post = magnitude * torch.sigmoid(post_l * post_s + post_b) + hc_eps
    out = (pre.unsqueeze(-1) * streams.float()).sum(dim=1)
    return post, out.to(streams.dtype)


def hc_post(post: torch.Tensor, sublayer_out: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    return (post.unsqueeze(-1) * sublayer_out.float().unsqueeze(-2) + residual.float()).to(residual.dtype)


def hc_head(streams: torch.Tensor, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor,
            eps_rms: float, hc_eps: float) -> torch.Tensor:
    flat = streams.flatten(1).float()
    mixes = (flat @ fn.float().t()) * unweighted_rsqrt(flat, eps_rms)
    pre = torch.sigmoid(mixes * scale.float() + base.float()) + hc_eps
    return (pre.unsqueeze(-1) * streams.float()).sum(dim=1).to(streams.dtype)


# ---------------------------------------------------------------------------------- Engram

def ngram_tokens(ids: torch.Tensor, history: torch.Tensor | None, n: int, pad_id: int) -> torch.Tensor:
    """ids [T] (one sequence, consecutive positions); history [n-1] tokens before ids[0] (pad if
    absent) -> [T, n] (current token first, then the previous n-1)."""
    if history is None:
        history = torch.full((n - 1,), pad_id, dtype=ids.dtype, device=ids.device)
    full = torch.cat([history.to(ids.dtype), ids])
    T = ids.shape[0]
    cols = [full[n - 1 - k: n - 1 - k + T] for k in range(n)]
    return torch.stack(cols, dim=-1)


def hash_ngrams(ngram: torch.Tensor, token_map: torch.Tensor, multipliers: torch.Tensor,
                head_vocab_sizes: torch.Tensor, n_head_per_ngram: int, pad_id: int) -> torch.Tensor:
    """[T, n] tokens -> [T, (n-1) * heads] hash ids (transformers `hash_ngrams`)."""
    size = token_map.shape[0]
    comp = token_map[ngram.clamp_max(size - 1).long()]
    comp = torch.where(ngram < size, comp, token_map[pad_id])
    mixed = comp.long() * multipliers.long()
    out = []
    n = ngram.shape[-1]
    for order in range(2, n + 1):
        mix = mixed[..., 0]
        for off in range(1, order):
            mix = torch.bitwise_xor(mix, mixed[..., off])
        s = (order - 2) * n_head_per_ngram
        out.append(mix.unsqueeze(-1) % head_vocab_sizes[s:s + n_head_per_ngram].long())
    return torch.cat(out, dim=-1)


def engram_residual(streams: torch.Tensor, rows: torch.Tensor, lookup_norms: torch.Tensor,
                    kv_weight: torch.Tensor, kv_bias: torch.Tensor, gate_w: torch.Tensor,
                    hc_scales: torch.Tensor, pre_gamma: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """streams [T, n, H]; rows [T, heads, d] looked-up embeddings; GGUF-fused weights:
    kv_weight [(n+1)*H, heads*d] (keys first, value last) or a callable applying it, kv_bias [(n+1), H],
    gate_w [n, H] (= norm_key * norm_query), hc_scales [n, H]. Returns the residual [T, n, H]."""
    n, H = streams.shape[1], streams.shape[2]
    emb = rms_norm(rows.float(), lookup_norms.float(), None).flatten(1)          # [T, heads*d]
    kv = kv_weight(emb) if callable(kv_weight) else emb @ kv_weight.float().t()   # callable: a GGUF linear
    proj = kv.float().view(-1, n + 1, H) + kv_bias.float()
    keys, value = proj[:, :n], proj[:, n]
    k_hat = rms_norm(keys, None, None)
    q_hat = rms_norm(streams.float(), None, None)
    g = (k_hat * q_hat * gate_w.float()).sum(-1) * H ** -0.5                    # [T, n]
    g = torch.sigmoid(g.abs().clamp_min(1e-6).sqrt() * g.sign())
    main = pre_gamma.float() * rms_norm(value, None, None)                       # [T, H]
    out = g.unsqueeze(-1) * main.unsqueeze(1) + hc_scales.float() * main.unsqueeze(1)
    if mask is not None:
        out = out.masked_fill(~mask.view(-1, 1, 1), 0.0)
    return out.to(streams.dtype)


# ---------------------------------------------------------------------------------- MoE / MLP

def route(h: torch.Tensor, w_router: torch.Tensor, bias: torch.Tensor, top_k: int, norm: bool,
          scaling: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid router with selection bias (n_group = topk_group = 1)."""
    logits = h.float() @ w_router.float().t()
    scores = logits.sigmoid()
    idx = torch.topk(scores + bias.float(), k=top_k, dim=-1, sorted=False).indices
    w = scores.gather(1, idx)
    if norm:
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    return w * scaling, idx


def clamped_swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    return F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)


def dense_experts(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor,
                  weights: torch.Tensor, idx: torch.Tensor, limit: float) -> torch.Tensor:
    """Reference experts: x [T, D], w_gate/w_up [E, I, D], w_down [E, D, I]."""
    out = torch.zeros_like(x, dtype=torch.float32)
    for e in idx.unique().tolist():
        t, slot = torch.where(idx == e)
        y = clamped_swiglu(x[t].float() @ w_gate[e].float().t(), x[t].float() @ w_up[e].float().t(), limit)
        out.index_add_(0, t, (y @ w_down[e].float().t()) * weights[t, slot, None])
    return out.to(x.dtype)


def mlp(x: torch.Tensor, w_gate, w_up, w_down) -> torch.Tensor:
    return (F.silu(x @ w_gate.t()) * (x @ w_up.t())) @ w_down.t()
