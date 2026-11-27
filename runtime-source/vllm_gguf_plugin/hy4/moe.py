# SPDX-License-Identifier: Apache-2.0
"""Hy4 latent-MoE experts on kaiLLM's GGUF MoE kernels, with Hy4's clamped SwiGLU.

Mirrors quantization/fused_moe.py `_fused_moe_gguf` (MMQ for batches > 64 tokens on standard /
K-quant types, MMVQ otherwise, a per-token fallback last); only the activation differs:
silu(clamp(gate, max=L)) * clamp(up, -L, L). Hy4 GGUFs stack experts as [E, I, D] tensors, so the
packed w13 is gate rows then up rows per expert (one GGML type for both, as kaiLLM's w13 needs).
"""

from __future__ import annotations

import torch
from torch import nn

from .blocks import clamped_swiglu


_UNQUANTIZED = {0: torch.float32, 1: torch.float16, 30: torch.bfloat16}


def _expert_dense(w: torch.Tensor, e: int, t: int, rows: int, cols: int, dtype) -> torch.Tensor:
    if t in _UNQUANTIZED:
        return w[e].view(_UNQUANTIZED[t]).view(rows, cols)
    from .. import ops
    return ops.ggml_dequantize(w[e].contiguous(), t, rows, cols, dtype)


def hy4_gguf_moe(x, w13, w2, topk_weights, topk_ids, type13: int, type2: int, limit: float) -> torch.Tensor:
    """Routed experts on packed GGUF weights: w13 [E, 2I, row bytes], w2 [E, D, row bytes]."""
    from ..quantization.utils import MMQ_QUANT_TYPES, MMVQ_QUANT_TYPES

    def act(inp):
        d = inp.shape[-1] // 2
        return clamped_swiglu(inp[..., :d], inp[..., d:], limit).to(inp.dtype)

    num_tokens, d_in = x.shape
    E = w13.shape[0]
    top_k = topk_ids.shape[1]
    d_out = w2.shape[1]
    N = w13.shape[1]
    if type13 in MMQ_QUANT_TYPES and type2 in MMQ_QUANT_TYPES and num_tokens > 64:
        from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size
        from .. import ops
        block = ops.ggml_moe_get_block_size(type13)
        sorted_ids, expert_ids, n_post = moe_align_block_size(topk_ids, block, E)
        h = ops.ggml_moe_a8(x, w13, sorted_ids, expert_ids, n_post, type13, N, top_k, num_tokens)
        h = act(h)
        h = ops.ggml_moe_a8(h, w2, sorted_ids, expert_ids, n_post, type2, d_out, 1, num_tokens * top_k)
    elif type13 in MMVQ_QUANT_TYPES and type2 in MMVQ_QUANT_TYPES:
        from .. import ops
        h = ops.ggml_moe_a8_vec(x, w13, topk_ids, top_k, type13, N, num_tokens)
        h = act(h)
        h = ops.ggml_moe_a8_vec(h, w2, topk_ids, 1, type2, d_out, num_tokens * top_k)
    else:
        # unquantized (F32 / F16 / BF16) or dequant-only types: dense matmuls per expert in the batch
        out = torch.zeros(num_tokens, d_out, dtype=torch.float32, device=x.device)
        for e in topk_ids.unique().tolist():
            t, slot = torch.where(topk_ids == e)
            g_u = _expert_dense(w13, e, type13, N, d_in, x.dtype)
            dn = _expert_dense(w2, e, type2, d_out, N // 2, x.dtype)
            y = act(x[t] @ g_u.to(x.dtype).t())
            out.index_add_(0, t, (y @ dn.to(x.dtype).t()).float() * topk_weights[t, slot, None].float())
        return out.to(x.dtype)
    from .. import ops
    out_hidden = torch.empty(num_tokens, d_out, dtype=x.dtype, device=x.device)
    h = h.reshape(num_tokens, top_k, d_out).mul_(topk_weights.view(num_tokens, top_k, 1).to(h.dtype))
    ops.moe_sum(h, out_hidden)
    return out_hidden


class Hy4GGUFExperts(nn.Module):
    def __init__(self, num_experts: int, intermediate: int, hidden: int, limit: float):
        super().__init__()
        self.E, self.I, self.D, self.limit = num_experts, intermediate, hidden, limit
        self.register_buffer("w13", torch.empty(0, dtype=torch.uint8), persistent=False)
        self.register_buffer("w2", torch.empty(0, dtype=torch.uint8), persistent=False)
        self.type13 = self.type2 = -1
        self._pending: dict[str, tuple[int, torch.Tensor]] = {}

    def offer(self, part: str, kind: str, tensor: torch.Tensor) -> bool:
        """Collect gate / up / down payloads and types from the loader; build when complete."""
        slot = self._pending.setdefault(part, [None, None])
        slot[0 if kind == "type" else 1] = tensor
        if not all(p in self._pending and None not in self._pending[p] for p in ("gate", "up", "down")):
            return False
        types = {p: int(self._pending[p][0].item()) for p in ("gate", "up", "down")}
        if types["gate"] != types["up"]:
            raise ValueError(f"kaiLLM loads ffn_gate_exps and ffn_up_exps as one w13 and needs one GGML type "
                             f"for both (got {types['gate']} and {types['up']})")
        from .attention import graph_backend_enabled
        if graph_backend_enabled():
            from ..quantization.utils import MMVQ_QUANT_TYPES
            if types["gate"] not in MMVQ_QUANT_TYPES or types["down"] not in MMVQ_QUANT_TYPES:
                raise ValueError("Hy4 graph execution needs a graph-compatible GGUF MoE kernel for both weights")
        dev = self.w13.device
        g, u, d = (self._pending[p][1] for p in ("gate", "up", "down"))
        self.w13 = torch.cat([g.view(self.E, self.I, -1), u.view(self.E, self.I, -1)], dim=1).contiguous().to(dev)
        self.w2 = d.view(self.E, self.D, -1).contiguous().to(dev)
        self.type13, self.type2 = types["gate"], types["down"]
        self._pending.clear()
        return True

    def forward(self, x, topk_weights, topk_ids):
        return hy4_gguf_moe(x, self.w13, self.w2, topk_weights, topk_ids, self.type13, self.type2, self.limit)
