# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 routed experts: sigmoid router + selection bias, top-k, normalized weights x expert_weights_scale, SITU
activation, per-expert scales, and each projection (gate_up, down) dispatched by its OWN format (files mix formats
per tensor: e.g. gate_up IQ2_S or ITQ, down NVFP4).

Stage kernels on the GPU (no host synchronisation: tokens are grouped by expert on the device):
  * GGML MMVQ / MMQ families (Q4_0..Q8_0, K-quants, IQ*): `ggml_moe_a8_vec` (any batch) / `ggml_moe_a8` (standard and
    K-quants above 64 tokens), the existing Triton kernels;
  * MXFP4 / NVFP4 / Q1_0 / Q2_0 and float experts: triton/fused_moe/new_types.py (MXFP4 / NVFP4 / Q2_0 up to
    new_types.gemv_rows routed rows: one GEMV program row per routed row, no sorting; KAILLM_INKLING_GEMV_ROWS caps
    it, 0: off);
  * ITQ (itq-moe v1): inkling/itq_moe.py (decode GEMV per routed row, grouped prefill).
CPU (tests): dequantized weight x activations per selected expert.
"""

from __future__ import annotations

import os

import torch
from torch import nn

from . import ggml_types as G

_FLOAT = {0: torch.float32, 1: torch.float16, 30: torch.bfloat16}
GEMV_ROWS = int(os.environ.get("KAILLM_INKLING_GEMV_ROWS", "4096"))      # cap over new_types.GEMV_TILES


def route(x: torch.Tensor, router: torch.Tensor, bias: torch.Tensor | None, top_k: int, w_scale: float):
    """(ids [T, k] int64, weights [T, k] float32); router [E, D] float32 (llama.cpp: router in F32)."""
    probs = torch.sigmoid(x.float() @ router.float().t())
    sel = probs + bias.float() if bias is not None else probs
    ids = sel.topk(top_k, dim=-1).indices
    w = probs.gather(1, ids)
    w = w / (w.sum(-1, keepdim=True) + 1e-20)
    if w_scale not in (0.0, 1.0):
        w = w * w_scale
    return ids, w


def situ(gate: torch.Tensor, up: torch.Tensor, beta: float, lb: float) -> torch.Tensor:
    act = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if lb > 0:
        up = lb * torch.tanh(up / lb)
    return act * up


class ExpertStage:
    """One projection of all experts: GGUF bytes [E, rows, row_bytes] (or float [E, rows, cols]) + its GGML type."""

    def __init__(self, ggml_type: int, data: torch.Tensor, rows: int, cols: int, scale: torch.Tensor | None):
        self.type, self.rows, self.cols = int(ggml_type), rows, cols
        E = data.shape[0]
        self.data = data.reshape(E, rows, -1)
        self.scale = scale

    @property
    def E(self) -> int:
        return self.data.shape[0]

    def to(self, device):
        self.data = self.data.to(device)
        if self.scale is not None:
            self.scale = self.scale.to(device)
        return self

    def dense(self, e: int, dtype=torch.float32) -> torch.Tensor:
        """Expert e as a float [rows, cols] matrix."""
        w = self.data[e]
        if self.type in _FLOAT:
            return w.reshape(self.rows, self.cols).to(dtype)
        if self.type in G.NEW_TYPES:
            return G.dequantize(w, self.type, self.cols).to(dtype)
        if w.is_cuda:
            from .. import ops
            return ops.ggml_dequantize(w.contiguous(), self.type, self.rows, self.cols, dtype)
        import gguf
        from gguf.quants import dequantize
        arr = dequantize(w.cpu().numpy(), gguf.GGMLQuantizationType(self.type))
        return torch.from_numpy(arr).reshape(self.rows, self.cols).to(device=w.device, dtype=dtype)

    def _kernel(self, tokens: int) -> str:
        from ..quantization.utils import MMQ_QUANT_TYPES, MMVQ_QUANT_TYPES
        if not self.data.is_cuda:
            return "dense"
        if self.type in _FLOAT:
            return "float"
        if self.type in G.NEW_TYPES:
            return "new"
        if self.type in MMQ_QUANT_TYPES and tokens > 64:
            return "mmq"
        if self.type in MMVQ_QUANT_TYPES:
            return "mmvq"
        raise ValueError(f"no GPU MoE kernel for GGML type {self.type}")

    def run(self, x: torch.Tensor, ids: torch.Tensor, shared: bool, pass_index=None) -> torch.Tensor:
        """shared: x [T, cols] read by every slot of ids [T, k] -> [T * k, rows];
        else x [T * k, cols] (one row per slot) -> [T * k, rows].  Output x.dtype or float32."""
        T, k = ids.shape
        kind = self._kernel(T if shared else T * k)
        from .. import ops
        if kind in ("float", "new"):
            from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size
            from ..triton.fused_moe import new_types as N
            if kind == "new" and self.type in N.GEMV_TYPES and T * k <= min(GEMV_ROWS, N.gemv_rows(self.type)):
                xin = x if x.dtype in (torch.bfloat16, torch.float16) else x.to(torch.bfloat16)
                return N.moe_new_type_gemv(xin.contiguous(), self.data, ids.reshape(-1).to(torch.int32).contiguous(),
                                           self.type, self.rows, k if shared else 1)
            route = (ids if shared else ids.reshape(-1, 1)).to(torch.int32)
            sorted_ids, expert_ids, n_post = moe_align_block_size(route, N.moe_block_m(), self.E)
            xin = x if x.dtype in (torch.bfloat16, torch.float16) else x.to(torch.bfloat16)
            xin = xin.contiguous()
            if kind == "float":
                w = self.data.view(_FLOAT[self.type]).reshape(self.E, self.rows, self.cols)
                return N.moe_dense(xin, w, sorted_ids, expert_ids, self.rows, k if shared else 1, xin.shape[0],
                                   out_dtype=torch.float32)
            return N.moe_new_type(xin, self.data, sorted_ids, expert_ids, n_post, self.type, self.rows,
                                  k if shared else 1, xin.shape[0], out_dtype=torch.float32)
        if kind == "mmvq":
            ids32 = ids.to(torch.int32).contiguous()            # the CUDA kernels read int32 expert ids
            if shared:
                return ops.ggml_moe_a8_vec(x, self.data, ids32, k, self.type, self.rows, T)
            return ops.ggml_moe_a8_vec(x, self.data, ids32.reshape(-1, 1), 1, self.type, self.rows, T * k)
        if kind == "mmq":
            from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size
            block = ops.ggml_moe_get_block_size(self.type)
            flat_ids = ids if shared else ids.reshape(-1, 1)
            sorted_ids, expert_ids, n_post = moe_align_block_size(flat_ids, block, self.E)
            return ops.ggml_moe_a8(x, self.data, sorted_ids, expert_ids, n_post, self.type, self.rows,
                                   k if shared else 1, T if shared else T * k)
        flat = ids.reshape(-1)
        out = torch.zeros(flat.shape[0], self.rows, dtype=torch.float32, device=x.device)
        xin = x.float()
        for e in flat.unique().tolist():
            sel = (flat == e).nonzero(as_tuple=True)[0]
            rows_in = xin[sel // k] if shared else xin[sel]
            out[sel] = rows_in @ self.dense(e, torch.float32).t()
        return out


class InklingExperts(nn.Module):
    """Routed experts of one physical layer (shared by every visit of a looped layer)."""

    def __init__(self, num_experts: int, hidden: int, ff: int, top_k: int, beta: float, lb: float, w_scale: float):
        super().__init__()
        self.E, self.D, self.F, self.k = num_experts, hidden, ff, top_k
        self.beta, self.lb, self.w_scale = beta, lb, w_scale
        self.gate_up: ExpertStage | None = None
        self.down: ExpertStage | None = None
        self.itq_gate_up = None              # inkling/itq_moe.py stages (ITQ files)
        self.itq_down = None
        self._pending: dict = {}

    def offer(self, part: str, kind: str, value: torch.Tensor) -> None:
        """Collect the loader's tensors: part in (gate_up, down, gate_up_scale, down_scale), kind in (type, data)."""
        self._pending.setdefault(part, {})[kind] = value
        for stage, rows, cols in (("gate_up", 2 * self.F, self.D), ("down", self.D, self.F)):
            p = self._pending.get(stage, {})
            if "type" in p and "data" in p and getattr(self, stage) is None:
                setattr(self, stage, ExpertStage(int(p["type"].item()), p["data"], rows, cols, None))

    def finalize(self, device) -> None:
        for stage in ("gate_up", "down"):
            st = getattr(self, stage)
            if st is None and getattr(self, "itq_" + stage) is None:
                raise ValueError(f"missing expert tensor {stage}")
            if st is not None:
                sc = self._pending.get(stage + "_scale", {}).get("data")
                st.scale = None if sc is None else sc.float().reshape(-1)
                st.to(device)
        self._pending.clear()

    def forward(self, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, pass_index=None,
                out_dtype: torch.dtype | None = None) -> torch.Tensor:
        """x [T, D] (model dtype) -> [T, D] out_dtype (default x.dtype)."""
        T = x.shape[0]
        k = ids.shape[1]
        gate_up = self.itq_gate_up if self.itq_gate_up is not None else self.gate_up
        from . import glue as G
        gu = gate_up.run(x, ids, True, pass_index)
        st = self.gate_up
        h = G.gate_up_epilogue(gu, st.scale if st is not None else None, ids, self.F, self.beta, self.lb, x.dtype)
        down = self.itq_down if self.itq_down is not None else self.down
        y = down.run(h, ids, False, pass_index)
        return G.combine(y, weights, self.down.scale if self.down is not None else None, ids, out_dtype or x.dtype)
