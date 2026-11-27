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
    return route_from_logits(x.float() @ router.float().t(), bias, top_k, w_scale)


def route_from_logits(logits, bias, top_k, w_scale):
    probs = torch.sigmoid(logits)
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
        self.iq_gate_tiles = None
        self.iq_gate_verifier_rows = False
        self.pro_dondon_prefill = None
        self.pro_dondon_routes = {}
        self.pro_iq4_tiles = None
        self.iq4_prefill_tile = None
        self.iq4_down_prefill_2048 = False
        self.dondon_m_down_prefill = False
        self.iq4_decode_tile = None
        self.iq4_sm86_packed = False
        self.decode_rows = 1
        self.decode_route = None
        self.q8_prefill = False
        self.q8_decode_tile = None

    @property
    def E(self) -> int:
        return self.data.shape[0]

    def to(self, device):
        self.data = self.data.to(device)
        self.iq_gate_tiles = None
        self.iq_gate_verifier_rows = False
        self.pro_dondon_prefill = None
        self.pro_dondon_routes = {}
        if (self.data.is_cuda and self.type in (16, 17, 18, 19, 22) and self.E == 128
                and self.rows == self.cols == 2304 and self.data.dtype == torch.uint8
                and self.data.is_contiguous()):
            from vllm.config import get_current_vllm_config_or_none
            from .. import ops
            from .iq_gate import measured_tiles, portable_tiles
            if not ops._cuda_kernel_available("ggml_moe_a8_vec", self.type):
                serving = get_current_vllm_config_or_none()
                gpu = torch.cuda.get_device_name(self.data.device)
                visible = torch.cuda.device_count()
                self.iq_gate_tiles = measured_tiles(
                    gpu, self.type, serving, visible, self.data.shape[-1])
                if self.iq_gate_tiles is not None:
                    from .dondon_5090 import released_tier
                    # The released S/M GGUFs use row-addressed IQ1/IQ2 and direct IQ3 gates.
                    self.iq_gate_verifier_rows = (
                        serving.speculative_config is not None or gpu == "NVIDIA GeForce RTX 3090"
                        or (self.type in (16, 17, 19, 22) and released_tier(serving, gpu, visible) is not None)
                    )
                override = portable_tiles(serving, self.data.device, self.type, self.data.shape[-1])
                if override is not None:
                    self.iq_gate_tiles = {**(self.iq_gate_tiles or {}), **override}
                if self.iq_gate_tiles is None and self.type in (18, 22):
                    from .pro_dondon import iq_gate_tuning

                    self.pro_dondon_prefill, self.pro_dondon_routes = iq_gate_tuning(
                        serving, gpu, visible, self.type, self.data.shape[-1], self.data.device)
        self.pro_iq4_tiles = None
        self.iq4_prefill_tile = None
        self.iq4_down_prefill_2048 = False
        self.dondon_m_down_prefill = False
        self.iq4_decode_tile = None
        self.iq4_sm86_packed = False
        self.decode_rows = 1
        self.decode_route = None
        self.q8_prefill = False
        self.q8_decode_tile = None
        if (self.data.is_cuda and self.type in (8, 20) and self.data.dtype == torch.uint8
                and self.E == 128 and self.rows == 2304 and self.cols in (1152, 2304)
                and self.data.shape[2] == self.cols // 32 * (34 if self.type == 8 else 18) and self.data.is_contiguous()):
            from vllm.config import get_current_vllm_config_or_none
            from .. import ops
            from .iq4_prefill import qualified_projection_tiles
            gpu_name = torch.cuda.get_device_name(self.data.device)
            is_ampere_3090 = gpu_name.endswith('GeForce RTX 3090')
            native_vec = ops._cuda_kernel_available('ggml_moe_a8_vec', self.type)
            # The Ampere Q8 measurements used the Triton paths for both MMQ and MMVQ.
            native = (ops._cuda_gemm_kernel_available('ggml_moe_a8', self.type)
                      if self.type == 8 and is_ampere_3090 else native_vec)
            serving = get_current_vllm_config_or_none()
            visible = torch.cuda.device_count()
            if self.type == 20 and self.cols == 1152 and not native:
                from .dondon_5090 import is_dondon_m
                from .iq4_prefill import supports_down_prefill_tile

                self.dondon_m_down_prefill = is_dondon_m(serving, gpu_name, visible)
                self.iq4_down_prefill_2048 = supports_down_prefill_tile(serving, self.data.device)
            if self.dondon_m_down_prefill:
                prefill, decode = 32, None
            else:
                prefill, decode = qualified_projection_tiles(serving, gpu_name, visible, native, self.cols, self.type)
            if self.type == 20:
                if not native:
                    from .iq4_prefill import pro_projection_tiles
                    self.pro_iq4_tiles = pro_projection_tiles(serving, gpu_name, visible, self.cols)
                self.iq4_sm86_packed = is_ampere_3090
                self.iq4_prefill_tile, self.iq4_decode_tile = prefill, decode
            else:
                self.q8_prefill = is_ampere_3090 and prefill is not None
                if not native_vec:
                    self.q8_decode_tile = decode
            if decode is not None:
                self.decode_rows = serving.scheduler_config.max_num_seqs
            if self.q8_decode_tile is not None or (self.iq4_decode_tile is not None and self.cols == 1152):
                from .iq4_decode import down_route
                self.decode_route = down_route(self.data.device, self.decode_rows)
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
        if (self.pro_iq4_tiles and T in self.pro_iq4_tiles and k == 8 and shared == (self.cols == 2304)
                and x.dtype == torch.bfloat16 and x.is_contiguous() and ids.is_contiguous()
                and ids.dtype in (torch.int32, torch.int64)):
            from .iq4_prefill import tiled_prefill
            block_m, block_n, k_blocks = self.pro_iq4_tiles[T]
            return tiled_prefill(x, self.data, ids, shared, self.rows, block_m, pro_geometry=(block_n, k_blocks))
        if (self.iq_gate_tiles and shared and k == 8 and x.dtype == torch.bfloat16
                and x.is_contiguous() and ids.is_contiguous() and ids.dtype in (torch.int32, torch.int64)):
            tile = self.iq_gate_tiles.get(T)
            if tile is not None:
                from .iq_gate import run_projection
                return run_projection(x, self.data, ids, self.type, tile, verifier_rows=self.iq_gate_verifier_rows)
        if ((self.iq4_decode_tile or self.q8_decode_tile) and T == self.decode_rows and k == 8 and shared == (self.cols == 2304) and x.dtype == torch.bfloat16
                and x.is_contiguous() and ids.is_contiguous() and ids.dtype in (torch.int32, torch.int64)):
            if self.q8_decode_tile is not None:
                from .q8_decode import grouped_q8_decode
                return grouped_q8_decode(x, self.data, ids, shared, self.q8_decode_tile, self.decode_route)
            if shared:
                from .iq4_decode import direct_gate_up
                return direct_gate_up(x, self.data, ids, self.iq4_decode_tile)
            from .iq4_decode import grouped_down
            return grouped_down(x, self.data, ids, self.iq4_decode_tile, self.decode_route)
        if ((self.dondon_m_down_prefill or self.iq4_down_prefill_2048) and T == 2048 and k == 8 and not shared
                and x.dtype == torch.bfloat16 and x.shape == (T * k, self.cols) and x.is_contiguous()
                and ids.is_contiguous() and ids.dtype in (torch.int32, torch.int64)):
            from .iq4_prefill import tiled_prefill
            return tiled_prefill(x, self.data, ids, shared, self.rows, 32, self.iq4_sm86_packed)
        if ((self.iq4_prefill_tile or self.q8_prefill) and T in (512, 2048) and k == 8 and x.dtype == torch.bfloat16
                and shared == (self.cols == 2304) and not self.dondon_m_down_prefill):
            if self.q8_prefill:
                from .q8_prefill import tiled_q8_prefill
                return tiled_q8_prefill(x, self.data, ids, shared, self.rows)
            from .iq4_prefill import tiled_prefill
            return tiled_prefill(x, self.data, ids, shared, self.rows, self.iq4_prefill_tile, self.iq4_sm86_packed)
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
            block_m = N.moe_block_m()
            if (self.type == N.NVFP4 and not shared and (T, k, self.cols, self.rows, self.E) == (2048, 8, 1152, 2304, 128)
                    and x.dtype == torch.bfloat16 and x.is_contiguous() and self.data.is_contiguous()
                    and self.data.dtype == torch.uint8 and self.data.shape[-1] == (1152 // 64) * 36
                    and os.getenv("KAILLM_INKLING_NVFP4_PREFILL_TILE") == "1"
                    and torch.cuda.get_device_capability(x.device) == (12, 0)):
                block_m = 32
            sorted_ids, expert_ids, n_post = moe_align_block_size(route, block_m, self.E)
            xin = x if x.dtype in (torch.bfloat16, torch.float16) else x.to(torch.bfloat16)
            xin = xin.contiguous()
            if kind == "float":
                w = self.data.view(_FLOAT[self.type]).reshape(self.E, self.rows, self.cols)
                return N.moe_dense(xin, w, sorted_ids, expert_ids, self.rows, k if shared else 1, xin.shape[0],
                                   out_dtype=torch.float32)
            return N.moe_new_type(xin, self.data, sorted_ids, expert_ids, n_post, self.type, self.rows,
                                  k if shared else 1, xin.shape[0], out_dtype=torch.float32, block_m=block_m)
        if kind == "mmvq":
            ids32 = ids.to(torch.int32).contiguous()            # the CUDA kernels read int32 expert ids
            if ((self.pro_dondon_prefill is not None or self.pro_dondon_routes)
                    and shared and k == 8 and x.dtype == torch.bfloat16 and x.is_contiguous()):
                prefill = self.pro_dondon_prefill if T == 512 else None
                routes = self.pro_dondon_routes.get(T)
                if prefill is not None or routes is not None:
                    from .pro_dondon import iq_gate_projection

                    return iq_gate_projection(x, self.data, ids32, self.type, self.rows, prefill, routes)
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
        from . import chima_verifier, glue as G
        gu = gate_up.run(x, ids, True, pass_index)
        st = self.gate_up
        h = G.gate_up_epilogue(gu, st.scale if st is not None else None, ids, self.F, self.beta, self.lb, x.dtype)
        del gu
        down = self.itq_down if self.itq_down is not None else self.down
        y = down.run(h, ids, False, pass_index)
        del h
        scale = self.down.scale if self.down is not None else None
        return chima_verifier.combine(G.combine, y, weights, scale, ids, out_dtype or x.dtype,
                                      itq=down is self.itq_down)
