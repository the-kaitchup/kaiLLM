# SPDX-License-Identifier: Apache-2.0
"""ITQ routed experts (codec itq-moe v1) as an Inkling MoE stage.

W_e = diag(pass_p,e) diag(pos_e) diag(sv) R_out (wscale_e T_e) R_in diag(su) diag(pis_e).  Run time, no host
synchronisation on the GPU:

  decode (<= DECODE_ROWS routed rows): itq_prep_rows (rotation + int8 per 32, once per token when the slots share the
      input) -> itq_moe_gemv_all (dp4a GEMV per routed row, expert and its format read on the device, every format in
      one launch; KAILLM_ITQ_MOE_ONE_LAUNCH=0: one launch per format code present) -> itq_post (R_out, sv)
      -> per-expert / per-pass output scales;
  prefill: itq_prep (rotated BF16 rows) -> itq_decode_i8 of every expert into a bounded int8 workspace (the raw table
      values, no scales) -> Triton grouped GEMM (tokens sorted by expert, x wscale_e) -> itq_post -> scales.

CPU (tests): the FP32 oracle per expert (yaqa_gguf.itq_moe_expert / reconstruct_itq).
"""

from __future__ import annotations

import ctypes
import os

import torch

# routed rows up to which the GEMV path runs: every decode batch the server captures (32 sequences x 8 experts = 256
# rows) and up to 64 sequences; above it (prefill) every expert is decoded once into the int8 workspace
DECODE_ROWS = int(os.getenv("KAILLM_ITQ_MOE_DECODE_ROWS", "512"))
# decode GEMV: every format of a tensor in one launch (0: one launch per format code present, the earlier path)
ONE_LAUNCH = os.getenv("KAILLM_ITQ_MOE_ONE_LAUNCH", "1") == "1"
_WORKSPACE: dict = {}


def _block(kind: str, width: int) -> int:
    return width if kind == "fwht" else int(kind[5:])


def _pack_table(t8: torch.Tensor, V: int) -> torch.Tensor:
    t = t8.to(torch.int32) & 0xFF
    if V == 4:
        return t[:, 0] | (t[:, 1] << 8) | (t[:, 2] << 16) | (t[:, 3] << 24)
    return t[:, 0] | (t[:, 1] << 8)


def _lib(device):
    from ..quantization import itq
    lib = itq.library(device)
    if not getattr(lib, "_inkling_moe", False):
        p, i = ctypes.c_void_p, ctypes.c_int
        lib.itq_moe_gemv.argtypes = [p, p, p, p, p, i, i, p, p, p, i, p, i, i, i, i, p]
        lib.itq_moe_gemv.restype = i
        lib.itq_prep_rows.argtypes = [p, i, p, p, p, p, i, i, i, i, i, p]
        lib.itq_prep_rows.restype = i
        lib.itq_moe_gemv_all.argtypes = [p, p, p, p, p, i, i, p, p, p, p, i, p, i, i, i, p]
        lib.itq_moe_gemv_all.restype = i
        lib._inkling_moe = True
    return lib


def _check(code, what):
    if code:
        raise RuntimeError(f"ITQ MoE {what} launch failed with CUDA error {code}")


class ITQExpertStage:
    """One itq-moe tensor: run(x, ids, shared, pass_index) -> float32 [T * k, m]."""

    def __init__(self, descriptor, components, device):
        from .. import yaqa_gguf as yg
        self.descriptor = descriptor
        E, m, n = descriptor["shape"]
        self.E, self.m, self.n = E, m, n
        self.formats = yg.itq_moe_formats(descriptor)
        offsets = yg.itq_moe_offsets(descriptor)
        self.block_in = _block(descriptor["rotation_in"], n)
        self.block_out = _block(descriptor["rotation_out"], m)
        dev = torch.device(device)
        self.device = dev
        self.components = components if dev.type == "cpu" else None       # CPU oracle path
        stream = components["stream"].reshape(-1).view(torch.uint8)
        pad = (-stream.numel()) % 4
        if pad:
            stream = torch.cat([stream, stream.new_zeros(pad)])
        self.words = stream.view(torch.int32).to(dev)
        self.word_off_host = [o // 4 for o in offsets[:-1]]
        self.woff = torch.tensor(self.word_off_host, dtype=torch.int64, device=dev)
        self.codes_host = [bps if V == 4 else 32 + bps for V, bps in self.formats]
        self.V_host = [V for V, _ in self.formats]
        self.fmt = torch.tensor(self.codes_host, dtype=torch.int32, device=dev)
        self.present = sorted({(c, V) for c, V in zip(self.codes_host, self.V_host)})
        self.wscale = components["wscale"].reshape(-1).float().to(dev)
        self.su = components["su"].to(torch.int8).to(dev)
        self.sv = components["sv"].to(torch.int8).to(dev)
        self.tables = {V: _pack_table(components[f"table_v{V}"], V).contiguous().to(dev)
                       for V in sorted(set(self.V_host))}
        pis = components.get("post_input_scale")
        self.pis = None if pis is None else pis.float().to(dev)
        pos = components.get("post_output_scale")
        self.pos = None if pos is None else pos.float().to(dev)
        ps = components.get("pass_output_scale")
        self.pass_scale = None if ps is None else ps.float().to(dev)
        self._dense: dict = {}

    @property
    def rows(self) -> int:
        return self.m

    # ---- CPU oracle (tests) ------------------------------------------------------------------------------------
    def _expert(self, e):
        if e not in self._dense:
            from .. import yaqa_gguf as yg
            sub, comps = yg.itq_moe_expert(self.descriptor, self.components, e)
            comps = {k: v for k, v in comps.items() if k not in ("post_output_scale",)}   # applied in run()
            if "post_input_scale" in comps and comps["post_input_scale"].dim() == 1 and self.pis is not None \
                    and self.pis.dim() == 2:
                comps.pop("post_input_scale")
            self._dense[e] = yg.reconstruct_itq(sub, comps)
        return self._dense[e]

    def _run_cpu(self, x, ids, shared):
        flat = ids.reshape(-1)
        k = ids.shape[1]
        out = torch.zeros(flat.numel(), self.m, dtype=torch.float32)
        xin = x.float()
        for e in flat.unique().tolist():
            sel = (flat == e).nonzero(as_tuple=True)[0]
            rows = xin[sel // k] if shared else xin[sel]
            if self.pis is not None and self.pis.dim() == 2:
                rows = rows * self.pis[e]
            out[sel] = rows @ self._expert(e).t()
        return out

    # ---- GPU ---------------------------------------------------------------------------------------------------
    def _prepared_input(self, x, ids, shared):
        """(activations [rows, n] contiguous, pis pointer): per-expert input scales folded into the rows."""
        if self.pis is not None and self.pis.dim() == 2:
            if shared:
                raise ValueError("per-expert post_input_scale needs one activation row per slot (down projections)")
            return (x.float() * self.pis[ids.reshape(-1)]).contiguous(), None     # float32 rows: no extra rounding
        return x.contiguous(), (None if self.pis is None else self.pis.data_ptr())

    def _finish(self, y, ids, pass_index):
        """y float32 [R, m] in the rotated basis -> out float32 [R, m]."""
        lib = _lib(y.device)
        R = y.shape[0]
        out = torch.empty(R, self.m, dtype=torch.bfloat16, device=y.device)
        stream = torch.cuda.current_stream(y.device).cuda_stream
        _check(lib.itq_post(y.data_ptr(), self.sv.data_ptr(), None, None, out.data_ptr(), R, self.m, self.m, self.m,
                            self.block_out, stream), "post")
        res = out.float()
        flat = ids.reshape(-1)
        if self.pos is not None:
            res = res * self.pos[flat]
        if self.pass_scale is not None and pass_index is not None:
            res = res * self.pass_scale[pass_index][flat]
        return res

    def _run_decode(self, x, ids, shared, pass_index):
        lib = _lib(x.device)
        T, k = ids.shape
        R = T * k
        xin, pis_ptr = self._prepared_input(x, ids, shared)
        rows_act = xin.shape[0]
        dev = x.device
        stream = torch.cuda.current_stream(dev).cuda_stream
        xq = torch.empty(rows_act * self.n // 4, dtype=torch.int32, device=dev)
        xs = torch.empty(rows_act * self.n // 32, dtype=torch.float32, device=dev)
        bf16 = 1 if xin.dtype == torch.bfloat16 else 0
        if xin.dtype not in (torch.bfloat16, torch.float32):
            xin, bf16 = xin.float(), 0
        _check(lib.itq_prep_rows(xin.data_ptr(), bf16, pis_ptr, self.su.data_ptr(), xq.data_ptr(), xs.data_ptr(),
                                 rows_act, self.n, xin.stride(0), self.block_in, rows_act, stream), "prep_rows")
        y = torch.empty(R, self.m, dtype=torch.float32, device=dev)
        ids32 = ids.reshape(-1).to(torch.int32).contiguous()
        if ONE_LAUNCH:
            t4, t2 = self.tables.get(4), self.tables.get(2)
            _check(lib.itq_moe_gemv_all(self.words.data_ptr(), self.woff.data_ptr(), self.fmt.data_ptr(),
                                        self.wscale.data_ptr(), ids32.data_ptr(), R, k if shared else 1,
                                        None if t4 is None else t4.data_ptr(), None if t2 is None else t2.data_ptr(),
                                        xq.data_ptr(), xs.data_ptr(), rows_act, y.data_ptr(), self.m, self.n, 2,
                                        stream), "gemv_all")
            return self._finish(y, ids, pass_index)
        for code, V in self.present:
            _check(lib.itq_moe_gemv(self.words.data_ptr(), self.woff.data_ptr(), self.fmt.data_ptr(),
                                    self.wscale.data_ptr(), ids32.data_ptr(), R, k if shared else 1,
                                    self.tables[V].data_ptr(), xq.data_ptr(), xs.data_ptr(), rows_act, y.data_ptr(),
                                    self.m, self.n, code, 2, stream), "gemv")
        return self._finish(y, ids, pass_index)

    def _run_prefill(self, x, ids, shared, pass_index):
        from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size
        from ..triton.fused_moe import new_types as N
        lib = _lib(x.device)
        T, k = ids.shape
        dev = x.device
        stream = torch.cuda.current_stream(dev).cuda_stream
        xin, pis_ptr = self._prepared_input(x, ids, shared)
        rows_act = xin.shape[0]
        bf16 = 1 if xin.dtype == torch.bfloat16 else 0
        if xin.dtype not in (torch.bfloat16, torch.float32):
            xin, bf16 = xin.float(), 0
        xr = torch.empty(rows_act, self.n, dtype=torch.bfloat16, device=dev)
        _check(lib.itq_prep(xin.data_ptr(), bf16, pis_ptr, self.su.data_ptr(), None, None, xr.data_ptr(), rows_act,
                            self.n, xin.stride(0), self.block_in, 0, None, 0, stream), "prep")
        key = (dev.index, self.E, self.m, self.n)
        w8 = _WORKSPACE.get(key)
        if w8 is None:
            w8 = _WORKSPACE[key] = torch.empty(self.E, self.m, self.n, dtype=torch.int8, device=dev)
        base = self.words.data_ptr()
        for e in range(self.E):
            _check(lib.itq_decode_i8(base + 4 * self.word_off_host[e], self.tables[self.V_host[e]].data_ptr(),
                                     w8[e].data_ptr(), self.m, self.n, self.codes_host[e], stream), "decode_i8")
        route = ids if shared else ids.reshape(-1, 1)
        sorted_ids, expert_ids, n_post = moe_align_block_size(route.to(torch.int32), N.moe_block_m(), self.E)
        y = N.moe_dense(xr, w8, sorted_ids, expert_ids, self.m, k if shared else 1, rows_act, scale=self.wscale,
                        out_dtype=torch.float32)
        return self._finish(y, ids, pass_index)

    def run(self, x, ids, shared: bool, pass_index=None) -> torch.Tensor:
        if not x.is_cuda:
            out = self._run_cpu(x, ids, shared)
            flat = ids.reshape(-1)
            if self.pos is not None:
                out = out * self.pos[flat]
            if self.pass_scale is not None and pass_index is not None:
                out = out * self.pass_scale[pass_index][flat]
            return out
        if ids.numel() <= DECODE_ROWS:
            return self._run_decode(x, ids, shared, pass_index)
        return self._run_prefill(x, ids, shared, pass_index)


class SplitGateUp:
    """Split gate / up itq-moe tensors run as one gate_up stage (rows [0, F) gate, [F, 2F) up)."""

    def __init__(self, gate: ITQExpertStage, up: ITQExpertStage):
        if gate.n != up.n or gate.E != up.E or not torch.equal(gate.su.cpu(), up.su.cpu()):
            raise ValueError("split gate / up itq-moe tensors must share E, n and su")
        self.gate, self.up = gate, up
        self.rows = gate.m + up.m

    def run(self, x, ids, shared, pass_index=None):
        return torch.cat([self.gate.run(x, ids, shared, pass_index), self.up.run(x, ids, shared, pass_index)], dim=1)
