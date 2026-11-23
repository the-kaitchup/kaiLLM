# SPDX-License-Identifier: Apache-2.0
"""Inkling decode short-conv Triton kernel (inkling/triton_conv.py) vs the batched torch path (conv.conv_step_batched)
on random states: static and dynamic taps (8 heads), BF16 and float32 states in both vLLM conv-state layouts,
channel offsets inside the state, shuffled state indices with null-block padding rows, float32 inputs and BF16
inputs (the projections' outputs, widened in the kernel) with float32 outputs.  Output and updated state.
GPU only.

    $PY tests/gpu/test_inkling_conv_kernel.py
"""

from __future__ import annotations

import itertools
import sys
from types import SimpleNamespace

import torch


class _State:
    def __init__(self, st, n_conv):
        self.st, self.n_conv = st, n_conv

    def states(self):
        return self.st


def main():
    from vllm_gguf_plugin.inkling.conv import conv_step_batched
    from vllm_gguf_plugin.inkling.triton_conv import conv_decode
    torch.manual_seed(0)
    n_conv, total, slots = 4, 6656, 40
    ok = True
    for st_dtype, dim_first, (C, offset, dyn_heads), T, in_dt in itertools.product(
            (torch.bfloat16, torch.float32), (True, False), ((1024, 0, 8), (1024, 1024, 8), (2304, 2048, 0),
                                                             (2304, 4352, 0)), (1, 5, 32),
            (torch.float32, torch.bfloat16)):
        if dim_first:
            base = torch.randn(slots, total, n_conv - 1, device="cuda").to(st_dtype)
            st = base
        else:
            base = torch.randn(slots, n_conv - 1, total, device="cuda").to(st_dtype)
            st = base.transpose(1, 2)
        idx = (torch.randperm(slots - 1, device="cuda")[:T] + 1).to(torch.int32)
        if T > 2:
            idx[-1] = 0                                  # CUDA-graph padding row: the null block
        x = torch.randn(T, C, device="cuda")
        w = torch.randn(C, n_conv, device="cuda") * 0.5
        dyn = torch.randn(T, dyn_heads * n_conv, device="cuda") * 0.5 if dyn_heads else None
        md = SimpleNamespace(num_decodes=T, num_prefills=0, num_decode_tokens=T, num_prefill_tokens=0,
                             state_indices_tensor_d=idx.view(T, 1), query_start_loc_d=None)
        if in_dt != torch.float32:                       # model path: bf16 projections in, float32 out
            x, dyn = x.to(in_dt), (dyn.to(in_dt) if dyn is not None else None)
        st_ref = st.clone()
        ref = conv_step_batched(_State(st_ref, n_conv), x.float(), w, offset, md,
                                dyn.float() if dyn is not None else None, max(dyn_heads, 1))
        st_got = st.clone() if dim_first else base.clone().transpose(1, 2)
        got = conv_decode(st_got, x, w, offset, idx, dyn, max(dyn_heads, 1), torch.float32)
        if got.dtype != torch.float32:
            ok = False
        real = slice(0, T - 1) if T > 2 else slice(0, T)          # the null-block row is garbage by design
        e_out = float((got[real] - ref[real]).abs().max() / ref[real].abs().max())
        rows = idx[real].long()
        e_st = float((st_got[rows].float() - st_ref[rows].float()).abs().max())
        untouched = torch.ones(slots, dtype=torch.bool, device="cuda")
        untouched[idx.long()] = False
        e_rest = float((st_got[untouched].float() - st[untouched].float()).abs().max())
        tol = 1e-5 if st_dtype == torch.float32 else 1e-2
        good = e_out < tol and e_st == 0.0 and e_rest == 0.0
        ok &= good
        print(f"{str(st_dtype)[6:]:9s} {'DS' if dim_first else 'SD'} in {str(in_dt)[6:]:8s} C {C} off {offset} "
              f"dyn {dyn_heads} T {T:2d}: "
              f"out rel {e_out:.1e} state {e_st:.1e} others {e_rest:.1e} {'ok' if good else 'FAIL'}", flush=True)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
