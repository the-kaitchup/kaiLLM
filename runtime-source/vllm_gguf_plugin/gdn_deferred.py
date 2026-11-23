# SPDX-License-Identifier: Apache-2.0
"""Deferred GDN state commit for speculative verification (batch 1; KAILLM_GDN_DEFERRED=1).

vLLM's fused GDN spec-decode kernel stores the recurrent state after every token of a verification step (one state
slot per token, so that the next step can start from the state after the last accepted token): 16 tokens x 48 value
heads x 128 x 128 x 2 bytes x 48 layers = 1.2 GB of writes per DSpine 15 step.  This kernel instead keeps the step's
inputs (k, v, a, b; v is per value head, k / a / b per head) in small double-buffered side buffers and, at the start of
the next step, recomputes the state after the accepted tokens from the step's start state, which it stored once.

Bit-exact with vLLM's Triton port (same update code; the recomputed state is rounded to the state dtype exactly where
the reference stores it), so outputs and the state vLLM reads from slot[accepted - 1] are unchanged.  Per step and
layer: one state read + one state write instead of one read + T writes.

Bookkeeping is per program (base slot, previous token count, epoch, buffer parity): each program reads and writes
only its own entries, so programs of one launch never race.  A prefill or plain decode of the layer bumps a global
epoch, which invalidates the side buffers (the next verification then starts from vLLM's own source slot).

Single-slot mode (KAILLM_GDN_SINGLE_SLOT=1, see gdn_spec): vLLM allocates one state block per GDN layer and request
instead of 1 + num_speculative_tokens, and every column of the verification's state indices names that block.  The
kernel then recomputes the accepted state in place (each program reads its whole state tile before writing it back),
so the slot always holds the state at the start of the current step.  With DSpine 7 on a 12 GB GPU the per-token
slots took 80 of 110 KV-cache blocks.
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path

import torch

from vllm.triton_utils import tl, triton

_EPOCH: torch.Tensor | None = None
_CUDA = None          # kai_gdn_deferred.cu (KAILLM_GDN_DEFERRED_CUDA=0 keeps the Triton kernel)
CUDA_SOURCE = Path(__file__).parent / "native/kernels/kai_gdn_deferred.cu"
CUDA_CONTRACT = "sm86-kai-gdn-deferred-v2-fp32state"


def _cuda_lib(device):
    global _CUDA
    if os.getenv("KAILLM_GDN_DEFERRED_CUDA", "1") == "0":
        return None
    if _CUDA is None:
        try:
            from .native.cuda_architecture import bind_library_architecture, device_architecture
            from .quantization.yaqa_cuda import build_cuda_library

            arch = device_architecture(torch.cuda.get_device_capability(device))
            identity = build_cuda_library(CUDA_SOURCE, CUDA_CONTRACT, "kai_gdn_deferred", architecture=arch)
            bind_library_architecture(json.dumps(identity, sort_keys=True), arch)
            lib = ctypes.CDLL(identity["library"])
            p, i, f, l = ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_long
            lib.kai_gdn_deferred.argtypes = [p, l, p, l, p, l, p, p, p, l, p, i, p, p, p, l, p, f, i, p, p, p, p, p,
                                             p, p, i, i, i, f, i, i, p]
            lib.kai_gdn_deferred.restype = i
            _CUDA = lib
            print("kaiLLM: GDN verification uses the CUDA deferred kernel", file=sys.stderr, flush=True)
        except Exception as error:      # no nvcc / unsupported device: the Triton kernel stays
            print(f"kaiLLM: CUDA gdn_deferred unavailable ({error}); using the Triton kernel", file=sys.stderr,
                  flush=True)
            _CUDA = False
    return _CUDA or None


def _f32(ctx, name, t):
    """Float32 copy of a small per-layer parameter, cached on the context."""
    cache = ctx.setdefault("f32", {})
    hit = cache.get(name)
    if hit is None or hit[0] != t.data_ptr():
        hit = (t.data_ptr(), t.detach().float().contiguous())
        cache[name] = hit
    return hit[1]


def epoch(device) -> torch.Tensor:
    global _EPOCH
    if _EPOCH is None:
        _EPOCH = torch.ones(1, dtype=torch.int32, device=device)
    return _EPOCH


def invalidate() -> None:
    if _EPOCH is not None:
        _EPOCH.add_(1)


def make_context(H, K, HV, V, maxt, dtype, device, single_slot=False):
    """Side buffers and per-program bookkeeping for one GDN layer.  Allocated (and zero-filled) eagerly at model
    construction: allocated lazily inside a CUDA graph capture, the fills would be captured and replayed every step."""
    from vllm.utils.platform_utils import num_compute_units

    idx = device.index if device.index is not None else torch.cuda.current_device()
    num_cus = num_compute_units(idx)
    bv = 32 if HV < num_cus else 64 if HV <= 4 * num_cus else V          # the reference's V tiling (same sums)
    epoch(device)
    return {
        "shape": (H, K, HV, V, maxt), "bv": bv, "single_slot": bool(single_slot),
        "k": torch.zeros(2, maxt, H * K, dtype=dtype, device=device),
        "v": torch.zeros(2, maxt, HV * V, dtype=dtype, device=device),
        "a": torch.zeros(2, maxt, HV, dtype=dtype, device=device),
        "b": torch.zeros(2, maxt, HV, dtype=dtype, device=device),
        "book": torch.zeros(HV * (V // bv), 4, dtype=torch.int32, device=device),
        "book_cuda": torch.zeros(HV, 4, dtype=torch.int32, device=device),      # one CUDA program per value head
    }


@triton.jit
def _gdn_deferred_update_kernel(
    mixed_qkv, b, a, A_log, dt_bias, state, state_indices, cu_seqlens, num_accepted_tokens, out, sum_sq,
    side_k, side_v, side_a, side_b, book, epoch_ptr,
    stride_qkv_row, stride_b_row, stride_a_row, stride_state_slot, num_indices_per_row, scale,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr, MAXT: tl.constexpr,
    SINGLE_SLOT: tl.constexpr = False,
):
    i_v, i_hv = tl.program_id(0), tl.program_id(1)
    NV: tl.constexpr = V // BV
    G: tl.constexpr = HV // H
    i_h = i_hv // G
    pid = i_hv * NV + i_v
    bos = tl.load(cu_seqlens).to(tl.int64)
    num_tokens = tl.load(cu_seqlens + 1).to(tl.int64) - bos
    if num_tokens <= 0:
        return

    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    p_out = out + (bos * HV + i_hv) * V + o_v
    p_sum_sq = sum_sq + (bos * HV + i_hv) * NV + i_v
    p_state = i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    neg_A = -tl.exp(tl.load(A_log + i_hv).to(tl.float32))
    bias = tl.load(dt_bias + i_hv).to(tl.float32)
    accepted = tl.load(num_accepted_tokens).to(tl.int64)
    cur_epoch = tl.load(epoch_ptr)
    base = tl.load(book + pid * 4 + 0).to(tl.int64)
    n_prev = tl.load(book + pid * 4 + 1).to(tl.int64)
    valid = (tl.load(book + pid * 4 + 2) == cur_epoch) & (n_prev > 0) & (base > 0)
    par = tl.where(valid, tl.load(book + pid * 4 + 3), 0)
    if SINGLE_SLOT:
        # one block per request: every column names it; the accepted state is recomputed in place
        dst = tl.load(state_indices).to(tl.int64)
        valid = valid & (base == dst)
    else:
        in_range = (accepted > 0) & (accepted <= num_indices_per_row)
        dst = tl.load(state_indices + accepted - 1, mask=in_range, other=0).to(tl.int64)
    if dst <= 0:
        # no state (padding / dummy runs): zero outputs as the reference, invalidate the side buffers
        for t in range(num_tokens):
            tl.store(p_out + t * HV * V, tl.zeros([BV], dtype=out.dtype.element_ty))
            tl.store(p_sum_sq + t * HV * NV, 0.0)
        tl.store(book + pid * 4 + 1, 0)
        return

    if valid:
        # State after the previous step's accepted tokens, from that step's start state and its inputs.
        b_h = tl.load(state + base * stride_state_slot + p_state).to(tl.float32)
        for t in range(tl.minimum(accepted, n_prev)):
            off = par * MAXT + t
            b_k = tl.load(side_k + off * (H * K) + i_h * K + o_k).to(tl.float32)
            b_v = tl.load(side_v + off * (HV * V) + i_hv * V + o_v).to(tl.float32)
            x = tl.load(side_a + off * HV + i_hv).to(tl.float32) + bias
            decay = tl.exp(neg_A * tl.where(x <= 20.0, tl.log(1 + tl.exp(x)), x))
            beta = tl.sigmoid(tl.load(side_b + off * HV + i_hv).to(tl.float32))
            b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
            b_h *= decay
            b_v = (b_v - tl.sum(b_h * b_k[None, :], 1)) * beta
            b_h += b_v[:, None] * b_k[None, :]
        b_h = b_h.to(state.dtype.element_ty).to(tl.float32)     # the reference stored and reloaded it here
    else:
        b_h = tl.load(state + dst * stride_state_slot + p_state).to(tl.float32)
    tl.store(state + dst * stride_state_slot + p_state, b_h.to(state.dtype.element_ty))

    w_par = 1 - par
    for t in range(num_tokens):
        row = bos + t
        p_qkv = mixed_qkv + row * stride_qkv_row
        raw_k = tl.load(p_qkv + (H + i_h) * K + o_k)
        raw_v = tl.load(p_qkv + 2 * H * K + i_hv * V + o_v)
        raw_a = tl.load(a + row * stride_a_row + i_hv)
        raw_b = tl.load(b + row * stride_b_row + i_hv)
        woff = w_par * MAXT + t
        tl.store(side_v + woff * (HV * V) + i_hv * V + o_v, raw_v)
        if i_v == 0:
            tl.store(side_a + woff * HV + i_hv, raw_a)
            tl.store(side_b + woff * HV + i_hv, raw_b)
            if i_hv % G == 0:
                tl.store(side_k + woff * (H * K) + i_h * K + o_k, raw_k)
        b_q = tl.load(p_qkv + i_h * K + o_k).to(tl.float32)
        b_k = raw_k.to(tl.float32)
        b_v = raw_v.to(tl.float32)
        x = raw_a.to(tl.float32) + bias
        decay = tl.exp(neg_A * tl.where(x <= 20.0, tl.log(1 + tl.exp(x)), x))
        beta = tl.sigmoid(raw_b.to(tl.float32))
        b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6) * scale
        b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
        b_h *= decay
        b_v = (b_v - tl.sum(b_h * b_k[None, :], 1)) * beta
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1).to(out.dtype.element_ty)
        tl.store(p_out + t * HV * V, b_o)
        b_o = b_o.to(tl.float32)
        tl.store(p_sum_sq + t * HV * NV, tl.sum(b_o * b_o))

    tl.store(book + pid * 4 + 0, dst.to(tl.int32))
    tl.store(book + pid * 4 + 1, num_tokens.to(tl.int32))
    tl.store(book + pid * 4 + 2, cur_epoch)
    tl.store(book + pid * 4 + 3, w_par)


def supports(ctx, mixed_qkv, a, b, state, state_indices, cu_seqlens, out) -> bool:
    if ctx is None or cu_seqlens.numel() != 2 or state_indices.shape[0] != 1:
        return False
    HV, V = out.shape[-2:]
    K = state.shape[-1]
    H = (mixed_qkv.shape[-1] - HV * V) // (2 * K)
    return (ctx["shape"] == (H, K, HV, V, state_indices.shape[1]) and ctx["k"].dtype == mixed_qkv.dtype
            and ctx["a"].dtype == a.dtype == b.dtype and mixed_qkv.stride(1) == 1 and a.stride(1) == b.stride(1) == 1
            and state.stride(1) == V * K and state.stride(2) == K and state.stride(3) == 1 and out.is_contiguous())


def _cuda_supports(mixed_qkv, a, b, state, state_indices, cu_seqlens, num_accepted_tokens, output_gate, out):
    bf = torch.bfloat16
    i32 = torch.int32
    return (mixed_qkv.dtype == a.dtype == b.dtype == output_gate.dtype == out.dtype == bf
            and state.dtype in (bf, torch.float32)
            and state_indices.dtype == cu_seqlens.dtype == num_accepted_tokens.dtype == i32
            and output_gate.dim() == 3 and output_gate.stride(2) == 1 and output_gate.stride(1) == output_gate.shape[2]
            and state_indices.stride(1) in (0, 1) and mixed_qkv.data_ptr() % 16 == 0 and mixed_qkv.stride(0) % 8 == 0
            and state.data_ptr() % 16 == 0)


def fused_gdn_decode_deferred(ctx, mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens,
                              state, output_gate, norm_weight, out, scale, norm_eps, output_gate_activation):
    """Same arguments and result as vLLM's fused_gdn_decode_post_conv_mtp_triton, for one request (see supports)."""
    from vllm.third_party.flash_linear_attention.ops.fused_gdn_decode_post_conv_mtp import (
        _gdn_mtp_gated_rmsnorm_kernel,
    )

    H, K, HV, V, maxt = ctx["shape"]
    lib = _cuda_lib(out.device) if (K == 128 and V == 128 and maxt <= 16) else None
    use_cuda = lib is not None and _cuda_supports(mixed_qkv, a, b, state, state_indices, cu_seqlens,
                                                  num_accepted_tokens, output_gate, out)
    if not ctx.get("reported"):
        ctx["reported"] = True
        if use_cuda or lib is None:
            print(f"kaiLLM: gdn_deferred first call: {'CUDA' if use_cuda else 'Triton'} kernel, "
                  f"{mixed_qkv.shape[0]} tokens", file=sys.stderr, flush=True)
        else:
            print("kaiLLM: gdn_deferred first call: Triton fallback (CUDA kernel unsupported for: "
                  f"qkv {mixed_qkv.dtype} {tuple(mixed_qkv.stride())} ptr%16={mixed_qkv.data_ptr() % 16}, "
                  f"a {a.dtype} b {b.dtype} state {state.dtype} gate {output_gate.dtype} {tuple(output_gate.shape)} "
                  f"{tuple(output_gate.stride())}, out {out.dtype}, idx {state_indices.dtype} "
                  f"{tuple(state_indices.stride())}, cu {cu_seqlens.dtype}, acc {num_accepted_tokens.dtype})",
                  file=sys.stderr, flush=True)
    if use_cuda:
        a_log32, dt_bias32, norm32 = _f32(ctx, "a_log", A_log), _f32(ctx, "dt_bias", dt_bias), _f32(ctx, "norm",
                                                                                                    norm_weight)
        rc = lib.kai_gdn_deferred(
            mixed_qkv.data_ptr(), mixed_qkv.stride(0), a.data_ptr(), a.stride(0), b.data_ptr(), b.stride(0),
            a_log32.data_ptr(), dt_bias32.data_ptr(), state.data_ptr(), state.stride(0), state_indices.data_ptr(),
            state_indices.shape[1], cu_seqlens.data_ptr(), num_accepted_tokens.data_ptr(), output_gate.data_ptr(),
            output_gate.stride(0), norm32.data_ptr(), norm_eps, 1 if output_gate_activation == "sigmoid" else 0,
            out.data_ptr(), ctx["k"].data_ptr(), ctx["v"].data_ptr(), ctx["a"].data_ptr(), ctx["b"].data_ptr(),
            ctx["book_cuda"].data_ptr(), epoch(state.device).data_ptr(), H, HV, maxt, scale,
            1 if ctx["single_slot"] else 0, 1 if state.dtype == torch.float32 else 0,
            torch.cuda.current_stream(out.device).cuda_stream)
        if rc:
            raise RuntimeError(f"kai_gdn_deferred failed with CUDA error {rc}")
        return out
    bv = ctx["bv"]
    sum_sq = torch.empty((out.shape[0], HV, V // bv), dtype=torch.float32, device=out.device)
    _gdn_deferred_update_kernel[(V // bv, HV)](
        mixed_qkv, b, a, A_log, dt_bias, state, state_indices, cu_seqlens, num_accepted_tokens, out, sum_sq,
        ctx["k"], ctx["v"], ctx["a"], ctx["b"], ctx["book"], epoch(state.device),
        mixed_qkv.stride(0), b.stride(0), a.stride(0), state.stride(0), state_indices.shape[1], scale,
        H=H, HV=HV, K=K, V=V, BV=bv, MAXT=maxt, SINGLE_SLOT=ctx["single_slot"], num_warps=4,
    )
    _gdn_mtp_gated_rmsnorm_kernel[(1, HV, maxt)](
        out, sum_sq, output_gate, norm_weight, cu_seqlens, output_gate.stride(0), norm_eps,
        HV=HV, V=V, NV=V // bv, SIGMOID_GATE=output_gate_activation == "sigmoid", num_warps=2,
    )
    return out
