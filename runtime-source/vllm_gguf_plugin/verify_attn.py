# SPDX-License-Identifier: Apache-2.0
"""CUDA split-KV attention for speculative verification (KAILLM_VERIFY_ATTN_CUDA=1; native/kernels/kai_verify_attn.cu).

One CTA per (KV head, context slice) holds every query row of the KV head's GQA group (up to 16 verification tokens x
8 query heads), so each KV tile is read once; vLLM's Triton kernel needs two 64-row query blocks for a 16-token,
6-way-GQA verification.  Used from spec_attention's unified_attention wrapper for batch-1 BF16 verification calls
(2-16 query tokens, causal, no window / softcap / ALiBi / sinks / KV quantization); everything else stays on Triton.
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path

import torch

SOURCE = Path(__file__).parent / "native/kernels/kai_verify_attn.cu"
CONTRACT = "sm86-kai-verify-attn-v2-cuq"
_LIB = None
_BUFFERS: dict = {}
_REPORTED: list = []


def enabled() -> bool:
    return os.getenv("KAILLM_VERIFY_ATTN_CUDA", "0") == "1"


def _lib(device):
    global _LIB
    if _LIB is None:
        try:
            from .native.cuda_architecture import bind_library_architecture, device_architecture
            from .quantization.yaqa_cuda import build_cuda_library

            arch = device_architecture(torch.cuda.get_device_capability(device))
            identity = build_cuda_library(SOURCE, CONTRACT, "kai_verify_attn", architecture=arch)
            bind_library_architecture(json.dumps(identity, sort_keys=True), arch)
            lib = ctypes.CDLL(identity["library"])
            p, i, f, l = ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_long
            lib.kai_verify_attn.argtypes = [p, l, l, p, l, l, l, p, l, l, l, p, i, p, p, i, i, i, f, p, l, l, p, p,
                                            p, i, p]
            lib.kai_verify_attn.restype = i
            _LIB = lib
            print("kaiLLM: verification attention uses the CUDA split-KV kernel", file=sys.stderr, flush=True)
        except Exception as error:
            print(f"kaiLLM: CUDA verification attention unavailable ({error}); Triton stays", file=sys.stderr,
                  flush=True)
            _LIB = False
    return _LIB or None


def num_segments(device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    sms = torch.cuda.get_device_properties(idx).multi_processor_count
    return int(os.getenv("KAILLM_VERIFY_ATTN_SEGMENTS", "0")) or max(1, sms // 4)   # x 4 KV heads = one wave


def buffers(device, Hq: int, S: int):
    """Partial results for up to 16 query tokens (allocated outside CUDA graph capture, reused by every layer)."""
    key = (str(device), Hq, S)
    if key not in _BUFFERS:
        if torch.cuda.is_current_stream_capturing():
            return None
        _BUFFERS[key] = (torch.empty(16 * Hq * S * 256, dtype=torch.float32, device=device),
                         torch.empty(16 * Hq * S, dtype=torch.float32, device=device),
                         torch.empty(16 * Hq * S, dtype=torch.float32, device=device))
    return _BUFFERS[key]


def supports(q, k, v, out, cu_seqlens_q, max_seqlen_q, block_table) -> bool:
    if not (2 <= max_seqlen_q <= 16) or q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        return False
    if q.dim() != 3 or q.shape[-1] != 256 or k.shape[-1] != 256 or out.dtype != torch.bfloat16:
        return False
    if cu_seqlens_q.numel() != 2 or block_table.dim() != 2 or block_table.shape[0] != 1:
        return False
    Hq, Hk = q.shape[1], k.shape[2]
    if Hq % Hk or Hq // Hk not in (2, 4, 6, 8):
        return False
    # head rows contiguous and 16-byte aligned (cp.async); K / V may be interleaved per head (LBNHC: head stride 512)
    return (q.stride(2) == 1 and k.stride(3) == 1 and v.stride(3) == 1 and out.stride(2) == 1
            and all(x % 8 == 0 for x in (*k.stride()[:3], *v.stride()[:3])) and k.data_ptr() % 16 == 0
            and v.data_ptr() % 16 == 0 and q.data_ptr() % 4 == 0)


def verify_attn(q, k, v, out, seqused_k, max_seqlen_k, softmax_scale, block_table, max_seqlen_q,
                cu_seqlens_q=None) -> bool:
    """Run the CUDA kernel; False if it cannot (the caller then uses Triton)."""
    lib = _lib(q.device)
    if lib is None:
        return False
    Hq, Hk = q.shape[1], k.shape[2]
    S = num_segments(q.device)
    buf = buffers(q.device, Hq, S)
    if buf is None:
        return False
    part_o, part_m, part_l = buf
    T = int(max_seqlen_q)              # bound; the kernels read the actual query length from cu_seqlens_q
    if cu_seqlens_q is None:
        cu_seqlens_q = torch.tensor([0, T], dtype=torch.int32, device=q.device)
    rc = lib.kai_verify_attn(
        q.data_ptr(), q.stride(0), q.stride(1), k.data_ptr(), k.stride(0), k.stride(1), k.stride(2),
        v.data_ptr(), v.stride(0), v.stride(1), v.stride(2), block_table.data_ptr(), k.shape[1],
        seqused_k.data_ptr(), cu_seqlens_q.data_ptr(), T, Hq, Hk, float(softmax_scale), out.data_ptr(), out.stride(0),
        out.stride(1), part_o.data_ptr(), part_m.data_ptr(), part_l.data_ptr(), S,
        torch.cuda.current_stream(q.device).cuda_stream)
    if rc:
        raise RuntimeError(f"kai_verify_attn failed with CUDA error {rc}")
    return True


def wrap(ops) -> None:
    """Route eligible verification calls of ops.unified_attention to the CUDA kernel."""
    if getattr(ops, "_kai_verify_wrapped", False):
        return
    original = ops.unified_attention

    def unified_attention(q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k, softmax_scale, causal,
                          window_size, block_table, softcap, q_descale, k_descale, v_descale, *args, **kwargs):
        plain = (causal and (window_size is None or tuple(window_size) in ((-1, -1), (None, None)))
                 and not softcap and not args
                 and all(kwargs.get(n) is None for n in ("alibi_slopes", "qq_bias", "sinks", "mm_prefix_range",
                                                         "rswa_prefix_lens", "output_scale", "k_scale_cache",
                                                         "v_scale_cache"))
                 and int(getattr(kwargs.get("kv_quant_mode", 0), "value", kwargs.get("kv_quant_mode", 0)) or 0) == 0
                 and kwargs.get("chunk_lookback", -1) in (-1, None))
        ok = plain and supports(q, k, v, out, cu_seqlens_q, max_seqlen_q, block_table)
        if not _REPORTED and 2 <= int(max_seqlen_q) <= 16:
            _REPORTED.append(1)
            print(f"kaiLLM: first verification attention call: {'CUDA' if ok else 'Triton'} (plain={plain}, "
                  f"q {tuple(q.shape)} {q.dtype} {tuple(q.stride())}, k {tuple(k.shape)} {k.dtype} "
                  f"{tuple(k.stride())}, out {out.dtype}, cu {cu_seqlens_q.numel()}, bt {tuple(block_table.shape)}, "
                  f"window {window_size}, softcap {softcap}, extra args {len(args)}, "
                  f"kwargs {sorted(k_ for k_, v_ in kwargs.items() if v_ is not None)})", file=sys.stderr, flush=True)
        if ok:
            if verify_attn(q, k, v, out, seqused_k, max_seqlen_k, softmax_scale, block_table, max_seqlen_q,
                           cu_seqlens_q):
                return out
        return original(q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k, softmax_scale, causal,
                        window_size, block_table, softcap, q_descale, k_descale, v_descale, *args, **kwargs)

    ops.unified_attention = unified_attention
    ops._kai_verify_wrapped = True
    backend = sys.modules.get("vllm.v1.attention.backends.triton_attn")
    if backend is not None:
        backend.unified_attention = unified_attention
