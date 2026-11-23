# SPDX-License-Identifier: Apache-2.0
"""Test-only CPU stand-ins for kaiLLM's GGML ops (Triton / CUDA only): dequantize with gguf-py (or kaiLLM's torch
dequantization for the new types), then a float matmul.  install() patches vllm_gguf_plugin.ops in place."""

from __future__ import annotations

import torch


def _dequant(W, t, m, n, dtype):
    from vllm_gguf_plugin.inkling.ggml_types import NEW_TYPES, dequantize as dq_new
    if t in NEW_TYPES:
        return dq_new(W.reshape(-1), t).reshape(m, n).to(dtype or torch.float32)
    import gguf
    from gguf.quants import dequantize
    arr = dequantize(W.reshape(-1).cpu().numpy(), gguf.GGMLQuantizationType(t))
    return torch.from_numpy(arr).reshape(m, n).to(dtype or torch.float32)


def install():
    import gguf
    from vllm_gguf_plugin import ops

    def ggml_dequantize(W, quant_type, m, n, dtype):
        return _dequant(W, quant_type, m, n, dtype)

    def mm(W, X, quant_type, row):
        bs, ts = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(quant_type)]
        n = W.shape[1] // ts * bs
        return X @ _dequant(W, quant_type, row, n, X.dtype).t()

    ops.ggml_dequantize = ggml_dequantize
    ops.ggml_mul_mat_vec_a8 = mm
    ops.ggml_mul_mat_a8 = mm
    # the GGUF linear / embedding custom ops are registered for CUDA only: call their Python bodies
    from vllm_gguf_plugin import quantization as q
    from vllm_gguf_plugin.quantization import linear, vocal_embeds
    q.fused_mul_mat_gguf = linear._fused_mul_mat_gguf
    q.apply_gguf_embedding = vocal_embeds._apply_gguf_embedding
    from vllm_gguf_plugin.triton.dequantize import interface
    interface.ggml_dequantize_triton = ggml_dequantize          # K-quant embedding rows call it directly
