# SPDX-License-Identifier: Apache-2.0
"""Lazy quantization exports; CPU metadata tools do not load CUDA kernels."""

from importlib import import_module

_EXPORTS = {
    "GGUFConfig": ("config", "GGUFConfig"),
    "DiffusionGGUFConfig": ("diffusion_config", "DiffusionGGUFConfig"),
    "DiffusionGGUFLinearMethod": ("diffusion_config", "DiffusionGGUFLinearMethod"),
    "dequant_gemm_gguf": ("diffusion_config", "dequant_gemm_gguf"),
    "GGUFMoEMethod": ("fused_moe", "GGUFMoEMethod"),
    "fused_moe_gguf": ("fused_moe", "fused_moe_gguf"),
    "GGUFLinearMethod": ("linear", "GGUFLinearMethod"),
    "fused_mul_mat_gguf": ("linear", "fused_mul_mat_gguf"),
    "GGUFUninitializedParameter": ("params", "GGUFUninitializedParameter"),
    "GGUFUninitializedWeightParameter": ("params", "GGUFUninitializedWeightParameter"),
    "GGUFUninitializedWeightTypeParameter": (
        "params",
        "GGUFUninitializedWeightTypeParameter",
    ),
    "GGUFWeightParameter": ("params", "GGUFWeightParameter"),
    "GGUFWeightTypeParameter": ("params", "GGUFWeightTypeParameter"),
    "DEQUANT_TYPES": ("utils", "DEQUANT_TYPES"),
    "IMATRIX_QUANT_TYPES": ("utils", "IMATRIX_QUANT_TYPES"),
    "KQUANT_TYPES": ("utils", "KQUANT_TYPES"),
    "MMQ_QUANT_TYPES": ("utils", "MMQ_QUANT_TYPES"),
    "MMVQ_QUANT_TYPES": ("utils", "MMVQ_QUANT_TYPES"),
    "STANDARD_QUANT_TYPES": ("utils", "STANDARD_QUANT_TYPES"),
    "UNQUANTIZED_TYPES": ("utils", "UNQUANTIZED_TYPES"),
    "is_layer_skipped_gguf": ("utils", "is_layer_skipped_gguf"),
    "GGUFEmbeddingMethod": ("vocal_embeds", "GGUFEmbeddingMethod"),
    "apply_gguf_embedding": ("vocal_embeds", "apply_gguf_embedding"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attr = _EXPORTS[name]
    value = getattr(import_module(f".{module}", __name__), attr)
    globals()[name] = value
    return value
