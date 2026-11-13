# SPDX-License-Identifier: Apache-2.0

from importlib import import_module


def _register_environment_variables():
    import os
    from functools import partial

    from vllm import envs

    for name in (
        "VLLM_GGUF_Q6_PRECISION",
        "VLLM_GGUF_Q6_PRECISION_CACHE_BYTES",
        "VLLM_GGUF_Q6_PRECISION_HEAD_LIBRARY",
        "VLLM_GGUF_NATIVE_Q6_MMA_DEPENDENCY_DIR",
        "VLLM_GGUF_NATIVE_SM120_B2",
        "VLLM_GGUF_BACKEND",
        "VLLM_GGUF_BODY_PREFILL_CACHE",
        "VLLM_GGUF_BODY_PREFILL_CACHE_TOTAL_BYTES",
        "VLLM_GGUF_ENV",
        "VLLM_GGUF_LFM3_DUAL_RMS",
        "VLLM_GGUF_LFM3_HEAD",
        "VLLM_GGUF_LFM3_HEAD_VALIDATE_B1",
        "VLLM_GGUF_NATIVE_BF16_ADAPTER",
        "VLLM_GGUF_NATIVE_BF16_PADDING_CAST",
        "VLLM_GGUF_NATIVE_BF16_Q3_HEAD",
        "VLLM_GGUF_NATIVE_BF16_Q3_HEAD_ROWS",
        "VLLM_GGUF_NATIVE_BF16_Q8Q6_POWER2",
        "VLLM_GGUF_NATIVE_BODY",
        "VLLM_GGUF_NATIVE_BODY_HALF_POWER2",
        "VLLM_GGUF_NATIVE_HEAD_LIBRARY",
        "VLLM_GGUF_NATIVE_HEAD_MMA_LIBRARY",
        "VLLM_GGUF_NATIVE_MANIFEST",
        "VLLM_GGUF_NATIVE_Q3_HEAD_LIBRARY",
        "VLLM_GGUF_NATIVE_Q3_HEAD_SM120_LIBRARY",
        "VLLM_GGUF_NATIVE_Q3_HEAD_SM120_METADATA_SHA256",
        "VLLM_GGUF_NATIVE_Q3_HEAD_SM120_SHA256",
        "VLLM_GGUF_NATIVE_Q6_MMA_ROWS",
        "VLLM_GGUF_NATIVE_Q6_MMA_LIBRARY",
        "VLLM_GGUF_NATIVE_BODY_LIBRARY",
        "VLLM_GGUF_NATIVE_SM120_B1",
        "VLLM_GGUF_NATIVE_SM120_LIBRARY",
        "VLLM_GGUF_SM120_GEMV_POLICY",
        "VLLM_GGUF_SM120_VERIFY_POLICY",
        "VLLM_GGUF_VLLM_SOURCE",
        "VLLM_GGUF_YAQA_CUDA_ARCH",
        "VLLM_GGUF_YAQA_GEMM",
        "VLLM_GGUF_YAQA_FACTOR_STORAGE",
        "VLLM_GGUF_YAQA_RAW_FACTOR_POST",
        "VLLM_GGUF_YAQA_RAW_FACTOR_POST_K3_LIBRARY",
        "VLLM_GGUF_YAQA_RAW_FACTOR_POST_K4_LIBRARY",
        "VLLM_GGUF_YAQA_HIN_HIGH_LOW",
        "VLLM_GGUF_YAQA_HIN_HIGH_LOW_LIBRARY",
        "VLLM_GGUF_YAQA_HIN_HIGH_LOW_METADATA",
        "VLLM_GGUF_YAQA_REGISTER_PRECISION",
        "VLLM_GGUF_YAQA_REGISTER_EPILOGUE",
        "VLLM_GGUF_YAQA_REGISTER_EPILOGUE_LIBRARY",
        "VLLM_GGUF_YAQA_REGISTER_EPILOGUE_METADATA",
        "VLLM_GGUF_YAQA_GEMV",
        "VLLM_GGUF_YAQA_HADAMARD",
        "VLLM_GGUF_YAQA_POWER_INPUT_RESOURCE",
        "VLLM_GGUF_YAQA_POWER_INPUT_RESOURCE_LIBRARY",
        "VLLM_GGUF_YAQA_POWER_INPUT_RESOURCE_METADATA",
        "VLLM_GGUF_YAQA_GROUPED_POWER_INPUT_RESOURCE",
        "VLLM_GGUF_YAQA_GROUPED_POWER_INPUT_RESOURCE_LIBRARY",
        "VLLM_GGUF_YAQA_GROUPED_POWER_INPUT_RESOURCE_METADATA",
        "VLLM_GGUF_YAQA_GROUPED_HIN",
        "VLLM_GGUF_YAQA_INDEX40_CACHE",
        "VLLM_GGUF_YAQA_INDEX40_CACHE_BYTES",
        "VLLM_GGUF_YAQA_K1_POST_FUSION",
        "VLLM_GGUF_YAQA_OPAQUE",
        "VLLM_GGUF_YAQA_GROUPED",
        "VLLM_GGUF_YAQA_GROUPED_K234",
        "VLLM_GGUF_YAQA_GROUPED_CORE_LIBRARY",
        "VLLM_GGUF_YAQA_GROUPED_HIN_LIBRARY",
        "VLLM_GGUF_YAQA_RESIDENT_SMALL_DECODE",
        "VLLM_GGUF_YAQA_PREFILL",
        "VLLM_GGUF_YAQA_SM120_LONG_K_DECODE",
        "VLLM_GGUF_YAQA_SM120_LONG_K_SPLITK",
        "VLLM_GGUF_YAQA_PREFILL_CACHE",
        "VLLM_GGUF_YAQA_PREFILL_CACHE_BYTES",
        "VLLM_GGUF_YAQA_PREFILL_CACHE_MERGED_OUT",
        "VLLM_GGUF_YAQA_PREFILL_CACHE_ROWS",
        "VLLM_GGUF_YAQA_PREFILL_CPU_CACHE",
        "VLLM_GGUF_YAQA_PREFILL_CPU_CACHE_BYTES",
        "VLLM_GGUF_YAQA_PREFILL_CPU_CACHE_DIR",
        "VLLM_GGUF_YAQA_QKV_CACHE_MERGE",
        "VLLM_GGUF_YAQA_SCALAR",
        "VLLM_GGUF_YAQA_VALIDATE_B1",
    ):
        envs.environment_variables.setdefault(name, partial(os.getenv, name))


def register():
    _register_environment_variables()
    from .plugin import register as register_plugin

    return register_plugin()


def __getattr__(name):
    modules = {
        "GGUFConfigParser": ".config_parser",
        "GGUFModelLoader": ".loader",
        "OOTGGUFConfig": ".plugin",
        "OOTGGUFModelLoader": ".plugin",
        "DiffusionGGUFConfig": ".quantization",
        "GGUFConfig": ".quantization",
    }
    if name not in modules:
        raise AttributeError(name)
    return getattr(import_module(modules[name], __name__), name)


__all__ = [
    "DiffusionGGUFConfig",
    "GGUFConfig",
    "GGUFConfigParser",
    "GGUFModelLoader",
    "OOTGGUFConfig",
    "OOTGGUFModelLoader",
    "register",
]
