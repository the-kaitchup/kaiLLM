"""Select compatible native defaults before vLLM freezes configuration hashes."""

import dataclasses
import logging
import os
import sys

from .manifest import (
    checkpoint_digest,
    prepared_manifest,
    selected_profile,
    selection_marker,
)

logger = logging.getLogger("vllm.gguf.native")
MARKER = "gguf_native_runtime"


def _capability():
    # vLLM's CUDA platform uses NVML here; do not initialize a parent CUDA
    # context before the worker process is started.
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        return None
    result = current_platform.get_device_capability()
    return None if result is None else tuple(result)


def _incompatible(args):
    if args.quantization not in (None, "gguf") or args.load_format not in (
        "auto",
        "gguf",
    ):
        return "an explicit non-GGUF weight loader or quantization"
    if args.quantization_config is not None:
        return "an explicit quantization configuration"
    if (
        args.cudagraph_capture_sizes is not None
        or args.max_cudagraph_capture_size is not None
    ):
        return "explicit legacy CUDA graph capture limits"
    if args.model_weights and args.model_weights != args.model:
        return "a separate model weight path"
    if args.tokenizer is not None and args.tokenizer != args.model:
        return "an explicit tokenizer override"
    if str(args.dtype) not in ("auto", "float32", "float", "torch.float32"):
        return "an explicit dtype other than FP32"
    if args.enforce_eager:
        return "explicit eager execution"
    for name in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    ):
        if getattr(args, name, 1) != 1:
            return f"{name} must be one"
    for name in (
        "speculative_config",
        "spec_method",
        "spec_model",
        "spec_tokens",
        "enable_lora",
        "enable_dbo",
        "enable_expert_parallel",
        "cpu_offload_gb",
        "kv_transfer_config",
        "ec_transfer_config",
        "hf_overrides",
        "model_class_overrides",
        "enable_prompt_embeds",
        "kv_sharing_fast_prefill",
        "use_replayssm",
        "kv_offloading_size",
        "offload_group_size",
    ):
        if getattr(args, name, None):
            return f"{name} is outside the validated native scope"
    if args.enable_prefix_caching is True:
        return "explicit prefix caching"
    if args.hf_config_path is not None:
        return "an explicit HF configuration override"
    if args.runner not in ("auto", "generate") or args.convert not in ("auto", "none"):
        return "a non-generation runner"
    for name, maximum in (
        ("max_model_len", 4096),
        ("max_num_seqs", 8),
        ("max_num_batched_tokens", 512),
    ):
        value = getattr(args, name, None)
        if value is not None and not 0 < value <= maximum:
            return f"{name} is outside 1..{maximum}"
    if str(args.kv_cache_dtype) not in ("auto", "float32"):
        return "an explicit non-FP32 authoritative KV cache"
    if str(args.mamba_cache_dtype) not in ("auto", "float32") or str(
        args.mamba_ssm_cache_dtype
    ) not in ("auto", "float32"):
        return "an explicit non-FP32 recurrent cache"
    if os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH") == "1":
        return "explicit breakable CUDA graphs"
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") == "0":
        return "the legacy model runner"
    if (
        args.attention_backend is not None
        and str(args.attention_backend).split(".")[-1] != "TRITON_ATTN"
    ):
        return "an explicit attention backend other than TRITON_ATTN"
    backend = getattr(args.attention_config, "backend", None)
    if backend is not None and str(backend).split(".")[-1] != "TRITON_ATTN":
        return "an explicit attention backend other than TRITON_ATTN"
    maximum = args.max_num_seqs or 8
    if (
        args.max_num_batched_tokens is not None
        and args.max_num_batched_tokens < maximum
    ):
        return "max_num_batched_tokens is smaller than max_num_seqs"
    return None


def configure_engine_args(args, gguf_path, *, capability=None):
    """Return a selection marker, or keep the existing Humming configuration.

    Explicit unsupported choices are preserved: auto falls back with a reason;
    VLLM_GGUF_BACKEND=native raises instead. No installed manifest means opt-out.
    """
    requested = os.environ.get("VLLM_GGUF_BACKEND", "auto").lower()
    if requested not in ("auto", "native", "humming", "compact"):
        raise ValueError("VLLM_GGUF_BACKEND must be auto, native, humming, or compact")
    if requested in ("humming", "compact") or os.environ.get(
        "GGUF_OFFLINE_BENCH_BACKEND"
    ):
        return None
    if isinstance(args.additional_config, dict) and any(
        key.startswith("gguf_parity_") for key in args.additional_config
    ):
        # Independent diagnostics already own their arithmetic configuration.
        return None

    def fallback(reason):
        if requested == "native":
            raise ValueError(f"Cannot select native GGUF: {reason}")
        logger.info("GGUF uses the existing Humming path: %s", reason)
        return None

    try:
        bundle = prepared_manifest()
    except FileNotFoundError:
        if requested == "native":
            return fallback("no prepared native bundle is installed")
        return None
    except (OSError, ValueError) as error:
        return fallback(str(error))
    if args.additional_config is not None and not isinstance(
        args.additional_config, dict
    ):
        return fallback("additional_config must be a dictionary")
    reason = _incompatible(args)
    if reason:
        return fallback(reason)
    try:
        profile = selected_profile(
            bundle,
            checkpoint_digest(gguf_path),
            allow_experimental=requested == "native",
        )
    except (OSError, ValueError) as error:
        return fallback(str(error))
    if not profile["validated"]:
        logger.warning(
            "Using explicitly requested experimental native checkpoint %s; "
            "automatic selection remains disabled until validation passes",
            profile["profile"],
        )
    capability = _capability() if capability is None else tuple(capability)
    if capability is None or list(capability) not in bundle["compute_capabilities"]:
        return fallback(f"native bundle does not support GPU {capability}")
    if list(capability) not in bundle["validated_compute_capabilities"]:
        logger.warning(
            "Using explicitly enabled experimental native GPU build %s; "
            "only SM86 has been validated",
            capability,
        )

    from vllm.config import CacheConfig, CompilationConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from .kernels.native_gdn_capture import native_gdn_capture_compilation_config

    maximum = args.max_num_seqs or 8
    # Native GGML reductions can depend on the physical matrix row count.
    # Capture every supported decode batch instead of rounding B3 up to B4.
    sizes = list(range(1, maximum + 1))
    desired = native_gdn_capture_compilation_config(sizes)
    defaults = CompilationConfig()
    candidate = CompilationConfig(**desired)
    for field in dataclasses.fields(defaults):
        if not field.init or field.name.startswith("_"):
            continue
        current = getattr(args.compilation_config, field.name)
        if current != getattr(defaults, field.name) and current != getattr(
            candidate, field.name
        ):
            return fallback(f"explicit compilation option {field.name}")

    marker = selection_marker(bundle, profile)
    os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "0"
    args.dtype = "float32"
    args.max_model_len = args.max_model_len or 4096
    args.max_num_seqs = maximum
    args.max_num_batched_tokens = args.max_num_batched_tokens or 512
    args.enable_prefix_caching = False
    args.attention_backend = AttentionBackendEnum.TRITON_ATTN
    args.compilation_config = candidate
    # A CLI memory-budget override is preserved. API callers can provide a
    # non-default value; explicit defaults have the same auto-policy meaning.
    if args.gpu_memory_utilization == CacheConfig.gpu_memory_utilization and not any(
        option.split("=")[0] == "--gpu-memory-utilization" for option in sys.argv
    ):
        args.gpu_memory_utilization = 0.65
    additional = dict(args.additional_config or {})
    additional.update(
        gguf_native_runtime=marker,
        gguf_parity_capture_gdn=True,
        gguf_parity_prefill_tail=4,
        gguf_parity_native_batch_fa=True,
        gguf_parity_batch_revision=1,
        gguf_parity_persistent_kv_revision=1,
        gguf_parity_native_batch_fa_revision=2,
        gguf_parity_ssm_channels_revision=1,
    )
    if profile["features"]["body_mma"]:
        additional["gguf_parity_body_mma"] = True
        additional["gguf_parity_body_mma_revision"] = 3
    if profile["features"]["fused_prep"]:
        additional["gguf_parity_fused_prep_revision"] = 1
    if profile["features"]["attention_graphs"]:
        additional["gguf_parity_attn_decode_graphs_revision"] = 2
    if profile["features"]["affine_mma"]:
        additional["gguf_parity_affine_mma_revision"] = 1
    if profile["features"].get("prefill_batch_fa", False):
        additional["gguf_parity_native_prefill_batch_fa"] = True
        additional["gguf_parity_native_prefill_batch_fa_revision"] = 1
    if profile["features"].get("llama_batch_schedule", False):
        additional["gguf_parity_llama_batch_schedule"] = True
        additional["gguf_parity_llama_batch_schedule_revision"] = 1
    args.additional_config = additional
    logger.info(
        "GGUF selected native profile %s: FP32, context=%s, max_requests=%s, "
        "batched_tokens=%s, gpu_memory_utilization=%s, PIECEWISE graphs.",
        profile["profile"],
        args.max_model_len,
        maximum,
        args.max_num_batched_tokens,
        args.gpu_memory_utilization,
    )
    return marker
