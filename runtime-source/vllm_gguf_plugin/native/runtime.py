"""Activate a verified native profile in the worker that owns its model.

The copied kernel adapters modify process-local classes. Selection therefore
happens from VllmConfig, before model construction, and a worker cannot switch
to another arithmetic profile after activation. Ordinary non-native workers
never install those adapters.
"""

import ctypes
import logging
import os
from functools import wraps

from .manifest import (
    ARCHITECTURE_DIMENSIONS,
    CHECKPOINTS,
    MODEL_SHA256,
    checkpoint_digest,
    prepared_manifest,
    selected_profile,
    selection_marker,
)
from .policy import MARKER

logger = logging.getLogger("vllm.gguf.native")
_active_marker = None
_preloaded = []


def selection(config):
    additional = config.additional_config
    return additional.get(MARKER) if isinstance(additional, dict) else None


def _configure_environment(bundle, features):
    """Set every native flag explicitly, including disabled profile choices."""
    os.environ.update(bundle["libraries"])
    flags = {
        "CURRENT_LLAMA": True,
        "CURRENT_DIRECT": True,
        "TRITON_IEEE": True,
        "NATIVE_ORDER_GDN": True,
        "NATIVE_NORM": True,
        "NATIVE_ATTN": True,
        "NATIVE_ROPE": True,
        "Q6_MMA_HEAD": features.get("q6_mma_head", True),
        "CAPTURE_GDN": True,
        "SHARED_Q8": True,
        "PERSISTENT_KV": True,
        "NATIVE_BATCH_FA": True,
        "BODY_MMA": features["body_mma"],
        "FUSED_PREP": features["fused_prep"],
        "ATTN_DECODE_GRAPHS": features["attention_graphs"],
        "AFFINE_MMA": features["affine_mma"],
        "NATIVE_PREFILL_BATCH_FA": features.get("prefill_batch_fa", False),
        "LLAMA_BATCH_SCHEDULE": features.get("llama_batch_schedule", False),
    }
    os.environ.update(
        {f"GGUF_PARITY_{key}": "1" if value else "0" for key, value in flags.items()}
    )
    os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "0"


def _validate_text_rope(rope):
    # vLLM's Qwen3_5ForCausalLMConfig removes BOTH multimodal fields during
    # full-engine configuration. Text positions are one-dimensional, so this
    # normalization preserves the verified 64-dimensional ordinary RoPE.
    expected = {
        "rope_type": "default",
        "rope_theta": 10000000.0,
        "partial_rotary_factor": 0.25,
    }
    if not isinstance(rope, dict) or any(
        rope.get(name) != value for name, value in expected.items()
    ):
        raise ValueError("Native profile requires the verified text RoPE configuration")
    multimodal = {"mrope_section", "mrope_interleaved"} & rope.keys()
    if multimodal and (
        multimodal != {"mrope_section", "mrope_interleaved"}
        or rope["mrope_section"] != [11, 11, 10]
        or rope["mrope_interleaved"] is not True
    ):
        raise ValueError("Native profile requires original or normalized text RoPE")
    if rope.get("rope_dim", 64) != 64 or rope.get("use_fope", False):
        raise ValueError("Native profile requires ordinary 64-dimensional text RoPE")


def validate_model_config(model_config, model_sha256=MODEL_SHA256):
    """Validate the actual parsed configuration, not just a filename."""
    import torch

    checkpoint = CHECKPOINTS.get(model_sha256)
    if checkpoint is None:
        raise ValueError("Native profile requires a registered checkpoint")
    text = model_config.hf_text_config
    expected = {
        "model_type": "qwen3_5_text",
        **ARCHITECTURE_DIMENSIONS[checkpoint["architecture"]],
        "num_hidden_layers": 24,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "vocab_size": 248320,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "hidden_act": "silu",
        "attention_bias": False,
        "tie_word_embeddings": True,
    }
    for name, value in expected.items():
        if getattr(text, name, None) != value:
            raise ValueError(f"Native profile requires {name}={value}")
    layers = [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(24)
    ]
    if text.layer_types != layers:
        raise ValueError("Native profile requires the verified 18 GDN / 6 FA layers")
    _validate_text_rope(text.rope_parameters)
    if model_config.dtype != torch.float32 or model_config.quantization != "gguf":
        raise ValueError("Native profile requires FP32 activations and GGUF weights")
    if model_config.max_model_len > 4096:
        raise ValueError("Native profile is limited to a 4096-token context")


def _validate_worker(vllm_config, model_config, marker, bundle):
    import torch

    if not isinstance(marker, dict):
        raise ValueError("Native configuration does not match its prepared manifest")
    profile = selected_profile(
        bundle,
        marker.get("model_sha256"),
        allow_experimental=marker.get("experimental_checkpoint") is True,
    )
    if marker != selection_marker(bundle, profile):
        raise ValueError("Native configuration does not match its prepared manifest")
    validate_model_config(model_config, profile["model_sha256"])
    if checkpoint_digest(model_config.model_weights) != profile["model_sha256"]:
        raise ValueError("Native model bytes changed after configuration selection")
    device = torch.device(vllm_config.device_config.device)
    if device.type != "cuda":
        raise ValueError("Native profile requires a CUDA worker")
    capability = tuple(torch.cuda.get_device_capability(device))
    if list(capability) not in bundle["compute_capabilities"]:
        raise ValueError(f"Native bundle does not support worker GPU {capability}")
    parallel = vllm_config.parallel_config
    if any(
        getattr(parallel, name, 1) != 1
        for name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
            "decode_context_parallel_size",
            "prefill_context_parallel_size",
        )
    ) or getattr(parallel, "enable_dbo", False):
        raise ValueError("Native profile requires one GPU without DBO")
    scheduler = vllm_config.scheduler_config
    cache = vllm_config.cache_config
    if not (
        1 <= scheduler.max_num_seqs <= 8
        and scheduler.max_num_seqs <= scheduler.max_num_batched_tokens <= 512
        and not cache.enable_prefix_caching
        and cache.cache_dtype in ("auto", "float32")
        and cache.mamba_cache_dtype in ("auto", "float32")
        and cache.mamba_ssm_cache_dtype in ("auto", "float32")
        and not vllm_config.offload_config.uva.cpu_offload_gb
        and not vllm_config.offload_config.prefetch.offload_group_size
        and vllm_config.speculative_config is None
        and vllm_config.lora_config is None
        and vllm_config.kv_transfer_config is None
    ):
        raise ValueError("Worker configuration is outside the native profile scope")


def validate_reload(model_config):
    """Keep an activated worker on its verified checkpoint during weight reload."""
    if _active_marker is not None:
        digest = _active_marker["model_sha256"]
        validate_model_config(model_config, digest)
        if checkpoint_digest(model_config.model_weights) != digest:
            raise ValueError("Native GGUF reload requires the same verified checkpoint")


def activate_for_model(vllm_config, model_config):
    """Install kernels only for a selected, validated worker configuration."""
    global _active_marker
    marker = selection(vllm_config)
    if marker is None:
        if _active_marker is not None:
            raise RuntimeError(
                "A native GGUF worker cannot load another backend in the same "
                "process; create a fresh engine worker"
            )
        return False
    if not isinstance(marker, dict) or not isinstance(marker.get("manifest"), str):
        raise ValueError("Invalid native GGUF configuration marker")
    bundle = prepared_manifest(marker["manifest"])
    _validate_worker(vllm_config, model_config, marker, bundle)
    profile = selected_profile(
        bundle,
        marker["model_sha256"],
        allow_experimental=marker.get("experimental_checkpoint") is True,
    )
    import torch

    from .kernels.native_gdn_capture import _validate as validate_capture

    validate_capture(vllm_config, torch.device(vllm_config.device_config.device))
    if _active_marker is not None:
        if marker != _active_marker:
            raise RuntimeError("Cannot change the native profile in an active worker")
        return True

    _configure_environment(bundle, profile["features"])
    for path in bundle["preloads"]:
        _preloaded.append(ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL))

    from .kernels.current_ggml_attention import install_native_attention
    from .kernels.current_ggml_attention_batch import library_provenance
    from .kernels.current_ggml_norm import install_native_norms
    from .kernels.current_ggml_rope import install_native_rope
    from .kernels.current_llama_vllm_adapter import install_current_llama_adapter
    from .kernels.native_gdn_capture import install_native_gdn_capture
    from .kernels.native_order_gdn import install_native_order_gdn

    library_provenance()  # Includes the corrected physical attention-mask ABI.
    # Mark before installing: a partial failed installation cannot safely be
    # reused with a different model in this process.
    _active_marker = marker
    install_current_llama_adapter()
    install_native_norms()
    install_native_order_gdn()
    install_native_attention()
    install_native_rope()
    install_native_gdn_capture()
    logger.info(
        "Activated verified native GGUF runtime %s (%s)",
        profile["profile"],
        bundle["sha256"],
    )
    return True


def install_scheduler_hook():
    """Match llama-server's two final prompt checkpoints for selected engines."""
    from vllm.v1.core.sched.scheduler import Scheduler

    from .scheduling import install_llama_batch_schedule, reserve_prefill_boundary

    original = Scheduler._reserve_prefill_lookahead
    if getattr(original, "_gguf_native_tail_hook", False):
        install_llama_batch_schedule()
        return

    @wraps(original)
    def reserve(self, request, num_computed_tokens, num_new_tokens):
        count = original(self, request, num_computed_tokens, num_new_tokens)
        if selection(self.vllm_config) is not None:
            if self.num_prefill_lookahead > 1:
                raise ValueError("Native GGUF prompt checkpoint does not support MTP")
            count = reserve_prefill_boundary(
                num_computed_tokens, request.num_prompt_tokens, count
            )
        return count

    reserve._gguf_native_tail_hook = True
    Scheduler._reserve_prefill_lookahead = reserve
    install_llama_batch_schedule()
