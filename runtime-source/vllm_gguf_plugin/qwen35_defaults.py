# SPDX-License-Identifier: Apache-2.0
"""Bound initial resource use for local Qwen3.5 GGUF path-only inference."""

import sys

from vllm.config import CacheConfig
from vllm.logger import init_logger

logger = init_logger("vllm.gguf.defaults")


def configure_local_qwen35_defaults(args, *, model_family="Qwen3.5") -> None:
    """Apply only unset defaults after native-profile selection and GGUF inspection.

    Context auto-fit is a vLLM engine policy; the GGUF's advertised maximum
    remains unchanged. This function does not select a numerical backend.
    """
    changed = []
    if args.max_model_len is None:
        args.max_model_len = -1
        changed.append("max_model_len=auto")
    if args.max_num_seqs is None:
        # An explicit smaller token budget also bounds the sequence default.
        tokens = args.max_num_batched_tokens
        args.max_num_seqs = min(8, tokens) if tokens is not None and tokens > 0 else 8
        changed.append(f"max_num_seqs={args.max_num_seqs}")
    if args.max_num_batched_tokens is None:
        args.max_num_batched_tokens = max(512, args.max_num_seqs)
        changed.append(f"max_num_batched_tokens={args.max_num_batched_tokens}")
    if args.enable_prefix_caching is None:
        args.enable_prefix_caching = False
        changed.append("enable_prefix_caching=False")
    # Match native profile semantics: preserve an explicit CLI value, including
    # vLLM's default, or a non-default API value. EngineArgs cannot distinguish
    # an API caller explicitly passing the default from omitting it.
    if args.gpu_memory_utilization == CacheConfig.gpu_memory_utilization and not any(
        option.split("=")[0] == "--gpu-memory-utilization" for option in sys.argv
    ):
        args.gpu_memory_utilization = 0.65
        changed.append("gpu_memory_utilization=0.65")
    if changed:
        logger.info(
            "Local %s GGUF automatic serving defaults: %s. "
            "Explicit settings are preserved; the loaded weight backend is "
            "reported separately.",
            model_family,
            ", ".join(changed),
        )
