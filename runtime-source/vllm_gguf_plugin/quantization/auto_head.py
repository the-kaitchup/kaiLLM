# SPDX-License-Identifier: Apache-2.0
"""Load-time vocabulary backend selection by tensor and hardware capability."""

import hashlib
import os
from pathlib import Path

import torch


GENERIC_HEAD_TYPES = (2, 3, 6, 7, 8, 12)  # Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K


def choose_backend(
    qtype, shape, dtype, capability, *, canonical=True, q5_capabilities=((8, 6),)
):
    if not canonical or len(shape) != 2 or dtype not in (torch.float32, torch.bfloat16):
        return "compact"
    n, k = shape
    capability = tuple(capability)
    if qtype == 13 and capability in tuple(map(tuple, q5_capabilities)):
        return "q5-native-mma" if k % 512 == 0 and n % 4 == 0 else "compact"
    if (
        qtype in GENERIC_HEAD_TYPES
        and os.getenv("KAILLM_GENERIC_HEAD", "1") == "1"
        and capability == (8, 6)
        and dtype == torch.bfloat16
        and n >= 128
    ):
        # Any vocabulary in a format with native GGML MMVQ/MMQ kernels and the exact
        # power-of-two activation guard; dedicated Q3/Q5/Q6 routes are kept.
        return "generic-bf16"
    if tuple(shape) != (128000, 2048):
        return "compact"
    if qtype == 14:
        if capability == (8, 6):
            return "q6-bf16" if dtype == torch.bfloat16 else "q6-f32"
        if capability == (12, 0) and dtype == torch.bfloat16:
            return "q6-bf16-sm120"
    if qtype == 11 and dtype == torch.bfloat16 and capability in ((8, 6), (12, 0)):
        return "q3-bf16"
    return "compact"


def prepare_output_head(part, capability=None):
    """Prepare the existing backend; unsupported/unavailable capability falls back.

    This is called only by the existing output-vocabulary weight lifecycle.
    It does not modify embedding lookup, tensor values or any forward method.
    """
    from . import native_head

    if capability is None:
        if not part.weight.is_cuda or torch.version.hip is not None:
            return "compact"
        capability = torch.cuda.get_device_capability(part.weight.device)
    route = choose_backend(
        part.weight_type,
        part.shape,
        part.dtype,
        capability,
        canonical=part.canonical_rounding,
        q5_capabilities=native_head.NATIVE_HEAD_COMPUTE_CAPABILITIES,
    )
    if route == "q5-native-mma":
        part.prepare_native_head(mode="native-mma")
    elif route == "q6-bf16":
        # HEAD=auto-native authorizes the already-qualified head-only adapter
        # independently of whether ordinary-body adaptation was enabled.
        from .native_bf16 import PreparedBF16NativeLinear

        part._native_bf16_head = PreparedBF16NativeLinear(
            part, scope="q6-head", selected_mode="body-q6"
        )
    elif route == "generic-bf16":
        from .native_bf16 import PreparedBF16NativeLinear

        part._native_bf16_head = PreparedBF16NativeLinear(
            part, scope="generic-head", selected_mode="body-q6"
        )
    elif route == "q6-f32":
        part.prepare_native_q6_head()
        if part._native_q6_head is None:
            return "compact"
    elif route == "q6-bf16-sm120":
        method = getattr(part, "prepare_native_sm120_q6_head", None)
        if method is None:
            return "compact"
        method(selected_mode="body-q6")
        if getattr(part, "_native_sm120_q6_head", None) is None:
            return "compact"
    elif route == "q3-bf16":
        method = getattr(part, "prepare_native_bf16_q3_head", None)
        if method is None:
            return "compact"
        from .native_bf16_q3 import mode

        expected_target = "sm86" if tuple(capability) == (8, 6) else "sm120"
        if mode() != expected_target:
            return "compact"  # Target backend was not enabled for this engine.
        method()
        if getattr(part, "_native_bf16_q3_head", None) is None:
            return "compact"
    return route


def runtime_marker():
    """Bind auto policy and every available host dependency before model cache."""
    from . import native_head

    path = Path(__file__).resolve()
    names = (
        "native_head.py",
        "native_head_large.py",
        "vocal_embeds.py",
        "compact.py",
        "native_bf16.py",
        "native_q6_head.py",
        "native_q6_mmvq_mma.py",
        "native_bf16_q3.py",
        "native_q3_head.py",
        "native_q3_head_sm120.py",
        "native_bf16_sm120.py",
        "native_q6_head_sm120.py",
    )
    files = [
        path,
        *[path.with_name(name) for name in names if path.with_name(name).is_file()],
    ]
    marker = {
        "mode": "auto-native",
        "contract": "output-head-hardware-qtype-dtype-geometry-dispatch-v1",
        "dispatch_inputs": [
            "compute_capability",
            "qtype",
            "dtype",
            "N",
            "K",
            "canonical_rounding",
        ],
        "checkpoint_or_logical_batch_selection": False,
        "unsupported_heads_keep_compact": True,
        "embedding_lookup_unchanged": True,
        "Q6_BF16_head_prepared_independently_of_body_mode": True,
        "Q5_native_backend_capabilities": [
            list(c) for c in native_head.NATIVE_HEAD_COMPUTE_CAPABILITIES
        ],
        "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "persistent_additional_weight_bytes": 0,
        "whole_model_quality_required": True,
    }
    # These are host-only identities; no CUDA operation or device query occurs.
    marker["Q5_direct_library"] = native_head.native_head_library_identity()
    marker["Q5_mma_library"] = native_head.native_head_mma_library_identity()
    capability = native_head.library_compute_capability(marker["Q5_direct_library"])
    marker["selected_library_compute_capability"] = list(capability)
    if capability == (8, 6):
        from .native_q6_head import native_q6_head_cache_marker
    elif capability == (12, 0):
        from .native_q6_head_sm120 import native_q6_head_cache_marker
    else:
        raise ValueError("Unqualified selected Q6 backend target")
    marker["Q6_backend"] = native_q6_head_cache_marker()
    if path.with_name("native_bf16_q3.py").is_file():
        from .native_bf16_q3 import runtime_marker as q3_marker

        marker["Q3_backend"] = q3_marker()
    return marker
