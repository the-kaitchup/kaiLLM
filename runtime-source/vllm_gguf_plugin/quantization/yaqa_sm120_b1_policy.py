# SPDX-License-Identifier: Apache-2.0
"""Measured SM120 policies; opt-in and captured before graph construction."""

import hashlib
import os
from pathlib import Path


def measured_gemv_enabled():
    value = os.getenv("VLLM_GGUF_SM120_GEMV_POLICY", "off")
    if value not in ("off", "measured-b1-v1"):
        raise ValueError("VLLM_GGUF_SM120_GEMV_POLICY must be off or measured-b1-v1")
    return value == "measured-b1-v1"


def fused_post_enabled():
    value = os.getenv("VLLM_GGUF_YAQA_K1_POST_FUSION", "0")
    if value not in ("0", "1"):
        raise ValueError("VLLM_GGUF_YAQA_K1_POST_FUSION must be0 or1")
    return value == "1"


def runtime_marker():
    return {
        "gemv": measured_gemv_enabled(),
        "K1_post": fused_post_enabled(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


_POLICIES = {
    (3, 16, 2048): (4, 8, 16, 0),
    (4, 16, 2048): (8, 8, 16, 0),
    (3, 512, 2048): (8, 8, 16, 1),
    (4, 512, 2048): (4, 8, 16, 0),
    # The old split8 square winners equal split16 timing, with smaller workspace.
    (3, 2048, 2048): (8, 8, 8, 1),
    (4, 2048, 2048): (8, 8, 8, 1),
    (3, 6144, 2048): (8, 8, 16, 1),
    (4, 6144, 2048): (4, 4, 16, 1),
    (3, 7680, 2048): (8, 8, 16, 1),
    (4, 7680, 2048): (8, 4, 16, 1),
    (3, 2048, 7680): (8, 8, 32, 1),
    (4, 2048, 7680): (4, 4, 32, 1),
}


def gemv_policy(bits, m, n):
    return _POLICIES.get((bits, m, n))


def k1_post_eligible(batch, m, dtype, global_dtype):
    # N2048/B1/F64 is a measured loser, so preserve its two original kernels.
    if str(dtype) not in ("torch.float32", "torch.bfloat16"):
        return False
    if str(global_dtype) not in ("torch.float32", "torch.float64"):
        return False
    if (
        batch > 0
        and m in (16, 512)
        and batch * m <= (1 << 31) - 1
        and str(global_dtype) == "torch.float32"
    ):
        return True
    if batch == 1:
        return m in (16, 512) or (
            m == 2048
            and str(dtype) == "torch.float32"
            and str(global_dtype) == "torch.float32"
        )
    if batch == 512:
        return m == 2048
    if batch == 2048:
        return m in (512, 2048)
    return False


def k1_post_power_resource(
    enabled, capability, output_width, factor_shape, dtype, global_dtype
):
    """Unselected small-row resource family; admission uses actual tensor types."""
    return (
        16
        if (
            enabled
            and capability == (12, 0)
            and output_width == 2048
            and tuple(factor_shape) == (1, 1)
            and str(dtype) == "torch.bfloat16"
            and str(global_dtype) == "torch.float32"
        )
        else None
    )
