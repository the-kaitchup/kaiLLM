# SPDX-License-Identifier: Apache-2.0
"""Bitwise cached QKV row join, explicit component API with no installed hooks."""

import torch
import triton

from vllm_gguf_plugin.quantization import yaqa_prefill_cache as cache

SHAPES = ((2048, 2048), (512, 2048), (512, 2048))
MODE = "physical-bf16-rn-sm120-small-decode"


def merge_cached_qkv(layer):
    parts = tuple(getattr(layer, "yaqa_parts", ()))
    if len(parts) != 3 or tuple(tuple(p.shape) for p in parts) != SHAPES:
        return False
    plan = getattr(parts[0], "prefill_cache_plan", None)
    if (
        plan is None
        or plan.mode != MODE
        or any(
            getattr(part, "prefill_cache_plan", None) is not plan
            or getattr(part, "codec", None) != "qtip"
            or not hasattr(part, "prefill_physical_weight")
            for part in parts
        )
    ):
        return False
    weights = [part.prefill_physical_weight for part in parts]
    device = weights[0].device
    if (
        device.type != "cuda"
        or torch.cuda.get_device_capability(device) != (12, 0)
        or any(
            w.device != device
            or w.dtype != torch.bfloat16
            or tuple(w.shape) != shape
            or not w.is_contiguous()
            for w, shape in zip(weights, SHAPES)
        )
    ):
        return False
    if hasattr(layer, "_yaqa_qkv_cache"):
        joint = layer._yaqa_qkv_cache
        offset = 0
        for weight in weights:
            assert weight.data_ptr() == joint[offset:].data_ptr()
            offset += weight.shape[0]
        return False
    joint = torch.cat(weights, dim=0)
    offset = 0
    for part, shape in zip(parts, SHAPES):
        part._buffers["prefill_physical_weight"] = joint[offset : offset + shape[0]]
        assert "prefill_physical_weight" in part._non_persistent_buffers_set
        offset += shape[0]
    layer.register_buffer("_yaqa_qkv_cache", joint, persistent=False)
    assert offset == 3072
    return True


@torch.library.custom_op("gguf_yaqa::sm120_cached_qkv", mutates_args=())
def fused_cached_qkv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if (
        x.device.type != "cuda"
        or torch.cuda.get_device_capability(x.device) != (12, 0)
        or x.dtype not in (torch.float32, torch.bfloat16)
        or weight.dtype != torch.bfloat16
        or weight.device != x.device
        or weight.shape != (3072, 2048)
        or not weight.is_contiguous()
        or x.shape[-1] != 2048
        or x.numel() // 2048 not in (1, 4)
    ):
        raise ValueError("Fused cache QKV requires qualified SM120 shape/dtype/rows")
    source = x.reshape(-1, 2048).contiguous()
    result = torch.empty((source.shape[0], 3072), device=x.device, dtype=x.dtype)
    if source.shape[0] == 1:
        cache._cached_gemv[(triton.cdiv(3072, 4),)](
            source,
            weight,
            result,
            M=3072,
            N=2048,
            R=4,
            K=2048,
            num_warps=4,
        )
    else:
        cache._cached_m4[(triton.cdiv(3072, 4),)](
            source,
            weight,
            result,
            M=3072,
            N=2048,
            R=4,
            K=2048,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return result.reshape(*x.shape[:-1], 3072)


@fused_cached_qkv.register_fake
def _fake(x, weight):
    return x.new_empty((*x.shape[:-1], 3072))
