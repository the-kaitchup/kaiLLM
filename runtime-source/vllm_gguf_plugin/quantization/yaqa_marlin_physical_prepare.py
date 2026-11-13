# SPDX-License-Identifier: Apache-2.0
"""Load-time, separately budgeted Marlin cache of canonical physical weights."""

import os
from dataclasses import dataclass

ENV = "VLLM_GGUF_YAQA_MARLIN_PHYSICAL"
BYTES_ENV = ENV + "_BYTES"
HEADROOM_ENV = ENV + "_HEADROOM_BYTES"
BUFFER_PREFIX = "_yaqa_marlin_physical_"
GROUP_SIZE = 32


def settings():
    mode = os.getenv(ENV, "none")
    if mode not in ("none", "auto"):
        raise ValueError(f"{ENV} requires none or auto")
    if mode == "none":
        return None
    if os.getenv("VLLM_MARLIN_USE_ATOMIC_ADD", "0") != "0":
        raise ValueError("Physical Marlin cache requires VLLM_MARLIN_USE_ATOMIC_ADD=0")
    values = []
    for name in (BYTES_ENV, HEADROOM_ENV):
        raw = os.getenv(name)
        if raw is None:
            raise ValueError(f"Enabled Marlin cache requires explicit {name}")
        try:
            value = int(raw)
        except ValueError as error:
            raise ValueError(f"{name} requires an integer byte count") from error
        if value < 0 or (name == BYTES_ENV and value == 0):
            raise ValueError(f"Invalid Marlin cache byte policy: {name}")
        values.append(value)
    return tuple(values)


@dataclass
class PreparedMarlinPhysical:
    wire: object
    scales: object
    zeros: object
    k: int
    n: int
    part_count: int
    device: object
    parts: tuple
    part_keys: tuple
    plan: object
    padded_k: int
    padded_n: int
    workspace: object = None
    key: object = None
    fallback: object = None
    buffers: tuple = ()


class MarlinPlan:
    """Full additional payload reservation; the BF16 plan is never modified."""

    def __init__(self, entries, bf16_plan, policy):
        self.entries = entries
        self.bf16_plan = bf16_plan
        self.policy = policy
        self.budget, self.headroom = policy
        self.expected_bytes = sum(row["bytes"] for row in entries.values())
        self.prepared = {}
        self.prepared_bytes = 0
        self.admitted = None
        self.device = None
        self.memory = None
        self.disabled_reason = None
        self.ready = False
        if self.expected_bytes > self.budget:
            self.admitted = False
            self.disabled_reason = "complete additional payload exceeds explicit cap"

    def admit(self, device, torch):
        if self.admitted is not None:
            if self.admitted and self.device != device:
                raise ValueError("One Marlin plan cannot span CUDA devices")
            return self.admitted
        if device.type != "cuda" or tuple(torch.cuda.get_device_capability(device)) != (12, 0):
            self.admitted = False
            self.disabled_reason = "actual device is outside the SM120 policy"
            return False
        free, total = torch.cuda.mem_get_info(device)
        remaining_bf16 = self.bf16_plan.expected_bytes - sum(
            row["bytes"] for row in self.bf16_plan.loaded.values()
        )
        if remaining_bf16 < 0:
            raise ValueError("Existing complete BF16 plan has inconsistent byte counts")
        required = self.expected_bytes + remaining_bf16 + self.headroom
        self.memory = {
            "free_bytes_before_Marlin": int(free),
            "total_device_bytes": int(total),
            "remaining_BF16_reservation_bytes": remaining_bf16,
            "additional_Marlin_reservation_bytes": self.expected_bytes,
            "headroom_bytes": self.headroom,
            "required_free_bytes": required,
            "headroom_scope": "KV, pending original weights, profile/workspace and preparation temporaries",
        }
        self.device = device
        self.admitted = int(free) >= required
        if not self.admitted:
            self.disabled_reason = "complete additional reservation and headroom do not fit"
        return self.admitted

    def finish(self, key, actual_bytes):
        if key in self.prepared or actual_bytes != self.entries[key]["bytes"]:
            raise ValueError("Marlin group allocation differs from its static reservation")
        self.prepared[key] = actual_bytes
        self.prepared_bytes += actual_bytes

    def ensure_complete(self):
        if self.admitted is False:
            return
        if self.admitted is not True or self.prepared.keys() != self.entries.keys():
            raise ValueError("Incomplete admitted Marlin cache before inference")
        if self.prepared_bytes != self.expected_bytes:
            raise ValueError("Marlin cache bytes differ from complete reservation")
        self.ready = True

    def summary(self):
        return {
            "mode": "auto",
            "admitted": self.admitted,
            "ready": self.ready,
            "disabled_reason": self.disabled_reason,
            "budget_bytes": self.budget,
            "headroom_bytes": self.headroom,
            "expected_groups": len(self.entries),
            "prepared_groups": len(self.prepared),
            "expected_bytes": self.expected_bytes,
            "prepared_bytes": self.prepared_bytes,
            "BF16_plan_expected_bytes_unchanged": self.bf16_plan.expected_bytes,
            "memory_admission": self.memory,
            "groups": [dict(row, part_keys=list(key)) for key, row in self.entries.items()],
        }


def create_plan(quant_config):
    policy = settings()
    if policy is None:
        return None
    bf16_plan = getattr(quant_config, "yaqa_prefill_cache_plan", None)
    if bf16_plan is None:
        return None
    from .yaqa_linear import get_yaqa_parts
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_padded_nk

    sources = quant_config.yaqa_sources
    reverse = {}
    for merged, names in getattr(quant_config, "packed_modules_mapping", {}).items():
        for name in names:
            if name in reverse and reverse[name] != merged:
                raise ValueError("Ambiguous model projection packing map")
            reverse[name] = merged
    entries = {}
    for name in sorted(sources):
        if not name.endswith(".weight"):
            continue
        prefix = name.removesuffix(".weight")
        parent, _, leaf = prefix.rpartition(".")
        if leaf in reverse:
            prefix = (parent + "." if parent else "") + reverse[leaf]
        parts = get_yaqa_parts(quant_config, prefix)
        if not parts:
            continue
        key = tuple(item[0] for item in parts)
        if key in entries:
            continue
        descriptors = [item[1][2] for item in parts]
        if any(d is None or d.get("codec") != "qtip" for d in descriptors):
            continue
        if any(item not in bf16_plan.entries for item in key):
            continue
        shapes = [tuple(d["shape"]) for d in descriptors]
        if any(len(shape) != 2 or min(shape) <= 0 for shape in shapes):
            continue
        k = shapes[0][1]
        if any(shape[1] != k for shape in shapes):
            continue
        n = sum(shape[0] for shape in shapes)
        padded_n, padded_k = marlin_padded_nk(n, k, GROUP_SIZE)
        entries[key] = {
            "k": k,
            "n": n,
            "padded_k": padded_k,
            "padded_n": padded_n,
            "part_count": len(parts),
            "shapes": [list(shape) for shape in shapes],
            "bytes": padded_k * padded_n + 2 * (padded_k // GROUP_SIZE) * padded_n,
        }
    return MarlinPlan(entries, bf16_plan, policy) if entries else None


def prepare(layer, plan):
    if plan is None:
        return None
    if settings() != plan.policy:
        raise ValueError("Marlin cache policy changed after complete planning")
    if plan.admitted is False:
        return None
    parts = tuple(getattr(layer, "yaqa_parts", ()))
    if not parts or any(getattr(part, "codec", None) != "qtip" for part in parts):
        return None
    key = tuple(getattr(part, "prefill_cache_key", None) for part in parts)
    expected = plan.entries.get(key)
    if expected is None:
        return None
    if key in plan.prepared:
        raise ValueError("A Marlin projection group was prepared more than once")
    import torch

    weights = [getattr(part, "prefill_physical_weight", None) for part in parts]
    device = parts[0].trellis.device
    if device.type != "cuda":
        plan.admit(device, torch)
        return None
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("Marlin preparation must finish before graph capture")
    if (
        any(weight is None for weight in weights)
        or len(parts) != expected["part_count"]
        or any(
            weight.dtype != torch.bfloat16
            or weight.device != device
            or not weight.is_contiguous()
            or tuple(weight.shape) != tuple(shape)
            or getattr(part, "prefill_cache_plan", None) is not plan.bf16_plan
            for part, weight, shape in zip(parts, weights, expected["shapes"])
        )
    ):
        raise ValueError("Actual canonical BF16 group differs from its Marlin plan")
    if not plan.admit(device, torch):
        return None
    from vllm import _custom_ops as ops
    import vllm.envs as envs
    from vllm.scalar_type import scalar_types
    from vllm.model_executor.layers.quantization.utils import marlin_utils, quant_utils

    if envs.VLLM_MARLIN_USE_ATOMIC_ADD:
        raise ValueError("Existing Marlin atomic reduction policy differs from preparation")
    weight = weights[0] if len(weights) == 1 else torch.cat(weights, dim=0)
    if (expected["n"], expected["k"]) != (expected["padded_n"], expected["padded_k"]):
        weight = torch.nn.functional.pad(
            weight,
            (0, expected["padded_k"] - expected["k"], 0, expected["padded_n"] - expected["n"]),
        )
    k, n = expected["padded_k"], expected["padded_n"]
    reference, quantized, scales = quant_utils.gptq_quantize_weights(
        weight.T.contiguous(), scalar_types.uint8b128, GROUP_SIZE
    )
    if (
        reference.dtype != torch.bfloat16
        or scales.dtype != torch.bfloat16
        or not bool(torch.isfinite(reference).all())
        or not bool(torch.isfinite(scales).all())
    ):
        raise ValueError("Marlin group32 quantization produced invalid BF16 values")
    packed = quant_utils.gptq_pack(quantized, 8, k, n)
    wire = ops.gptq_marlin_repack(packed, k, n, 8, is_a_8bit=False)
    scales = marlin_utils.marlin_permute_scales(scales, k, n, GROUP_SIZE, is_a_8bit=False)
    zeros = marlin_utils.marlin_make_empty(device)
    if (
        wire.dtype != torch.int32
        or tuple(wire.shape) != (k // 16, n * 4)
        or tuple(scales.shape) != (k // GROUP_SIZE, n)
        or any(value.device != device or not value.is_contiguous() for value in (wire, scales, zeros))
    ):
        raise ValueError("Existing Marlin repack returned an invalid physical layout")
    actual_bytes = sum(value.numel() * value.element_size() for value in (wire, scales, zeros))
    if actual_bytes != expected["bytes"]:
        raise ValueError("Marlin storage does not match the exact planned payload")
    for suffix, value in (("weight", wire), ("scales", scales), ("zeros", zeros)):
        name = BUFFER_PREFIX + suffix
        if hasattr(layer, name):
            raise ValueError("Marlin cache buffer already exists")
        layer.register_buffer(name, value, persistent=False)
    plan.finish(key, actual_bytes)
    return PreparedMarlinPhysical(
        wire=wire, scales=scales, zeros=zeros,
        k=expected["k"], n=expected["n"], part_count=len(parts), device=device,
        parts=parts, part_keys=key, plan=plan,
        padded_k=k, padded_n=n, buffers=(wire, scales, zeros),
    )
