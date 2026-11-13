# SPDX-License-Identifier: Apache-2.0
"""Optional load-time Marlin cache of canonical ordinary GGUF body weights."""

import os
import weakref
from dataclasses import dataclass

ENV = "VLLM_GGUF_ORDINARY_MARLIN_PHYSICAL"
BYTES_ENV = ENV + "_BYTES"
HEADROOM_ENV = ENV + "_HEADROOM_BYTES"
BUFFER_PREFIX = "_ordinary_marlin_physical_"
GROUP_SIZE = 32


def settings():
    mode = os.getenv(ENV, "none")
    if mode not in ("none", "auto"):
        raise ValueError(f"{ENV} requires none or auto")
    if mode == "none":
        return None
    if os.getenv("VLLM_MARLIN_USE_ATOMIC_ADD", "0") != "0":
        raise ValueError("Ordinary Marlin requires VLLM_MARLIN_USE_ATOMIC_ADD=0")
    values = []
    for name in (BYTES_ENV, HEADROOM_ENV):
        raw = os.getenv(name)
        if raw is None:
            raise ValueError(f"Enabled ordinary Marlin requires explicit {name}")
        try:
            value = int(raw)
        except ValueError as error:
            raise ValueError(f"{name} requires an integer byte count") from error
        if value < 0 or (name == BYTES_ENV and value == 0):
            raise ValueError(f"Invalid ordinary Marlin byte policy: {name}")
        values.append(value)
    return tuple(values)


def _wire_pin(weight):
    try:
        version = weight._version
    except RuntimeError as error:
        if "Inference tensors do not track version counter" not in str(error):
            raise
        version = "inference tensor; no version counter"
    return (weight.data_ptr(), tuple(weight.shape), str(weight.dtype),
            str(weight.device), tuple(weight.stride()), version)


@dataclass
class PreparedOrdinaryMarlinPhysical:
    wire: object
    scales: object
    zeros: object
    k: int
    n: int
    padded_k: int
    padded_n: int
    device: object
    original_ref: object
    original_pin: tuple
    plan: object
    buffers: tuple
    key: object = None
    fallback: object = None


class OrdinaryMarlinPlan:
    """Complete additional payload admission; existing caches are unchanged."""

    def __init__(self, entries, policy, device, skipped):
        self.entries = entries
        self.policy = policy
        self.budget, self.headroom = policy
        self.device = device
        self.skipped = skipped
        self.expected_bytes = sum(row["bytes"] for row in entries.values())
        self.prepared = {}
        self.prepared_bytes = 0
        self.admitted = None
        self.ready = False
        self.memory = None
        self.disabled_reason = None
        if not entries:
            self.admitted = False
            self.disabled_reason = "no eligible ordinary GGUF body matrices"
        elif self.expected_bytes > self.budget:
            self.admitted = False
            self.disabled_reason = "complete additional payload exceeds explicit cap"

    def admit(self, torch):
        if self.admitted is not None:
            return self.admitted
        with torch.cuda.device(self.device):
            if torch.cuda.is_current_stream_capturing():
                raise ValueError("Ordinary Marlin preparation must precede graph capture")
            free, total = torch.cuda.mem_get_info(self.device)
        required = self.expected_bytes + self.headroom
        self.memory = {
            "free_bytes_after_original_and_existing_cache_preparation": int(free),
            "total_device_bytes": int(total),
            "additional_Marlin_reservation_bytes": self.expected_bytes,
            "headroom_bytes": self.headroom,
            "required_free_bytes": required,
            "headroom_scope": "KV, profile/workspace and serial preparation temporaries",
        }
        self.admitted = int(free) >= required
        if not self.admitted:
            self.disabled_reason = "complete additional reservation and headroom do not fit"
        return self.admitted

    def finish(self, name, actual_bytes):
        if name in self.prepared or actual_bytes != self.entries[name]["bytes"]:
            raise ValueError("Ordinary Marlin allocation differs from static reservation")
        self.prepared[name] = actual_bytes
        self.prepared_bytes += actual_bytes

    def ensure_complete(self):
        if self.admitted is False:
            return
        if self.admitted is not True or self.prepared.keys() != self.entries.keys():
            raise ValueError("Incomplete ordinary Marlin cache before inference")
        if self.prepared_bytes != self.expected_bytes:
            raise ValueError("Ordinary Marlin bytes differ from complete reservation")
        self.ready = True

    def summary(self):
        return {
            "mode": "auto", "admitted": self.admitted, "ready": self.ready,
            "disabled_reason": self.disabled_reason,
            "budget_bytes": self.budget, "headroom_bytes": self.headroom,
            "expected_groups": len(self.entries), "prepared_groups": len(self.prepared),
            "expected_bytes": self.expected_bytes, "prepared_bytes": self.prepared_bytes,
            "memory_admission": self.memory,
            "groups": [dict(row, name=name) for name, row in self.entries.items()],
            "skipped": list(self.skipped),
            "original_wire_and_native_fallback_retained": True,
            "existing_QA_and_BF16_plans_unchanged": True,
            "persistent_canonical_BF16_bytes_added": 0,
        }


def _catalog(model, torch, packed_type, padded_nk, yaqa_type=None):
    entries, layers, skipped, capabilities = {}, {}, [], {}
    modules = list(model.named_modules())
    excluded = set()
    if yaqa_type is not None:
        for _, owner in modules:
            parts = getattr(owner, "yaqa_parts", ())
            if any(isinstance(part, yaqa_type) for part in parts):
                # The same wrapper also hosts all-ordinary projections. Only
                # a group with an actual YAQA part is outside this candidate.
                excluded.update(id(child) for part in parts for child in part.modules())
    device, seen = None, set()
    for name, layer in sorted(modules, key=lambda item: item[0]):
        if not isinstance(layer, packed_type) or id(layer) in seen or id(layer) in excluded:
            continue
        seen.add(id(layer))
        weight = layer.weight
        storage_supported = (weight.dtype == torch.uint8 if layer.is_quantized
                             else weight.dtype == torch.bfloat16)
        if not (layer.role == "body" and layer.canonical_rounding
                and layer.dtype == torch.bfloat16 and storage_supported
                and weight.ndim == 2 and weight.is_contiguous()
                and weight.device.type == "cuda"):
            continue
        actual_device = weight.device
        if actual_device not in capabilities:
            capabilities[actual_device] = tuple(torch.cuda.get_device_capability(actual_device))
        if capabilities[actual_device] != (12, 0):
            continue
        if getattr(layer, "_ordinary_marlin_physical", None) is not None:
            raise ValueError("Ordinary Marlin is prepared only once per model load")
        shape = tuple(layer.shape)
        if len(shape) != 2 or any(type(value) is not int or value <= 0 for value in shape):
            skipped.append({"name": name, "reason": "invalid positive matrix geometry"})
            continue
        n, k = shape
        padded_n, padded_k = map(int, padded_nk(n, k, GROUP_SIZE))
        if (padded_n < n or padded_k < k or padded_n % 64 or padded_k % 32
                or not ((padded_n % 64 == 0 and padded_k % 128 == 0)
                        or (padded_n % 128 == 0 and padded_k % 64 == 0))):
            raise ValueError("Stock Marlin padding returned unsupported geometry")
        if device is not None and actual_device != device:
            raise ValueError("One ordinary Marlin reservation cannot span CUDA devices")
        device = actual_device
        entries[name] = {
            "n": n, "k": k, "padded_n": padded_n, "padded_k": padded_k,
            "qtype": layer.weight_type,
            "bytes": padded_k * padded_n + 2 * (padded_k // GROUP_SIZE) * padded_n,
        }
        layers[name] = layer
    return entries, layers, device, skipped


def _prepare(layer, expected, plan, torch):
    from vllm import _custom_ops as ops
    import vllm.envs as envs
    from vllm.scalar_type import scalar_types
    from vllm.model_executor.layers.quantization.utils import marlin_utils, quant_utils

    if envs.VLLM_MARLIN_USE_ATOMIC_ADD:
        raise ValueError("Stock Marlin atomic reduction policy differs from preparation")
    if settings() != plan.policy:
        raise ValueError("Ordinary Marlin policy changed during preparation")
    device = layer.weight.device
    if device != plan.device:
        raise ValueError("Ordinary Marlin matrix moved after planning")
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("Ordinary Marlin reconstruction must precede graph capture")
    original = layer.weight
    original_pin = _wire_pin(original)
    n, k = expected["n"], expected["k"]
    # Existing compact gather performs the canonical BF16 decoded-weight store.
    rows = torch.arange(n, device=device)
    weight = layer.embedding(rows)
    if (tuple(weight.shape) != (n, k) or weight.dtype != torch.bfloat16
            or weight.device != device or not weight.is_contiguous()
            or not bool(torch.isfinite(weight).all())):
        raise ValueError("Canonical ordinary GGUF reconstruction is invalid")
    if layer.weight is not original or _wire_pin(layer.weight) != original_pin:
        raise ValueError("Canonical reconstruction changed original GGUF storage")
    if (n, k) != (expected["padded_n"], expected["padded_k"]):
        weight = torch.nn.functional.pad(
            weight, (0, expected["padded_k"] - k, 0, expected["padded_n"] - n),
        )
    k, n = expected["padded_k"], expected["padded_n"]
    reference, quantized, scales = quant_utils.gptq_quantize_weights(
        weight.T.contiguous(), scalar_types.uint8b128, GROUP_SIZE
    )
    if (reference.dtype != torch.bfloat16 or scales.dtype != torch.bfloat16
            or not bool(torch.isfinite(reference).all())
            or not bool(torch.isfinite(scales).all())):
        raise ValueError("Stock Marlin group32 quantization returned invalid BF16 values")
    packed = quant_utils.gptq_pack(quantized, 8, k, n)
    wire = ops.gptq_marlin_repack(packed, k, n, 8, is_a_8bit=False)
    scales = marlin_utils.marlin_permute_scales(scales, k, n, GROUP_SIZE, is_a_8bit=False)
    zeros = marlin_utils.marlin_make_empty(device)
    if (wire.dtype != torch.int32 or tuple(wire.shape) != (k // 16, n * 4)
            or tuple(scales.shape) != (k // GROUP_SIZE, n)
            or any(value.device != device or not value.is_contiguous()
                   for value in (wire, scales, zeros))):
        raise ValueError("Stock Marlin repack returned an invalid physical layout")
    actual_bytes = sum(value.numel() * value.element_size() for value in (wire, scales, zeros))
    if actual_bytes != expected["bytes"]:
        raise ValueError("Ordinary Marlin storage differs from exact reservation")
    if layer.weight is not original or _wire_pin(layer.weight) != original_pin:
        raise ValueError("Marlin preparation changed original GGUF storage")
    # All dense reconstruction/quantization temporaries die on return. Only this
    # encoded wire, BF16 scales and empty zero-point tensor remain resident.
    return PreparedOrdinaryMarlinPhysical(
        wire=wire, scales=scales, zeros=zeros,
        k=expected["k"], n=expected["n"], padded_k=k, padded_n=n,
        device=device, original_ref=weakref.ref(layer), original_pin=original_pin,
        plan=plan, buffers=(wire, scales, zeros),
    )


def prepare_model(model, vllm_config=None):
    """Prepare after existing caches/Q6 adapters and before profiling/capture."""
    del vllm_config
    policy = settings()
    if policy is None:
        return None
    if hasattr(model, "_ordinary_marlin_physical_plan"):
        raise ValueError("Ordinary Marlin model preparation cannot run twice")
    import torch
    from .compact import PackedGGUFLinear
    from .yaqa import PackedYAQALinear
    from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_padded_nk

    entries, layers, device, skipped = _catalog(
        model, torch, PackedGGUFLinear, marlin_padded_nk, PackedYAQALinear
    )
    plan = OrdinaryMarlinPlan(entries, policy, device, skipped)
    if plan.admit(torch):
        staged = []
        with torch.inference_mode(), torch.cuda.device(device):
            for name, expected in entries.items():
                state = _prepare(layers[name], expected, plan, torch)
                plan.finish(name, expected["bytes"])
                staged.append((layers[name], state))
        plan.ensure_complete()
        for layer, state in staged:
            for suffix, value in (("weight", state.wire), ("scales", state.scales), ("zeros", state.zeros)):
                name = BUFFER_PREFIX + suffix
                if hasattr(layer, name):
                    raise ValueError("Ordinary Marlin numerical buffer already exists")
                layer.register_buffer(name, value, persistent=False)
            layer._ordinary_marlin_physical = state
    model._ordinary_marlin_physical_plan = plan
    model.gguf_ordinary_marlin_physical = plan.summary()
    return plan
