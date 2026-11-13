# SPDX-License-Identifier: Apache-2.0
"""Default-off lossless F32 storage for exact small QTIP Hadamard factors.

Original H64 buffers remain registered and byte-identical. Existing kernels,
routes and grouped descriptors consume the active H32 buffers normally. Factors
are immutable after preparation; known original-buffer writes require refresh()
outside capture before replay. Arbitrary external writes are not tracked.
"""

import copy
import hashlib
import os
from pathlib import Path

ENV = "VLLM_GGUF_YAQA_FACTOR_STORAGE"


def mode():
    value = os.getenv(ENV, "none")
    if value not in ("none", "auto"):
        raise ValueError(ENV + " requires none or auto")
    return value


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _tensor_sha(value):
    import torch

    return hashlib.sha256(
        value.detach().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _exact_shadow_cpu(value):
    import torch

    if (
        value.device.type != "cpu"
        or value.dtype != torch.float64
        or value.ndim != 2
        or value.shape[0] <= 1
        or value.shape[0] != value.shape[1]
        or not bool(torch.isfinite(value).all())
    ):
        return None
    original = value.detach().contiguous()
    shadow = original.float().contiguous()
    if not torch.equal(original.view(torch.uint8), shadow.double().view(torch.uint8)):
        return None
    return shadow


def _pin(value):
    return dict(
        pointer=value.data_ptr(),
        shape=list(value.shape),
        dtype=str(value.dtype),
        device=str(value.device),
        stride=list(value.stride()),
        contiguous=value.is_contiguous(),
    )


def prepare_layer(layer, components):
    """Construct shadows once, before opaque/group/native pointer capture."""
    if mode() == "none":
        return None
    if getattr(layer, "codec", None) != "qtip" or getattr(
        layer, "_native_device_capability", None
    ) != (12, 0):
        return None
    if getattr(layer, "_yaqa_factor_storage", None) is not None:
        raise ValueError("Factor storage must be prepared exactly once")
    import torch

    state = dict(mode="auto", compute_capability=[12, 0], axes={}, shadow_bytes=0)
    for axis in ("input", "output"):
        name = "hadamard_" + axis
        cpu = components.get(name)
        if cpu is None:
            continue
        shadow_cpu = _exact_shadow_cpu(cpu)
        if shadow_cpu is None:
            continue
        original = getattr(layer, name)
        if original.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare factor storage before CUDA graph capture")
        if original.dtype != torch.float64 or original.shape != cpu.shape:
            raise ValueError("Original factor differs from constructor components")
        original_name = "_factor_storage_original_" + name
        if hasattr(layer, original_name):
            raise ValueError("Original factor storage name already exists")
        shadow = shadow_cpu.to(device=original.device).contiguous()
        layer.register_buffer(original_name, original)
        layer.register_buffer(name, shadow)
        state["axes"][axis] = dict(
            original_buffer=original_name,
            active_buffer=name,
            original=_pin(original),
            shadow=_pin(shadow),
            original_sha256=_tensor_sha(cpu),
            shadow_sha256=_tensor_sha(shadow_cpu),
            original_bytes=original.numel() * original.element_size(),
            shadow_bytes=shadow.numel() * shadow.element_size(),
        )
        state["shadow_bytes"] += shadow.numel() * shadow.element_size()
    layer._yaqa_factor_storage = state
    return status(layer)


def status(layer):
    """Startup metadata only; no tensor scans, hashes or device calls."""
    state = getattr(layer, "_yaqa_factor_storage", None)
    return None if state is None else copy.deepcopy(state)


def refresh(layer):
    """Refresh known in-place original H64 writes into captured H32 storage.

    Call after an intentional factor edit and before any graph replay. Replacing
    buffers, changing shape/device/dtype, or nonrepresentable values requires
    restoring the original route and rebuilding prepared states/graphs instead.
    """
    state = getattr(layer, "_yaqa_factor_storage", None)
    if state is None:
        return None
    import torch

    updates = []
    for axis, metadata in state["axes"].items():
        original = getattr(layer, metadata["original_buffer"])
        shadow = getattr(layer, metadata["active_buffer"])
        if original.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Refresh factors outside CUDA graph capture")
        if _pin(original) != metadata["original"] or _pin(shadow) != metadata["shadow"]:
            raise ValueError("Factor identity changed; rebuild prepared states/graphs")
        cpu = original.detach().cpu().contiguous()
        shadow_cpu = _exact_shadow_cpu(cpu)
        if shadow_cpu is None:
            raise ValueError(
                "Factor is no longer losslessly F32; rebuild original route"
            )
        updates.append((axis, shadow, cpu, shadow_cpu))
    # Validate every factor before changing any captured shadow buffer.
    with torch.no_grad():
        for axis, shadow, cpu, shadow_cpu in updates:
            shadow.copy_(shadow_cpu)
            state["axes"][axis]["original_sha256"] = _tensor_sha(cpu)
            state["axes"][axis]["shadow_sha256"] = _tensor_sha(shadow_cpu)
    return status(layer)


def runtime_marker(*, target=None):
    """Stamp mode/source identity before model construction and graph caching."""
    selected_mode = mode()
    target = tuple(target or ())
    sources = (
        Path(__file__).absolute(),
        Path(__file__).with_name("yaqa.py").absolute(),
    )
    return dict(
        mode=selected_mode,
        active_for_target=selected_mode == "auto" and target == (12, 0),
        target=list(target),
        sources={str(path): _sha(path) for path in sources},
        selection="captured SM120 QTIP constructor; exact CPU H64 square K>1 only",
        arithmetic="unchanged kernels load the same F32 factor values",
        original_H64_retained=True,
        factor_mutation=(
            "immutable; refresh explicitly outside capture after known writes"
        ),
        persistent_dense_weights_added=False,
    )
