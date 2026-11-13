"""Lossless YAQA payloads in GGUF using ordinary GGML storage types.

The logical matrix is described by ``yaqa.manifest``; its components are real
GGUF tensors. No private GGML type numbers or dense shadow weights are needed.
"""

from __future__ import annotations

import json
from pathlib import Path

MANIFEST_KEY = "yaqa.manifest"
RECONSTRUCTION_PRECISION = "inverse-Hadamard-Wscale-signs-FP64-then-FP32-post-scales-v1"


def validate_yaqa_descriptor(descriptor, tensor_names=None):
    """Validate the version-one logical matrix contract without loading weights."""
    if not isinstance(descriptor, dict):
        raise ValueError("YAQA matrix descriptor must be a dictionary")
    shape = descriptor.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(n) is not int or n < 1 for n in shape)
    ):
        raise ValueError("YAQA matrix shape requires two positive dimensions")
    components = descriptor.get("components")
    if not isinstance(components, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v for k, v in components.items()
    ):
        raise ValueError("YAQA components must map fields to GGUF tensor names")
    codec = descriptor.get("codec")
    if codec == "qtip":
        bits = descriptor.get("bits")
        if (
            type(bits) is not int
            or bits not in range(1, 9)
            or (descriptor.get("L"), descriptor.get("K"), descriptor.get("V"))
            != (16, bits, 2)
            or descriptor.get("tlut_bits") != 9
            or descriptor.get("decode_mode") != "quantlut_sym"
            or any(n % 16 for n in shape)
            or type(descriptor.get("tensor_core_permutation")) is not bool
            or descriptor.get("reconstruction_precision") != RECONSTRUCTION_PRECISION
        ):
            raise ValueError("Unsupported YAQA QTIP reconstruction contract")
        required = {
            "trellis",
            "SU",
            "SV",
            "Wscale",
            "tlut",
            "hadamard_input",
            "hadamard_output",
        }
    elif codec == "ternary":
        if descriptor.get("bits") != 1.6:
            raise ValueError("YAQA base3 ternary stores five symbols per byte")
        required = {"packed", "row_scales"}
    elif codec == "scalar-int4":
        if descriptor.get("bits") != 4:
            raise ValueError("YAQA scalar-int4 requires four bits per value")
        required = {"packed", "scales"}
    else:
        raise ValueError(f"Unsupported YAQA codec: {codec!r}")
    if not required <= components.keys():
        raise ValueError(
            f"Missing YAQA components: {sorted(required - components.keys())}"
        )
    if tensor_names is not None and not set(components.values()) <= set(tensor_names):
        raise ValueError("YAQA manifest references absent GGUF component tensors")
    return descriptor


def read_yaqa_profile(reader_or_path):
    """Return the validated container profile, or None for ordinary GGUF.

    A valid profile may contain no YAQA matrices: an RCO winner can use only
    ordinary GGUF tensors while retaining fitted state/cast semantics and its
    original provenance. Profile presence must not depend on tensor count.
    """
    if isinstance(reader_or_path, (str, Path)):
        import gguf

        reader_or_path = gguf.GGUFReader(str(reader_or_path))
    field = reader_or_path.fields.get(MANIFEST_KEY)
    if field is None:
        return None
    value = field.contents()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    manifest = json.loads(value)
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != 1
        or not isinstance(manifest.get("tensors"), dict)
    ):
        raise ValueError("Unsupported YAQA GGUF manifest version")
    names = {tensor.name for tensor in reader_or_path.tensors}
    for name, descriptor in manifest["tensors"].items():
        if not isinstance(name, str) or not name.endswith(".weight") or name in names:
            raise ValueError("YAQA logical weight must have one unambiguous owner")
        validate_yaqa_descriptor(descriptor, names)
    return manifest


def read_yaqa_manifest(reader_or_path):
    """Return logical matrix descriptors; both empty profiles and GGUF give {}."""
    profile = read_yaqa_profile(reader_or_path)
    return {} if profile is None else profile["tensors"]


def load_yaqa_components(reader, descriptor):
    """Copy a logical matrix's packed components into CPU Torch tensors."""
    import numpy as np
    import torch

    tensors = {tensor.name: tensor for tensor in reader.tensors}
    validate_yaqa_descriptor(descriptor, tensors)
    dtypes = {
        0: torch.float32,
        1: torch.float16,
        24: torch.int8,
        25: torch.int16,
        26: torch.int32,
        27: torch.int64,
        28: torch.float64,
        30: torch.bfloat16,
    }
    result = {}
    for field, name in descriptor["components"].items():
        tensor = tensors[name]
        if int(tensor.tensor_type) not in dtypes:
            raise ValueError("YAQA components require ordinary scalar GGUF types")
        raw = np.array(tensor.data, copy=True).view(np.uint8)
        value = torch.from_numpy(raw.reshape(-1)).view(dtypes[int(tensor.tensor_type)])
        result[field] = value.reshape(tuple(reversed(tensor.shape.tolist())))
    m, n = descriptor["shape"]
    expected = {}
    if descriptor["codec"] == "qtip":
        expected.update(
            trellis=(m * n // 256, 16 * descriptor["bits"]),
            SU=(n,),
            SV=(m,),
            tlut=(512, 2),
        )
        if result["trellis"].dtype != torch.int16:
            raise ValueError("QTIP trellis requires exact signed int16 storage")
        if result["Wscale"].numel() != 1 or not bool(result["Wscale"].reshape(()) > 0):
            raise ValueError("QTIP requires one positive global scale")
        for axis, width in (("input", n), ("output", m)):
            factor = result["hadamard_" + axis]
            k = factor.shape[0] if factor.ndim == 2 else 0
            if k < 1 or tuple(factor.shape) != (k, k) or width % k:
                raise ValueError("Invalid Hadamard factor dimensions")
            p = width // k
            if p & (p - 1):
                raise ValueError("Hadamard butterfly width must be a power of two")
    else:
        expected["packed"] = (
            (m * n + 4) // 5 if descriptor["codec"] == "ternary" else (m * n + 1) // 2,
        )
        expected["row_scales" if descriptor["codec"] == "ternary" else "scales"] = (m,)
        if result["packed"].dtype != torch.int8:
            raise ValueError("Packed scalar codes require exact byte storage")
    for field, size in (("post_input_scale", n), ("post_output_scale", m)):
        if field in result:
            expected[field] = (size,)
    for field, shape in expected.items():
        if tuple(result[field].shape) != shape:
            raise ValueError("YAQA component shape mismatch: " + field)
    for field, value in result.items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("Nonfinite YAQA component: " + field)
    return result


def _hadamard(value, factor):
    import torch

    n = value.shape[-1]
    k = factor.shape[0]
    if factor.shape != (k, k) or n % k or (n // k) & (n // k - 1):
        raise ValueError("Invalid YAQA Hadamard factor geometry")
    result = value.reshape(-1, k, n // k)
    width = 1
    while width < n // k:
        pairs = result.reshape(*result.shape[:-1], -1, 2, width)
        first, second = pairs[..., 0, :], pairs[..., 1, :]
        result = torch.stack((first + second, first - second), -2).reshape(result.shape)
        width *= 2
    if k > 1:
        result = factor.to(value) @ result
    return (result * n**-0.5).reshape(value.shape)


def reconstruct_yaqa(descriptor, components, *, dtype=None):
    """Canonical dense CPU oracle; inference should use packed GPU kernels."""
    import torch

    validate_yaqa_descriptor(descriptor)
    dtype = dtype or torch.bfloat16
    m, n = descriptor["shape"]
    if descriptor["codec"] == "qtip":
        bits = descriptor["bits"]
        words = components["trellis"].to(torch.int64) & 65535
        if tuple(words.shape) != (m * n // 256, 16 * bits):
            raise ValueError("YAQA trellis shape disagrees with its declared bit depth")
        positions = (torch.arange(128)[:, None] * (2 * bits) + torch.arange(16)) % (
            256 * bits
        )
        index, shift = positions // 16, 15 - positions % 16
        states = (((words[:, index] >> shift) & 1) << torch.arange(15, -1, -1)).sum(-1)
        hashed = (states + 1) * states
        lut_index = (hashed >> 6) & 511
        values = components["tlut"][lut_index].clone()
        values[..., 0] *= (1 - ((hashed >> 15) & 1) * 2).to(values.dtype)
        values = values.reshape(-1, 256)
        if descriptor["tensor_core_permutation"]:
            permutation = (
                torch.arange(256)
                .reshape(2, 8, 2, 4, 2)
                .permute(1, 3, 2, 0, 4)
                .flatten()
            )
            values = values[:, torch.argsort(permutation)]
        value = (
            values.reshape(m // 16, n // 16, 16, 16)
            .transpose(1, 2)
            .reshape(m, n)
            .double()
        )
        value *= components["Wscale"].double()
        value = (
            _hadamard(value, components["hadamard_input"]) * components["SU"].double()
        )
        value = (
            (
                _hadamard(value.T, components["hadamard_output"])
                * components["SV"].double()
            )
            .T.contiguous()
            .float()
        )
        if "post_output_scale" in components:
            value *= components["post_output_scale"].float()[:, None]
        if "post_input_scale" in components:
            value *= components["post_input_scale"].float()[None, :]
    elif descriptor["codec"] == "ternary":
        packed = components["packed"].view(torch.uint8).to(torch.int16).flatten()
        if len(packed) != (m * n + 4) // 5 or bool((packed > 242).any()):
            raise ValueError("Invalid base3 ternary payload")
        codes = ((packed[:, None] // torch.tensor([1, 3, 9, 27, 81])) % 3).flatten()
        if bool((codes[m * n :] != 0).any()):
            raise ValueError("Nonzero ternary padding digits")
        codes = (codes[: m * n] - 1).reshape(m, n)
        scales = components.get("post_output_scale", components["row_scales"])
        value = (codes.double() * scales.double()[:, None]).float()
    else:
        packed = components["packed"].view(torch.uint8).flatten()
        codes = (
            torch.stack((packed & 15, packed >> 4), -1)
            .flatten()[: m * n]
            .to(torch.int8)
            - 8
        )
        value = codes.reshape(m, n).float() * components["scales"].float()[:, None]
    return value.to(dtype)
