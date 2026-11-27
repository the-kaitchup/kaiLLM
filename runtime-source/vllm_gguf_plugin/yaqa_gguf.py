"""Lossless YAQA payloads in GGUF using ordinary GGML storage types.

The logical matrix is described by ``yaqa.manifest``; its components are real
GGUF tensors. No private GGML type numbers or dense shadow weights are needed.
"""

from __future__ import annotations

import json
from pathlib import Path

MANIFEST_KEY = "yaqa.manifest"
RECONSTRUCTION_PRECISION = "inverse-Hadamard-Wscale-signs-FP64-then-FP32-post-scales-v1"

# itq-moe (routed experts): rotation block sizes the prep / post kernels instantiate
ITQ_MOE_BLOCKS_IN = (32, 64, 128, 256, 512, 1024, 2048, 4096)
ITQ_MOE_BLOCKS_OUT = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
ITQ_TABLE_BITS = {4: 12, 2: 10}
ITQ_BPS = {4: range(4, 13), 2: range(3, 9)}      # V4: 1-3 bits per weight; V2: 1.5-4 (b = 7 / 8 since 0.1.0rc6)


def itq_moe_formats(descriptor):
    """[(V, bps)] per expert of an itq-moe descriptor (scalars mean every expert)."""
    E = descriptor["shape"][0]
    out = []
    for key in ("V", "bps"):
        value = descriptor.get(key)
        if type(value) is int:
            value = [value] * E
        if not isinstance(value, list) or len(value) != E or any(type(v) is not int for v in value):
            raise ValueError(f"itq-moe {key} must be an int or a list of one int per expert")
        out.append(value)
    formats = list(zip(*out))
    for V, bps in formats:
        if V not in ITQ_BPS or bps not in ITQ_BPS[V]:
            raise ValueError(f"Unsupported ITQ expert format V{V} bps {bps}")
    return formats


def itq_moe_stream_bytes(m, n, V, bps):
    """Bytes of one expert's tile stream: (m n / 256) tiles of 256 / V steps x bps bits."""
    return (m * n // 256) * (32 * bps // V)


def itq_moe_offsets(descriptor):
    """Byte offsets [E + 1] of the experts' streams (concatenated in expert order, no padding)."""
    _, m, n = descriptor["shape"]
    offsets = [0]
    for V, bps in itq_moe_formats(descriptor):
        offsets.append(offsets[-1] + itq_moe_stream_bytes(m, n, V, bps))
    return offsets


def validate_yaqa_descriptor(descriptor, tensor_names=None):
    """Validate the version-one logical matrix contract without loading weights."""
    if not isinstance(descriptor, dict):
        raise ValueError("YAQA matrix descriptor must be a dictionary")
    shape = descriptor.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != (3 if descriptor.get("codec") == "itq-moe" else 2)
        or any(type(n) is not int or n < 1 for n in shape)
    ):
        raise ValueError("YAQA matrix shape requires two positive dimensions (three for itq-moe)")
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
    elif codec == "itq":
        bps = descriptor.get("bps")
        fmt = (descriptor.get("L"), descriptor.get("V"), descriptor.get("table_bits"))
        if (
            descriptor.get("version") != 1
            or fmt not in ((16, 4, 12), (16, 2, 10))
            or descriptor.get("hash") != "mul"
            or descriptor.get("layout") != "tile16-rowmajor-steps-lsb"
            or type(bps) is not int
            or bps not in (range(4, 13) if fmt[1] == 4 else range(3, 9))     # V2 7 / 8 (3.5 / 4 bits): 0.1.0rc6
            or any(n % 16 for n in shape)
            or shape[1] % 32
        ):
            raise ValueError("Unsupported ITQ trellis contract")
        for axis in ("rotation_in", "rotation_out"):
            kind = descriptor.get(axis)
            if kind != "fwht" and not (isinstance(kind, str) and kind.startswith("block")):
                raise ValueError("Unsupported ITQ rotation")
        required = {"stream", "su", "sv", "wscale", "table"}
    elif codec == "itq-moe":
        # routed experts: E ITQ matrices [m, n] with per-expert trellis formats
        if not isinstance(shape, list) or len(shape) != 3:
            raise ValueError("ITQ expert tensors need a shape [experts, m, n]")
        if descriptor.get("version") != 1 or descriptor.get("L") != 16 or descriptor.get("hash") != "mul" \
                or descriptor.get("layout") != "tile16-rowmajor-steps-lsb":
            raise ValueError("Unsupported ITQ expert contract")
        formats = itq_moe_formats(descriptor)
        E, m, n = shape
        if m % 16 or n % 32:
            raise ValueError("ITQ expert matrices need m % 16 == 0 and n % 32 == 0")
        for axis, width, sizes in (("rotation_in", n, ITQ_MOE_BLOCKS_IN), ("rotation_out", m, ITQ_MOE_BLOCKS_OUT)):
            kind = descriptor.get(axis)
            size = width if kind == "fwht" else (
                int(kind[5:]) if isinstance(kind, str) and kind.startswith("block") and kind[5:].isdigit() else 0)
            if size not in sizes or width % size:
                raise ValueError(f"Unsupported ITQ expert {axis} {kind!r} for width {width}")
        required = {"stream", "su", "sv", "wscale"} | {f"table_v{V}" for V, _ in formats}
    elif codec == "itqvocab":
        vocab, dim = shape
        rotation = descriptor.get("rotation")
        block = dim if rotation == "fwht" else (
            int(rotation[5:]) if isinstance(rotation, str) and rotation.startswith("block") and rotation[5:].isdigit()
            else 0)
        classes = descriptor.get("classes")
        if (
            descriptor.get("version") != 1
            or dim % 1024
            or block not in (256, 512, 1024, 2048)
            or dim % block
            or not isinstance(classes, list)
            or not 1 <= len(classes) <= 8
            or vocab >= 1 << 24
        ):
            raise ValueError("Unsupported ITQ vocabulary contract")
        required = {"su", "map"}
        total = 0
        for k, item in enumerate(classes):
            kind, rows = item.get("kind") if isinstance(item, dict) else None, item.get("rows") if isinstance(item, dict) else None
            if kind not in ("int8", "nf4", "itq") or type(rows) is not int or rows < 1:
                raise ValueError("Unsupported ITQ vocabulary class")
            total += rows
            required |= {f"c{k}.rows", f"c{k}.scale"}
            if kind == "int8":
                required.add(f"c{k}.q")
            elif kind == "nf4":
                required |= {f"c{k}.codes", f"c{k}.levels"}
            else:
                bps = item.get("bps")
                if (
                    type(bps) is not int
                    or bps not in range(4, 13)
                    or (item.get("L"), item.get("V"), item.get("table_bits")) != (16, 4, 12)
                    or item.get("hash") != "mul"
                    or item.get("layout") != "tile16-rowmajor-steps-lsb"
                ):
                    raise ValueError("Unsupported ITQ vocabulary trellis class")
                required |= {f"c{k}.stream", f"c{k}.wscale", "table"}
        if total != vocab:
            raise ValueError("ITQ vocabulary classes must cover every token once")
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
    if descriptor["codec"] == "itq-moe":
        return check_itq_moe_components(descriptor, result)
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
    elif descriptor["codec"] == "itq":
        bps, V = descriptor["bps"], descriptor["V"]   # tile bytes: 256 / V steps of bps bits
        expected.update(stream=(m * n // 256, 32 * bps // V), su=(n,), sv=(m,),
                        table=(1 << descriptor["table_bits"], V))
        for field in ("stream", "su", "sv", "table"):
            if result[field].dtype != torch.int8:
                raise ValueError("ITQ " + field + " requires int8 storage")
        if result["wscale"].numel() != 1 or not bool(result["wscale"].reshape(()) > 0):
            raise ValueError("ITQ requires one positive global scale")
        if "bias" in result:
            expected["bias"] = (m,)
    elif descriptor["codec"] == "itqvocab":
        vocab, dim = m, n
        expected.update(su=(dim,), map=(vocab,))
        dtypes_needed = {"su": torch.int8, "map": torch.int32}
        token_map = torch.full((vocab,), -1, dtype=torch.int64, device="cpu")   # models load under a CUDA device context
        for k, item in enumerate(descriptor["classes"]):
            rows = item["rows"]
            expected[f"c{k}.rows"] = (rows,)
            expected[f"c{k}.scale"] = (rows,)
            dtypes_needed[f"c{k}.rows"] = torch.int32
            dtypes_needed[f"c{k}.scale"] = torch.float32
            if item["kind"] == "int8":
                expected[f"c{k}.q"] = (rows, dim)
                dtypes_needed[f"c{k}.q"] = torch.int8
            elif item["kind"] == "nf4":
                expected[f"c{k}.codes"] = (rows, dim // 2)
                expected[f"c{k}.levels"] = (16,)
                dtypes_needed[f"c{k}.codes"] = torch.int8
                dtypes_needed[f"c{k}.levels"] = torch.float32
            else:
                padded = (rows + 15) // 16 * 16
                expected[f"c{k}.stream"] = (padded * dim // 256, 8 * item["bps"])
                expected[f"c{k}.wscale"] = (1,)
                expected["table"] = (4096, 4)
                dtypes_needed[f"c{k}.stream"] = torch.int8
                dtypes_needed[f"c{k}.wscale"] = torch.float32
                dtypes_needed["table"] = torch.int8
                scale = result[f"c{k}.wscale"]
                if scale.numel() != 1 or not bool(scale.reshape(-1)[0] > 0):
                    raise ValueError("ITQ vocabulary trellis classes need one positive scale")
            ids = result[f"c{k}.rows"].to(torch.int64)
            if tuple(ids.shape) != (rows,) or bool(((ids < 0) | (ids >= vocab)).any()):
                raise ValueError("ITQ vocabulary row ids out of range")
            if bool((token_map[ids] != -1).any()):
                raise ValueError("ITQ vocabulary token assigned to two classes")
            token_map[ids] = (k << 24) | torch.arange(rows, dtype=torch.int64, device="cpu")
        for field, dtype in dtypes_needed.items():
            if result[field].dtype != dtype:
                raise ValueError(f"ITQ vocabulary {field} requires {dtype}")
        if bool((token_map < 0).any()) or not torch.equal(token_map, result["map"].to(torch.int64)):
            raise ValueError("ITQ vocabulary map disagrees with the class rows")
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
    if descriptor["codec"] == "itq" and "pass_output_scale" in result:   # [passes, m], looped (core) layers only
        p = result["pass_output_scale"]
        if p.dim() != 2 or p.shape[0] < 1:
            raise ValueError("ITQ pass_output_scale must be [passes, m]")
        expected["pass_output_scale"] = (p.shape[0], m)
    for field, shape in expected.items():
        if tuple(result[field].shape) != shape:
            raise ValueError("YAQA component shape mismatch: " + field)
    for field, value in result.items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("Nonfinite YAQA component: " + field)
    return result


def check_itq_moe_components(descriptor, result):
    """Validate the loaded components of an itq-moe tensor (CPU tensors, numpy axis order); stream flattened."""
    import torch

    E, m, n = descriptor["shape"]
    formats = itq_moe_formats(descriptor)
    total = itq_moe_offsets(descriptor)[-1]
    for field in ("stream", "su", "sv") + tuple(f"table_v{V}" for V in sorted({V for V, _ in formats})):
        if result[field].dtype != torch.int8:
            raise ValueError(f"ITQ expert {field} requires int8 storage")
    stream = result["stream"].reshape(-1)
    if stream.numel() != total:
        raise ValueError(f"ITQ expert stream has {stream.numel()} bytes, the formats need {total}")
    result["stream"] = stream
    expected = {"su": (n,), "sv": (m,), "wscale": (E,)}
    for V in {V for V, _ in formats}:
        expected[f"table_v{V}"] = (1 << ITQ_TABLE_BITS[V], V)
    if "post_output_scale" in result:
        expected["post_output_scale"] = (E, m)
    if "post_input_scale" in result:
        expected["post_input_scale"] = tuple(result["post_input_scale"].shape) if tuple(
            result["post_input_scale"].shape) in ((n,), (E, n)) else (n,)
    if "pass_output_scale" in result:
        p = result["pass_output_scale"]
        if p.dim() != 3 or p.shape[0] < 1:
            raise ValueError("ITQ expert pass_output_scale must be [passes, experts, m]")
        expected["pass_output_scale"] = (p.shape[0], E, m)
    for field, shape in expected.items():
        if tuple(result[field].shape) != shape:
            raise ValueError(f"ITQ expert component shape mismatch: {field} {tuple(result[field].shape)} != {shape}")
    for field in ("su", "sv"):
        if not bool((result[field].abs() == 1).all()):
            raise ValueError(f"ITQ expert {field} must hold +1 / -1 signs")
    if result["wscale"].dtype != torch.float32 or not bool((result["wscale"] > 0).all()):
        raise ValueError("ITQ expert wscale must be positive F32, one per expert")
    for field, value in result.items():
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("Nonfinite ITQ expert component: " + field)
    return result


def itq_moe_expert(descriptor, components, e):
    """(dense itq descriptor, components) of expert e of an itq-moe tensor: the per-expert view the dense oracle and
    kernels read.  Post scales are the expert's rows; pass scales stay out (see reconstruct_itq_moe)."""
    import torch

    E, m, n = descriptor["shape"]
    V, bps = itq_moe_formats(descriptor)[e]
    offsets = itq_moe_offsets(descriptor)
    stream = components["stream"].reshape(-1)[offsets[e]:offsets[e + 1]].reshape(m * n // 256, 32 * bps // V)
    sub = {"codec": "itq", "version": 1, "shape": [m, n], "bps": bps, "V": V, "L": 16,
           "table_bits": ITQ_TABLE_BITS[V], "hash": "mul", "layout": "tile16-rowmajor-steps-lsb",
           "rotation_in": descriptor["rotation_in"], "rotation_out": descriptor["rotation_out"]}
    comps = {"stream": stream, "su": components["su"], "sv": components["sv"],
             "wscale": components["wscale"].reshape(-1)[e:e + 1].to(torch.float32),
             "table": components[f"table_v{V}"]}
    if "post_output_scale" in components:
        comps["post_output_scale"] = components["post_output_scale"][e]
    if "post_input_scale" in components:
        pis = components["post_input_scale"]
        comps["post_input_scale"] = pis[e] if pis.dim() == 2 else pis
    return sub, comps


def reconstruct_itq_moe(descriptor, components, experts=None, pass_index=None):
    """Dense FP32 oracle [len(experts), m, n] of an itq-moe tensor (tests and decode-to-BF16 scoring; not a serving
    path).  pass_index: 0-based loop pass for the optional pass_output_scale (core layers)."""
    import torch

    E, m, n = descriptor["shape"]
    experts = range(E) if experts is None else experts
    out = []
    for e in experts:
        sub, comps = itq_moe_expert(descriptor, components, e)
        w = reconstruct_itq(sub, comps)
        if pass_index is not None and "pass_output_scale" in components:
            w = w * components["pass_output_scale"][pass_index, e].float()[:, None]
        out.append(w)
    return torch.stack(out)


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


def _block_hadamard(value, kind):
    """Orthonormal Sylvester Hadamard along the last axis, per power-of-two block."""
    n = value.shape[-1]
    size = n if kind == "fwht" else int(kind[5:])
    result = value.reshape(-1, n // size, size)
    width = 1
    while width < size:
        pairs = result.reshape(*result.shape[:-1], -1, 2, width)
        first, second = pairs[..., 0, :], pairs[..., 1, :]
        result = __import__("torch").stack((first + second, first - second), -2).reshape(result.shape)
        width *= 2
    return (result * size**-0.5).reshape(value.shape)


def _trellis_tiles(stream, bps, table_bits, table):
    """Raw table values of ITQ tiles: [tiles, 256] in tile row-major order (FP64)."""
    import torch

    stream = stream.view(torch.uint8).to(torch.int64)
    bits = ((stream[:, :, None] >> torch.arange(8)) & 1).reshape(stream.shape[0], -1)
    total = bits.shape[1]
    ends = (torch.arange(64) + 1) * bps
    offsets = torch.arange(16)
    window = bits[:, (ends[:, None] - 16 + offsets[None]) % total]
    states = (window << offsets).sum(-1)
    index = ((states * 0x9E3779B1) % (1 << 32)) >> (32 - table_bits)
    return table.to(torch.float64)[index].reshape(-1, 256)


def reconstruct_itq_vocab(descriptor, components):
    """Dense FP32 oracle [vocab, dim] of an ITQ vocabulary (tests; not a serving path)."""
    import torch

    vocab, dim = descriptor["shape"]
    rotated = torch.zeros(vocab, dim, dtype=torch.float64)
    for k, item in enumerate(descriptor["classes"]):
        ids = components[f"c{k}.rows"].to(torch.int64)
        scale = components[f"c{k}.scale"].double()[:, None]
        if item["kind"] == "int8":
            rows = components[f"c{k}.q"].double()
        elif item["kind"] == "nf4":
            packed = components[f"c{k}.codes"].view(torch.uint8).to(torch.int64)
            codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(len(ids), dim)
            rows = components[f"c{k}.levels"].double()[codes]
        else:
            padded = (len(ids) + 15) // 16 * 16
            tiles = _trellis_tiles(components[f"c{k}.stream"], item["bps"], 12, components["table"])
            rows = tiles.reshape(padded // 16, dim // 16, 16, 16).transpose(1, 2).reshape(padded, dim)[: len(ids)]
            rows = rows * components[f"c{k}.wscale"].double().reshape(())
        rotated[ids] = rows * scale
    rotation = descriptor["rotation"]
    return (_block_hadamard(rotated, rotation) * components["su"].double()).float()


def reconstruct_itq(descriptor, components):
    """Dense FP32 oracle of an ITQ matrix (tests and canonical scoring; not a serving path)."""
    import torch

    m, n = descriptor["shape"]
    bps, steps = descriptor["bps"], 256 // descriptor["V"]
    stream = components["stream"].view(torch.uint8).to(torch.int64)
    bits = ((stream[:, :, None] >> torch.arange(8)) & 1).reshape(stream.shape[0], -1)
    total = bits.shape[1]
    ends = (torch.arange(steps) + 1) * bps
    offsets = torch.arange(16)
    window = bits[:, (ends[:, None] - 16 + offsets[None]) % total]
    states = (window << offsets).sum(-1)
    index = ((states * 0x9E3779B1) % (1 << 32)) >> (32 - descriptor["table_bits"])
    values = components["table"].to(torch.float64)[index].reshape(-1, 256)
    value = values.reshape(m // 16, n // 16, 16, 16).transpose(1, 2).reshape(m, n)
    value = value * components["wscale"].double().reshape(())
    value = _block_hadamard(value, descriptor["rotation_in"]) * components["su"].double()
    value = (_block_hadamard(value.T, descriptor["rotation_out"]) * components["sv"].double()).T
    if "post_output_scale" in components:
        value = value * components["post_output_scale"].double()[:, None]
    if "post_input_scale" in components:
        value = value * components["post_input_scale"].double()[None, :]
    return value.contiguous().float()


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
    elif descriptor["codec"] == "itq":
        value = reconstruct_itq(descriptor, components)
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
