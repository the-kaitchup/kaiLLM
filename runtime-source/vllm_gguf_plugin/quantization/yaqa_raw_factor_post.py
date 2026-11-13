# SPDX-License-Identifier: Apache-2.0
"""Default-off original RAW GEMV plus qualified fused split/factor post.

Preparation authenticates the unchanged original arithmetic and host-only RAW
exports. Runtime admission uses explicit decode and tensor metadata only. All
unsupported inputs retain the original individual core and postprocessor.
"""

import ctypes
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

ENV = "VLLM_GGUF_YAQA_RAW_FACTOR_POST"
LIBRARY_ENVS = {bit: ENV + "_K" + str(bit) + "_LIBRARY" for bit in (3, 4)}
RAW_PINS = {
    3: ("4c30757e092cc6ed558837be6303d3d5aba23dc30ae692dbd154a1cd78d861ce",
        "8b20cacefd55b894d578d3d291cab40928d402572b0b4dd2b5cc2b7aab4ff23a"),
    4: ("01370e736656fa6d9169fd7de40ab334d92ab7349ef84848d90b229d2691a1c2",
        "c9854eb728aedb8287bb2cc1fac755d2e0ef89328db1905aa5e74d21ebede346"),
}
ORIGINAL_PINS = {
    3: "e801676ff6daa463a9dc9d33c444dffc73c699559a1ac1bc86773e7cd519a106",
    4: "d692f3f58fe626d328b5ce665da95159a4756d94c7fc23f951fe3f67e0e5d979",
}
POST_PINS = {
    "yaqa_grouped_k4.py": "addc6b21e1de9146792239210dbfcbabe55e1de2a62ee4eecd78423cb5c66d23",
    "yaqa_grouped_k234.py": "c4fd02423259b5f29b7f18db5759ea5a9c16789229559ac24574672d490683c4",
}
POLICIES = {3: (8, 8, 16, 1), 4: (4, 4, 16, 1)}
_BINDINGS = {}


def mode():
    value = os.getenv(ENV, "none")
    if value not in ("none", "auto"):
        raise ValueError(ENV + " requires none or auto")
    return value


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _pin(tensor):
    return (tensor.data_ptr(), tuple(tensor.shape), str(tensor.dtype),
            str(tensor.device), tuple(tensor.stride()), tensor.is_contiguous())


def _tensors(layer):
    return (layer.trellis, layer.tlut, layer.input_scale, layer.hadamard_output,
            layer.output_scale, layer.Wscale)


def structural(layer):
    """Measured resource family, independent of model or logical tensor names."""
    bit = getattr(layer, "bits", None)
    if (getattr(layer, "_native_device_capability", None) != (12, 0)
            or getattr(layer, "codec", None) != "qtip" or bit not in POLICIES
            or tuple(getattr(layer, "shape", ())) != (6144, 2048)
            or not getattr(layer, "permutation", False)
            or getattr(layer, "validate_b1", False)
            or str(getattr(layer, "dtype", None)) not in
               ("torch.bfloat16", "torch.float32")):
        return False
    expected = (("torch.int16", (49152, bit * 16), 4),
                ("torch.bfloat16", (512, 2), 4),
                ("torch.float32", (2048,), 4),
                ("torch.float32", (12, 12), 4),
                ("torch.float32", (6144,), 4))
    tensors = _tensors(layer)
    device = layer.trellis.device
    if device.type != "cuda":
        return False
    for tensor, (dtype, shape, alignment) in zip(tensors, expected):
        if (str(tensor.dtype) != dtype or tuple(tensor.shape) != shape
                or tensor.device != device or not tensor.is_contiguous()
                or tensor.data_ptr() % alignment):
            return False
    scale = layer.Wscale
    return (str(scale.dtype) == "torch.float32" and scale.numel() == 1
            and scale.device == device and scale.is_contiguous()
            and scale.data_ptr() % 4 == 0)


def original_policy(layer):
    key = getattr(layer, "cuda_k34_key", None)
    try:
        identity = json.loads(key) if isinstance(key, str) else None
    except ValueError:
        return None
    if (not isinstance(identity, dict)
            or identity.get("bits") != layer.bits
            or identity.get("measured_b1_policy") is not True
            or identity.get("source_sha256") != ORIGINAL_PINS[layer.bits]
            or not isinstance(identity.get("architecture"), dict)
            or identity["architecture"].get("compute_capability") != [12, 0]):
        return None
    from .yaqa_cuda_k34 import gemv_policy

    actual = gemv_policy(layer.bits, *layer.shape)
    return actual if actual == POLICIES[layer.bits] else None


@lru_cache(maxsize=8)
def _binding(bit, library):
    source = (Path(__file__).resolve().parents[1] / "native/kernels" /
              ("yaqa_sm120_k" + str(bit) + "_split32_raw_r1.cu"))
    library = Path(library).absolute()
    source_sha, library_sha = RAW_PINS[bit]
    if sha(source) != source_sha or sha(library) != library_sha:
        raise ValueError("Qualified original RAW source/library pair differs")
    root = Path(__file__).resolve().parent
    for name, digest in POST_PINS.items():
        if sha(root / name) != digest:
            raise ValueError("Qualified existing grouped post source differs")
    native = ctypes.CDLL(str(library))
    call = getattr(native, "yaqa_k" + str(bit) + "_tile_rows_raw")
    call.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 7 + [ctypes.c_void_p]
    call.restype = ctypes.c_int
    binding = {"library": str(library), "library_sha256": library_sha,
               "source": str(source), "source_sha256": source_sha,
               "compute_capability": [12, 0], "library_object": native, "call": call}
    _BINDINGS[(bit, str(library))] = binding
    return binding


def prepare_body_layer(layer):
    if mode() == "none" or not structural(layer):
        return None
    if getattr(layer, "_yaqa_raw_factor_post", None) is not None:
        return layer._yaqa_raw_factor_post
    policy = original_policy(layer)
    if policy is None:
        return None
    import torch
    from .yaqa_grouped_k234 import PreparedGroupedK234
    from .yaqa_grouped_k234_dispatch import library as grouped_library

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Prepare RAW factor post before graph capture")
    binding = _binding(layer.bits, os.environ[LIBRARY_ENVS[layer.bits]])
    # The existing constructor owns descriptor layout and post metadata. The
    # original module's tensors are retained without retaining a module cycle.
    proxy = SimpleNamespace(**{name: getattr(layer, name) for name in (
        "shape", "bits", "codec", "permutation", "dtype", "trellis", "tlut",
        "input_scale", "output_scale", "hadamard_output", "Wscale")})
    core_identity = grouped_library()
    prepared = PreparedGroupedK234([proxy], 1, core_identity["library"],
                                  core_identity["library_sha256"], policies=[policy])
    layer.register_buffer("_raw_factor_post_descriptors", prepared.descriptors,
                          persistent=False)
    layer.register_buffer("_raw_factor_post_ptrs", prepared.ptrs, persistent=False)
    state = SimpleNamespace(prepared=prepared, binding=binding, policy=policy,
                            pins=tuple(_pin(t) for t in _tensors(layer)),
                            device=layer.trellis.device,
                            grouped_library=core_identity, original_key=layer.cuda_k34_key)
    layer._yaqa_raw_factor_post = state
    return state


def _decode():
    from vllm.forward_context import get_forward_context

    try:
        context = get_forward_context()
    except AssertionError:
        return False
    return getattr(getattr(context, "batch_descriptor", None), "is_prefill", None) is False


def selected(x, source, layer):
    state = getattr(layer, "_yaqa_raw_factor_post", None)
    return (state is not None and structural(layer) and _decode()
            and layer.cuda_k34_key == state.original_key
            and x.device == state.device
            and str(x.dtype) in ("torch.bfloat16", "torch.float32")
            and x.numel() == 2048 and x.shape[-1] == 2048
            and source.device == state.device
            and str(source.dtype) == "torch.float32"
            and tuple(source.shape) == (1, 2048)
            and source.is_contiguous() and source.data_ptr() % 8 == 0
            and tuple(_pin(t) for t in _tensors(layer)) == state.pins
            and layer._raw_factor_post_descriptors is state.prepared.descriptors
            and layer._raw_factor_post_ptrs is state.prepared.ptrs)


def maybe_apply(x, source, layer):
    if not selected(x, source, layer):
        return None
    import torch

    state = layer._yaqa_raw_factor_post
    # Fresh transient partials belong to this ordinary opaque invocation. CUDA
    # graph capture retains their allocation and dependencies; concurrent calls
    # never share a mutable workspace. The layout is exactly [1,S,N].
    partial = torch.empty((state.prepared.elements,), device=source.device,
                          dtype=torch.float32)
    with torch.cuda.device(source.device):
        code = state.binding["call"](
            source.data_ptr(), layer.trellis.data_ptr(), layer.tlut.data_ptr(),
            partial.data_ptr(), partial.data_ptr(), 6144, 2048, 1,
            *state.policy, torch.cuda.current_stream(source.device).cuda_stream)
    if code:
        raise RuntimeError("Qualified original RAW core failed: " + str(code))
    output = state.prepared.post(partial, x.dtype, fuse_k1=False)
    return output


def runtime_marker(target=None):
    selected_mode = mode()
    root = Path(__file__).resolve().parent
    sources = {str(path): sha(path) for path in (
        Path(__file__).resolve(), root / "yaqa_linear.py", root / "yaqa_kernels.py",
        *(root / name for name in POST_PINS))}
    bindings = {}
    if selected_mode == "auto" and tuple(target or ()) == (12, 0):
        for bit in (3, 4):
            binding = _binding(bit, os.environ[LIBRARY_ENVS[bit]])
            bindings[str(bit)] = {k: v for k, v in binding.items()
                                  if k not in ("library_object", "call")}
            sources[binding["source"]] = binding["source_sha256"]
    return {"mode": selected_mode, "active": selected_mode == "auto" and
            tuple(target or ()) == (12, 0), "sources": sources,
            "bindings": bindings, "scope": "explicit decode M1; SM120 QTIP Q3/Q4 "
            "N6144 K2048 output factor12; original measured split16 policies",
            "original_fallback": "all other rows, phases, types, resources and layouts",
            "persistent_weight_bytes_added": 0,
            "partials": "fresh [1,S,N] F32 allocation retained by CUDA graph capture"}
