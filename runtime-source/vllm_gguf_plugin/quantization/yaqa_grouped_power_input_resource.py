# SPDX-License-Identifier: Apache-2.0
"""Default-off SM120 grouped input resource; existing math and metadata views."""

import ctypes
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

ENV = "VLLM_GGUF_YAQA_GROUPED_POWER_INPUT_RESOURCE"
LIBRARY_ENV = ENV + "_LIBRARY"
METADATA_ENV = ENV + "_METADATA"
SOURCE_SHA = "e7206ec25bf75058b07ef2aa5a534cc3a6cacb2fc928f2252652f0620d17e63e"
LIBRARY_SHA = "63b0ec6d88219965b6d98a1d4551b93a5ea44b387a22783eb2b5195bcc350645"
METADATA_SHA = "0d61dca39ba1f3bac79f342bda5ea2b949468b799adb3014113e46cd0921ca6c"
LEGACY_LIBRARY_SHA = "8a49cb25b98bfcd46bc5ab936317c70eadc844d8dd8dcb5ead30e32c176db4a4"
LEGACY_METADATA_SHA = "76aa0df7682fc60b1cf77e4fcf753af526359b653ff2e836b590e7341c733408"
SOURCE = (
    Path(__file__).parents[1] / "native/kernels/yaqa_grouped_power_input_1024_r1.cu"
)
_BINDINGS = {}


def mode():
    value = os.getenv(ENV, "none")
    if value not in ("none", "auto"):
        raise ValueError(ENV + " requires none or auto")
    return value


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@lru_cache(maxsize=8)
def metadata(library, raw):
    library, raw = Path(library).absolute(), Path(raw).absolute()
    if (
        sha(library) != LIBRARY_SHA
        or sha(raw) != METADATA_SHA
        or sha(SOURCE) != SOURCE_SHA
    ):
        raise ValueError("Qualified grouped Hin1024 native/source bytes differ")
    identity = json.loads(raw.read_text())
    if (
        identity.get("library_sha256") != LIBRARY_SHA
        or identity.get("source_sha256") != SOURCE_SHA
        or identity.get("architecture", {}).get("compute_capability") != [12, 0]
    ):
        raise ValueError("Qualified grouped Hin1024 target identity differs")
    return {
        "library": str(library),
        "library_sha256": LIBRARY_SHA,
        "raw_metadata": str(raw),
        "raw_metadata_sha256": METADATA_SHA,
        "source_sha256": SOURCE_SHA,
        "compute_capability": [12, 0],
    }


@lru_cache(maxsize=8)
def legacy_metadata(library):
    path = Path(library).absolute()
    raw = path.with_name("identity.json")
    if sha(path) != LEGACY_LIBRARY_SHA or sha(raw) != LEGACY_METADATA_SHA:
        raise ValueError("Original grouped input library/raw metadata differs")
    return {str(path): LEGACY_LIBRARY_SHA, str(raw): LEGACY_METADATA_SHA}


def prepare(library, expected_sha256):
    """Runs in the existing grouped body loader, before any graph capture."""
    if mode() == "none" or expected_sha256 != LEGACY_LIBRARY_SHA:
        return None
    import torch

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Prepare grouped input resource before graph capture")
    device = torch.cuda.current_device()
    if tuple(torch.cuda.get_device_capability(device)) != (12, 0):
        return None
    identity = metadata(os.environ[LIBRARY_ENV], os.environ[METADATA_ENV])
    original = legacy_metadata(str(Path(library).absolute()))
    key = json.dumps(
        {**identity, "legacy_sources": original, "device_index": device}, sort_keys=True
    )
    if key not in _BINDINGS:
        native = ctypes.CDLL(identity["library"])
        call = native.power_hadamard_grouped_input_1024_run
        call.argtypes = (
            [ctypes.c_void_p] * 3
            + [ctypes.c_int] * 3
            + [ctypes.c_float, ctypes.c_size_t]
        )
        call.restype = ctypes.c_int
        _BINDINGS[key] = {
            **identity,
            "legacy_sources": original,
            "device_index": device,
            "library_object": native,
            "call": call,
        }
    _registered_op()
    return key


def eligible(x, scale, binding, context):
    from .yaqa_power_input_resource import explicit_decode

    return (
        x.ndim == 2
        and scale.ndim == 2
        and type(x.shape[0]) is int
        and 1 <= x.shape[0] <= 8
        and type(scale.shape[0]) is int
        and 2 <= scale.shape[0] <= 3
        and x.shape[1] == 2048
        and scale.shape[1] == 2048
        and str(x.dtype) in ("torch.bfloat16", "torch.float32")
        and str(scale.dtype) == "torch.float32"
        and x.device.type == "cuda"
        and scale.device == x.device
        and x.device.index == binding["device_index"]
        and x.is_contiguous()
        and scale.is_contiguous()
        and x.data_ptr() % 16 == 0
        and scale.data_ptr() % 16 == 0
        and explicit_decode(context)
    )


def context_or_none():
    from vllm.forward_context import get_forward_context

    try:
        return get_forward_context()
    except AssertionError:
        return None


def maybe_apply(x, scale, key):
    binding = _BINDINGS.get(key)
    if binding is None:
        raise RuntimeError("Grouped input resource must be constructor-bound")
    if not eligible(x, scale, binding, context_or_none()):
        return None
    return _registered_op()(x, scale, key)


@lru_cache(maxsize=1)
def _registered_op():
    import torch

    @torch.library.custom_op(
        "gguf_yaqa::cuda_grouped_power_input_resource", mutates_args=()
    )
    def op(x: torch.Tensor, scale: torch.Tensor, key: str) -> torch.Tensor:
        binding = _BINDINGS.get(key)
        if binding is None or not eligible(x, scale, binding, context_or_none()):
            raise ValueError("Qualified grouped input resource/phase required")
        groups, rows = scale.shape[0], x.shape[0]
        output = torch.empty((groups, rows, 2048), dtype=torch.float32, device=x.device)
        with torch.cuda.device(x.device):
            code = binding["call"](
                x.data_ptr(),
                scale.data_ptr(),
                output.data_ptr(),
                groups,
                rows,
                1 if x.dtype == torch.bfloat16 else 0,
                2048**-0.5,
                torch.cuda.current_stream(x.device).cuda_stream,
            )
        if code:
            raise RuntimeError("Qualified grouped Hin1024 launch failed: " + str(code))
        return output

    @op.register_fake
    def fake(x, scale, key):
        return x.new_empty((scale.shape[0], x.shape[0], 2048), dtype=torch.float32)

    return op


def runtime_marker(target=None, legacy_identity=None):
    target = tuple(target or ())
    active = mode() == "auto" and target == (12, 0)
    identity = (
        metadata(os.environ[LIBRARY_ENV], os.environ[METADATA_ENV]) if active else {}
    )
    native = {}
    if identity:
        if not isinstance(legacy_identity, dict) or "library" not in legacy_identity:
            raise ValueError("Before-constructor marker needs original grouped library")
        native.update(legacy_metadata(legacy_identity["library"]))
        native[identity["library"]] = LIBRARY_SHA
        native[identity["raw_metadata"]] = METADATA_SHA
    root = Path(__file__).parent
    files = [
        Path(__file__),
        root / "yaqa_grouped_hin_input.py",
        root / "yaqa_power_input_resource.py",
        root.parent / "plugin.py",
        root.parent / "__init__.py",
        SOURCE,
    ]
    return {
        "mode": mode(),
        "target": list(target),
        "threads": 1024,
        "scope": "grouped body; SM120; width2048; groups2/3; decode rows1..8",
        "head_SM86_prefill_unknown_and_other_shapes_keep_original": True,
        "native_identity": identity,
        "native_sources": native,
        "loaded_bindings": [
            {k: v for k, v in b.items() if k not in ("library_object", "call")}
            for b in _BINDINGS.values()
        ],
        "eligible_new_path_per_token_CC_hash_or_env_query": False,
        "sources": {str(p): sha(p) for p in files},
    }
