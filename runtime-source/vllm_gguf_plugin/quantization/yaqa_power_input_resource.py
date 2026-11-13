# SPDX-License-Identifier: Apache-2.0
"""Default-off exact SM120 body power-input resource; no math/weight changes.

Libraries/target/device are authenticated once before capture. The opaque call
uses tensor metadata and the explicit graph phase, never per-token CC/hash/env.
"""

import ctypes
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

ENV = "VLLM_GGUF_YAQA_POWER_INPUT_RESOURCE"
LIBRARY_ENV = ENV + "_LIBRARY"
METADATA_ENV = ENV + "_METADATA"
SOURCE_SHA = "7a99052cd7a4447f32cbf72a9352b19da47668ffc4f4f7e624ee5ec3247518c9"
LIBRARY_SHA = "056906d7235bab12d39db173cac1fc69ce306281c41fed8013f0bce3b6caf365"
METADATA_SHA = "49da06c46bc6ca2e4812c799305617fe7d75aaed262f48488331feefb12deeeb"
LEGACY_LIBRARY_SHA = "67583dadb44ad1fbd35c1b50d162f871a9e4a200c97916f500db7045fa95cb7a"
LEGACY_METADATA_SHA = "e8cfd79e656cfc9f2c7882f877ccd5622b2701ecc674eedc67ddabed7f842544"
_BINDINGS = {}
SOURCE = (
    Path(__file__).parents[1] / "native/kernels/yaqa_power_input_large_threads_r1.cu"
)


def mode():
    value = os.getenv(ENV, "none")
    if value not in ("none", "auto"):
        raise ValueError(ENV + " requires none or auto")
    return value


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@lru_cache(maxsize=8)
def metadata(library, raw_metadata):
    library, raw_metadata = Path(library).absolute(), Path(raw_metadata).absolute()
    if (
        sha(library) != LIBRARY_SHA
        or sha(raw_metadata) != METADATA_SHA
        or sha(SOURCE) != SOURCE_SHA
    ):
        raise ValueError("Qualified Hin1024 binary/raw metadata differs")
    identity = json.loads(raw_metadata.read_text())
    if (
        identity.get("library_sha256") != LIBRARY_SHA
        or identity.get("source_sha256") != SOURCE_SHA
        or identity.get("architecture", {}).get("compute_capability") != [12, 0]
    ):
        raise ValueError("Qualified Hin1024 source/target identity differs")
    return {
        "library": str(library),
        "library_sha256": LIBRARY_SHA,
        "raw_metadata": str(raw_metadata),
        "raw_metadata_sha256": METADATA_SHA,
        "source_sha256": SOURCE_SHA,
        "compute_capability": (12, 0),
    }


@lru_cache(maxsize=8)
def legacy_metadata(library):
    library = Path(library).absolute()
    raw = library.with_name("identity.json")
    if sha(library) != LEGACY_LIBRARY_SHA or sha(raw) != LEGACY_METADATA_SHA:
        raise ValueError("Original Hin binary/raw metadata source pair differs")
    return {str(library): LEGACY_LIBRARY_SHA, str(raw): LEGACY_METADATA_SHA}


def explicit_decode(context):
    # Unknown/eager/untagged phases retain the old path, including tiny prefill.
    descriptor = getattr(context, "batch_descriptor", None)
    return getattr(descriptor, "is_prefill", None) is False


def eligible(rows, width, dtype, device_index, binding, context):
    return (
        type(rows) is int
        and 1 <= rows <= 8
        and width == 2048
        and str(dtype) in ("torch.bfloat16", "torch.float32")
        and device_index == binding["device_index"]
        and explicit_decode(context)
    )


def prepare_body_layer(layer):
    """Called only from the generic LinearMethod body factory, after old ctor."""
    if mode() == "none":
        return None
    if (
        getattr(layer, "codec", None) != "qtip"
        or getattr(layer, "_native_device_capability", None) != (12, 0)
        or getattr(layer, "shape", (None, None))[1] != 2048
        or getattr(layer, "cuda_power_key", None) is None
    ):
        return None
    import torch

    from ..native.cuda_architecture import validate_library_device
    from . import yaqa_power as original

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Prepare Hin resource before graph capture")
    legacy_key = layer.cuda_power_key
    validate_library_device(legacy_key, (12, 0))
    if legacy_key != LEGACY_LIBRARY_SHA or legacy_key not in original._LIBRARIES:
        raise ValueError("Original qualified Hin library must remain loaded")
    identity = metadata(os.environ[LIBRARY_ENV], os.environ[METADATA_ENV])
    device_index = layer.trellis.device.index
    if type(device_index) is not int:
        raise ValueError("Actual constructor-bound CUDA device index required")
    old_path = Path(original._LIBRARIES[legacy_key]._name).absolute()
    key = json.dumps(
        {
            **identity,
            "legacy_key": legacy_key,
            "legacy_library": str(old_path),
            "device_index": device_index,
        },
        sort_keys=True,
    )
    if key not in _BINDINGS:
        old_metadata = old_path.with_name("identity.json")
        legacy_metadata(str(old_path))
        library = ctypes.CDLL(identity["library"])
        call = library.power_hadamard_large_threads_run
        call.argtypes = (
            [ctypes.c_void_p] * 4
            + [ctypes.c_int] * 9
            + [ctypes.c_float, ctypes.c_size_t]
        )
        call.restype = ctypes.c_int
        _BINDINGS[key] = {
            **identity,
            "device_index": device_index,
            "legacy_key": legacy_key,
            "legacy_library": str(old_path),
            "legacy_metadata": str(old_metadata),
            "library_object": library,
            "call": call,
        }
    _registered_op()
    layer.cuda_power_input_resource_key = key
    return key


@lru_cache(maxsize=1)
def _registered_op():
    import torch

    @torch.library.custom_op("gguf_yaqa::cuda_power_input_resource", mutates_args=())
    def op(
        x: torch.Tensor, scale: torch.Tensor, legacy_key: str, resource_key: str
    ) -> torch.Tensor:
        from vllm.forward_context import get_forward_context

        from .yaqa_power import cuda_power_input

        binding = _BINDINGS.get(resource_key)
        if binding is None or binding["legacy_key"] != legacy_key:
            raise RuntimeError("Hin resource must be bound by the body constructor")
        try:
            context = get_forward_context()
        except AssertionError:
            context = None
        rows = x.shape[0] if x.ndim == 2 else 0
        if not (
            x.device.type == "cuda"
            and x.ndim == 2
            and x.is_contiguous()
            and scale.shape == (2048,)
            and scale.dtype == torch.float32
            and scale.device == x.device
            and scale.is_contiguous()
            and x.data_ptr() % 16 == 0
            and scale.data_ptr() % 16 == 0
            and eligible(rows, x.shape[-1], x.dtype, x.device.index, binding, context)
        ):
            return cuda_power_input(x, scale, legacy_key)
        output = torch.empty_like(x, dtype=torch.float32)
        with torch.cuda.device(x.device):
            code = binding["call"](
                x.data_ptr(),
                scale.data_ptr(),
                scale.data_ptr(),
                output.data_ptr(),
                2048,
                2048,
                rows,
                1 if x.dtype == torch.bfloat16 else 0,
                0,
                0,
                1024,
                2,
                0,
                2048**-0.5,
                torch.cuda.current_stream(x.device).cuda_stream,
            )
        if code:
            raise RuntimeError("Qualified Hin1024 launch failed: " + str(code))
        return output

    @op.register_fake
    def fake(x, scale, legacy_key, resource_key):
        return x.new_empty(x.shape, dtype=torch.float32)

    return op


def power_input(x, scale, legacy_key, resource_key):
    return _registered_op()(x, scale, legacy_key, resource_key)


def runtime_marker(target=None, legacy_identity=None):
    root = Path(__file__).resolve().parent
    active = mode() == "auto" and tuple(target or ()) == (12, 0)
    identity = (
        metadata(os.environ[LIBRARY_ENV], os.environ[METADATA_ENV]) if active else {}
    )
    paths = [
        Path(__file__),
        root / "yaqa_power.py",
        root / "yaqa_kernels.py",
        root / "yaqa_linear.py",
        root / "config.py",
        root.parent / "plugin.py",
        root.parent / "__init__.py",
        root.parent / "native/cuda_architecture.py",
        root.parent / "native/kernels/yaqa_power_hadamard.cu",
        SOURCE,
    ]
    native = {}
    for binding in _BINDINGS.values():
        native[binding["library"]] = LIBRARY_SHA
        native[binding["raw_metadata"]] = METADATA_SHA
        native[binding["legacy_library"]] = LEGACY_LIBRARY_SHA
        native[binding["legacy_metadata"]] = LEGACY_METADATA_SHA
    if identity:
        native[identity["library"]] = LIBRARY_SHA
        native[identity["raw_metadata"]] = METADATA_SHA
        if not isinstance(legacy_identity, dict) or "library" not in legacy_identity:
            raise ValueError(
                "Before-constructor marker requires actual old Hin identity"
            )
        native.update(legacy_metadata(legacy_identity["library"]))
    return {
        "mode": mode(),
        "target": list(target or ()),
        "scope": "body; SM120; P2048; explicit decode; physicalrows1..8",
        "threads": 1024,
        "head_prefill_SM86_and_unknown_keep_256": True,
        "grouped_input_keeps_256": True,
        "native_identity": identity,
        "native_sources": native,
        "loaded_bindings": [
            {k: v for k, v in b.items() if k not in ("library_object", "call")}
            for b in _BINDINGS.values()
        ],
        "eligible_new_path_per_token_CC_hash_or_env_query": False,
        "sources": {str(p): sha(p) for p in paths},
    }
