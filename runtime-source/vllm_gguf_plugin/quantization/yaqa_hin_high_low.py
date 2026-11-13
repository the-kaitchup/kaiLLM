# SPDX-License-Identifier: Apache-2.0
"""Default-off exact Hin1024 output preparation, retaining original core guards.

Constructor preparation owns environment/source/native/device checks. Runtime
admission keeps the current residual selector and explicit decode phase, and
never reads tensor contents. RawF32 remains the original unsafe-repair input.
"""

import ctypes
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

ENV = "VLLM_GGUF_YAQA_HIN_HIGH_LOW"
LIBRARY_ENV = ENV + "_LIBRARY"
METADATA_ENV = ENV + "_METADATA"
SOURCE_SHA = "5e2d2c27f45b340c34cefa8f549c436f2efb7498fbf363fd0de7926125cbc9d8"
LIBRARY_SHA = "9388ae5ddce46bffa243ea3ef800e3d6e1e18437eecd031f0a8507ab29631f8b"
METADATA_SHA = "305af112731bda6f8db32fc95b3aaeaa5b0111ffc7d1e9e8c3f79820a63f21ec"
SOURCE = Path(__file__).parents[1] / "native/kernels/yaqa_hin_high_low_candidate.cu"
_BINDINGS = {}


def mode():
    value = os.getenv(ENV, "none")
    if value not in ("none", "auto"):
        raise ValueError(ENV + " requires none or auto")
    return value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=8)
def metadata(library, raw_metadata):
    library, raw_metadata = Path(library).absolute(), Path(raw_metadata).absolute()
    if (
        sha(library) != LIBRARY_SHA
        or sha(raw_metadata) != METADATA_SHA
        or sha(SOURCE) != SOURCE_SHA
    ):
        raise ValueError("Qualified HinHiLo native/source/raw pair differs")
    identity = json.loads(raw_metadata.read_text())
    if (
        identity.get("source_sha256") != SOURCE_SHA
        or identity.get("library_sha256") != LIBRARY_SHA
        or identity.get("architecture", {}).get("compute_capability") != [12, 0]
    ):
        raise ValueError("HinHiLo source/target identity differs")
    return {
        "library": str(library),
        "library_sha256": LIBRARY_SHA,
        "raw_metadata": str(raw_metadata),
        "raw_metadata_sha256": METADATA_SHA,
        "source_sha256": SOURCE_SHA,
        "compute_capability": (12, 0),
    }


def geometry(groups, rows):
    return (
        type(groups) is int
        and type(rows) is int
        and groups in (1, 2)
        and 2 <= rows <= 8
        and groups * rows * (2048 * 8 + 4) <= 4 << 20
    )


def _decode():
    from vllm.forward_context import get_forward_context

    from .yaqa_power_input_resource import explicit_decode

    try:
        return explicit_decode(get_forward_context())
    except AssertionError:
        return False


def prepare_body_layer(layer):
    if mode() == "none":
        return None
    if (
        getattr(layer, "_native_device_capability", None) != (12, 0)
        or getattr(layer, "codec", None) != "qtip"
        or getattr(layer, "bits", None) not in (2, 3, 4)
        or getattr(layer, "shape", (None, None))[1] != 2048
        or getattr(getattr(layer, "hadamard_input", None), "shape", None) != (1, 1)
        or not getattr(layer, "native_fp16_gemm", False)
        or getattr(layer, "cuda_residual_mma_key", None) is None
        or getattr(layer, "cuda_power_input_resource_key", None) is None
    ):
        return None
    import torch

    from . import yaqa_power_input_resource as old

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Prepare HinHiLo before capture")
    prior = old._BINDINGS.get(layer.cuda_power_input_resource_key)
    device = layer.trellis.device
    if (
        prior is None
        or prior["device_index"] != device.index
        or prior["library_sha256"] != old.LIBRARY_SHA
    ):
        raise ValueError("Original qualified body Hin binding required")
    identity = metadata(os.environ[LIBRARY_ENV], os.environ[METADATA_ENV])
    key = json.dumps({**identity, "device_index": device.index}, sort_keys=True)
    if key not in _BINDINGS:
        native = ctypes.CDLL(identity["library"])
        call = native.power_hadamard_high_low_run
        call.argtypes = (
            [ctypes.c_void_p] * 7
            + [ctypes.c_int] * 4
            + [ctypes.c_float, ctypes.c_size_t]
        )
        call.restype = ctypes.c_int
        _BINDINGS[key] = {
            **identity,
            "device_index": device.index,
            "library_object": native,
            "call": call,
        }
    _registered_op()
    layer.cuda_hin_high_low_key = key
    return key


def selected(x, layer):
    key = getattr(layer, "cuda_hin_high_low_key", None)
    if key not in _BINDINGS or getattr(layer, "validate_b1", False) or not _decode():
        return False
    from .yaqa_residual_mma import selected as original_selected

    return (
        layer.shape[1] == 2048
        and geometry(1, x.numel() // 2048)
        and layer.hadamard_input.shape == (1, 1)
        and x.device.index == _BINDINGS[key]["device_index"]
        and original_selected(x, layer)
        and x.data_ptr() % 16 == 0
        and layer.input_scale.shape == (2048,)
        and str(layer.input_scale.dtype) == "torch.float32"
        and layer.input_scale.device == x.device
        and layer.input_scale.is_contiguous()
        and layer.input_scale.data_ptr() % 16 == 0
    )


@lru_cache(maxsize=1)
def _registered_op():
    import torch

    @torch.library.custom_op("gguf_yaqa::cuda_hin_high_low", mutates_args=())
    def op(
        x: torch.Tensor, scales: torch.Tensor, padding: torch.Tensor | None, key: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        binding = _BINDINGS.get(key)
        groups = scales.shape[0] if scales.ndim == 2 else 0
        rows = x.shape[0] if x.ndim == 2 else 0
        if not (
            binding is not None
            and _decode()
            and geometry(groups, rows)
            and x.shape == (rows, 2048)
            and x.dtype in (torch.bfloat16, torch.float32)
            and x.device.type == "cuda"
            and x.device.index == binding["device_index"]
            and scales.shape == (groups, 2048)
            and scales.dtype == torch.float32
            and scales.device == x.device
            and x.data_ptr() % 16 == 0
            and scales.data_ptr() % 16 == 0
            and x.is_contiguous()
            and scales.is_contiguous()
        ):
            raise ValueError("HinHiLo requires a prepared selected decode input")
        if padding is not None and not (
            padding.shape == (rows,)
            and padding.dtype == torch.bool
            and padding.device == x.device
            and padding.is_contiguous()
        ):
            raise ValueError("Official shared body bool[M] padding required")
        shape = (groups, rows, 2048)
        raw = torch.empty(shape, device=x.device, dtype=torch.float32)
        high = torch.empty(shape, device=x.device, dtype=torch.float16)
        low = torch.empty_like(high)
        safe = torch.empty((groups, rows), device=x.device, dtype=torch.int32)
        with torch.cuda.device(x.device):
            code = binding["call"](
                x.data_ptr(),
                scales.data_ptr(),
                raw.data_ptr(),
                high.data_ptr(),
                low.data_ptr(),
                safe.data_ptr(),
                0 if padding is None else padding.data_ptr(),
                groups,
                rows,
                int(x.dtype == torch.bfloat16),
                int(padding is not None),
                2048**-0.5,
                torch.cuda.current_stream(x.device).cuda_stream,
            )
        if code:
            raise RuntimeError("Qualified HinHiLo native launch failed: " + str(code))
        return raw, high, low, safe

    @op.register_fake
    def fake(x, scales, padding, key):
        shape = (scales.shape[0], x.shape[0], 2048)
        return (
            x.new_empty(shape, dtype=torch.float32),
            x.new_empty(shape, dtype=torch.float16),
            x.new_empty(shape, dtype=torch.float16),
            x.new_empty(shape[:2], dtype=torch.int32),
        )

    return op


def _apply(x, scales, padding, key):
    import torch

    return torch.ops.gguf_yaqa.cuda_hin_high_low(x, scales, padding, key)


def maybe_individual(x, source, layer):
    if not selected(x, layer):
        return None
    from .inactive_padding import mask_for_rows

    rows = source.shape[0]
    padding = mask_for_rows(source, rows, role="body")
    outputs = _apply(
        source, layer.input_scale.reshape(1, 2048), padding, layer.cuda_hin_high_low_key
    )
    return tuple(value[0] for value in outputs)


def maybe_grouped(x, parts, scales):
    if len(parts) != 2 or not all(selected(x, part) for part in parts):
        return None
    if not (
        scales.shape == (2, 2048)
        and str(scales.dtype) == "torch.float32"
        and scales.device == x.device
        and scales.is_contiguous()
        and scales.data_ptr() % 16 == 0
    ):
        return None
    keys = {part.cuda_hin_high_low_key for part in parts}
    if len(keys) != 1:
        return None
    from .inactive_padding import mask_for_rows

    source = x.reshape(-1, 2048).contiguous()
    if not geometry(2, source.shape[0]):
        return None
    padding = mask_for_rows(source, source.shape[0], role="body")
    return _apply(source, scales, padding, next(iter(keys)))


def validate_prepared(source, prepared, rows, width):
    """Validate borrowed tensor metadata without reading any device values."""
    import torch

    if (
        not isinstance(prepared, tuple)
        or len(prepared) != 3
        or width != 2048
        or not geometry(1, rows)
    ):
        raise ValueError("Prepared residual requires the exact High/Low/Safe tuple")
    high, low, safe = prepared
    if not (
        source.dtype == torch.float32
        and source.device.type == "cuda"
        and source.shape == (rows, width)
        and source.is_contiguous()
        and high.shape == low.shape == source.shape
        and high.dtype == low.dtype == torch.float16
        and safe.shape == (rows,)
        and safe.dtype == torch.int32
        and all(
            value.device == source.device and value.is_contiguous()
            for value in prepared
        )
        and all(value.data_ptr() % 4 == 0 for value in prepared)
    ):
        raise ValueError("Prepared residual tensor metadata differs")


def runtime_marker(target=None):
    active = mode() == "auto" and tuple(target or ()) == (12, 0)
    identity = (
        metadata(os.environ[LIBRARY_ENV], os.environ[METADATA_ENV]) if active else {}
    )
    root = Path(__file__).resolve().parent
    paths = [
        Path(__file__),
        SOURCE,
        root / "yaqa_residual_prepare.py",
        root / "yaqa_residual_mma.py",
        root / "yaqa_kernels.py",
        root / "yaqa_grouped_k234_dispatch.py",
        root / "yaqa_linear.py",
        root / "inactive_padding.py",
        root / "yaqa_power_input_resource.py",
    ]
    return {
        "mode": mode(),
        "active": active,
        "compute_capability": [12, 0],
        "sources": {str(path): sha(path) for path in paths},
        "native": {
            identity["library"]: LIBRARY_SHA,
            identity["raw_metadata"]: METADATA_SHA,
        }
        if active
        else {},
        "rows": "2..8, original residual selector authoritative",
        "groups": [1, 2],
        "width": 2048,
        "decode_only": True,
        "prepared_tuple": "Original RawF32 plus borrowed HalfHigh/HalfLow/I32Safe",
        "padding": "One live official body Bool[M], shared by independent groups",
        "new_weight_bytes": 0,
        "new_persistent_state_bytes": 0,
        "fallback": "Original Hin/preparation/core/post for every unselected case",
        "old_head_factory_unchanged": True,
    }
