# SPDX-License-Identifier: Apache-2.0
"""Default-off SM120 residual-A register MMA over the original packed QTIP wire.

Imports/default-none perform no Torch import, device discovery or native build.
The constructor loads and binds both original and residual libraries before
capture; original policy and whole-BM16 unsafe overwrite remain authoritative.
"""

import ctypes
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

ENV = "VLLM_GGUF_YAQA_REGISTER_PRECISION"
SOURCE = Path(__file__).parents[1] / "native/kernels/yaqa_register_mma_residual.cu"
SOURCE_SHA = "59a1290c6d0cc7168bfd8f8bbe22d42e9c653ae15d23810845dbec37180ae384"
ORIGINAL_SOURCE_SHA = "dd841ab110ba52a1729a6a4cad57f332c142bed6e682e56af401bb1fa1138b56"
BUILD_CONTRACT = "sm120-original-qtip2-4-residual-A-register-mma-v1"
PARTIAL_BYTES = 4 << 20
_LIBRARIES = {}


def mode():
    value = os.getenv(ENV, "none")
    if value not in ("none", "auto"):
        raise ValueError(ENV + " must be none or auto")
    return value


def __getattr__(name):
    if name == "prepare":
        from .yaqa_residual_prepare import prepare

        globals()["prepare"] = prepare
        return prepare
    raise AttributeError(name)


@lru_cache(maxsize=64)
def _original_identity(key):
    if not isinstance(key, str):
        return None
    try:
        identity = json.loads(key)
    except (TypeError, ValueError):
        return None
    if (
        not isinstance(identity, dict)
        or identity.get("source_sha256") != ORIGINAL_SOURCE_SHA
    ):
        return None
    architecture = identity.get("architecture")
    if not isinstance(architecture, dict) or tuple(
        architecture.get("compute_capability", ())
    ) != (12, 0):
        return None
    return identity


def _qualified_original(key):
    from ..native.cuda_architecture import validate_library_device
    from . import yaqa_mma as mma

    if (
        not isinstance(key, str)
        or key not in mma._LIBRARIES
        or _original_identity(key) is None
    ):
        return False
    try:
        validate_library_device(key, (12, 0))
    except (ValueError, RuntimeError):
        return False
    return True


@lru_cache(maxsize=1)
def _build_library():
    from .yaqa_cuda import build_cuda_library

    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("Qualified residual CUDA source differs")
    identity = build_cuda_library(
        SOURCE, BUILD_CONTRACT, "yaqa_register_mma_residual", architecture=(12, 0)
    )
    if (
        identity.get("source_sha256") != SOURCE_SHA
        or tuple(identity.get("architecture", {}).get("compute_capability", ()))
        != (12, 0)
        or hashlib.sha256(Path(identity["library"]).read_bytes()).hexdigest()
        != identity.get("library_sha256")
    ):
        raise ValueError("Residual library source/target/binary identity differs")
    return identity


def load(device, bits, original_mma_key):
    if mode() == "none" or type(bits) is not int or bits not in (2, 3, 4):
        return None
    import torch

    device = torch.device(device)
    if device.type != "cuda" or tuple(torch.cuda.get_device_capability(device)) != (
        12,
        0,
    ):
        return None
    from ..native.cuda_architecture import (
        bind_library_architecture,
        device_architecture,
    )

    if device_architecture((12, 0)) != (12, 0) or not _qualified_original(
        original_mma_key
    ):
        return None
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Residual MMA must load and register before graph capture")
    identity = _build_library()
    key = json.dumps(
        {
            **identity,
            "original_mma_key_sha256": hashlib.sha256(
                original_mma_key.encode()
            ).hexdigest(),
        },
        sort_keys=True,
    )
    bind_library_architecture(key, (12, 0))
    if key not in _LIBRARIES:
        library = ctypes.CDLL(identity["library"])
        launch = library.yaqa_register_mma_residual_run
        launch.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 8 + [ctypes.c_void_p]
        launch.restype = ctypes.c_int
        _LIBRARIES[key] = (library, launch, original_mma_key, dict(identity))
    _registered_op()
    from .yaqa_residual_epilogue import prepare as prepare_epilogue

    prepare_epilogue(device, bits, original_mma_key, key)
    return key


def _policy(m, n, rows, bits, original_key):
    from . import yaqa_mma as mma

    value = tuple(mma._launch_policy(m, n, rows))
    if 2 <= rows <= 8 and original_key in mma._QUERY4_KEYS:
        value = tuple(mma._QUERY4_POLICIES.get((bits, m, n), value))
    if len(value) != 4:
        raise ValueError("Original register launch policy differs")
    warps, ring, shared, splits = value
    if (
        warps not in (2, 4, 8)
        or ring not in (2, 4, 8)
        or shared not in (0, 1)
        or type(splits) is not int
        or not 1 <= splits <= 64
        or splits & (splits - 1)
        or rows * m * splits * 4 > PARTIAL_BYTES
    ):
        raise ValueError("Original register resources exceed residual CABI bounds")
    return value


def selected(x, layer):
    if mode() == "none":
        return False
    import torch

    residual_key = getattr(layer, "cuda_residual_mma_key", None)
    original_key = getattr(layer, "cuda_mma_key", None)
    if (
        x.device.type != "cuda"
        or x.dtype not in (torch.float32, torch.bfloat16)
        or x.ndim < 1
        or not x.is_contiguous()
        or getattr(layer, "codec", None) != "qtip"
        or type(getattr(layer, "bits", None)) is not int
        or getattr(layer, "bits", None) not in (2, 3, 4)
        or not getattr(layer, "permutation", False)
        or not getattr(layer, "native_fp16_gemm", False)
        or residual_key not in _LIBRARIES
    ):
        return False
    m, n = layer.shape
    rows = x.numel() // n if n else 0
    if (
        not 2 <= rows <= 32
        or x.shape[-1] != n
        or min(m, n) < 32
        or m % 16
        or n % 16
        or n > 65536
        or rows * m > 2**31 - 1
        or (m // 16) * (n // 16) * (8 * layer.bits) > 2**31 - 1
        or tuple(torch.cuda.get_device_capability(x.device)) != (12, 0)
        or _LIBRARIES[residual_key][2] != original_key
        or not _qualified_original(original_key)
    ):
        return False
    from ..native.cuda_architecture import validate_library_device
    from .yaqa_index40 import selected as index_selected
    from .yaqa_prefill_cache import selected as cache_selected

    try:
        validate_library_device(residual_key, (12, 0))
        _policy(m, n, rows, layer.bits, original_key)
    except (ValueError, RuntimeError):
        return False
    operands = tuple(
        getattr(layer, name, None) for name in ("trellis", "tlut", "mma_lut")
    )
    for value in operands:
        if (
            value is None
            or value.device != x.device
            or not value.is_contiguous()
            or value.data_ptr() % 4
        ):
            return False
    weight, lut, signed = operands
    if (
        weight.dtype != torch.int16
        or weight.shape != (m * n // 256, 16 * layer.bits)
        or lut.dtype != torch.bfloat16
        or lut.shape != (512, 2)
        or signed.dtype != torch.float16
        or signed.shape != (1024, 2)
    ):
        return False
    return not cache_selected(x, layer) and not index_selected(x, layer)


@lru_cache(maxsize=1)
def _registered_op():
    import torch
    import triton

    from ..native.cuda_architecture import validate_library_device
    from . import yaqa_mma as mma

    @torch.library.custom_op("vllm_gguf::yaqa_residual_mma", mutates_args=())
    def op(
        source: torch.Tensor,
        high: torch.Tensor,
        low: torch.Tensor,
        safe: torch.Tensor,
        trellis: torch.Tensor,
        lut: torch.Tensor,
        signed_half_lut: torch.Tensor,
        m: int,
        n: int,
        bits: int,
        original_key: str,
        residual_key: str,
    ) -> torch.Tensor:
        rows = source.shape[0] if source.ndim == 2 else 0
        values = (high, low, safe, trellis, lut, signed_half_lut)
        if (
            source.device.type != "cuda"
            or source.dtype != torch.float32
            or source.ndim != 2
            or source.shape != (rows, n)
            or not 2 <= rows <= 32
            or bits not in (2, 3, 4)
            or min(m, n) < 32
            or m % 16
            or n % 16
            or n > 65536
            or rows * m > 2**31 - 1
            or (m // 16) * (n // 16) * (8 * bits) > 2**31 - 1
            or high.dtype != torch.float16
            or low.dtype != torch.float16
            or high.shape != source.shape
            or low.shape != source.shape
            or safe.dtype != torch.int32
            or safe.shape != (rows,)
            or trellis.dtype != torch.int16
            or trellis.shape != (m * n // 256, 16 * bits)
            or lut.dtype != torch.bfloat16
            or lut.shape != (512, 2)
            or signed_half_lut.dtype != torch.float16
            or signed_half_lut.shape != (1024, 2)
            or not source.is_contiguous()
            or any(t.device != source.device or not t.is_contiguous() for t in values)
            or any(t.data_ptr() % 4 for t in (high, low, trellis, signed_half_lut))
        ):
            raise ValueError("Invalid residual original-QTIP register operands")
        if residual_key not in _LIBRARIES or not _qualified_original(original_key):
            raise RuntimeError("Original/residual MMA must load before capture")
        if _LIBRARIES[residual_key][2] != original_key:
            raise ValueError("Residual library was prepared for another original key")
        capability = tuple(torch.cuda.get_device_capability(source.device))
        validate_library_device(original_key, capability)
        validate_library_device(residual_key, capability)
        warps, ring, shared, splits = _policy(m, n, rows, bits, original_key)
        from .yaqa_residual_epilogue import maybe_apply as fused_epilogue

        fused = fused_epilogue(
            source,
            high,
            low,
            safe,
            trellis,
            lut,
            signed_half_lut,
            m,
            n,
            bits,
            original_key,
            residual_key,
            (warps, ring, shared, splits),
        )
        if fused is not None:
            return fused
        output = torch.empty((rows, m), device=source.device, dtype=torch.float32)
        partial = (
            torch.empty((splits, rows, m), device=source.device, dtype=torch.float32)
            if splits > 1
            else output
        )
        with torch.cuda.device(source.device):
            error = _LIBRARIES[residual_key][1](
                high.data_ptr(),
                low.data_ptr(),
                trellis.data_ptr(),
                signed_half_lut.data_ptr(),
                partial.data_ptr(),
                output.data_ptr(),
                m,
                n,
                rows,
                bits,
                warps,
                ring,
                shared,
                splits,
                torch.cuda.current_stream(source.device).cuda_stream,
            )
            if error:
                raise RuntimeError(f"Residual register MMA launch failed: {error}")
            mma._overwrite_unsafe[(triton.cdiv(rows, 16), triton.cdiv(m, 32))](
                source,
                high,
                safe,
                trellis,
                lut,
                output,
                M=m,
                N=n,
                BATCH=rows,
                BITS=bits,
                BM=16,
                BN=32,
                BK=64,
                num_warps=4,
                num_stages=2,
            )
        return output

    @op.register_fake
    def fake(
        source,
        high,
        low,
        safe,
        trellis,
        lut,
        signed_half_lut,
        m,
        n,
        bits,
        original_key,
        residual_key,
    ):
        return source.new_empty((source.shape[0], m), dtype=torch.float32)

    return op


def residual_mma_linear(
    source,
    high,
    low,
    safe,
    trellis,
    lut,
    signed_half_lut,
    m,
    n,
    bits,
    original_key,
    residual_key,
):
    return _registered_op()(
        source,
        high,
        low,
        safe,
        trellis,
        lut,
        signed_half_lut,
        m,
        n,
        bits,
        original_key,
        residual_key,
    )


def runtime_marker(target=None):
    root = Path(__file__).resolve().parent
    paths = [
        Path(__file__),
        root / "yaqa_residual_prepare.py",
        SOURCE,
        root / "yaqa_mma.py",
        root / "yaqa.py",
        root / "yaqa_kernels.py",
        root / "yaqa_cuda.py",
        root / "yaqa_index40.py",
        root / "yaqa_prefill_cache.py",
        root / "yaqa_prefill_cache_out.py",
        root / "resident_small_decode.py",
        root / "inactive_padding.py",
        root / "prefill_context.py",
        root.parent / "__init__.py",
        root.parent / "native/cuda_architecture.py",
        root.parent / "native/cuda_toolchain.py",
        root.parent / "native/kernels/yaqa_register_mma.cu",
    ]
    active = mode() == "auto"
    identity = _build_library() if active and target == (12, 0) else {}
    return {
        "mode": mode(),
        "scope": "SM120 original QTIP2..4 physicalM2..32, bounded original residual core",
        "library": identity,
        "loaded_libraries": {key: record[3] for key, record in _LIBRARIES.items()},
        "original_policy_and_QUERY4_preserved": False,
        "original_M4_and_unlisted_geometry_policy_preserved": True,
        "small_row_resource_policy": {
            "version": "original-query4-resources-for-physical-M2-through-M8-v1",
            "same_physical_WMMA_M16_tile": True,
            "source_table": "original yaqa_mma._QUERY4_POLICIES",
            "requires_original_QUERY4_key_admission": True,
            "partial_buffer_ceiling_bytes": PARTIAL_BYTES,
            "new_shader_or_weight_cache": False,
        },
        "whole_BM16_unsafe_BK64_overwrite_preserved": True,
        "weight_cache_added_bytes": 0,
        "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
