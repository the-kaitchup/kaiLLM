# SPDX-License-Identifier: Apache-2.0
"""Unselected SM86 lossless pair-index cache with unchanged native MMA math."""

import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl

from . import yaqa_mma as original
from .yaqa_cuda import build_cuda_library

SOURCE = Path(__file__).parents[1] / "native/kernels/yaqa_index40_mma.cu"
_LIBRARIES = {}


@triton.jit
def _pack40(W, Out, TOTAL: tl.constexpr, BITS: tl.constexpr, TILES: tl.constexpr):
    tile = tl.program_id(0) * TILES + tl.arange(0, TILES)
    lane = tl.arange(0, 32)
    packed = tl.zeros((TILES, 32), tl.uint32)
    final = tl.zeros((TILES, 32), tl.uint32)
    for phase in tl.static_range(4):
        start = (lane[None, :] * 4 + phase) * (2 * BITS)
        pos = start // 16
        shift = start % 16
        lo = (
            tl.load(W + tile[:, None] * (16 * BITS) + pos, tile[:, None] < TOTAL, 0)
            .to(tl.uint16)
            .to(tl.uint32)
        )
        hi = (
            tl.load(
                W + tile[:, None] * (16 * BITS) + (pos + 1) % (16 * BITS),
                tile[:, None] < TOTAL,
                0,
            )
            .to(tl.uint16)
            .to(tl.uint32)
        )
        state = (((lo << 16) | hi) >> (16 - shift)) & 65535
        hashed = state * (state + 1)
        code = (hashed >> 6) & 1023
        packed |= code << (10 * phase)
        if phase == 3:
            final = code >> 2
    tl.store(Out + tile[:, None] * 40 + lane[None, :], packed, tile[:, None] < TOTAL)
    bytes_out = tl.cast(Out, tl.pointer_type(tl.uint8))
    tl.store(
        bytes_out + tile[:, None] * 160 + 128 + lane[None, :],
        final.to(tl.uint8),
        tile[:, None] < TOTAL,
    )


def build():
    return build_cuda_library(
        SOURCE,
        "sm86-index40-original-register-mma-F16-same-sumtree-v1",
        "yaqa_index40_register_mma",
    )


def load():
    if torch.cuda.get_device_capability() != (8, 6):
        raise ValueError("Candidate requires SM86")
    identity = build()
    key = identity["library"]
    if key not in _LIBRARIES:
        library = ctypes.CDLL(key)
        launch = library.yaqa_index40_register_mma_run
        launch.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 8 + [ctypes.c_void_p]
        launch.restype = ctypes.c_int
        _LIBRARIES[key] = (library, launch)
    return key


def pack(trellis, m, n, bits):
    if (
        bits not in (3, 4)
        or trellis.dtype != torch.int16
        or not trellis.is_cuda
        or not trellis.is_contiguous()
        or tuple(trellis.shape) != (m * n // 256, 16 * bits)
        or m % 16
        or n % 16
    ):
        raise ValueError("Only original permuted QTIP3/4 tiles")
    count = m * n // 256
    codes = torch.empty((count, 40), device=trellis.device, dtype=torch.int32)
    _pack40[(triton.cdiv(count, 16),)](
        trellis, codes, TOTAL=count, BITS=bits, TILES=16, num_warps=4
    )
    return codes


@torch.library.custom_op("vllm_gguf::index40_register_mma", mutates_args=())
def linear(
    source: torch.Tensor,
    half: torch.Tensor,
    safe_rows: torch.Tensor,
    trellis: torch.Tensor,
    lut: torch.Tensor,
    signed_half_lut: torch.Tensor,
    codes: torch.Tensor,
    m: int,
    n: int,
    bits: int,
    library_key: str,
) -> torch.Tensor:
    batch = source.shape[0]
    if (
        source.dtype != torch.float32
        or not source.is_cuda
        or not source.is_contiguous()
        or source.shape != (batch, n)
        or batch not in (2, 4, 8, 16)
        or bits not in (3, 4)
        or codes.dtype != torch.int32
        or codes.shape != (m * n // 256, 40)
        or half.dtype != torch.float16
        or half.shape != source.shape
        or safe_rows.dtype != torch.int32
        or safe_rows.shape != (batch,)
        or trellis.shape != (m * n // 256, 16 * bits)
        or trellis.dtype != torch.int16
        or lut.dtype != torch.bfloat16
        or lut.shape != (512, 2)
        or signed_half_lut.dtype != torch.float16
        or signed_half_lut.shape != (1024, 2)
        or any(
            x.device != source.device or not x.is_contiguous()
            for x in (half, safe_rows, trellis, lut, signed_half_lut, codes)
        )
    ):
        raise ValueError("Invalid indexed register-MMA component inputs")
    if library_key not in _LIBRARIES:
        raise ValueError("Load candidate library before graph capture")
    warps, ring, shared, splits = original._launch_policy(m, n, batch)
    output = source.new_empty((batch, m))
    partial = source.new_empty((splits, batch, m)) if splits > 1 else output
    error = _LIBRARIES[library_key][1](
        half.data_ptr(),
        codes.data_ptr(),
        signed_half_lut.data_ptr(),
        partial.data_ptr(),
        output.data_ptr(),
        m,
        n,
        batch,
        bits,
        warps,
        ring,
        shared,
        splits,
        torch.cuda.current_stream(source.device).cuda_stream,
    )
    if error:
        raise RuntimeError(f"Indexed MMA CUDA error{error}")
    original._overwrite_unsafe[(triton.cdiv(batch, 16), triton.cdiv(m, 32))](
        source,
        half,
        safe_rows,
        trellis,
        lut,
        output,
        M=m,
        N=n,
        BATCH=batch,
        BITS=bits,
        BM=16,
        BN=32,
        BK=64,
        num_warps=4,
        num_stages=2,
    )
    return output


@linear.register_fake
def _fake(
    source,
    half,
    safe_rows,
    trellis,
    lut,
    signed_half_lut,
    codes,
    m,
    n,
    bits,
    library_key,
):
    return source.new_empty((source.shape[0], m))


def identity():
    return {
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "original_helper_sha256": hashlib.sha256(
            Path(original.__file__).read_bytes()
        ).hexdigest(),
        "bytes_per_tile": 160,
        "bits_per_weight": 5,
        "same_MMA_and_repair": True,
        "whole_model_EAR_required": True,
    }


ENV = "VLLM_GGUF_YAQA_INDEX40_CACHE"
BUDGET_ENV = "VLLM_GGUF_YAQA_INDEX40_CACHE_BYTES"
MODE = "qtip4-gate-down-sm86"
SHAPES = ((7680, 2048), (2048, 7680))
ROWS = (2, 4, 8)
MAX_BUDGET = 1 << 30
ORIGINAL_MMA_SHA = "d5c3ca315cfaa6c055d759fd019199801eb261a4c7c7e9a131fa7431fc4acbb1"
CUDA_SOURCE_SHA = "f1f3e244ee6da6f71276c9c88ccf7cbabae147f7b0a71c4e43e697c99735241f"
_WARMED = set()


def mode():
    value = os.getenv(ENV, "none")
    if value not in ("none", MODE):
        raise ValueError(f"{ENV} must be none or {MODE}")
    return value


def budget_bytes():
    if mode() == "none":
        return 0
    value = int(os.getenv(BUDGET_ENV, "0"))
    if not 0 < value <= MAX_BUDGET:
        raise ValueError("Index40 requires an explicit positive budget <=1GiB")
    return value


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def descriptor_sha(descriptor):
    return hashlib.sha256(
        json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def eligible_descriptor(descriptor):
    return (
        descriptor is not None
        and descriptor.get("codec") == "qtip"
        and descriptor.get("bits") == 4
        and descriptor.get("tensor_core_permutation") is True
        and tuple(descriptor.get("shape", ())) in SHAPES
    )


class CachePlan:
    """One checkpoint/rank; all selected matrices fit before GPU allocation."""

    def __init__(self, entries, budget):
        self.entries = copy.deepcopy(entries)
        self.budget = budget
        self.expected_bytes = sum(row["bytes"] for row in entries.values())
        if self.expected_bytes > budget:
            raise ValueError(
                f"Whole index40 needs{self.expected_bytes}bytes, budget{budget}; "
                "no partial cache"
            )
        self.loaded = {}
        self.complete = not entries

    def require(self, key, descriptor):
        row = self.entries.get(key)
        if row is None or row["descriptor_sha256"] != descriptor_sha(descriptor):
            raise ValueError("Indexed layer differs from whole-profile plan")
        if key in self.loaded or self.complete:
            raise ValueError("Index40 cache matrix already constructed")
        return row

    def finish(self, key, record):
        if key in self.loaded or record["bytes"] != self.entries[key]["bytes"]:
            raise ValueError("Index40 actual allocation differs from preflight")
        self.loaded[key] = copy.deepcopy(record)
        if self.loaded.keys() == self.entries.keys():
            if sum(row["bytes"] for row in self.loaded.values()) != self.expected_bytes:
                raise ValueError("Index40 complete cache byte mismatch")
            self.complete = True

    def ensure_complete(self):
        # No O(layers) work on the per-token path; completion is set once atload.
        if not self.complete:
            raise ValueError("Incomplete indexed cache; inference forbidden")

    def summary(self):
        return {
            "mode": MODE,
            "budget_bytes": self.budget,
            "expected_matrices": len(self.entries),
            "expected_bytes": self.expected_bytes,
            "loaded_matrices": len(self.loaded),
            "loaded_bytes": sum(x["bytes"] for x in self.loaded.values()),
            "complete": self.complete,
            "entries": copy.deepcopy(self.entries),
            "loaded": copy.deepcopy(self.loaded),
        }


def create_plan(sources):
    if mode() == "none":
        return None
    budget = budget_bytes()
    entries = {}
    for key, (path, logical, descriptor) in sources.items():
        if not eligible_descriptor(descriptor):
            continue
        m, n = descriptor["shape"]
        entries[key] = {
            "source": str(path),
            "logical_name": logical,
            "shape": [m, n],
            "bits": 4,
            "bytes": (m * n // 256) * 160,
            "descriptor_sha256": descriptor_sha(descriptor),
        }
    return CachePlan(entries, budget)


def rebind_plan(plan, sources):
    if plan is None:
        return None
    if plan.loaded:
        raise ValueError("Cannot remap indexed cache after allocation starts")
    rebound = create_plan(sources)

    def records(value):
        return sorted(
            (
                r["source"],
                r["logical_name"],
                tuple(r["shape"]),
                r["bits"],
                r["bytes"],
                r["descriptor_sha256"],
            )
            for r in value.entries.values()
        )

    if (
        rebound is None
        or rebound.budget != plan.budget
        or records(rebound) != records(plan)
    ):
        raise ValueError("Mapper changed indexed sources/budget/complete selection")
    return rebound


def _tensor_identity(value):
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "bytes": value.numel() * value.element_size(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def expected_code_sha(trellis):
    """Independent CPU bitstream packing; no persistent CPU copy is retained."""
    import numpy as np

    if (
        trellis.device.type != "cpu"
        or trellis.dtype != torch.int16
        or trellis.shape[1] != 64
    ):
        raise ValueError("Index40 CPU identity requires original QTIP4 trellis")
    words = trellis.detach().contiguous().numpy().view(np.uint16)
    lanes = np.arange(32, dtype=np.int64)
    low = np.zeros((len(words), 32), dtype=np.uint32)
    for phase in range(4):
        start = (lanes * 4 + phase) * 8
        pos, shift = start // 16, start % 16
        lo = words[:, pos].astype(np.uint32)
        hi = words[:, (pos + 1) % 64].astype(np.uint32)
        state = (((lo << 16) | hi) >> (16 - shift).astype(np.uint32)) & np.uint32(65535)
        code = ((state * (state + np.uint32(1))) >> 6) & np.uint32(1023)
        low |= code << np.uint32(10 * phase)
        if phase == 3:
            high = (code >> 2).astype(np.uint8)
    packed = np.empty((len(words), 160), dtype=np.uint8)
    packed[:, :128] = low.view(np.uint8).reshape(len(words), 128)
    packed[:, 128:] = high
    return hashlib.sha256(packed).hexdigest()


def _warm(layer):
    m, n = layer.shape
    for batch in ROWS:
        key = (layer.trellis.device, m, n, batch, layer.cuda_index40_key)
        if key in _WARMED:
            continue
        source = torch.ones(
            (batch, n), dtype=torch.float32, device=layer.trellis.device
        )
        half = source.half()
        safe = torch.ones(batch, dtype=torch.int32, device=source.device)
        linear(
            source,
            half,
            safe,
            layer.trellis,
            layer.tlut,
            layer.mma_lut,
            layer.index40_codes,
            m,
            n,
            4,
            layer.cuda_index40_key,
        )
        torch.cuda.synchronize(source.device)
        _WARMED.add(key)


def prepare_layer(layer, descriptor, components, plan, key):
    if mode() == "none":
        if plan is not None:
            raise ValueError("Index40 plan conflicts with disabled policy")
        return
    if not eligible_descriptor(descriptor):
        return
    if plan is None or plan.budget != budget_bytes():
        raise ValueError("Index40 requires complete whole-model preflight")
    expected = plan.require(key, descriptor)
    if (
        layer.dtype != torch.bfloat16
        or layer.trellis.device.type != "cuda"
        or torch.cuda.get_device_capability(layer.trellis.device) != (8, 6)
        or not layer.native_fp16_gemm
        or layer.cuda_mma_key is None
        or not hasattr(layer, "mma_lut")
        or layer.gemm_precision != "native-fp16-mma"
    ):
        raise ValueError(
            "Opted-in index40 profile requires BF16/SM86/original exact-Half MMA"
        )
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Prepare indexed buffers/library before capture")
    if _sha(original.__file__) != ORIGINAL_MMA_SHA or _sha(SOURCE) != CUDA_SOURCE_SHA:
        raise ValueError(
            "Index40 original launch/repair/CUDA source differs from proof"
        )
    trellis_cpu, lut_cpu = components["trellis"], components["tlut"]
    if (
        trellis_cpu.device.type != "cpu"
        or lut_cpu.device.type != "cpu"
        or not bool(torch.isfinite(lut_cpu).all())
        or not torch.equal(lut_cpu, lut_cpu.half().bfloat16())
    ):
        raise ValueError(
            "Index40 needs original CPU components and finite exact-Half LUT"
        )
    with torch.cuda.device(layer.trellis.device):
        layer.cuda_index40_key = load()
    codes = pack(layer.trellis, *layer.shape, 4)
    if codes.numel() * codes.element_size() != expected["bytes"]:
        raise ValueError("Index40 GPU allocation differs from complete plan")
    layer.register_buffer("index40_codes", codes, persistent=False)
    signed = lut_cpu.half().repeat(2, 1)
    signed[512:, 0].neg_()
    expected_sha = expected_code_sha(trellis_cpu)
    actual = _tensor_identity(codes)
    if (
        actual["sha256"] != expected_sha
        or actual["bytes"] != expected["bytes"]
        or actual["shape"] != [layer.shape[0] * layer.shape[1] // 256, 40]
        or actual["dtype"] != "torch.int32"
    ):
        raise ValueError("Actual GPU index40 codes differ from independent CPU packing")
    identity = {
        **actual,
        "CPU_expected_code_sha256": expected_sha,
        "GPU_code_hash_verified_at_load": True,
        "original_trellis": _tensor_identity(trellis_cpu),
        "original_tlut": _tensor_identity(lut_cpu),
        "signed_half_lut": _tensor_identity(signed),
        "descriptor_sha256": expected["descriptor_sha256"],
        "library": build(),
        "original_trellis_pointer": layer.trellis.data_ptr(),
        "original_tlut_pointer": layer.tlut.data_ptr(),
    }
    layer.index40_identity = identity
    layer.index40_device = layer.trellis.device
    plan.finish(key, identity)
    _warm(layer)


def selected(x, layer):
    if (
        getattr(layer, "index40_codes", None) is None
        or layer.dtype != torch.bfloat16
        or x.dtype != torch.bfloat16
        or x.device.type != "cuda"
        or x.device != layer.index40_device
        or layer.codec != "qtip"
        or layer.bits != 4
        or tuple(layer.shape) not in SHAPES
        or not layer.permutation
        or x.numel() // x.shape[-1] not in ROWS
    ):
        return False
    layer.index40_plan.ensure_complete()
    return True


def runtime_marker(*, parallel=(1, 1, 1), activation_dtype=None, compilation=None):
    selected_mode = mode()
    if selected_mode != "none":
        mode_value = (
            compilation.get("mode")
            if isinstance(compilation, dict)
            else getattr(compilation, "mode", None)
        )
        if (
            tuple(parallel) != (1, 1, 1)
            or activation_dtype != "torch.bfloat16"
            or mode_value != 0
        ):
            raise ValueError(
                "Index40 requires single-rank BF16 explicit compilation mode0"
            )
        if (
            _sha(original.__file__) != ORIGINAL_MMA_SHA
            or _sha(SOURCE) != CUDA_SOURCE_SHA
        ):
            raise ValueError("Qualified index40 CUDA/original native source changed")
    files = [Path(__file__).resolve(), SOURCE, Path(original.__file__).resolve()]
    files += [
        Path(__file__).parents[1] / "plugin.py",
        Path(__file__).parents[1] / "__init__.py",
    ]
    files += [
        Path(__file__).with_name(name)
        for name in ("yaqa.py", "yaqa_kernels.py", "yaqa_linear.py", "config.py")
    ]
    return {
        "contract": "lossless-QTIP4-index40-complete-cache-v1",
        "mode": selected_mode,
        "budget_bytes": budget_bytes(),
        "SM": 86,
        "activation_dtype": "bfloat16",
        "activation_rows": list(ROWS),
        "shapes": [list(s) for s in SHAPES],
        "conv6144_selected": False,
        "bytes_per_tile": 160,
        "bits_per_weight": 5,
        "sources": {str(p.resolve()): _sha(p) for p in files},
        "library": build() if selected_mode != "none" else None,
        "original_guard_BF16_admission_and_F32_unsafe_repair_retained": True,
        "whole_profile_required": True,
        "persistent_checkpoint_bytes_added": 0,
        "new_weight_matrix_expansion": False,
        "cache_is_nonpersistent_GPU_buffer": True,
        "GPU_hash_verification_default": False,
        "whole_model_performance_and_same8_EAR_pending": True,
    }


def model_status(model, verify=False):
    rows = []
    plans = {}
    for name, layer in model.named_modules():
        plan = getattr(layer, "index40_plan", None)
        if plan is None:
            continue
        plans[id(plan)] = plan
        if getattr(layer, "index40_codes", None) is None:
            continue
        identity = layer.index40_identity
        record = {
            "name": name,
            "shape": list(layer.shape),
            "bits": layer.bits,
            "key": layer.index40_cache_key,
            "nonpersistent": "index40_codes" in layer._non_persistent_buffers_set,
            "actual_bytes": layer.index40_codes.numel()
            * layer.index40_codes.element_size(),
            "original_trellis_retained": layer.trellis.data_ptr()
            == identity["original_trellis_pointer"],
            "original_tlut_retained": layer.tlut.data_ptr()
            == identity["original_tlut_pointer"],
            "identity": copy.deepcopy(identity),
            "GPU_hashes_verified": False,
        }
        if verify:
            for value, expected in (
                (layer.index40_codes, identity),
                (layer.trellis, identity["original_trellis"]),
                (layer.tlut, identity["original_tlut"]),
                (layer.mma_lut, identity["signed_half_lut"]),
            ):
                actual = _tensor_identity(value)
                if any(
                    actual[k] != expected[k]
                    for k in ("shape", "dtype", "bytes", "sha256")
                ):
                    raise ValueError(
                        "Actual indexed cache/original buffer differs from CPU identity"
                    )
            record["GPU_hashes_verified"] = True
        rows.append(record)
    if len(plans) > 1:
        raise ValueError("Index40 requires one model/rank cache plan")
    for plan in plans.values():
        plan.ensure_complete()
    return {
        "mode": mode(),
        "budget_bytes": budget_bytes(),
        "actual_cache_matrices": len(rows),
        "actual_cache_bytes": sum(r["actual_bytes"] for r in rows),
        "buffers": rows,
        "plans": [p.summary() for p in plans.values()],
        "GPU_hashes_verified": bool(rows and verify),
        "default_status_GPU_sync": False,
        "all_codes_GPU_verified_at_load": all(
            r["identity"].get("GPU_code_hash_verified_at_load") is True for r in rows
        ),
        "original_GGUF_buffers_retained": all(
            r["original_trellis_retained"] and r["original_tlut_retained"] for r in rows
        ),
        "generation_route_changed": bool(rows),
        "checkpoint_GGUF_bytes_added": 0,
    }
