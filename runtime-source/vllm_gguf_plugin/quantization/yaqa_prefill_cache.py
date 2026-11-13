# SPDX-License-Identifier: Apache-2.0
"""Opt-in whole-profile physical-BF16 cache for512/2048 prefills only.

Original compressed buffers remain resident. Cache preparation is load-time
source code, not an installed worker/constructor/forward override. RN chunk
kernel arithmetic is identical to the validated component candidate.
"""

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl

LONG_K_ENV = "VLLM_GGUF_YAQA_SM120_LONG_K_DECODE"
MODE_ENV = "VLLM_GGUF_YAQA_PREFILL_CACHE"
BUDGET_ENV = "VLLM_GGUF_YAQA_PREFILL_CACHE_BYTES"
SELECTED_ROWS = (512, 2048)
ROWS_ENV = "VLLM_GGUF_YAQA_PREFILL_CACHE_ROWS"
GENERIC_MIN_ROWS = 128
SM120_PREFILL_MODE = "physical-bf16-rn-sm120"
SM120_SMALL_MODE = "physical-bf16-rn-sm120-small-decode"
MODES = ("none", "physical-bf16-rn", SM120_PREFILL_MODE, SM120_SMALL_MODE)


def long_k_small_decode_enabled():
    value = os.getenv(LONG_K_ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{LONG_K_ENV} requires an explicit0/1 flag")
    if value == "1" and os.getenv("VLLM_GGUF_YAQA_OPAQUE", "0") != "1":
        raise ValueError("Long-K decode requires opaque runtime row selection")
    return value == "1"


def long_k_small_decode_geometry(rows, shape, dtype, bits):
    return (
        type(rows) is int
        and 2 <= rows <= 8
        and len(shape) == 2
        and shape[0] >= 2048
        and shape[1] > shape[0]
        and all(value % 16 == 0 for value in shape)
        and dtype == torch.bfloat16
        and 2 <= bits <= 4
    )


def qkv_merge_enabled():
    value = os.getenv("VLLM_GGUF_YAQA_QKV_CACHE_MERGE", "0")
    if value not in ("0", "1"):
        raise ValueError("QKV cache merge requires an explicit 0/1 flag")
    return value == "1"


def b2_prefill_enabled():
    value = os.getenv("VLLM_GGUF_NATIVE_SM120_B2", "0")
    if value not in ("0", "1"):
        raise ValueError("VLLM_GGUF_NATIVE_SM120_B2 must be0or1")
    return value == "1" and mode() in (SM120_PREFILL_MODE, SM120_SMALL_MODE)


def mode():
    value = os.getenv(MODE_ENV, "none")
    if value not in MODES:
        raise ValueError(f"{MODE_ENV} must be one of{MODES}")
    return value


def rows_mode():
    value = os.getenv(ROWS_ENV, "legacy")
    if value not in ("legacy", "generic-prefill"):
        raise ValueError(f"{ROWS_ENV} must be legacy or generic-prefill")
    return value


def row_supported(rows, policy=None):
    policy = rows_mode() if policy is None else policy
    if policy == "legacy":
        return rows in SELECTED_ROWS
    if policy == "generic-prefill":
        return rows >= GENERIC_MIN_ROWS
    raise ValueError("Unknown physical-cache row policy")


def budget_bytes():
    try:
        value = int(os.getenv(BUDGET_ENV, "0"))
    except ValueError as error:
        raise ValueError(
            f"{BUDGET_ENV} requires an explicit integer byte budget"
        ) from error
    if value < 0 or (mode() != "none" and value == 0):
        raise ValueError(
            "Enabled prefill cache requires a positive explicit byte budget"
        )
    return value


def descriptor_sha(descriptor):
    return hashlib.sha256(
        json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def tensor_identity(value):
    value = value.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


@lru_cache(maxsize=8)
def _is_sm86(device):
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == (8, 6)


@lru_cache(maxsize=8)
def _is_sm120(device):
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == (12, 0)


def _supports_device(device, captured_mode):
    if captured_mode in (SM120_PREFILL_MODE, SM120_SMALL_MODE):
        return _is_sm120(device)
    return _is_sm86(device) or _is_sm120(device)


@triton.jit(do_not_specialize=["BATCH"], do_not_specialize_on_alignment=["BATCH"])
def _cached_gemm(
    X,
    W,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    BATCH,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    batch = tl.program_id(0) * BM + tl.arange(0, BM)
    rows = tl.program_id(1) * BN + tl.arange(0, BN)
    columns = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(N, BK)):
        c = start * BK + columns
        x = tl.load(
            X + batch[:, None] * N + c[None, :],
            (batch[:, None] < BATCH) & (c[None, :] < N),
            0,
        ).to(tl.bfloat16)
        w = tl.load(
            W + rows[None, :] * N + c[:, None],
            (rows[None, :] < M) & (c[:, None] < N),
            0,
        ).to(tl.bfloat16)
        partial = tl.dot(x, w)
        acc = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=(acc, partial),
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
    tl.store(
        Y + batch[:, None] * M + rows[None, :],
        acc,
        (batch[:, None] < BATCH) & (rows[None, :] < M),
    )


@triton.jit
def _cached_gemv(
    X, W, Y, M: tl.constexpr, N: tl.constexpr, R: tl.constexpr, K: tl.constexpr
):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    cols = tl.arange(0, K)
    x = tl.load(X + cols, cols < N, 0).to(tl.float32)
    weight = tl.load(
        W + rows[:, None] * N + cols[None, :],
        (rows[:, None] < M) & (cols[None, :] < N),
        0,
    ).to(tl.float32)
    output = tl.sum(weight * x[None, :], 1)
    tl.store(Y + rows, output, rows < M)


@triton.jit
def _cached_m4(
    X, W, Y, M: tl.constexpr, N: tl.constexpr, R: tl.constexpr, K: tl.constexpr
):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    columns = tl.arange(0, K)
    weight = tl.load(
        W + rows[:, None] * N + columns[None, :],
        (rows[:, None] < M) & (columns[None, :] < N),
        0,
    ).to(tl.float32)
    for token in tl.static_range(4):
        x = tl.load(X + token * N + columns, columns < N, 0).to(tl.float32)
        result = tl.sum(weight * x[None, :], 1)
        tl.store(Y + token * M + rows, result, rows < M)


@torch.library.custom_op("gguf_yaqa::physical_prefill_cache_mm", mutates_args=())
def physical_prefill_mm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if (
        not x.is_cuda
        or not _is_sm86(x.device)
        or x.dtype not in (torch.float32, torch.bfloat16)
        or weight.dtype != torch.bfloat16
        or weight.ndim != 2
        or not weight.is_contiguous()
        or x.device != weight.device
        or x.shape[-1] != weight.shape[1]
        or max(weight.shape) > 7680
        or min(weight.shape) < 16
        or any(n % 16 for n in weight.shape)
        or not row_supported(x.numel() // x.shape[-1])
    ):
        raise ValueError(
            "Physical cache requires SM86, supported prefill rows and BF16 weights"
        )
    m, n = weight.shape
    source = x.reshape(-1, n).contiguous()
    output = torch.empty((source.shape[0], m), dtype=x.dtype, device=x.device)
    bn = (
        128
        if source.dtype == torch.bfloat16
        and source.shape[0] == 2048
        and (m, n) in ((7680, 2048), (2048, 7680), (6144, 2048), (2048, 2048))
        else 64
    )
    _cached_gemm[(triton.cdiv(source.shape[0], 64), triton.cdiv(m, bn))](
        source,
        weight,
        output,
        M=m,
        N=n,
        BATCH=source.shape[0],
        BM=64,
        BN=bn,
        BK=64,
        num_warps=4,
        num_stages=3,
    )
    return output.reshape(*x.shape[:-1], m)


@physical_prefill_mm.register_fake
def _fake(x, weight):
    return x.new_empty((*x.shape[:-1], weight.shape[0]), dtype=x.dtype)


@torch.library.custom_op("gguf_yaqa::physical_sm120_cache_mm", mutates_args=())
def physical_sm120_cache_mm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if (
        x.device.type != "cuda"
        or not _is_sm120(x.device)
        or x.dtype not in (torch.float32, torch.bfloat16)
        or weight.dtype != torch.bfloat16
        or weight.ndim != 2
        or not weight.is_contiguous()
        or x.device != weight.device
        or x.shape[-1] != weight.shape[1]
        or min(weight.shape) <= 0
        or (
            rows_mode() == "legacy"
            and (
                max(weight.shape) > 7680
                or min(weight.shape) < 16
                or any(n % 16 for n in weight.shape)
            )
        )
        or not (
            x.numel() // x.shape[-1] in (1, 4)
            or row_supported(x.numel() // x.shape[-1])
            or (b2_prefill_enabled() and x.numel() // x.shape[-1] == 1024)
        )
        or (x.numel() // x.shape[-1] in (1, 4) and max(weight.shape) > 2048)
    ):
        raise ValueError(
            "SM120 cache requires F32/BF16 activations and qualified shape/rows"
        )
    m, n = weight.shape
    source = x.reshape(-1, n).contiguous()
    result = torch.empty((source.shape[0], m), dtype=x.dtype, device=x.device)
    if source.shape[0] == 1:
        _cached_gemv[(triton.cdiv(m, 4),)](
            source,
            weight,
            result,
            M=m,
            N=n,
            R=4,
            K=triton.next_power_of_2(n),
            num_warps=4,
        )
    elif source.shape[0] == 4:
        rows_per_program = 1 if m == 16 else 4
        _cached_m4[(triton.cdiv(m, rows_per_program),)](
            source,
            weight,
            result,
            M=m,
            N=n,
            R=rows_per_program,
            K=triton.next_power_of_2(n),
            num_warps=4,
            enable_fp_fusion=False,
        )
    elif source.shape[0] > 0:
        _cached_gemm[(triton.cdiv(source.shape[0], 64), triton.cdiv(m, 64))](
            source,
            weight,
            result,
            M=m,
            N=n,
            BATCH=source.shape[0],
            BM=64,
            BN=64,
            BK=64,
            num_warps=4,
            num_stages=3,
        )
    return result.reshape(*x.shape[:-1], m)


@physical_sm120_cache_mm.register_fake
def _fake_sm120(x, weight):
    return x.new_empty((*x.shape[:-1], weight.shape[0]), dtype=x.dtype)


@torch.library.custom_op("gguf_yaqa::physical_sm120_long_k_decode_mm", mutates_args=())
def physical_sm120_long_k_decode_mm(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    rows = x.numel() // x.shape[-1]
    if (
        not x.is_cuda
        or not _is_sm120(x.device)
        or not long_k_small_decode_geometry(rows, weight.shape, x.dtype, 2)
        or weight.dtype != torch.bfloat16
        or weight.ndim != 2
        or not weight.is_contiguous()
        or x.device != weight.device
        or x.shape[-1] != weight.shape[1]
    ):
        raise ValueError("Require SM120/BF16 aligned long-K weight and2..8rows")
    n, k = weight.shape
    source = x.reshape(rows, k).contiguous()
    output = torch.empty((rows, n), device=x.device, dtype=x.dtype)
    _cached_gemm[(triton.cdiv(rows, 16), triton.cdiv(n, 16))](
        source,
        weight,
        output,
        M=n,
        N=k,
        BATCH=rows,
        BM=16,
        BN=16,
        BK=64,
        num_warps=4,
        num_stages=3,
    )
    return output.reshape(*x.shape[:-1], n)


@physical_sm120_long_k_decode_mm.register_fake
def _fake_sm120_long_k(x, weight):
    return x.new_empty((*x.shape[:-1], weight.shape[0]), dtype=x.dtype)


class CachePlan:
    """One model/rank's complete descriptor and byte contract; no global cache."""

    def __init__(self, entries, budget):
        self.entries, self.budget = entries, budget
        self.mode = mode()
        self.sm120_long_k_small_decode = long_k_small_decode_enabled()
        from .physical_cache_splitk import mode as split_mode

        self.sm120_long_k_splitk = split_mode()
        if self.sm120_long_k_splitk != "none" and not self.sm120_long_k_small_decode:
            raise ValueError("Split-K requires the canonical long-K cache policy")
        if self.sm120_long_k_small_decode and self.mode != SM120_SMALL_MODE:
            raise ValueError("Long-K decode requires complete SM120 canonical cache")
        from .resident_small_decode import enabled

        self.resident_small_decode = enabled()
        self.qkv_merge = qkv_merge_enabled()
        self.native_b2_prefill = b2_prefill_enabled()
        self.qkv_joined = []
        if self.qkv_merge and self.mode != SM120_SMALL_MODE:
            raise ValueError("QKV merge needs the qualified SM120 small-cache mode")
        self.row_policy = rows_mode()
        self.decode_query_limit = None
        from .yaqa_prefill_cache_out import enabled

        self.merged_out = enabled()
        self.expected_bytes = sum(row["bytes"] for row in entries.values())
        if self.expected_bytes > budget:
            raise ValueError(
                f"Whole prefill cache needs{self.expected_bytes}bytes, "
                f"budget{budget}; no partial cache"
            )
        self.loaded = {}

    def require(self, key, descriptor):
        expected = self.entries.get(key)
        if expected is None or expected["descriptor_sha256"] != descriptor_sha(
            descriptor
        ):
            raise ValueError("Packed layer differs from whole-profile cache plan")
        if key in self.loaded:
            raise ValueError("A cache matrix was constructed more than once")
        return expected

    def finish(self, key, identity):
        if identity["bytes"] != self.entries[key]["bytes"]:
            raise ValueError("Actual physical cache bytes differ from complete plan")
        self.loaded[key] = identity

    def ensure_complete(self):
        if (
            self.loaded.keys() != self.entries.keys()
            or sum(x["bytes"] for x in self.loaded.values()) != self.expected_bytes
        ):
            raise ValueError(
                "Incomplete physical prefill cache; inference is not allowed"
            )

    def summary(self):
        return {
            "mode": self.mode,
            "resident_small_decode": self.resident_small_decode,
            "SM120_long_K_small_decode": self.sm120_long_k_small_decode,
            "SM120_long_K_splitK": self.sm120_long_k_splitk,
            "row_policy": self.row_policy,
            "qkv_cache_merge": self.qkv_merge,
            "qkv_joined": list(self.qkv_joined),
            "B2_prefill_1024_admission_enabled": self.native_b2_prefill,
            "activation_row_policy": self.row_policy,
            "decode_query_limit": self.decode_query_limit,
            "generic_minimum_rows": GENERIC_MIN_ROWS
            if self.row_policy == "generic-prefill"
            else None,
            "budget_bytes": self.budget,
            "merged_prefill_out": self.merged_out,
            "expected_matrices": len(self.entries),
            "expected_bytes": self.expected_bytes,
            "loaded_matrices": len(self.loaded),
            "loaded_bytes": sum(x["bytes"] for x in self.loaded.values()),
            "complete": self.loaded.keys() == self.entries.keys(),
            "entries": self.entries,
            "loaded": self.loaded,
            "canonical_CPU_reuse": (
                self._canonical_cpu_session.summary()
                if getattr(self, "_canonical_cpu_session", None) is not None
                else {"mode": "none", "extra_GPU_bytes": 0}
            ),
        }


def create_plan(sources):
    if mode() == "none":
        return None
    budget = budget_bytes()
    entries = {}
    for key, (path, logical, descriptor) in sources.items():
        if descriptor is None:
            continue
        if descriptor["codec"] != "qtip":
            continue  # Other original YAQA codecs retain their original routes.
        shape = descriptor["shape"]
        if (
            len(shape) != 2
            or min(shape) < 16
            or max(shape) > 7680
            or any(n % 16 for n in shape)
        ):
            raise ValueError(
                "Whole-profile cache needs bounded QTIP; no partial selection"
            )
        entries[key] = {
            "source": str(path),
            "logical_name": logical,
            "shape": list(shape),
            "bytes": 2 * shape[0] * shape[1],
            "descriptor_sha256": descriptor_sha(descriptor),
        }
    return CachePlan(entries, budget) if entries else None


def rebind_plan(plan, sources):
    """Apply the existing HF→vLLM rename before any physical allocation."""
    if plan is None:
        return None
    if plan.loaded:
        raise ValueError("Cannot remap a physical cache after allocation has started")
    rebound = create_plan(sources)
    if rebound is None:
        raise ValueError("Mapper discarded the complete physical cache selection")

    def records(value):
        return sorted(
            (
                row["source"],
                row["logical_name"],
                tuple(row["shape"]),
                row["bytes"],
                row["descriptor_sha256"],
            )
            for row in value.entries.values()
        )

    if (
        rebound.merged_out != plan.merged_out
        or rebound.mode != plan.mode
        or rebound.qkv_merge != plan.qkv_merge
        or rebound.sm120_long_k_small_decode != plan.sm120_long_k_small_decode
        or rebound.budget != plan.budget
        or rebound.expected_bytes != plan.expected_bytes
        or records(rebound) != records(plan)
    ):
        raise ValueError(
            "Mapper changed physical-cache sources/bytes; no partial cache"
        )
    return rebound


def prepare_layer(layer, descriptor, components, plan, key):
    if descriptor["codec"] != "qtip":
        return
    if mode() == "none":
        if plan is not None:
            raise ValueError("Cache plan conflicts with disabled process policy")
        return
    if plan is None or budget_bytes() != plan.budget or mode() != plan.mode:
        raise ValueError("Enabled cache requires its authenticated whole-profile plan")
    from .physical_cache_splitk import mode as split_mode
    from .resident_small_decode import enabled

    if split_mode() != plan.sm120_long_k_splitk:
        raise ValueError("Split-K cache policy changed after complete planning")
    if long_k_small_decode_enabled() != plan.sm120_long_k_small_decode:
        raise ValueError("Long-K cache policy changed after complete planning")
    if enabled() != plan.resident_small_decode:
        raise ValueError("Small-cache policy changed after whole-cache planning")
    expected = plan.require(key, descriptor)
    if plan.decode_query_limit is None:
        from .prefill_context import decode_query_limit

        plan.decode_query_limit = decode_query_limit()
    if layer.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Physical prefill cache supports F32/BF16 activations only")
    target = next(layer.buffers()).device
    if not _supports_device(target, plan.mode):
        raise ValueError("Physical cache mode does not admit the actual GPU")
    from .yaqa import reconstruct_yaqa_reference

    # Only exact original components enter canonical reconstruction. The layer's
    # auxiliary folded scales and any factor-storage convenience buffers do not.
    host = {name: value.detach().cpu() for name, value in components.items()}
    from .yaqa_prefill_cpu_cache import canonical_weight

    with canonical_weight(
        plan, key, descriptor, host, reconstruct_yaqa_reference, tensor_identity, torch
    ) as weight:
        session = getattr(plan, "_canonical_cpu_session", None)
        if (
            session is not None
            and session.hit
            and not session.closed
            and session.views_open > 0
        ):
            # The complete-store check hashed these BF16 bytes, and lookup
            # already matched the current original components for this key.
            # Published records passed the cold path's finite check below.
            if tuple(weight.shape) != tuple(expected["shape"]):
                raise ValueError(
                    "Canonical physical cache has invalid shape/nonfinite weights"
                )
            from copy import deepcopy

            identity = deepcopy(session.records[key]["identity"])
        else:
            if tuple(weight.shape) != tuple(expected["shape"]) or not bool(
                torch.isfinite(weight).all()
            ):
                raise ValueError(
                    "Canonical physical cache has invalid shape/nonfinite weights"
                )
            identity = {
                **tensor_identity(weight),
                "descriptor_sha256": descriptor_sha(descriptor),
                "original_components": {
                    name: tensor_identity(value) for name, value in host.items()
                },
            }
        # The original packed/source buffers are retained; only this additional
        # physical matrix is nonpersistent and the full expansion is explicit.
        layer.register_buffer(
            "prefill_physical_weight", weight.to(target), persistent=False
        )
        layer.prefill_cache_identity = identity
        plan.finish(key, identity)
        # The existing synchronous CPU-to-device copy is complete; release
        # the caller's CPU view before the helper closes its mmap/lease.
        del weight
    if plan.resident_small_decode:
        from .resident_small_decode import warm

        warm(layer)


def selected(x, layer):
    plan = getattr(layer, "prefill_cache_plan", None)
    if (
        plan is None
        or x.device.type != "cuda"
        or x.dtype not in (torch.float32, torch.bfloat16)
    ):
        return False
    rows = x.numel() // x.shape[-1]
    if (
        getattr(plan, "sm120_long_k_small_decode", False)
        and plan.mode == SM120_SMALL_MODE
        and _is_sm120(x.device)
        and long_k_small_decode_geometry(rows, layer.shape, x.dtype, layer.bits)
    ):
        return True
    from .resident_small_decode import geometry

    if (
        plan.resident_small_decode
        and _is_sm86(x.device)
        and geometry(rows, layer.shape)
    ):
        return True
    if (
        plan.mode == SM120_SMALL_MODE
        and _is_sm120(x.device)
        and rows in (1, 4)
        and max(layer.shape) <= 2048
    ):
        return True
    capable = row_supported(rows, plan.row_policy) or (
        plan.native_b2_prefill
        and rows == 1024
        and plan.mode in (SM120_PREFILL_MODE, SM120_SMALL_MODE)
    )
    if not capable:
        return False
    if plan.row_policy == "legacy":
        return True
    from .prefill_context import context_is_prefill

    return context_is_prefill(plan.decode_query_limit)


def apply(x, layer):
    layer.prefill_cache_plan.ensure_complete()
    if (
        getattr(layer.prefill_cache_plan, "sm120_long_k_small_decode", False)
        and layer.prefill_cache_plan.mode == SM120_SMALL_MODE
        and _is_sm120(x.device)
        and long_k_small_decode_geometry(
            x.numel() // x.shape[-1], layer.shape, x.dtype, layer.bits
        )
    ):
        if layer.prefill_cache_plan.sm120_long_k_splitk == "auto":
            from .physical_cache_splitk import physical_sm120_long_k_split_mm

            return physical_sm120_long_k_split_mm(x, layer.prefill_physical_weight)
        return physical_sm120_long_k_decode_mm(x, layer.prefill_physical_weight)
    from .resident_small_decode import geometry

    if (
        layer.prefill_cache_plan.resident_small_decode
        and _is_sm86(x.device)
        and geometry(x.numel() // x.shape[-1], layer.shape)
    ):
        from .resident_small_decode_kernel import cached_linear

        k = layer.shape[1]
        result = cached_linear(
            x.reshape(-1, k).contiguous(), layer.prefill_physical_weight, False
        )
        return result.reshape(*x.shape[:-1], layer.shape[0])
    if _is_sm120(x.device):
        return physical_sm120_cache_mm(x, layer.prefill_physical_weight)
    return physical_prefill_mm(x, layer.prefill_physical_weight).to(x.dtype)


def layer_status(layer, verify=False):
    plan = getattr(layer, "prefill_cache_plan", None)
    if plan is None:
        return None
    value = layer.prefill_physical_weight
    result = {
        "key": layer.prefill_cache_key,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "actual_cache_bytes": value.numel() * value.element_size(),
        "SM120_long_K_small_decode_eligible": long_k_small_decode_geometry(
            4, layer.shape, torch.bfloat16, layer.bits
        ),
        "nonpersistent": "prefill_physical_weight" in layer._non_persistent_buffers_set,
        "canonical_CPU_identity": layer.prefill_cache_identity,
        "original_components_retained": all(
            hasattr(layer, name)
            for name in layer.prefill_cache_identity["original_components"]
        ),
        "GPU_hashes_verified": False,
    }
    if verify:
        actual = tensor_identity(value)
        if actual["sha256"] != layer.prefill_cache_identity["sha256"]:
            raise ValueError(
                "Actual physical cache bytes differ from canonical CPU reconstruction"
            )
        for name, identity in layer.prefill_cache_identity[
            "original_components"
        ].items():
            if tensor_identity(getattr(layer, name)) != identity:
                raise ValueError("An original packed/source component changed")
        result.update(GPU_hashes_verified=True, actual_identity=actual)
    return result


def runtime_marker(parallel=(1, 1, 1)):
    selected_mode, budget = mode(), budget_bytes()
    long_k_decode = long_k_small_decode_enabled()
    from .physical_cache_splitk import mode as split_mode

    split_policy = split_mode()
    if split_policy != "none" and not long_k_decode:
        raise ValueError("Split-K requires the canonical long-K cache policy")
    if long_k_decode and selected_mode != SM120_SMALL_MODE:
        raise ValueError("Long-K decode requires the complete SM120 cache mode")
    from .resident_small_decode import marker

    small_decode = marker()
    if small_decode["enabled"] and selected_mode == "none":
        raise ValueError("Small decode requires the complete physical weight cache")
    merge_qkv = qkv_merge_enabled()
    if merge_qkv and selected_mode != SM120_SMALL_MODE:
        raise ValueError("QKV merge needs qualified SM120 small-cache mode")
    if selected_mode != "none" and tuple(parallel) != (1, 1, 1):
        raise ValueError(
            "Physical prefill cache requires a single TP/PP/DP worker/rank"
        )
    from .yaqa_prefill_cache_out import runtime_marker as merged_marker

    merged = merged_marker()
    from .yaqa_prefill_cpu_cache import runtime_marker as cpu_cache_marker

    cpu_cache = cpu_cache_marker()
    path = Path(__file__).resolve()
    files = [
        path,
        path.with_name("yaqa.py"),
        path.with_name("yaqa_linear.py"),
        path.with_name("config.py"),
        path.with_name("yaqa_prefill_cpu_cache.py"),
        path.with_name("prefill_context.py"),
        path.with_name("physical_cache_splitk.py"),
    ]
    if merge_qkv:
        files.append(path.with_name("yaqa_qkv_cache.py"))
    return {
        "resident_small_decode": small_decode,
        "SM120_long_K_splitK": {
            "mode": split_policy,
            "selection": "smallest power2<=8 reaching2CTAs/actualSM; <=4MiB partial",
            "new_F32_partial_grouping_requires_EAR": True,
            "additional_weight_bytes": 0,
        },
        "SM120_long_K_small_decode": {
            "enabled": long_k_decode,
            "rows": "actual2..8",
            "geometry": "N>=2048,K>N,16aligned,BF16,QTIPbits2..4",
            "tile": {"BM": 16, "BN": 16, "BK": 64, "warps": 4, "stages": 3},
            "existing_canonical_weights_only": True,
            "additional_weight_bytes": 0,
            "original_RN_gemm_JIT_unchanged": True,
        },
        "qkv_cache_merge": merge_qkv,
        "qkv_merge_rows": [1, 4] if merge_qkv else [],
        "small_decode_rows": [1, 4] if selected_mode == SM120_SMALL_MODE else [],
        "small_decode_max_dimension": 2048,
        "SM120_native_B2_prefill": b2_prefill_enabled(),
        "merged_out": merged,
        "canonical_CPU_reuse": cpu_cache,
        "single_square_BF16_2048_BN128": True,
        "contract": "whole-QTIP-physical-BF16-prefill-RN-typed-output-v2",
        "mode": selected_mode,
        "budget_bytes": budget,
        "activation_rows": list(SELECTED_ROWS) if rows_mode() == "legacy" else None,
        "activation_row_policy": rows_mode(),
        "generic_minimum_rows": GENERIC_MIN_ROWS
        if rows_mode() == "generic-prefill"
        else None,
        "decode_rows_up_to64_remain_compressed": not small_decode["enabled"]
        and selected_mode != SM120_SMALL_MODE,
        "generic_route_requires_prefill_semantics": True,
        "graph_key_semantic_field": "BatchDescriptor.is_prefill",
        "activation_dtypes": ["float32", "bfloat16"],
        "SM": 120 if selected_mode in (SM120_PREFILL_MODE, SM120_SMALL_MODE) else 86,
        "supported_target_SM": [86, 120],
        "parallel": list(parallel),
        "no_partial_cache": True,
        "extra_resident_weight_budget_bytes": budget if selected_mode != "none" else 0,
        "cache_buffers_persistent_in_state_dict": False,
        "GGUF_and_original_components_unchanged": True,
        "precision_change": selected_mode != "none",
        "whole_model_EAR_required": selected_mode != "none",
        "sources": {
            **small_decode["sources"],
            **{
                str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in files
            },
            **merged["sources"],
        },
    }


def model_status(model, verify=False):
    """Explicit untimed read-only attestation; GPU hashes only with verify=True."""
    rows, plans = [], {}
    for name, layer in model.named_modules():
        plan = getattr(layer, "prefill_cache_plan", None)
        if plan is None:
            continue
        plans[id(plan)] = plan
        rows.append({"name": name, **layer_status(layer, verify=verify)})
    if len(plans) > 1:
        raise ValueError("Physical cache attestation requires one whole-model plan")
    for plan in plans.values():
        plan.ensure_complete()
    return {
        "mode": mode(),
        "activation_row_policy": rows_mode(),
        "generic_minimum_rows": GENERIC_MIN_ROWS
        if rows_mode() == "generic-prefill"
        else None,
        "budget_bytes": budget_bytes(),
        "actual_cache_matrices": len(rows),
        "actual_cache_bytes": sum(row["actual_cache_bytes"] for row in rows),
        "buffers": rows,
        "plans": [plan.summary() for plan in plans.values()],
        "GPU_hashes_verified": bool(verify and rows),
        "default_status_GPU_sync": False,
        "SM120_long_K_small_decode_enabled": any(
            getattr(plan, "sm120_long_k_small_decode", False) for plan in plans.values()
        ),
        "generation_route_changed": any(
            plan.mode == SM120_SMALL_MODE for plan in plans.values()
        ),
        "small_decode_cache_matrices": sum(max(row["shape"]) <= 2048 for row in rows)
        if any(plan.mode == SM120_SMALL_MODE for plan in plans.values())
        else 0,
        "small_decode_cache_bytes": sum(
            row["actual_cache_bytes"] for row in rows if max(row["shape"]) <= 2048
        )
        if any(plan.mode == SM120_SMALL_MODE for plan in plans.values())
        else 0,
        "B2_prefill_1024_admission_enabled": any(
            plan.native_b2_prefill for plan in plans.values()
        ),
        "merged_prefill_out_enabled": any(plan.merged_out for plan in plans.values()),
        "merged_prefill_extra_weight_bytes": 0,
        "original_GGUF_buffers_retained": all(
            row["original_components_retained"] for row in rows
        ),
    }
