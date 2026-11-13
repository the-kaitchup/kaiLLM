# SPDX-License-Identifier: Apache-2.0
"""Shared-budget canonical GGUF prefill cache with native-route priority.

Install as quantization/shared_body_prefill_cache.py in an isolated runtime.
The original packed buffers remain authoritative and resident. The optional
BF16 representation serves prefill, independent of checkpoint or batch IDs.
"""

import os

import torch
import triton

ENV = "VLLM_GGUF_BODY_PREFILL_CACHE"
BUDGET_ENV = "VLLM_GGUF_BODY_PREFILL_CACHE_TOTAL_BYTES"
MIN_ROWS = 128


def settings():
    mode = os.getenv(ENV, "none")
    if mode not in ("none", "shared-bf16"):
        raise ValueError(ENV + " must be none or shared-bf16")
    budget = int(os.getenv(BUDGET_ENV, "0")) if mode != "none" else 0
    if budget < 0 or (mode != "none" and budget == 0):
        raise ValueError("An enabled shared cache requires an explicit total budget")
    return {"mode": mode, "total_budget_bytes": budget}


@torch.library.custom_op("gguf_yaqa::shared_body_prefill_mm", mutates_args=())
def cached_mm(x: torch.Tensor, weight: torch.Tensor, fp32_output: bool) -> torch.Tensor:
    from .yaqa_prefill_cache import _cached_gemm

    if (
        not x.is_cuda
        or torch.cuda.get_device_capability(x.device) not in ((8, 6), (12, 0))
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or weight.device != x.device
        or weight.ndim != 2
        or not weight.is_contiguous()
        or x.shape[-1] != weight.shape[1]
        or x.numel() // x.shape[-1] < MIN_ROWS
    ):
        raise ValueError("Require supported CUDA BF16 prefill rows and cached weight")
    n, k = weight.shape
    source = x.reshape(-1, k).contiguous()
    rows = source.shape[0]
    dtype = torch.float32 if fp32_output else torch.bfloat16
    result = torch.empty((rows, n), device=x.device, dtype=dtype)
    bn = (
        128
        if rows == 2048
        and (n, k) in ((7680, 2048), (2048, 7680), (6144, 2048), (2048, 2048))
        else 64
    )
    _cached_gemm[(triton.cdiv(rows, 64), triton.cdiv(n, bn))](
        source,
        weight,
        result,
        M=n,
        N=k,
        BATCH=rows,
        BM=64,
        BN=bn,
        BK=64,
        num_warps=4,
        num_stages=3,
    )
    return result.reshape(*x.shape[:-1], n)


@cached_mm.register_fake
def _fake_cached_mm(x, weight, fp32_output):
    return x.new_empty(
        (*x.shape[:-1], weight.shape[0]),
        dtype=torch.float32 if fp32_output else torch.bfloat16,
    )


def selected(x, layer):
    if not (
        getattr(layer, "body_prefill_weight", None) is not None
        and x.is_cuda
        and torch.cuda.get_device_capability(x.device) in ((8, 6), (12, 0))
        and x.dtype == torch.bfloat16
        and x.numel() // x.shape[-1] >= MIN_ROWS
    ):
        return False
    from .prefill_context import context_is_prefill

    return context_is_prefill(layer.body_prefill_decode_query_limit)


def apply(x, layer, bias=None, *, output_dtype=None):
    dtype = layer.dtype if output_dtype is None else output_dtype
    if dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Cached BF16 body output must be BF16 or F32")
    result = cached_mm(x, layer.body_prefill_weight, dtype == torch.float32)
    return result if bias is None else result + bias


def _body_candidates(model):
    from .compact import PackedGGUFLinear
    from .vocal_embeds import GGUFEmbeddingMethod

    # Vocabulary roles are semantic quantization methods, not name/shape rules.
    excluded = set()
    for module in model.modules():
        if isinstance(getattr(module, "quant_method", None), GGUFEmbeddingMethod):
            excluded.update(id(child) for child in module.modules())
    return [
        (name, layer)
        for name, layer in model.named_modules()
        if isinstance(layer, PackedGGUFLinear)
        and id(layer) not in excluded
        and layer.is_quantized
        and layer.canonical_rounding
        and layer.dtype == torch.bfloat16
        and layer.weight.device.type == "cuda"
    ]


def _fused_prefill_parts(model):
    """Ordinary GGUF matrices whose whole projection group the fused MMQ path runs."""
    from .compact import PackedGGUFLinear
    from .native_fused import prepare as prepare_fused

    covered = set()
    for module in model.modules():
        parts = getattr(module, "yaqa_parts", None) or getattr(module, "gguf_compact", None)
        if not parts or not all(type(part) is PackedGGUFLinear for part in parts):
            continue
        fused = getattr(module, "_kai_fused", None)
        if fused is None:
            fused = prepare_fused(list(parts)) or False
            module._kai_fused = fused
        if fused and fused._mmq:
            covered.update(id(part) for part in parts)
    return covered


def prepare_model(model, vllm_config=None):
    """Allocate once after weight loading, within QA+ordinary total budget."""
    from .yaqa_prefill_cache import tensor_identity

    policy = settings()
    if policy["mode"] == "none":
        return {**policy, "ordinary_cached_bytes": 0, "selected": []}
    qa_buffers = {
        value.data_ptr(): value
        for layer in model.modules()
        if (value := getattr(layer, "prefill_physical_weight", None)) is not None
    }
    reserved = sum(v.numel() * v.element_size() for v in qa_buffers.values())
    if reserved > policy["total_budget_bytes"]:
        raise ValueError("Existing YAQA cache exceeds the combined prefill budget")
    remaining = policy["total_budget_bytes"] - reserved
    from .prefill_context import decode_query_limit

    query_limit = decode_query_limit(vllm_config)
    candidates = _body_candidates(model)
    # Geometry supplies the cost estimate. Names only give deterministic ties.
    candidates.sort(key=lambda row: (-2 * row[1].shape[0] * row[1].shape[1], row[0]))
    covered = _fused_prefill_parts(model)
    records, skipped = [], []
    for name, layer in candidates:
        if id(layer) in covered:
            skipped.append(
                {"name": name, "bytes": 2 * layer.shape[0] * layer.shape[1],
                 "reason": "fused MMQ projection group serves prefill from the GGUF bytes"}
            )
            continue
        capability = torch.cuda.get_device_capability(layer.weight.device)
        if capability == (12, 0) and getattr(layer, "_native_sm120_body", None) is not None:
            skipped.append(
                {"name": name, "bytes": 2 * layer.shape[0] * layer.shape[1],
                 "reason": "existing native body adapter retains prefill priority"}
            )
            continue
        if capability not in ((8, 6), (12, 0)):
            skipped.append(
                {
                    "name": name,
                    "bytes": 2 * layer.shape[0] * layer.shape[1],
                    "reason": "ordinary cache target not component-qualified",
                }
            )
            continue
        if getattr(layer, "body_prefill_weight", None) is not None:
            raise ValueError("Body cache is prepared only once per model load")
        size = 2 * layer.shape[0] * layer.shape[1]
        if size > remaining:
            skipped.append({"name": name, "bytes": size, "reason": "total budget"})
            continue
        original = tensor_identity(layer.weight)
        pointer = layer.weight.data_ptr()
        with torch.inference_mode():
            rows = torch.arange(layer.shape[0], device=layer.weight.device)
            physical = layer.embedding(rows)
        if physical.shape != layer.shape or physical.dtype != torch.bfloat16:
            raise ValueError("Canonical GGUF body reconstruction shape/dtype differs")
        if not bool(torch.isfinite(physical).all()):
            raise ValueError("Canonical GGUF body cache contains nonfinite weights")
        if (
            layer.weight.data_ptr() != pointer
            or tensor_identity(layer.weight) != original
        ):
            raise ValueError("Canonical reconstruction changed original GGUF storage")
        layer.register_buffer(
            "body_prefill_weight", physical.contiguous(), persistent=False
        )
        layer.body_prefill_decode_query_limit = query_limit
        layer.body_prefill_identity = {
            "original": original,
            "cache": tensor_identity(physical),
            "qtype": layer.weight_type,
            "canonical_rounding": True,
        }
        records.append({"name": name, "bytes": size, **layer.body_prefill_identity})
        remaining -= size
    summary = {
        **policy,
        "QA_reserved_bytes": reserved,
        "decode_query_limit": query_limit,
        "ordinary_cached_bytes": sum(row["bytes"] for row in records),
        "unused_budget_bytes": remaining,
        "selected": records,
        "skipped": skipped,
        "original_wire_retained": True,
        "checkpoint_or_logical_batch_rules": False,
        "ordinary_precision": "canonical BF16 decoded GGUF weights",
    }
    model.gguf_shared_body_prefill_cache = summary
    return summary


def verify_model(model):
    from .yaqa_prefill_cache import tensor_identity

    count = 0
    for layer in model.modules():
        identity = getattr(layer, "body_prefill_identity", None)
        if identity is None:
            continue
        if tensor_identity(layer.weight) != identity["original"]:
            raise ValueError("Original GGUF storage changed after cache preparation")
        if tensor_identity(layer.body_prefill_weight) != identity["cache"]:
            raise ValueError("Resident canonical body cache changed")
        count += 1
    return {"verified_matrices": count, "original_and_cached_GPU_hashes_verified": True}


def runtime_marker():
    import hashlib
    from pathlib import Path

    path = Path(__file__).resolve()
    files = [
        path,
        path.with_name("compact.py"),
        path.with_name("prefill_context.py"),
        path.with_name("yaqa_prefill_cache.py"),
        path.parents[1] / "loader.py",
    ]
    return {
        "contract": "source-bound-shared-ordinary-canonical-BF16-prefill-v1",
        **settings(),
        "component_qualified_compute_capabilities": [[8, 6]],
        "candidate_compute_capabilities": [[12, 0]],
        "SM120_policy": "cache only canonical body matrices without a native body adapter",
        "other_target_fallback": "original native or compact",
        "sources": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
    }
