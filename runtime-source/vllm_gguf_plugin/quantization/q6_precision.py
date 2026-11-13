# SPDX-License-Identifier: Apache-2.0
"""Default-off generic Q6 body/head precision candidate with bounded caches."""

import hashlib
import importlib.metadata
import json
import os
from functools import lru_cache
from pathlib import Path

import torch

ENV = "VLLM_GGUF_Q6_PRECISION"
BUDGET_ENV = ENV + "_CACHE_BYTES"
LIBRARY_ENV = ENV + "_HEAD_LIBRARY"
LIBRARY_SHA = "e0a1715f216f1f7c1a9eab9b6119134bcac0e84437f7d76dc03e293814c9e7ec"
CUDA_SHA = "c9e1db12578e408e12cbc078611b831bece71ec0d9a0abbc56ad3d456252cdb5"
HEAD_LIBRARIES = {
    (8, 6): (
        LIBRARY_SHA,
        "af97331fa06355fbc60024830aa5a587a7b8c498fb63dc7927cdcdcfe7296265",
    ),
    (12, 0): (
        "bc31aa455e794a3c005d36ea648e365164eeaaf073198070d69640bde79dd6f5",
        "208ba27aace86aa2b6af2737fac9b5911fae07b9986d4e7190aa1c8f207b0a7d",
    ),
}
MAX_BUDGET = 1536 << 20
MAX_ROWS = 32
HUMMING_WORKSPACE_RESERVE = 65536


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@lru_cache(maxsize=16)
def _head_library_identity(path):
    """Authenticate once before preparation/capture, never per decoded token."""
    path = Path(path)
    metadata = path.with_suffix(".build.json")
    binary_sha = sha(path)
    targets = [
        target for target, pins in HEAD_LIBRARIES.items() if pins[0] == binary_sha
    ]
    if len(targets) != 1:
        raise ValueError("Unqualified canonical Q6 target library")
    target = targets[0]
    metadata_sha = sha(metadata)
    if metadata_sha != HEAD_LIBRARIES[target][1]:
        raise ValueError("Canonical Q6 target build metadata identity differs")
    build = json.loads(metadata.read_text())
    arch = f"{target[0]}{target[1]}"
    command = build.get("command", ())
    if (
        build.get("library_sha256") != binary_sha
        or build.get("returncode") != 0
        or CUDA_SHA not in build.get("source_sha256", {}).values()
        or sha(Path(__file__).with_name("canonical_q6_tile.cu")) != CUDA_SHA
        or command.count(f"arch=compute_{arch},code=sm_{arch}") != 1
        or command.count(f"arch=compute_{arch},code=compute_{arch}") != 1
        or any(
            flag not in command
            for flag in (
                "--fmad=false",
                "--ftz=false",
                "--prec-div=true",
                "--prec-sqrt=true",
            )
        )
        or "--use_fast_math" in command
    ):
        raise ValueError("Canonical Q6 binary/target/CUDA arithmetic identity differs")
    return {
        "path": str(path),
        "sha256": binary_sha,
        "target": target,
        "metadata_sha256": metadata_sha,
    }


def head_library_identity(path):
    return dict(_head_library_identity(str(Path(path).absolute())))


def settings():
    mode = os.getenv(ENV, "none")
    if mode not in ("none", "f16-body-canonical-head"):
        raise ValueError(ENV + " must be none or f16-body-canonical-head")
    if mode == "none":
        return {"mode": mode, "additional_cache_budget_bytes": 0}
    budget = int(os.getenv(BUDGET_ENV, "0"))
    if not 0 < budget <= MAX_BUDGET:
        raise ValueError("Explicit additional Q6 cache budget must be at most1.5GiB")
    path = Path(
        os.getenv(
            LIBRARY_ENV,
            str(Path(__file__).parents[1] / "native/libraries/libcanonical_q6_tile.so"),
        )
    ).absolute()
    return {
        "mode": mode,
        "additional_cache_budget_bytes": budget,
        "canonical_head_library": head_library_identity(path),
        "Humming_partition_policy": "stock-geometry-no-StreamK",
    }


def estimate_bytes(scope, n, k):
    if (
        scope not in ("body", "q6-head")
        or type(n) is not int
        or type(k) is not int
        or not 0 < n <= 2**31 - 1
        or not 0 < k <= 2**31 - 1
        or k % 256
    ):
        raise ValueError("Positive block-complete Q6 geometry required")
    if scope == "q6-head":
        return ((n + 15) // 16) * (k // 16) * 256 + 8
    # Stock Q6 uint6 codes + FP16 scales every16 weights; bounded StreamK locks.
    padded_n, padded_k = ((n + 255) // 256) * 256, ((k + 127) // 128) * 128
    return padded_n * padded_k * 7 // 8 + HUMMING_WORKSPACE_RESERVE + 8


def plan_catalog(catalog, budget):
    if type(budget) is not int or not 0 <= budget <= MAX_BUDGET:
        raise ValueError("Invalid additional Q6 representation budget")
    entries, seen = [], set()
    for row in catalog:
        if row["name"] in seen:
            raise ValueError("Duplicate Q6 representation plan name")
        seen.add(row["name"])
        if (
            row["qtype"] != 14
            or row["dtype"] != "torch.bfloat16"
            or tuple(row["capability"]) not in HEAD_LIBRARIES
            or not row["canonical_rounding"]
        ):
            continue
        scope = (
            "body"
            if row["role"] == "body"
            else (
                "q6-head"
                if row["is_output_head"] and row["role"] == "vocabulary"
                else None
            )
        )
        if scope is None:
            continue
        n, k = row["shape"]
        size = estimate_bytes(scope, n, k)
        entries.append(
            {
                "name": row["name"],
                "scope": scope,
                "qtype": 14,
                "shape": [n, k],
                "reserved_bytes": size,
            }
        )
    total = sum(row["reserved_bytes"] for row in entries)
    if total > budget:
        raise ValueError(
            f"Complete Q6 plan needs {total} additional bytes; budget{budget}"
        )
    return {
        "entries": entries,
        "reserved_additional_bytes": total,
        "explicit_budget_bytes": budget,
        "full_preflight": True,
        "partial_selection": False,
    }


def runtime_marker():
    from . import (
        canonical_q6_tile,
        compact_decode,
        humming,
        native_bf16_q8q6_power2,
        q6_precision_guard,
    )

    policy = settings()
    files = [
        Path(__file__),
        Path(canonical_q6_tile.__file__),
        canonical_q6_tile.CUDA_SOURCE,
        Path(q6_precision_guard.__file__),
        Path(compact_decode.__file__),
        Path(native_bf16_q8q6_power2.__file__),
        Path(humming.__file__),
    ]
    if policy["mode"] != "none":
        library = Path(policy["canonical_head_library"]["path"])
        metadata = library.with_suffix(".build.json")
        build = json.loads(metadata.read_text())
        if (
            sha(library) != policy["canonical_head_library"]["sha256"]
            or sha(metadata) != policy["canonical_head_library"]["metadata_sha256"]
            or build.get("library_sha256") != policy["canonical_head_library"]["sha256"]
            or sha(canonical_q6_tile.CUDA_SOURCE)
            not in build.get("source_sha256", {}).values()
        ):
            raise ValueError("Canonical Q6 binary/CUDA source identity differs")
        if importlib.metadata.version("humming-kernels") != "0.1.18":
            raise ValueError("Qualified Humming0.1.18 package required")
        files.extend((library, metadata))
    return {
        **policy,
        "scope": "actual Q6_K/BF16 body and semantic output head on SM86/SM120",
        "physical_row_limit": MAX_ROWS,
        "original_wire_retained": True,
        "original_whole_M_F32_unsafe_repair": True,
        "canonical_head_weight_rounding": "BF16",
        "body_input_metadata_output": "FP16 Humming with BF16 API and FP32 accumulation",
        "checkpoint_or_logical_batch_policy": False,
        "sources": {str(p.absolute()): sha(p) for p in files},
    }


class PreparedQ6Precision:
    def __init__(self, original, scope, policy):
        from . import canonical_q6_tile
        from .humming import prepare_gguf_humming

        if not original.weight.is_cuda:
            raise ValueError("Q6 precision preparation requires a CUDA tensor")
        self.target = tuple(torch.cuda.get_device_capability(original.weight.device))
        selected = head_library_identity(policy["canonical_head_library"]["path"])
        if (
            self.target not in HEAD_LIBRARIES
            or self.target != tuple(selected["target"])
            or selected != policy["canonical_head_library"]
        ):
            raise ValueError("Actual Q6 tensor target and selected head library differ")
        self.original, self.scope = original, scope
        self.packed = original.weight
        self.n, self.k = original.shape
        self.mode = policy["mode"]
        self.fallback = getattr(
            original,
            "_native_bf16_body" if scope == "body" else "_native_bf16_head",
            None,
        )
        self.pointers = torch.tensor(
            [self.packed.data_ptr()], device=self.packed.device, dtype=torch.int64
        )
        self.original_pointer = self.packed.data_ptr()
        if scope == "body":
            self.humming = prepare_gguf_humming(self.packed, 14, torch.float16)
            self.humming.compute_config = json.dumps(
                {
                    **json.loads(self.humming.compute_config),
                    "use_f16_accum": False,
                    "use_batch_invariant": False,
                }
            )
            cfg = self.humming.layer_config
            if (
                any(
                    str(getattr(cfg, key)) != "float16"
                    for key in ("a_dtype", "bs_dtype", "c_dtype")
                )
                or cfg.input_quant_mode.value != "none"
            ):
                raise ValueError("Original Half Humming configuration changed")
            self.humming._non_persistent_buffers_set.update(self.humming._buffers)
            self.layer_config = cfg.to_str()
            from humming.tune import get_heuristics_config

            self._stock_tunings = get_heuristics_config(
                cfg,
                device=self.packed.device,
                **json.loads(self.humming.compute_config),
            )
            self._tuning_cache = {}
        else:
            self.raw = canonical_q6_tile.PreparedRawPackedGEMV(
                self.packed,
                14,
                self.k,
                policy["canonical_head_library"]["path"],
                canonical_rounding=True,
            )

    def tuning_config(self, rows):
        if rows not in self._tuning_cache:
            stock = self._stock_tunings
            if isinstance(stock, dict):
                selected = stock
            else:
                matches = [cfg for low, high, cfg in stock if low < rows <= high]
                if len(matches) != 1:
                    raise ValueError("No unique stock Humming geometry interval")
                selected = matches[0]
            self._tuning_cache[rows] = json.dumps(
                {**selected, "use_stream_k": False}, sort_keys=True
            )
        return self._tuning_cache[rows]

    def supports_rows(self, rows):
        return (
            type(rows) is int
            and rows > 0
            and (
                rows <= MAX_ROWS
                or (self.fallback is not None and self.fallback.supports_rows(rows))
            )
        )

    def __call__(self, x, bias=None, *, output_dtype=None):
        from . import q6_precision_guard as guard

        dtype = self.original.dtype if output_dtype is None else output_dtype
        rows = x.numel() // self.k
        if rows > MAX_ROWS:
            if self.fallback is None:
                raise ValueError("Original fallback is unavailable for this row count")
            return self.fallback(x, bias, output_dtype=dtype)
        if (
            self.original.weight is not self.packed
            or self.packed.data_ptr() != self.original_pointer
            or dtype not in (torch.bfloat16, torch.float32)
        ):
            raise ValueError("Original Q6 buffer or output dtype changed")
        source = x.reshape(-1, self.k).contiguous()
        if self.scope == "body":
            result = guard.body(
                source,
                self.packed,
                self.pointers,
                self.humming.weight,
                self.humming.weight_scale,
                self.humming.locks,
                self.n,
                self.k,
                self.layer_config,
                self.humming.compute_config,
                self.tuning_config(rows),
                dtype == torch.float32,
            )
        else:
            result = guard.head(
                source,
                self.packed,
                self.pointers,
                self.raw.cache,
                self.n,
                self.k,
                self.raw.path,
                self.raw.multiprocessors,
                dtype == torch.float32,
            )
        result = result.reshape(*x.shape[:-1], self.n)
        return result if bias is None else result + bias


def prepare_model(model):
    from .compact import PackedGGUFLinear
    from .yaqa_prefill_cache import tensor_identity

    policy = settings()
    if policy["mode"] == "none":
        return {**policy, "prepared_matrices": 0, "additional_GPU_bytes": 0}
    runtime_marker()
    from .vocal_embeds import ParallelLMHead

    output_parts = {
        id(part)
        for layer in model.modules()
        if isinstance(layer, ParallelLMHead)
        or getattr(layer, "_gguf_tied_lm_head", False)
        for part in getattr(layer, "gguf_compact", ())
    }
    layers = {}
    catalog = []
    for name, part in model.named_modules():
        if not isinstance(part, PackedGGUFLinear):
            continue
        if getattr(part, "q6_precision_identity", None) is not None:
            raise RuntimeError(
                "Q6 precision is prepared once after original weight loading"
            )
        catalog.append(
            {
                "name": name,
                "qtype": part.weight_type,
                "dtype": str(part.dtype),
                "capability": torch.cuda.get_device_capability(part.weight.device)
                if part.weight.is_cuda
                else (0, 0),
                "canonical_rounding": bool(part.canonical_rounding),
                "role": part.role,
                "is_output_head": id(part) in output_parts,
                "shape": tuple(part.shape),
            }
        )
        layers[name] = part
    plan = plan_catalog(catalog, policy["additional_cache_budget_bytes"])
    staged = []
    for entry in plan["entries"]:
        part = layers[entry["name"]]
        original = tensor_identity(part.weight)
        prepared = PreparedQ6Precision(part, entry["scope"], policy)
        buffers = {"pointers": prepared.pointers}
        if entry["scope"] == "body":
            buffers.update(dict(prepared.humming.named_buffers()))
        else:
            buffers["canonical_codes"] = prepared.raw.cache
        identities = {
            key: {**tensor_identity(tensor), "pointer": tensor.data_ptr()}
            for key, tensor in buffers.items()
        }
        actual_bytes = sum(row["bytes"] for row in identities.values())
        if (
            actual_bytes > entry["reserved_bytes"]
            or tensor_identity(part.weight) != original
        ):
            raise ValueError(
                "Actual Q6 allocation exceeds full preflight or original wire changed"
            )
        identity = {
            **entry,
            "original": original,
            "buffers": identities,
            "actual_additional_bytes": actual_bytes,
            "mutable_workspace_names": ["locks"] if entry["scope"] == "body" else [],
            "fallback_retained": prepared.fallback is not None,
        }
        staged.append((part, prepared, identity))
    for part, prepared, identity in staged:
        part.register_buffer(
            "q6_precision_pointers", prepared.pointers, persistent=False
        )
        if prepared.scope == "body":
            part.add_module("q6_precision_humming", prepared.humming)
            part._native_bf16_body = prepared
        else:
            part.register_buffer(
                "q6_precision_codes", prepared.raw.cache, persistent=False
            )
            part._native_bf16_head = prepared
        part.q6_precision_identity = identity
    summary = {
        **policy,
        **plan,
        "prepared_matrices": len(staged),
        "additional_GPU_bytes": sum(
            row[2]["actual_additional_bytes"] for row in staged
        ),
        "identities": [row[2] for row in staged],
        "original_weight_GPU_hashes_verified": True,
        "new_buffer_GPU_hashes_verified": True,
    }
    model.gguf_q6_precision = summary
    return summary


def model_status(model, verify=False):
    from .yaqa_prefill_cache import tensor_identity

    result = getattr(
        model,
        "gguf_q6_precision",
        {"mode": "none", "prepared_matrices": 0, "additional_GPU_bytes": 0},
    )
    count = 0
    for part in model.modules():
        identity = getattr(part, "q6_precision_identity", None)
        if identity is None:
            continue
        count += 1
        prepared = getattr(
            part,
            "_native_bf16_body" if identity["scope"] == "body" else "_native_bf16_head",
        )
        if (
            type(prepared) is not PreparedQ6Precision
            or prepared.original is not part
            or prepared.packed is not part.weight
            or prepared.packed.data_ptr() != prepared.original_pointer
            or part.q6_precision_pointers is not prepared.pointers
        ):
            raise ValueError("Actual Q6 prepared object/original identity changed")
        if verify:
            if tensor_identity(part.weight) != identity["original"]:
                raise ValueError("Original Q6 wire changed")
            buffers = {"pointers": prepared.pointers}
            if prepared.scope == "body":
                buffers.update(dict(prepared.humming.named_buffers()))
            else:
                buffers["canonical_codes"] = prepared.raw.cache
            if set(buffers) != set(identity["buffers"]):
                raise ValueError("Q6 prepared tensor inventory changed")
            for key, value in buffers.items():
                expected = identity["buffers"][key]
                if value.data_ptr() != expected["pointer"]:
                    raise ValueError("Q6 prepared buffer pointer changed")
                if key in identity["mutable_workspace_names"]:
                    if (
                        list(value.shape) != expected["shape"]
                        or str(value.dtype) != expected["dtype"]
                    ):
                        raise ValueError("Q6 workspace metadata changed")
                elif tensor_identity(value) != {
                    k: v for k, v in expected.items() if k != "pointer"
                }:
                    raise ValueError("Q6 prepared weight/code changed")
    if count != result["prepared_matrices"]:
        raise ValueError("Q6 full plan/model inventory changed")
    return {**result, "verified": bool(verify), "observed_matrices": count}
