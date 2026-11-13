# SPDX-License-Identifier: Apache-2.0
"""Share launches for independent YAQA input transforms; keep original cores."""

import hashlib
import json
import os
import weakref
from functools import lru_cache
from pathlib import Path

import torch

ENV = "VLLM_GGUF_YAQA_GROUPED_HIN"
_STATES = weakref.WeakValueDictionary()
_OCCURRENCES = {}
_CALLS = {}
_TARGETS = ((8, 6), (12, 0))


def enabled():
    value = os.getenv(ENV, "none")
    if value not in ("none", "auto"):
        raise ValueError(f"{ENV} requires none or auto")
    if value == "auto" and os.getenv("VLLM_GGUF_YAQA_OPAQUE", "0") != "1":
        raise ValueError("Grouped input transforms require opaque YAQA dispatch")
    return value == "auto"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=2)
def _library(target):
    if target not in _TARGETS:
        raise ValueError("Unqualified grouped-input CUDA target")
    from .yaqa_cuda import build_cuda_library

    source = (
        Path(__file__).resolve().parents[1]
        / "native/kernels/yaqa_grouped_power_input.cu"
    )
    return build_cuda_library(
        source,
        "sm86-independent-original-power-input-GxBx2048-v1",
        f"yaqa_grouped_power_input_sm{target[0]}{target[1]}",
        architecture=f"sm{target[0]}{target[1]}",
    )


def compatible(parts):
    if not 2 <= len(parts) <= 65535 or len(parts) * 32 * 2048 * 4 > 4 << 20:
        return False
    device = getattr(getattr(parts[0], "trellis", None), "device", None)
    return all(
        getattr(p, "codec", None) == "qtip"
        and 1 <= int(p.bits) <= 8
        and p.shape[1] == 2048
        and p.shape[0] > 0
        and p.hadamard_input.shape == (1, 1)
        and getattr(p, "cuda_power_key", None) is not None
        and not getattr(p, "validate_b1", False)
        and p.input_scale.shape == (2048,)
        and p.input_scale.dtype == torch.float32
        and p.input_scale.is_contiguous()
        and p.input_scale.is_cuda
        and p.input_scale.device == device
        and p.trellis.device == device
        and p.dtype in (torch.float32, torch.bfloat16, torch.float16)
        for p in parts
    )


class PreparedGroup:
    def __init__(self, layer, target):
        from . import yaqa_grouped_hin_input as hin
        from .prefill_context import decode_query_limit

        self.parts = tuple(layer.yaqa_parts)
        self.target = target
        self.query_limit = decode_query_limit()
        self.total = sum(p.shape[0] for p in self.parts)
        self.device = self.parts[0].trellis.device
        # Replace the small SU buffers with views, preserving mutation semantics.
        self.scales = torch.stack([p.input_scale for p in self.parts])
        layer.register_buffer("_yaqa_grouped_hin_scales", self.scales, persistent=False)
        for index, part in enumerate(self.parts):
            part.input_scale = self.scales[index]
        identity = _library(target)
        hin.load(identity["library"], identity["library_sha256"])
        self.library_identity = identity
        descriptor = {
            "shapes": [list(p.shape) for p in self.parts],
            "bits": [int(p.bits) for p in self.parts],
            "dtype": str(self.parts[0].dtype),
            "target": target,
        }
        digest = hashlib.sha256(
            json.dumps(descriptor, sort_keys=True).encode()
        ).hexdigest()
        occurrence = _OCCURRENCES.get(digest, 0)
        _OCCURRENCES[digest] = occurrence + 1
        self.key = f"{digest}:{occurrence}"
        _STATES[self.key] = self
        # The only new kernel is compiled CUDA, with the same arithmetic for M.
        sample = torch.zeros((1, 2048), device=self.device, dtype=self.parts[0].dtype)
        hin.apply(sample, self.scales)

    def grouped(self, x):
        from .yaqa_grouped_hin_input import apply
        from .yaqa_kernels import packed_yaqa_linear

        rows = x.numel() // 2048
        transformed = apply(x.reshape(rows, 2048).contiguous(), self.scales)
        outputs = [
            packed_yaqa_linear(x, part, prepared_source=transformed[index])
            for index, part in enumerate(self.parts)
        ]
        return torch.ops.vllm._gguf_concat_projections(outputs)

    def dispatch(self, x):
        from .prefill_context import context_is_prefill
        from .yaqa_prefill_cache import selected as selected_cache
        from .yaqa_prefill_cache_out import apply as apply_prefill
        from .yaqa_prefill_cache_out import selected as selected_prefill

        rows = x.numel() // x.shape[-1]
        if selected_prefill(x, self.parts):
            route = "merged-prefill"
            result = apply_prefill(x, self.parts)
        elif (
            x.device == self.device
            and x.dtype in (torch.float32, torch.bfloat16, torch.float16)
            and x.shape[-1] == 2048
            and 1 <= rows <= 32
            and x.is_contiguous()
            and x.data_ptr() % 16 == 0
            and not context_is_prefill(self.query_limit)
            and not any(selected_cache(x, part) for part in self.parts)
        ):
            route = "grouped-input-original-cores"
            result = self.grouped(x)
        else:
            route = "original-parts"
            outputs = [part(x) for part in self.parts]
            result = torch.ops.vllm._gguf_concat_projections(outputs)
        key = f"{route}:{rows}"
        _CALLS[key] = _CALLS.get(key, 0) + 1
        return result

    def buffers(self):
        return [self.scales] + [tensor for p in self.parts for tensor in p.buffers()]


def prepare(layer):
    if (
        not enabled()
        or getattr(layer, "_yaqa_qkv_cache", None) is not None
        or getattr(layer, "_yaqa_grouped_state", None) is not None
        or not compatible(layer.yaqa_parts)
    ):
        return None
    from ..native.cuda_architecture import resolve_architecture

    target = resolve_architecture()
    return PreparedGroup(layer, target) if target in _TARGETS else None


@torch.library.custom_op("vllm_gguf::yaqa_grouped_hin_linear", mutates_args=())
def opaque_grouped_hin(
    x: torch.Tensor, buffers: list[torch.Tensor], key: str, columns: int
) -> torch.Tensor:
    state = _STATES.get(key)
    if state is None or state.total != columns or not buffers:
        raise RuntimeError("Grouped-input state/buffers were not prepared")
    return state.dispatch(x)


@opaque_grouped_hin.register_fake
def _fake(x, buffers, key, columns):
    return x.new_empty((*x.shape[:-1], columns))


def runtime_marker():
    active = enabled()
    target = None
    if active:
        from ..native.cuda_architecture import resolve_architecture

        target = resolve_architecture()
    supported = target in _TARGETS
    root = Path(__file__).resolve().parent
    paths = [
        root / "yaqa_grouped_hin.py",
        root / "yaqa_grouped_hin_input.py",
        root / "yaqa_kernels.py",
        root / "yaqa_linear.py",
        root.parent / "native/kernels/yaqa_grouped_power_input.cu",
    ]
    return {
        "contract": "grouped-independent-YAQA-Hin-original-cores-v1",
        "enabled": active,
        "target": target,
        "hardware_supported": supported if active else None,
        "library": _library(target) if active and supported else {},
        "sources": {str(p): _sha(p) for p in paths},
        "selection": "QTIP bits1..8/K2048/power input/M1..32/metadata resource cap",
        "semantic_prefill_and_any_selected_part_cache_priority": True,
        "original_decode_reduction_post_and_concat": True,
        "persistent_weight_bytes_added": 0,
        "SU_metadata_replaces_original_buffers_with_views": True,
    }


def status():
    return {
        "prepared_groups": len(_STATES),
        "Python_dispatch_counts_include_capture_not_replay": dict(_CALLS),
        "groups": [
            {
                "shapes": [list(p.shape) for p in s.parts],
                "bits": [int(p.bits) for p in s.parts],
                "SU_bytes": s.scales.numel() * 4,
                "library": s.library_identity,
            }
            for s in _STATES.values()
        ],
    }
