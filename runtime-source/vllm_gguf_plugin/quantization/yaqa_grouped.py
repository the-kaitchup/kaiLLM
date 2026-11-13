# SPDX-License-Identifier: Apache-2.0
"""Optional grouped original-code projections behind one opaque dispatch."""

import hashlib
import json
import os
import weakref
from functools import lru_cache
from pathlib import Path

import torch

ENV = "VLLM_GGUF_YAQA_GROUPED"
CORE_ENV = "VLLM_GGUF_YAQA_GROUPED_CORE_LIBRARY"
HIN_ENV = "VLLM_GGUF_YAQA_GROUPED_HIN_LIBRARY"
_PINNED = {
    CORE_ENV: "5f4951b835b587e4b1051a92b2886f03cfe2a50366684e57cf390375d66cbc5d",
    HIN_ENV: "292ab6d3495a6560fc0bbab56c224ee67a75a1e4a12820820bcd4b0d50ed557b",
}
_STATES = weakref.WeakValueDictionary()
_OCCURRENCES = {}
_CALLS = {}
_TARGETS = ((8, 6), (12, 0))


def enabled():
    value = os.getenv(ENV, "none")
    if value not in ("none", "auto"):
        raise ValueError(f"{ENV} requires none or auto")
    if value == "auto" and os.getenv("VLLM_GGUF_YAQA_OPAQUE", "0") != "1":
        raise ValueError("Grouped dispatch requires VLLM_GGUF_YAQA_OPAQUE=1")
    return value == "auto"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=4)
def _library(env, target=(8, 6)):
    if target not in _TARGETS:
        raise ValueError("Grouped Q4 target is not qualified")
    source_name = (
        "yaqa_grouped_k4.cu" if env == CORE_ENV else "yaqa_grouped_power_input.cu"
    )
    source = Path(__file__).resolve().parents[1] / "native/kernels" / source_name
    configured = os.getenv(env)
    if configured:
        if target != (8, 6):
            raise ValueError("A pinned SM86 grouped override cannot select SM120")
        path = Path(configured).absolute()
        actual = _sha(path)
        if actual != _PINNED[env]:
            raise ValueError("Grouped library override differs from qualified binary")
        return path, actual
    if target == (12, 0) and env == HIN_ENV:
        from .yaqa_grouped_hin import _library as input_library

        identity = input_library(target)
        return Path(identity["library"]), identity["library_sha256"]
    from .yaqa_cuda import build_cuda_library

    identity = build_cuda_library(
        source,
        "sm86-original-grouped-v1",
        source.stem,
        architecture=f"sm{target[0]}{target[1]}",
    )
    return Path(identity["library"]), identity["library_sha256"]


def _captured_measured_policy(part):
    key = getattr(part, "cuda_k34_key", None)
    if not isinstance(key, str):
        return False
    try:
        identity = json.loads(key)
    except (TypeError, ValueError):
        return False
    return isinstance(identity, dict) and identity.get("measured_b1_policy") is True


def _uniform_resource_policy(policies):
    resources = {(p[0], p[1], p[3]) for p in policies}
    if len(resources) != 1:
        return None
    warps, prefetch, shared = next(iter(resources))
    return warps, prefetch, shared, tuple(p[2] for p in policies)


def _sm120_group_policy(parts):
    from .yaqa_cuda_k34 import gemv_policy, tile_splits

    policies = []
    for part in parts:
        policy = (
            gemv_policy(4, *part.shape) if _captured_measured_policy(part) else None
        )
        if policy is None:
            policy = (4, 4, tile_splits(*part.shape), 1)
        policies.append(policy)
    return _uniform_resource_policy(policies)


def compatible(parts):
    if not 2 <= len(parts) <= 65535:
        return False
    for p in parts:
        if (
            getattr(p, "codec", None) != "qtip"
            or p.bits != 4
            or not p.permutation
            or getattr(p, "cuda_k34_key", None) is None
            or getattr(p, "validate_b1", False)
            or p.shape[1] != 2048
            or p.shape[0] < 16
            or p.shape[0] % 16
            or p.hadamard_input.shape != (1, 1)
            or p.input_scale.shape != (2048,)
            or p.input_scale.dtype != torch.float32
            or p.dtype not in (torch.float32, torch.bfloat16)
            or p.trellis.device.type != "cuda"
            or not p.input_scale.is_contiguous()
            or p.input_scale.device != p.trellis.device
            or p.trellis.device != parts[0].trellis.device
            or p.output_scale.dtype != torch.float32
            or p.output_scale.shape != (p.shape[0],)
            or p.hadamard_output.ndim != 2
            or p.hadamard_output.shape[0] < 1
            or p.hadamard_output.shape
            != (p.hadamard_output.shape[0], p.hadamard_output.shape[0])
            or p.shape[0] % p.hadamard_output.shape[0]
            or p.Wscale.numel() != 1
            or p.Wscale.dtype not in (torch.float32, torch.float64)
            or p.hadamard_output.dtype not in (torch.float32, torch.float64)
            or any(
                not t.is_contiguous() or t.device != p.trellis.device
                for t in (
                    p.trellis,
                    p.tlut,
                    p.output_scale,
                    p.hadamard_output,
                    p.Wscale,
                )
            )
        ):
            return False
        power = p.shape[0] // p.hadamard_output.shape[0]
        if power & (power - 1):
            return False
    from .yaqa_cuda_k34 import tile_splits
    from .yaqa_grouped_k4 import descriptor_rows

    try:
        descriptor_rows(
            [p.shape for p in parts],
            [tile_splits(*p.shape) for p in parts],
            1,
            4,
            [(p.trellis.data_ptr(), p.tlut.data_ptr()) for p in parts],
        )
    except ValueError:
        return False
    target = tuple(torch.cuda.get_device_capability(parts[0].trellis.device))
    if (
        target == (12, 0)
        and any(_captured_measured_policy(p) for p in parts)
        and _sm120_group_policy(parts) is None
    ):
        return False
    return target in _TARGETS


class PreparedGroup:
    def __init__(self, layer, target=(8, 6)):
        if target not in _TARGETS:
            raise ValueError("Grouped Q4 target is not qualified")
        if target == (12, 0):
            from . import yaqa_grouped_hin_input as hin
        else:
            from . import yaqa_grouped_power_input as hin
        from .prefill_context import decode_query_limit
        from .yaqa_grouped_k4 import PreparedGroupedQ4

        self.target = target
        self._input_apply = hin.apply
        self.parts = tuple(layer.yaqa_parts)
        self.query_limit = decode_query_limit()
        self.total = sum(p.shape[0] for p in self.parts)
        self.device = self.parts[0].trellis.device
        # Consolidate only small mutable SU metadata. Original buffers become
        # views, so an in-place update is visible to both execution paths.
        self.scales = torch.stack([p.input_scale for p in self.parts])
        layer.register_buffer("_yaqa_grouped_scales", self.scales, persistent=False)
        for index, part in enumerate(self.parts):
            part.input_scale = self.scales[index]
        core_path, core_sha = _library(CORE_ENV, target)
        hin_path, hin_sha = _library(HIN_ENV, target)
        hin.load(hin_path, hin_sha)
        if target == (12, 0) and any(_captured_measured_policy(p) for p in self.parts):
            policy = _sm120_group_policy(self.parts)
            if policy is None:
                raise ValueError("Grouped parts require one actual resource policy")
            warps, prefetch, shared, splits = policy
            self.prepared = PreparedGroupedQ4(
                self.parts,
                1,
                core_path,
                core_sha,
                warps=warps,
                prefetch=prefetch,
                shared=shared,
                splits_override=splits,
            )
        else:
            self.prepared = PreparedGroupedQ4(self.parts, 1, core_path, core_sha)
        self.library_identities = {str(core_path): core_sha, str(hin_path): hin_sha}
        descriptor = {
            "shapes": [list(p.shape) for p in self.parts],
            "bits": [p.bits for p in self.parts],
            "dtype": str(self.parts[0].dtype),
        }
        digest = hashlib.sha256(
            json.dumps(descriptor, sort_keys=True).encode()
        ).hexdigest()
        count = _OCCURRENCES.get(digest, 0)
        _OCCURRENCES[digest] = count + 1
        self.key = f"{digest}:{count}"
        _STATES[self.key] = self
        sample = torch.zeros((1, 2048), device=self.device, dtype=self.parts[0].dtype)
        self.grouped(sample)  # Prepare every new kernel before engine capture.

    def grouped(self, x):
        transformed = self._input_apply(x.reshape(1, 2048).contiguous(), self.scales)
        out = self.prepared.post(
            self.prepared.core(transformed), x.dtype, fuse_k1=False
        )
        return out.reshape(*x.shape[:-1], self.total)

    def dispatch(self, x):
        from .prefill_context import context_is_prefill
        from .yaqa_prefill_cache_out import apply as apply_prefill
        from .yaqa_prefill_cache_out import selected as selected_prefill

        rows = x.numel() // x.shape[-1]
        if selected_prefill(x, self.parts):
            route = "merged-prefill"
            result = apply_prefill(x, self.parts)
        elif (
            rows == 1
            and x.device == self.device
            and x.dtype in (torch.float32, torch.bfloat16)
            and not context_is_prefill(self.query_limit)
        ):
            route = "grouped-original"
            result = self.grouped(x)
        else:
            route = "original-parts"
            outputs = [p(x) for p in self.parts]
            result = torch.ops.vllm._gguf_concat_projections(outputs)
        key = f"{route}:{rows}"
        _CALLS[key] = _CALLS.get(key, 0) + 1
        return result

    def buffers(self):
        return [self.scales, self.prepared.descriptors, self.prepared.ptrs] + [
            t for p in self.parts for t in p.buffers()
        ]


def prepare(layer):
    if os.getenv("VLLM_GGUF_YAQA_GROUPED_K234", "none") != "none":
        from .yaqa_grouped_k234_dispatch import prepare as prepare_mixed

        candidate = prepare_mixed(layer)
        if candidate is not None:
            return candidate
    if not enabled() or getattr(layer, "_yaqa_qkv_cache", None) is not None:
        return None
    if not compatible(layer.yaqa_parts):
        return None
    return PreparedGroup(
        layer,
        tuple(torch.cuda.get_device_capability(layer.yaqa_parts[0].trellis.device)),
    )


@torch.library.custom_op("vllm_gguf::yaqa_grouped_linear", mutates_args=())
def opaque_grouped(
    x: torch.Tensor, buffers: list[torch.Tensor], key: str, columns: int
) -> torch.Tensor:
    state = _STATES.get(key)
    if state is None or state.total != columns or not buffers:
        raise RuntimeError(
            "Grouped state and actual buffers must be prepared before capture"
        )
    return state.dispatch(x)


@opaque_grouped.register_fake
def _fake(x, buffers, key, columns):
    return x.new_empty((*x.shape[:-1], columns))


def runtime_marker():
    active = enabled()
    root = Path(__file__).resolve().parent
    names = [
        "yaqa_grouped.py",
        "yaqa_grouped_k4.py",
        "yaqa_grouped_power_input.py",
        "yaqa_grouped_hin_input.py",
        "yaqa_grouped_hin.py",
        "yaqa_grouped_k234.py",
        "yaqa_grouped_k234_metadata.py",
        "yaqa_grouped_k234_dispatch.py",
    ]
    sources = {str(root / n): _sha(root / n) for n in names}
    for name in (
        "yaqa_grouped_k4.cu",
        "yaqa_grouped_power_input.cu",
        "yaqa_grouped_k234_original.cu",
    ):
        path = root.parent / "native/kernels" / name
        sources[str(path)] = _sha(path)
    libraries = {}
    target = None
    if active:
        from ..native.cuda_architecture import resolve_architecture

        target = resolve_architecture()
    if active and target in _TARGETS:
        libraries = {
            str(p): digest
            for p, digest in (_library(CORE_ENV, target), _library(HIN_ENV, target))
        }
    mixed = {}
    if os.getenv("VLLM_GGUF_YAQA_GROUPED_K234", "none") != "none":
        from .yaqa_grouped_k234_dispatch import runtime_marker as mixed_marker

        mixed = mixed_marker(target)
    return {
        "mixed_bit_original": mixed,
        "contract": (
            f"grouped-original-QTIP4-SM{target[0]}{target[1]}-v1"
            if active and target in _TARGETS
            else "grouped-original-QTIP4-SM86-v1"
        ),
        "enabled": active,
        "native_target": target,
        "hardware_supported": target in _TARGETS if active else None,
        "original_CUDA_k34_capability_required": True,
        "physical_row_cost_policy": "grouped M1; original fallback for other rows",
        "geometry": "common K2048,power input,positive16-aligned outputs,original wire",
        "prefill_priority": True,
        "bias_applied_once_by_original_linear": True,
        "persistent_weight_bytes_added": 0,
        "SU_metadata_replaces_original_buffers_with_views": True,
        "sources": sources,
        "libraries": libraries,
    }


def status():
    return {
        "prepared_groups": len(_STATES),
        "Python_dispatch_counts_include_capture_not_replay": dict(_CALLS),
        "groups": [
            {
                "shapes": list(s.prepared.shapes),
                "SU_bytes": s.scales.numel() * 4,
                "libraries": s.library_identities,
            }
            for s in _STATES.values()
        ],
    }
