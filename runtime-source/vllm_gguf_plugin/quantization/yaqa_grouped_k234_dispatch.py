# SPDX-License-Identifier: Apache-2.0
"""Unselected SM120 fusion retaining each original K2/K3/K4 CTA policy."""

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import torch

from . import yaqa_grouped as original

ENV = "VLLM_GGUF_YAQA_GROUPED_K234"
SOURCE_SHA256 = "3ae746465ca00edb17c22f8592dec151bae94d38342eb96141a2893f1fbb1ed2"
ORIGINAL_SOURCES = {
    2: "9faedeec1fb779f035893ff7cdc4eda043584f885b1161e73191770af4dc6d98",
    3: "e801676ff6daa463a9dc9d33c444dffc73c699559a1ac1bc86773e7cd519a106",
    4: "d692f3f58fe626d328b5ce665da95159a4756d94c7fc23f951fe3f67e0e5d979",
}


def enabled():
    value = os.getenv(ENV, "none")
    if value not in ("none", "auto"):
        raise ValueError(f"{ENV} requires none or auto")
    if value == "auto" and not original.enabled():
        raise ValueError("Mixed-bit grouping requires original grouped/opaque dispatch")
    return value == "auto"


def original_policy(part, target):
    """Recover the original selected source and resources, without retuning."""
    from .yaqa_cuda_k34 import gemv_policy, tile_splits

    bits = getattr(part, "bits", None)
    if bits not in ORIGINAL_SOURCES:
        return None
    key = getattr(part, "cuda_k2_key" if bits == 2 else "cuda_k34_key", None)
    if not isinstance(key, str):
        return None
    try:
        identity = json.loads(key)
    except (TypeError, ValueError):
        return None
    if not isinstance(identity, dict):
        return None
    # Unmeasured K3 may select its distinct funnel shader. Keep its old path.
    if bits != 2 and (
        identity.get("bits") != bits or identity.get("measured_b1_policy") is not True
    ):
        return None
    source = identity.get("source_sha256")
    arch = identity.get("architecture", {})
    if not isinstance(arch, dict):
        return None
    if (
        tuple(arch.get("compute_capability", ())) != target
        or source != ORIGINAL_SOURCES[bits]
    ):
        return None
    if bits == 2:
        return (4, 4, 1, 1)
    policy = gemv_policy(bits, *part.shape)
    return policy if policy is not None else (4, 4, tile_splits(*part.shape), 1)


def compatible(parts, target):
    from .yaqa_grouped_hin import compatible as compatible_input
    from .yaqa_grouped_k234_metadata import descriptor_rows

    if target != (12, 0) or not compatible_input(parts):
        return None
    for part in parts:
        if (
            part.bits not in (2, 3, 4)
            or not part.permutation
            or part.dtype not in (torch.float32, torch.bfloat16)
            or part.dtype != parts[0].dtype
            or part.shape[0] < 16
            or part.shape[0] % 16
            or part.output_scale.dtype != torch.float32
            or part.output_scale.shape != (part.shape[0],)
            or part.hadamard_output.ndim != 2
            or part.hadamard_output.shape[0] < 1
            or part.hadamard_output.shape
            != (part.hadamard_output.shape[0], part.hadamard_output.shape[0])
            or part.shape[0] % part.hadamard_output.shape[0]
            or part.Wscale.numel() != 1
            or part.Wscale.dtype not in (torch.float32, torch.float64)
            or part.hadamard_output.dtype not in (torch.float32, torch.float64)
            or any(
                not t.is_contiguous() or t.device != part.trellis.device
                for t in (
                    part.trellis,
                    part.tlut,
                    part.output_scale,
                    part.hadamard_output,
                    part.Wscale,
                )
            )
        ):
            return None
        power = part.shape[0] // part.hadamard_output.shape[0]
        if power & (power - 1):
            return None
    policies = tuple(original_policy(part, target) for part in parts)
    if any(policy is None for policy in policies):
        return None
    try:
        descriptor_rows(
            [p.shape for p in parts],
            [p.bits for p in parts],
            policies,
            1,
            [(p.trellis.data_ptr(), p.tlut.data_ptr()) for p in parts],
        )
    except ValueError:
        return None
    return policies


@lru_cache(maxsize=1)
def library():
    from .yaqa_cuda import build_cuda_library

    source = (
        Path(__file__).resolve().parents[1]
        / "native/kernels/yaqa_grouped_k234_original.cu"
    )
    if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Mixed-bit original arithmetic source differs")
    return build_cuda_library(
        source,
        "sm86-original-grouped-k234-per-CTA-v1",
        "yaqa_grouped_k234_original",
        architecture="sm120",
    )


class PreparedMixedGroup(original.PreparedGroup):
    def __init__(self, layer, policies):
        from . import yaqa_grouped_hin_input as hin
        from .prefill_context import decode_query_limit
        from .yaqa_grouped_hin import _library as input_library
        from .yaqa_grouped_k234 import PreparedGroupedK234

        self.target = (12, 0)
        self._input_apply = hin.apply
        self.parts = tuple(layer.yaqa_parts)
        self.query_limit = decode_query_limit()
        self.total = sum(part.shape[0] for part in self.parts)
        self.device = self.parts[0].trellis.device
        self.scales = torch.stack([part.input_scale for part in self.parts])
        layer.register_buffer("_yaqa_grouped_scales", self.scales, persistent=False)
        for index, part in enumerate(self.parts):
            part.input_scale = self.scales[index]
        resources = {(p[0], p[1], p[3]) for p in policies}
        static_q4 = all(part.bits == 4 for part in self.parts) and len(resources) == 1
        if static_q4:
            core_path, core_sha = original._library(original.CORE_ENV, self.target)
            core_identity = {"library": str(core_path), "library_sha256": core_sha}
            self.core_backend = "qualified-static-Q4"
        else:
            core_identity = library()
            self.core_backend = "per-CTA-mixed-K234"
        input_identity = input_library(self.target)
        hin.load(input_identity["library"], input_identity["library_sha256"])
        if static_q4:
            from .yaqa_grouped_k4 import PreparedGroupedQ4

            warps, prefetch, shared = next(iter(resources))
            self.prepared = PreparedGroupedQ4(
                self.parts,
                1,
                core_identity["library"],
                core_identity["library_sha256"],
                warps=warps,
                prefetch=prefetch,
                shared=shared,
                splits_override=tuple(policy[2] for policy in policies),
            )
            # Metadata only: leave .policy unchanged because core() consumes it.
            self.prepared.policies = tuple(tuple(policy) for policy in policies)
        else:
            self.prepared = PreparedGroupedK234(
                self.parts,
                1,
                core_identity["library"],
                core_identity["library_sha256"],
                policies=policies,
            )
        self.library_identities = {
            core_identity["library"]: core_identity["library_sha256"],
            input_identity["library"]: input_identity["library_sha256"],
        }
        descriptor = {
            "contract": "mixed-original-k234-per-CTA-v1",
            "core_backend": self.core_backend,
            "core_library_sha256": core_identity["library_sha256"],
            "shapes": [list(part.shape) for part in self.parts],
            "bits": [part.bits for part in self.parts],
            "dtype": str(self.parts[0].dtype),
            "policies": policies,
            "target": self.target,
        }
        digest = hashlib.sha256(
            json.dumps(descriptor, sort_keys=True).encode()
        ).hexdigest()
        count = original._OCCURRENCES.get(digest, 0)
        original._OCCURRENCES[digest] = count + 1
        self.key = f"{digest}:{count}"
        original._STATES[self.key] = self
        sample = torch.zeros((1, 2048), device=self.device, dtype=self.parts[0].dtype)
        self.grouped(sample)

    # grouped() and buffers() are exactly the original full-group implementation.
    # Its post alias retains the original reduction, Hadamard, scale and cast.

    def dispatch(self, x):
        from .prefill_context import context_is_prefill
        from .yaqa_kernels import packed_yaqa_linear
        from .yaqa_prefill_cache import selected as selected_cache
        from .yaqa_prefill_cache_out import apply as apply_prefill
        from .yaqa_prefill_cache_out import selected as selected_prefill

        rows = x.numel() // x.shape[-1]
        if selected_prefill(x, self.parts):
            route = "merged-prefill"
            result = apply_prefill(x, self.parts)
        elif (
            x.device == self.device
            and x.dtype in (torch.float32, torch.bfloat16)
            and x.shape[-1] == 2048
            and x.is_contiguous()
            and x.data_ptr() % 16 == 0
            and 1 <= rows <= 32
            and not context_is_prefill(self.query_limit)
            and not any(selected_cache(x, part) for part in self.parts)
        ):
            if rows == 1:
                route = "grouped-mixed-original"
                result = self.grouped(x)
            else:
                # Preserve the previously qualified Hin benefit for physical
                # M>1; those original cores may select a different MMA backend.
                route = "grouped-input-original-cores"
                from .yaqa_hin_high_low import maybe_grouped

                prepared = maybe_grouped(x, self.parts, self.scales)
                if prepared is None:
                    transformed = self._input_apply(x.reshape(rows, 2048), self.scales)
                    outputs = [
                        packed_yaqa_linear(x, part, prepared_source=transformed[g])
                        for g, part in enumerate(self.parts)
                    ]
                else:
                    transformed, high, low, safe = prepared
                    outputs = [
                        packed_yaqa_linear(
                            x,
                            part,
                            prepared_source=transformed[g],
                            prepared_residual=(high[g], low[g], safe[g]),
                        )
                        for g, part in enumerate(self.parts)
                    ]
                result = torch.ops.vllm._gguf_concat_projections(outputs)
        else:
            route = "original-parts"
            result = torch.ops.vllm._gguf_concat_projections(
                [part(x) for part in self.parts]
            )
        key = f"{route}:{rows}"
        original._CALLS[key] = original._CALLS.get(key, 0) + 1
        return result


def prepare(layer):
    if not enabled() or getattr(layer, "_yaqa_qkv_cache", None) is not None:
        return None
    parts = layer.yaqa_parts
    if not parts:
        return None
    device = getattr(getattr(parts[0], "trellis", None), "device", None)
    if getattr(device, "type", None) != "cuda":
        return None
    target = tuple(torch.cuda.get_device_capability(device))
    policies = compatible(parts, target)
    return PreparedMixedGroup(layer, policies) if policies is not None else None


def runtime_marker(target=None):
    root = Path(__file__).resolve().parent
    paths = [
        Path(__file__),
        root.parent / "__init__.py",
        root / "yaqa_cuda.py",
        root / "yaqa_cuda_k34.py",
        root / "yaqa_sm120_b1_policy.py",
        root / "prefill_context.py",
        root / "yaqa_prefill_cache.py",
        root / "yaqa_prefill_cache_out.py",
        root / "yaqa_kernels.py",
        root / "yaqa_grouped.py",
        root / "yaqa_grouped_k4.py",
        root / "yaqa_grouped_hin.py",
        root / "yaqa_grouped_hin_input.py",
        root / "yaqa_grouped_k234.py",
        root / "yaqa_grouped_k234_metadata.py",
        root.parent / "native/kernels/yaqa_grouped_k234_original.cu",
        root.parent / "native/kernels/yaqa_grouped_power_input.cu",
    ]
    return {
        "enabled": enabled(),
        "contract": "mixed-original-k234-per-CTA-v1",
        "hardware_selection": "SM120",
        "physical_rows": "1; original Hin/cores for2..32",
        "core_factory": (
            "all Q4 and uniform actual resources: qualified static Q4; "
            "otherwise per-CTA K234"
        ),
        "prepared_core_backends": sorted(
            {
                state.core_backend
                for state in original._STATES.values()
                if isinstance(state, PreparedMixedGroup)
            }
        ),
        "per_part_policy_preserved": True,
        "persistent_weight_bytes_added": 0,
        "library": library() if enabled() and target == (12, 0) else {},
        "sources": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
    }
