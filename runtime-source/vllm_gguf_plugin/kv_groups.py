# SPDX-License-Identifier: Apache-2.0
"""KV-cache group size for hybrid models (KAILLM_KV_GROUPS=auto; opt-in, memory layout only).

vLLM groups the layers of each KV-cache type into groups of equal size (one block table per group, every block spans
one page per layer of its group) and pads the last group of each type.  The size is the smallest type's layer count
(or the largest when all are close).  Qwen3.8 27B + DSpine (48 GDN + 16 full-attention + 5 sliding-window drafter
layers) gets 5: the 16 full-attention layers are padded to 20, so a quarter of every attention block -- the only
memory that grows with the context -- is padding.

This picks the size that pads the unbounded (full-attention) types least, then pads least overall, then makes the
fewest groups, among sizes giving at most max(16, vLLM's count) groups; vLLM's own choice wins ties.  Qwen3.8 +
DSpine -> 8 (full attention unpadded, 3 padding layers in the drafter's window-bounded group).  Same KV numerics.
"""
from __future__ import annotations

import os


def _choose(sizes: list[int], unbounded: list[bool], default: int) -> int:
    def pad(n, g):
        return (g - n % g) % g

    def cost(g):
        pad_u = sum(pad(n, g) for n, u in zip(sizes, unbounded) if u)
        n_u = sum(n for n, u in zip(sizes, unbounded) if u) or 1
        return (pad_u / n_u, sum(pad(n, g) for n in sizes), sum(-(-n // g) for n in sizes))

    cap = max(16, cost(default)[2])
    best = default
    for g in range(1, max(sizes) + 1):
        c = cost(g)
        if c[2] <= cap and c < cost(best):
            best = g
    return best


def install() -> bool:
    if os.getenv("KAILLM_KV_GROUPS", "") != "auto":
        return False
    from collections import defaultdict

    from vllm.utils.math_utils import cdiv
    from vllm.v1.core import kv_cache_planning as planning
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    if getattr(planning, "_kai_kv_groups", False):
        return True

    def grouped(kv_cache_spec):
        # Same bucketing as vLLM's _get_kv_cache_groups_uniform_page_size; only the group size differs.
        same_type_layers = defaultdict(list)
        for layer_name, layer_spec in kv_cache_spec.items():
            same_type_layers[layer_spec].append(layer_name)
        layer_buckets, spec_buckets = [], []
        for layer_spec, layer_names in same_type_layers.items():
            for names, specs in zip(layer_buckets, spec_buckets):
                try:
                    type(specs[0]).merge([*specs, layer_spec])
                except (AssertionError, ValueError):
                    continue
                names.extend(layer_names)
                specs.append(layer_spec)
                break
            else:
                layer_buckets.append(list(layer_names))
                spec_buckets.append([layer_spec])
        sizes = [len(layers) for layers in layer_buckets]
        default = min(sizes)
        if max(sizes) < min(sizes) * 1.5:
            default = max(sizes)
        unbounded = [isinstance(specs[0], FullAttentionSpec) for specs in spec_buckets]
        group_size = _choose(sizes, unbounded, default)
        if group_size != default:
            planning.logger.info("kaiLLM: KV-cache group size %d instead of %d (layers per type %s)",
                                 group_size, default, sizes)
        grouped_layers = []
        for layers in layer_buckets:
            num_padding_layers = group_size - len(layers) % group_size
            if num_padding_layers != group_size:
                planning.logger.warning("Add %d padding layers, may waste at most %.2f%% KV cache memory",
                                        num_padding_layers, num_padding_layers / len(layers) * 100)
            num_groups = cdiv(len(layers), group_size)
            for i in range(num_groups):
                grouped_layers.append(layers[i::num_groups])
        return planning.create_kv_cache_group_specs(kv_cache_spec, grouped_layers)

    planning._get_kv_cache_groups_uniform_page_size = grouped
    planning._kai_kv_groups = True
    return True
