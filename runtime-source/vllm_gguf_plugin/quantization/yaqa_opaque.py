# SPDX-License-Identifier: Apache-2.0
"""Opaque YAQA dispatch boundary for an isolated compiled-inference candidate.

Tensor inputs include every actual layer buffer. The process-local weak registry
only supplies immutable dispatch descriptors; no dense weight copy is created.
"""

import hashlib
import json
import weakref

import torch

_LAYERS = weakref.WeakValueDictionary()
_DESCRIPTOR_COUNTS = {}
_CALLS_BY_ROWS = {}


def register_layer(layer, descriptor):
    digest = hashlib.sha256(
        json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    # Stable descriptor/order identity across identical model-worker startups.
    occurrence = _DESCRIPTOR_COUNTS.get(digest, 0)
    _DESCRIPTOR_COUNTS[digest] = occurrence + 1
    key = f"{digest}:{occurrence}:{layer.dtype}"
    if key in _LAYERS:
        raise ValueError("Opaque YAQA registry key collision")
    _LAYERS[key] = layer
    return key


@torch.library.custom_op("vllm_gguf::yaqa_opaque_linear", mutates_args=())
def opaque_linear(
    x: torch.Tensor, buffers: list[torch.Tensor], key: str, m: int
) -> torch.Tensor:
    from vllm_gguf_plugin.quantization.yaqa_kernels import packed_yaqa_linear

    layer = _LAYERS.get(key)
    if layer is None or layer.shape[0] != m:
        raise RuntimeError("Opaque YAQA layer must be registered before execution")
    if not buffers or x.device.type != "cuda":
        raise ValueError("Opaque YAQA dispatch needs actual CUDA layer buffers")
    rows = x.numel() // x.shape[-1]
    # These count Python dispatcher invocations/captures, not graph replays.
    _CALLS_BY_ROWS[rows] = _CALLS_BY_ROWS.get(rows, 0) + 1
    from .yaqa_prefill_cache import apply as apply_cache
    from .yaqa_prefill_cache import selected as cache_selected

    if cache_selected(x, layer):
        return apply_cache(x, layer)
    return packed_yaqa_linear(x, layer)


@opaque_linear.register_fake
def _fake(x, buffers, key, m):
    del buffers, key
    return x.new_empty((*x.shape[:-1], m))


def status():
    return {
        "registered_layers": len(_LAYERS),
        "Python_dispatch_invocations_by_token_rows": dict(_CALLS_BY_ROWS),
        "counts_include_capture_not_replay": True,
        "persistent_dense_weights_added": False,
    }
