# SPDX-License-Identifier: Apache-2.0
"""Optional small-row Marlin route over the prepared canonical physical cache."""

import weakref
from functools import lru_cache

_STATES = weakref.WeakValueDictionary()
_OCCURRENCES = {}
_marlin_apply = None
_WEIGHT_TYPE = None


def _phase_is_prefill():
    from vllm.forward_context import get_forward_context

    try:
        context = get_forward_context()
    except AssertionError:
        return False
    return getattr(getattr(context, "batch_descriptor", None), "is_prefill", None) is True


def selected(x, state):
    """Run inside the opaque op; unknown small-row phase remains eligible."""
    import torch

    if (
        x.dtype != torch.bfloat16
        or x.device != state.device
        or x.ndim < 1
        or x.shape[-1] != state.k
        or not x.is_contiguous()
        or state.plan.admitted is not True
        or not state.plan.ready
    ):
        return False
    rows = x.numel() // state.k
    if not 1 <= rows <= 4 or (rows == 1 and state.part_count > 1):
        return False
    if _phase_is_prefill():
        return False
    # Keep physical W8 requantization within its original QTIP4 domain.
    if not state.parts or any(getattr(part, "bits", None) != 4 for part in state.parts):
        return False
    # Retain the canonical QKV/small-cache and other existing cache priorities.
    from .yaqa_prefill_cache import selected as selected_cache

    return not any(selected_cache(x, part) for part in state.parts)


def _native(x, state, wire, scales, zeros):
    source = x.reshape(-1, state.k)
    if state.padded_k != state.k:
        import torch.nn.functional as functional

        source = functional.pad(source, (0, state.padded_k - state.k))
    output = _marlin_apply(
        source,
        wire,
        scales,
        zeros,
        None,  # Existing persistent stream/ubatch/lane workspace manager.
        _WEIGHT_TYPE,
        state.padded_n,
        state.padded_k,
        bias=None,
        use_fp32_reduce=True,
        input_dtype=None,
    )
    if state.padded_n != state.n:
        output = output[:, : state.n].contiguous()
    return output.reshape(*x.shape[:-1], state.n)


@lru_cache(maxsize=1)
def _registered_op():
    import torch

    @torch.library.custom_op("vllm_gguf::yaqa_marlin_physical", mutates_args=())
    def op(
        x: torch.Tensor,
        buffers: list[torch.Tensor],
        key: str,
        output_width: int,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        state = _STATES.get(key)
        if state is None or state.n != output_width:
            raise RuntimeError("Prepared Marlin state no longer belongs to a live layer")
        if not selected(x, state):
            return state.fallback(x, bias)
        output = _native(x, state, *buffers[:3])
        return output if bias is None else output + bias

    @op.register_fake
    def fake(x, buffers, key, output_width, bias):
        dtype = x.dtype if bias is None else torch.promote_types(x.dtype, bias.dtype)
        return x.new_empty((*x.shape[:-1], output_width), dtype=dtype)

    return op


def bind(layer, original_apply):
    """Freeze final native/fallback buffers and register before model capture."""
    state = getattr(layer, "_yaqa_marlin_physical", None)
    if state is None:
        return
    if state.key is not None:
        raise ValueError("Marlin runtime state must bind exactly once")
    import torch
    from vllm.model_executor.layers.quantization.utils import marlin_utils
    from vllm.scalar_type import scalar_types

    global _marlin_apply, _WEIGHT_TYPE
    _marlin_apply = marlin_utils.apply_gptq_marlin_linear
    _WEIGHT_TYPE = scalar_types.uint8b128
    with torch.cuda.device(state.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Bind Marlin runtime outside CUDA graph capture")
    # Retain every final original/grouped buffer as an opaque-op argument too.
    # Layer construction and existing grouped preparation have already finished.
    seen = set()
    frozen = []
    for value in (*state.buffers, *layer.buffers()):
        if id(value) not in seen:
            seen.add(id(value))
            frozen.append(value)
    state.buffers = tuple(frozen)
    layer_ref = weakref.ref(layer)

    def fallback(x, bias):
        owner = layer_ref()
        if owner is None:
            raise RuntimeError("Original Marlin fallback layer has been released")
        return original_apply(owner, x, bias)

    state.fallback = fallback
    identity = repr((state.part_keys, state.k, state.n, state.device))
    occurrence = _OCCURRENCES.get(identity, 0)
    _OCCURRENCES[identity] = occurrence + 1
    state.key = identity + ":" + str(occurrence)
    _STATES[state.key] = state
    _registered_op()
    # Compile/repack never occurs at inference. Existing manager allocation is
    # warmed here and by normal model warmups on each actual capture stream.
    with torch.inference_mode(), torch.cuda.device(state.device):
        for rows in (1, 4):
            sample = torch.zeros((rows, state.k), device=state.device, dtype=torch.bfloat16)
            _native(sample, state, *state.buffers[:3])


def apply_prepared(x, state, bias=None):
    """Trace only an existing registered op; phase/cache checks stay inside it."""
    import torch

    return torch.ops.vllm_gguf.yaqa_marlin_physical(
        x, list(state.buffers), state.key, state.n, bias
    )
