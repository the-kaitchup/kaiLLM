# SPDX-License-Identifier: Apache-2.0
"""Optional ordinary-body Marlin dispatch over a separately prepared cache."""

import itertools
import weakref
from functools import lru_cache

MAX_DECODE_ROWS = 32
WARMUP_ROWS = (1, 4, 8, 16, 32)
_STATES = weakref.WeakValueDictionary()
_KEYS = itertools.count()
_marlin_apply = None
_WEIGHT_TYPE = None


def _phase_is_prefill():
    from vllm.forward_context import get_forward_context

    try:
        context = get_forward_context()
    except AssertionError:
        return False
    return getattr(getattr(context, "batch_descriptor", None), "is_prefill", None) is True


def selected(x, state, output_dtype=None):
    """Evaluate tensor/phase policy inside the opaque operation, never tracing."""
    import torch

    layer = state.original_ref()
    if (
        layer is None
        or layer.role != "body"
        or not layer.canonical_rounding
        or layer.dtype != torch.bfloat16
        or output_dtype not in (None, torch.bfloat16)
        or x.dtype != torch.bfloat16
        or x.device != state.device
        or x.device.type != "cuda"
        or x.ndim < 1
        or x.shape[-1] != state.k
        or not x.is_contiguous()
        or state.plan.admitted is not True
        or not state.plan.ready
    ):
        return False
    rows = x.numel() // state.k
    if not 1 <= rows <= MAX_DECODE_ROWS or _phase_is_prefill():
        return False
    # Existing canonical body prefill storage retains its original priority.
    if getattr(layer, "body_prefill_weight", None) is not None:
        from .shared_body_prefill_cache import selected as selected_prefill

        if selected_prefill(x, layer):
            return False
    return True


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
        None,  # Existing stream/ubatch/lane workspace manager, warmed at bind.
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

    @torch.library.custom_op("vllm_gguf::ordinary_marlin_physical", mutates_args=())
    def op(
        x: torch.Tensor,
        buffers: list[torch.Tensor],
        key: str,
        output_width: int,
        bias: torch.Tensor | None,
        output_dtype: torch.dtype | None,
    ) -> torch.Tensor:
        state = _STATES.get(key)
        if state is None or state.n != output_width:
            raise RuntimeError("Ordinary Marlin state must belong to a live layer")
        if not selected(x, state, output_dtype):
            return state.fallback(x, bias, output_dtype=output_dtype)
        output = _native(x, state, *buffers[:3])
        return output if bias is None else output + bias

    @op.register_fake
    def fake(x, buffers, key, output_width, bias, output_dtype):
        dtype = x.dtype if output_dtype is None else output_dtype
        if bias is not None:
            dtype = torch.promote_types(dtype, bias.dtype)
        return x.new_empty((*x.shape[:-1], output_width), dtype=dtype)

    return op


def bind(layer, state):
    """Bind original fallback and existing native API once before capture."""
    if state is not getattr(layer, "_ordinary_marlin_physical", None):
        raise ValueError("Ordinary Marlin state is not owned by the selected layer")
    if state.key is not None or state.original_ref() is not layer:
        raise ValueError("Ordinary Marlin state must bind exactly once")
    if not state.plan.ready or state.plan.admitted is not True:
        raise ValueError("Bind only after the complete ordinary cache is ready")
    import torch
    from vllm.model_executor.layers.quantization.utils import marlin_utils
    from vllm.scalar_type import scalar_types

    global _marlin_apply, _WEIGHT_TYPE
    _marlin_apply = marlin_utils.apply_gptq_marlin_linear
    _WEIGHT_TYPE = scalar_types.uint8b128
    with torch.cuda.device(state.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Bind ordinary Marlin outside graph capture")
    seen, frozen = set(), []
    for value in (*state.buffers, *layer.buffers()):
        if id(value) not in seen:
            seen.add(id(value))
            frozen.append(value)
    state.buffers = tuple(frozen)
    layer_ref = weakref.ref(layer)
    original_forward = type(layer)._forward_original

    def fallback(x, bias, *, output_dtype=None):
        owner = layer_ref()
        if owner is None:
            raise RuntimeError("Ordinary GGUF fallback layer has been released")
        return original_forward(owner, x, bias, output_dtype=output_dtype)

    state.fallback = fallback
    state.key = f"{state.device}:{state.k}:{state.n}:{next(_KEYS)}"
    _STATES[state.key] = state
    _registered_op()
    with torch.inference_mode(), torch.cuda.device(state.device):
        for rows in WARMUP_ROWS:
            sample = torch.zeros((rows, state.k), device=state.device, dtype=torch.bfloat16)
            _native(sample, state, *state.buffers[:3])


def apply_prepared(x, state, bias=None, *, output_dtype=None):
    """Trace only the existing opaque operation and frozen buffer arguments."""
    import torch

    return torch.ops.vllm_gguf.ordinary_marlin_physical(
        x, list(state.buffers), state.key, state.n, bias, output_dtype
    )
