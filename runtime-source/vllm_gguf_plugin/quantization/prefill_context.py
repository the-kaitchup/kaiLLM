# SPDX-License-Identifier: Apache-2.0
"""CPU metadata semantics for cache routing; never inspect GPU tensor values."""

from numbers import Integral


def decode_query_limit(config=None):
    """Capture LFM-style standard verifier query width at model construction."""
    if config is None:
        try:
            from vllm.config import get_current_vllm_config_or_none
        except ImportError:
            return None
        config = get_current_vllm_config_or_none()
    if config is None:
        return None
    count = getattr(config, "num_speculative_tokens", None)
    if count is None:
        speculative = getattr(config, "speculative_config", None)
        count = getattr(speculative, "num_speculative_tokens", 0)
        if isinstance(speculative, dict):
            count = speculative.get("num_speculative_tokens", 0)
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
        return None
    return int(count) + 1


def _max_query_len(metadata):
    if isinstance(metadata, list):
        if len(metadata) != 1:
            return None  # Unbound microbatch index: preserve the original path.
        metadata = metadata[0]
    values = metadata.values() if isinstance(metadata, dict) else (metadata,)
    lengths = []
    for item in values:
        value = getattr(item, "max_query_len", None)
        if isinstance(value, Integral) and not isinstance(value, bool) and value > 0:
            lengths.append(int(value))
    return max(lengths) if lengths else None


def context_is_prefill(query_limit, context=None):
    """Use the explicit graph-key tag; only eager execution may infer metadata.

    `BatchDescriptor.is_prefill` must participate in the stock graph cache key.
    PIECEWISE without this tag is ambiguous and stays on the original route.
    The low-level physical GEMM remains an explicitly invoked prefill primitive.
    """
    if context is None:
        try:
            from vllm.forward_context import get_forward_context
        except ImportError:
            return False
        try:
            context = get_forward_context()
        except AssertionError:
            return False
    descriptor = getattr(context, "batch_descriptor", None)
    tagged = getattr(descriptor, "is_prefill", None)
    if type(tagged) is bool:
        return tagged
    mode = getattr(context, "cudagraph_runtime_mode", None)
    mode = getattr(mode, "name", mode)
    if mode in ("FULL", "PIECEWISE"):
        return False  # An existing captured graph must never freeze an inference.
    if getattr(descriptor, "uniform", False):
        return False
    if (
        isinstance(query_limit, bool)
        or not isinstance(query_limit, Integral)
        or query_limit < 1
    ):
        return False
    maximum = _max_query_len(getattr(context, "attn_metadata", None))
    return maximum is not None and maximum > query_limit
