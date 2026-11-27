# SPDX-License-Identifier: Apache-2.0
"""Preserve singleton arithmetic in the qualified Chimachima XS DSpark graph."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
import hashlib
import os
from pathlib import Path


ENV = "KAILLM_INKLING_CHIMA_XS_ROW_STABLE_DS5"
_ACTIVE = ContextVar("inkling_chima_xs_verifier_rows", default=False)
_TARGET = "Inkling-2-Mini-chimachima-kaiPTQ-Q2_K_XS.gguf"
_TARGET_SHA256 = "4e71fe95f7dbb05fd349b8b1f1000ec426bb81dc0366252e5bca8c513e32dd5a"
_MAX_WIDTH = 6


def enabled():
    return os.getenv(ENV) == "1"


def validate(config, device):
    """Reject an explicit opt-in outside the verified placement."""
    import torch

    model = config.model_config
    spec = config.speculative_config
    name = getattr(model, "model_weights", None) or getattr(model, "model", None)
    card = torch.cuda.get_device_properties(device)
    if not (
        spec and spec.method == "dflash" and config.num_speculative_tokens == _MAX_WIDTH - 1
        and config.scheduler_config.max_num_seqs == 1
        and config.scheduler_config.max_num_batched_tokens == 512
        and config.parallel_config.world_size_across_dp == 1 and model.max_model_len == 73728
        and model.dtype == torch.bfloat16 and str(config.cache_config.cache_dtype) in ("auto", "bfloat16")
        and getattr(config.attention_config.backend, "name", None) == "TRITON_ATTN"
        and os.getenv("KAILLM_INKLING_ATTN", "auto") in ("auto", "triton")
        and getattr(model.hf_config, "model_type", None) == "inkling2_loop"
        and (card.major, card.minor, card.multi_processor_count) == (12, 0, 170)
        and card.name == "NVIDIA GeForce RTX 5090" and torch.cuda.device_count() == 1
        and isinstance(name, (str, Path)) and Path(name).name == _TARGET
    ):
        raise ValueError("Chimachima XS row-stable verification requires its qualified RTX 5090 placement")
    with Path(name).open("rb") as model_file:
        if hashlib.file_digest(model_file, "sha256").hexdigest() != _TARGET_SHA256:
            raise ValueError("Chimachima XS row-stable verification requires the qualified target weights")


def capture_twins(captures):
    twins = {
        desc: replace(desc, inkling_row_stable=True) for desc in captures
        if 2 <= desc.num_tokens <= _MAX_WIDTH and desc.num_reqs == 1
        and desc.num_ubatches == 1 and not desc.num_active_loras
    }
    if {desc.num_tokens for desc in twins} != set(range(2, _MAX_WIDTH + 1)):
        raise ValueError("Every DSpark verifier width requires a dedicated FULL graph capture")
    captures.extend(twins.values())
    captures.sort(key=lambda desc: desc.num_tokens, reverse=True)
    return twins


def real_width(state):
    if state is None or state.has_prefill or len(state.req_ids) != 1 or state.num_draft_tokens_np is None:
        return False
    count = int(state.num_scheduled_tokens[0])
    drafts = int(state.num_draft_tokens_np[0])
    return 2 <= count == drafts + 1 <= _MAX_WIDTH and state.num_tokens == count


@contextmanager
def capture():
    token = _ACTIVE.set(True)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def active():
    return _ACTIVE.get()


def projected(singleton, q, k, v, relative, q_clip, relative_clip, tau, config, kwargs):
    import torch

    rows = []
    for i in range(q.shape[0]):
        token = slice(i, i + 1)
        rows.append(singleton(q[token], k[token], v[token], relative[token], q_clip,
                              relative_clip, tau[token], *config, **kwargs))
    return tuple(torch.cat(values, dim=0) for values in zip(*rows, strict=True))


def ffn_pair(singleton, source, weight, eps, dtype):
    if not active():
        return singleton(source, weight, eps, dtype)
    import torch

    if (source.dtype == torch.float32 and dtype == torch.bfloat16 and source.ndim == 2
            and 2 <= source.shape[0] <= _MAX_WIDTH and source.shape[1] == 2304 and weight.shape == (2304,)):
        rows = [singleton(row.contiguous(), weight, eps, dtype) for row in source.split(1)]
        return tuple(torch.cat(values, dim=0) for values in zip(*rows, strict=True))
    return singleton(source, weight, eps, dtype)


def combine(original, source, weights, scale, ids, dtype, itq):
    if not active():
        return original(source, weights, scale, ids, dtype)
    import torch

    if (scale is None and source.dtype == weights.dtype == dtype == torch.float32
            and ids.dtype == torch.int32 and source.is_contiguous() and weights.is_contiguous()
            and weights.ndim == 2 and 2 <= weights.shape[0] <= _MAX_WIDTH and weights.shape[1] == 8
            and source.shape == (weights.shape[0] * 8, 2304) and ids.shape == weights.shape):
        if itq:
            # The ordinary ITQ path uses the one-token FP32 reduction.
            return torch.cat([original(rows, targets, scale, indices, dtype)
                              for rows, targets, indices in
                              zip(source.split(8), weights.split(1), ids.split(1), strict=True)])
        from . import mixture_balanced

        return mixture_balanced.combine(source, weights, block=256, warps=4)
    return original(source, weights, scale, ids, dtype)
