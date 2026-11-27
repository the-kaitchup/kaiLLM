"""Exact-row verification for qualified single-request Inkling DSpark setups on DGX Spark."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from functools import lru_cache
import hashlib
import os
from pathlib import Path

import torch

from ..dspine import INKLING_ARCHS


SHA256 = 'd2e18b505ed0885e620e4f54c8d7d9734ecdaf5d07c857a32aca069043a862d1'
CHIMACHIMA_XS_SHA256 = '4e71fe95f7dbb05fd349b8b1f1000ec426bb81dc0366252e5bca8c513e32dd5a'
CHIMACHIMA_S_SHA256 = '62a97eef7ac458cb7831df31fe526ee42ce88b4318abcb2dc2b94b57208a6655'
ROWS = range(2, 7)
_active = ContextVar('inkling_spark_router_verify', default=False)


@lru_cache(maxsize=2)
def _hash(path, inode, size, mtime_ns):
    with Path(path).open('rb') as reader:
        return hashlib.file_digest(reader, 'sha256').hexdigest()


def supported(config, device):
    if os.environ.get('KAILLM_SPARK_DSPARK5_ROW_ROUTER') != '1':
        return False
    model, spec = config.model_config, config.speculative_config
    weights = getattr(model, 'model_weights', None) or getattr(model, 'model', None)
    draft = getattr(spec, 'draft_model_config', None)
    if not (spec and spec.method == 'dflash' and config.num_speculative_tokens in (3, 5)
            and set(INKLING_ARCHS).intersection(getattr(draft, 'architectures', ()) or ())
            and config.parallel_config.world_size_across_dp == 1 and config.scheduler_config.max_num_seqs == 1
            and config.scheduler_config.max_num_batched_tokens == 2048 and model.max_model_len == 73728
            and model.dtype == torch.bfloat16 and getattr(model.hf_config, 'model_type', None) == 'inkling2_loop'
            and str(config.cache_config.cache_dtype) in ('auto', 'bfloat16') and torch.cuda.device_count() == 1
            and torch.cuda.get_device_name(device) == 'NVIDIA GB10' and torch.cuda.get_device_capability(device) == (12,1)
            and isinstance(weights, (str, Path))):
        return False
    p = Path(weights)
    if p.name == 'Inkling-2-Mini-dondon-Q2_K_XS.gguf' and config.num_speculative_tokens == 5:
        expected = SHA256
    elif (p.name == 'Inkling-2-Mini-chimachima-kaiPTQ-Q2_K_XS.gguf'
          and config.num_speculative_tokens == 5
          and os.environ.get('KAILLM_SPARK_DSPARK5_CHIMACHIMA_XS') == '1'
          and os.environ.get('KAILLM_SPARK_DSPARK5_MIXTURE_FMA4') != '1'):
        expected = CHIMACHIMA_XS_SHA256
    elif (p.name == 'Inkling-2-Mini-chimachima-kaiPTQ-Q3_K_S.gguf'
          and os.environ.get('KAILLM_SPARK_DSPARK5_CHIMACHIMA_S') == '1'
          and os.environ.get('KAILLM_SPARK_DSPARK5_MIXTURE_FMA4') != '1'):
        expected = CHIMACHIMA_S_SHA256
    else:
        return False
    try:
        info = p.stat()
        return _hash(str(p), info.st_ino, info.st_size, info.st_mtime_ns) == expected
    except OSError:
        return False


def capture_twins(captures, verify_width):
    """Preserve the selected padded shape and leave normal candidate lookup unchanged."""
    twins = {d: replace(d, inkling_router_verify=True) for d in captures
             if 2 <= d.num_tokens <= verify_width and d.num_reqs == 1 and d.num_ubatches == 1 and not d.num_active_loras}
    if verify_width == 4 and {d.num_tokens for d in twins} != {2, 3, 4}:
        raise ValueError('Spark DSpark-3 requires captured verifier widths 2 through 4')
    captures.extend(twins.values())
    captures.sort(key=lambda d: d.num_tokens, reverse=True)
    return twins


def real_verify_width(state):
    if state is None or state.has_prefill or len(state.req_ids) != 1 or state.num_draft_tokens_np is None:
        return False
    drafts = int(state.num_draft_tokens_np[0])
    width = int(state.num_scheduled_tokens[0])
    return width in ROWS and drafts == width - 1 and state.num_tokens == width


@contextmanager
def capture():
    token = _active.set(True)
    try:
        yield
    finally:
        _active.reset(token)


def active():
    return _active.get()


@torch.compiler.disable
def project(x, router):
    """Issue a discrete ATen mm per row; Inductor may compile only the unchanged normalization."""
    if x.ndim != 2 or x.shape[0] not in ROWS or x.shape[1] != 2304 or router.shape != (128, 2304):
        raise ValueError('Unexpected Spark DSpark5 target router shape')
    if x.dtype != torch.float32 or router.dtype != torch.float32:
        raise ValueError('The Spark DSpark verifier requires the original FP32 router and hidden state')
    logits = x.new_empty((x.shape[0], router.shape[0]))
    transpose = router.t()
    for row in range(x.shape[0]):
        torch.mm(x[row:row+1], transpose, out=logits[row:row+1])
    return logits


def authenticate_mixture_scales(model, config, device):
    """Bind each routed down scale to the exact same layer in the published target."""
    if not supported(config, device):
        return
    import gguf
    from vllm.logger import init_logger
    source = getattr(config.model_config, 'model_weights', None) or config.model_config.model
    paths = [Path(path).resolve() for path in model.gguf_paths]
    if paths != [Path(source).resolve()]:
        raise ValueError('Cannot authenticate mixture scales from a different GGUF')
    reader = gguf.GGUFReader(str(paths[0]))
    directory = {t.name: t for t in reader.tensors}
    checked = 0
    for index, layer in enumerate(model.model.layers):
        stage = layer.experts.down
        scale = getattr(stage, 'scale', None)
        field = directory.get(f'blk.{index}.ffn_down_exps.scale')
        if field is None and scale is None:
            continue
        if field is None or scale is None or int(field.tensor_type) != 0 or tuple(field.shape) != (128,):
            raise ValueError(f'Invalid FP32 expert down scale at layer {index}')
        original = field.data.tobytes()
        if index == 0 and hashlib.sha256(original).hexdigest() != 'ebde1afa69dd69e567cb895a5ea80e6d81334292a33be3cd1d46d782c10df45b':
            raise ValueError('The published layer-zero expert down scale changed')
        mark_mixture_scale(scale, original)
        checked += 1
    if not checked:
        raise ValueError('Missing original FP32 expert scales')
    init_logger("vllm.inkling.spark_verify_router").info("Authenticated %d GB10 expert mixture scale tensors against the published target", checked)


def mark_mixture_scale(scale, original):
    if scale.dtype != torch.float32 or tuple(scale.shape) != (128,) or not scale.is_contiguous():
        raise ValueError('Invalid loaded mixture scale layout')
    if len(original) != 512 or scale.detach().cpu().numpy().tobytes() != original:
        raise ValueError('Loaded expert down weights differ from the original GGUF FP32 scale')
    scale._inkling_spark_verified_mixture_scale = True


def mixture_supported(y, weights, scale, ids, out_dtype):
    return (active() and getattr(scale, '_inkling_spark_verified_mixture_scale', False)
            and y.is_cuda and y.dtype == out_dtype == torch.float32 and weights.dtype == torch.float32
            and ids.dtype == torch.int32 and weights.shape[0] in ROWS and weights.shape[1:] == (8,)
            and y.shape == (weights.shape[0] * 8, 2304) and ids.shape == weights.shape
            and y.device == weights.device == ids.device == scale.device
            and y.is_contiguous() and weights.is_contiguous() and ids.is_contiguous())
