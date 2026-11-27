# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 attention for kaiLLM: one paged KV cache per layer visit (48 cache slots for 24 physical layers),
relative-position bias + per-query sink (llama.cpp semantics, see ref.attention), sliding window on the SWA layers.

Full-attention slots are FullAttentionSpec layers, sliding ones SlidingWindowSpec layers (vLLM frees the blocks
below the window; the kernels never read them). Decode-only batches replay as full CUDA graphs. The cache view is
vLLM's logical [blocks, kv_heads, block_size, K | V] page (strides set by the KV layout).

Implementations (KAILLM_INKLING_ATTN = auto | torch | triton): torch = gather + softmax per request (reference,
CPU and GPU); triton = paged kernel with the bias computed on the fly (inkling/triton_attention.py).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn

IMPL = os.environ.get("KAILLM_INKLING_ATTN", "auto")


# ----------------------------------------------------------------------------------------------------------------
# torch reference over the paged cache

_UNIT_SCALE: dict = {}


def write_kv(cache: torch.Tensor, k: torch.Tensor, v: torch.Tensor, slot_mapping: torch.Tensor) -> None:
    """cache [B, Hkv, bs, 2D] (logical view), k / v [T, Hkv, D], slot_mapping [>= T] (-1 = padding).

    CUDA: vLLM's reshape_and_cache_flash on the K and V halves of the page (strided heads, padded slots skipped, no
    host synchronisation: CUDA-graph safe); FP8 caches take float32 rows (one E4M3 rounding, saturating)."""
    T = k.shape[0]
    D = k.shape[-1]
    if cache.is_cuda and slot_mapping.dtype == torch.int64:
        from vllm import _custom_ops as ops
        page = cache.transpose(1, 2)                                   # [B, bs, Hkv, 2D]
        fp8 = cache.dtype in (torch.float8_e4m3fn, torch.uint8)
        src_dtype = torch.float32 if fp8 else cache.dtype
        one = _UNIT_SCALE.get(cache.device)
        if one is None:
            one = _UNIT_SCALE[cache.device] = torch.ones(1, dtype=torch.float32, device=cache.device)
        ops.reshape_and_cache_flash(k.to(src_dtype).contiguous(), v.to(src_dtype).contiguous(), page[..., :D],
                                    page[..., D:], slot_mapping[:T], "fp8_e4m3" if fp8 else "auto", one, one)
        return
    bs = cache.shape[2]
    s = slot_mapping[:T].long()
    ok = s >= 0
    if not bool(ok.all()):
        s, k, v = s[ok], k[ok], v[ok]
    blk, off = s // bs, s % bs
    cache[blk, :, off, :D] = k.to(cache.dtype)
    cache[blk, :, off, D:] = v.to(cache.dtype)


def read_kv(cache: torch.Tensor, block_row: torch.Tensor, start: int, end: int, D: int):
    """Keys / values [end - start, Hkv, D] at positions [start, end) of one request (block_row: its block table)."""
    bs = cache.shape[2]
    pos = torch.arange(start, end, device=cache.device)
    blk = block_row[pos // bs].long()
    kv = cache[blk, :, pos % bs, :]                       # [P, Hkv, 2D]
    return kv[..., :D], kv[..., D:]


def rel_attention_torch(q, rel, sink, bias, positions, cache, md, scale: float, window: int | None, extent: int):
    """q [T, H, D], rel [T, H, n_rel], sink [T, H], bias [extent, n_rel] -> [T, H, D] (float32 math)."""
    from .ref import attention
    T, H, D = q.shape
    out = q.new_empty(T, H, D)
    qsl = md.query_start_loc_cpu.tolist()
    seq = (md.seq_lens_cpu if md.seq_lens_cpu is not None else md.seq_lens.cpu()).tolist()
    for r in range(md.num_reqs):
        a, b = qsl[r], qsl[r + 1]
        if b <= a:
            continue
        L = seq[r]
        qpos = positions[a:b]
        first = 0 if window is None else max(0, int(qpos[0]) - window + 1)
        k, v = read_kv(cache, md.block_table[r], first, L, D)
        kpos = torch.arange(first, L, device=q.device)
        out[a:b] = attention(q[a:b], k, v, rel[a:b], bias, sink[a:b], qpos, kpos, scale, window,
                             extent).to(out.dtype)
    return out


# ----------------------------------------------------------------------------------------------------------------
# vLLM backend (metadata only: the model calls the kernels itself)

@dataclass
class InklingAttnMetadata:
    num_reqs: int
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor | None
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    tiles: torch.Tensor | None = None        # decode-only batches: persistent (request, 0) tiles (CUDA graphs)
    verify_width: int = 1
    one_pass_decode: bool = False


def _backend_classes():
    from vllm.config import VllmConfig
    from vllm.v1.attention.backend import (AttentionBackend, AttentionCGSupport, AttentionImplBase,
                                           AttentionMetadataBuilder, CommonAttentionMetadata)
    from vllm.v1.kv_cache_interface import AttentionSpec

    class InklingAttnMetadataBuilder(AttentionMetadataBuilder[InklingAttnMetadata]):
        # The torch reference reads metadata on the host. The Triton path computes
        # prefill tiles during replay from the runner's persistent query offsets.
        _cudagraph_support: ClassVar[AttentionCGSupport] = (
            AttentionCGSupport.ALWAYS if IMPL in ("auto", "triton") else AttentionCGSupport.NEVER
        )

        def __init__(self, kv_cache_spec: AttentionSpec, layer_names, vllm_config: VllmConfig, device):
            self.kv_cache_spec = kv_cache_spec
            self.max_verify_width = vllm_config.num_speculative_tokens + 1
            n = max(int(vllm_config.scheduler_config.max_num_seqs), 1)
            self.decode_tiles = torch.stack([torch.arange(n, dtype=torch.int32), torch.zeros(n, dtype=torch.int32)],
                                            dim=1).to(device)

        def build(self, common_prefix_len, common_attn_metadata: CommonAttentionMetadata, fast_build=False):
            m = common_attn_metadata
            seq_cpu = getattr(m, "seq_lens_cpu_upper_bound", None)
            if seq_cpu is None and IMPL == "torch":
                seq_cpu = m.seq_lens.cpu()
            tiles = None
            if m.max_query_len <= 1 and m.num_reqs <= self.decode_tiles.shape[0]:
                tiles = self.decode_tiles[:max(m.num_reqs, 1)]
            return InklingAttnMetadata(
                num_reqs=m.num_reqs, num_actual_tokens=m.num_actual_tokens, max_query_len=m.max_query_len,
                max_seq_len=m.max_seq_len, query_start_loc=m.query_start_loc,
                query_start_loc_cpu=m.query_start_loc_cpu, seq_lens=m.seq_lens, seq_lens_cpu=seq_cpu,
                block_table=m.block_table_tensor, slot_mapping=m.slot_mapping, tiles=tiles,
                verify_width=m.max_query_len if 1 < m.max_query_len <= self.max_verify_width else 1)

    class InklingAttnImpl(AttentionImplBase):
        def __init__(self, *args, **kwargs):
            pass

    class InklingAttnBackend(AttentionBackend):
        supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16, torch.float32]
        supported_kv_cache_dtypes: ClassVar[list] = ["auto", "float16", "bfloat16", "float32", "fp8", "fp8_e4m3"]

        @staticmethod
        def get_name() -> str:
            return "INKLING_REL"

        @staticmethod
        def get_builder_cls():
            return InklingAttnMetadataBuilder

        @staticmethod
        def get_impl_cls():
            return InklingAttnImpl

        @classmethod
        def get_supported_head_sizes(cls) -> list[int]:
            return []

        @classmethod
        def supports_head_size(cls, head_size: int) -> bool:
            return True

        @classmethod
        def supports_compute_capability(cls, capability) -> bool:
            return True

        @classmethod
        def supports_sink(cls) -> bool:
            return True

    return InklingAttnBackend


_BACKEND = None


def inkling_attn_backend():
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = _backend_classes()
    return _BACKEND


class InklingKVCache(nn.Module):
    """KV cache owner of one layer visit (vLLM AttentionLayerBase; name model.slots.<slot>.attn)."""

    def __init__(self, num_kv_heads: int, head_dim: int, dtype: torch.dtype, prefix: str, cache_config,
                 window: int | None = None):
        super().__init__()
        self.window = window if (window and os.getenv("KAILLM_INKLING_SWA_CACHE", "1") != "0") else None
        from vllm.config import get_current_vllm_config
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
        if AttentionLayerBase not in type(self).__mro__:
            raise TypeError("use make_kv_cache()")
        self.kv_cache = torch.tensor([])
        self.num_kv_heads, self.head_dim, self.dtype = num_kv_heads, head_dim, dtype
        self.prefix, self.cache_config = prefix, cache_config
        ctx = get_current_vllm_config().compilation_config.static_forward_context
        if prefix in ctx:
            raise ValueError(f"Duplicate layer name: {prefix}")
        ctx[prefix] = self

    def get_kv_cache_spec(self, vllm_config):
        """Sliding layers: a SlidingWindowSpec (vLLM releases blocks below the window; the kernels never read them)."""
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
        if self.window:
            return SlidingWindowSpec(block_size=self.cache_config.block_size, num_kv_heads=self.num_kv_heads,
                                     head_size=self.head_dim, dtype=self.dtype, sliding_window=self.window)
        return FullAttentionSpec(block_size=self.cache_config.block_size, num_kv_heads=self.num_kv_heads,
                                 head_size=self.head_dim, dtype=self.dtype)

    def get_attn_backend(self):
        return inkling_attn_backend()

    def metadata(self):
        from vllm.forward_context import get_forward_context
        md = get_forward_context().attn_metadata
        return None if md is None else (md[self.prefix] if isinstance(md, dict) else md)


_KV_CLS = None


def make_kv_cache(*args, **kwargs) -> InklingKVCache:
    global _KV_CLS
    if _KV_CLS is None:
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
        _KV_CLS = type("InklingKVCacheLayer", (InklingKVCache, AttentionLayerBase), {})
    return _KV_CLS(*args, **kwargs)


def attend(cache_layer: InklingKVCache, q, k, v, rel, sink, bias, positions, scale, window, extent):
    """Write k / v of the scheduled tokens into the slot's cache, then attend.  Without metadata (vLLM's profile
    run) the call attends within the batch as one sequence (the result is unused)."""
    md = cache_layer.metadata()
    if md is None:
        from .ref import attention
        pos = torch.arange(q.shape[0], device=q.device)
        return attention(q, k, v, rel, bias, sink, pos, pos, scale, window, extent).to(q.dtype)
    write_kv(cache_layer.kv_cache, k, v, md.slot_mapping)
    use_triton = q.is_cuda and IMPL in ("auto", "triton")
    if use_triton:
        from .triton_attention import rel_attention_triton
        return rel_attention_triton(q, rel, sink, bias, positions, cache_layer.kv_cache, md, scale, window, extent)
    return rel_attention_torch(q, rel, sink, bias, positions, cache_layer.kv_cache, md, scale, window, extent)


def triton_available() -> bool:
    try:
        from . import triton_attention  # noqa: F401
        return True
    except Exception:
        return False


__all__ = ["attend", "make_kv_cache", "inkling_attn_backend", "write_kv", "read_kv", "rel_attention_torch",
           "InklingAttnMetadata", "math"]
