# SPDX-License-Identifier: Apache-2.0
"""Noncausal draft-block attention with a common left boundary at the anchor."""

import torch
from torch.nn import functional as F


def local_block_attention(q, k, v, positions, block_table, cache, window, block_length, *, prepare_window=None):
    """Read committed context from the runtime's [page, head, token, K|V] cache.

    Every query in a draft block sees all of that block and the same `window`
    tokens before its anchor. The uncommitted query K/V are never persisted.
    """
    rows, heads, dim = q.shape
    kv_heads = k.shape[1]
    requests = rows // block_length
    if requests * block_length != rows:
        raise ValueError('Inkling DSpark expects one complete query block per request')
    if (prepare_window is not None and requests == 1 and k.shape[1:] == (8, 128)
            and window == 1024 and block_length == 6 and cache.shape[2] in (16, 32, 64)
            and block_table.stride(1) == 1 and cache.dtype == k.dtype == torch.bfloat16):
        keys, values, keep = prepare_window(k, v, positions, block_table, cache, window, block_length, tile=512)
    else:
        anchors = positions.reshape(requests, block_length)[:, :1]
        indexes = anchors + torch.arange(-window, 0, device=q.device)
        valid = indexes >= 0
        indexes = indexes.clamp_min(0)
        page_size = cache.shape[2]
        pages = block_table[:requests].gather(1, indexes // page_size).long()
        slots = indexes % page_size
        history = cache.transpose(1, 2)[pages, slots]
        history = history.masked_fill(~valid[:, :, None, None], 0)
        context_k, context_v = history.split(dim, dim=-1)
        keys = torch.cat((context_k, k.reshape(requests, block_length, kv_heads, dim)), dim=1)
        values = torch.cat((context_v, v.reshape(requests, block_length, kv_heads, dim)), dim=1)
        keep = F.pad(valid, (0, block_length), value=True)
    # Keep GQA grouped without replicating the cached keys and values per query head.
    group = heads // kv_heads
    query = q.reshape(requests, block_length, kv_heads, group, dim).permute(0, 2, 1, 3, 4)
    query = query.reshape(requests, kv_heads, block_length * group, dim)
    out = F.scaled_dot_product_attention(query, keys.transpose(1, 2), values.transpose(1, 2),
                                         attn_mask=keep[:, None, None, :])
    return out.reshape(requests, kv_heads, block_length, group, dim).permute(0, 2, 1, 3, 4).reshape(rows, heads * dim)


def local_query(attention, hidden, positions, window, block_length):
    from vllm.forward_context import get_forward_context

    qkv, _ = attention.qkv_proj(hidden)
    q, k, v = qkv.split((attention.q_size, attention.kv_size, attention.kv_size), dim=-1)
    dim = attention.head_dim
    q = attention.q_norm(q.reshape(-1, attention.num_heads, dim)).reshape_as(q)
    k = attention.k_norm(k.reshape(-1, attention.num_kv_heads, dim)).reshape_as(k)
    q, k = attention.rotary_emb(positions, q, k)
    ctx = get_forward_context().attn_metadata
    if ctx is None:
        return attention.o_proj(torch.zeros_like(q))[0]
    layer = attention.attn
    metadata = ctx[layer.layer_name]
    out = local_block_attention(q.reshape(-1, attention.num_heads, dim), k.reshape(-1, attention.num_kv_heads, dim),
                                v.reshape(-1, attention.num_kv_heads, dim), positions, metadata.block_table,
                                layer.kv_cache, window, block_length,
                                prepare_window=getattr(attention, '_inkling_prepare_window', None))
    return attention.o_proj(out)[0]
