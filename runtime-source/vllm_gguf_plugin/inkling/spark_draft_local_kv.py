# SPDX-License-Identifier: Apache-2.0
"""Copy the committed window and the uncommitted query block into dense KV."""

import torch
import triton
import triton.language as tl


@triton.jit
def _window(
    POS, PAGES, CACHE, KEY, VALUE, OUT_KEY, OUT_VALUE, KEEP,
    WINDOW:tl.constexpr, QUERIES:tl.constexpr, PAGE:tl.constexpr, HEADS:tl.constexpr, WIDTH:tl.constexpr,
    POS_S:tl.constexpr, TABLE_S:tl.constexpr, CACHE_P:tl.constexpr, CACHE_H:tl.constexpr,
    CACHE_T:tl.constexpr, CACHE_D:tl.constexpr, KEY_T:tl.constexpr, KEY_H:tl.constexpr, KEY_D:tl.constexpr,
    VALUE_T:tl.constexpr, VALUE_H:tl.constexpr, VALUE_D:tl.constexpr, SIZE:tl.constexpr,
):
    entries:tl.constexpr = WINDOW + QUERIES
    q_stride:tl.constexpr = HEADS * WIDTH
    index = tl.program_id(0) * SIZE + tl.arange(0,SIZE)
    request = tl.program_id(1)
    column = index % WIDTH
    head = index // WIDTH % HEADS
    token = index // q_stride
    anchor = tl.load(POS + request * QUERIES * POS_S)
    position = anchor - WINDOW + token
    history = (position >= 0) & (token < WINDOW)
    page = tl.load(PAGES + request * TABLE_S + position // PAGE, history, 0).to(tl.int64)
    slot = position % PAGE
    old = CACHE + page * CACHE_P + head * CACHE_H + slot * CACHE_T + column * CACHE_D
    k0 = tl.load(old, history, 0)
    v0 = tl.load(old + WIDTH * CACHE_D, history, 0)
    query = request * QUERIES + token - WINDOW
    current = (token >= WINDOW) & (token < entries)
    k1 = tl.load(KEY + query * KEY_T + head * KEY_H + column * KEY_D, current, 0)
    v1 = tl.load(VALUE + query * VALUE_T + head * VALUE_H + column * VALUE_D, current, 0)
    output = request * entries * q_stride + index
    tl.store(OUT_KEY + output,tl.where(current,k1,k0),token < entries)
    tl.store(OUT_VALUE + output,tl.where(current,v1,v0),token < entries)
    tl.store(KEEP + request * entries + token,history | current,(head == 0) & (column == 0) & (token < entries))


def prepare_window(k,v,positions,block_table,cache,window,block_length,tile=256):
    rows,heads,width = k.shape
    requests = rows // block_length
    length = window + block_length
    shape = (requests,length,heads,width)
    keys = k.new_empty(shape)
    values = v.new_empty(shape)
    keep = torch.empty((requests,length),device=k.device,dtype=torch.bool)
    assert rows == requests * block_length and positions.numel() == rows
    assert k.dtype == v.dtype == cache.dtype and block_table.stride(1) == 1
    assert cache.ndim == 4 and cache.shape[1] == heads and cache.shape[-1] == 2 * width
    _window[(triton.cdiv(length*heads*width,tile),requests)](
        positions,block_table,cache,k,v,keys,values,keep,
        window,block_length,cache.shape[2],heads,width,positions.stride(0),block_table.stride(0),
        *cache.stride(),*k.stride(),*v.stride(),tile,num_warps=4 if tile>128 else 2,
    )
    return keys,values,keep
