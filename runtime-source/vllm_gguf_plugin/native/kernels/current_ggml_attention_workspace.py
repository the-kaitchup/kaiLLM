"""Opt-in incremental FP16 KV workspace for the native attention diagnostic.

Prefix caching must be disabled: active requests must have distinct first KV
blocks. A per-page generation distinguishes successive requests that reuse a
physical block. Each current request ordinal owns fixed contiguous storage.
"""

import weakref

import torch
import triton
import triton.language as tl


@triton.jit
def _update_workspace(
    cache,
    blocks,
    starts,
    lengths,
    keys,
    values,
    owners,
    page_generations,
    slot_generations,
    valid_lengths,
    cleared_lengths,
    stats,
    totals,
    STRIDE_B: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_T: tl.constexpr,
    STRIDE_D: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CAPACITY: tl.constexpr,
    COMMON_PADDED: tl.constexpr,
    TILE_TOKENS: tl.constexpr,
):
    # One program owns all data and metadata for one request. Updating owner or
    # valid length cannot race with another program deciding whether to refill.
    request = tl.program_id(0)
    count = tl.load(starts + request + 1) - tl.load(starts + request)
    length = tl.load(lengths + request)
    active = (count > 0) & (length > 0)
    owner = tl.load(blocks + request * TABLE_STRIDE, mask=active, other=-1)
    previous_owner = tl.load(owners + request)
    previous_length = tl.load(valid_lengths + request)
    previous_generation = tl.load(slot_generations + request)
    generation = tl.load(page_generations + owner, mask=active, other=-1)
    # Prefix caching is disabled, so distinct active requests cannot touch the
    # same counter. Increment on the first prompt chunk, even when the physical
    # page and length happen to match a stale cache at another request ordinal.
    if active & (length == count):
        generation = tl.atomic_add(page_generations + owner, 1, sem="relaxed") + 1
    append = (
        active
        & (owner == previous_owner)
        & (generation == previous_generation)
        & (previous_length == length - count)
    )
    copy_start = tl.where(append, previous_length, 0)
    copy_stop = tl.where(active, length, 0)
    padded = tl.maximum(tl.maximum(256, tl.cdiv(length, 256) * 256), COMMON_PADDED)
    previous_padded = tl.load(cleared_lengths + request)
    clear_start = tl.where(append, tl.maximum(length, previous_padded), length)
    clear_start = tl.where(active, clear_start, 0)
    clear_stop = tl.where(
        active,
        padded,
        tl.where((previous_owner == -1) & (previous_padded >= padded), 0, padded),
    )

    offsets = tl.arange(0, TILE_TOKENS * 512)
    local_token = offsets // 512
    head, dim = offsets % 512 // 256, offsets % 256
    for first in range(copy_start, copy_stop, TILE_TOKENS):
        token = first + local_token
        valid = token < copy_stop
        block = tl.load(
            blocks + request * TABLE_STRIDE + token // BLOCK_SIZE, mask=valid, other=0
        ).to(tl.int64)
        address = (
            block * STRIDE_B
            + head * STRIDE_H
            + token % BLOCK_SIZE * STRIDE_T
            + dim * STRIDE_D
        )
        k = tl.load(cache + address, mask=valid, other=0).to(tl.float16)
        v = tl.load(cache + address + 256 * STRIDE_D, mask=valid, other=0).to(
            tl.float16
        )
        destination = (request * CAPACITY + token) * 512 + head * 256 + dim
        tl.store(keys + destination, k, valid)
        tl.store(values + destination, v, valid)
    for first in range(clear_start, clear_stop, TILE_TOKENS):
        token = first + local_token
        destination = (request * CAPACITY + token) * 512 + head * 256 + dim
        tl.store(keys + destination, 0.0, token < clear_stop)
        tl.store(values + destination, 0.0, token < clear_stop)
    tl.store(owners + request, tl.where(active, owner, -1))
    tl.store(slot_generations + request, tl.where(active, generation, -1))
    tl.store(valid_lengths + request, tl.where(active, length, 0))
    tl.store(
        cleared_lengths + request,
        tl.where(append | ~active, tl.maximum(padded, previous_padded), padded),
    )
    tl.store(stats + request * 3, copy_stop - copy_start)
    tl.store(stats + request * 3 + 1, tl.maximum(0, clear_stop - clear_start))
    tl.store(stats + request * 3 + 2, active & ~append)
    metric = tl.arange(0, 4)
    increments = tl.where(
        metric == 0,
        copy_stop - copy_start,
        tl.where(
            metric == 1,
            tl.maximum(0, clear_stop - clear_start),
            tl.where(metric == 2, active & ~append, append),
        ),
    )
    old = tl.load(totals + request * 4 + metric)
    tl.store(totals + request * 4 + metric, old + increments)


class PersistentNativeKV:
    """Fixed-pointer layer storage; callers own stream ordering and scope."""

    def __init__(self, requests, context, device, cache_blocks):
        self.requests = requests
        self.cache_blocks = cache_blocks
        self.context = triton.cdiv(context, 256) * 256
        self.keys = torch.empty(
            (requests, self.context, 2, 256), device=device, dtype=torch.float16
        )
        self.values = torch.empty_like(self.keys)
        # -2 means never initialized; -1 means an initialized inactive slot.
        self.owners = torch.full((requests,), -2, device=device, dtype=torch.int32)
        self.page_generations = torch.zeros(
            cache_blocks, device=device, dtype=torch.int64
        )
        self.slot_generations = torch.full(
            (requests,), -1, device=device, dtype=torch.int64
        )
        self.valid_lengths = torch.zeros(requests, device=device, dtype=torch.int32)
        self.cleared_lengths = torch.zeros(requests, device=device, dtype=torch.int32)
        self.last_update = torch.zeros((requests, 3), device=device, dtype=torch.int32)
        self.totals = torch.zeros((requests, 4), device=device, dtype=torch.int64)
        self._cache_ref = None
        self._cache_identity = None
        self.rebind_count = 0

    def bind_cache(self, cache):
        # vLLM executes attention against a minimal profiling cache before
        # allocating the real cache. Keep the large FP16 buffers, but never keep
        # a strong reference that would prevent releasing the profiling cache.
        identity = (cache.data_ptr(), tuple(cache.shape), cache.stride(), cache.device)
        allocation_expired = self._cache_ref is not None and self._cache_ref() is None
        if self._cache_identity is not None and (
            identity != self._cache_identity or allocation_expired
        ):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Cannot rebind native KV during CUDA graph capture")
            self.owners.fill_(-2)
            self.slot_generations.fill_(-1)
            self.valid_lengths.zero_()
            self.cleared_lengths.zero_()
            self.last_update.zero_()
            self.totals.zero_()
            if cache.shape[0] != self.cache_blocks:
                self.page_generations = torch.zeros(
                    cache.shape[0], device=cache.device, dtype=torch.int64
                )
            else:
                self.page_generations.zero_()
            self.cache_blocks = cache.shape[0]
            self.rebind_count += 1
        elif cache.shape[0] != self.cache_blocks:
            raise ValueError("Initial native KV allocation differs from construction")
        self._cache_ref = weakref.ref(cache)
        self._cache_identity = identity

    def update(self, cache, blocks, starts, lengths, common_padded=0):
        requests = lengths.shape[0]
        if requests > self.requests:
            raise ValueError("Native KV workspace request capacity exceeded")
        self.bind_cache(cache)
        if common_padded > self.context or common_padded % 256:
            raise ValueError("Invalid common native KV padding")
        _update_workspace[(requests,)](
            cache,
            blocks,
            starts,
            lengths,
            self.keys,
            self.values,
            self.owners,
            self.page_generations,
            self.slot_generations,
            self.valid_lengths,
            self.cleared_lengths,
            self.last_update,
            self.totals,
            *cache.stride(),
            blocks.stride(0),
            cache.shape[2],
            self.context,
            common_padded,
            4,
            num_warps=4,
        )
        return self.keys[:requests], self.values[:requests]
