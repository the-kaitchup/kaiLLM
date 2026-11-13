// SPDX-License-Identifier: MIT
// Adapted from llama.cpp ggml/src/ggml-cuda/gated_delta_net.cu and common.cuh.
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

// Diagnostic only: 128-wide Q/K/V, identical head counts, F32.
// Match native lane assignment, FMA opportunities, butterfly reduction, scalar
// decay placement, and output scaling. Compile with native compile_commands.
static __device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, 32);
    }
    return x;
}

__global__ void __launch_bounds__(128, 2) native_order_gdn_kernel(
        const float * q, const float * k, const float * v,
        const float * g, const float * beta, const float * initial_state,
        float * output, float * final_state, int heads, int tokens, float scale) {
    const int head = blockIdx.x;
    const int lane = threadIdx.x;
    const int col = blockIdx.z * blockDim.y + threadIdx.y;
    const int state_offset = head * 128 * 128 + col * 128;
    float state[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        state[r] = initial_state[state_offset + r * 32 + lane];
    }
    for (int token = 0; token < tokens; ++token) {
        const int offset = (token * heads + head) * 128;
        float key[4], query[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            key[r] = k[offset + r * 32 + lane];
            query[r] = q[offset + r * 32 + lane];
        }
        const float decay = expf(g[token * heads + head]);
        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            kv_shard += state[r] * key[r];
        }
        const float kv = warp_sum(kv_shard);
        const float delta = (v[offset + col] - decay * kv) * beta[token * heads + head];
        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            state[r] = decay * state[r] + key[r] * delta;
            attn_partial += state[r] * query[r];
        }
        const float attn = warp_sum(attn_partial);
        if (lane == 0) {
            output[offset + col] = attn * scale;
        }
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        final_state[state_offset + r * 32 + lane] = state[r];
    }
}

extern "C" int native_order_gdn_run(
        const float * q, const float * k, const float * v,
        const float * g, const float * beta, const float * initial_state,
        float * output, float * final_state, int heads, int tokens,
        uintptr_t stream_ptr) {
    if (heads <= 0 || tokens <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const float scale = 1.0f / sqrtf(128.0f);
    native_order_gdn_kernel<<<dim3(heads, 1, 32), dim3(32, 4), 0,
            reinterpret_cast<cudaStream_t>(stream_ptr)>>>(
        q, k, v, g, beta, initial_state, output, final_state, heads, tokens, scale);
    return static_cast<int>(cudaPeekAtLastError());
}

// Each warp still owns one value column and performs exactly the original
// sequential recurrence. Requests differ only in token range and state slot.
// Keep the original B1 kernel above as an independent regression oracle.
template<bool INDEXED_DECODE>
__global__ void __launch_bounds__(128, 2) native_order_gdn_batched_kernel(
        const float * q, const float * k, const float * v,
        const float * g, const float * beta, const float * initial_state,
        float * output, float * final_state, const int32_t * cu_seqlens,
        const int32_t * state_indices, int heads, int tokens, int state_slots,
        int64_t state_stride, float scale) {
    const int head = blockIdx.x;
    const int request = blockIdx.y;
    const int lane = threadIdx.x;
    const int col = blockIdx.z * blockDim.y + threadIdx.y;
    const int begin = cu_seqlens[request];
    const int end = cu_seqlens[request + 1];
    const int slot = INDEXED_DECODE ? state_indices[request] : request;
    // Slot zero is vLLM's reserved NULL_BLOCK_ID. Negative/out-of-range slots
    // and repeated padded cu_seqlens never read or write the shared cache.
    if (INDEXED_DECODE &&
            (slot <= 0 || slot >= state_slots || end <= begin || end - begin != 1)) {
        return;
    }
    const bool valid_range = begin >= 0 && end >= begin && end <= tokens;
    if (INDEXED_DECODE && !valid_range) return;
    const int64_t state_offset =
            int64_t(slot) * state_stride + int64_t(head) * 128 * 128 + col * 128;
    float state[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        state[r] = initial_state[state_offset + r * 32 + lane];
    }
    // Invalid prefill ranges safely preserve their initial state. The caller
    // supplies valid monotonic ranges; the guard also prevents an OOB access.
    for (int token = begin; valid_range && token < end; ++token) {
        const int64_t offset = (int64_t(token) * heads + head) * 128;
        float key[4], query[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            key[r] = k[offset + r * 32 + lane];
            query[r] = q[offset + r * 32 + lane];
        }
        const float decay = expf(g[int64_t(token) * heads + head]);
        float kv_shard = 0.0f;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            kv_shard += state[r] * key[r];
        }
        const float kv = warp_sum(kv_shard);
        const float delta =
                (v[offset + col] - decay * kv) * beta[int64_t(token) * heads + head];
        float attn_partial = 0.0f;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            state[r] = decay * state[r] + key[r] * delta;
            attn_partial += state[r] * query[r];
        }
        const float attn = warp_sum(attn_partial);
        if (lane == 0) output[offset + col] = attn * scale;
    }
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        final_state[state_offset + r * 32 + lane] = state[r];
    }
}

extern "C" int native_order_gdn_batched_run(
        const float * q, const float * k, const float * v,
        const float * g, const float * beta, const float * initial_state,
        float * output, float * final_state, const int32_t * cu_seqlens,
        const int32_t * state_indices, int heads, int tokens, int requests,
        int state_slots, int indexed_decode, int64_t state_stride,
        uintptr_t stream_ptr) {
    if (heads <= 0 || tokens < 0 || requests <= 0 || requests > 65535 ||
            state_slots <= 0 || state_stride < int64_t(heads) * 128 * 128 ||
            !cu_seqlens ||
            (indexed_decode && (!state_indices || initial_state != final_state)) ||
            (!indexed_decode && requests != state_slots)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const float scale = 1.0f / sqrtf(128.0f);
    const dim3 grid(heads, requests, 32);
    const dim3 block(32, 4);
    const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    if (indexed_decode) {
        native_order_gdn_batched_kernel<true><<<grid, block, 0, stream>>>(
            q, k, v, g, beta, initial_state, output, final_state, cu_seqlens,
            state_indices, heads, tokens, state_slots, state_stride, scale);
    } else {
        native_order_gdn_batched_kernel<false><<<grid, block, 0, stream>>>(
            q, k, v, g, beta, initial_state, output, final_state, cu_seqlens,
            nullptr, heads, tokens, state_slots, state_stride, scale);
    }
    return static_cast<int>(cudaPeekAtLastError());
}
