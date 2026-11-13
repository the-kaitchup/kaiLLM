// SPDX-License-Identifier: Apache-2.0 AND MIT
// GGML vecdot/reduction arithmetic: Copyright (c) 2023-2026 The ggml authors.
// See LLAMA_CPP_LICENSE.txt. Two active native M1 warp chains plus native zeros.
#pragma once
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/vecdotq.cuh"

namespace gguf_native_q4_m1 {
inline bool eligible(int cc, ggml_type type, int64_t n, int64_t k, int64_t m) {
    return cc == 860 && type == GGML_TYPE_Q4_K && n == 6144 && k == 1024 && m == 1;
}

__launch_bounds__(64, 1)
__global__ void folded(const block_q4_K * __restrict__ weights,
        const block_q8_1 * __restrict__ activation, float * __restrict__ output) {
    const int lane = threadIdx.x;
    const int row = 2 * blockIdx.x + threadIdx.y;
    float sums[4] = {0.0f};
    #pragma unroll
    for (int logical = 0; logical < 2; ++logical) {
        const int block = 2 * logical + lane / 16;
        sums[logical] += vec_dot_q4_K_q8_1(
            weights + row * 4 + block, activation + block * 8, 0, 2 * (lane % 16));
    }
    // Native M1 adds all four logical warp partials in this order, including
    // inactive warp2/warp3 zeros, before its lane XOR reduction.
    #pragma unroll
    for (int logical = 1; logical < 4; ++logical) sums[0] += sums[logical];
    const float value = warp_reduce_sum<32>(sums[0]);
    if (lane == 0) output[row] = value;
}

inline void run(const void * weights, const void * native_q8,
                float * output, cudaStream_t stream) {
    folded<<<3072, dim3(32, 2), 0, stream>>>(
        static_cast<const block_q4_K *>(weights),
        static_cast<const block_q8_1 *>(native_q8), output);
}
}  // namespace gguf_native_q4_m1
