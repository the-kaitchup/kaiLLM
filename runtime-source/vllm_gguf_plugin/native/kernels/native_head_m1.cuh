// SPDX-License-Identifier: Apache-2.0 AND MIT
// GGML dot/reduction arithmetic: Copyright (c) 2023-2026 The ggml authors.
// See LLAMA_CPP_LICENSE.txt. Exact M1 Q6_K vocabulary-head experiment.
#pragma once
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/vecdotq.cuh"

namespace gguf_native_head_m1 {
inline bool eligible(int cc, ggml_type type, int64_t n, int64_t k, int64_t m) {
    return cc == 860 && type == GGML_TYPE_Q6_K && m == 1 &&
        n == 248320 && k == 2048;
}

// Selected head_m1.cu folded<Q6_K,2,1,0>: each physical warp owns two
// independent rows and the same four logical-warp chains as native M1.
// Each chain keeps native block order; the four chains are added in native
// order before the unmodified warp reduction. The original weights/Q8 blocks
// are read directly, so there is no repacking or extra workspace.
__launch_bounds__(32, 1)
__global__ void q6_k2048_two_rows(
        const block_q6_K * __restrict__ weights,
        const block_q8_1 * __restrict__ activation,
        float * __restrict__ output) {
    constexpr int blocks = 8, rows = 2;
    const int lane = threadIdx.x, row = rows * blockIdx.x;
    float sums[4][rows] = {{0.0f}};
    #pragma unroll
    for (int logical = 0; logical < 4; ++logical) {
        #pragma unroll
        for (int block = logical; block < blocks; block += 4) {
            #pragma unroll
            for (int r = 0; r < rows; ++r) {
                sums[logical][r] += vec_dot_q6_K_q8_1(
                    weights + (row+r)*blocks + block,
                    activation + block*8, 0, lane);
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < rows; ++r) {
        #pragma unroll
        for (int logical = 1; logical < 4; ++logical) {
            sums[0][r] += sums[logical][r];
        }
        const float sum = warp_reduce_sum<32>(sums[0][r]);
        if (lane == r) output[row+r] = sum;
    }
}

inline void run(const void * weights, const void * activation, float * output,
                int64_t n, cudaStream_t stream) {
    q6_k2048_two_rows<<<n/2, 32, 0, stream>>>(
        static_cast<const block_q6_K *>(weights),
        static_cast<const block_q8_1 *>(activation), output);
}
} // namespace gguf_native_head_m1
