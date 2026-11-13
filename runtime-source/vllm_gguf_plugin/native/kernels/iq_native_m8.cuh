// SPDX-License-Identifier: Apache-2.0 AND MIT
// GGML codebook/decode/reduction portions: Copyright (c) 2023-2026 The ggml authors.
// See LLAMA_CPP_LICENSE.txt for the upstream notice and permission terms.
// Exact native-order M8 IQ kernels selected by the SM86 measurements.
// Original matrix boundaries and native Q8_1 activation blocks are preserved.
#pragma once
#include "ggml-cuda/vecdotq.cuh"
#include <climits>

namespace current_iq_m8 {
struct DecodedIQ {
    int grid[8];
    int scale0, scale1;
    float d;
};

template<ggml_type Type>
__device__ __forceinline__ DecodedIQ decode_weight(const void * input, int block, int iqs) {
    DecodedIQ out;
    if constexpr (Type == GGML_TYPE_IQ2_S) {
        const block_iq2_s * b = static_cast<const block_iq2_s *>(input) + block;
        const int packed = get_int_b2(b->qs, iqs/2);
        const auto * qs = reinterpret_cast<const uint8_t *>(&packed);
        const int qh = b->qh[iqs/2];
        const int signs_packed = get_int_b2(b->qs, QK_K/32 + iqs/2);
        const auto * signs = reinterpret_cast<const uint8_t *>(&signs_packed);
        out.scale0 = b->scales[iqs/2] & 0x0f;
        out.scale1 = b->scales[iqs/2] >> 4;
        out.d = __half2float(b->d);
        #pragma unroll
        for (int i = 0; i < 8; i += 2) {
            const int * grid = reinterpret_cast<const int *>(iq2s_grid + (qs[i/2] | ((qh << (8-i)) & 0x300)));
            const int s0 = __vcmpne4(((signs[i/2] & 0x03) << 7) | ((signs[i/2] & 0x0c) << 21), 0);
            const int s1 = __vcmpne4(((signs[i/2] & 0x30) << 3) | ((signs[i/2] & 0xc0) << 17), 0);
            out.grid[i] = __vsub4(grid[0] ^ s0, s0);
            out.grid[i+1] = __vsub4(grid[1] ^ s1, s1);
        }
    } else if constexpr (Type == GGML_TYPE_IQ3_S) {
        const block_iq3_s * b = static_cast<const block_iq3_s *>(input) + block;
        const int2 packed = make_int2(get_int_b2(b->qs, iqs), get_int_b2(b->qs, iqs+1));
        const auto * qs = reinterpret_cast<const uint8_t *>(&packed);
        const int qh = b->qh[iqs/2];
        const int signs_packed = get_int_b2(b->signs, iqs/2);
        const auto * signs = reinterpret_cast<const uint8_t *>(&signs_packed);
        out.scale0 = 1 + 2*((b->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
        out.scale1 = 0;
        out.d = __half2float(b->d);
        #pragma unroll
        for (int i = 0; i < 8; i += 2) {
            const int g0 = iq3s_grid[qs[i] | ((qh << (8-i)) & 0x100)];
            const int g1 = iq3s_grid[qs[i+1] | ((qh << (7-i)) & 0x100)];
            const int s0 = __vcmpne4(((signs[i/2] & 0x03) << 7) | ((signs[i/2] & 0x0c) << 21), 0);
            const int s1 = __vcmpne4(((signs[i/2] & 0x30) << 3) | ((signs[i/2] & 0xc0) << 17), 0);
            out.grid[i] = __vsub4(g0 ^ s0, s0);
            out.grid[i+1] = __vsub4(g1 ^ s1, s1);
        }
    } else {
        const block_iq3_xxs * b = static_cast<const block_iq3_xxs *>(input) + block;
        const int2 packed = make_int2(get_int_b2(b->qs, iqs), get_int_b2(b->qs, iqs+1));
        const auto * qs = reinterpret_cast<const uint8_t *>(&packed);
        const uint32_t aux = get_int_b2(b->qs, QK_K/16 + iqs/2);
        out.scale0 = aux >> 28;
        out.scale1 = 0;
        out.d = __half2float(b->d);
        #pragma unroll
        for (int i = 0; i < 8; i += 2) {
            const uint32_t signs = unpack_ksigns(aux >> (7*i/2));
            const int s0 = __vcmpne4(signs & 0x08040201, 0);
            const int s1 = __vcmpne4(signs & 0x80402010, 0);
            out.grid[i] = __vsub4(iq3xxs_grid[qs[i]] ^ s0, s0);
            out.grid[i+1] = __vsub4(iq3xxs_grid[qs[i+1]] ^ s1, s1);
        }
    }
    return out;
}

template<ggml_type Type>
__device__ __forceinline__ float native_dot(
        const DecodedIQ & w, const block_q8_1 * activation, int iqs) {
    const block_q8_1 & b = activation[iqs/2];
    int sum0 = 0, sum1 = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int u = get_int_b4(b.qs, i);
        if constexpr (Type == GGML_TYPE_IQ2_S) {
            if (i < 4) sum0 = ggml_cuda_dp4a(w.grid[i], u, sum0);
            else sum1 = ggml_cuda_dp4a(w.grid[i], u, sum1);
        } else {
            sum0 = ggml_cuda_dp4a(w.grid[i], u, sum0);
        }
    }
    // Keep the literal native integer parenthesization and signed division.
    // In particular, expanding scales into floating weights is not equivalent.
    int sum;
    if constexpr (Type == GGML_TYPE_IQ2_S) {
        sum = (sum0*w.scale0 + sum1*w.scale1 + (sum0 + sum1)/2)/4;
    } else if constexpr (Type == GGML_TYPE_IQ3_S) {
        sum = sum0*w.scale0;
    } else {
        sum = (w.scale0*sum0 + sum0/2)/2;
    }
    const float d = w.d * __low2float(b.ds);
    return d * sum;
}

template<ggml_type Type, int Groups, int ActiveWarps>
__launch_bounds__(32 * ActiveWarps * Groups, 1)
__global__ void iq_m8(
        const void * __restrict__ weights,
        const block_q8_1 * __restrict__ activation,
        float * __restrict__ output, int64_t output_stride, int k, int padded_k) {
    static_assert(QI2_S == 16 && QI3_XXS == 16 && QI3_S == 16, "Native IQ lane mapping changed");
    constexpr int rows = 2, cols = 8, warp_size = 32;
    // Logical native nwarps remains TWO even for the one-active-warp case.
    constexpr int qi = 16, vdr = 2, blocks_per_iter = vdr * 64 / qi;
    const int group = threadIdx.z;
    const int tid = warp_size * threadIdx.y + threadIdx.x;
    const int row = rows * (blockIdx.x * Groups + group);
    const int blocks = k / QK_K;
    const int iqs = vdr * (tid % (qi/vdr));
    float sums[cols][rows] = {{0.0f}};
    for (int block = tid / (qi/vdr); block < blocks; block += blocks_per_iter) {
        DecodedIQ decoded[rows];
        #pragma unroll
        for (int i = 0; i < rows; ++i) {
            decoded[i] = decode_weight<Type>(weights, (row+i)*blocks+block, iqs);
        }
        #pragma unroll
        for (int j = 0; j < cols; ++j) {
            const auto * x = activation + j*(padded_k/QK8_1) + block*(QK_K/QK8_1);
            #pragma unroll
            for (int i = 0; i < rows; ++i) sums[j][i] += native_dot<Type>(decoded[i], x, iqs);
        }
    }
    __shared__ float partial[ActiveWarps == 2 ? Groups : 1][cols][rows][warp_size];
    if constexpr (ActiveWarps == 2) {
        if (threadIdx.y) {
            #pragma unroll
            for (int j = 0; j < cols; ++j) {
                #pragma unroll
                for (int i = 0; i < rows; ++i) partial[group][j][i][threadIdx.x] = sums[j][i];
            }
        }
        __syncthreads();
        if (threadIdx.y) return;
    }
    #pragma unroll
    for (int j = 0; j < cols; ++j) {
        #pragma unroll
        for (int i = 0; i < rows; ++i) {
            if constexpr (ActiveWarps == 2) sums[j][i] += partial[group][j][i][threadIdx.x];
            else sums[j][i] += 0.0f; // Native warp1 is identically zero for K<=1024.
            sums[j][i] = warp_reduce_sum<warp_size>(sums[j][i]);
            if (threadIdx.x == i) output[j*output_stride+row+i] = sums[j][i];
        }
    }
}

template<ggml_type Type, int Groups, int ActiveWarps>
void launch(const void * weights, const void * q8, float * output,
            int n, int k, int64_t output_stride, cudaStream_t stream) {
    iq_m8<Type, Groups, ActiveWarps><<<n/(2*Groups), dim3(32, ActiveWarps, Groups), 0, stream>>>(
        weights, static_cast<const block_q8_1 *>(q8), output, output_stride, k, GGML_PAD(k, MATRIX_ROW_PADDING));
}

// The one-warp launch retains the native TWO-warp logical lane mapping.
// At K=1024 warp1 owns no K blocks. Its +0 contribution remains in the kernel.
// Other architectures, dimensions, formats, and incomplete row groups use GGML.
inline bool eligible(int cc, ggml_type type, int64_t n, int64_t k, int64_t m) {
    return cc == 860 && m == 8 && k == 1024 && n > 0 &&
        n <= INT_MAX / 4 &&
        ((type == GGML_TYPE_IQ2_S && n % 4 == 0) ||
         (type == GGML_TYPE_IQ3_XXS && n % 2 == 0) ||
         (type == GGML_TYPE_IQ3_S && n == 512));
}

inline void run(ggml_type type, const void * weights, const void * q8,
                float * output, int n, int64_t output_stride, cudaStream_t stream) {
    if (type == GGML_TYPE_IQ2_S) {
        launch<GGML_TYPE_IQ2_S, 2, 1>(weights, q8, output, n, 1024, output_stride, stream);
    } else if (type == GGML_TYPE_IQ3_S) {
        launch<GGML_TYPE_IQ3_S, 1, 1>(weights, q8, output, n, 1024, output_stride, stream);
    } else {
        launch<GGML_TYPE_IQ3_XXS, 1, 1>(weights, q8, output, n, 1024, output_stride, stream);
    }
}
}  // namespace current_iq_m8
