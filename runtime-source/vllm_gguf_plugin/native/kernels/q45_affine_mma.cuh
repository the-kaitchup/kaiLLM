// SPDX-License-Identifier: Apache-2.0 AND MIT
// GGML tile/packing portions: Copyright (c) 2023-2026 The ggml authors.
// See LLAMA_CPP_LICENSE.txt for the upstream notice and permission terms.
// Experimental J8 Q4_K/Q5_K MMA. Keep raw FP16 d/dmin and integer group
// coefficients, never their FP16 products. Activation metadata contains the
// native MMVQ half scale and the exact int16 sum of its 32 quantized codes.
#pragma once

template<ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void q45_affine_load(
        const char * __restrict__ x, int * tile_x, int kbx0, int i_max, int stride) {
#if defined(TURING_MMA_AVAILABLE)
    static_assert(type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K, "Q4/Q5 only");
    constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int warps = ggml_cuda_mmq_get_nthreads(type, J, fallback) / 32;
    constexpr int pitch = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    const int lane = threadIdx.x;
    for (int first = 0; first < I; first += warps) {
        const int row = first + threadIdx.y;
        const int source_row = fallback ? min(row, i_max) : row;
        int low, high = 0;
        if constexpr (type == GGML_TYPE_Q4_K) {
            const auto * b = reinterpret_cast<const block_q4_K *>(x) + kbx0 + source_row * stride;
            low = get_int_b4(b->qs, lane);
        } else {
            const auto * b = reinterpret_cast<const block_q5_K *>(x) + kbx0 + source_row * stride;
            low = get_int_b4(b->qs, lane);
            high = get_int_b4(b->qh, lane % 8);
        }
        const int k0 = 16 * (lane / 8) + lane % 8;
        tile_x[row * pitch + k0] = (low & 0x0f0f0f0f)
            | (((high >> (2 * (lane / 8))) << 4) & 0x10101010);
        tile_x[row * pitch + k0 + 8] = ((low >> 4) & 0x0f0f0f0f)
            | (((high >> (2 * (lane / 8) + 1)) << 4) & 0x10101010);
    }
    // Five words fit inside the existing eight-word metadata region:
    // raw half2(d,dmin), uint8 scales[8], uint8 minima[8].
    for (int first = 0; first < I; first += warps * 16) {
        const int row = (first + threadIdx.y * 16 + lane / 2) % I;
        const int source_row = fallback ? min(row, i_max) : row;
        const int * scales;
        uint32_t dm;
        if constexpr (type == GGML_TYPE_Q4_K) {
            const auto * b = reinterpret_cast<const block_q4_K *>(x) + kbx0 + source_row * stride;
            scales = reinterpret_cast<const int *>(b->scales);
            dm = *reinterpret_cast<const uint32_t *>(&b->dm);
        } else {
            const auto * b = reinterpret_cast<const block_q5_K *>(x) + kbx0 + source_row * stride;
            scales = reinterpret_cast<const int *>(b->scales);
            dm = *reinterpret_cast<const uint32_t *>(&b->dm);
        }
        const int half = lane % 2;
        int * metadata = tile_x + row * pitch + 64;
        if (!half) metadata[0] = dm;
        metadata[1 + half] = unpack_scales_q45_K(scales, half);
        metadata[3 + half] = unpack_scales_q45_K(scales, half + 2);
    }
#else
    GGML_UNUSED_VARS(x, tile_x, kbx0, i_max, stride);
    NO_DEVICE_CODE;
#endif
}

template<ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void q45_affine_dot(
        const int * __restrict__ x, const int * __restrict__ y,
        float * __restrict__ sum, const int k00) {
#if defined(TURING_MMA_AVAILABLE)
    static_assert(J == 8, "Only physical M8 decode is supported");
    typedef tile<16, 8, int> tile_A;
    typedef tile<8, 8, int> tile_B;
    typedef tile<16, 8, int> tile_C;
    constexpr int pitch = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    static_assert(ggml_cuda_mmq_get_rows_per_warp(type, J, fallback) == 16,
                  "The J8 kernel requires one row fragment per warp");
    const int row0 = threadIdx.y * 16;
    tile_A A[4];
    uint32_t scales[2], minima[2];
    float2 dm[2];
    #pragma unroll
    for (int group = 0; group < 4; ++group) {
        load_ldmatrix(A[group], x + row0 * pitch + k00 + group * 8, pitch);
    }
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int row = row0 + tile_C::get_i(2 * r);
        const int * metadata = x + row * pitch + 64;
        dm[r] = __half22float2(*reinterpret_cast<const half2 *>(metadata));
        scales[r] = metadata[1 + k00 / 32];
        minima[r] = metadata[3 + k00 / 32];
    }
    float dot_sum[4] = {0, 0, 0, 0}, min_sum[4] = {0, 0, 0, 0};
    #pragma unroll
    for (int group = 0; group < 4; ++group) {
        tile_B B;
        tile_C C;
        load_generic(B, y + 4 + group * 8, MMQ_TILE_Y_K);
        mma(C, A[group], B);
        float d8[2];
        int sums[2];
        #pragma unroll
        for (int col = 0; col < 2; ++col) {
            const int j = tile_C::get_j(col);
            const uint32_t word = y[j * MMQ_TILE_Y_K + group];
            d8[col] = __half2float(__ushort_as_half(word & 0xffff));
            sums[col] = int16_t(word >> 16);
        }
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int scale = (scales[l / 2] >> (group * 8)) & 255;
            const int minimum = (minima[l / 2] >> (group * 8)) & 255;
            dot_sum[l] += d8[l % 2] * (C.x[l] * scale);
            min_sum[l] += d8[l % 2] * (sums[l % 2] * minimum);
        }
    }
    #pragma unroll
    for (int l = 0; l < 4; ++l) {
        sum[l] += dm[l / 2].x * dot_sum[l] - dm[l / 2].y * min_sum[l];
    }
#else
    GGML_UNUSED_VARS(x, y, sum, k00);
    NO_DEVICE_CODE;
#endif
}

#define AFFINE_SPECIALIZATION(TYPE, FALLBACK) \
template<> constexpr __device__ ggml_cuda_mmq_load_tiles_t \
ggml_cuda_mmq_get_load_tiles<TYPE, 8, FALLBACK, GGML_PREC_Q8>() { \
    return q45_affine_load<TYPE, 8, FALLBACK>; \
} \
template<> constexpr __device__ ggml_cuda_mmq_vec_dot_t \
ggml_cuda_mmq_get_vec_dot<TYPE, 8, FALLBACK, GGML_PREC_Q8>() { \
    return q45_affine_dot<TYPE, 8, FALLBACK>; \
}

AFFINE_SPECIALIZATION(GGML_TYPE_Q4_K, false)
AFFINE_SPECIALIZATION(GGML_TYPE_Q4_K, true)
AFFINE_SPECIALIZATION(GGML_TYPE_Q5_K, false)
AFFINE_SPECIALIZATION(GGML_TYPE_Q5_K, true)
#undef AFFINE_SPECIALIZATION
