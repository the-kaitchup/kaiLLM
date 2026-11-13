// SPDX-License-Identifier: Apache-2.0 AND MIT
#pragma once
// GGML arithmetic portions: Copyright (c) 2023-2026 The ggml authors.
// See LLAMA_CPP_LICENSE.txt.
// Native T1 SSM lane/K mapping and four-warp FP tree, with independent
// request channels fused inside one CTA so packed weights are decoded once.
// Validated against public GGML and r4 by check_native_ssm_fused.py
// and check_native_ssm_extended.py (all four local GGUF profiles).
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/vecdotq.cuh"

namespace gguf_native_ssm_fused {
struct DecodedQ8 {
    int v[2];
    float d;
};

struct DecodedQ5 {
    int vl[2], vh[2];
    uint16_t aux[2];
    half2 dm;
};

template<ggml_type Type> struct WeightTraits;

template<> struct WeightTraits<GGML_TYPE_Q8_0> {
    using Decoded = DecodedQ8;
    static constexpr int qk = QK8_0, qi = QI8_0, vdr = VDR_Q8_0_Q8_1_MMVQ;
    __device__ static __forceinline__ Decoded decode(const void * data, int block, int iqs) {
        const auto * b = static_cast<const block_q8_0 *>(data) + block;
        return {{get_int_b2(b->qs, iqs), get_int_b2(b->qs, iqs + 1)}, __half2float(b->d)};
    }
    __device__ static __forceinline__ float dot(const Decoded & w, const block_q8_1 * x, int iqs) {
        const int u[2] = {get_int_b4(x->qs, iqs), get_int_b4(x->qs, iqs + 1)};
        return vec_dot_q8_0_q8_1_impl<float, vdr>(w.v, u, w.d, __low2float(x->ds));
    }
};

template<> struct WeightTraits<GGML_TYPE_Q5_K> {
    using Decoded = DecodedQ5;
    static constexpr int qk = QK_K, qi = QI5_K, vdr = VDR_Q5_K_Q8_1_MMVQ;
    __device__ static __forceinline__ Decoded decode(const void * data, int block, int iqs) {
        const auto * b = static_cast<const block_q5_K *>(data) + block;
        Decoded out;
        const int offset = QR5_K * ((iqs/2) / (QI8_1/2));
        const int * ql = reinterpret_cast<const int *>(b->qs + 16*offset + 4*((iqs/2)%4));
        const int * qh = reinterpret_cast<const int *>(b->qh + 4*((iqs/2)%4));
        out.vl[0] = ql[0]; out.vl[1] = ql[4];
        out.vh[0] = qh[0] >> offset; out.vh[1] = qh[4] >> offset;
        const uint16_t * scales = reinterpret_cast<const uint16_t *>(b->scales);
        const int j = offset/2, jm = j & 1;
        const uint32_t s0 = scales[jm], s2 = scales[jm+2], s4 = scales[jm+4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        out.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) |
            ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        out.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) |
            ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        out.dm = b->dm;
        return out;
    }
    __device__ static __forceinline__ float dot(const Decoded & w, const block_q8_1 * x, int iqs) {
        int u[2*QR5_K];
        float d8[QR5_K];
        const int offset = QR5_K * ((iqs/2) / (QI8_1/2));
        #pragma unroll
        for (int i=0; i<QR5_K; ++i) {
            const block_q8_1 * b = x + offset + i;
            const int * q8 = reinterpret_cast<const int *>(b->qs) + ((iqs/2)%4);
            u[2*i] = q8[0]; u[2*i+1] = q8[4]; d8[i] = __low2float(b->ds);
        }
        const auto * sc = reinterpret_cast<const uint8_t *>(w.aux);
        return vec_dot_q5_K_q8_1_impl_vmmq(w.vl, w.vh, u, sc, sc+2, w.dm, d8);
    }
};

struct DecodedQ6 {
    int vl, vh;
    int8_t scales[8];
    float d;
};
template<> struct WeightTraits<GGML_TYPE_Q6_K> {
    using Decoded = DecodedQ6;
    static constexpr int qk = QK_K, qi = QI6_K, vdr = VDR_Q6_K_Q8_1_MMVQ;
    __device__ static __forceinline__ Decoded decode(const void * data, int block, int iqs) {
        const auto * b = static_cast<const block_q6_K *>(data) + block;
        Decoded out{};
        const int scale_offset = (QI6_K/4)*(iqs/(QI6_K/2)) + (iqs%(QI6_K/2))/(QI6_K/8);
        const int vh_shift = 2*((iqs%(QI6_K/2))/(QI6_K/4));
        out.vl = get_int_b2(b->ql, iqs);
        out.vh = get_int_b2(b->qh, (QI6_K/4)*(iqs/(QI6_K/2)) + iqs%(QI6_K/4)) >> vh_shift;
        out.scales[0] = b->scales[scale_offset];
        out.scales[4] = b->scales[scale_offset+4];
        out.d = __half2float(b->d);
        return out;
    }
    __device__ static __forceinline__ float dot(const Decoded & w, const block_q8_1 * x, int iqs) {
        const int offset = 2*QR6_K*(iqs/(QI6_K/2)) + (iqs%(QI6_K/2))/(QI6_K/4);
        int u[QR6_K]; float d8[QR6_K];
        #pragma unroll
        for (int i=0; i<QR6_K; ++i) {
            u[i] = get_int_b4(x[offset+2*i].qs, iqs%QI8_1);
            d8[i] = __low2float(x[offset+2*i].ds);
        }
        return vec_dot_q6_K_q8_1_impl_mmvq(w.vl, w.vh, u, w.scales, w.d, d8);
    }
};

template<ggml_type Type, int Channels, int Rows>
__launch_bounds__(128, 1)
__global__ void fused_channels(
        const void * __restrict__ weights, const block_q8_1 * __restrict__ input,
        float * __restrict__ output, int n) {
    using Traits = WeightTraits<Type>;
    constexpr int k = 2048, warp_size = 32, warps = 4;
    constexpr int blocks_per_row = k / Traits::qk;
    constexpr int blocks_per_iter = Traits::vdr * warps * warp_size / Traits::qi;
    const int tid = threadIdx.y * warp_size + threadIdx.x;
    const int row = blockIdx.x * Rows;
    const int iqs = Traits::vdr * (tid % (Traits::qi / Traits::vdr));
    float sums[Channels][Rows] = {{0.0f}};
    for (int block = tid / (Traits::qi/Traits::vdr); block < blocks_per_row; block += blocks_per_iter) {
        typename Traits::Decoded decoded[Rows];
        #pragma unroll
        for (int i=0; i<Rows; ++i) decoded[i] = Traits::decode(weights, (row+i)*blocks_per_row+block, iqs);
        #pragma unroll
        for (int channel=0; channel<Channels; ++channel) {
            const auto * x = input + channel*(k/QK8_1) + block*(Traits::qk/QK8_1);
            #pragma unroll
            for (int i=0; i<Rows; ++i) sums[channel][i] += Traits::dot(decoded[i], x, iqs);
        }
    }
    __shared__ float partial[warps-1][Channels][Rows][warp_size];
    if (threadIdx.y) {
        #pragma unroll
        for (int channel=0; channel<Channels; ++channel) {
            #pragma unroll
            for (int i=0; i<Rows; ++i) partial[threadIdx.y-1][channel][i][threadIdx.x] = sums[channel][i];
        }
    }
    __syncthreads();
    if (threadIdx.y) return;
    #pragma unroll
    for (int channel=0; channel<Channels; ++channel) {
        #pragma unroll
        for (int i=0; i<Rows; ++i) {
            // Retain the native left-to-right warp addition, then XOR reduction.
            #pragma unroll
            for (int warp=0; warp<warps-1; ++warp) sums[channel][i] += partial[warp][channel][i][threadIdx.x];
            sums[channel][i] = warp_reduce_sum<warp_size>(sums[channel][i]);
            if (threadIdx.x == i) output[channel*n + row+i] = sums[channel][i];
        }
    }
}

template<ggml_type Type, int Channels>
void launch(const void * weights, const void * q8, float * output, int n, int rows, cudaStream_t stream) {
    #define LAUNCH(R) fused_channels<Type, Channels, R><<<n/R, dim3(32,4), 0, stream>>>(weights, static_cast<const block_q8_1 *>(q8), output, n)
    if (rows == 1) LAUNCH(1);
    else if (rows == 2) LAUNCH(2);
    else LAUNCH(4);
    #undef LAUNCH
}

template<ggml_type Type>
void dispatch(const void * weights, const void * q8, float * output, int n, int rows, int channels, cudaStream_t stream) {
    #define CASE(B) case B: launch<Type,B>(weights,q8,output,n,rows,stream); break
    switch (channels) { CASE(1); CASE(2); CASE(3); CASE(4); CASE(5); CASE(6); CASE(7); CASE(8); }
    #undef CASE
}

inline bool selected(int cc, ggml_type type, int64_t n, int64_t k,
                     int64_t tokens, int64_t channels) {
    return cc == 860 && k == 2048 && tokens == 1 && channels >= 2 && channels <= 8 &&
        (((type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K) && n == 1024) ||
         (type == GGML_TYPE_Q8_0 && (n == 1024 || n == 2048)));
}

// q8 has original native MMVQ storage [channels, 64 block_q8_1]. The
// producer must retain half ds, byte quantization, and original T1 geometry.
inline void run_q8(const void * weights, const void * q8, float * output,
                   ggml_type type, int n, int channels, cudaStream_t stream) {
    if (type == GGML_TYPE_Q8_0) {
        const int rows = n == 1024 || channels >= 6 ? 4 : 2;
        dispatch<GGML_TYPE_Q8_0>(weights, q8, output, n, rows, channels, stream);
    } else if (type == GGML_TYPE_Q6_K) {
        dispatch<GGML_TYPE_Q6_K>(weights, q8, output, n, 4, channels, stream);
    } else {
        dispatch<GGML_TYPE_Q5_K>(weights, q8, output, n, 2, channels, stream);
    }
}

inline void run(const void * weights, const float * input, float * output,
                void * scratch, ggml_type type, int n, int channels,
                cudaStream_t stream) {
    quantize_row_q8_1_cuda(input, nullptr, scratch, type,
        2048, 2048, 2048, int64_t(channels)*2048, 2048, 1, channels, 1, stream);
    run_q8(weights, scratch, output, type, n, channels, stream);
}
} // namespace gguf_native_ssm_fused
