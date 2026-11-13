// SPDX-License-Identifier: Apache-2.0 AND MIT
#pragma once
// GGML arithmetic: Copyright (c) 2023-2026 The ggml authors.
// See LLAMA_CPP_LICENSE.txt. Original packed Q4_K/Q5_K, native M8 two-warp
// lane/K mapping and FP tree. Only independent output rows/columns are grouped.
// Measured and bitwise checked by check_q45_body_variants.py (r5).
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/vecdotq.cuh"
#include <cstdint>
namespace gguf_native_body {
struct DecodedQ5 {
    int vl[2], vh[2];
    uint16_t aux[2];
    half2 dm;
};

template<ggml_type Type> struct WeightTraits;

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

template<> struct WeightTraits<GGML_TYPE_Q4_K> {
    using Decoded = DecodedQ5;
    static constexpr int qk = QK_K, qi = QI4_K, vdr = VDR_Q4_K_Q8_1_MMVQ;
    __device__ static __forceinline__ Decoded decode(const void * data, int block, int iqs) {
        const auto * b = static_cast<const block_q4_K *>(data) + block;
        Decoded out;
        const int offset = QR4_K * ((iqs/2) / (QI8_1/2));
        const int * ql = reinterpret_cast<const int *>(b->qs + 16*offset + 4*((iqs/2)%4));
        out.vl[0] = ql[0]; out.vl[1] = ql[4];
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
        int u[2*QR4_K];
        float d8[QR4_K];
        const int offset = QR4_K * ((iqs/2) / (QI8_1/2));
        #pragma unroll
        for (int i=0; i<QR4_K; ++i) {
            const block_q8_1 * b = x + offset + i;
            const int * q8 = reinterpret_cast<const int *>(b->qs) + ((iqs/2)%4);
            u[2*i] = q8[0]; u[2*i+1] = q8[4]; d8[i] = __low2float(b->ds);
        }
        const auto * sc = reinterpret_cast<const uint8_t *>(w.aux);
        return vec_dot_q4_K_q8_1_impl_vmmq(w.vl, u, sc, sc+2, w.dm, d8);
    }
};


template<ggml_type Type, int Rows, int ColumnGroups>
__launch_bounds__(64*ColumnGroups, 1)
__global__ void body(const void * __restrict__ weights,
                     const block_q8_1 * __restrict__ input,
                     float * __restrict__ output, int64_t output_stride, int k) {
    using Traits = WeightTraits<Type>;
    constexpr int Cols = 8/ColumnGroups;
    const int group = threadIdx.y / 2, warp = threadIdx.y % 2;
    const int tid = warp*32 + threadIdx.x;
    const int row = blockIdx.x*Rows;
    const int blocks_per_row = k/256, q8_stride = GGML_PAD(k,MATRIX_ROW_PADDING)/32;
    const int iqs = 2*(tid%16);
    float sums[Cols][Rows] = {{0.0f}};
    for (int block=tid/16; block<blocks_per_row; block+=4) {
        typename Traits::Decoded decoded[Rows];
        #pragma unroll
        for (int i=0;i<Rows;++i) decoded[i]=Traits::decode(weights,(row+i)*blocks_per_row+block,iqs);
        #pragma unroll
        for (int col=0;col<Cols;++col) {
            const block_q8_1 * x=input+(group*Cols+col)*q8_stride+block*8;
            #pragma unroll
            for(int i=0;i<Rows;++i) sums[col][i]+=Traits::dot(decoded[i],x,iqs);
        }
    }
    __shared__ float partial[ColumnGroups][Cols][Rows][32];
    if (warp) {
        #pragma unroll
        for(int col=0;col<Cols;++col) {
            #pragma unroll
            for(int i=0;i<Rows;++i) partial[group][col][i][threadIdx.x]=sums[col][i];
        }
    }
    __syncthreads();
    if(warp) return;
    #pragma unroll
    for(int col=0;col<Cols;++col) {
        #pragma unroll
        for(int i=0;i<Rows;++i) {
            sums[col][i]+=partial[group][col][i][threadIdx.x];
            sums[col][i]=warp_reduce_sum<32>(sums[col][i]);
            if(threadIdx.x==i) output[int64_t(group*Cols+col)*output_stride+row+i]=sums[col][i];
        }
    }
}


inline bool eligible(int cc, ggml_type type, int64_t n, int64_t k, int64_t m) {
    return cc == 860 && m == 8 &&
        (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) &&
        ((k == 2048 && (n == 2048 || n == 4096 || n == 6144)) ||
         (k == 6144 && n == 2048) ||
         (type == GGML_TYPE_Q4_K && k == 1024 && n == 6144));
}

inline void run(ggml_type type, const void * weights, const void * q8,
                float * output, int n, int k, int64_t output_stride,
                cudaStream_t stream) {
    const auto * x = static_cast<const block_q8_1 *>(q8);
    // The N=K=2048 projections favor two groups of four columns; larger
    // projections favor sharing each weight across all eight columns.
    const bool split = n == 2048 && k == 2048;
    #define LAUNCH(T,G) body<T,4,G><<<n/4,dim3(32,2*G),0,stream>>>(weights,x,output,output_stride,k)
    if (type == GGML_TYPE_Q4_K) {
        if (split) { LAUNCH(GGML_TYPE_Q4_K,2); }
        else { LAUNCH(GGML_TYPE_Q4_K,1); }
    } else {
        if (split) { LAUNCH(GGML_TYPE_Q5_K,2); }
        else { LAUNCH(GGML_TYPE_Q5_K,1); }
    }
    #undef LAUNCH
}
} // namespace gguf_native_body
