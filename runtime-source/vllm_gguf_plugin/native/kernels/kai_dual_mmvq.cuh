// SPDX-License-Identifier: Apache-2.0
// Dual-Q8_1 activation MMVQ for every GGML weight format with a Q8_1 vec_dot.
//
// Activations are represented as x = s * (d1*q1 + d2*q2) per 32-value block:
//   * s  is an exact per-row power of two that puts the row's absolute maximum
//        into [64, 128); scaling by s and back by 1/s is exact in binary floating
//        point, keeps every half-precision block scale normal for all but
//        negligible blocks and keeps every half-precision block sum far below
//        65504, so no activation row needs a guarded F32 repair;
//   * q1 is the regular GGML Q8_1 quantization of s*x (same arithmetic as
//        quantize_q8_1, so the first pass equals what GGML/llama.cpp computes);
//   * q2 quantizes the first pass's residual s*x - half(d1)*q1 with its own
//        half scale, and carries the block-sum rounding correction in its half
//        sum, so the offset terms of Q4_0/Q4_1/Q5_x add up to the true block sum.
// The weight is read once per tile; each GGML vec_dot routine runs on both
// activation passes, which gives ~16-bit activation precision with exact
// GGML weight decoding. The epilogue applies 1/s and stores F32 or BF16 into a
// strided (merged-projection) destination.
#pragma once

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/vecdotq.cuh"

#include <cuda_bf16.h>
#include <stdexcept>

namespace kai_dual {

// Same signature as GGML's (mmvq.cu-local) vec_dot_q_cuda_t.
typedef float (*vec_dot_t)(const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
                           const int & kbx, const int & iqs);

constexpr int MAX_COLS = 8;
constexpr int PREP_THREADS = 1024;

static constexpr __host__ __device__ int nwarps_for(int ncols) {
    // GGML's generic (Ampere/Ada/Blackwell) MMVQ table.
    return ncols <= 4 ? 4 : 2;
}

static constexpr __host__ __device__ int rows_per_block_for(int ncols) {
    return ncols == 1 ? 1 : 2;
}

static constexpr __host__ __device__ int vdr_for(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:    return VDR_Q4_0_Q8_1_MMVQ;
        case GGML_TYPE_Q4_1:    return VDR_Q4_1_Q8_1_MMVQ;
        case GGML_TYPE_Q5_0:    return VDR_Q5_0_Q8_1_MMVQ;
        case GGML_TYPE_Q5_1:    return VDR_Q5_1_Q8_1_MMVQ;
        case GGML_TYPE_Q8_0:    return VDR_Q8_0_Q8_1_MMVQ;
        case GGML_TYPE_Q2_K:    return VDR_Q2_K_Q8_1_MMVQ;
        case GGML_TYPE_Q3_K:    return VDR_Q3_K_Q8_1_MMVQ;
        case GGML_TYPE_Q4_K:    return VDR_Q4_K_Q8_1_MMVQ;
        case GGML_TYPE_Q5_K:    return VDR_Q5_K_Q8_1_MMVQ;
        case GGML_TYPE_Q6_K:    return VDR_Q6_K_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_S:   return VDR_IQ2_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_XXS: return VDR_IQ3_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ3_S:   return VDR_IQ3_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_XS:  return VDR_IQ4_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ4_NL:  return VDR_IQ4_NL_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XXS: return VDR_IQ2_XXS_Q8_1_MMVQ;
        case GGML_TYPE_IQ2_XS:  return VDR_IQ2_XS_Q8_1_MMVQ;
        case GGML_TYPE_IQ1_S:   return VDR_IQ1_S_Q8_1_MMVQ;
        case GGML_TYPE_IQ1_M:   return VDR_IQ1_M_Q8_1_MMVQ;
        default:                return 1;
    }
}

static constexpr __device__ vec_dot_t vec_dot_for(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:    return vec_dot_q4_0_q8_1;
        case GGML_TYPE_Q4_1:    return vec_dot_q4_1_q8_1;
        case GGML_TYPE_Q5_0:    return vec_dot_q5_0_q8_1;
        case GGML_TYPE_Q5_1:    return vec_dot_q5_1_q8_1;
        case GGML_TYPE_Q8_0:    return vec_dot_q8_0_q8_1;
        case GGML_TYPE_Q2_K:    return vec_dot_q2_K_q8_1;
        case GGML_TYPE_Q3_K:    return vec_dot_q3_K_q8_1;
        case GGML_TYPE_Q4_K:    return vec_dot_q4_K_q8_1;
        case GGML_TYPE_Q5_K:    return vec_dot_q5_K_q8_1;
        case GGML_TYPE_Q6_K:    return vec_dot_q6_K_q8_1;
        case GGML_TYPE_IQ2_S:   return vec_dot_iq2_s_q8_1;
        case GGML_TYPE_IQ3_XXS: return vec_dot_iq3_xxs_q8_1;
        case GGML_TYPE_IQ3_S:   return vec_dot_iq3_s_q8_1;
        case GGML_TYPE_IQ4_XS:  return vec_dot_iq4_xs_q8_1;
        case GGML_TYPE_IQ4_NL:  return vec_dot_iq4_nl_q8_1;
        case GGML_TYPE_IQ2_XXS: return vec_dot_iq2_xxs_q8_1;
        case GGML_TYPE_IQ2_XS:  return vec_dot_iq2_xs_q8_1;
        case GGML_TYPE_IQ1_S:   return vec_dot_iq1_s_q8_1;
        case GGML_TYPE_IQ1_M:   return vec_dot_iq1_m_q8_1;
        default:                return nullptr;
    }
}

static inline bool type_supported(int type) {
    switch (type) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0: case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K: case GGML_TYPE_IQ2_S: case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ4_XS: case GGML_TYPE_IQ4_NL: case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS: case GGML_TYPE_IQ1_S: case GGML_TYPE_IQ1_M:
            return true;
        default:
            return false;
    }
}

template <typename T> static __device__ __forceinline__ float load_value(const T * p);
template <> __device__ __forceinline__ float load_value<float>(const float * p) { return *p; }
template <> __device__ __forceinline__ float load_value<nv_bfloat16>(const nv_bfloat16 * p) { return __bfloat162float(*p); }
template <> __device__ __forceinline__ float load_value<half>(const half * p) { return __half2float(*p); }

template <typename T> static __device__ __forceinline__ T store_value(float v);
template <> __device__ __forceinline__ float store_value<float>(float v) { return v; }
template <> __device__ __forceinline__ nv_bfloat16 store_value<nv_bfloat16>(float v) { return __float2bfloat16(v); }

// In-thread replica of GGML's lane-0 warp_reduce_sum<32> (xor butterfly 16..1),
// so block sums are bit-identical to quantize_q8_1.
static __device__ __forceinline__ float butterfly_sum32(const float (&v)[QK8_1]) {
    float s[QK8_1];
#pragma unroll
    for (int j = 0; j < QK8_1; ++j) {
        s[j] = v[j];
    }
#pragma unroll
    for (int offset = QK8_1 / 2; offset > 0; offset >>= 1) {
#pragma unroll
        for (int j = 0; j < offset; ++j) {
            s[j] += s[j + offset];
        }
    }
    return s[0];
}

static __device__ __forceinline__ void store_block(block_q8_1 * b, const int8_t (&q)[QK8_1], const half2 ds) {
    int * qs = reinterpret_cast<int *>(b->qs);
#pragma unroll
    for (int j = 0; j < QK8_1 / 4; ++j) {
        qs[j] = int(uint8_t(q[4*j])) | (int(uint8_t(q[4*j+1])) << 8) |
                (int(uint8_t(q[4*j+2])) << 16) | (int(uint8_t(q[4*j+3])) << 24);
    }
    b->ds = ds;
}

// grid (m), block round_up(k/32, 32) threads: one CTA per activation row and one
// thread per 32-value block; the row is read once and kept in registers.
template <typename src_t, bool two_pass>
__launch_bounds__(PREP_THREADS, 1)
static __global__ void prepare_dual(const src_t * __restrict__ x, const int64_t stride_x, const int k,
                                    block_q8_1 * __restrict__ y1, block_q8_1 * __restrict__ y2,
                                    float * __restrict__ inverse) {
    const int row = blockIdx.x;
    const int ib = threadIdx.x;
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int nblocks = k / QK8_1;
    const src_t * xb = x + int64_t(row) * stride_x + int64_t(ib) * QK8_1;

    float v[QK8_1];
    float block_max = 0.0f;
    if (ib < nblocks) {
        constexpr int vec = 16 / sizeof(src_t);
        if (reinterpret_cast<uintptr_t>(xb) % 16 == 0) {
#pragma unroll
            for (int c = 0; c < QK8_1 / vec; ++c) {
                const int4 packed = reinterpret_cast<const int4 *>(xb)[c];
                const src_t * values = reinterpret_cast<const src_t *>(&packed);
#pragma unroll
                for (int j = 0; j < vec; ++j) {
                    v[c * vec + j] = load_value(values + j);
                }
            }
        } else {
#pragma unroll
            for (int j = 0; j < QK8_1; ++j) {
                v[j] = load_value(xb + j);
            }
        }
#pragma unroll
        for (int j = 0; j < QK8_1; ++j) {
            block_max = fmaxf(block_max, fabsf(v[j]));
        }
    }

    // Exact row absolute maximum.
    __shared__ float warp_max[PREP_THREADS / WARP_SIZE];
    float amax = warp_reduce_max<WARP_SIZE>(block_max);
    if (lane == 0) {
        warp_max[warp] = amax;
    }
    __syncthreads();
    if (warp == 0) {
        amax = lane < int(blockDim.x / WARP_SIZE) ? warp_max[lane] : 0.0f;
        amax = warp_reduce_max<WARP_SIZE>(amax);
        if (lane == 0) {
            warp_max[0] = amax;
        }
    }
    __syncthreads();
    amax = warp_max[0];

    // Binary normalization into [64, 128); zero, subnormal and non-finite rows keep 1.
    const int biased = int((__float_as_uint(amax) >> 23) & 255u);
    int scaled_exponent = 127;
    if (biased != 0 && biased != 255) {
        scaled_exponent = min(max(260 - biased, 1), 253);
    }
    const float scale = __uint_as_float(uint32_t(scaled_exponent) << 23);
    if (threadIdx.x == 0) {
        inverse[row] = __uint_as_float(uint32_t(254 - scaled_exponent) << 23);
    }
    if (ib >= nblocks) {
        return;
    }

    // First pass: identical to GGML quantize_q8_1 on the normalized row.
    float a1 = 0.0f;
#pragma unroll
    for (int j = 0; j < QK8_1; ++j) {
        v[j] *= scale;
        a1 = fmaxf(a1, fabsf(v[j]));
    }
    const float d1 = a1 / 127.0f;
    int8_t q1[QK8_1];
    int isum1 = 0;
#pragma unroll
    for (int j = 0; j < QK8_1; ++j) {
        q1[j] = a1 == 0.0f ? 0 : roundf(v[j] / d1);
        isum1 += q1[j];
    }
    // As current GGML quantize_q8_1: the stored sum is that of the quantized values.
    const half2 ds1 = __floats2half2_rn(d1, d1 * float(isum1));
    store_block(y1 + int64_t(row) * nblocks + ib, q1, ds1);

    if constexpr (two_pass) {
        // Second pass: residual of the value the dot product actually uses.
        const float d1h = __low2float(ds1);
        float a2 = 0.0f;
#pragma unroll
        for (int j = 0; j < QK8_1; ++j) {
            v[j] -= d1h * float(q1[j]);
            a2 = fmaxf(a2, fabsf(v[j]));
        }
        const float d2 = a2 / 127.0f;
        int8_t q2[QK8_1];
        int isum2 = 0;
#pragma unroll
        for (int j = 0; j < QK8_1; ++j) {
            q2[j] = a2 == 0.0f ? 0 : roundf(v[j] / d2);
            isum2 += q2[j];
        }
        store_block(y2 + int64_t(row) * nblocks + ib, q2, __floats2half2_rn(d2, d2 * float(isum2)));
    }
}

// grid ceil(nrows / rows_per_block), block (32, nwarps). dst column j of row r is
// dst[j*stride_col_dst + r] (activation row j == output row j).
template <ggml_type type, int ncols, typename dst_t, bool two_pass>
__launch_bounds__(nwarps_for(ncols) * WARP_SIZE, 1)
static __global__ void mmvq_dual(const void * __restrict__ vx, const block_q8_1 * __restrict__ y1,
                                 const block_q8_1 * __restrict__ y2, const float * __restrict__ inverse,
                                 dst_t * __restrict__ dst, const int ncols_x, const int nrows_x,
                                 const int stride_col_y, const int stride_col_dst) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int qi = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = vdr_for(type);
    constexpr int nwarps = nwarps_for(ncols);
    constexpr int rows_per_block = rows_per_block_for(ncols);
    constexpr vec_dot_t vec_dot = vec_dot_for(type);

    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
    const int row0 = rows_per_block * blockIdx.x;
    const int blocks_per_row_x = ncols_x / qk;
    constexpr int blocks_per_iter = vdr * nwarps * WARP_SIZE / qi;

    float tmp[ncols][rows_per_block] = {{0.0f}};

    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (tid % (qi / vdr));
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_block; ++i) {
                // Clamp reads of a partial last tile; those lanes are never stored.
                const int ibx = min(row0 + i, nrows_x - 1) * blocks_per_row_x + kbx;
                if constexpr (two_pass) {
                    tmp[j][i] += vec_dot(vx, &y1[j * stride_col_y + kby], ibx, kqs)
                               + vec_dot(vx, &y2[j * stride_col_y + kby], ibx, kqs);
                } else {
                    tmp[j][i] += vec_dot(vx, &y1[j * stride_col_y + kby], ibx, kqs);
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps - 1 > 0 ? nwarps - 1 : 1][ncols][rows_per_block][WARP_SIZE];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_block; ++i) {
                tmp_shared[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        const float unscale = inverse[j];
#pragma unroll
        for (int i = 0; i < rows_per_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps - 1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
            }
            tmp[j][i] = warp_reduce_sum<WARP_SIZE>(tmp[j][i]);
            if (threadIdx.x == i && row0 + i < nrows_x) {
                dst[int64_t(j) * stride_col_dst + row0 + i] = store_value<dst_t>(tmp[j][i] * unscale);
            }
        }
    }
}

template <ggml_type type, int ncols>
static void launch_cols(const void * vx, const block_q8_1 * y1, const block_q8_1 * y2, const float * inverse,
                        void * dst, bool dst_bf16, int ncols_x, int nrows_x, int stride_col_dst,
                        cudaStream_t stream) {
    constexpr int rows_per_block = rows_per_block_for(ncols);
    const dim3 grid((nrows_x + rows_per_block - 1) / rows_per_block, 1, 1);
    const dim3 block(WARP_SIZE, nwarps_for(ncols), 1);
    const int stride_col_y = ncols_x / QK8_1;
    const bool two_pass = y2 != nullptr;
    if (dst_bf16 && two_pass) {
        mmvq_dual<type, ncols, nv_bfloat16, true><<<grid, block, 0, stream>>>(
            vx, y1, y2, inverse, static_cast<nv_bfloat16 *>(dst), ncols_x, nrows_x, stride_col_y, stride_col_dst);
    } else if (dst_bf16) {
        mmvq_dual<type, ncols, nv_bfloat16, false><<<grid, block, 0, stream>>>(
            vx, y1, y2, inverse, static_cast<nv_bfloat16 *>(dst), ncols_x, nrows_x, stride_col_y, stride_col_dst);
    } else if (two_pass) {
        mmvq_dual<type, ncols, float, true><<<grid, block, 0, stream>>>(
            vx, y1, y2, inverse, static_cast<float *>(dst), ncols_x, nrows_x, stride_col_y, stride_col_dst);
    } else {
        mmvq_dual<type, ncols, float, false><<<grid, block, 0, stream>>>(
            vx, y1, y2, inverse, static_cast<float *>(dst), ncols_x, nrows_x, stride_col_y, stride_col_dst);
    }
}

template <ggml_type type>
static void launch_type(const void * vx, const block_q8_1 * y1, const block_q8_1 * y2, const float * inverse,
                        void * dst, bool dst_bf16, int ncols_x, int nrows_x, int ncols_y, int stride_col_dst,
                        cudaStream_t stream) {
    switch (ncols_y) {
        case 1: launch_cols<type, 1>(vx, y1, y2, inverse, dst, dst_bf16, ncols_x, nrows_x, stride_col_dst, stream); break;
        case 2: launch_cols<type, 2>(vx, y1, y2, inverse, dst, dst_bf16, ncols_x, nrows_x, stride_col_dst, stream); break;
        case 3: launch_cols<type, 3>(vx, y1, y2, inverse, dst, dst_bf16, ncols_x, nrows_x, stride_col_dst, stream); break;
        case 4: launch_cols<type, 4>(vx, y1, y2, inverse, dst, dst_bf16, ncols_x, nrows_x, stride_col_dst, stream); break;
        case 5: launch_cols<type, 5>(vx, y1, y2, inverse, dst, dst_bf16, ncols_x, nrows_x, stride_col_dst, stream); break;
        case 6: launch_cols<type, 6>(vx, y1, y2, inverse, dst, dst_bf16, ncols_x, nrows_x, stride_col_dst, stream); break;
        case 7: launch_cols<type, 7>(vx, y1, y2, inverse, dst, dst_bf16, ncols_x, nrows_x, stride_col_dst, stream); break;
        case 8: launch_cols<type, 8>(vx, y1, y2, inverse, dst, dst_bf16, ncols_x, nrows_x, stride_col_dst, stream); break;
        default: throw std::runtime_error("Dual MMVQ supports 1..8 activation rows");
    }
}

static void mmvq(int type, const void * vx, const block_q8_1 * y1, const block_q8_1 * y2, const float * inverse,
                 void * dst, bool dst_bf16, int ncols_x, int nrows_x, int ncols_y, int stride_col_dst,
                 cudaStream_t stream) {
    switch (type) {
#define KAI_DUAL_CASE(T) case T: launch_type<T>(vx, y1, y2, inverse, dst, dst_bf16, ncols_x, nrows_x, ncols_y, stride_col_dst, stream); break;
        KAI_DUAL_CASE(GGML_TYPE_Q4_0)
        KAI_DUAL_CASE(GGML_TYPE_Q4_1)
        KAI_DUAL_CASE(GGML_TYPE_Q5_0)
        KAI_DUAL_CASE(GGML_TYPE_Q5_1)
        KAI_DUAL_CASE(GGML_TYPE_Q8_0)
        KAI_DUAL_CASE(GGML_TYPE_Q2_K)
        KAI_DUAL_CASE(GGML_TYPE_Q3_K)
        KAI_DUAL_CASE(GGML_TYPE_Q4_K)
        KAI_DUAL_CASE(GGML_TYPE_Q5_K)
        KAI_DUAL_CASE(GGML_TYPE_Q6_K)
        KAI_DUAL_CASE(GGML_TYPE_IQ2_S)
        KAI_DUAL_CASE(GGML_TYPE_IQ3_XXS)
        KAI_DUAL_CASE(GGML_TYPE_IQ3_S)
        KAI_DUAL_CASE(GGML_TYPE_IQ4_XS)
        KAI_DUAL_CASE(GGML_TYPE_IQ4_NL)
        KAI_DUAL_CASE(GGML_TYPE_IQ2_XXS)
        KAI_DUAL_CASE(GGML_TYPE_IQ2_XS)
        KAI_DUAL_CASE(GGML_TYPE_IQ1_S)
        KAI_DUAL_CASE(GGML_TYPE_IQ1_M)
#undef KAI_DUAL_CASE
        default: throw std::runtime_error("Unsupported dual MMVQ weight type");
    }
}

static inline size_t align256(size_t bytes) { return (bytes + 255) & ~size_t(255); }

static inline size_t pass_bytes(int64_t m, int64_t k) {
    return align256(size_t(m) * size_t(k / QK8_1) * sizeof(block_q8_1));
}

}  // namespace kai_dual

namespace kai_dual {

template <typename src_t>
static void prepare(const src_t * x, int64_t m, int64_t k, int64_t stride_x, block_q8_1 * y1, block_q8_1 * y2,
                    float * inverse, cudaStream_t stream) {
    const int nblocks = int(k / QK8_1);
    if (nblocks > PREP_THREADS) {
        throw std::runtime_error("Fused activation preparation supports K <= 32768");
    }
    const dim3 grid(unsigned(m), 1, 1);
    const int threads = (nblocks + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
    if (y2) {
        prepare_dual<src_t, true><<<grid, threads, 0, stream>>>(x, stride_x, int(k), y1, y2, inverse);
    } else {
        prepare_dual<src_t, false><<<grid, threads, 0, stream>>>(x, stride_x, int(k), y1, y2, inverse);
    }
}

}  // namespace kai_dual
