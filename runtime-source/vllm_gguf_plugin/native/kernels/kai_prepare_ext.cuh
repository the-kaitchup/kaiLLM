// SPDX-License-Identifier: Apache-2.0
// Fused producers of the single-pass Q8_1 activation used by kai_fused_mmvq (decode, <= 8 rows):
//   * prepare_view:   activation of x[..., view] where view moves S-wide segments (A, B, S) -> (B, A, S)
//                     (llama.cpp's tiled Qwen3.5 V heads at the out_proj input; replaces a gather kernel);
//   * prepare_swiglu: activation of silu(gate) * up read straight from the merged gate_up output
//                     (replaces the act_and_mul kernel; same BF16 roundings as vLLM's SiluAndMul);
//   * norm_prepare:   Gemma RMSNorm (+ residual add, F32 residual stream, F32 (1 + w) multiplier) that also
//                     writes the activation of its BF16 output (replaces one prepare per consumer group).
// Quantization is prepare_dual's single pass on the same values (exact power-of-two row normalization,
// GGML Q8_1 blocks with sums of the quantized values), so the MMVQ consumers are unchanged.
#pragma once

#include "kai_dual_mmvq.cuh"

namespace kai_dual {

// Row amax, exact binary normalization and Q8_1 quantization of this thread's 32 values.
// Every thread of the CTA calls it (ib >= nblocks threads pass block_max = 0 and store nothing).
static __device__ __forceinline__ void quantize_row_single(float (&v)[QK8_1], float block_max, int ib, int nblocks,
                                                           int row, block_q8_1 * __restrict__ y1,
                                                           float * __restrict__ inverse) {
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
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
    store_block(y1 + int64_t(row) * nblocks + ib, q1, __floats2half2_rn(d1, d1 * float(isum1)));
}

template <typename src_t>
static __device__ __forceinline__ void load32(const src_t * __restrict__ p, float (&v)[QK8_1]) {
    constexpr int vec = 16 / sizeof(src_t);
    if (reinterpret_cast<uintptr_t>(p) % 16 == 0) {
#pragma unroll
        for (int c = 0; c < QK8_1 / vec; ++c) {
            const int4 packed = reinterpret_cast<const int4 *>(p)[c];
            const src_t * values = reinterpret_cast<const src_t *>(&packed);
#pragma unroll
            for (int j = 0; j < vec; ++j) {
                v[c * vec + j] = load_value(values + j);
            }
        }
    } else {
#pragma unroll
        for (int j = 0; j < QK8_1; ++j) {
            v[j] = load_value(p + j);
        }
    }
}

// v[j] += r[j] without a second 32-value register array (16-byte loads when aligned).
template <typename res_t>
static __device__ __forceinline__ void add32(const res_t * __restrict__ p, float (&v)[QK8_1]) {
    constexpr int vec = 16 / sizeof(res_t);
    if (reinterpret_cast<uintptr_t>(p) % 16 == 0) {
#pragma unroll
        for (int c = 0; c < QK8_1 / vec; ++c) {
            const int4 packed = reinterpret_cast<const int4 *>(p)[c];
            const res_t * values = reinterpret_cast<const res_t *>(&packed);
#pragma unroll
            for (int j = 0; j < vec; ++j) {
                v[c * vec + j] += load_value(values + j);
            }
        }
    } else {
#pragma unroll
        for (int j = 0; j < QK8_1; ++j) {
            v[j] += load_value(p + j);
        }
    }
}

static __device__ __forceinline__ float block_abs_max(const float (&v)[QK8_1]) {
    float m = 0.0f;
#pragma unroll
    for (int j = 0; j < QK8_1; ++j) {
        m = fmaxf(m, fabsf(v[j]));
    }
    return m;
}

// grid (m), block round_up(k/32, 32): stored column c of the activation is x column src(c).
template <typename src_t>
__launch_bounds__(PREP_THREADS, 1)
static __global__ void prepare_view(const src_t * __restrict__ x, const int64_t stride_x, const int k,
                                    const int a, const int b, const int s,
                                    block_q8_1 * __restrict__ y1, float * __restrict__ inverse) {
    const int row = blockIdx.x;
    const int ib = threadIdx.x;
    const int nblocks = k / QK8_1;
    float v[QK8_1];
    float block_max = 0.0f;
    if (ib < nblocks) {
        const int c = ib * QK8_1;
        const int src = ((c / s) % a) * b * s + (c / (a * s)) * s + c % s;
        load32(x + int64_t(row) * stride_x + src, v);
        block_max = block_abs_max(v);
    }
    quantize_row_single(v, block_max, ib, nblocks, row, y1, inverse);
}

template <typename src_t> static __device__ __forceinline__ float round_act(float v);
template <> __device__ __forceinline__ float round_act<float>(float v) { return v; }
template <> __device__ __forceinline__ float round_act<nv_bfloat16>(float v) { return __bfloat162float(__float2bfloat16(v)); }
template <> __device__ __forceinline__ float round_act<half>(float v) { return __half2float(__float2half(v)); }

// grid (m), block round_up(k/32, 32): activation of act[c] = round(round(silu(gu[c])) * gu[k + c]).
template <typename src_t>
__launch_bounds__(PREP_THREADS, 1)
static __global__ void prepare_swiglu(const src_t * __restrict__ gu, const int64_t stride, const int k,
                                      block_q8_1 * __restrict__ y1, float * __restrict__ inverse) {
    const int row = blockIdx.x;
    const int ib = threadIdx.x;
    const int nblocks = k / QK8_1;
    float v[QK8_1];
    float block_max = 0.0f;
    if (ib < nblocks) {
        const src_t * base = gu + int64_t(row) * stride + int64_t(ib) * QK8_1;
        load32(base, v);
#pragma unroll
        for (int j = 0; j < QK8_1; ++j) {
            v[j] = round_act<src_t>(v[j] / (1.0f + expf(-v[j])));
        }
        const src_t * up = base + k;
        constexpr int vec = 16 / sizeof(src_t);
        if (reinterpret_cast<uintptr_t>(up) % 16 == 0) {
#pragma unroll
            for (int c = 0; c < QK8_1 / vec; ++c) {
                const int4 packed = reinterpret_cast<const int4 *>(up)[c];
                const src_t * values = reinterpret_cast<const src_t *>(&packed);
#pragma unroll
                for (int j = 0; j < vec; ++j) {
                    v[c * vec + j] = round_act<src_t>(v[c * vec + j] * load_value(values + j));
                }
            }
        } else {
#pragma unroll
            for (int j = 0; j < QK8_1; ++j) {
                v[j] = round_act<src_t>(v[j] * load_value(up + j));
            }
        }
        block_max = block_abs_max(v);
    }
    quantize_row_single(v, block_max, ib, nblocks, row, y1, inverse);
}

// grid (m), block round_up(k/32, 32). y = bf16((x [+ r]) * rsqrt(mean((x [+ r])^2) + eps) * w32);
// residual_out (F32) = x + r; the activation quantizes the stored BF16 y.
template <typename src_t, typename res_t>
__launch_bounds__(PREP_THREADS, 1)
static __global__ void norm_prepare(const src_t * __restrict__ x, const int64_t stride_x,
                                    const res_t * __restrict__ r, const int64_t stride_r,
                                    float * __restrict__ r_out, const float * __restrict__ w32, const float eps,
                                    nv_bfloat16 * __restrict__ y, const int k,
                                    block_q8_1 * __restrict__ y1, float * __restrict__ inverse) {
    const int row = blockIdx.x;
    const int ib = threadIdx.x;
    const int nblocks = k / QK8_1;
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    float v[QK8_1];
    float ss = 0.0f;
    if (ib < nblocks) {
        load32(x + int64_t(row) * stride_x + int64_t(ib) * QK8_1, v);
        if (r) {
            add32(r + int64_t(row) * stride_r + int64_t(ib) * QK8_1, v);
            float * ro = r_out + int64_t(row) * k + int64_t(ib) * QK8_1;
#pragma unroll
            for (int j = 0; j < QK8_1; j += 4) {
                reinterpret_cast<float4 *>(ro)[j / 4] = make_float4(v[j], v[j + 1], v[j + 2], v[j + 3]);
            }
        }
#pragma unroll
        for (int j = 0; j < QK8_1; ++j) {
            ss += v[j] * v[j];
        }
    }
    __shared__ float warp_ss[PREP_THREADS / WARP_SIZE];
    ss = warp_reduce_sum<WARP_SIZE>(ss);
    if (lane == 0) {
        warp_ss[warp] = ss;
    }
    __syncthreads();
    if (warp == 0) {
        ss = lane < int(blockDim.x / WARP_SIZE) ? warp_ss[lane] : 0.0f;
        ss = warp_reduce_sum<WARP_SIZE>(ss);
        if (lane == 0) {
            warp_ss[0] = ss;
        }
    }
    __syncthreads();
    const float rstd = rsqrtf(warp_ss[0] / float(k) + eps);
    float block_max = 0.0f;
    if (ib < nblocks) {
        const float4 * w = reinterpret_cast<const float4 *>(w32 + int64_t(ib) * QK8_1);
        int4 * yo = reinterpret_cast<int4 *>(y + int64_t(row) * k + int64_t(ib) * QK8_1);
#pragma unroll
        for (int c = 0; c < QK8_1 / 8; ++c) {
            const float4 w0 = w[2 * c], w1 = w[2 * c + 1];
            const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
            int4 packed;
            nv_bfloat16 * h = reinterpret_cast<nv_bfloat16 *>(&packed);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                h[j] = __float2bfloat16(v[8 * c + j] * rstd * wv[j]);
                v[8 * c + j] = __bfloat162float(h[j]);
            }
            yo[c] = packed;
        }
        block_max = block_abs_max(v);
    }
    quantize_row_single(v, block_max, ib, nblocks, row, y1, inverse);
}

static inline int prep_threads(int64_t k) {
    const int nblocks = int(k / QK8_1);
    if (nblocks > PREP_THREADS) {
        throw std::runtime_error("Fused activation preparation supports K <= 32768");
    }
    return (nblocks + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
}

}  // namespace kai_dual
