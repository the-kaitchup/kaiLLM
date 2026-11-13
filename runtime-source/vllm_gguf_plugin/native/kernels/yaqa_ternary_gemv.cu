// SPDX-License-Identifier: Apache-2.0
// Experimental original five-trit bytes, BF16 weights/input and FP32 sums.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

template<int Digit>
__device__ __forceinline__ int trit(uint32_t byte) {
    constexpr int powers[] = {1,3,9,27,81};
    return int((byte / powers[Digit]) % 3) - 1;
}

template<int N, int Batch, int Warps, int Prefetch>
__global__ void ternary_gemv(
    const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ w,
    const float* __restrict__ scales, float* __restrict__ y, int m, int batch) {
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * Warps + threadIdx.x / 32;
    if (row >= m) return;
    const int start_digit = (row * N) % 5;
    const int bytes = (N + start_digit + 4) / 5;
    const uint8_t* row_bytes = w + row * N / 5;
    const float amplitude = __bfloat162float(__float2bfloat16_rn(scales[row]));
    float accum[Batch][2] = {};
    for (int first = 0; first < bytes; first += 32 * Prefetch) {
        uint32_t packed[Prefetch];
        #pragma unroll
        for (int p = 0; p < Prefetch; ++p) {
            const int byte = first + lane + p * 32;
            packed[p] = byte < bytes ? __ldg(row_bytes + byte) : 0;
        }
        #pragma unroll
        for (int p = 0; p < Prefetch; ++p) {
            const int column = (first + lane + p * 32) * 5 - start_digit;
            #pragma unroll
            for (int d = 0; d < 5; ++d) {
                int code;
                if (d == 0) code = trit<0>(packed[p]);
                else if (d == 1) code = trit<1>(packed[p]);
                else if (d == 2) code = trit<2>(packed[p]);
                else if (d == 3) code = trit<3>(packed[p]);
                else code = trit<4>(packed[p]);
                const float weight = __fmul_rn(float(code), amplitude);
                const int c = column + d;
                if (c >= 0 && c < N) {
                    #pragma unroll
                    for (int b = 0; b < Batch; ++b) {
                        if (b < batch) {
                            const float value = __bfloat162float(__ldg(x + b * N + c));
                            accum[b][d % 2] = fmaf(value, weight, accum[b][d % 2]);
                        }
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int b = 0; b < Batch; ++b) {
        if (b < batch) {
            float value = accum[b][0] + accum[b][1];
            #pragma unroll
            for (int delta = 16; delta; delta /= 2)
                value += __shfl_down_sync(0xffffffff, value, delta);
            if (lane == 0) y[b * m + row] = value;
        }
    }
}

template<int N, int Batch>
bool choose_ternary(const __nv_bfloat16* x, const uint8_t* w, const float* scales,
                    float* y, int m, int batch, int warps, int prefetch,
                    cudaStream_t stream) {
    #define CASE(W,P) if (warps==W && prefetch==P) { \
        ternary_gemv<N,Batch,W,P><<<(m+W-1)/W,32*W,0,stream>>>(x,w,scales,y,m,batch); \
        return true; }
    CASE(4,1) CASE(4,2) CASE(4,4)
    CASE(8,1) CASE(8,2) CASE(8,4)
    #undef CASE
    return false;
}

extern "C" int yaqa_ternary_run(
    const __nv_bfloat16* x, const uint8_t* w, const float* scales, float* y,
    int m, int n, int batch, int warps, int prefetch, void* raw_stream) {
    if (m <= 0 || batch < 1 || batch > 8) return -1;
    const auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
    bool okay = false;
    #define WIDTH(N) if (n == N) { \
        if (batch==1) okay=choose_ternary<N,1>(x,w,scales,y,m,batch,warps,prefetch,stream); \
        else okay=choose_ternary<N,8>(x,w,scales,y,m,batch,warps,prefetch,stream); }
    WIDTH(2048) WIDTH(7680)
    #undef WIDTH
    return okay ? int(cudaGetLastError()) : -2;
}
