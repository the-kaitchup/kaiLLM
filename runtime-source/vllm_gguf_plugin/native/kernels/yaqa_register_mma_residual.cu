// SPDX-License-Identifier: Apache-2.0
// Experimental L16/V2 QTIP decoder straight into MMA operand registers.
// Fragment layout: NVIDIA PTX ISA, mma.m16n8k16 floating-point fragments.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

template<bool Small>
__device__ __forceinline__ void mma(float* c, const uint32_t* a,
                                    uint32_t b0, uint32_t b1) {
    if constexpr (Small) {
        float unused0 = 0, unused1 = 0;
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(c[0]), "+f"(c[1]), "+f"(unused0), "+f"(unused1)
            : "r"(a[0]), "r"(0), "r"(a[2]), "r"(0), "r"(b0), "r"(b1));
    } else {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
}

template<int BITS>
__device__ __forceinline__ uint32_t window(uint32_t packed, int word) {
    const int next = (word + 1) % (16 * BITS);
    const uint32_t q0 = __shfl_sync(0xffffffff, packed, word / 2);
    const uint32_t q1 = __shfl_sync(0xffffffff, packed, next / 2);
    const uint32_t lo = (q0 >> (16 * (word % 2))) & 65535;
    const uint32_t hi = (q1 >> (16 * (next % 2))) & 65535;
    return (lo << 16) | hi;
}

template<int BITS, int WARPS, int RING, bool SHARED_LUT, bool Small>
__global__ void register_qtip(
    const half* __restrict__ input, const uint32_t* __restrict__ weights,
    const uint32_t* __restrict__ signed_half_lut, float* __restrict__ output,
    int rows, int width, int batch, int splits) {
    __shared__ uint32_t table[SHARED_LUT ? 1024 : 1];
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    if (SHARED_LUT) {
        for (int i = tid; i < 1024; i += 32 * WARPS)
            table[i] = __ldg(signed_half_lut + i);
        __syncthreads();
    }
    const int output_row = (blockIdx.x * WARPS + warp) * 16;
    const int sample0 = blockIdx.y * 16 + lane / 4;
    const int sample1 = sample0 + 8;
    const int column = (lane % 4) * 2;
    const int chunks = (width / 16 + splits - 1) / splits;
    const int start_tile = blockIdx.z * chunks;
    const int end_tile = min(width / 16, start_tile + chunks);
    float c0[4] = {0,0,0,0}, c1[4] = {0,0,0,0};
    for (int base = start_tile; base < end_tile; base += RING) {
        uint32_t encoded[RING], a[RING][4];
        #pragma unroll
        for (int i = 0; i < RING; ++i) {
            const int tile_k = base + i;
            encoded[i] = lane < 8 * BITS && output_row < rows && tile_k < end_tile
                ? __ldg(weights + (output_row / 16 * (width / 16) + tile_k)
                         * (8 * BITS) + lane) : 0;
            const auto* x0 = reinterpret_cast<const uint32_t*>(
                input + sample0 * width + tile_k * 16 + column);
            const bool valid0 = sample0 < batch && tile_k < end_tile;
            a[i][0] = valid0 ? __ldg(x0) : 0;
            a[i][2] = valid0 ? __ldg(x0 + 4) : 0;
            if constexpr (!Small) {
                const auto* x1 = reinterpret_cast<const uint32_t*>(
                    input + sample1 * width + tile_k * 16 + column);
                const bool valid1 = sample1 < batch && tile_k < end_tile;
                a[i][1] = valid1 ? __ldg(x1) : 0;
                a[i][3] = valid1 ? __ldg(x1 + 4) : 0;
            }
        }
        #pragma unroll
        for (int i = 0; i < RING; ++i) {
            const int start = 4 * lane * (2 * BITS);
            const int word = start / 16, initial_shift = start % 16;
            const uint32_t w0 = window<BITS>(encoded[i], word);
            uint32_t w1 = 0;
            if (BITS > 2) w1 = window<BITS>(encoded[i], (word + 1) % (16 * BITS));
            uint32_t b[4];
            #pragma unroll
            for (int group = 0; group < 4; ++group) {
                const int shift = initial_shift + group * (2 * BITS);
                const uint32_t w = shift < 16 ? w0 : w1;
                const uint32_t state = (w >> (16 - shift % 16)) & 65535;
                const uint32_t hash = state * (state + 1);
                const int code = (hash >> 6) & 1023;
                b[group] = SHARED_LUT ? table[code] : __ldg(signed_half_lut + code);
            }
            mma<Small>(c0,a[i],b[0],b[2]);
            mma<Small>(c1,a[i],b[1],b[3]);
        }
    }
    const int r0 = output_row + (lane % 4) * 2;
    const int r1 = r0 + 8;
    const size_t stride = static_cast<size_t>(batch) * rows;
    #define STORE(S,R,C,I) if (S < batch && R < rows) { \
        float* dest = output + blockIdx.z * stride + static_cast<size_t>(S) * rows + R; \
        *reinterpret_cast<float2*>(dest) = make_float2(C[I],C[I+1]); }
    STORE(sample0,r0,c0,0)
    STORE(sample0,r1,c1,0)
    if constexpr (!Small) {
        STORE(sample1,r0,c0,2)
        STORE(sample1,r1,c1,2)
    }
    #undef STORE
}

__global__ void sum_register_partials(const float* p, float* y, int count, int splits) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float value = 0;
    for (int s = 0; s < splits; ++s) value += p[s * count + i];
    y[i] = value;
}

template<int BITS, int WARPS, int RING>
void launch_register(bool shared, const half* x, const uint32_t* w,
                     const uint32_t* lut, float* y, int m, int n, int b,
                     int splits, cudaStream_t stream) {
    const dim3 grid((m + 16 * WARPS - 1) / (16 * WARPS),(b + 15) / 16,splits);
    #define LAUNCH(S,L) register_qtip<BITS,WARPS,RING,S,L><<<grid,32*WARPS,0,stream>>>(x,w,lut,y,m,n,b,splits)
    if (b <= 8) {
        if (shared) LAUNCH(true,true);
        else LAUNCH(false,true);
    } else {
        if (shared) LAUNCH(true,false);
        else LAUNCH(false,false);
    }
    #undef LAUNCH
}

template<int BITS>
bool choose_register(int warps, int ring, bool shared, const half* x,
                     const uint32_t* w, const uint32_t* lut, float* y,
                     int m, int n, int b, int splits, cudaStream_t stream) {
    #define CASE(W,R) if (warps==W && ring==R) { \
        launch_register<BITS,W,R>(shared,x,w,lut,y,m,n,b,splits,stream); return true; }
    CASE(2,2) CASE(2,4) CASE(2,8)
    CASE(4,2) CASE(4,4) CASE(4,8)
    CASE(8,2) CASE(8,4) CASE(8,8)
    #undef CASE
    return false;
}

extern "C" int yaqa_register_mma_run(
    const half* x, const uint32_t* w, const uint32_t* lut, float* partial, float* y,
    int m, int n, int b, int bits, int warps, int ring, int shared, int splits,
    void* raw_stream) {
    if (splits < 1 || splits > 64 || n % 16 || m % 16) return -1;
    const auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
    float* target = splits == 1 ? y : partial;
    bool found = false;
    #define BIT(B) case B: found=choose_register<B>(warps,ring,shared,x,w,lut,target,m,n,b,splits,stream); break
    switch (bits) { BIT(1);BIT(2);BIT(3);BIT(4);default:return -2; }
    #undef BIT
    if (!found) return -3;
    if (splits>1) sum_register_partials<<<(b*m+255)/256,256,0,stream>>>(partial,y,b*m,splits);
    return static_cast<int>(cudaGetLastError());
}

// Standalone residual-A export; old definitions above are byte-identical.
template<int BITS, int WARPS, int RING, bool SHARED_LUT, bool Small>
__global__ void register_qtip_residual(
    const half* __restrict__ input, const half* __restrict__ residual,
    const uint32_t* __restrict__ weights,
    const uint32_t* __restrict__ signed_half_lut, float* __restrict__ output,
    int rows, int width, int batch, int splits) {
    __shared__ uint32_t table[SHARED_LUT ? 1024 : 1];
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    if (SHARED_LUT) {
        for (int i = tid; i < 1024; i += 32 * WARPS)
            table[i] = __ldg(signed_half_lut + i);
        __syncthreads();
    }
    const int output_row = (blockIdx.x * WARPS + warp) * 16;
    const int sample0 = blockIdx.y * 16 + lane / 4;
    const int sample1 = sample0 + 8;
    const int column = (lane % 4) * 2;
    const int chunks = (width / 16 + splits - 1) / splits;
    const int start_tile = blockIdx.z * chunks;
    const int end_tile = min(width / 16, start_tile + chunks);
    float c0[4] = {0,0,0,0}, c1[4] = {0,0,0,0};
    float low_c0[4] = {0,0,0,0}, low_c1[4] = {0,0,0,0};
    for (int base = start_tile; base < end_tile; base += RING) {
        uint32_t encoded[RING], a[RING][4], low_a[RING][4];
        #pragma unroll
        for (int i = 0; i < RING; ++i) {
            const int tile_k = base + i;
            encoded[i] = lane < 8 * BITS && output_row < rows && tile_k < end_tile
                ? __ldg(weights + (output_row / 16 * (width / 16) + tile_k)
                         * (8 * BITS) + lane) : 0;
            const auto* x0 = reinterpret_cast<const uint32_t*>(
                input + sample0 * width + tile_k * 16 + column);
            const bool valid0 = sample0 < batch && tile_k < end_tile;
            a[i][0] = valid0 ? __ldg(x0) : 0;
            a[i][2] = valid0 ? __ldg(x0 + 4) : 0;
            const auto* low_x0 = reinterpret_cast<const uint32_t*>(
                residual + sample0 * width + tile_k * 16 + column);
            low_a[i][0] = valid0 ? __ldg(low_x0) : 0;
            low_a[i][2] = valid0 ? __ldg(low_x0 + 4) : 0;
            if constexpr (!Small) {
                const auto* x1 = reinterpret_cast<const uint32_t*>(
                    input + sample1 * width + tile_k * 16 + column);
                const bool valid1 = sample1 < batch && tile_k < end_tile;
                a[i][1] = valid1 ? __ldg(x1) : 0;
                a[i][3] = valid1 ? __ldg(x1 + 4) : 0;
                const auto* low_x1 = reinterpret_cast<const uint32_t*>(
                    residual + sample1 * width + tile_k * 16 + column);
                low_a[i][1] = valid1 ? __ldg(low_x1) : 0;
                low_a[i][3] = valid1 ? __ldg(low_x1 + 4) : 0;
            }
        }
        #pragma unroll
        for (int i = 0; i < RING; ++i) {
            const int start = 4 * lane * (2 * BITS);
            const int word = start / 16, initial_shift = start % 16;
            const uint32_t w0 = window<BITS>(encoded[i], word);
            uint32_t w1 = 0;
            if (BITS > 2) w1 = window<BITS>(encoded[i], (word + 1) % (16 * BITS));
            uint32_t b[4];
            #pragma unroll
            for (int group = 0; group < 4; ++group) {
                const int shift = initial_shift + group * (2 * BITS);
                const uint32_t w = shift < 16 ? w0 : w1;
                const uint32_t state = (w >> (16 - shift % 16)) & 65535;
                const uint32_t hash = state * (state + 1);
                const int code = (hash >> 6) & 1023;
                b[group] = SHARED_LUT ? table[code] : __ldg(signed_half_lut + code);
            }
            mma<Small>(c0,a[i],b[0],b[2]);
            mma<Small>(c1,a[i],b[1],b[3]);
            mma<Small>(low_c0,low_a[i],b[0],b[2]);
            mma<Small>(low_c1,low_a[i],b[1],b[3]);
        }
    }
    const int r0 = output_row + (lane % 4) * 2;
    const int r1 = r0 + 8;
    const size_t stride = static_cast<size_t>(batch) * rows;
    #define STORE(S,R,C,L,I) if (S < batch && R < rows) { \
        float* dest = output + blockIdx.z * stride + static_cast<size_t>(S) * rows + R; \
        *reinterpret_cast<float2*>(dest) = make_float2(C[I] + L[I],C[I+1] + L[I+1]); }
    STORE(sample0,r0,c0,low_c0,0)
    STORE(sample0,r1,c1,low_c1,0)
    if constexpr (!Small) {
        STORE(sample1,r0,c0,low_c0,2)
        STORE(sample1,r1,c1,low_c1,2)
    }
    #undef STORE
}

template<int BITS, int WARPS, int RING>
void launch_register_residual(bool shared, const half* x, const half* low_x, const uint32_t* w,
                     const uint32_t* lut, float* y, int m, int n, int b,
                     int splits, cudaStream_t stream) {
    const dim3 grid((m + 16 * WARPS - 1) / (16 * WARPS),(b + 15) / 16,splits);
    #define LAUNCH(S,L) register_qtip_residual<BITS,WARPS,RING,S,L><<<grid,32*WARPS,0,stream>>>(x,low_x,w,lut,y,m,n,b,splits)
    if (b <= 8) {
        if (shared) LAUNCH(true,true);
        else LAUNCH(false,true);
    } else {
        if (shared) LAUNCH(true,false);
        else LAUNCH(false,false);
    }
    #undef LAUNCH
}

template<int BITS>
bool choose_register_residual(int warps, int ring, bool shared, const half* x, const half* low_x,
                     const uint32_t* w, const uint32_t* lut, float* y,
                     int m, int n, int b, int splits, cudaStream_t stream) {
    #define CASE(W,R) if (warps==W && ring==R) { \
        launch_register_residual<BITS,W,R>(shared,x,low_x,w,lut,y,m,n,b,splits,stream); return true; }
    CASE(2,2) CASE(2,4) CASE(2,8)
    CASE(4,2) CASE(4,4) CASE(4,8)
    CASE(8,2) CASE(8,4) CASE(8,8)
    #undef CASE
    return false;
}

extern "C" int yaqa_register_mma_residual_run(
    const half* x, const half* low_x, const uint32_t* w, const uint32_t* lut,
    float* partial, float* y, int m, int n, int b, int bits, int warps, int ring,
    int shared, int splits, void* raw_stream) {
    if (!x || !low_x || !w || !lut || !y || (splits > 1 && !partial) ||
        uintptr_t(x) % 4 || uintptr_t(low_x) % 4 || uintptr_t(w) % 4 ||
        uintptr_t(lut) % 4 || uintptr_t(y) % 8 || uintptr_t(partial) % 8 ||
        m <= 0 || n <= 0 || b <= 0 || b > 32 || n % 16 || m % 16 ||
        uint64_t(b) * m > 0x7fffffffu ||
        uint64_t(m / 16) * (n / 16) * (8 * bits) > 0x7fffffffu ||
        splits < 1 || splits > 64 || (splits & (splits - 1)) ||
        uint64_t(b) * m * splits * sizeof(float) > (4u << 20) ||
        (warps != 2 && warps != 4 && warps != 8) ||
        (ring != 2 && ring != 4 && ring != 8) || (shared != 0 && shared != 1) ||
        bits < 1 || bits > 4) return int(cudaErrorInvalidValue);
    const auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
    float* target = splits == 1 ? y : partial;
    bool found = false;
#define BIT_RES(B) case B: found=choose_register_residual<B>(warps,ring,shared,x,low_x,w,lut,target,m,n,b,splits,stream); break
    switch(bits) { BIT_RES(1); BIT_RES(2); BIT_RES(3); BIT_RES(4); default: return int(cudaErrorInvalidValue); }
#undef BIT_RES
    if (!found) return int(cudaErrorInvalidValue);
    if (splits > 1) sum_register_partials<<<(b*m+255)/256,256,0,stream>>>(partial,y,b*m,splits);
    return int(cudaGetLastError());
}
