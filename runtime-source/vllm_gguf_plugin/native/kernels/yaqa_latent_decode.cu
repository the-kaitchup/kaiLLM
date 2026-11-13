// SPDX-License-Identifier: Apache-2.0
// Decode each original QTIP1--4 tile once into bounded transient F16 storage.
#include <cuda_runtime.h>
#include <cstdint>

template<int Bits>
__device__ __forceinline__ void four_states(uint32_t packed, int lane,
                                           uint32_t* states) {
    const int first = (Bits * lane) >> 2;
    const int next = first == 8 * Bits - 1 ? 0 : first + 1;
    const int shift = (8 * Bits * lane) & 31;
    const uint32_t q0 = __shfl_sync(0xffffffffu, packed, first);
    const uint32_t q1 = __shfl_sync(0xffffffffu, packed, next);
    const uint32_t a = (q0 << 16) | (q0 >> 16);
    const uint32_t b = (q1 << 16) | (q1 >> 16);
    const uint32_t top = __funnelshift_l(b, a, shift);
    states[0] = top >> 16;
    states[1] = (top >> (16 - 2 * Bits)) & 65535;
    states[2] = (top >> (16 - 4 * Bits)) & 65535;
    if constexpr (6 * Bits <= 16)
        states[3] = (top >> (16 - 6 * Bits)) & 65535;
    else
        states[3] = __funnelshift_l(b, a, shift + 6 * Bits - 16) & 65535;
}

template<int Bits, int Warps, bool VectorStore>
__global__ __launch_bounds__(Warps * 32, 2) void decode_latent_cuda(
    const uint32_t* __restrict__ words,
    const uint32_t* __restrict__ signed_half_lut,
    uint32_t* __restrict__ dense_pairs, int m, int n) {
    __shared__ uint32_t lut[1024];
    __shared__ __align__(16) uint32_t scratch[VectorStore ? Warps * 128 : 1];
    for (int i = threadIdx.x; i < 1024; i += Warps * 32)
        lut[i] = __ldg(signed_half_lut + i);
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int column_tiles = n / 16;
    const int tile_count = (m / 16) * column_tiles;
    for (int tile = blockIdx.x * Warps + warp; tile < tile_count;
         tile += gridDim.x * Warps) {
        const uint32_t packed = lane < 8 * Bits
            ? __ldg(words + tile * (8 * Bits) + lane) : 0;
        uint32_t states[4], values[4];
        four_states<Bits>(packed, lane, states);
        #pragma unroll
        for (int group = 0; group < 4; ++group) {
            const uint32_t state = states[group];
            const uint32_t hash = state * (state + 1);
            values[group] = lut[(hash >> 6) & 1023];
        }
        const int row_tile = tile / column_tiles, col_tile = tile % column_tiles;
        const int r0 = lane >> 2, j = lane & 3;
        if constexpr (VectorStore) {
            uint32_t* temp = scratch + warp * 128;
            temp[r0 * 8 + j] = values[0];
            temp[(r0 + 8) * 8 + j] = values[1];
            temp[r0 * 8 + j + 4] = values[2];
            temp[(r0 + 8) * 8 + j + 4] = values[3];
            __syncwarp();
            const uint4 value = *reinterpret_cast<const uint4*>(temp + lane * 4);
            const int row = row_tile * 16 + lane / 2;
            const int pair_col = col_tile * 8 + (lane & 1) * 4;
            *reinterpret_cast<uint4*>(dense_pairs + size_t(row) * (n / 2) + pair_col)
                = value;
            __syncwarp();
        } else {
            const int row = row_tile * 16 + r0;
            const size_t base = size_t(row) * (n / 2) + col_tile * 8 + j;
            dense_pairs[base] = values[0];
            dense_pairs[base + size_t(8) * (n / 2)] = values[1];
            dense_pairs[base + 4] = values[2];
            dense_pairs[base + size_t(8) * (n / 2) + 4] = values[3];
        }
    }
}

template<int Bits, int Warps>
void launch(int vector_store, const uint32_t* words, const uint32_t* lut,
            uint32_t* dense, int m, int n, int grid_cap, cudaStream_t stream) {
    int blocks = ((m / 16) * (n / 16) + Warps - 1) / Warps;
    if (grid_cap > 0) blocks = min(blocks, grid_cap);
    if (vector_store)
        decode_latent_cuda<Bits,Warps,true><<<blocks,Warps*32,0,stream>>>(
            words,lut,dense,m,n);
    else
        decode_latent_cuda<Bits,Warps,false><<<blocks,Warps*32,0,stream>>>(
            words,lut,dense,m,n);
}

extern "C" int yaqa_latent_cuda_run(
    const uint32_t* words, const uint32_t* signed_half_lut, uint32_t* dense,
    int m, int n, int bits, int warps, int vector_store, int grid_cap,
    void* raw_stream) {
    if (!words || !signed_half_lut || !dense || m <= 0 || n <= 0 || m % 16 || n % 16
        || bits < 1 || bits > 4 || (warps != 4 && warps != 8)
        || (vector_store != 0 && vector_store != 1) || grid_cap < 0
        || (reinterpret_cast<uintptr_t>(words) & 3)
        || (reinterpret_cast<uintptr_t>(signed_half_lut) & 3)
        || (reinterpret_cast<uintptr_t>(dense) & (vector_store ? 15 : 3)))
        return int(cudaErrorInvalidValue);
    const auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
    #define PICK(B,W) if (bits==B && warps==W) { \
        launch<B,W>(vector_store,words,signed_half_lut,dense,m,n,grid_cap,stream); \
        return int(cudaGetLastError()); }
    PICK(1,4) PICK(1,8) PICK(2,4) PICK(2,8)
    PICK(3,4) PICK(3,8) PICK(4,4) PICK(4,8)
    #undef PICK
    return int(cudaErrorInvalidValue);
}
