// SPDX-License-Identifier: Apache-2.0
// Optional K4 whole-tile one-word/lane decode with bounded Split-K workspace.
#include <cuda_runtime.h>
#include <cstdint>

template <bool Shared>
__device__ __forceinline__ float tile_state_product(uint32_t state, float2 x,
                                                   const uint32_t* lut,
                                                   const uint32_t* table) {
  const uint32_t hash = state * (state + 1);
  const uint32_t index = (hash >> 6) & 511;
  uint32_t pair;
  if constexpr (Shared) pair = table[index];
  else pair = __ldg(lut + index);
  pair ^= hash & 32768;
  return __uint_as_float(pair << 16) * x.x +
         __uint_as_float(pair & 0xffff0000u) * x.y;
}

template <int Bits, int Warps, int Prefetch, bool Shared>
__global__ __launch_bounds__(Warps * 32, 2) void warp_tile_gemv(
    const float* __restrict__ x, const uint32_t* __restrict__ words,
    const uint32_t* __restrict__ lut, float* __restrict__ partial,
    int m, int n, int tiles_per_split) {
  x += size_t(blockIdx.z) * n;
  __shared__ uint32_t table[Shared ? 512 : 1];
  if constexpr (Shared) {
    for (int i = threadIdx.x; i < 512; i += Warps * 32) table[i] = lut[i];
    __syncthreads();
  }
  const int lane = threadIdx.x & 31;
  const int j = lane & 3;
  const int row_base = (blockIdx.x * Warps + (threadIdx.x >> 5)) * 16;
  if (row_base >= m) return;
  const int tiles = n >> 4;
  const int first_tile = blockIdx.y * tiles_per_split;
  const int end_tile = min(first_tile + tiles_per_split, tiles);
  const int row_word_base = (row_base >> 4) * tiles * (8 * Bits);
  float a0 = 0, a1 = 0, b0 = 0, b1 = 0;
  for (int first = first_tile; first < end_tile; first += Prefetch) {
    uint32_t packed[Prefetch];
    float2 low_x[Prefetch], high_x[Prefetch];
#pragma unroll
    for (int part = 0; part < Prefetch; ++part) {
      const int tile = first + part;
      packed[part] = tile < end_tile && lane < 8 * Bits
          ? __ldg(words + row_word_base + tile * (8 * Bits) + lane) : 0;
      low_x[part] = tile < end_tile
          ? __ldg(reinterpret_cast<const float2*>(x) + tile * 8 + j)
          : make_float2(0, 0);
      high_x[part] = tile < end_tile
          ? __ldg(reinterpret_cast<const float2*>(x) + tile * 8 + j + 4)
          : make_float2(0, 0);
    }
#pragma unroll
    for (int part = 0; part < Prefetch; ++part) {
      const uint32_t own = packed[part];
      const uint32_t next = __shfl_sync(
          0xffffffffu, own, (lane + 1) & 31);
      const uint32_t s0 = own & 65535;
      const uint32_t s1 = ((own & 255) << 8) | (own >> 24);
      const uint32_t s2 = own >> 16;
      const uint32_t s3 = (((own >> 16) & 255) << 8) | ((next >> 8) & 255);
      a0 += tile_state_product<Shared>(s0, low_x[part], lut, table);
      b0 += tile_state_product<Shared>(s1, low_x[part], lut, table);
      a1 += tile_state_product<Shared>(s2, high_x[part], lut, table);
      b1 += tile_state_product<Shared>(s3, high_x[part], lut, table);
    }
  }
  float a = a0 + a1, b = b0 + b1;
  a += __shfl_xor_sync(0xffffffffu, a, 1);
  b += __shfl_xor_sync(0xffffffffu, b, 1);
  a += __shfl_xor_sync(0xffffffffu, a, 2);
  b += __shfl_xor_sync(0xffffffffu, b, 2);
  if (j == 0) {
    const int row = row_base + (lane >> 2);
    partial[(blockIdx.z * gridDim.y + blockIdx.y) * m + row] = a;
    partial[(blockIdx.z * gridDim.y + blockIdx.y) * m + row + 8] = b;
  }
}

__global__ void reduce_tile_splits(const float* partial, float* y, int m, int splits) {
  partial += size_t(blockIdx.y) * splits * m;
  y += size_t(blockIdx.y) * m;
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < m) {
    float value = partial[row];
    for (int split = 1; split < splits; ++split) value += partial[split * m + row];
    y[row] = value;
  }
}

template <int Bits, int Warps, int Prefetch>
void launch_tile(bool shared, const float* x, const uint32_t* words,
                 const uint32_t* lut, float* partial, float* y,
                 int m, int n, int splits, cudaStream_t stream, int batch) {
  const dim3 grid((m / 16 + Warps - 1) / Warps, splits, batch);
  const int chunk = (n / 16 + splits - 1) / splits;
  float* destination = splits == 1 ? y : partial;
  if (shared)
    warp_tile_gemv<Bits, Warps, Prefetch, true>
        <<<grid, Warps * 32, 0, stream>>>(x, words, lut, destination, m, n, chunk);
  else
    warp_tile_gemv<Bits, Warps, Prefetch, false>
        <<<grid, Warps * 32, 0, stream>>>(x, words, lut, destination, m, n, chunk);
  if (splits > 1) reduce_tile_splits<<<dim3((m + 127) / 128, batch), 128, 0, stream>>>(partial, y, m, splits);
}

extern "C" int yaqa_k4_tile_rows(
    const float* x, const uint32_t* words, const uint32_t* lut, float* partial,
    float* y, int m, int n, int batch, int warps, int prefetch, int splits,
    int shared, void* stream_pointer) {
  if (m <= 0 || n <= 0 || batch <= 0 || m % 16 || n % 16 ||
      (warps != 4 && warps != 8) || (prefetch != 2 && prefetch != 4 && prefetch != 8) ||
      (splits != 1 && splits != 2 && splits != 4 && splits != 8))
    return int(cudaErrorInvalidValue);
  auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
#define PICK(W, P) if (warps == W && prefetch == P) { \
  launch_tile<4, W, P>(shared != 0, x, words, lut, partial, y, m, n, splits, stream, batch); \
  return int(cudaGetLastError()); }
  PICK(4,2) PICK(4,4) PICK(4,8) PICK(8,2) PICK(8,4) PICK(8,8)
#undef PICK
  return int(cudaErrorInvalidValue);
}
