// SPDX-License-Identifier: Apache-2.0
// Experimental K1 row-pair GEMV: original packed words, FP32 activations.
#include <cuda_runtime.h>
#include <cstdint>

template <bool Shared>
__device__ __forceinline__ float state_product(uint32_t state, float2 value,
                                              const uint32_t* lut,
                                              const uint32_t* table) {
  const uint32_t hashed = state * (state + 1);
  const uint32_t index = (hashed >> 6) & 511;
  uint32_t pair;
  if constexpr (Shared) pair = table[index];
  else pair = __ldg(lut + index);
  pair ^= ((hashed >> 15) & 1) << 15;
  const float w0 = __uint_as_float((pair & 65535) << 16);
  const float w1 = __uint_as_float(pair & 0xffff0000u);
  return w0 * value.x + w1 * value.y;
}

template <int Warps, int Prefetch, bool Shared>
__global__ __launch_bounds__(Warps * 32, 2) void k1_row_pair_gemv(
    const float* __restrict__ x, const uint32_t* __restrict__ words,
    const uint32_t* __restrict__ lut, float* __restrict__ y, int m, int n) {
  x += size_t(blockIdx.y) * n;
  y += size_t(blockIdx.y) * m;
  __shared__ uint32_t table[Shared ? 512 : 1];
  if constexpr (Shared) {
    for (int i = threadIdx.x; i < 512; i += Warps * 32) table[i] = lut[i];
    __syncthreads();
  }
  const int lane = threadIdx.x & 31;
  const int j = lane & 3;
  const int group_lane = lane & ~3;
  const int logical_pair = blockIdx.x * Warps + (threadIdx.x >> 5);
  const int row = (logical_pair >> 3) * 16 + (logical_pair & 7);
  if (row >= m) return;
  const int input_tiles = n >> 4;
  const int tile_row_base = (row >> 4) * input_tiles * 8;
  const int word_offset = ((row & 7) + j) & 7;
  float a0 = 0, a1 = 0, b0 = 0, b1 = 0;
  for (int first = 0; first < input_tiles; first += 8 * Prefetch) {
    uint32_t packed[Prefetch];
    float2 low_x[Prefetch], high_x[Prefetch];
#pragma unroll
    for (int part = 0; part < Prefetch; ++part) {
      const int tile = first + (lane >> 2) + part * 8;
      packed[part] = tile < input_tiles && j < 2
          ? __ldg(words + tile_row_base + tile * 8 + word_offset) : 0;
      low_x[part] = tile < input_tiles
          ? __ldg(reinterpret_cast<const float2*>(x) + tile * 8 + j)
          : make_float2(0, 0);
      high_x[part] = tile < input_tiles
          ? __ldg(reinterpret_cast<const float2*>(x) + tile * 8 + j + 4)
          : make_float2(0, 0);
    }
#pragma unroll
    for (int part = 0; part < Prefetch; ++part) {
      const uint32_t low_word = __shfl_sync(
          0xffffffffu, packed[part], group_lane);
      const uint32_t high_word = __shfl_sync(
          0xffffffffu, packed[part], group_lane + (((j >> 1) + 1) >> 1));
      const uint32_t lo = (low_word >> (16 * ((j >> 1) & 1))) & 65535;
      const uint32_t hi = (high_word >> (16 * (((j >> 1) + 1) & 1))) & 65535;
      const uint32_t window = (lo << 16) | hi;
      const int shift = 16 - 8 * (j & 1);
      a0 += state_product<Shared>((window >> shift) & 65535, low_x[part], lut, table);
      a1 += state_product<Shared>((window >> (shift - 4)) & 65535, high_x[part], lut, table);
      b0 += state_product<Shared>((window >> (shift - 2)) & 65535, low_x[part], lut, table);
      b1 += state_product<Shared>((window >> (shift - 6)) & 65535, high_x[part], lut, table);
    }
  }
  float a = a0 + a1, b = b0 + b1;
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    a += __shfl_down_sync(0xffffffffu, a, offset);
    b += __shfl_down_sync(0xffffffffu, b, offset);
  }
  if (lane == 0) {
    y[row] = a;
    y[row + 8] = b;
  }
}

template <int Warps, int Prefetch>
void launch_k1(bool shared, const float* x, const uint32_t* words,
               const uint32_t* lut, float* y, int m, int n, cudaStream_t stream,
               int batch = 1) {
  const dim3 grid((m / 2 + Warps - 1) / Warps, batch);
  if (shared)
    k1_row_pair_gemv<Warps, Prefetch, true>
        <<<grid, Warps * 32, 0, stream>>>(x, words, lut, y, m, n);
  else
    k1_row_pair_gemv<Warps, Prefetch, false>
        <<<grid, Warps * 32, 0, stream>>>(x, words, lut, y, m, n);
}

extern "C" int yaqa_k1_gemv_rows(
    const float* x, const uint32_t* words, const uint32_t* lut, float* y,
    int m, int n, int batch, int warps, int prefetch, int shared,
    void* stream_pointer) {
  if (m <= 0 || n <= 0 || batch <= 0 || m % 16 || n % 16 ||
      (warps != 4 && warps != 8 && warps != 16) ||
      (prefetch != 1 && prefetch != 2 && prefetch != 4))
    return int(cudaErrorInvalidValue);
  auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
#define PICK(W, P) if (warps == W && prefetch == P) { \
  launch_k1<W, P>(shared != 0, x, words, lut, y, m, n, stream, batch); \
  return int(cudaGetLastError()); }
  PICK(4, 1) PICK(4, 2) PICK(4, 4)
  PICK(8, 1) PICK(8, 2) PICK(8, 4)
  PICK(16, 1) PICK(16, 2) PICK(16, 4)
#undef PICK
  return int(cudaErrorInvalidValue);
}

extern "C" int yaqa_k1_gemv(
    const float* x, const uint32_t* words, const uint32_t* lut, float* y,
    int m, int n, int warps, int prefetch, int shared, void* stream_pointer) {
  return yaqa_k1_gemv_rows(x, words, lut, y, m, n, 1, warps, prefetch, shared,
                          stream_pointer);
}
