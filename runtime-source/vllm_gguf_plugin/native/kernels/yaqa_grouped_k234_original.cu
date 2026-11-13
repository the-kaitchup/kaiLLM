// SPDX-License-Identifier: Apache-2.0
// Unselected: per-CTA original K2/K3/K4 arithmetic and policies.
#include <cuda_runtime.h>
#include <cstdint>
template <bool Shared>
__device__ __forceinline__ float original_k2_product(uint32_t state, float2 value,
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
__device__ __forceinline__ void original_k2_body(
    const float* x, const uint32_t* words, const uint32_t* lut,
    float* y, const uint32_t* table, int m, int n, int row_block) {
  const int lane = threadIdx.x & 31;
  const int j = lane & 3;
  const int group_lane = lane & ~3;
  const int logical_pair = row_block * Warps + (threadIdx.x >> 5);
  const int row = (logical_pair >> 3) * 16 + (logical_pair & 7);
  if (row >= m) return;
  const int input_tiles = n >> 4;
  const int tile_row_base = (row >> 4) * input_tiles * 16;
  const int word_offset = (2 * (row & 7) + j) & 15;
  float a0 = 0, a1 = 0, b0 = 0, b1 = 0;
  for (int first = 0; first < input_tiles; first += 8 * Prefetch) {
    uint32_t packed[Prefetch];
    float2 low_x[Prefetch], high_x[Prefetch];
#pragma unroll
    for (int part = 0; part < Prefetch; ++part) {
      const int tile = first + (lane >> 2) + part * 8;
      packed[part] = tile < input_tiles && j < 3
          ? __ldg(words + tile_row_base + tile * 16 + word_offset) : 0;
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
          0xffffffffu, packed[part], group_lane + (j >> 1));
      const uint32_t high_word = __shfl_sync(
          0xffffffffu, packed[part], group_lane + ((j + 1) >> 1));
      const uint32_t lo = (low_word >> (16 * (j & 1))) & 65535;
      const uint32_t hi = (high_word >> (16 * ((j + 1) & 1))) & 65535;
      const uint32_t window = (lo << 16) | hi;
      a0 += original_k2_product<Shared>(window >> 16, low_x[part], lut, table);
      a1 += original_k2_product<Shared>((window >> 8) & 65535, high_x[part], lut, table);
      b0 += original_k2_product<Shared>((window >> 12) & 65535, low_x[part], lut, table);
      b1 += original_k2_product<Shared>((window >> 4) & 65535, high_x[part], lut, table);
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
template <bool Shared>
__device__ __forceinline__ float original_k3_product(uint32_t state, float2 x,
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
template <int Warps, int Prefetch, bool Shared>
__device__ __forceinline__ void original_k3_body(
    const float* x, const uint32_t* words, const uint32_t* lut,
    float* partial, const uint32_t* table, int m, int n, int row_block,
    int split, int splits, int tiles_per_split) {
  constexpr int Bits = 3;
  const int lane = threadIdx.x & 31;
  const int j = lane & 3;
  const int row_base = (row_block * Warps + (threadIdx.x >> 5)) * 16;
  if (row_base >= m) return;
  const int tiles = n >> 4;
  const int first_tile = split * tiles_per_split;
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
      const int first_word = (3 * lane) >> 2;
      const int next_word = first_word == 23 ? 0 : first_word + 1;
      const int shift = (24 * lane) & 31;
      const uint32_t q0 = __shfl_sync(0xffffffffu, packed[part], first_word);
      const uint32_t q1 = __shfl_sync(0xffffffffu, packed[part], next_word);
      const uint32_t swapped0 = (q0 << 16) | (q0 >> 16);
      const uint32_t swapped1 = (q1 << 16) | (q1 >> 16);
      const uint64_t aligned = ((uint64_t(swapped0) << 32) | swapped1) << shift;
      a0 += original_k3_product<Shared>((aligned >> 48) & 65535, low_x[part], lut, table);
      b0 += original_k3_product<Shared>((aligned >> 42) & 65535, low_x[part], lut, table);
      a1 += original_k3_product<Shared>((aligned >> 36) & 65535, high_x[part], lut, table);
      b1 += original_k3_product<Shared>((aligned >> 30) & 65535, high_x[part], lut, table);
    }
  }
  float a = a0 + a1, b = b0 + b1;
  a += __shfl_xor_sync(0xffffffffu, a, 1);
  b += __shfl_xor_sync(0xffffffffu, b, 1);
  a += __shfl_xor_sync(0xffffffffu, a, 2);
  b += __shfl_xor_sync(0xffffffffu, b, 2);
  if (j == 0) {
    const int row = row_base + (lane >> 2);
    partial[(size_t(blockIdx.z) * splits + split) * m + row] = a;
    partial[(size_t(blockIdx.z) * splits + split) * m + row + 8] = b;
  }

}
template <bool Shared>
__device__ __forceinline__ float original_k4_product(uint32_t state, float2 x,
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
template <int Warps, int Prefetch, bool Shared>
__device__ __forceinline__ void original_k4_body(
    const float* x, const uint32_t* words, const uint32_t* lut,
    float* partial, const uint32_t* table, int m, int n, int row_block,
    int split, int splits, int tiles_per_split) {
  constexpr int Bits = 4;
  const int lane = threadIdx.x & 31;
  const int j = lane & 3;
  const int row_base = (row_block * Warps + (threadIdx.x >> 5)) * 16;
  if (row_base >= m) return;
  const int tiles = n >> 4;
  const int first_tile = split * tiles_per_split;
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
      a0 += original_k4_product<Shared>(s0, low_x[part], lut, table);
      b0 += original_k4_product<Shared>(s1, low_x[part], lut, table);
      a1 += original_k4_product<Shared>(s2, high_x[part], lut, table);
      b1 += original_k4_product<Shared>(s3, high_x[part], lut, table);
    }
  }
  float a = a0 + a1, b = b0 + b1;
  a += __shfl_xor_sync(0xffffffffu, a, 1);
  b += __shfl_xor_sync(0xffffffffu, b, 1);
  a += __shfl_xor_sync(0xffffffffu, a, 2);
  b += __shfl_xor_sync(0xffffffffu, b, 2);
  if (j == 0) {
    const int row = row_base + (lane >> 2);
    partial[(size_t(blockIdx.z) * splits + split) * m + row] = a;
    partial[(size_t(blockIdx.z) * splits + split) * m + row + 8] = b;
  }

}

// Descriptor: wire,LUT,N,S,firstCTA,CTAcount,partialOffset,bits,warps,prefetch,shared.
// Input is [G,B,K]; partial storage is independently [B,S,N] per projection.
template<int MaxWarps>
__global__ __launch_bounds__(MaxWarps * 32, 2) void grouped_k234_original(
    const float* __restrict__ grouped_x,
    const uint64_t* __restrict__ descriptors,
    float* __restrict__ partial, int groups, int n, int batch) {
  int group = 0;
  while (group < groups && uint64_t(blockIdx.x) >=
      descriptors[size_t(group) * 11 + 4] + descriptors[size_t(group) * 11 + 5]) ++group;
  if (group == groups) return;
  const uint64_t* d = descriptors + size_t(group) * 11;
  const int m = int(d[2]), splits = int(d[3]), bits = int(d[7]);
  const int warps = int(d[8]), prefetch = int(d[9]), shared = int(d[10]);
  if (m <= 0 || m % 16 || (bits != 2 && bits != 3 && bits != 4) ||
      (warps != 4 && warps != 8 && warps != 16) || warps > MaxWarps ||
      (bits != 2 && warps == 16) || (shared != 0 && shared != 1) ||
      (bits == 2 && (splits != 1 || (prefetch != 1 && prefetch != 2 && prefetch != 4))) ||
      (bits != 2 && (splits < 1 || splits > 32 || (splits & (splits - 1)) ||
                    (prefetch != 2 && prefetch != 4 && prefetch != 8)))) return;
  const uint64_t row_groups = bits == 2 ? uint64_t(m) / 2 : uint64_t(m) / 16;
  const uint64_t base_ctas = (row_groups + warps - 1) / warps;
  if (d[5] != base_ctas * splits || uint64_t(blockIdx.x) < d[4]) return;
  const uint64_t local = uint64_t(blockIdx.x) - d[4];
  const int row_block = int(local % base_ctas), split = int(local / base_ctas);
  const auto* words = reinterpret_cast<const uint32_t*>(uintptr_t(d[0]));
  const auto* lut = reinterpret_cast<const uint32_t*>(uintptr_t(d[1]));
  const float* x = grouped_x + (size_t(group) * batch + blockIdx.z) * n;
  partial += size_t(d[6]);
  __shared__ uint32_t table[512];
  if (shared) {
    for (int i = threadIdx.x; i < 512; i += MaxWarps * 32) table[i] = lut[i];
  }
  // All CTA threads participate before unused warps exit. Every active warp
  // then executes exactly its original bit-specific instruction ordering.
  __syncthreads();
  if (threadIdx.x >= warps * 32) return;
  if (bits == 2) {
    float* y = partial + size_t(blockIdx.z) * m;
#define K2(W,P) if (warps == W && prefetch == P) { \
  if (shared) original_k2_body<W,P,true>(x,words,lut,y,table,m,n,row_block); \
  else original_k2_body<W,P,false>(x,words,lut,y,table,m,n,row_block); return; }
    K2(4,1) K2(4,2) K2(4,4) K2(8,1) K2(8,2) K2(8,4)
    K2(16,1) K2(16,2) K2(16,4)
#undef K2
  } else {
    const int tiles_per_split = int((uint64_t(n) / 16 + splits - 1) / splits);
#define K34(B,W,P) if (bits == B && warps == W && prefetch == P) { \
  if (shared) original_k##B##_body<W,P,true>(x,words,lut,partial,table,m,n,row_block,split,splits,tiles_per_split); \
  else original_k##B##_body<W,P,false>(x,words,lut,partial,table,m,n,row_block,split,splits,tiles_per_split); return; }
    K34(3,4,2) K34(3,4,4) K34(3,4,8) K34(3,8,2) K34(3,8,4) K34(3,8,8)
    K34(4,4,2) K34(4,4,4) K34(4,4,8) K34(4,8,2) K34(4,8,4) K34(4,8,8)
#undef K34
  }
}

extern "C" int yaqa_k234_grouped_rows(
    const float* x, const uint64_t* descriptors, float* partial,
    int groups, int n, int batch, int blocks, int max_warps, uintptr_t raw_stream) {
  if (!x || !descriptors || !partial || uintptr_t(x) % 8 ||
      uintptr_t(descriptors) % 8 || uintptr_t(partial) % 4 ||
      groups <= 0 || groups > 65535 || n <= 0 || n % 16 ||
      batch <= 0 || batch > 65535 || blocks <= 0 ||
      (max_warps != 4 && max_warps != 8 && max_warps != 16))
    return int(cudaErrorInvalidValue);
  const auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
  const dim3 grid(blocks, 1, batch);
#define MAX_W(W) if (max_warps == W) { \
  grouped_k234_original<W><<<grid,W*32,0,stream>>>(x,descriptors,partial,groups,n,batch); \
  return int(cudaGetLastError()); }
  MAX_W(4) MAX_W(8) MAX_W(16)
#undef MAX_W
  return int(cudaErrorInvalidValue);
}
