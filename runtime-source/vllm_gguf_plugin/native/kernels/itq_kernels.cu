// SPDX-License-Identifier: Apache-2.0
// ITQ (integer trellis) codec kernels for kaiLLM.
//
// Weight format (yaqa.manifest codec "itq", version 1):
//   16x16 tiles, row-block-major; one tile = 64 steps of BPS bits (2*BPS uint32 words, BPS 4..12), LSB-first;
//   step t = row t/4, columns 4(t%4)..4(t%4)+3; state s_t = 16-bit stream window ending at
//   bit (t+1)*BPS (circular inside the tile); weights = table[(s * 0x9E3779B1) >> (32 - TB)],
//   one int32 table entry = four int8 weights.
//   W = diag(pos) diag(sv) R_out (wscale * table[.]) R_in diag(su) diag(pis), R = block Hadamard.
// Kernels:
//   itq_prep_*  : x -> (x * pis * su) R_in^T  then int8 per 32 columns (decode) or BF16 (prefill)
//   itq_gemv    : y_rot = wscale * sum_g xs_g * sum_j T[s] . xq_j           (1..16 rows, dp4a)
//   itq_post    : out = bf16((R_out y_rot) * sv * pos + bias)
//   itq_decode  : W_rot = bf16(wscale * T[s]) dense, for prefill GEMMs
// Programmatic dependent launch (sm_90+, default on; ITQ_PDL=0 disables): prep / GEMV / post may start
// before the previous kernel finishes; each waits (griddepcontrol.wait) before reading its producer's
// output, and the GEMV stages its table in shared memory before waiting.  Without the launch attribute
// both instructions are no-ops, so one build serves both modes.  ITQ_FAST_FWHT=0 selects the original
// shared-memory Hadamard prep/post instead of the register/shuffle ones.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>

namespace {

constexpr uint32_t HASH_MUL = 0x9E3779B1u;

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

bool use_pdl() {   // default on for sm_90+; ITQ_PDL=0 disables
  static int cached = -1;
  if (cached < 0) {
    const char* env = std::getenv("ITQ_PDL");
    int on = !(env != nullptr && env[0] == '0');
    if (on) {
      int device = 0, major = 0;
      cudaGetDevice(&device);
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
      on = major >= 9;
    }
    cached = on;
  }
  return cached == 1;
}

template <typename... KArgs, typename... Args>
int launch_k(void (*kern)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t stream, Args... args) {
  if (!use_pdl()) {
    kern<<<grid, block, smem, stream>>>(args...);
    return int(cudaGetLastError());
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  return int(cudaLaunchKernelEx(&cfg, kern, args...));
}

template <int TB>
__device__ __forceinline__ int lut(const int* __restrict__ table, uint32_t s) {
  return table[(s * HASH_MUL) >> (32 - TB)];
}

// Decode activations (int8 x4 words and per-32 scales) are pair-major: the 32 columns of tile pair p for padded row b
// are words [(p * nb + b) * 8, + 8) and its scale is p * nb + b, so the 8-64 rows of a pair are contiguous for the
// tensor-core fragment loads (a couple of L1 wavefronts instead of one per row).  nb = pad_rows(rows) = the GEMVs' NB
// (48: the 48-row MTP-2 verification of 16 requests, 25 % less tensor-core work than 64).
// 3-4 rows pad to 8 when the tensor-core GEMV runs (ITQ_SMALL_MMA, default on): the 8-row mma kernel (one decode per
// weight tile for all rows, activations hoisted; bit-identical to dp4a) beat the 4-row dp4a kernel on every Qwen3.8
// shape and rate (RTX PRO 6000, cold L2: gate+up 2.5 bits 61.7 -> 53.5 us); 2 rows stay on dp4a (2-6 % faster there).
bool use_mma();
bool use_small_mma() {
  static int cached = -1;
  if (cached < 0) {
    const char* env = std::getenv("ITQ_SMALL_MMA");
    cached = use_mma() && !(env != nullptr && env[0] == '0');
  }
  return cached == 1;
}
inline int pad_rows(int rows) {
  if (rows <= 2) return rows <= 1 ? 1 : 2;
  if (rows <= 4) return use_small_mma() ? 8 : 4;
  return rows <= 8 ? 8 : rows <= 16 ? 16 : rows <= 32 ? 32 : rows <= 48 ? 48 : 64;
}
__device__ __forceinline__ size_t pm_word(int b, int c, int nb) { return (size_t(c >> 5) * nb + b) * 8 + ((c >> 2) & 7); }
__device__ __forceinline__ size_t pm_scale(int b, int c, int nb) { return size_t(c >> 5) * nb + b; }

// ---- trellis formats: the BPS template parameter of the decode kernels ---------------------------------------------
// BPS 4..12: "V4", four weights per step (32-bit entries of a 12-bit table), BPS bits per step, 64 steps per tile.
// BPS 32 + b, b = 3..8: "V2" (action 10 of the 2026-11-14 plan; b = 7, 8 = 3.5 / 4 bits per weight since 2026-11-17), two
// weights per step (an int8 pair in the low 16 bits of
// each entry of a 10-bit table), b bits per step, 128 steps per tile: the tile of V4 at BPS 2 b (4 b words, rate b / 2).
// Row r of a tile is steps 8 r .. 8 r + 7 (V4: 4 r .. 4 r + 3); its dp4a word k = the pairs of steps 8 r + 2 k and
// 8 r + 2 k + 1 (bytes 0-1 and 2-3).  Same 16-bit circular windows, same multiplicative hash.
__host__ __device__ constexpr bool fmt_v2(int F) { return F >= 32; }
__host__ __device__ constexpr int fmt_step_bits(int F) { return F >= 32 ? F - 32 : F; }
__host__ __device__ constexpr int fmt_steps_per_row(int F) { return F >= 32 ? 8 : 4; }
__host__ __device__ constexpr int fmt_words(int F) { return F >= 32 ? 4 * (F - 32) : 2 * F; }   // 32-bit words per tile
__host__ __device__ constexpr int fmt_table_bits(int F) { return F >= 32 ? 10 : 12; }
__host__ __device__ constexpr int fmt_bps4(int F) { return F >= 32 ? 2 * (F - 32) : F; }       // V4 BPS of the same rate

template <int BPS>
__device__ __forceinline__ void row_states(uint32_t myword, int r, int lane_base, uint32_t st[4]) {
  constexpr int TW = fmt_words(BPS);
  if constexpr (BPS == 8) {
    const uint32_t prev = __shfl_sync(0xffffffffu, myword, lane_base | ((r + 15) & 15));
    st[0] = __funnelshift_r(prev, myword, 24) & 0xffffu;
    st[1] = myword & 0xffffu;
    st[2] = (myword >> 8) & 0xffffu;
    st[3] = (myword >> 16) & 0xffffu;
  } else if constexpr (BPS == 4) {
    const uint32_t w = __shfl_sync(0xffffffffu, myword, lane_base | (r >> 1));
    const uint32_t mine = (r & 1) ? (w >> 16) : (w & 0xffffu);
    const uint32_t prev = __shfl_sync(0xffffffffu, mine, lane_base | ((r + 15) & 15));
    const uint32_t win = (mine << 16) | prev;
    st[0] = (win >> 4) & 0xffffu;
    st[1] = (win >> 8) & 0xffffu;
    st[2] = (win >> 12) & 0xffffu;
    st[3] = win >> 16;
  } else {
    constexpr int TBITS = 64 * BPS;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      int start = (4 * r + k + 1) * BPS - 16;
      start = (start + TBITS) % TBITS;
      const int wi = start >> 5, sh = start & 31;
      const uint32_t lo = __shfl_sync(0xffffffffu, myword, lane_base | wi);
      const uint32_t hi = __shfl_sync(0xffffffffu, myword, lane_base | ((wi + 1) % TW));
      st[k] = __funnelshift_r(lo, hi, sh) & 0xffffu;
    }
  }
}

// Stream words of one tile for lane r (row r) of its 16-lane group.  The four states of row r are the 16-bit
// windows ending at bits (4r+k+1)*BPS (circular), all inside bits [(4r+1)*BPS-16, (4r+4)*BPS): three consecutive
// words from word window_start(r) >> 5 always hold them.
//  BPS 4 / 8: lane r holds word r and gets the rest from its neighbours (one or two shuffles).
//  Other BPS, DIRECT: lane r loads its three words itself (three L1 loads, no shuffle).
//  Other BPS, shuffle: lane r holds word r (and r + 16 when the tile has more than 16 words) and fetches its three
//    words with one shuffle each (two above 16 words: the owner holds the word in w0 or w1; two words at BPS 5 / 6).
// With the old extraction (four variable shifts + word selects) DIRECT was faster at one activation row; with the
// aligned extraction the shuffles are faster at every row count (cold L2, RTX PRO 6000, Qwen3.8 27B shapes: one row
// 2.5 bits down 25.6 -> 20.3 us, gate+up 48.5 -> 46.6; 3 bits 26.7 -> 25.1, 51.9 -> 50.6), so DIRECT is off by default.
// On sm_8x the shuffles lose instead (RTX 3060 cold L2, Qwen3.8 shapes, BPS 6 / 7 / 9 / 10: one row 1.25-1.53x faster
// direct, 8-16 rows 1.00-1.22x; never slower), so the default is per architecture (__CUDA_ARCH__ is defined only in the
// device passes, and every use of direct_words is device code).
#ifndef ITQ_WORDS_MODE
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
#define ITQ_WORDS_MODE 1   // benchmarks: 0 = always shuffle, 1 = always direct, 2 = per-kernel choice
#else
#define ITQ_WORDS_MODE 2
#endif
#endif
#ifndef ITQ_DIRECT8_MAX_NB  // BPS 8 with direct loads (word r and its predecessor) instead of one shuffle, up to NB rows
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
#define ITQ_DIRECT8_MAX_NB 2   // RTX 3060 cold L2, one row: down 171 -> 109 us, gate+up / qkv+z 1.03x; 8-16 rows 0.98-1.00x
#else
#define ITQ_DIRECT8_MAX_NB 0
#endif
#endif
#ifndef ITQ_DIRECT_MAX_NB
#define ITQ_DIRECT_MAX_NB 0   // mode 2: DIRECT for BPS > 8 up to this many activation rows
#endif
#ifndef ITQ_ALIGN_EXTRACT
#define ITQ_ALIGN_EXTRACT 1   // generic BPS: align once to the lane's first window, then constant-shift extractions
#endif
#ifndef ITQ_GEMV_WARPS
#define ITQ_GEMV_WARPS 4   // warps per block of the dp4a decode GEMVs (1-4 rows); the 16 KB table is per block
#endif
#ifndef ITQ_GROUP_WARPS
#define ITQ_GROUP_WARPS 8  // grouped dp4a GEMV at 1-2 rows and >= 2 bits per weight: one 16 KB table for 8 warps
#endif                     // (RTX PRO 6000 cold L2, gate+up / qkv+z at 2-3 bits: x1.19 at one row, x1.32 at two)
__host__ __device__ constexpr int group_warps(int bps, int nb) { return (nb <= 2 && fmt_bps4(bps) >= 8) ? ITQ_GROUP_WARPS : ITQ_GEMV_WARPS; }
#ifndef ITQ_MMA_WARPS
#define ITQ_MMA_WARPS 4    // warps per block of the tensor-core decode GEMVs (8-64 rows)
#endif
#ifndef ITQ_MMA_HOIST
#define ITQ_MMA_HOIST 1   // tensor-core GEMV: activations of a round loaded before its decode (NB <= 32)
#endif
#ifndef ITQ_MMA_DEPTH
#define ITQ_MMA_DEPTH 2   // tensor-core GEMV: rounds of tile words in flight per warp (1 = load, then decode)
#endif
#ifndef ITQ_MMA_FAST
#define ITQ_MMA_FAST 1    // tensor-core GEMV: full rounds without per-pair clamps (depth 2; 0 = generic ITQ_MMA_DEPTH path)
#endif
#ifndef ITQ_VOCAB_DP4A_MAX_ROWS
#define ITQ_VOCAB_DP4A_MAX_ROWS 4   // vocabulary head: dp4a up to this many activation rows, tensor cores above
#endif

// The windows of every lane lie in two consecutive words (V4 BPS 4, 5, 6): the third word is never needed.
template <int BPS>
__host__ __device__ constexpr bool two_words() {
  constexpr int B = fmt_step_bits(BPS), SPR = fmt_steps_per_row(BPS), TBITS = 32 * fmt_words(BPS);
  for (int r = 0; r < 16; ++r)
    if (((((SPR * r + 1) * B - 16 + TBITS) % TBITS) & 31) + (SPR - 1) * B + 16 > 64) return false;
  return true;
}

// Every lane's eight V2 windows lie in its three words (bits [o, o + 7 b + 16) with o = the lane's offset in its first
// word): always up to b = 7; at b = 8 every window starts at bit 24 of a word, so o + 72 = 96 exactly.
template <int BPS>
__host__ __device__ constexpr bool rows_fit_three_words() {
  constexpr int B = fmt_step_bits(BPS), SPR = fmt_steps_per_row(BPS), TBITS = 32 * fmt_words(BPS);
  for (int r = 0; r < 16; ++r)
    if (((((SPR * r + 1) * B - 16 + TBITS) % TBITS) & 31) + (SPR - 1) * B + 16 > 96) return false;
  return true;
}

template <int BPS, int NB>
__host__ __device__ constexpr bool direct_words() {
  if constexpr (BPS == 8) return NB <= ITQ_DIRECT8_MAX_NB;
  else if constexpr (BPS == 4) return false;
  else if constexpr (ITQ_WORDS_MODE == 0) return false;
  else if constexpr (ITQ_WORDS_MODE == 1) return true;
  else return fmt_bps4(BPS) > 8 && NB <= ITQ_DIRECT_MAX_NB;
}

template <int BPS>
struct TileWords {
  uint32_t w0, w1, w2;
};

template <int BPS>
__device__ __forceinline__ int window_start(int r) {
  constexpr int B = fmt_step_bits(BPS), SPR = fmt_steps_per_row(BPS), TBITS = 32 * fmt_words(BPS);
  return ((SPR * r + 1) * B - 16 + TBITS) % TBITS;
}

template <int BPS, bool DIRECT>
__device__ __forceinline__ TileWords<BPS> tile_words(const uint32_t* __restrict__ tile, int r, bool valid) {
  constexpr int TW = fmt_words(BPS);
  static_assert(fmt_v2(BPS) ? rows_fit_three_words<BPS>()
                            : (fmt_steps_per_row(BPS) - 1) * fmt_step_bits(BPS) + 16 <= 64,
                "V4: the aligned 64-bit extraction holds a row's windows up to 16 bits per step; V2: the 96-bit "
                "extraction of row_words up to 8");
  TileWords<BPS> t;
  if constexpr (DIRECT && BPS == 8) {        // word r and its circular predecessor (TW = 16 = one word per lane)
    t.w0 = valid ? __ldg(tile + r) : 0u;
    t.w1 = valid ? __ldg(tile + ((r + 15) & 15)) : 0u;
    t.w2 = 0u;
  } else if constexpr (DIRECT && BPS != 4) {
    const int w = window_start<BPS>(r) >> 5;
    t.w0 = valid ? __ldg(tile + w) : 0u;
    t.w1 = valid ? __ldg(tile + (w + 1) % TW) : 0u;
    if constexpr (two_words<BPS>()) t.w2 = 0u;
    else t.w2 = valid ? __ldg(tile + (w + 2) % TW) : 0u;
  } else {
    t.w0 = (valid && r < TW) ? __ldg(tile + r) : 0u;
    t.w1 = (TW > 16 && valid && r + 16 < TW) ? __ldg(tile + r + 16) : 0u;
    t.w2 = 0u;
  }
  return t;
}

template <int BPS, bool DIRECT>
__device__ __forceinline__ void row_states(const TileWords<BPS>& t, int r, int lane_base, uint32_t st[4]) {
  if constexpr (BPS == 8 && DIRECT) {
    st[0] = __funnelshift_r(t.w1, t.w0, 24) & 0xffffu;
    st[1] = t.w0 & 0xffffu;
    st[2] = (t.w0 >> 8) & 0xffffu;
    st[3] = (t.w0 >> 16) & 0xffffu;
  } else if constexpr (BPS == 4 || BPS == 8) {
    row_states<BPS>(t.w0, r, lane_base, st);
  } else {
    constexpr int TW = fmt_words(BPS);
    const int w = window_start<BPS>(r) >> 5, o = window_start<BPS>(r) & 31;
    uint32_t v[3];
    if constexpr (DIRECT) {
      v[0] = t.w0; v[1] = t.w1; v[2] = t.w2;
    } else {
      constexpr int NW = two_words<BPS>() ? 2 : 3;
      v[2] = 0u;
#pragma unroll
      for (int j = 0; j < NW; ++j) {
        const int i = (w + j) % TW;                            // word index; its owner is lane i & 15
        const uint32_t a = __shfl_sync(0xffffffffu, t.w0, lane_base | (i & 15));
        if constexpr (TW > 16) {
          const uint32_t b = __shfl_sync(0xffffffffu, t.w1, lane_base | (i & 15));
          v[j] = i < 16 ? a : b;
        } else {
          v[j] = a;
        }
      }
    }
#if ITQ_ALIGN_EXTRACT
    // x1:x0 = bits [o, o + 64) of v2:v1:v0; window k = bits [k * BPS, k * BPS + 16) of it (3 * BPS + 16 <= 64), so
    // two shifts by the lane's offset and four compile-time extractions replace four selects + variable shifts.
    const uint32_t x0 = __funnelshift_r(v[0], v[1], o), x1 = __funnelshift_r(v[1], v[2], o);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int s = k * BPS;
      const uint32_t win = s + 16 <= 32 ? x0 >> s : s >= 32 ? x1 >> (s - 32) : __funnelshift_r(x0, x1, s);
      st[k] = win & 0xffffu;
    }
#else
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int pos = o + k * BPS, wp = pos >> 5;              // pos <= 31 + 3 * BPS < 80: wp in {0, 1, 2}
      const uint32_t lo = wp == 0 ? v[0] : wp == 1 ? v[1] : v[2];
      const uint32_t hi = wp == 0 ? v[1] : v[2];
      st[k] = __funnelshift_r(lo, hi, pos & 31) & 0xffffu;
    }
#endif
  }
}

// The four dp4a words (16 int8 weights) of row r of a tile, from its stream words (tile_words): V4 = four windows, four
// lookups; V2 = eight windows, eight lookups of int8 pairs, four byte permutes.  Every lane must call it (shuffles).
template <int BPS, int TB, bool DIRECT>
__device__ __forceinline__ void row_words(const TileWords<BPS>& t, int r, int lane_base, const int* __restrict__ table,
                                          int (&w)[4]) {
  if constexpr (!fmt_v2(BPS)) {
    uint32_t st[4];
    row_states<BPS, DIRECT>(t, r, lane_base, st);
#pragma unroll
    for (int k = 0; k < 4; ++k) w[k] = lut<TB>(table, st[k]);
  } else {
    constexpr int B = fmt_step_bits(BPS), TW = fmt_words(BPS);
    const int ws = window_start<BPS>(r);
    const int wi = ws >> 5, o = ws & 31;
    uint32_t v[3];
    if constexpr (DIRECT) {
      v[0] = t.w0; v[1] = t.w1; v[2] = t.w2;
    } else {
      constexpr int NW = two_words<BPS>() ? 2 : 3;
      v[2] = 0u;
#pragma unroll
      for (int j = 0; j < NW; ++j) {
        const int i = (wi + j) % TW;                           // word index; its owner is lane i & 15
        const uint32_t a = __shfl_sync(0xffffffffu, t.w0, lane_base | (i & 15));
        if constexpr (TW > 16) {
          const uint32_t b = __shfl_sync(0xffffffffu, t.w1, lane_base | (i & 15));
          v[j] = i < 16 ? a : b;
        } else {
          v[j] = a;
        }
      }
    }
    // x2:x1:x0 = bits [o, o + 96) of the row's words (the third word's tail reads as zeros); window k = bits
    // [k B, k B + 16) of it: 7 B + 16 <= 64 up to B = 6, 65 / 72 at B = 7 / 8 (rows_fit_three_words: o + 7 B + 16 <= 96)
    const uint32_t x0 = __funnelshift_r(v[0], v[1], o), x1 = __funnelshift_r(v[1], v[2], o);
    const uint32_t x2 = (7 * B + 16 > 64) ? __funnelshift_r(v[2], 0u, o) : 0u;
    int e[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int s = k * B;
      const uint32_t win = s + 16 <= 32 ? x0 >> s
                         : s + 16 <= 64 ? (s >= 32 ? x1 >> (s - 32) : __funnelshift_r(x0, x1, s))
                         : s >= 64 ? x2 >> (s - 64) : __funnelshift_r(x1, x2, s - 32);
      e[k] = lut<TB>(table, win & 0xffffu);
    }
#pragma unroll
    for (int k = 0; k < 4; ++k) w[k] = int(__byte_perm(uint32_t(e[2 * k]), uint32_t(e[2 * k + 1]), 0x5410));
  }
}

template <int BPS, int TB, int NB, int WARPS, int PF>
__global__ void __launch_bounds__(WARPS * 32)
itq_gemv_kernel(const uint32_t* __restrict__ words, const int* __restrict__ table_g,
                const int* __restrict__ xq, const float* __restrict__ xs, float* __restrict__ y,
                int m, int n, int ksplit, float out_scale) {
  extern __shared__ int table[];
  for (int i = threadIdx.x; i < (1 << TB); i += WARPS * 32) table[i] = table_g[i];   // model constant
  __syncthreads();
  pdl_trigger();
  pdl_wait();
  constexpr int TW = fmt_words(BPS);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int r = lane & 15, half = lane >> 4, lane_base = lane & 16;
  const int rt = m >> 4, ct = n >> 4, pairs = ct >> 1;
  const int per = (pairs + ksplit - 1) / ksplit;
  const int items = rt * ksplit;
  const int xq_row = n >> 2, xs_row = n >> 5;
  for (int item = blockIdx.x * WARPS + warp; item < items; item += gridDim.x * WARPS) {
    const int rb = item / ksplit, ks = item - rb * ksplit;
    const int p0 = ks * per, p1 = min(pairs, p0 + per);
    float acc[NB];
#pragma unroll
    for (int b = 0; b < NB; ++b) acc[b] = 0.f;
    const uint32_t* rowtiles = words + size_t(rb) * ct * TW;
    for (int q = p0; q < p1; q += 2 * PF) {
      TileWords<BPS> wbuf[PF][2];
      int pidx[PF];
      bool valid[PF];
#pragma unroll
      for (int i = 0; i < PF; ++i) {
        const int p = q + half + 2 * i;
        valid[i] = p < p1;
        pidx[i] = valid[i] ? p : p0;
#pragma unroll
        for (int t = 0; t < 2; ++t)
          wbuf[i][t] = tile_words<BPS, direct_words<BPS, NB>()>(rowtiles + size_t(2 * pidx[i] + t) * TW, r, true);
      }
#pragma unroll
      for (int i = 0; i < PF; ++i) {
        int part[NB];
#pragma unroll
        for (int b = 0; b < NB; ++b) part[b] = 0;
#pragma unroll
        for (int t = 0; t < 2; ++t) {
          const int c = 2 * pidx[i] + t;
          int w[4];
          row_words<BPS, TB, direct_words<BPS, NB>()>(wbuf[i][t], r, lane_base, table, w);
#pragma unroll
          for (int b = 0; b < NB; ++b) {
            const int4 xv = __ldg(reinterpret_cast<const int4*>(xq) + (size_t(pidx[i]) * NB + b) * 2 + t);
            part[b] = __dp4a(w[0], xv.x, part[b]);
            part[b] = __dp4a(w[1], xv.y, part[b]);
            part[b] = __dp4a(w[2], xv.z, part[b]);
            part[b] = __dp4a(w[3], xv.w, part[b]);
          }
        }
        if (valid[i]) {
#pragma unroll
          for (int b = 0; b < NB; ++b) acc[b] += float(part[b]) * __ldg(xs + size_t(pidx[i]) * NB + b);
        }
      }
    }
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      const float total = acc[b] + __shfl_xor_sync(0xffffffffu, acc[b], 16);
      if (half == 0) {
        float* dst = y + size_t(b) * m + rb * 16 + r;
        if (ksplit > 1) atomicAdd(dst, total * out_scale);
        else *dst = total * out_scale;
      }
    }
  }
}

// In-place orthonormal Walsh-Hadamard transform of RB values in shared memory.
template <int RB>
__device__ __forceinline__ void block_fwht(float* v) {
  for (int h = 1; h < RB; h <<= 1) {
    for (int i = threadIdx.x; i < RB / 2; i += blockDim.x) {
      const int j = (i / h) * 2 * h + (i % h);
      const float a = v[j], c = v[j + h];
      v[j] = a + c;
      v[j + h] = a - c;
    }
    __syncthreads();
  }
}

// grid (n / RB, rows); input element type T (bf16 or f32); outputs int8x4 + scales or bf16.
template <int RB, typename T, bool INT8>
__global__ void itq_prep_kernel(const T* __restrict__ x, const float* __restrict__ pis,
                                const int8_t* __restrict__ su, int* __restrict__ xq,
                                float* __restrict__ xs, __nv_bfloat16* __restrict__ xr, int n,
                                int x_stride, float* __restrict__ zero, int zero_count, int nb) {
  __shared__ float v[RB];
  const int b = blockIdx.y, base = blockIdx.x * RB;
  pdl_wait();
  pdl_trigger();
  if (zero) {   // split-K GEMV output that follows this launch (saves a memset launch)
    const int nblk = gridDim.x * gridDim.y, id = blockIdx.y * gridDim.x + blockIdx.x;
    for (int i = id * blockDim.x + threadIdx.x; i < zero_count; i += nblk * blockDim.x) zero[i] = 0.f;
  }
  for (int i = threadIdx.x; i < RB; i += blockDim.x) {
    float value = float(x[size_t(b) * x_stride + base + i]);
    if (pis) value *= pis[base + i];
    v[i] = value * float(su[base + i]);
  }
  __syncthreads();
  block_fwht<RB>(v);
  const float norm = rsqrtf(float(RB));
  if constexpr (!INT8) {
    for (int i = threadIdx.x; i < RB; i += blockDim.x)
      xr[size_t(b) * n + base + i] = __float2bfloat16(v[i] * norm);
  } else {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for (int g = warp; g < RB / 32; g += warps) {
      const float val = v[g * 32 + lane] * norm;
      float amax = fabsf(val);
#pragma unroll
      for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
      const float scale = amax > 0.f ? amax / 127.f : 1.f;
      const int qv = __float2int_rn(val / scale);
      const int b0 = __shfl_sync(0xffffffffu, qv, (lane & ~3) + 0) & 0xff;
      const int b1 = __shfl_sync(0xffffffffu, qv, (lane & ~3) + 1) & 0xff;
      const int b2 = __shfl_sync(0xffffffffu, qv, (lane & ~3) + 2) & 0xff;
      const int b3 = __shfl_sync(0xffffffffu, qv, (lane & ~3) + 3) & 0xff;
      if ((lane & 3) == 0)
        xq[pm_word(b, base + g * 32 + lane, nb)] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
      if (lane == 0) xs[pm_scale(b, base + g * 32, nb)] = scale;
    }
  }
}

// grid (m / RB, rows): out[b, i] = bf16((R_out y)[i] * sv[i] * pos[i] + bias[i])
template <int RB>
__global__ void itq_post_kernel(const float* __restrict__ y, const int8_t* __restrict__ sv,
                                const float* __restrict__ pos, const float* __restrict__ bias,
                                __nv_bfloat16* __restrict__ out, int m, int out_stride, int y_stride,
                                const int* __restrict__ yi = nullptr, const float* __restrict__ row_scale = nullptr) {
  __shared__ float v[RB];
  const int b = blockIdx.y, base = blockIdx.x * RB;
  pdl_wait();
  if (yi) {
    const float rs = row_scale[b];
    for (int i = threadIdx.x; i < RB; i += blockDim.x) v[i] = __fmul_rn(float(yi[size_t(b) * y_stride + base + i]), rs);
  } else {
    for (int i = threadIdx.x; i < RB; i += blockDim.x) v[i] = y[size_t(b) * y_stride + base + i];
  }
  __syncthreads();
  block_fwht<RB>(v);
  const float norm = rsqrtf(float(RB));
  for (int i = threadIdx.x; i < RB; i += blockDim.x) {
    float value = v[i] * norm * float(sv[base + i]);
    if (pos) value *= pos[base + i];
    if (bias) value += bias[base + i];
    out[size_t(b) * out_stride + base + i] = __float2bfloat16(value);
  }
}

// ---- Register/shuffle block Hadamard for RB = 256 W (W = 1, 2, 4, 8): T = 32 W threads, 8 values each ----
// Input layout A: thread t = 32 w + l holds element e = 256 w + 8 l + r in v[r].  Bits 0-2 are
// transformed in registers, bits 3-7 by warp shuffles, and (W > 1) the warp bits after one shared-memory
// exchange into layout B: thread t then holds e = 256 k + T j + t in v[j W + k] (j < 8 / W, k < W).
// Unnormalized.  Inactive threads (whole warps) only join the barrier.
template <int W>
__device__ __forceinline__ void fwht_regs(float (&v)[8], float* sm, bool active) {
#pragma unroll
  for (int h = 1; h < 8; h <<= 1)
#pragma unroll
    for (int r = 0; r < 8; ++r)
      if ((r & h) == 0) {
        const float a = v[r], c = v[r + h];
        v[r] = a + c;
        v[r + h] = a - c;
      }
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int mk = 1; mk < 32; mk <<= 1) {
    const bool upper = (lane & mk) != 0;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      const float o = __shfl_xor_sync(0xffffffffu, v[r], mk);
      v[r] = upper ? o - v[r] : v[r] + o;
    }
  }
  if constexpr (W > 1) {
    constexpr int T = 32 * W, J = 8 / W;
    const int t = threadIdx.x;
    if (active) {
      float4* dst = reinterpret_cast<float4*>(sm + (t >> 5) * 256 + (t & 31) * 8);
      dst[0] = make_float4(v[0], v[1], v[2], v[3]);
      dst[1] = make_float4(v[4], v[5], v[6], v[7]);
    }
    __syncthreads();
    if (active) {
#pragma unroll
      for (int j = 0; j < J; ++j)
#pragma unroll
        for (int k = 0; k < W; ++k) v[j * W + k] = sm[k * 256 + j * T + t];
#pragma unroll
      for (int j = 0; j < J; ++j)
#pragma unroll
        for (int h = 1; h < W; h <<= 1)
#pragma unroll
          for (int k = 0; k < W; ++k)
            if ((k & h) == 0) {
              const float a = v[j * W + k], c = v[j * W + k + h];
              v[j * W + k] = a + c;
              v[j * W + k + h] = a - c;
            }
    }
  }
}

// element index held in v[i] after fwht_regs<W> (layout A for W == 1, layout B otherwise)
template <int W>
__device__ __forceinline__ int fwht_index(int i) {
  if constexpr (W == 1) {
    return 8 * int(threadIdx.x) + i;
  } else {
    return (i % W) * 256 + (i / W) * (32 * W) + int(threadIdx.x);
  }
}

__device__ __forceinline__ void load8(const __nv_bfloat16* p, float (&v)[8]) {
  const uint4 raw = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __bfloat1622float2(h[i]);
    v[2 * i] = f.x;
    v[2 * i + 1] = f.y;
  }
}

__device__ __forceinline__ void load8(const float* p, float (&v)[8]) {
  const float4 a = reinterpret_cast<const float4*>(p)[0], c = reinterpret_cast<const float4*>(p)[1];
  v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
  v[4] = c.x; v[5] = c.y; v[6] = c.z; v[7] = c.w;
}

// itq_prep_fast_kernel after the load: input scale, signs, block Hadamard, int8 per 32 (pair-major).
template <int RB>
__device__ __forceinline__ void prep_fast_tail(float (&v)[8], const float (&pisf)[8], bool has_pis, uint2 sg,
                                               int* __restrict__ xq, float* __restrict__ xs, int b, int base, int e0,
                                               int nb, float* sm) {
  constexpr int W = RB / 256;
  const int lane = threadIdx.x & 31;
  if (has_pis) {
#pragma unroll
    for (int r = 0; r < 8; ++r) v[r] *= pisf[r];
  }
#pragma unroll
  for (int r = 0; r < 8; ++r) v[r] *= float(int8_t(((r < 4 ? sg.x : sg.y) >> (8 * (r & 3))) & 0xffu));
  fwht_regs<W>(v, sm, true);
  const float norm = rsqrtf(float(RB));
  if constexpr (W == 1) {
    float amax = 0.f;
#pragma unroll
    for (int r = 0; r < 8; ++r) amax = fmaxf(amax, fabsf(v[r]));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    const float scale = amax > 0.f ? amax / 127.f : 1.f;
    unsigned w0 = 0, w1 = 0;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      w0 |= (unsigned(__float2int_rn(v[r] / scale)) & 0xffu) << (8 * r);
      w1 |= (unsigned(__float2int_rn(v[r + 4] / scale)) & 0xffu) << (8 * r);
    }
    reinterpret_cast<uint2*>(xq)[pm_word(b, base + e0, nb) >> 1] = make_uint2(w0, w1);
    if ((lane & 3) == 0) xs[pm_scale(b, base + e0, nb)] = scale * norm;
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) {   // 32 consecutive elements per warp for each i
      const int e = fwht_index<W>(i);
      float amax = fabsf(v[i]);
#pragma unroll
      for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
      const float scale = amax > 0.f ? amax / 127.f : 1.f;
      unsigned q = unsigned(__float2int_rn(v[i] / scale)) & 0xffu;
      q |= __shfl_down_sync(0xffffffffu, q, 1) << 8;
      q |= __shfl_down_sync(0xffffffffu, q, 2) << 16;
      if ((lane & 3) == 0) reinterpret_cast<unsigned*>(xq)[pm_word(b, base + e, nb)] = q;
      if (lane == 0) xs[pm_scale(b, base + e, nb)] = scale * norm;
    }
  }
}

// Decode prep for RB in [256, 2048]: grid (n / RB, rows), RB / 8 threads, 16-byte aligned rows.
// The int8 scale absorbs the 1/sqrt(RB) normalization (quantized values are scale invariant).
template <int RB, typename T>
__global__ void __launch_bounds__(RB / 8)
itq_prep_fast_kernel(const T* __restrict__ x, const float* __restrict__ pis, const int8_t* __restrict__ su,
                     int* __restrict__ xq, float* __restrict__ xs, int n, int x_stride,
                     float* __restrict__ zero, int zero_count, int nb) {
  constexpr int W = RB / 256;
  __shared__ __align__(16) float sm[W > 1 ? RB : 4];
  const int b = blockIdx.y, base = blockIdx.x * RB, t = threadIdx.x, lane = t & 31;
  const int e0 = (t >> 5) * 256 + lane * 8;
  float pisf[8];                       // model constants before the dependency wait
  if (pis) load8(pis + base + e0, pisf);
  const uint2 sg = *reinterpret_cast<const uint2*>(su + base + e0);
  pdl_wait();
  pdl_trigger();
  if (zero) {   // split-K GEMV output that follows this launch (saves a memset launch)
    const int nblk = gridDim.x * gridDim.y, id = blockIdx.y * gridDim.x + blockIdx.x;
    for (int i = id * blockDim.x + t; i < zero_count; i += nblk * blockDim.x) zero[i] = 0.f;
  }
  float v[8];
  load8(x + size_t(b) * x_stride + base + e0, v);
  prep_fast_tail<RB>(v, pisf, pis != nullptr, sg, xq, xs, b, base, e0, nb, sm);
}

// Decode: vLLM's gated RMSNorm (RMSNormGated: norm_before_gate, silu gate, 128-wide heads, F32 weight; output rounded
// to BF16 like the reference's BF16 tensor) fused into the consumer's prep (itq_prep_fast_kernel's layout: thread t
// holds elements 8 t .. 8 t + 7 of its rotation block, so a head is one half-warp).  Not bit-identical to the Triton
// norm (sum order, sigmoid): BF16 ties of y may round the other way.
template <int RB>
__global__ void __launch_bounds__(RB / 8)
itq_gated_norm_prep_kernel(const __nv_bfloat16* __restrict__ x, int x_stride, const __nv_bfloat16* __restrict__ z,
                           int z_stride, const float* __restrict__ w, float eps, const float* __restrict__ pis,
                           const int8_t* __restrict__ su, int* __restrict__ xq, float* __restrict__ xs,
                           float* __restrict__ zero, int zero_count, int nb) {
  constexpr int W = RB / 256;
  __shared__ __align__(16) float sm[W > 1 ? RB : 4];
  const int b = blockIdx.y, base = blockIdx.x * RB, t = threadIdx.x, lane = t & 31;
  const int e0 = (t >> 5) * 256 + lane * 8;
  float v[8], g[8], wv[8], pisf[8];    // model constants before the dependency wait
  load8(w + ((base + e0) & 127), wv);
  if (pis) load8(pis + base + e0, pisf);
  const uint2 sg = *reinterpret_cast<const uint2*>(su + base + e0);
  pdl_wait();
  pdl_trigger();
  if (zero) {   // split-K GEMV output that follows this launch
    const int nblk = gridDim.x * gridDim.y, id = blockIdx.y * gridDim.x + blockIdx.x;
    for (int i = id * blockDim.x + t; i < zero_count; i += nblk * blockDim.x) zero[i] = 0.f;
  }
  load8(x + size_t(b) * x_stride + base + e0, v);
  load8(z + size_t(b) * z_stride + base + e0, g);
  float ss = 0.f;
#pragma unroll
  for (int r = 0; r < 8; ++r) ss += v[r] * v[r];
#pragma unroll
  for (int o = 8; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);   // the head's 16 threads
  const float rstd = 1.f / sqrtf(ss / 128.f + eps);
#pragma unroll
  for (int r = 0; r < 8; ++r) {
    const float y = v[r] * rstd * wv[r];
    v[r] = __bfloat162float(__float2bfloat16_rn(y * (g[r] / (1.f + __expf(-g[r])))));
  }
  prep_fast_tail<RB>(v, pisf, pis != nullptr, sg, xq, xs, b, base, e0, nb, sm);
}


// Post for one rotation block of RB = 256 W outputs; pointers already offset to the block.
template <int W>
__device__ __forceinline__ void post_fast(const float* __restrict__ y, const int8_t* __restrict__ sv,
                                          const float* __restrict__ pos, const float* __restrict__ bias,
                                          __nv_bfloat16* __restrict__ out, float* sm, bool active) {
  float v[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  const int t = threadIdx.x;
  if (active) load8(y + (t >> 5) * 256 + (t & 31) * 8, v);
  fwht_regs<W>(v, sm, active);
  if (!active) return;
  const float norm = rsqrtf(float(256 * W));
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int e = fwht_index<W>(i);
    float value = v[i] * norm * float(sv[e]);
    if (pos) value *= pos[e];
    if (bias) value += bias[e];
    out[e] = __float2bfloat16(value);
  }
}

// post_fast with the dependency wait inside: the model constants (signs, output scales, bias) are loaded first, so
// their latency overlaps the producer GEMV's tail; then pdl_wait() and the GEMV output.  Same arithmetic.
template <int W>
__device__ __forceinline__ void post_fast_pdl(const float* __restrict__ y, const int8_t* __restrict__ sv,
                                              const float* __restrict__ pos, const float* __restrict__ bias,
                                              __nv_bfloat16* __restrict__ out, float* sm, bool active,
                                              const int* __restrict__ yi = nullptr, float rs = 1.f) {
  float sf[8], pf[8], bf[8];
  if (active) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int e = fwht_index<W>(i);
      sf[i] = float(sv[e]);
      pf[i] = pos ? pos[e] : 1.f;
      bf[i] = bias ? bias[e] : 0.f;
    }
  }
  pdl_wait();
  float v[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  const int t = threadIdx.x;
  if (active) {
    if (yi) {        // int32 GEMM output: float(y) * row scale, as torch's y.float().mul_(scale * wscale)
      const int4* q = reinterpret_cast<const int4*>(yi + (t >> 5) * 256 + (t & 31) * 8);
      const int4 a = q[0], c = q[1];
      const int iv[8] = {a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w};
#pragma unroll
      for (int i = 0; i < 8; ++i) v[i] = __fmul_rn(float(iv[i]), rs);
    } else {
      load8(y + (t >> 5) * 256 + (t & 31) * 8, v);
    }
  }
  fwht_regs<W>(v, sm, active);
  if (!active) return;
  const float norm = rsqrtf(float(256 * W));
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int e = fwht_index<W>(i);
    float value = v[i] * norm * sf[i];
    if (pos) value *= pf[i];
    if (bias) value += bf[i];
    out[e] = __float2bfloat16(value);
  }
}

// grid (m / RB, rows), RB / 8 threads, RB in [256, 2048], 16-byte aligned y rows
template <int RB>
__global__ void __launch_bounds__(RB / 8)
itq_post_fast_kernel(const float* __restrict__ y, const int8_t* __restrict__ sv, const float* __restrict__ pos,
                     const float* __restrict__ bias, __nv_bfloat16* __restrict__ out, int out_stride, int y_stride,
                     const int* __restrict__ yi = nullptr, const float* __restrict__ row_scale = nullptr) {
  constexpr int W = RB / 256;
  __shared__ __align__(16) float sm[W > 1 ? RB : 4];
  const int b = blockIdx.y, base = blockIdx.x * RB;
  post_fast_pdl<W>(yi ? nullptr : y + size_t(b) * y_stride + base, sv + base, pos ? pos + base : nullptr,
                   bias ? bias + base : nullptr, out + size_t(b) * out_stride + base, sm, true,
                   yi ? yi + size_t(b) * y_stride + base : nullptr, yi ? row_scale[b] : 1.f);
}

// vLLM's SiLU-and-mul on BF16 operands: silu computed in F32 and rounded to BF16, then a BF16 product (one
// rounding of the exact product), as act_and_mul_kernel<BFloat16, __nv_bfloat162, silu, packed_silu>.
#ifndef ITQ_SILU_FAST_MATH
#define ITQ_SILU_FAST_MATH 0   // 1 if the deployed vLLM build uses --use_fast_math (__expf, __fdividef)
#endif
__device__ __forceinline__ float bf16_round(float x) { return __bfloat162float(__float2bfloat16(x)); }

__device__ __forceinline__ float vllm_silu_mul(float g, float u) {
#if ITQ_SILU_FAST_MATH
  const float s = bf16_round(__fdividef(g, 1.0f + __expf(-g)));
#else
  const float s = bf16_round(g / (1.0f + expf(-g)));
#endif
  return bf16_round(s * u);
}

// Decode MLP middle in one launch (replaces post_group(gate, up) + act_and_mul + prep(down)): per RB-wide chunk and
// row, the inverse rotation of the gate and up GEMV outputs (post_fast arithmetic, BF16 outputs), SiLU-and-mul, then
// down's input preparation (input scale, signs, block Hadamard, per-32 int8) exactly as itq_prep_fast_kernel.
// Requires gate / up output blocks == down input block == RB.  grid (m / RB, rows), RB / 8 threads.
template <int RB>
__global__ void __launch_bounds__(RB / 8)
itq_swiglu_prep_kernel(const float* __restrict__ yg, const float* __restrict__ yu, int y_stride,
                       const int8_t* __restrict__ svg, const int8_t* __restrict__ svu, const float* __restrict__ posg,
                       const float* __restrict__ posu, const float* __restrict__ pis, const int8_t* __restrict__ su,
                       int* __restrict__ xq, float* __restrict__ xs, int m, float* __restrict__ zero, int zero_count,
                       int nb) {
  constexpr int W = RB / 256;
  __shared__ __align__(16) float sm[RB];
  const int b = blockIdx.y, base = blockIdx.x * RB, t = threadIdx.x, lane = t & 31;
  const int e0 = (t >> 5) * 256 + lane * 8;
  // model constants before the dependency wait (never written; their latency overlaps the gate+up GEMV's tail)
  float sgf[8], suf[8], pgf[8], puf[8], pisf[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int e = fwht_index<W>(i);
    sgf[i] = float(svg[base + e]);
    suf[i] = float(svu[base + e]);
    pgf[i] = posg ? posg[base + e] : 1.f;
    puf[i] = posu ? posu[base + e] : 1.f;
  }
  if (pis) load8(pis + base + e0, pisf);
  const uint2 sg = *reinterpret_cast<const uint2*>(su + base + e0);
  pdl_wait();
  pdl_trigger();
  if (zero) {   // split-K output of the down GEMV that follows
    const int nblk = gridDim.x * gridDim.y, id = blockIdx.y * gridDim.x + blockIdx.x;
    for (int i = id * blockDim.x + t; i < zero_count; i += nblk * blockDim.x) zero[i] = 0.f;
  }
  const float norm = rsqrtf(float(RB));
  float g[8], u[8];
  load8(yg + size_t(b) * y_stride + base + e0, g);   // both GEMV outputs in one round trip
  load8(yu + size_t(b) * y_stride + base + e0, u);
  fwht_regs<W>(g, sm, true);
  __syncthreads();
  fwht_regs<W>(u, sm, true);
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float gv = g[i] * norm * sgf[i];
    if (posg) gv *= pgf[i];
    float uv = u[i] * norm * suf[i];
    if (posu) uv *= puf[i];
    const int e = fwht_index<W>(i);
    sm[e] = vllm_silu_mul(bf16_round(gv), bf16_round(uv));
  }
  __syncthreads();
  float v[8];
#pragma unroll
  for (int r = 0; r < 8; ++r) v[r] = sm[e0 + r];
  __syncthreads();
  if (pis) {
#pragma unroll
    for (int r = 0; r < 8; ++r) v[r] *= pisf[r];
  }
#pragma unroll
  for (int r = 0; r < 8; ++r) v[r] *= float(int8_t(((r < 4 ? sg.x : sg.y) >> (8 * (r & 3))) & 0xffu));
  fwht_regs<W>(v, sm, true);
  if constexpr (W == 1) {
    float amax = 0.f;
#pragma unroll
    for (int r = 0; r < 8; ++r) amax = fmaxf(amax, fabsf(v[r]));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    const float scale = amax > 0.f ? amax / 127.f : 1.f;
    unsigned w0 = 0, w1 = 0;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      w0 |= (unsigned(__float2int_rn(v[r] / scale)) & 0xffu) << (8 * r);
      w1 |= (unsigned(__float2int_rn(v[r + 4] / scale)) & 0xffu) << (8 * r);
    }
    reinterpret_cast<uint2*>(xq)[pm_word(b, base + e0, nb) >> 1] = make_uint2(w0, w1);
    if ((lane & 3) == 0) xs[pm_scale(b, base + e0, nb)] = scale * norm;
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int e = fwht_index<W>(i);
      float amax = fabsf(v[i]);
#pragma unroll
      for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
      const float scale = amax > 0.f ? amax / 127.f : 1.f;
      unsigned q = unsigned(__float2int_rn(v[i] / scale)) & 0xffu;
      q |= __shfl_down_sync(0xffffffffu, q, 1) << 8;
      q |= __shfl_down_sync(0xffffffffu, q, 2) << 16;
      if ((lane & 3) == 0) reinterpret_cast<unsigned*>(xq)[pm_word(b, base + e, nb)] = q;
      if (lane == 0) xs[pm_scale(b, base + e, nb)] = scale * norm;
    }
  }
}

// fwht_regs for one RB-wide chunk handled by threads [c * RB / 8, (c + 1) * RB / 8) of a larger block: tl is the
// thread index inside the chunk, sm the chunk's own scratch (RB floats).  Same arithmetic and layouts as fwht_regs.
template <int W>
__device__ __forceinline__ void fwht_regs_chunk(float (&v)[8], float* sm, int tl) {
#pragma unroll
  for (int h = 1; h < 8; h <<= 1)
#pragma unroll
    for (int r = 0; r < 8; ++r)
      if ((r & h) == 0) {
        const float a = v[r], c = v[r + h];
        v[r] = a + c;
        v[r + h] = a - c;
      }
  const int lane = tl & 31;
#pragma unroll
  for (int mk = 1; mk < 32; mk <<= 1) {
    const bool upper = (lane & mk) != 0;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
      const float o = __shfl_xor_sync(0xffffffffu, v[r], mk);
      v[r] = upper ? o - v[r] : v[r] + o;
    }
  }
  if constexpr (W > 1) {
    constexpr int T = 32 * W, J = 8 / W;
    float4* dst = reinterpret_cast<float4*>(sm + (tl >> 5) * 256 + (tl & 31) * 8);
    dst[0] = make_float4(v[0], v[1], v[2], v[3]);
    dst[1] = make_float4(v[4], v[5], v[6], v[7]);
    __syncthreads();
#pragma unroll
    for (int j = 0; j < J; ++j)
#pragma unroll
      for (int k = 0; k < W; ++k) v[j * W + k] = sm[k * 256 + j * T + tl];
#pragma unroll
    for (int j = 0; j < J; ++j)
#pragma unroll
      for (int h = 1; h < W; h <<= 1)
#pragma unroll
        for (int k = 0; k < W; ++k)
          if ((k & h) == 0) {
            const float a = v[j * W + k], c = v[j * W + k + h];
            v[j * W + k] = a + c;
            v[j * W + k + h] = a - c;
          }
  } else {
    __syncthreads();   // keep the barrier count equal across W
  }
}

template <int W>
__device__ __forceinline__ int fwht_index_chunk(int i, int tl) {
  if constexpr (W == 1) return 8 * tl + i;
  else return (i % W) * 256 + (i / W) * (32 * W) + tl;
}

__device__ __forceinline__ float ggml_warp_sum(float x) {   // ggml warp_reduce_sum<32>
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) x += __shfl_xor_sync(0xffffffffu, x, offset, 32);
  return x;
}

// kai_dual::norm_prepare is built with ggml's --use_fast_math: its mean (ss / k) is not an IEEE division.
#ifndef ITQ_NORM_DIV
#define ITQ_NORM_DIV 2   // 0 IEEE '/', 1 __fdividef (div.approx), 2 div.full.f32 (--prec-div=false)
#endif
__device__ __forceinline__ float norm_mean_div(float a, float b) {
#if ITQ_NORM_DIV == 1
  return __fdividef(a, b);
#elif ITQ_NORM_DIV == 2
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
#else
  return a / b;
#endif
}

constexpr int ADD_NORM_MAX_H = 5120;   // two F32 rows of static shared memory (+ warp sums) under 48 KB

// Residual stream step in one launch (decode, one row per block, H / 8 threads), replacing the producer's post, the
// fused add + Gemma RMSNorm (kai_dual::norm_prepare) and the consumer's itq_prep:
//   x = BF16(post(y_raw))                      producer (down / o_proj / out_proj) output rotation, post_fast arithmetic
//   s = x + residual   -> r_out (F32)          then y = BF16(s * rsqrt(mean(s^2) + eps) * w32), norm_prepare's order
//   consumer prep of y (input scale, signs, block Hadamard, per-32 int8), itq_prep_fast_kernel arithmetic.
// Producer output blocks and consumer input blocks are RB; H % RB == 0, H <= ADD_NORM_MAX_H.
template <int RB, typename R>
__global__ void __launch_bounds__(ADD_NORM_MAX_H / 8)
itq_add_norm_prep_kernel(const float* __restrict__ y_raw, int y_stride, const int8_t* __restrict__ sv,
                         const float* __restrict__ pos, const R* __restrict__ res, int r_stride,
                         float* __restrict__ r_out, const float* __restrict__ w32, float eps,
                         __nv_bfloat16* __restrict__ y, int H, const float* __restrict__ pis,
                         const int8_t* __restrict__ su, int* __restrict__ xq, float* __restrict__ xs,
                         float* __restrict__ zero, int zero_count, int nb) {
  constexpr int W = RB / 256, TPC = RB / 8;
  __shared__ __align__(16) float srow[ADD_NORM_MAX_H];
  __shared__ __align__(16) float scratch[ADD_NORM_MAX_H];
  __shared__ float warp_ss[32];
  const int b = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const int c = t / TPC, tl = t - c * TPC, base = c * RB;
  const int e0 = (tl >> 5) * 256 + (tl & 31) * 8;
  float* sc = scratch + base;
  pdl_wait();
  pdl_trigger();
  if (zero) {
    for (int i = b * blockDim.x + t; i < zero_count; i += gridDim.x * blockDim.x) zero[i] = 0.f;
  }
  const float norm = rsqrtf(float(RB));
  float v[8];
  load8(y_raw + size_t(b) * y_stride + base + e0, v);
  fwht_regs_chunk<W>(v, sc, tl);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int e = base + fwht_index_chunk<W>(i, tl);
    float x = v[i] * norm * float(sv[e]);
    if (pos) x *= pos[e];
    const float s = bf16_round(x) + float(res[size_t(b) * r_stride + e]);
    r_out[size_t(b) * H + e] = s;
    srow[e] = s;
  }
  __syncthreads();
  const int nblocks = H / 32, nred = (nblocks + 31) / 32;
  if (warp < nred) {
    float ss = 0.f;
    if (t < nblocks) {
#pragma unroll
      for (int j = 0; j < 32; ++j) ss += srow[32 * t + j] * srow[32 * t + j];
    }
    ss = ggml_warp_sum(ss);
    if (lane == 0) warp_ss[warp] = ss;
  }
  __syncthreads();
  if (warp == 0) {
    float ss = lane < nred ? warp_ss[lane] : 0.f;
    ss = ggml_warp_sum(ss);
    if (lane == 0) warp_ss[0] = ss;
  }
  __syncthreads();
  const float rstd = rsqrtf(norm_mean_div(warp_ss[0], float(H)) + eps);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int e = base + fwht_index_chunk<W>(i, tl);
    const __nv_bfloat16 h = __float2bfloat16(srow[e] * rstd * w32[e]);
    y[size_t(b) * H + e] = h;
    scratch[e] = __bfloat162float(h);
  }
  if (xq == nullptr) return;
  __syncthreads();
#pragma unroll
  for (int r = 0; r < 8; ++r) v[r] = scratch[base + e0 + r];
  __syncthreads();
  if (pis) {
    float p[8];
    load8(pis + base + e0, p);
#pragma unroll
    for (int r = 0; r < 8; ++r) v[r] *= p[r];
  }
  const uint2 sg = *reinterpret_cast<const uint2*>(su + base + e0);
#pragma unroll
  for (int r = 0; r < 8; ++r) v[r] *= float(int8_t(((r < 4 ? sg.x : sg.y) >> (8 * (r & 3))) & 0xffu));
  fwht_regs_chunk<W>(v, sc, tl);
  if constexpr (W == 1) {
    float amax = 0.f;
#pragma unroll
    for (int r = 0; r < 8; ++r) amax = fmaxf(amax, fabsf(v[r]));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    const float scale = amax > 0.f ? amax / 127.f : 1.f;
    unsigned w0 = 0, w1 = 0;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      w0 |= (unsigned(__float2int_rn(v[r] / scale)) & 0xffu) << (8 * r);
      w1 |= (unsigned(__float2int_rn(v[r + 4] / scale)) & 0xffu) << (8 * r);
    }
    reinterpret_cast<uint2*>(xq)[pm_word(b, base + e0, nb) >> 1] = make_uint2(w0, w1);
    if ((lane & 3) == 0) xs[pm_scale(b, base + e0, nb)] = scale * norm;
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int e = fwht_index_chunk<W>(i, tl);
      float amax = fabsf(v[i]);
#pragma unroll
      for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
      const float scale = amax > 0.f ? amax / 127.f : 1.f;
      unsigned q = unsigned(__float2int_rn(v[i] / scale)) & 0xffu;
      q |= __shfl_down_sync(0xffffffffu, q, 1) << 8;
      q |= __shfl_down_sync(0xffffffffu, q, 2) << 16;
      if ((lane & 3) == 0) reinterpret_cast<unsigned*>(xq)[pm_word(b, base + e, nb)] = q;
      if (lane == 0) xs[pm_scale(b, base + e, nb)] = scale * norm;
    }
  }
}

// Cluster version of itq_add_norm_prep_kernel for 1024-wide rotation blocks (sm_90+): a cluster of H / 1024 blocks per
// row, block c handling chunk c (128 threads).  Chunk c is exactly warp c of kai_dual::norm_prepare's reduction (32
// blocks of 32 elements), so each block computes that warp's butterfly sum, and every block repeats warp 0's cross-warp
// butterfly over the partials read through distributed shared memory: same arithmetic, bit-identical, but the two
// Hadamard transforms and the norm run on H / 1024 SMs instead of one.
template <typename R>
__global__ void __launch_bounds__(128)
itq_add_norm_prep_cluster_kernel(const float* __restrict__ y_raw, int y_stride, const int8_t* __restrict__ sv,
                                 const float* __restrict__ pos, const R* __restrict__ res, int r_stride,
                                 float* __restrict__ r_out, const float* __restrict__ w32, float eps,
                                 __nv_bfloat16* __restrict__ y, int H, const float* __restrict__ pis,
                                 const int8_t* __restrict__ su, int* __restrict__ xq, float* __restrict__ xs,
                                 float* __restrict__ zero, int zero_count, int nb) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  namespace cg = cooperative_groups;
  constexpr int RB = 1024, W = 4;
  __shared__ __align__(16) float srow[RB];
  __shared__ __align__(16) float sm[RB];
  __shared__ float part, rstd_sh;
  cg::cluster_group cluster = cg::this_cluster();
  const int c = int(cluster.block_rank()), b = blockIdx.y, t = threadIdx.x, lane = t & 31, base = c * RB;
  const int e0 = (t >> 5) * 256 + lane * 8;
  // Model constants (signs, scales, norm weight) are never written: load them before the dependency wait so their
  // latency overlaps the producer GEMV's tail (the residual and the GEMV output still wait).
  float svf[8], posf[8], wf[8], pisf[8];
  uint2 sg = make_uint2(0u, 0u);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int e = fwht_index<W>(i);
    svf[i] = float(sv[base + e]);
    posf[i] = pos ? pos[base + e] : 1.f;
    wf[i] = w32[base + e];
  }
  if (xq != nullptr) {
    if (pis) load8(pis + base + e0, pisf);
    sg = *reinterpret_cast<const uint2*>(su + base + e0);
  }
  pdl_wait();
  pdl_trigger();
  if (zero) {
    const int nblk = gridDim.x * gridDim.y, id = blockIdx.y * gridDim.x + blockIdx.x;
    for (int i = id * blockDim.x + t; i < zero_count; i += nblk * blockDim.x) zero[i] = 0.f;
  }
  const float norm = rsqrtf(float(RB));
  float v[8];
  load8(y_raw + size_t(b) * y_stride + base + e0, v);
  fwht_regs<W>(v, sm, true);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int e = fwht_index<W>(i);
    float x = v[i] * norm * svf[i];
    if (pos) x *= posf[i];
    const float s = bf16_round(x) + float(res[size_t(b) * r_stride + base + e]);
    r_out[size_t(b) * H + base + e] = s;
    srow[e] = s;
  }
  __syncthreads();
  if (t < 32) {
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 32; ++j) ss += srow[32 * t + j] * srow[32 * t + j];
    ss = ggml_warp_sum(ss);
    if (t == 0) part = ss;
  }
  cluster.sync();
  if (t < 32) {
    float ss = t < int(cluster.num_blocks()) ? *cluster.map_shared_rank(&part, t) : 0.f;
    ss = ggml_warp_sum(ss);
    if (t == 0) rstd_sh = rsqrtf(norm_mean_div(ss, float(H)) + eps);
  }
  cluster.sync();      // every block has read the partials (DSMEM) and its rstd before any block moves on
  const float rstd = rstd_sh;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int e = fwht_index<W>(i);
    const __nv_bfloat16 h = __float2bfloat16(srow[e] * rstd * wf[i]);
    y[size_t(b) * H + base + e] = h;
    srow[e] = __bfloat162float(h);
  }
  if (xq == nullptr) return;
  __syncthreads();
#pragma unroll
  for (int r = 0; r < 8; ++r) v[r] = srow[e0 + r];
  __syncthreads();
  if (pis) {
#pragma unroll
    for (int r = 0; r < 8; ++r) v[r] *= pisf[r];
  }
#pragma unroll
  for (int r = 0; r < 8; ++r) v[r] *= float(int8_t(((r < 4 ? sg.x : sg.y) >> (8 * (r & 3))) & 0xffu));
  fwht_regs<W>(v, sm, true);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int e = fwht_index<W>(i);
    float amax = fabsf(v[i]);
#pragma unroll
    for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float scale = amax > 0.f ? amax / 127.f : 1.f;
    unsigned q = unsigned(__float2int_rn(v[i] / scale)) & 0xffu;
    q |= __shfl_down_sync(0xffffffffu, q, 1) << 8;
    q |= __shfl_down_sync(0xffffffffu, q, 2) << 16;
    if ((lane & 3) == 0) reinterpret_cast<unsigned*>(xq)[pm_word(b, base + e, nb)] = q;
    if (lane == 0) xs[pm_scale(b, base + e, nb)] = scale * norm;
  }
#endif
}

bool use_cluster_norm() {   // default on for sm_90+; ITQ_CLUSTER=0 keeps the one-block-per-row kernel
  static int cached = -1;
  if (cached < 0) {
    const char* env = std::getenv("ITQ_CLUSTER");
    int device = 0, major = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cached = major >= 9 && !(env != nullptr && env[0] == '0');
  }
  return cached == 1;
}

bool use_fast_fwht() {
  static int cached = -1;
  if (cached < 0) {
    const char* env = std::getenv("ITQ_FAST_FWHT");
    cached = !(env != nullptr && env[0] == '0');
  }
  return cached == 1;
}

inline bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

// one warp per (row block, tile): dense rotated weight in BF16, row-major [m, n]
template <int BPS, int TB>
__global__ void itq_decode_kernel(const uint32_t* __restrict__ words, const int* __restrict__ table_g,
                                  __nv_bfloat16* __restrict__ w, int m, int n, float wscale) {
  extern __shared__ int table[];
  for (int i = threadIdx.x; i < (1 << TB); i += blockDim.x) table[i] = table_g[i];
  __syncthreads();
  constexpr int TW = fmt_words(BPS);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
  const int r = lane & 15, half = lane >> 4, lane_base = lane & 16;
  const int ct = n >> 4, tiles = (m >> 4) * ct;
  for (int pair = blockIdx.x * warps + warp; pair * 2 < tiles; pair += gridDim.x * warps) {
    const int tile = pair * 2 + half;
    const bool valid = tile < tiles;
    const TileWords<BPS> myword = tile_words<BPS, direct_words<BPS, 1>()>(words + size_t(valid ? tile : 0) * TW, r, valid);
    int wd[4];
    row_words<BPS, TB, direct_words<BPS, 1>()>(myword, r, lane_base, table, wd);
    if (!valid) continue;
    const int rb = tile / ct, cb = tile - rb * ct;
    // lane r owns row r of the tile: 16 BF16 = two 16-byte stores
    __nv_bfloat162 v[8];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int packed = wd[k];
      const float f0 = float(int8_t(packed & 0xff)) * wscale, f1 = float(int8_t((packed >> 8) & 0xff)) * wscale;
      const float f2 = float(int8_t((packed >> 16) & 0xff)) * wscale, f3 = float(int8_t((packed >> 24) & 0xff)) * wscale;
      v[2 * k] = __floats2bfloat162_rn(f0, f1);
      v[2 * k + 1] = __floats2bfloat162_rn(f2, f3);
    }
    uint4* dst = reinterpret_cast<uint4*>(w + size_t(rb * 16 + r) * n + cb * 16);
    dst[0] = *reinterpret_cast<const uint4*>(&v[0]);
    dst[1] = *reinterpret_cast<const uint4*>(&v[4]);
  }
}

// Same tiles, raw int8 table values (no wscale): row-major int8 [m, n] for the int8 tensor-core prefill.
template <int BPS, int TB>
__global__ void itq_decode_i8_kernel(const uint32_t* __restrict__ words, const int* __restrict__ table_g,
                                     int8_t* __restrict__ w, int m, int n) {
  extern __shared__ int table[];
  for (int i = threadIdx.x; i < (1 << TB); i += blockDim.x) table[i] = table_g[i];
  __syncthreads();
  constexpr int TW = fmt_words(BPS);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
  const int r = lane & 15, half = lane >> 4, lane_base = lane & 16;
  const int ct = n >> 4, tiles = (m >> 4) * ct;
  for (int pair = blockIdx.x * warps + warp; pair * 2 < tiles; pair += gridDim.x * warps) {
    const int tile = pair * 2 + half;
    const bool valid = tile < tiles;
    const TileWords<BPS> myword = tile_words<BPS, direct_words<BPS, 1>()>(words + size_t(valid ? tile : 0) * TW, r, valid);
    int wd[4];
    row_words<BPS, TB, direct_words<BPS, 1>()>(myword, r, lane_base, table, wd);
    if (!valid) continue;
    const int rb = tile / ct, cb = tile - rb * ct;
    uint4 v;
    v.x = uint32_t(wd[0]);
    v.y = uint32_t(wd[1]);
    v.z = uint32_t(wd[2]);
    v.w = uint32_t(wd[3]);
    *reinterpret_cast<uint4*>(w + size_t(rb * 16 + r) * n + cb * 16) = v;
  }
}

int sm_count() {
  static int cached = 0;
  if (!cached) {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&cached, cudaDevAttrMultiProcessorCount, device);
    if (cached <= 0) cached = 1;
  }
  return cached;
}

// ---- grouped launches for merged projections that share one prepared input ----------------
constexpr int MAX_PARTS = 8;
struct GemvGroup {
  const uint32_t* words[MAX_PARTS];
  float* y[MAX_PARTS];
  int m[MAX_PARTS];
  int ksplit[MAX_PARTS];
  float scale[MAX_PARTS];
  int item_end[MAX_PARTS];   // cumulative work items
  int parts;
};

template <int BPS, int TB, int NB, int WARPS, int PF>
__global__ void __launch_bounds__(WARPS * 32)
itq_gemv_group_kernel(GemvGroup g, const int* __restrict__ table_g, const int* __restrict__ xq,
                      const float* __restrict__ xs, int n) {
  extern __shared__ int table[];
  for (int i = threadIdx.x; i < (1 << TB); i += WARPS * 32) table[i] = table_g[i];   // model constant
  __syncthreads();
  pdl_trigger();
  pdl_wait();
  constexpr int TW = fmt_words(BPS);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int r = lane & 15, half = lane >> 4, lane_base = lane & 16;
  const int ct = n >> 4, pairs = ct >> 1;
  const int xq_row = n >> 2, xs_row = n >> 5;
  const int items = g.item_end[g.parts - 1];
  for (int item = blockIdx.x * WARPS + warp; item < items; item += gridDim.x * WARPS) {
    int part = 0;
    while (item >= g.item_end[part]) ++part;
    const int local = item - (part ? g.item_end[part - 1] : 0);
    const int ksplit = g.ksplit[part], m = g.m[part];
    const int rb = local / ksplit, ks = local - rb * ksplit;
    const int per = (pairs + ksplit - 1) / ksplit;
    const int p0 = ks * per, p1 = min(pairs, p0 + per);
    float acc[NB];
#pragma unroll
    for (int b = 0; b < NB; ++b) acc[b] = 0.f;
    const uint32_t* rowtiles = g.words[part] + size_t(rb) * ct * TW;
    for (int q = p0; q < p1; q += 2 * PF) {
      TileWords<BPS> wbuf[PF][2];
      int pidx[PF];
      bool valid[PF];
#pragma unroll
      for (int i = 0; i < PF; ++i) {
        const int p = q + half + 2 * i;
        valid[i] = p < p1;
        pidx[i] = valid[i] ? p : p0;
#pragma unroll
        for (int t = 0; t < 2; ++t)
          wbuf[i][t] = tile_words<BPS, direct_words<BPS, NB>()>(rowtiles + size_t(2 * pidx[i] + t) * TW, r, true);
      }
#pragma unroll
      for (int i = 0; i < PF; ++i) {
        int prt[NB];
#pragma unroll
        for (int b = 0; b < NB; ++b) prt[b] = 0;
#pragma unroll
        for (int t = 0; t < 2; ++t) {
          const int c = 2 * pidx[i] + t;
          int w[4];
          row_words<BPS, TB, direct_words<BPS, NB>()>(wbuf[i][t], r, lane_base, table, w);
#pragma unroll
          for (int b = 0; b < NB; ++b) {
            const int4 xv = __ldg(reinterpret_cast<const int4*>(xq) + (size_t(pidx[i]) * NB + b) * 2 + t);
            prt[b] = __dp4a(w[0], xv.x, prt[b]);
            prt[b] = __dp4a(w[1], xv.y, prt[b]);
            prt[b] = __dp4a(w[2], xv.z, prt[b]);
            prt[b] = __dp4a(w[3], xv.w, prt[b]);
          }
        }
        if (valid[i]) {
#pragma unroll
          for (int b = 0; b < NB; ++b) acc[b] += float(prt[b]) * __ldg(xs + size_t(pidx[i]) * NB + b);
        }
      }
    }
    const float scale = g.scale[part];
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      const float total = acc[b] + __shfl_xor_sync(0xffffffffu, acc[b], 16);
      if (half == 0) {
        float* dst = g.y[part] + size_t(b) * m + rb * 16 + r;
        if (ksplit > 1) atomicAdd(dst, total * scale);
        else *dst = total * scale;
      }
    }
  }
}

// ---- tensor-core decode GEMV for 8-64 activation rows -------------------------------------------------------------
// Same work split as itq_gemv_kernel: a warp owns one item (a 16-row block x a k-split range of tile pairs) and its two
// half-warps decode alternate pairs exactly as there (lane r: row r's 32 int8 weights of a pair).  The decoded pairs are
// staged in shared memory and the whole warp applies each one with mma.m16n8k32.s8 to 8-row chunks of the int8
// activations, instead of NB x 8 dp4a per lane and tile.  The int32 dot product of a pair equals the dp4a one, and it is
// scaled and summed in the dp4a kernel's order (one F32 accumulator per half-warp pair sequence, the halves added at the
// end), so the result is bit-identical to itq_gemv_kernel (16 rows or fewer).
constexpr int MMA_ROW_WORDS = 12;   // 32 int8 weights + 16 bytes of padding per staged row: conflict-free A loads
constexpr int MMA_STAGE_WORDS = 2 * 16 * MMA_ROW_WORDS;   // one pair per half-warp

#ifndef ITQ_MMA_LDMATRIX
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
#define ITQ_MMA_LDMATRIX 1   // sm_8x: one ldmatrix.x4 instead of four LDS per A fragment (not yet measured on sm_120)
#else
#define ITQ_MMA_LDMATRIX 0
#endif
#endif

// A fragment of mma.m16n8k32.s8 from a staged 16 x 32 int8 tile (rows of MMA_ROW_WORDS words): word tq of rows gq and
// gq + 8, then word 4 + tq of both.  ldmatrix.x4 returns exactly these registers (matrices 0-3 = rows 0-7 / 8-15 of
// bytes 0-15, then of bytes 16-31; thread t gets row t / 4, 32-bit column t % 4 of each); with 48-byte rows the eight
// 16-byte row reads of each matrix start on banks 0, 12, 24, 4, 16, 28, 8, 20: conflict-free.
__device__ __forceinline__ void load_a_frag(const uint32_t* s, int lane, uint32_t (&a)[4]) {
#if ITQ_MMA_LDMATRIX
  const int row = (lane & 7) + (lane & 8), woff = (lane >> 4) * 4;
  const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(s + row * MMA_ROW_WORDS + woff));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(addr) : "memory");
#else
  const int gq = lane >> 2, tq = lane & 3;
  a[0] = s[gq * MMA_ROW_WORDS + tq];
  a[1] = s[(gq + 8) * MMA_ROW_WORDS + tq];
  a[2] = s[gq * MMA_ROW_WORDS + 4 + tq];
  a[3] = s[(gq + 8) * MMA_ROW_WORDS + 4 + tq];
#endif
}

__device__ __forceinline__ void mma_s8_16832(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%10,%10,%10,%10};"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
}

// tot[c][j]: rows (lane >> 2) + 8 * (j >> 1) of the block, activation row 8 c + 2 (lane & 3) + (j & 1)
template <int BPS, int TB, int NB, int PF>
__device__ __forceinline__ void mma_item(const uint32_t* __restrict__ rowtiles, int p0, int p1,
                                         const int* __restrict__ table, const int* __restrict__ xq,
                                         const float* __restrict__ xs, int xq_row, int xs_row, uint32_t* stage,
                                         float (&tot)[NB / 8][4]) {
  constexpr int TW = fmt_words(BPS), CH = NB / 8;
  constexpr bool D = direct_words<BPS, NB>();
  const int lane = threadIdx.x & 31, r = lane & 15, half = lane >> 4, lane_base = lane & 16;
  const int gq = lane >> 2, tq = lane & 3;
  float acc[2][CH][4];
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int c = 0; c < CH; ++c)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[h][c][j] = 0.f;
  uint32_t* mine = stage + (half * 16 + r) * MMA_ROW_WORDS;
  // round qq's tile words (pairs past p1 load pair p0 again; their products are skipped below)
  auto load_round = [&](int qq, TileWords<BPS> (&w)[PF][2]) {
#pragma unroll
    for (int i = 0; i < PF; ++i) {
      const int p = qq + half + 2 * i, pi = p < p1 ? p : p0;
#pragma unroll
      for (int t = 0; t < 2; ++t) w[i][t] = tile_words<BPS, D>(rowtiles + size_t(2 * pi + t) * TW, r, true);
    }
  };
  // round q: its activations (hoisted), the decode of its tile words, the products in each half-warp's dp4a order
  auto run_round = [&](const int q, const TileWords<BPS> (&wbuf)[PF][2]) {
    // ITQ_MMA_HOIST: this round's activation words and scales are loaded before the decode (which hides their
    // latency) instead of right before each mma, when PF x CH <= 8 (registers: NB <= 32 at PF 2, NB 48 / 64 at PF 1).
    constexpr int HC = (ITQ_MMA_HOIST && PF * CH <= 8) ? CH : 0;
    uint32_t xb[PF][2][HC > 0 ? HC : 1][2];
    float2 xv[PF][2][HC > 0 ? HC : 1];
    if constexpr (HC > 0) {
#pragma unroll
      for (int i = 0; i < PF; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int p = q + h + 2 * i, pv = p < p1 ? p : p0;
#pragma unroll
          for (int c = 0; c < HC; ++c) {
            const int* xr = xq + (size_t(pv) * NB + 8 * c + gq) * 8;
            xb[i][h][c][0] = uint32_t(__ldg(xr + tq));
            xb[i][h][c][1] = uint32_t(__ldg(xr + 4 + tq));
            xv[i][h][c] = __ldg(reinterpret_cast<const float2*>(xs + size_t(pv) * NB + 8 * c + 2 * tq));
          }
        }
    }
#pragma unroll
    for (int i = 0; i < PF; ++i) {
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        int wd[4];
        row_words<BPS, TB, D>(wbuf[i][t], r, lane_base, table, wd);
        uint4 w;
        w.x = uint32_t(wd[0]);
        w.y = uint32_t(wd[1]);
        w.z = uint32_t(wd[2]);
        w.w = uint32_t(wd[3]);
        *reinterpret_cast<uint4*>(mine + 4 * t) = w;
      }
      __syncwarp();
#pragma unroll
      for (int h = 0; h < 2; ++h) {   // the two pairs of this round, each in its half-warp's dp4a order
        const int p = q + h + 2 * i;
        if (p < p1) {                  // warp-uniform
          uint32_t a[4];
          load_a_frag(stage + h * 16 * MMA_ROW_WORDS, lane, a);
#pragma unroll
          for (int c = 0; c < CH; ++c) {
            int d[4];
            float s0, s1;
            if constexpr (HC > 0) {
              mma_s8_16832(d, a, xb[i][h][c][0], xb[i][h][c][1]);
              s0 = xv[i][h][c].x;
              s1 = xv[i][h][c].y;
            } else {
              const int* xr = xq + (size_t(p) * NB + 8 * c + gq) * 8;
              mma_s8_16832(d, a, uint32_t(__ldg(xr + tq)), uint32_t(__ldg(xr + 4 + tq)));
              s0 = __ldg(xs + size_t(p) * NB + 8 * c + 2 * tq);
              s1 = __ldg(xs + size_t(p) * NB + 8 * c + 2 * tq + 1);
            }
            acc[h][c][0] += float(d[0]) * s0;
            acc[h][c][1] += float(d[1]) * s1;
            acc[h][c][2] += float(d[2]) * s0;
            acc[h][c][3] += float(d[3]) * s1;
          }
        }
      }
      __syncwarp();
    }
  };
#if ITQ_MMA_FAST
  // Full rounds (all 2 PF pairs below p1: every round but possibly the last) address the tile words and activations
  // from one base per round plus compile-time offsets, without per-pair clamps and checks; the last partial round
  // takes run_round.  Same loads, same products, same order.  The generic round compiled to ~336 instructions per
  // 2048 weights on sm_86, ~90 of them 64-bit activation address arithmetic, and the RTX 3060 runs it issue-bound.
  // The next round's tile words are requested before this round's decode (ITQ_MMA_DEPTH 2; deeper did not help).
  auto load_full = [&](int qq, TileWords<BPS> (&w)[PF][2]) {
    const uint32_t* tp = rowtiles + size_t(2 * (qq + half)) * TW;
#pragma unroll
    for (int i = 0; i < PF; ++i)
#pragma unroll
      for (int t = 0; t < 2; ++t) w[i][t] = tile_words<BPS, D>(tp + (4 * i + t) * TW, r, true);
  };
  auto run_full = [&](const int q, const TileWords<BPS> (&wbuf)[PF][2]) {
    const int* xqr = xq + (size_t(q) * NB + gq) * 8 + tq;     // pair q + j, activation chunk c: + 8 (j NB + 8 c)
    const float* xsr = xs + size_t(q) * NB + 2 * tq;           //                                   + j NB + 8 c
    constexpr int HC = (ITQ_MMA_HOIST && PF * CH <= 8) ? CH : 0;
    uint32_t xb[PF][2][HC > 0 ? HC : 1][2];
    float2 xv[PF][2][HC > 0 ? HC : 1];
    if constexpr (HC > 0) {
#pragma unroll
      for (int i = 0; i < PF; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
          for (int c = 0; c < HC; ++c) {
            const int o = (h + 2 * i) * NB + 8 * c;
            xb[i][h][c][0] = uint32_t(__ldg(xqr + 8 * o));
            xb[i][h][c][1] = uint32_t(__ldg(xqr + 8 * o + 4));
            xv[i][h][c] = __ldg(reinterpret_cast<const float2*>(xsr + o));
          }
    }
#pragma unroll
    for (int i = 0; i < PF; ++i) {
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        int wd[4];
        row_words<BPS, TB, D>(wbuf[i][t], r, lane_base, table, wd);
        uint4 w;
        w.x = uint32_t(wd[0]);
        w.y = uint32_t(wd[1]);
        w.z = uint32_t(wd[2]);
        w.w = uint32_t(wd[3]);
        *reinterpret_cast<uint4*>(mine + 4 * t) = w;
      }
      __syncwarp();
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        uint32_t a[4];
        load_a_frag(stage + h * 16 * MMA_ROW_WORDS, lane, a);
#pragma unroll
        for (int c = 0; c < CH; ++c) {
          int d[4];
          float s0, s1;
          if constexpr (HC > 0) {
            mma_s8_16832(d, a, xb[i][h][c][0], xb[i][h][c][1]);
            s0 = xv[i][h][c].x;
            s1 = xv[i][h][c].y;
          } else {
            const int o = (h + 2 * i) * NB + 8 * c;
            mma_s8_16832(d, a, uint32_t(__ldg(xqr + 8 * o)), uint32_t(__ldg(xqr + 8 * o + 4)));
            s0 = __ldg(xsr + o);
            s1 = __ldg(xsr + o + 1);
          }
          acc[h][c][0] += float(d[0]) * s0;
          acc[h][c][1] += float(d[1]) * s1;
          acc[h][c][2] += float(d[2]) * s0;
          acc[h][c][3] += float(d[3]) * s1;
        }
      }
      __syncwarp();
    }
  };
  {
    constexpr int STEP = 2 * PF;
    const int qf = p0 + (p1 - p0) / STEP * STEP;   // rounds [p0, qf) are full
    TileWords<BPS> next[PF][2];
    if (p0 < qf) load_full(p0, next);
    else if (p0 < p1) load_round(p0, next);
    int q = p0;
    for (; q < qf; q += STEP) {
      TileWords<BPS> cur[PF][2];
#pragma unroll
      for (int i = 0; i < PF; ++i)
#pragma unroll
        for (int t = 0; t < 2; ++t) cur[i][t] = next[i][t];
      if (q + STEP < qf) load_full(q + STEP, next);        // warp-uniform
      else if (q + STEP < p1) load_round(q + STEP, next);
      run_full(q, cur);
    }
    if (q < p1) run_round(q, next);                        // the partial last round
  }
#else
  // ITQ_MMA_DEPTH rounds of tile words in flight per warp: round q + DEPTH - 1 is requested before round q's decode
  // (DEPTH 1: load, then decode).  The DRAM latency (~1500 cycles on an RTX PRO 6000) was exposed once per round
  // (~600 cycles of decode and products; 7-12 warps per SM on Qwen3.8 27B shapes).  Same products, same order.
  constexpr int DEPTH = ITQ_MMA_DEPTH, STEP = 2 * PF;
  if constexpr (DEPTH <= 1) {
    for (int q = p0; q < p1; q += STEP) {
      TileWords<BPS> wbuf[PF][2];
      load_round(q, wbuf);
      run_round(q, wbuf);
    }
  } else {
    // a shift register of the next DEPTH - 1 rounds (moves, one round per iteration; an unrolled ring of DEPTH rounds
    // per iteration ran slower: cold L2, 6-12-bit shapes, depth 2 1531 vs 1453 us)
    TileWords<BPS> pipe[DEPTH - 1][PF][2];
#pragma unroll
    for (int d = 0; d < DEPTH - 1; ++d)
      if (p0 + d * STEP < p1) load_round(p0 + d * STEP, pipe[d]);
    for (int q = p0; q < p1; q += STEP) {
      TileWords<BPS> wbuf[PF][2];
#pragma unroll
      for (int i = 0; i < PF; ++i)
#pragma unroll
        for (int t = 0; t < 2; ++t) {
          wbuf[i][t] = pipe[0][i][t];
#pragma unroll
          for (int d = 0; d + 1 < DEPTH - 1; ++d) pipe[d][i][t] = pipe[d + 1][i][t];
        }
      if (q + (DEPTH - 1) * STEP < p1) load_round(q + (DEPTH - 1) * STEP, pipe[DEPTH - 2]);   // warp-uniform
      run_round(q, wbuf);
    }
  }
#endif
#pragma unroll
  for (int c = 0; c < CH; ++c)
#pragma unroll
    for (int j = 0; j < 4; ++j) tot[c][j] = acc[0][c][j] + acc[1][c][j];
}

template <int NB>
__device__ __forceinline__ void mma_store(float* __restrict__ y, int m, int rb, int ksplit, float scale,
                                          const float (&tot)[NB / 8][4]) {
  const int lane = threadIdx.x & 31, gq = lane >> 2, tq = lane & 3;
#pragma unroll
  for (int c = 0; c < NB / 8; ++c)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float* dst = y + size_t(8 * c + 2 * tq + (j & 1)) * m + rb * 16 + gq + 8 * (j >> 1);
      if (ksplit > 1) atomicAdd(dst, tot[c][j] * scale);
      else *dst = tot[c][j] * scale;
    }
}

template <int BPS, int TB, int NB, int WARPS, int PF>
__global__ void __launch_bounds__(WARPS * 32)
itq_gemv_mma_kernel(const uint32_t* __restrict__ words, const int* __restrict__ table_g,
                    const int* __restrict__ xq, const float* __restrict__ xs, float* __restrict__ y,
                    int m, int n, int ksplit, float out_scale) {
  extern __shared__ int table[];
  for (int i = threadIdx.x; i < (1 << TB); i += WARPS * 32) table[i] = table_g[i];
  __syncthreads();
  pdl_trigger();
  pdl_wait();
  constexpr int TW = fmt_words(BPS);
  const int warp = threadIdx.x >> 5;
  uint32_t* stage = reinterpret_cast<uint32_t*>(table + (1 << TB)) + warp * MMA_STAGE_WORDS;
  const int rt = m >> 4, ct = n >> 4, pairs = ct >> 1;
  const int per = (pairs + ksplit - 1) / ksplit, items = rt * ksplit;
  for (int item = blockIdx.x * WARPS + warp; item < items; item += gridDim.x * WARPS) {
    const int rb = item / ksplit, ks = item - rb * ksplit;
    const int p0 = ks * per, p1 = min(pairs, p0 + per);
    float tot[NB / 8][4];
    mma_item<BPS, TB, NB, PF>(words + size_t(rb) * ct * TW, p0, p1, table, xq, xs, n >> 2, n >> 5, stage, tot);
    mma_store<NB>(y, m, rb, ksplit, out_scale, tot);
  }
}

template <int BPS, int TB, int NB, int WARPS, int PF>
__global__ void __launch_bounds__(WARPS * 32)
itq_gemv_group_mma_kernel(GemvGroup g, const int* __restrict__ table_g, const int* __restrict__ xq,
                          const float* __restrict__ xs, int n) {
  extern __shared__ int table[];
  for (int i = threadIdx.x; i < (1 << TB); i += WARPS * 32) table[i] = table_g[i];
  __syncthreads();
  pdl_trigger();
  pdl_wait();
  constexpr int TW = fmt_words(BPS);
  const int warp = threadIdx.x >> 5;
  uint32_t* stage = reinterpret_cast<uint32_t*>(table + (1 << TB)) + warp * MMA_STAGE_WORDS;
  const int ct = n >> 4, pairs = ct >> 1;
  const int items = g.item_end[g.parts - 1];
  for (int item = blockIdx.x * WARPS + warp; item < items; item += gridDim.x * WARPS) {
    int part = 0;
    while (item >= g.item_end[part]) ++part;
    const int local = item - (part ? g.item_end[part - 1] : 0);
    const int ksplit = g.ksplit[part], m = g.m[part];
    const int rb = local / ksplit, ks = local - rb * ksplit;
    const int per = (pairs + ksplit - 1) / ksplit;
    const int p0 = ks * per, p1 = min(pairs, p0 + per);
    float tot[NB / 8][4];
    mma_item<BPS, TB, NB, PF>(g.words[part] + size_t(rb) * ct * TW, p0, p1, table, xq, xs, n >> 2, n >> 5, stage, tot);
    mma_store<NB>(g.y[part], m, rb, ksplit, g.scale[part], tot);
  }
}

// Tensor-core decode GEMVs for 8+ activation rows: default on sm_80+, ITQ_MMA=0 keeps the dp4a kernels (<= 16 rows).
bool use_mma() {
  static int cached = -1;
  if (cached < 0) {
    const char* env = std::getenv("ITQ_MMA");
    int device = 0, major = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cached = major >= 8 && !(env != nullptr && env[0] == '0');
  }
  return cached == 1;
}

struct PostGroup {
  const float* y[MAX_PARTS];
  const int8_t* sv[MAX_PARTS];
  const float* pos[MAX_PARTS];
  const float* bias[MAX_PARTS];
  int m[MAX_PARTS];
  int rb[MAX_PARTS];          // rotation block size per part
  int col[MAX_PARTS];         // output column offset per part
  int block_end[MAX_PARTS];   // cumulative rotation blocks
  int parts;
  int fast;                   // register/shuffle Hadamard for 256 <= rb <= 2048 (16-byte aligned y)
};

// grid (sum_p m_p / rb_p, rows), 256 threads; runtime rotation width <= 4096
__global__ void itq_post_group_kernel(PostGroup g, __nv_bfloat16* __restrict__ out, int out_stride) {
  __shared__ __align__(16) float v[4096];
  const int b = blockIdx.y;
  int part = 0;
  while (int(blockIdx.x) >= g.block_end[part]) ++part;
  const int local = blockIdx.x - (part ? g.block_end[part - 1] : 0);
  const int RB = g.rb[part], m = g.m[part], base = local * RB;
  const float* y = g.y[part] + size_t(b) * m;
  if (g.fast && RB >= 256 && RB <= 2048) {
    const bool active = int(threadIdx.x) < RB / 8;
    const float* pos = g.pos[part] ? g.pos[part] + base : nullptr;
    const float* bias = g.bias[part] ? g.bias[part] + base : nullptr;
    __nv_bfloat16* o = out + size_t(b) * out_stride + g.col[part] + base;
    const int8_t* sv = g.sv[part] + base;
    switch (RB) {   // post_fast_pdl: constants, then pdl_wait(), then the GEMV output
      case 256: post_fast_pdl<1>(y + base, sv, pos, bias, o, v, active); return;
      case 512: post_fast_pdl<2>(y + base, sv, pos, bias, o, v, active); return;
      case 1024: post_fast_pdl<4>(y + base, sv, pos, bias, o, v, active); return;
      default: post_fast_pdl<8>(y + base, sv, pos, bias, o, v, active); return;
    }
  }
  pdl_wait();
  for (int i = threadIdx.x; i < RB; i += blockDim.x) v[i] = y[base + i];
  __syncthreads();
  for (int h = 1; h < RB; h <<= 1) {
    for (int i = threadIdx.x; i < RB / 2; i += blockDim.x) {
      const int j = (i / h) * 2 * h + (i % h);
      const float a = v[j], c = v[j + h];
      v[j] = a + c;
      v[j + h] = a - c;
    }
    __syncthreads();
  }
  const float norm = rsqrtf(float(RB));
  const int8_t* sv = g.sv[part];
  const float* pos = g.pos[part];
  const float* bias = g.bias[part];
  for (int i = threadIdx.x; i < RB; i += blockDim.x) {
    float value = v[i] * norm * float(sv[base + i]);
    if (pos) value *= pos[base + i];
    if (bias) value += bias[base + i];
    out[size_t(b) * out_stride + g.col[part] + base + i] = __float2bfloat16(value);
  }
}

template <int BPS, int TB, int NB, int PF>
int launch_gemv(const uint32_t* words, const int* table, const int* xq, const float* xs, float* y,
                int m, int n, int ksplit, float scale, int blocks, bool zeroed, cudaStream_t stream) {
  constexpr int WARPS = ITQ_GEMV_WARPS;
  const size_t smem = sizeof(int) << TB;
  auto kern = itq_gemv_kernel<BPS, TB, NB, WARPS, PF>;
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
    configured = true;
  }
  if (ksplit > 1 && !zeroed) cudaMemsetAsync(y, 0, sizeof(float) * size_t(NB) * m, stream);
  return launch_k(kern, dim3(blocks), dim3(WARPS * 32), smem, stream, words, table, xq, xs, y, m, n, ksplit, scale);
}

template <int BPS, int TB, int NB, int PF>
int launch_gemv_mma(const uint32_t* words, const int* table, const int* xq, const float* xs, float* y,
                    int m, int n, int ksplit, float scale, int blocks, bool zeroed, cudaStream_t stream) {
  constexpr int WARPS = ITQ_MMA_WARPS;
  const size_t smem = (sizeof(int) << TB) + sizeof(uint32_t) * WARPS * MMA_STAGE_WORDS;
  auto kern = itq_gemv_mma_kernel<BPS, TB, NB, WARPS, PF>;
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
    configured = true;
  }
  if (ksplit > 1 && !zeroed) cudaMemsetAsync(y, 0, sizeof(float) * size_t(NB) * m, stream);
  return launch_k(kern, dim3(blocks), dim3(WARPS * 32), smem, stream, words, table, xq, xs, y, m, n, ksplit, scale);
}

template <int RB, typename T>
int launch_prep(const T* x, const float* pis, const int8_t* su, int* xq, float* xs, __nv_bfloat16* xr,
                int rows, int n, int x_stride, bool to_int8, float* zero, int zero_count, cudaStream_t stream) {
  const dim3 grid(n / RB, rows);
  if constexpr (RB >= 256 && RB <= 2048) {
    if (to_int8 && use_fast_fwht() && aligned16(x) && x_stride % int(16 / sizeof(T)) == 0 &&
        (pis == nullptr || aligned16(pis)) && (reinterpret_cast<uintptr_t>(su) & 7u) == 0 &&
        (reinterpret_cast<uintptr_t>(xq) & 7u) == 0)
      return launch_k(itq_prep_fast_kernel<RB, T>, grid, dim3(RB / 8), 0, stream, x, pis, su, xq, xs, n, x_stride,
                      zero, zero_count, pad_rows(rows));
  }
  const int threads = RB >= 256 ? 256 : (RB >= 32 ? RB : 32);
  if (to_int8)
    return launch_k(itq_prep_kernel<RB, T, true>, grid, dim3(threads), 0, stream, x, pis, su, xq, xs, xr, n, x_stride,
                    zero, zero_count, pad_rows(rows));
  return launch_k(itq_prep_kernel<RB, T, false>, grid, dim3(threads), 0, stream, x, pis, su, xq, xs, xr, n, x_stride,
                  zero, zero_count, 0);
}

template <int RB>
int launch_post(const float* y, const int8_t* sv, const float* pos, const float* bias, __nv_bfloat16* out,
                int rows, int m, int out_stride, int y_stride, cudaStream_t stream,
                const int* yi = nullptr, const float* row_scale = nullptr) {
  const dim3 grid(m / RB, rows);
  const void* src = yi ? static_cast<const void*>(yi) : static_cast<const void*>(y);
  if constexpr (RB >= 256 && RB <= 2048) {
    if (use_fast_fwht() && aligned16(src) && y_stride % 4 == 0)
      return launch_k(itq_post_fast_kernel<RB>, grid, dim3(RB / 8), 0, stream, y, sv, pos, bias, out, out_stride,
                      y_stride, yi, row_scale);
  }
  return launch_k(itq_post_kernel<RB>, grid, dim3(RB >= 256 ? 256 : (RB >= 32 ? RB : 32)), 0, stream, y, sv, pos, bias,
                  out, m, out_stride, y_stride, yi, row_scale);
}

// Prefill int8 activations, one block per row: scale = max(|x|, 1e-30) * (1 / 127), q = clamp(rint(x / scale), +-127),
// IEEE operations and half-even rounding = torch's xr.abs().amax(1).float().clamp_min_(1e-30).div_(127) and
// round(xr.float().div_(scale)).clamp_(-127, 127).to(int8) bit for bit (the max is exact in any order).
__global__ void __launch_bounds__(1024)
itq_quant_rows_kernel(const __nv_bfloat16* __restrict__ x, int8_t* __restrict__ q, float* __restrict__ scale, int n) {
  __shared__ float red[32];
  const int b = blockIdx.x;
  const __nv_bfloat16* xr = x + size_t(b) * n;
  float m = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, fabsf(__bfloat162float(xr[i])));
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
  __syncthreads();
  if (threadIdx.x < 32) {
    m = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    // torch divides a tensor by a Python scalar as a product with the scalar's reciprocal (div_true_kernel_cuda)
    if (threadIdx.x == 0) red[0] = __fmul_rn(fmaxf(m, 1e-30f), __fdiv_rn(1.0f, 127.0f));
  }
  __syncthreads();
  const float s = red[0];
  if (threadIdx.x == 0) scale[b] = s;
  int8_t* qr = q + size_t(b) * n;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float v = rintf(__fdiv_rn(__bfloat162float(xr[i]), s));
    qr[i] = int8_t(fminf(fmaxf(v, -127.f), 127.f));
  }
}

// ---- vocabulary codec (yaqa.manifest codec "itqvocab", version 1) ----------------------------------
// Tied embedding / LM head rows stored in the rotated basis r = R (e * su) (R = orthonormal block Hadamard
// of RB columns) and split into rate classes.  kind 0: int8 rows [n, d]; kind 1: NF4 nibbles [n, d/2]
// (low nibble = even column) with 16 float levels; kind 2: ITQ trellis rows in the tile16 layout above
// (rows padded to 16).  Row j of class c decodes to its codes times scale[j] (kind 2: times wscale too);
// map[token] = (c << 24) | j.  Embedding: e = (R^T r) * su.  Logits: scale_j <codes_j, R (su * h)>, with
// R (su * h) prepared once by itq_prep (int8 per 32 columns).
constexpr int VOCAB_MAX_CLASSES = 8;
struct VocabClasses {
  const void* data[VOCAB_MAX_CLASSES];
  const float* scale[VOCAB_MAX_CLASSES];
  const float* levels[VOCAB_MAX_CLASSES];
  float wscale[VOCAB_MAX_CLASSES];
  int kind[VOCAB_MAX_CLASSES];
  int bps[VOCAB_MAX_CLASSES];
};

// the 8 consecutive columns e0..e0+7 (e0 % 8 == 0) of row j of class c, scaled
template <int TB>
__device__ __forceinline__ void vocab_decode8(const VocabClasses& vc, const int* __restrict__ table, int c, int j,
                                              int d, int e0, float (&v)[8]) {
  const int kind = vc.kind[c];
  const float s = __ldg(vc.scale[c] + j);
  if (kind == 0) {
    const uint2 raw = __ldg(reinterpret_cast<const uint2*>(static_cast<const int8_t*>(vc.data[c]) + size_t(j) * d + e0));
#pragma unroll
    for (int r = 0; r < 8; ++r) v[r] = float(int8_t(((r < 4 ? raw.x : raw.y) >> (8 * (r & 3))) & 0xffu)) * s;
  } else if (kind == 1) {
    const uint32_t raw =
        __ldg(reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(vc.data[c]) + size_t(j) * (d >> 1) + (e0 >> 1)));
    const float* lv = vc.levels[c];
#pragma unroll
    for (int r = 0; r < 8; ++r) v[r] = __ldg(lv + ((raw >> (4 * r)) & 15u)) * s;
  } else {
    const int bps = vc.bps[c], tw = 2 * bps, tbits = 64 * bps;
    const float ws = s * vc.wscale[c];
    const uint32_t* tile = static_cast<const uint32_t*>(vc.data[c]) + (size_t(j >> 4) * (d >> 4) + (e0 >> 4)) * tw;
    const int k0 = (e0 & 15) >> 2;
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      const int t = 4 * (j & 15) + k0 + kk;
      const int start = ((t + 1) * bps - 16 + tbits) % tbits;
      const int wi = start >> 5, sh = start & 31;
      const uint32_t st = __funnelshift_r(__ldg(tile + wi), __ldg(tile + (wi + 1) % tw), sh) & 0xffffu;
      const int packed = __ldg(table + ((st * HASH_MUL) >> (32 - TB)));
#pragma unroll
      for (int q = 0; q < 4; ++q) v[4 * kk + q] = float(int8_t((packed >> (8 * q)) & 0xff)) * ws;
    }
  }
}

// grid (tokens, d / RB), RB / 8 threads; out-of-range ids embed to zeros
template <int RB, int TB>
__global__ void __launch_bounds__(RB / 8)
itq_vocab_embed_kernel(const int* __restrict__ ids32, const long long* __restrict__ ids64,
                       const int* __restrict__ map, VocabClasses vc, const int* __restrict__ table,
                       const int8_t* __restrict__ su, __nv_bfloat16* __restrict__ out, int d, int vocab) {
  constexpr int W = RB / 256;
  __shared__ __align__(16) float sm[W > 1 ? RB : 4];
  pdl_wait();
  pdl_trigger();
  const int token = blockIdx.x, base = blockIdx.y * RB, t = threadIdx.x;
  const long long id = ids64 ? ids64[token] : (long long)ids32[token];
  float v[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  if (id >= 0 && id < vocab) {
    const int code = __ldg(map + id);
    vocab_decode8<TB>(vc, table, code >> 24, code & 0xffffff, d, base + (t >> 5) * 256 + (t & 31) * 8, v);
  }
  fwht_regs<W>(v, sm, true);
  const float norm = rsqrtf(float(RB));
  __nv_bfloat16* row = out + size_t(token) * d + base;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int e = fwht_index<W>(i);
    row[e] = __float2bfloat16(v[i] * norm * float(su[base + e]));
  }
}

// int8 (kind 0) or NF4 (kind 1, levels as int8 in lv8) rows x prepared activations; one warp per row,
// writes logits[b][rows[j]] = scale[j] * dot.  d % 1024 == 0.
template <int NB, bool NF4>
__global__ void __launch_bounds__(128)
itq_vocab_gemv_kernel(const uint8_t* __restrict__ data, const int* __restrict__ lv8_g, const int* __restrict__ rows,
                      const float* __restrict__ scale, const int* __restrict__ xq, const float* __restrict__ xs,
                      float* __restrict__ logits, int n_rows, int d, int logits_stride, int act_rows) {
  __shared__ int lv8[16];
  if (NF4 && threadIdx.x < 16) lv8[threadIdx.x] = lv8_g[threadIdx.x] & 0xff;
  __syncthreads();
  pdl_trigger();
  pdl_wait();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int xq_row = d >> 2, xs_row = d >> 5, groups = d >> 10;
  for (int j = blockIdx.x * 4 + warp; j < n_rows; j += gridDim.x * 4) {
    float acc[NB];
#pragma unroll
    for (int b = 0; b < NB; ++b) acc[b] = 0.f;
    for (int g = 0; g < groups; ++g) {
      const int e = g * 1024 + lane * 32;   // one 32-column activation group per lane
      int w[8];
      if constexpr (!NF4) {
        const int4* src = reinterpret_cast<const int4*>(data + size_t(j) * d + e);
        const int4 a = __ldg(src), c = __ldg(src + 1);
        w[0] = a.x; w[1] = a.y; w[2] = a.z; w[3] = a.w; w[4] = c.x; w[5] = c.y; w[6] = c.z; w[7] = c.w;
      } else {
        const int4 raw = __ldg(reinterpret_cast<const int4*>(data + size_t(j) * (d >> 1) + (e >> 1)));
        const uint32_t words[4] = {uint32_t(raw.x), uint32_t(raw.y), uint32_t(raw.z), uint32_t(raw.w)};
#pragma unroll
        for (int q = 0; q < 4; ++q)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const uint32_t x = words[q] >> (16 * h);
            w[2 * q + h] = lv8[x & 15u] | (lv8[(x >> 4) & 15u] << 8) | (lv8[(x >> 8) & 15u] << 16) |
                           (lv8[(x >> 12) & 15u] << 24);
          }
      }
#pragma unroll
      for (int b = 0; b < NB; ++b) {
        if (b >= act_rows) break;
        const int4* xv = reinterpret_cast<const int4*>(xq + (size_t(e >> 5) * NB + b) * 8);
        const int4 x0 = __ldg(xv), x1 = __ldg(xv + 1);
        int p = 0;
        p = __dp4a(w[0], x0.x, p); p = __dp4a(w[1], x0.y, p); p = __dp4a(w[2], x0.z, p); p = __dp4a(w[3], x0.w, p);
        p = __dp4a(w[4], x1.x, p); p = __dp4a(w[5], x1.y, p); p = __dp4a(w[6], x1.z, p); p = __dp4a(w[7], x1.w, p);
        acc[b] += float(p) * __ldg(xs + size_t(e >> 5) * NB + b);
      }
    }
    const float sj = __ldg(scale + j);
    const int col = __ldg(rows + j);
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      if (b >= act_rows) break;
      float total = acc[b];
#pragma unroll
      for (int o = 16; o; o >>= 1) total += __shfl_xor_sync(0xffffffffu, total, o);
      if (lane == 0) logits[size_t(b) * logits_stride + col] = total * sj;
    }
  }
}

// ---- tensor-core vocabulary head, 8-32 activation rows (int8 / NF4 classes) ---------------------------------------
// Bit-identical to itq_vocab_gemv_kernel.  There, lane L of a row's warp sums the 32-column groups kb = 32 g + L
// (g = 0 .. d/1024 - 1) in order (one FFMA per group), then the warp adds the 32 lane sums with an xor tree (offsets
// 16, 8, 4, 2, 1), whose bracketing pairs leaf L with L ^ 16 first, ...: leaves in bit-reversed order, added pairwise.
// Here a warp owns 16 class rows and computes every lane sum ("leaf") with mma.m16n8k32 over the same k-blocks (one
// staged 16 x 32 int8 A tile per k-block, as in mma_item), then adds the leaves with that bracketing (vocab_tree).
__host__ __device__ constexpr int bitrev5(int i) {
  return ((i & 1) << 4) | ((i & 2) << 2) | (i & 4) | ((i & 8) >> 2) | ((i & 16) >> 4);
}

struct VocabMmaCtx {
  const uint8_t* data;
  const int* lv8;          // shared: NF4 level -> int8
  const int* xq;
  const float* xs;
  uint32_t* stage;         // this warp's 16 x MMA_ROW_WORDS staging tile
  int j0, n_rows, d, groups;
};

template <bool NF4>
__device__ __forceinline__ uint4 vocab_tile_load(const VocabMmaCtx& c, int srow, int half, int kb, bool valid) {
  if (!valid) return make_uint4(0u, 0u, 0u, 0u);
  if constexpr (!NF4) {
    return __ldg(reinterpret_cast<const uint4*>(c.data + size_t(c.j0 + srow) * c.d + kb * 32 + 16 * half));
  } else {
    // 16 nibbles = columns 16 half .. +15 of the k-block (low nibble = even column), mapped to int8 levels
    const uint2 raw = __ldg(reinterpret_cast<const uint2*>(c.data + size_t(c.j0 + srow) * (c.d >> 1) + kb * 16 + 8 * half));
    uint32_t o[4];
#pragma unroll
    for (int q = 0; q < 2; ++q)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const uint32_t x = (q ? raw.y : raw.x) >> (16 * h);
        o[2 * q + h] = uint32_t(c.lv8[x & 15u] | (c.lv8[(x >> 4) & 15u] << 8) | (c.lv8[(x >> 8) & 15u] << 16) |
                                (c.lv8[(x >> 12) & 15u] << 24));
      }
    return make_uint4(o[0], o[1], o[2], o[3]);
  }
}

// acc[ch][j] = lane L's sum (rows gq + 8 (j >> 1), activation row 8 ch + 2 tq + (j & 1))
template <int NB, bool NF4>
__device__ __forceinline__ void vocab_leaf(const VocabMmaCtx& c, int L, float (&acc)[NB / 8][4]) {
  constexpr int CH = NB / 8;
  const int lane = threadIdx.x & 31, gq = lane >> 2, tq = lane & 3, srow = lane >> 1, half = lane & 1;
  const bool valid = c.j0 + srow < c.n_rows;
#pragma unroll
  for (int ch = 0; ch < CH; ++ch)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[ch][j] = 0.f;
  uint4 next = vocab_tile_load<NF4>(c, srow, half, L, valid);
  for (int g = 0; g < c.groups; ++g) {
    const int kb = 32 * g + L;
    const uint4 w = next;
    if (g + 1 < c.groups) next = vocab_tile_load<NF4>(c, srow, half, kb + 32, valid);
    *reinterpret_cast<uint4*>(c.stage + srow * MMA_ROW_WORDS + 4 * half) = w;
    __syncwarp();
    uint32_t a[4];
    load_a_frag(c.stage, lane, a);
    __syncwarp();
#pragma unroll
    for (int ch = 0; ch < CH; ++ch) {
      const int* xr = c.xq + (size_t(kb) * NB + 8 * ch + gq) * 8;
      int dd[4];
      mma_s8_16832(dd, a, uint32_t(__ldg(xr + tq)), uint32_t(__ldg(xr + 4 + tq)));
      const float s0 = __ldg(c.xs + size_t(kb) * NB + 8 * ch + 2 * tq);
      const float s1 = __ldg(c.xs + size_t(kb) * NB + 8 * ch + 2 * tq + 1);
      acc[ch][0] += float(dd[0]) * s0;
      acc[ch][1] += float(dd[1]) * s1;
      acc[ch][2] += float(dd[2]) * s0;
      acc[ch][3] += float(dd[3]) * s1;
    }
  }
}

// sum of the leaves bitrev5(I0 .. I0 + 2^K - 1), bracketed as the xor tree
template <int K, int I0, int NB, bool NF4>
__device__ __forceinline__ void vocab_tree(const VocabMmaCtx& c, float (&out)[NB / 8][4]) {
  if constexpr (K == 0) {
    vocab_leaf<NB, NF4>(c, bitrev5(I0), out);
  } else {
    float right[NB / 8][4];
    vocab_tree<K - 1, I0, NB, NF4>(c, out);
    vocab_tree<K - 1, I0 + (1 << (K - 1)), NB, NF4>(c, right);
#pragma unroll
    for (int ch = 0; ch < NB / 8; ++ch)
#pragma unroll
      for (int j = 0; j < 4; ++j) out[ch][j] = out[ch][j] + right[ch][j];
  }
}

template <int NB, bool NF4>
__global__ void __launch_bounds__(128)
itq_vocab_mma_kernel(const uint8_t* __restrict__ data, const int* __restrict__ lv8_g, const int* __restrict__ rows,
                     const float* __restrict__ scale, const int* __restrict__ xq, const float* __restrict__ xs,
                     float* __restrict__ logits, int n_rows, int d, int logits_stride, int act_rows) {
  __shared__ int lv8[16];
  __shared__ __align__(16) uint32_t stage[4][16 * MMA_ROW_WORDS];
  if (NF4 && threadIdx.x < 16) lv8[threadIdx.x] = lv8_g[threadIdx.x] & 0xff;
  __syncthreads();
  pdl_trigger();
  pdl_wait();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, gq = lane >> 2, tq = lane & 3;
  const int items = (n_rows + 15) >> 4;
  for (int item = blockIdx.x * 4 + warp; item < items; item += gridDim.x * 4) {
    const VocabMmaCtx c{data, lv8, xq, xs, stage[warp], item * 16, n_rows, d, d >> 10};
    float out[NB / 8][4];
    vocab_tree<5, 0, NB, NF4>(c, out);
#pragma unroll
    for (int ch = 0; ch < NB / 8; ++ch)
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int jj = item * 16 + gq + 8 * (j >> 1), b = 8 * ch + 2 * tq + (j & 1);
        if (jj < n_rows && b < act_rows)
          logits[size_t(b) * logits_stride + __ldg(rows + jj)] = out[ch][j] * __ldg(scale + jj);
      }
  }
}

// ---- tensor-core vocabulary head v2: same numerics, streamed weights ------------------------------------------------
// Bit-identical to itq_vocab_mma_kernel (and so to itq_vocab_gemv_kernel): leaf L of a class row is the FMA chain over
// the k-blocks kb = 32 g + L (g = 0 .. d/1024 - 1, in order) of float(int32 dot) * xs, and the 32 leaves are added with
// vocab_tree's bracketing.  Only the memory side changes.  v1 has each warp read 32-byte pieces of its 16 rows in
// bit-reversed leaf order with one load in flight (RTX 3060: ~110 GB/s int8, ~60 GB/s NF4 of 360; RTX PRO 6000:
// ~0.5 TB/s).  Here a block of 8 warps owns 16 class rows at a time and streams them through shared memory in
// 1024-column chunks (one chunk = the 32 leaves' k-blocks of one g) with 16-byte cp.async, VM2_STAGES deep; warp w
// computes leaves w, w + 8, w + 16, w + 24 (vocab_tree's two lowest levels, in registers) and the three upper levels
// are added through shared memory.  The weights do not depend on earlier kernels, so the first stages are requested
// before the PDL wait.
constexpr int VM2_WARPS = 8;
constexpr int VM2_STAGES = 3;
template <bool NF4>
struct Vm2Tile {
  static constexpr int ROW_BYTES = NF4 ? 512 : 1024;   // one 1024-column chunk of a row
  static constexpr int PITCH = ROW_BYTES + 16;         // padded: conflict-free fragment loads (row r -> banks + 4 r)
  static constexpr int BYTES = 16 * PITCH;
  static constexpr int CHUNKS = ROW_BYTES;             // 16-byte pieces per 16-row tile (16 rows x ROW_BYTES / 16)
};

__device__ __forceinline__ void vm2_cp_async16(void* smem, const void* gmem, bool valid) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(s), "l"(gmem), "r"(valid ? 16 : 0) : "memory");
}
__device__ __forceinline__ void vm2_cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void vm2_cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

__device__ __forceinline__ uint32_t vm2_nf4_word(const int* lv8, uint32_t x) {   // 4 nibbles -> 4 int8 levels
  return uint32_t(lv8[x & 15u] | (lv8[(x >> 4) & 15u] << 8) | (lv8[(x >> 8) & 15u] << 16) | (lv8[(x >> 12) & 15u] << 24));
}

template <int NB, bool NF4>
__global__ void __launch_bounds__(VM2_WARPS * 32, 1)
itq_vocab_mma2_kernel(const uint8_t* __restrict__ data, const int* __restrict__ lv8_g, const int* __restrict__ rows,
                      const float* __restrict__ scale, const int* __restrict__ xq, const float* __restrict__ xs,
                      float* __restrict__ logits, int n_rows, int d, int logits_stride, int act_rows) {
  using T = Vm2Tile<NF4>;
  constexpr int CH = NB / 8, THREADS = VM2_WARPS * 32;
  extern __shared__ __align__(16) uint8_t vm2_smem[];
  uint8_t* tiles = vm2_smem;                                                      // VM2_STAGES x T::BYTES
  float* tree = reinterpret_cast<float*>(vm2_smem + VM2_STAGES * T::BYTES);      // [VM2_WARPS][16][NB]
  __shared__ int lv8[16];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gq = lane >> 2, tq = lane & 3;
  const int groups = d >> 10;
  const int items = (n_rows + 15) >> 4;
  const int my_items = int(blockIdx.x) < items ? (items - 1 - int(blockIdx.x)) / int(gridDim.x) + 1 : 0;
  const int stages = my_items * groups;
  const size_t row_bytes = NF4 ? size_t(d >> 1) : size_t(d);

  auto issue = [&](int s) {   // stage s = (item s / groups, chunk g = s % groups) -> buffer s % VM2_STAGES
    if (s < stages) {
      const int item = int(blockIdx.x) + (s / groups) * int(gridDim.x), g = s % groups;
      uint8_t* tile = tiles + (s % VM2_STAGES) * T::BYTES;
#pragma unroll
      for (int i = 0; i < T::CHUNKS / THREADS; ++i) {
        const int c = tid + i * THREADS, r = c / (T::ROW_BYTES / 16), col = c % (T::ROW_BYTES / 16);
        const int j = item * 16 + r;
        const bool valid = j < n_rows;   // rows past the class end are zero-filled (their outputs are not written)
        vm2_cp_async16(tile + r * T::PITCH + col * 16,
                       data + size_t(valid ? j : 0) * row_bytes + size_t(g) * T::ROW_BYTES + col * 16, valid);
      }
    }
    vm2_cp_async_commit();   // always one group per stage slot: uniform wait counts
  };

#pragma unroll
  for (int s = 0; s < VM2_STAGES - 1; ++s) issue(s);
  if (NF4 && tid < 16) lv8[tid] = lv8_g[tid] & 0xff;
  pdl_trigger();
  pdl_wait();

  float acc[4][CH][4];
  for (int s = 0; s < stages; ++s) {
    const int g = s % groups;
    vm2_cp_async_wait<VM2_STAGES - 2>();
    __syncthreads();                    // stage s is in shared memory for everyone; buffer (s - 1) % STAGES is free
    issue(s + VM2_STAGES - 1);
    if (g == 0) {
#pragma unroll
      for (int li = 0; li < 4; ++li)
#pragma unroll
        for (int ch = 0; ch < CH; ++ch)
#pragma unroll
          for (int j = 0; j < 4; ++j) acc[li][ch][j] = 0.f;
    }
    const uint8_t* tile = tiles + (s % VM2_STAGES) * T::BYTES;
#pragma unroll
    for (int li = 0; li < 4; ++li) {
      const int L = warp + 8 * li, kb = 32 * g + L;
      uint32_t a[4];
      if constexpr (!NF4) {
        const uint8_t* base = tile + L * 32 + 4 * tq;
        a[0] = *reinterpret_cast<const uint32_t*>(base + gq * T::PITCH);
        a[1] = *reinterpret_cast<const uint32_t*>(base + (gq + 8) * T::PITCH);
        a[2] = *reinterpret_cast<const uint32_t*>(base + gq * T::PITCH + 16);
        a[3] = *reinterpret_cast<const uint32_t*>(base + (gq + 8) * T::PITCH + 16);
      } else {
        // a k-block is 16 bytes; columns 4 tq .. 4 tq + 3 are bytes 2 tq, 2 tq + 1 (low nibble = even column)
        const uint8_t* base = tile + L * 16 + 2 * tq;
        a[0] = vm2_nf4_word(lv8, *reinterpret_cast<const uint16_t*>(base + gq * T::PITCH));
        a[1] = vm2_nf4_word(lv8, *reinterpret_cast<const uint16_t*>(base + (gq + 8) * T::PITCH));
        a[2] = vm2_nf4_word(lv8, *reinterpret_cast<const uint16_t*>(base + gq * T::PITCH + 8));
        a[3] = vm2_nf4_word(lv8, *reinterpret_cast<const uint16_t*>(base + (gq + 8) * T::PITCH + 8));
      }
#pragma unroll
      for (int ch = 0; ch < CH; ++ch) {
        const int* xr = xq + (size_t(kb) * NB + 8 * ch + gq) * 8;
        int dd[4];
        mma_s8_16832(dd, a, uint32_t(__ldg(xr + tq)), uint32_t(__ldg(xr + 4 + tq)));
        const float s0 = __ldg(xs + size_t(kb) * NB + 8 * ch + 2 * tq);
        const float s1 = __ldg(xs + size_t(kb) * NB + 8 * ch + 2 * tq + 1);
        acc[li][ch][0] += float(dd[0]) * s0;
        acc[li][ch][1] += float(dd[1]) * s1;
        acc[li][ch][2] += float(dd[2]) * s0;
        acc[li][ch][3] += float(dd[3]) * s1;
      }
    }
    if (g == groups - 1) {
      // vocab_tree<2, 4 i>: (leaf w + leaf w + 16) + (leaf w + 8 + leaf w + 24), w = bitrev5(4 i)
      float* tw = tree + warp * 16 * NB;
#pragma unroll
      for (int ch = 0; ch < CH; ++ch)
#pragma unroll
        for (int j = 0; j < 4; ++j)
          tw[(gq + 8 * (j >> 1)) * NB + 8 * ch + 2 * tq + (j & 1)] =
              (acc[0][ch][j] + acc[2][ch][j]) + (acc[1][ch][j] + acc[3][ch][j]);
      __syncthreads();
      // the three upper levels over i = 0 .. 7 (w = 0, 4, 2, 6, 1, 5, 3, 7)
      const int item = int(blockIdx.x) + (s / groups) * int(gridDim.x);
      for (int o = tid; o < 16 * NB; o += THREADS) {
        const float* t = tree + o;
        const float total = ((t[0] + t[4 * 16 * NB]) + (t[2 * 16 * NB] + t[6 * 16 * NB])) +
                            ((t[1 * 16 * NB] + t[5 * 16 * NB]) + (t[3 * 16 * NB] + t[7 * 16 * NB]));
        const int jj = item * 16 + o / NB, b = o % NB;
        if (jj < n_rows && b < act_rows) logits[size_t(b) * logits_stride + __ldg(rows + jj)] = total * __ldg(scale + jj);
      }
    }
  }
  vm2_cp_async_wait<0>();
}

bool use_vocab_mma2() {   // ITQ_VOCAB_MMA2=0 keeps v1
  static int cached = -1;
  if (cached < 0) {
    const char* env = std::getenv("ITQ_VOCAB_MMA2");
    cached = !(env != nullptr && env[0] == '0');
  }
  return cached == 1;
}

template <int NB, bool NF4>
int launch_vocab_mma2(cudaStream_t stream, const uint8_t* dd, const int* l, const int* r, const float* sc, const int* q,
                      const float* s, float* o, int n_rows, int d, int logits_stride, int act_rows) {
  using T = Vm2Tile<NF4>;
  constexpr size_t smem = size_t(VM2_STAGES) * T::BYTES + size_t(VM2_WARPS) * 16 * NB * sizeof(float);
  static int per_sm = 0;
  if (per_sm == 0) {
    cudaFuncSetAttribute(itq_vocab_mma2_kernel<NB, NF4>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, itq_vocab_mma2_kernel<NB, NF4>, VM2_WARPS * 32, smem) !=
            cudaSuccess || per_sm < 1)
      per_sm = 1;
  }
  const int items = (n_rows + 15) / 16;
  const int blocks = items < sm_count() * per_sm ? items : sm_count() * per_sm;
  return launch_k(itq_vocab_mma2_kernel<NB, NF4>, dim3(blocks), dim3(VM2_WARPS * 32), smem, stream, dd, l, r, sc, q, s,
                  o, n_rows, d, logits_stride, act_rows);
}

// Tensor-core vocabulary head (sm_80+ with use_mma()): ITQ_VOCAB_MMA=0 keeps the dp4a kernel (<= 16 rows).
bool use_vocab_mma() {
  static int cached = -1;
  if (cached < 0) {
    const char* env = std::getenv("ITQ_VOCAB_MMA");
    cached = use_mma() && !(env != nullptr && env[0] == '0');
  }
  return cached == 1;
}

// logits[b][rows[j]] = y[b][j] * scale[j]   (ITQ classes: y from itq_gemv, wscale already applied)
__global__ void itq_vocab_scatter_kernel(const float* __restrict__ y, const int* __restrict__ rows,
                                         const float* __restrict__ scale, float* __restrict__ logits, int n_rows,
                                         int y_stride, int logits_stride) {
  pdl_wait();
  const int b = blockIdx.y;
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n_rows; j += gridDim.x * blockDim.x)
    logits[size_t(b) * logits_stride + __ldg(rows + j)] = y[size_t(b) * y_stride + j] * __ldg(scale + j);
}

}  // namespace

extern "C" int itq_vocab_embed(const void* ids, int ids_are_64, const void* map, int classes, const void* const* data,
                               const void* const* scales, const void* const* levels, const float* wscales,
                               const int* kinds, const int* bpss, const void* table, const void* su, void* out,
                               int tokens, int d, int block, int vocab, void* stream_pointer) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  if (classes < 1 || classes > VOCAB_MAX_CLASSES || d % block || tokens < 0) return int(cudaErrorInvalidValue);
  if (tokens == 0) return 0;
  VocabClasses vc = {};
  for (int c = 0; c < classes; ++c) {
    vc.data[c] = data[c];
    vc.scale[c] = static_cast<const float*>(scales[c]);
    vc.levels[c] = static_cast<const float*>(levels[c]);
    vc.wscale[c] = wscales[c];
    vc.kind[c] = kinds[c];
    vc.bps[c] = bpss[c];
    if (kinds[c] < 0 || kinds[c] > 2 || (kinds[c] == 2 && (bpss[c] < 4 || bpss[c] > 12))) return int(cudaErrorInvalidValue);
  }
  auto i32 = ids_are_64 ? nullptr : static_cast<const int*>(ids);
  auto i64 = ids_are_64 ? static_cast<const long long*>(ids) : nullptr;
  auto m = static_cast<const int*>(map);
  auto t = static_cast<const int*>(table);
  auto s = static_cast<const int8_t*>(su);
  auto o = static_cast<__nv_bfloat16*>(out);
  const dim3 grid(tokens, d / block);
#define E(RB) if (block == RB) return launch_k(itq_vocab_embed_kernel<RB, 12>, grid, dim3(RB / 8), 0, stream, \
                                               i32, i64, m, vc, t, s, o, d, vocab);
  E(256) E(512) E(1024) E(2048)
#undef E
  return int(cudaErrorInvalidValue);
}

extern "C" int itq_vocab_gemv(const void* data, const void* lv8, const void* rows, const void* scale, const void* xq,
                              const void* xs, void* logits, int n_rows, int d, int logits_stride, int act_rows,
                              int nf4, void* stream_pointer) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  if (d % 1024 || act_rows < 1 || act_rows > 32 || n_rows < 0) return int(cudaErrorInvalidValue);
  if (act_rows > 16 && !use_vocab_mma()) return int(cudaErrorInvalidValue);
  if (n_rows == 0) return 0;
  const int nb = pad_rows(act_rows);
  auto dd = static_cast<const uint8_t*>(data);
  auto l = static_cast<const int*>(lv8);
  auto r = static_cast<const int*>(rows);
  auto sc = static_cast<const float*>(scale);
  auto q = static_cast<const int*>(xq);
  auto s = static_cast<const float*>(xs);
  auto o = static_cast<float*>(logits);
  if (nb >= 8 && act_rows > ITQ_VOCAB_DP4A_MAX_ROWS && use_vocab_mma()) {
    if (use_vocab_mma2()) {
#define VM2(N) if (nb == N) return nf4 ? launch_vocab_mma2<N, true>(stream, dd, l, r, sc, q, s, o, n_rows, d, logits_stride, \
                                                                  act_rows) \
                                   : launch_vocab_mma2<N, false>(stream, dd, l, r, sc, q, s, o, n_rows, d, logits_stride, \
                                                                  act_rows);
      VM2(8) VM2(16) VM2(32)
#undef VM2
    }
    const int items = (n_rows + 15) / 16;
    const int mblocks = (items + 3) / 4 < sm_count() * 8 ? (items + 3) / 4 : sm_count() * 8;
#define VM(N) if (nb == N) { \
    if (nf4) return launch_k(itq_vocab_mma_kernel<N, true>, dim3(mblocks), dim3(128), 0, stream, dd, l, r, sc, q, s, o, \
                             n_rows, d, logits_stride, act_rows); \
    return launch_k(itq_vocab_mma_kernel<N, false>, dim3(mblocks), dim3(128), 0, stream, dd, l, r, sc, q, s, o, n_rows, \
                    d, logits_stride, act_rows); }
    VM(8) VM(16) VM(32)
#undef VM
    return int(cudaErrorInvalidValue);
  }
  const int blocks = (n_rows + 3) / 4 < sm_count() * 16 ? (n_rows + 3) / 4 : sm_count() * 16;
#define V(N) if (nb == N) { \
    if (nf4) return launch_k(itq_vocab_gemv_kernel<N, true>, dim3(blocks), dim3(128), 0, stream, dd, l, r, sc, q, s, o, \
                             n_rows, d, logits_stride, act_rows); \
    return launch_k(itq_vocab_gemv_kernel<N, false>, dim3(blocks), dim3(128), 0, stream, dd, l, r, sc, q, s, o, n_rows, \
                    d, logits_stride, act_rows); }
  V(1) V(2) V(4) V(8) V(16)
#undef V
  return int(cudaErrorInvalidValue);
}

extern "C" int itq_vocab_scatter(const void* y, const void* rows, const void* scale, void* logits, int n_rows,
                                 int y_stride, int logits_stride, int act_rows, void* stream_pointer) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  if (n_rows <= 0 || act_rows <= 0) return 0;
  const int threads = 256;
  int blocks = (n_rows + threads - 1) / threads;
  if (blocks > 1024) blocks = 1024;
  return launch_k(itq_vocab_scatter_kernel, dim3(blocks, act_rows), dim3(threads), 0, stream,
                  static_cast<const float*>(y), static_cast<const int*>(rows), static_cast<const float*>(scale),
                  static_cast<float*>(logits), n_rows, y_stride, logits_stride);
}

// prefetch argument: PF + 1000 * zeroed (zeroed = the output was already cleared by itq_prep)
extern "C" int itq_gemv(const void* words, const void* table, const void* xq, const void* xs, void* y,
                        int m, int n, int bps, int table_bits, int rows, int ksplit, float wscale,
                        int blocks, int prefetch, void* stream_pointer) {
  const bool zeroed = prefetch >= 1000;
  prefetch %= 1000;
  auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  if (m % 16 || n % 32 || table_bits != fmt_table_bits(bps)) return int(cudaErrorInvalidValue);
  const int nb = pad_rows(rows);
  if (rows > 64 || (rows > 16 && !use_mma())) return int(cudaErrorInvalidValue);
  auto w = reinterpret_cast<const uint32_t*>(words);
  auto t = reinterpret_cast<const int*>(table);
  auto q = reinterpret_cast<const int*>(xq);
  auto s = reinterpret_cast<const float*>(xs);
  auto o = reinterpret_cast<float*>(y);
  if (nb >= 8 && use_mma()) {
    const int pf = prefetch >= 2 ? 2 : 1;
#define M(B, N, P) if (bps == B && nb == N && pf == P) \
    return launch_gemv_mma<B, fmt_table_bits(B), N, P>(w, t, q, s, o, m, n, ksplit, wscale, blocks, zeroed, stream);
#define MP(B, N) M(B, N, 1) M(B, N, 2)
#define MN(B) MP(B, 8) MP(B, 16) MP(B, 32) MP(B, 48) MP(B, 64)
    MN(4) MN(5) MN(6) MN(7) MN(8) MN(9) MN(10) MN(11) MN(12) MN(35) MN(36) MN(37) MN(38) MN(39) MN(40)
#undef MN
#undef MP
#undef M
    return int(cudaErrorInvalidValue);
  }
#define G(B, N, P) if (bps == B && nb == N && prefetch == P) \
    return launch_gemv<B, fmt_table_bits(B), N, P>(w, t, q, s, o, m, n, ksplit, wscale, blocks, zeroed, stream);
#define GP(B, N) G(B, N, 1) G(B, N, 2) G(B, N, 4)
#define GN(B) GP(B, 1) GP(B, 2) GP(B, 4) GP(B, 8) GP(B, 16)
  GN(4) GN(5) GN(6) GN(7) GN(8) GN(9) GN(10) GN(11) GN(12) GN(35) GN(36) GN(37) GN(38) GN(39) GN(40)
#undef GN
#undef GP
#undef G
  return int(cudaErrorInvalidValue);
}

extern "C" int itq_prep(const void* x, int x_is_bf16, const void* pis, const void* su, void* xq, void* xs,
                        void* xr, int rows, int n, int x_stride, int block, int to_int8, void* zero,
                        int zero_count, void* stream_pointer) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  auto p = reinterpret_cast<const float*>(pis);
  auto sgn = reinterpret_cast<const int8_t*>(su);
  auto q = reinterpret_cast<int*>(xq);
  auto s = reinterpret_cast<float*>(xs);
  auto r = reinterpret_cast<__nv_bfloat16*>(xr);
  auto z = reinterpret_cast<float*>(zero);
  if (n % block || (to_int8 && rows > 64)) return int(cudaErrorInvalidValue);   // pair-major int8 rows <= 64
#define P(RB) if (block == RB) { \
    if (x_is_bf16) return launch_prep<RB>(reinterpret_cast<const __nv_bfloat16*>(x), p, sgn, q, s, r, rows, n, x_stride, to_int8 != 0, z, zero_count, stream); \
    return launch_prep<RB>(reinterpret_cast<const float*>(x), p, sgn, q, s, r, rows, n, x_stride, to_int8 != 0, z, zero_count, stream); }
  P(32) P(64) P(128) P(256) P(512) P(1024) P(2048) P(4096)
#undef P
  return int(cudaErrorInvalidValue);
}

// Decode: RMSNormGated(x, z) (128-wide heads, silu, F32 weight w) + the consumer's int8 prep, one launch
// (itq_gated_norm_prep_kernel).  x / z: BF16 rows of n elements (row strides in elements, 16-byte aligned).
extern "C" int itq_gated_norm_prep(const void* x, int x_stride, const void* z, int z_stride, const void* w, float eps,
                                   int head_dim, const void* pis, const void* su, void* xq, void* xs, int rows, int n,
                                   int block, void* zero, int zero_count, void* stream_pointer) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  if (head_dim != 128 || rows < 1 || rows > 64 || n % block || block % 128 || x_stride % 8 || z_stride % 8 ||
      !aligned16(x) || !aligned16(z) || !aligned16(w) || (pis != nullptr && !aligned16(pis)) ||
      (reinterpret_cast<uintptr_t>(su) & 7u) || (reinterpret_cast<uintptr_t>(xq) & 7u))
    return int(cudaErrorInvalidValue);
  const dim3 grid(n / block, rows);
  auto xx = reinterpret_cast<const __nv_bfloat16*>(x);
  auto zz = reinterpret_cast<const __nv_bfloat16*>(z);
  auto ww = reinterpret_cast<const float*>(w);
  auto p = reinterpret_cast<const float*>(pis);
  auto sgn = reinterpret_cast<const int8_t*>(su);
  auto q = reinterpret_cast<int*>(xq);
  auto s = reinterpret_cast<float*>(xs);
  auto zr = reinterpret_cast<float*>(zero);
#define GN(RB) if (block == RB) return launch_k(itq_gated_norm_prep_kernel<RB>, grid, dim3(RB / 8), 0, stream, xx, \
                                                x_stride, zz, z_stride, ww, eps, p, sgn, q, s, zr, zero_count, \
                                                pad_rows(rows));
  GN(256) GN(512) GN(1024) GN(2048)
#undef GN
  return int(cudaErrorInvalidValue);
}

// Warps per block of the decode GEMV that itq_gemv / itq_gemv_group run for this many rows (the Python side sizes
// its grids with it: items per block).
extern "C" int itq_gemv_warps(int rows, int bps, int group) {
  const int nb = pad_rows(rows);
  if (nb >= 8 && use_mma()) return ITQ_MMA_WARPS;
  return group ? group_warps(bps, nb) : ITQ_GEMV_WARPS;
}

extern "C" int itq_quant_rows(const void* x, void* q, void* scale, int rows, int n, void* stream_pointer) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  if (rows < 1 || n < 1) return int(cudaErrorInvalidValue);
  itq_quant_rows_kernel<<<rows, 1024, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(x),
                                                    reinterpret_cast<int8_t*>(q), reinterpret_cast<float*>(scale), n);
  return int(cudaGetLastError());
}

// int32 GEMM output (torch._int_mm) with a per-row F32 scale (activation scale x wscale): the int8 prefill epilogue.
extern "C" int itq_post_i32(const void* y, const void* row_scale, const void* sv, const void* pos, const void* bias,
                            void* out, int rows, int m, int out_stride, int y_stride, int block, void* stream_pointer) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  auto yi = reinterpret_cast<const int*>(y);
  auto rs = reinterpret_cast<const float*>(row_scale);
  auto s = reinterpret_cast<const int8_t*>(sv);
  auto p = reinterpret_cast<const float*>(pos);
  auto bb = reinterpret_cast<const float*>(bias);
  auto o = reinterpret_cast<__nv_bfloat16*>(out);
  if (m % block) return int(cudaErrorInvalidValue);
#define Q(RB) if (block == RB) return launch_post<RB>(nullptr, s, p, bb, o, rows, m, out_stride, y_stride, stream, yi, rs);
  Q(16) Q(32) Q(64) Q(128) Q(256) Q(512) Q(1024) Q(2048) Q(4096)
#undef Q
  return int(cudaErrorInvalidValue);
}

extern "C" int itq_post(const void* y, const void* sv, const void* pos, const void* bias, void* out,
                        int rows, int m, int out_stride, int y_stride, int block, void* stream_pointer) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  auto yy = reinterpret_cast<const float*>(y);
  auto s = reinterpret_cast<const int8_t*>(sv);
  auto p = reinterpret_cast<const float*>(pos);
  auto bb = reinterpret_cast<const float*>(bias);
  auto o = reinterpret_cast<__nv_bfloat16*>(out);
  if (m % block) return int(cudaErrorInvalidValue);
#define Q(RB) if (block == RB) return launch_post<RB>(yy, s, p, bb, o, rows, m, out_stride, y_stride, stream);
  Q(16) Q(32) Q(64) Q(128) Q(256) Q(512) Q(1024) Q(2048) Q(4096)
#undef Q
  return int(cudaErrorInvalidValue);
}

// Fused decode MLP middle (see itq_swiglu_prep_kernel): yg / yu are the gate / up GEMV outputs (row stride
// y_stride), xq / xs down's int8 activations ([nb, m / 4] / [nb, m / 32]); zero clears down's split-K output.
extern "C" int itq_swiglu_prep(const void* yg, const void* yu, int y_stride, const void* svg, const void* svu,
                               const void* posg, const void* posu, const void* pis, const void* su, void* xq,
                               void* xs, int rows, int m, int block, void* zero, int zero_count,
                               void* stream_pointer) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  if (rows < 1 || rows > 64 || block < 256 || block > 2048 || m % block || y_stride % 4) return int(cudaErrorInvalidValue);
  if (!aligned16(yg) || !aligned16(yu) || (pis && !aligned16(pis)) || (reinterpret_cast<uintptr_t>(su) & 7u))
    return int(cudaErrorInvalidValue);
  const dim3 grid(m / block, rows);
  auto f = [](const void* p) { return reinterpret_cast<const float*>(p); };
  auto s8 = [](const void* p) { return reinterpret_cast<const int8_t*>(p); };
#define S(RB) if (block == RB) return launch_k(itq_swiglu_prep_kernel<RB>, grid, dim3(RB / 8), 0, stream, f(yg), f(yu), \
    y_stride, s8(svg), s8(svu), f(posg), f(posu), f(pis), s8(su), reinterpret_cast<int*>(xq), \
    reinterpret_cast<float*>(xs), m, reinterpret_cast<float*>(zero), zero_count, pad_rows(rows));
  S(256) S(512) S(1024) S(2048)
#undef S
  return int(cudaErrorInvalidValue);
}

// Fused residual-stream step (see itq_add_norm_prep_kernel).  res_bf16: residual dtype (0 = F32).  xq == nullptr:
// norm only (y and r_out), no consumer preparation.  zero clears the consumer's split-K output.
extern "C" int itq_add_norm_prep(const void* y_raw, int y_stride, const void* sv, const void* pos, const void* res,
                                 int res_bf16, int r_stride, void* r_out, const void* w32, float eps, void* y, int rows,
                                 int H, int block, const void* pis, const void* su, void* xq, void* xs, void* zero,
                                 int zero_count, void* stream_pointer) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  if (rows < 1 || H > ADD_NORM_MAX_H || block < 256 || block > 2048 || H % block || y_stride % 4 ||
      !aligned16(y_raw) || (xq && (su == nullptr || (reinterpret_cast<uintptr_t>(su) & 7u) || (pis && !aligned16(pis)))))
    return int(cudaErrorInvalidValue);
  auto f = [](const void* p) { return reinterpret_cast<const float*>(p); };
  auto s8 = [](const void* p) { return reinterpret_cast<const int8_t*>(p); };
  if (block == 1024 && H % 1024 == 0 && H / 1024 <= 8 && use_cluster_norm()) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(H / 1024, rows);
    cfg.blockDim = dim3(128);
    cfg.stream = stream;
    cudaLaunchAttribute attr[2];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = H / 1024;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    int na = 1;
    if (use_pdl()) {
      attr[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attr[1].val.programmaticStreamSerializationAllowed = 1;
      na = 2;
    }
    cfg.attrs = attr;
    cfg.numAttrs = na;
#define C(RT) return int(cudaLaunchKernelEx(&cfg, itq_add_norm_prep_cluster_kernel<RT>, f(y_raw), y_stride, s8(sv), f(pos), \
    reinterpret_cast<const RT*>(res), r_stride, reinterpret_cast<float*>(r_out), f(w32), eps, \
    reinterpret_cast<__nv_bfloat16*>(y), H, f(pis), s8(su), reinterpret_cast<int*>(xq), reinterpret_cast<float*>(xs), \
    reinterpret_cast<float*>(zero), zero_count, pad_rows(rows)));
    if (res_bf16) { C(__nv_bfloat16) } else { C(float) }
#undef C
  }
#define A(RB, RT) if (block == RB) return launch_k(itq_add_norm_prep_kernel<RB, RT>, dim3(rows), dim3(H / 8), 0, stream, \
    f(y_raw), y_stride, s8(sv), f(pos), reinterpret_cast<const RT*>(res), r_stride, reinterpret_cast<float*>(r_out), \
    f(w32), eps, reinterpret_cast<__nv_bfloat16*>(y), H, f(pis), s8(su), reinterpret_cast<int*>(xq), \
    reinterpret_cast<float*>(xs), reinterpret_cast<float*>(zero), zero_count, pad_rows(rows));
  if (res_bf16) { A(256, __nv_bfloat16) A(512, __nv_bfloat16) A(1024, __nv_bfloat16) A(2048, __nv_bfloat16) }
  else { A(256, float) A(512, float) A(1024, float) A(2048, float) }
#undef A
  return int(cudaErrorInvalidValue);
}

template <int BPS, int NB, int PF>
int launch_gemv_group(const GemvGroup& g, const int* table, const int* xq, const float* xs, int n, int blocks,
                      cudaStream_t stream) {
  constexpr int WARPS = group_warps(BPS, NB);
  const size_t smem = sizeof(int) << fmt_table_bits(BPS);
  auto kern = itq_gemv_group_kernel<BPS, fmt_table_bits(BPS), NB, WARPS, PF>;
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
    configured = true;
  }
  return launch_k(kern, dim3(blocks), dim3(WARPS * 32), smem, stream, g, table, xq, xs, n);
}

template <int BPS, int NB, int PF>
int launch_gemv_group_mma(const GemvGroup& g, const int* table, const int* xq, const float* xs, int n, int blocks,
                          cudaStream_t stream) {
  constexpr int WARPS = ITQ_MMA_WARPS;
  const size_t smem = (sizeof(int) << fmt_table_bits(BPS)) + sizeof(uint32_t) * WARPS * MMA_STAGE_WORDS;
  auto kern = itq_gemv_group_mma_kernel<BPS, fmt_table_bits(BPS), NB, WARPS, PF>;
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
    configured = true;
  }
  return launch_k(kern, dim3(blocks), dim3(WARPS * 32), smem, stream, g, table, xq, xs, n);
}

// words/ys: host arrays of device pointers; the output buffers must already be zero for split-K
extern "C" int itq_gemv_group(int parts, const void* const* words, void* const* ys, const int* ms,
                              const int* ksplits, const float* scales, const void* table, const void* xq,
                              const void* xs, int n, int bps, int rows, int blocks, int prefetch,
                              void* stream_pointer) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  if (parts < 1 || parts > MAX_PARTS || n % 32 || rows > 64 || (rows > 16 && !use_mma()))
    return int(cudaErrorInvalidValue);
  GemvGroup g{};
  int items = 0;
  for (int i = 0; i < parts; ++i) {
    if (ms[i] % 16) return int(cudaErrorInvalidValue);
    g.words[i] = reinterpret_cast<const uint32_t*>(words[i]);
    g.y[i] = reinterpret_cast<float*>(ys[i]);
    g.m[i] = ms[i];
    g.ksplit[i] = ksplits[i];
    g.scale[i] = scales[i];
    items += (ms[i] / 16) * ksplits[i];
    g.item_end[i] = items;
  }
  g.parts = parts;
  const int nb = pad_rows(rows);
  auto t = reinterpret_cast<const int*>(table);
  auto q = reinterpret_cast<const int*>(xq);
  auto s = reinterpret_cast<const float*>(xs);
  if (nb >= 8 && use_mma()) {
    const int pf = prefetch >= 2 ? 2 : 1;
#define M(B, N, P) if (bps == B && nb == N && pf == P) return launch_gemv_group_mma<B, N, P>(g, t, q, s, n, blocks, stream);
#define MP(B, N) M(B, N, 1) M(B, N, 2)
#define MN(B) MP(B, 8) MP(B, 16) MP(B, 32) MP(B, 48) MP(B, 64)
    MN(4) MN(5) MN(6) MN(7) MN(8) MN(9) MN(10) MN(11) MN(12) MN(35) MN(36) MN(37) MN(38) MN(39) MN(40)
#undef MN
#undef MP
#undef M
    return int(cudaErrorInvalidValue);
  }
#define G(B, N, P) if (bps == B && nb == N && prefetch == P) return launch_gemv_group<B, N, P>(g, t, q, s, n, blocks, stream);
#define GP(B, N) G(B, N, 1) G(B, N, 2) G(B, N, 4)
#define GN(B) GP(B, 1) GP(B, 2) GP(B, 4) GP(B, 8) GP(B, 16)
  GN(4) GN(5) GN(6) GN(7) GN(8) GN(9) GN(10) GN(11) GN(12) GN(35) GN(36) GN(37) GN(38) GN(39) GN(40)
#undef GN
#undef GP
#undef G
  return int(cudaErrorInvalidValue);
}

extern "C" int itq_post_group(int parts, const void* const* ys, const void* const* svs, const void* const* poss,
                              const void* const* biases, const int* ms, const int* blocks_out, void* out,
                              int out_stride, int rows, void* stream_pointer) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  if (parts < 1 || parts > MAX_PARTS) return int(cudaErrorInvalidValue);
  PostGroup g{};
  int blocks = 0, col = 0;
  for (int i = 0; i < parts; ++i) {
    if (blocks_out[i] > 4096 || ms[i] % blocks_out[i]) return int(cudaErrorInvalidValue);
    g.y[i] = reinterpret_cast<const float*>(ys[i]);
    g.sv[i] = reinterpret_cast<const int8_t*>(svs[i]);
    g.pos[i] = reinterpret_cast<const float*>(poss[i]);
    g.bias[i] = reinterpret_cast<const float*>(biases[i]);
    g.m[i] = ms[i];
    g.rb[i] = blocks_out[i];
    g.col[i] = col;
    col += ms[i];
    blocks += ms[i] / blocks_out[i];
    g.block_end[i] = blocks;
  }
  g.parts = parts;
  g.fast = use_fast_fwht();
  for (int i = 0; i < parts; ++i)
    if (!aligned16(ys[i]) || ms[i] % 4) g.fast = 0;
  return launch_k(itq_post_group_kernel, dim3(blocks, rows), dim3(256), 0, stream, g,
                  reinterpret_cast<__nv_bfloat16*>(out), out_stride);
}

extern "C" int itq_decode(const void* words, const void* table, void* w, int m, int n, int bps,
                          int table_bits, float wscale, void* stream_pointer) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  if (m % 16 || n % 16 || table_bits != fmt_table_bits(bps)) return int(cudaErrorInvalidValue);
  const int tiles = (m / 16) * (n / 16);
  // persistent blocks (8 warps, one tile pair each per iteration): the 16 KB (V2: 4 KB) table is loaded once per block
  const int want = (tiles / 2 + 7) / 8, cap = 4 * sm_count();
  const int blocks = want < cap ? (want > 0 ? want : 1) : cap;
  auto ww = reinterpret_cast<const uint32_t*>(words);
  auto t = reinterpret_cast<const int*>(table);
  auto o = reinterpret_cast<__nv_bfloat16*>(w);
#define D(B) if (bps == B) { itq_decode_kernel<B, fmt_table_bits(B)><<<blocks, 256, (sizeof(int) << fmt_table_bits(B)), \
    stream>>>(ww, t, o, m, n, wscale); return int(cudaGetLastError()); }
  D(4) D(5) D(6) D(7) D(8) D(9) D(10) D(11) D(12) D(35) D(36) D(37) D(38) D(39) D(40)
#undef D
  return int(cudaErrorInvalidValue);
}

// int8 [m, n] row-major (raw table values; the product is scaled by wscale and the activation scales later)
extern "C" int itq_decode_i8(const void* words, const void* table, void* w, int m, int n, int bps,
                             void* stream_pointer) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  if (m % 16 || n % 16) return int(cudaErrorInvalidValue);
  const int tiles = (m / 16) * (n / 16);
  const int want = (tiles / 2 + 7) / 8, cap = 4 * sm_count();
  const int blocks = want < cap ? (want > 0 ? want : 1) : cap;
  auto ww = reinterpret_cast<const uint32_t*>(words);
  auto t = reinterpret_cast<const int*>(table);
  auto o = reinterpret_cast<int8_t*>(w);
#define D(B) if (bps == B) { itq_decode_i8_kernel<B, fmt_table_bits(B)><<<blocks, 256, (sizeof(int) << fmt_table_bits(B)), \
    stream>>>(ww, t, o, m, n); return int(cudaGetLastError()); }
  D(4) D(5) D(6) D(7) D(8) D(9) D(10) D(11) D(12) D(35) D(36) D(37) D(38) D(39) D(40)
#undef D
  return int(cudaErrorInvalidValue);
}

// ---- routed experts (codec itq-moe v1) ------------------------------------------------------------------------------
// One dp4a GEMV row per routed slot: row r of y is W_{ids[r]} x_{r / act_div}.  The expert (its stream offset in words,
// format code and wscale) is read on the device, so a decode step needs no host synchronisation (CUDA-graph safe).
// Experts of one tensor may use different formats: the host launches the kernel once per format code present, and
// work items whose expert has another format return at once.  Activations: pair-major int8 rows of itq_prep_rows
// (stride nb rows), shared by the top-k slots of a token (act_div = k) or one per slot (act_div = 1).
namespace {

template <int BPS, int TB, int WARPS, int PF>
__global__ void __launch_bounds__(WARPS * 32)
itq_moe_gemv_kernel(const uint32_t* __restrict__ words, const long long* __restrict__ woff,
                    const int* __restrict__ fmt, const float* __restrict__ wscale, const int* __restrict__ ids,
                    int rows, int act_div, const int* __restrict__ table_g, const int* __restrict__ xq,
                    const float* __restrict__ xs, int nb, float* __restrict__ y, int m, int n) {
  extern __shared__ int table[];
  for (int i = threadIdx.x; i < (1 << TB); i += WARPS * 32) table[i] = table_g[i];
  __syncthreads();
  constexpr int TW = fmt_words(BPS);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int r = lane & 15, half = lane >> 4, lane_base = lane & 16;
  const int rt = m >> 4, ct = n >> 4, pairs = ct >> 1;
  const int items = rows * rt;
  for (int item = blockIdx.x * WARPS + warp; item < items; item += gridDim.x * WARPS) {
    const int row = item / rt, rb = item - row * rt;
    const int e = __ldg(ids + row);
    if (e < 0 || __ldg(fmt + e) != BPS) continue;            // uniform over the warp: one item per warp
    const int b = row / act_div;
    const uint32_t* rowtiles = words + __ldg(woff + e) + size_t(rb) * ct * TW;
    float acc = 0.f;
    for (int q = 0; q < pairs; q += 2 * PF) {
      TileWords<BPS> wbuf[PF][2];
      int pidx[PF];
      bool valid[PF];
#pragma unroll
      for (int i = 0; i < PF; ++i) {
        const int p = q + half + 2 * i;
        valid[i] = p < pairs;
        pidx[i] = valid[i] ? p : 0;
#pragma unroll
        for (int t = 0; t < 2; ++t)
          wbuf[i][t] = tile_words<BPS, direct_words<BPS, 1>()>(rowtiles + size_t(2 * pidx[i] + t) * TW, r, true);
      }
#pragma unroll
      for (int i = 0; i < PF; ++i) {
        int part = 0;
#pragma unroll
        for (int t = 0; t < 2; ++t) {
          int w[4];
          row_words<BPS, TB, direct_words<BPS, 1>()>(wbuf[i][t], r, lane_base, table, w);
          const int4 xv = __ldg(reinterpret_cast<const int4*>(xq) + (size_t(pidx[i]) * nb + b) * 2 + t);
          part = __dp4a(w[0], xv.x, part);
          part = __dp4a(w[1], xv.y, part);
          part = __dp4a(w[2], xv.z, part);
          part = __dp4a(w[3], xv.w, part);
        }
        if (valid[i]) acc += float(part) * __ldg(xs + size_t(pidx[i]) * nb + b);
      }
    }
    const float total = acc + __shfl_xor_sync(0xffffffffu, acc, 16);
    if (half == 0) y[size_t(row) * m + rb * 16 + r] = total * __ldg(wscale + e);
  }
}

template <int BPS, int PF>
int launch_moe_gemv(const uint32_t* words, const long long* woff, const int* fmt, const float* wscale, const int* ids,
                    int rows, int act_div, const int* table, const int* xq, const float* xs, int nb, float* y, int m,
                    int n, cudaStream_t stream) {
  constexpr int WARPS = ITQ_GEMV_WARPS, TB = fmt_table_bits(BPS);
  const size_t smem = sizeof(int) << TB;
  auto kern = itq_moe_gemv_kernel<BPS, TB, WARPS, PF>;
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
    configured = true;
  }
  const int items = rows * (m >> 4);
  const int blocks = std::max(1, std::min(sm_count() * 4, (items + WARPS - 1) / WARPS));
  kern<<<blocks, WARPS * 32, smem, stream>>>(words, woff, fmt, wscale, ids, rows, act_div, table, xq, xs, nb, y, m, n);
  return int(cudaGetLastError());
}

}  // namespace

// y [rows, m] float32 = W_{ids[r]} (int8 activations of row r / act_div); launch once per format code bps present.
extern "C" int itq_moe_gemv(const void* words, const void* woff, const void* fmt, const void* wscale, const void* ids,
                            int rows, int act_div, const void* table, const void* xq, const void* xs, int nb, void* y,
                            int m, int n, int bps, int prefetch, void* stream_pointer) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  if (m % 16 || n % 32 || rows < 1 || act_div < 1 || nb < 1) return int(cudaErrorInvalidValue);
  auto w = reinterpret_cast<const uint32_t*>(words);
  auto wo = reinterpret_cast<const long long*>(woff);
  auto f = reinterpret_cast<const int*>(fmt);
  auto sc = reinterpret_cast<const float*>(wscale);
  auto id = reinterpret_cast<const int*>(ids);
  auto t = reinterpret_cast<const int*>(table);
  auto q = reinterpret_cast<const int*>(xq);
  auto s = reinterpret_cast<const float*>(xs);
  auto o = reinterpret_cast<float*>(y);
#define G(B, P) if (bps == B && prefetch == P) \
    return launch_moe_gemv<B, P>(w, wo, f, sc, id, rows, act_div, t, q, s, nb, o, m, n, stream);
#define GP(B) G(B, 1) G(B, 2) G(B, 4)
  GP(4) GP(5) GP(6) GP(7) GP(8) GP(9) GP(10) GP(11) GP(12) GP(35) GP(36) GP(37) GP(38) GP(39) GP(40)
#undef GP
#undef G
  return int(cudaErrorInvalidValue);
}

// ---- one launch for every format of an itq-moe tensor -------------------------------------------------------------
// The per-format kernel above runs once per format code present (5-10 per tensor in the Inkling files), and every
// launch's blocks load a table and skip the rows of other formats.  Here each warp's work item (routed row, 16-row
// tile) reads its expert's format and runs that format's inner loop (warp-uniform switch); both tables (V4: 4,096
// entries, V2: 1,024) sit in shared memory once per block.  Same arithmetic, same output.
namespace {

template <int BPS, int TB, int PF>
__device__ __forceinline__ float moe_gemv_item(const uint32_t* __restrict__ ebase, int rb, int ct,
                                               const int* __restrict__ table, const int* __restrict__ xq,
                                               const float* __restrict__ xs, int nb, int b, int lane) {
  constexpr int TW = fmt_words(BPS);
  const int r = lane & 15, half = lane >> 4, lane_base = lane & 16;
  const int pairs = ct >> 1;
  const uint32_t* rowtiles = ebase + size_t(rb) * ct * TW;
  float acc = 0.f;
  for (int q = 0; q < pairs; q += 2 * PF) {
    TileWords<BPS> wbuf[PF][2];
    int pidx[PF];
    bool valid[PF];
#pragma unroll
    for (int i = 0; i < PF; ++i) {
      const int p = q + half + 2 * i;
      valid[i] = p < pairs;
      pidx[i] = valid[i] ? p : 0;
#pragma unroll
      for (int t = 0; t < 2; ++t)
        wbuf[i][t] = tile_words<BPS, direct_words<BPS, 1>()>(rowtiles + size_t(2 * pidx[i] + t) * TW, r, true);
    }
#pragma unroll
    for (int i = 0; i < PF; ++i) {
      int part = 0;
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        int w[4];
        row_words<BPS, TB, direct_words<BPS, 1>()>(wbuf[i][t], r, lane_base, table, w);
        const int4 xv = __ldg(reinterpret_cast<const int4*>(xq) + (size_t(pidx[i]) * nb + b) * 2 + t);
        part = __dp4a(w[0], xv.x, part);
        part = __dp4a(w[1], xv.y, part);
        part = __dp4a(w[2], xv.z, part);
        part = __dp4a(w[3], xv.w, part);
      }
      if (valid[i]) acc += float(part) * __ldg(xs + size_t(pidx[i]) * nb + b);
    }
  }
  return acc;
}

template <int WARPS, int PF>
__global__ void __launch_bounds__(WARPS * 32)
itq_moe_gemv_all_kernel(const uint32_t* __restrict__ words, const long long* __restrict__ woff,
                        const int* __restrict__ fmt, const float* __restrict__ wscale, const int* __restrict__ ids,
                        int rows, int act_div, const int* __restrict__ table4_g, const int* __restrict__ table2_g,
                        const int* __restrict__ xq, const float* __restrict__ xs, int nb, float* __restrict__ y,
                        int m, int n) {
  extern __shared__ int tables[];
  int* t4 = tables;                                       // 1 << 12 entries when the tensor has V4 experts
  int* t2 = tables + (table4_g ? 4096 : 0);               // 1 << 10 entries when it has V2 experts
  if (table4_g)
    for (int i = threadIdx.x; i < 4096; i += WARPS * 32) t4[i] = table4_g[i];
  if (table2_g)
    for (int i = threadIdx.x; i < 1024; i += WARPS * 32) t2[i] = table2_g[i];
  __syncthreads();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int rt = m >> 4, ct = n >> 4;
  const int items = rows * rt;
  for (int item = blockIdx.x * WARPS + warp; item < items; item += gridDim.x * WARPS) {
    const int row = item / rt, rb = item - row * rt;
    const int e = __ldg(ids + row);
    if (e < 0) continue;
    const int b = row / act_div;
    const uint32_t* ebase = words + __ldg(woff + e);
    float acc = 0.f;
    switch (__ldg(fmt + e)) {                             // uniform over the warp: one item per warp
#define C4(B) case B: acc = moe_gemv_item<B, 12, PF>(ebase, rb, ct, t4, xq, xs, nb, b, lane); break;
#define C2(B) case B: acc = moe_gemv_item<B, 10, PF>(ebase, rb, ct, t2, xq, xs, nb, b, lane); break;
      C4(4) C4(5) C4(6) C4(7) C4(8) C4(9) C4(10) C4(11) C4(12)
      C2(35) C2(36) C2(37) C2(38) C2(39) C2(40)
#undef C4
#undef C2
      default: break;
    }
    const float total = acc + __shfl_xor_sync(0xffffffffu, acc, 16);
    if ((lane >> 4) == 0) y[size_t(row) * m + rb * 16 + (lane & 15)] = total * __ldg(wscale + e);
  }
}

}  // namespace

// y [rows, m] float32 = W_{ids[r]} (int8 activations of row r / act_div), every format in one launch; table4 / table2:
// the tensor's V4 / V2 tables (nullptr when it has no expert of that kind).
extern "C" int itq_moe_gemv_all(const void* words, const void* woff, const void* fmt, const void* wscale,
                                const void* ids, int rows, int act_div, const void* table4, const void* table2,
                                const void* xq, const void* xs, int nb, void* y, int m, int n, int prefetch,
                                void* stream_pointer) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  if (m % 16 || n % 32 || rows < 1 || act_div < 1 || nb < 1 || (!table4 && !table2)) return int(cudaErrorInvalidValue);
  constexpr int WARPS = ITQ_GEMV_WARPS;
  const size_t smem = sizeof(int) * ((table4 ? 4096 : 0) + (table2 ? 1024 : 0));
  const int items = rows * (m >> 4);
  const int blocks = std::max(1, std::min(sm_count() * 4, (items + WARPS - 1) / WARPS));
#define L(P) if (prefetch == P) { \
    itq_moe_gemv_all_kernel<WARPS, P><<<blocks, WARPS * 32, smem, stream>>>( \
        reinterpret_cast<const uint32_t*>(words), reinterpret_cast<const long long*>(woff), \
        reinterpret_cast<const int*>(fmt), reinterpret_cast<const float*>(wscale), reinterpret_cast<const int*>(ids), \
        rows, act_div, reinterpret_cast<const int*>(table4), reinterpret_cast<const int*>(table2), \
        reinterpret_cast<const int*>(xq), reinterpret_cast<const float*>(xs), nb, reinterpret_cast<float*>(y), m, n); \
    return int(cudaGetLastError()); }
  L(1) L(2) L(4)
#undef L
  return int(cudaErrorInvalidValue);
}

// itq_prep with an explicit pair-major row stride nb (>= rows, any count): routed-expert activations.
extern "C" int itq_prep_rows(const void* x, int x_is_bf16, const void* pis, const void* su, void* xq, void* xs,
                             int rows, int n, int x_stride, int block, int nb, void* stream_pointer) {
  auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  if (n % block || rows < 1 || nb < rows) return int(cudaErrorInvalidValue);
  auto p = reinterpret_cast<const float*>(pis);
  auto sgn = reinterpret_cast<const int8_t*>(su);
  auto q = reinterpret_cast<int*>(xq);
  auto s = reinterpret_cast<float*>(xs);
  const dim3 grid(n / block, rows);
#define P(RB) if (block == RB) { \
    const int threads = RB >= 256 ? 256 : (RB >= 32 ? RB : 32); \
    if (x_is_bf16) return launch_k(itq_prep_kernel<RB, __nv_bfloat16, true>, grid, dim3(threads), 0, stream, \
        reinterpret_cast<const __nv_bfloat16*>(x), p, sgn, q, s, (__nv_bfloat16*) nullptr, n, x_stride, (float*) nullptr, 0, nb); \
    return launch_k(itq_prep_kernel<RB, float, true>, grid, dim3(threads), 0, stream, reinterpret_cast<const float*>(x), \
        p, sgn, q, s, (__nv_bfloat16*) nullptr, n, x_stride, (float*) nullptr, 0, nb); }
  P(32) P(64) P(128) P(256) P(512) P(1024) P(2048) P(4096)
#undef P
  return int(cudaErrorInvalidValue);
}
