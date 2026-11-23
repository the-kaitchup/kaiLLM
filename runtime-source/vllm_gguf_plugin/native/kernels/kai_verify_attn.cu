// SPDX-License-Identifier: Apache-2.0
// kaiLLM: split-KV attention for speculative verification (batch 1, 2-16 query tokens, GQA, head 256, BF16 paged KV).
//
// vLLM's Triton unified attention covers a 16-token verification of a 6-way GQA group (96 query rows per KV head) with
// two 64-row query blocks, so every KV tile is fetched twice (best Triton config at 100k context: 497 us per layer on
// an RTX PRO 6000, 824 GB/s for one KV pass).  Here one CTA covers ALL query rows of a KV head for a slice of the
// context: 12 warps, two per query head (each owns half of the 256 output columns; both compute the same scores), all
// 16 query tokens of the head as the MMA's 16 rows.  K / V tiles of 32 tokens stream through a 3-stage cp.async
// pipeline into XOR-swizzled shared memory; Q stays in registers; online softmax in exp2.  A second kernel merges the
// context slices.  Causal masking over the verification tokens; padding rows (tokens >= T) are masked.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int D = 256;           // head size
constexpr int TN = 32;           // KV tokens per tile
constexpr int QPK_MAX = 8;       // query heads per KV head (warps = 2 * QPK)
constexpr int STAGES = 2;      // 2 x 32 KB of K / V + the S exchange (24 KB at 96 rows) stays under 99 KB
constexpr int ROW_BYTES = D * 2; // 512 bytes per K / V row in shared memory
constexpr int TILE_BYTES = TN * ROW_BYTES;   // 16 KB

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
  const int n = valid ? 16 : 0;      // src-size 0 -> zero fill
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                                  uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

// D(16x8, f32) += A(16x16, bf16, row) * B(16x8, bf16, col)
__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

// byte offset of (row, 16-byte chunk) in a swizzled 512-byte-row tile
__device__ __forceinline__ uint32_t swz(int row, int chunk) {
  return static_cast<uint32_t>(row * ROW_BYTES + ((chunk ^ (row & 7)) << 4));
}

struct Params {
  const __nv_bfloat16* q;        // [T, Hq, D]
  const __nv_bfloat16* kcache;   // paged: block stride, token stride, head stride (elements)
  const __nv_bfloat16* vcache;
  long k_sb, k_st, k_sh, v_sb, v_st, v_sh;
  const int* block_table;        // [max_blocks] of the request
  int block_size;
  const int* seq_len;            // device scalar: tokens in the context incl. the T new ones
  const int* cu_q;               // device query offsets [0, T]: the actual query length (T below is the bound)
  int T, Hq, Hk, num_segments;
  long q_st, q_sh;               // query strides (token, head) in elements
  float scale_log2;              // softmax scale * log2(e)
  float* part_o;                 // [T, Hq, S, D]  unnormalized
  float* part_m;                 // [T, Hq, S]     running max (log2 units)
  float* part_l;                 // [T, Hq, S]     running sum
};

// Rows = (query token, query head of the KV group) pairs, R = T * QPK <= 128, in NG = ceil(R / 16) groups of 16; a
// pair of warps per group: both compute half of S = Q K^T (dims 0-127 / 128-255), add the halves through shared
// memory, run the same online softmax, and each accumulates half of O.
__global__ void __launch_bounds__(512, 1) verify_attn_kernel(Params p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int g = blockIdx.y;                       // KV head
  const int s = blockIdx.x;                       // context slice
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int grp = warp >> 1, dhalf = warp & 1;
  const int nthreads = blockDim.x;
  const int QPK = p.Hq / p.Hk;
  const int Tq = min(p.cu_q[1] - p.cu_q[0], p.T);  // actual query tokens (rows beyond them are padding)
  const int R = Tq * QPK;
  const int L = *p.seq_len;
  const int t0 = L - Tq;                          // absolute position of query token 0
  const int seg_tiles = ((L + TN - 1) / TN + p.num_segments - 1) / p.num_segments;
  const int key_begin = s * seg_tiles * TN;
  const int key_end = min(L, key_begin + seg_tiles * TN);
  // this lane's two rows (MMA rows lane/4 and lane/4 + 8 of the group)
  const int rA = grp * 16 + (lane >> 2), rB = rA + 8;
  const int tA = rA / QPK, hA = g * QPK + rA % QPK;
  const int tB = rB / QPK, hB = g * QPK + rB % QPK;
  const bool vA = rA < R, vB = rB < R;
  if (key_begin >= key_end) {                     // empty slice: neutral partials
    if (dhalf == 0 && (lane & 3) == 0) {
      if (vA) { const long b = (static_cast<long>(tA) * p.Hq + hA) * p.num_segments + s; p.part_m[b] = -INFINITY; p.part_l[b] = 0.f; }
      if (vB) { const long b = (static_cast<long>(tB) * p.Hq + hB) * p.num_segments + s; p.part_m[b] = -INFINITY; p.part_l[b] = 0.f; }
    }
    return;
  }
  const int ntiles = (key_end - key_begin + TN - 1) / TN;

  // ---- Q fragments for this warp's half of the head dims: 8 k-steps of 16 ----
  uint32_t qa[8][4];
  {
    const int c = (lane & 3) * 2;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const int d0 = dhalf * 128 + ks * 16 + c;
      auto ld = [&](bool v, int t, int h, int d) -> uint32_t {
        if (!v) return 0u;
        return *reinterpret_cast<const uint32_t*>(p.q + t * p.q_st + h * p.q_sh + d);
      };
      qa[ks][0] = ld(vA, tA, hA, d0);
      qa[ks][1] = ld(vB, tB, hB, d0);
      qa[ks][2] = ld(vA, tA, hA, d0 + 8);
      qa[ks][3] = ld(vB, tB, hB, d0 + 8);
    }
  }

  float oacc[16][4];
#pragma unroll
  for (int i = 0; i < 16; ++i) oacc[i][0] = oacc[i][1] = oacc[i][2] = oacc[i][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

  uint8_t* ks_base = smem;                                   // STAGES K tiles
  uint8_t* vs_base = smem + STAGES * TILE_BYTES;             // STAGES V tiles
  float* xs = reinterpret_cast<float*>(smem + 2 * STAGES * TILE_BYTES);   // S halves: [group][warp-in-pair][16 x 32]

  const int nwarps = nthreads >> 5;
  auto load_tile = [&](int tile, int stage) {
    // a 32-token tile spans at most two KV-cache blocks (block_size >= 32): two lookups per tile, a warp per row,
    // a lane per 16-byte chunk
    const int kt0 = key_begin + tile * TN;
    const int b0 = kt0 / p.block_size;
    const int off0 = kt0 - b0 * p.block_size;
    const long blk0 = p.block_table[b0];
    const long blk1 = (off0 + TN > p.block_size) ? p.block_table[b0 + 1] : blk0;
    const uint32_t kdst = smem_u32(ks_base + stage * TILE_BYTES);
    const uint32_t vdst = smem_u32(vs_base + stage * TILE_BYTES);
    for (int row = warp; row < TN; row += nwarps) {
      const int key = kt0 + row;
      const bool valid = key < key_end;
      int off = off0 + row;
      long blk = blk0;
      if (off >= p.block_size) { off -= p.block_size; blk = blk1; }
      if (!valid) { off = off0; blk = blk0; }          // any mapped address; zero-filled
      const __nv_bfloat16* ksrc = p.kcache + blk * p.k_sb + off * p.k_st + g * p.k_sh + lane * 8;
      const __nv_bfloat16* vsrc = p.vcache + blk * p.v_sb + off * p.v_st + g * p.v_sh + lane * 8;
      cp_async16(kdst + swz(row, lane), ksrc, valid);
      cp_async16(vdst + swz(row, lane), vsrc, valid);
    }
  };

#pragma unroll
  for (int st = 0; st < STAGES - 1; ++st) {
    if (st < ntiles) load_tile(st, st);
    cp_async_commit();
  }

  float* my_x = xs + (grp * 2 + dhalf) * (16 * TN);
  const float* other_x = xs + (grp * 2 + (dhalf ^ 1)) * (16 * TN);
  const int pair_bar = 1 + grp;                    // named barrier per warp pair (0 = __syncthreads)

  for (int tile = 0; tile < ntiles; ++tile) {
    const int pre = tile + STAGES - 1;
    if (pre < ntiles) load_tile(pre, pre % STAGES);
    cp_async_commit();
    cp_async_wait<STAGES - 1>();
    __syncthreads();
    const int stage = tile % STAGES;
    const uint32_t kb = smem_u32(ks_base + stage * TILE_BYTES);
    const uint32_t vb = smem_u32(vs_base + stage * TILE_BYTES);

    // ---- half of S = Q K^T over this warp's 128 dims: 16 rows x 32 keys ----
    float sacc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) sacc[i][0] = sacc[i][1] = sacc[i][2] = sacc[i][3] = 0.f;
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
#pragma unroll
      for (int np = 0; np < 2; ++np) {
        const int mtx = lane >> 3, rr = lane & 7;
        const int key = np * 16 + (mtx >> 1) * 8 + rr;
        const int chunk = dhalf * 16 + ks * 2 + (mtx & 1);
        uint32_t b0, b1, b2, b3;
        ldmatrix_x4(kb + swz(key, chunk), b0, b1, b2, b3);
        mma_bf16(sacc[np * 2], qa[ks], b0, b1);
        mma_bf16(sacc[np * 2 + 1], qa[ks], b2, b3);
      }
    }
    // exchange halves with the pair warp (C-fragment order: [nt][e] per lane)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
      *reinterpret_cast<float4*>(my_x + (nt * 32 + lane) * 4) = make_float4(sacc[nt][0], sacc[nt][1], sacc[nt][2], sacc[nt][3]);
    asm volatile("bar.sync %0, 64;\n" ::"r"(pair_bar));
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const float4 o = *reinterpret_cast<const float4*>(other_x + (nt * 32 + lane) * 4);
      // same summation order in both warps (dims 0-127 first): identical scores and softmax in the pair
      if (dhalf == 0) {
        sacc[nt][0] += o.x; sacc[nt][1] += o.y; sacc[nt][2] += o.z; sacc[nt][3] += o.w;
      } else {
        sacc[nt][0] = o.x + sacc[nt][0]; sacc[nt][1] = o.y + sacc[nt][1];
        sacc[nt][2] = o.z + sacc[nt][2]; sacc[nt][3] = o.w + sacc[nt][3];
      }
    }

    // ---- mask, online softmax ----
    const int kt0 = key_begin + tile * TN;
    float tmax0 = -INFINITY, tmax1 = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        const int key = kt0 + nt * 8 + (lane & 3) * 2 + e;
        const bool ok0 = key < key_end && vA && key <= t0 + tA;
        const bool ok1 = key < key_end && vB && key <= t0 + tB;
        sacc[nt][e] = ok0 ? sacc[nt][e] * p.scale_log2 : -INFINITY;
        sacc[nt][2 + e] = ok1 ? sacc[nt][2 + e] * p.scale_log2 : -INFINITY;
        tmax0 = fmaxf(tmax0, sacc[nt][e]);
        tmax1 = fmaxf(tmax1, sacc[nt][2 + e]);
      }
    }
    tmax0 = fmaxf(tmax0, __shfl_xor_sync(0xffffffffu, tmax0, 1));
    tmax0 = fmaxf(tmax0, __shfl_xor_sync(0xffffffffu, tmax0, 2));
    tmax1 = fmaxf(tmax1, __shfl_xor_sync(0xffffffffu, tmax1, 1));
    tmax1 = fmaxf(tmax1, __shfl_xor_sync(0xffffffffu, tmax1, 2));
    const float mn0 = fmaxf(m0, tmax0), mn1 = fmaxf(m1, tmax1);
    const float a0 = (mn0 == -INFINITY) ? 1.f : exp2f(m0 - mn0);
    const float a1 = (mn1 == -INFINITY) ? 1.f : exp2f(m1 - mn1);
    float rs0 = 0.f, rs1 = 0.f;
    uint32_t pa[2][4];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const float p00 = (mn0 == -INFINITY) ? 0.f : exp2f(sacc[nt][0] - mn0);
      const float p01 = (mn0 == -INFINITY) ? 0.f : exp2f(sacc[nt][1] - mn0);
      const float p10 = (mn1 == -INFINITY) ? 0.f : exp2f(sacc[nt][2] - mn1);
      const float p11 = (mn1 == -INFINITY) ? 0.f : exp2f(sacc[nt][3] - mn1);
      rs0 += p00 + p01;
      rs1 += p10 + p11;
      const int kk = nt >> 1, hi = nt & 1;
      pa[kk][hi * 2 + 0] = pack_bf16(p00, p01);
      pa[kk][hi * 2 + 1] = pack_bf16(p10, p11);
    }
    rs0 += __shfl_xor_sync(0xffffffffu, rs0, 1);
    rs0 += __shfl_xor_sync(0xffffffffu, rs0, 2);
    rs1 += __shfl_xor_sync(0xffffffffu, rs1, 1);
    rs1 += __shfl_xor_sync(0xffffffffu, rs1, 2);
    l0 = l0 * a0 + rs0;
    l1 = l1 * a1 + rs1;
    m0 = mn0;
    m1 = mn1;
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
      oacc[nt][0] *= a0;
      oacc[nt][1] *= a0;
      oacc[nt][2] *= a1;
      oacc[nt][3] *= a1;
    }

    // ---- O += P V over this warp's 128 dims ----
#pragma unroll
    for (int ks = 0; ks < 2; ++ks) {
#pragma unroll
      for (int np = 0; np < 8; ++np) {
        const int mtx = lane >> 3, rr = lane & 7;
        const int key = ks * 16 + (mtx & 1) * 8 + rr;
        const int chunk = dhalf * 16 + np * 2 + (mtx >> 1);
        uint32_t b0, b1, b2, b3;
        ldmatrix_x4_trans(vb + swz(key, chunk), b0, b1, b2, b3);
        mma_bf16(oacc[np * 2], pa[ks], b0, b1);
        mma_bf16(oacc[np * 2 + 1], pa[ks], b2, b3);
      }
    }
    __syncthreads();
  }
  cp_async_wait<0>();

  // ---- partials ----
  const int c = (lane & 3) * 2;
  if (vA) {
    const long base = (static_cast<long>(tA) * p.Hq + hA) * p.num_segments + s;
    float* o = p.part_o + base * D + dhalf * 128;
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) *reinterpret_cast<float2*>(o + nt * 8 + c) = make_float2(oacc[nt][0], oacc[nt][1]);
    if (dhalf == 0 && (lane & 3) == 0) {
      p.part_m[base] = m0;
      p.part_l[base] = l0;
    }
  }
  if (vB) {
    const long base = (static_cast<long>(tB) * p.Hq + hB) * p.num_segments + s;
    float* o = p.part_o + base * D + dhalf * 128;
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) *reinterpret_cast<float2*>(o + nt * 8 + c) = make_float2(oacc[nt][2], oacc[nt][3]);
    if (dhalf == 0 && (lane & 3) == 0) {
      p.part_m[base] = m1;
      p.part_l[base] = l1;
    }
  }
}

// out[t, h, :] = sum_s O_s 2^(m_s - M) / sum_s l_s 2^(m_s - M)
__global__ void __launch_bounds__(D) verify_attn_reduce(const float* __restrict__ part_o, const float* __restrict__ part_m,
                                                       const float* __restrict__ part_l, __nv_bfloat16* __restrict__ out,
                                                       long out_st, long out_sh, int Hq, int S, const int* __restrict__ cu_q) {
  const int t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
  if (t >= cu_q[1] - cu_q[0]) {                   // padding row
    out[t * out_st + h * out_sh + d] = __float2bfloat16(0.f);
    return;
  }
  const long base = (static_cast<long>(t) * Hq + h) * S;
  float M = -INFINITY;
  for (int s = 0; s < S; ++s) M = fmaxf(M, part_m[base + s]);
  float L = 0.f, acc = 0.f;
  for (int s = 0; s < S; ++s) {
    const float m = part_m[base + s];
    if (m == -INFINITY) continue;
    const float w = exp2f(m - M);
    L += part_l[base + s] * w;
    acc += part_o[(base + s) * D + d] * w;
  }
  out[t * out_st + h * out_sh + d] = __float2bfloat16(L > 0.f ? acc / L : 0.f);
}

}  // namespace

extern "C" int kai_verify_attn(const void* q, long q_st, long q_sh, const void* kcache, long k_sb, long k_st, long k_sh,
                               const void* vcache, long v_sb, long v_st, long v_sh, const void* block_table,
                               int block_size, const void* seq_len, const void* cu_q, int T, int Hq, int Hk,
                               float scale, void* out, long out_st, long out_sh, void* part_o, void* part_m,
                               void* part_l, int num_segments, void* stream) {
  if (T < 1 || T > 16 || Hq % Hk) return static_cast<int>(cudaErrorInvalidValue);
  const int qpk = Hq / Hk;
  auto s = static_cast<cudaStream_t>(stream);
  Params p;
  p.q = static_cast<const __nv_bfloat16*>(q);
  p.kcache = static_cast<const __nv_bfloat16*>(kcache);
  p.vcache = static_cast<const __nv_bfloat16*>(vcache);
  p.k_sb = k_sb; p.k_st = k_st; p.k_sh = k_sh; p.v_sb = v_sb; p.v_st = v_st; p.v_sh = v_sh;
  p.block_table = static_cast<const int*>(block_table);
  p.block_size = block_size;
  p.seq_len = static_cast<const int*>(seq_len);
  p.cu_q = static_cast<const int*>(cu_q);
  p.T = T; p.Hq = Hq; p.Hk = Hk; p.num_segments = num_segments;
  p.q_st = q_st; p.q_sh = q_sh;
  p.scale_log2 = scale * 1.4426950408889634f;
  p.part_o = static_cast<float*>(part_o);
  p.part_m = static_cast<float*>(part_m);
  p.part_l = static_cast<float*>(part_l);
  const int groups = (T * qpk + 15) / 16;                     // 16-row MMA groups, a warp pair each
  if (groups > 8) return static_cast<int>(cudaErrorInvalidValue);
  const size_t smem = 2 * STAGES * TILE_BYTES + static_cast<size_t>(groups) * 2 * 16 * TN * sizeof(float);
  cudaError_t err = cudaFuncSetAttribute(verify_attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(smem));
  if (err) return static_cast<int>(err);
  verify_attn_kernel<<<dim3(num_segments, Hk), 64 * groups, smem, s>>>(p);
  verify_attn_reduce<<<dim3(T, Hq), D, 0, s>>>(p.part_o, p.part_m, p.part_l, static_cast<__nv_bfloat16*>(out), out_st,
                                               out_sh, Hq, num_segments, p.cu_q);
  return static_cast<int>(cudaGetLastError());
}
