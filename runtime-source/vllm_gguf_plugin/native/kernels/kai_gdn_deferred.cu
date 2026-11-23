// SPDX-License-Identifier: Apache-2.0
// kaiLLM: deferred-commit Gated DeltaNet update for speculative verification (batch 1) with the gated RMSNorm fused.
//
// Same algorithm and bookkeeping as gdn_deferred.py's Triton kernel (side buffers double-buffered by parity, one book
// entry per program, a global epoch for invalidation, single-slot in-place mode), written for latency: one CTA per
// value head, 512 threads, a 4-lane group per value row holding 32 state columns in registers, so each token's two
// dot products over K = 128 need two shuffles.  The CTA owns the whole head, so the gated RMSNorm over V runs in the
// same launch.  K = V = 128 (Qwen3.5 / Qwen3.8 GDN), up to 16 tokens.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int K = 128;
constexpr int V = 128;
constexpr int THREADS = 512;     // 16 warps x 8 value rows; each lane holds 4 key columns of each of its warp's rows
constexpr int ROWS = 8;          // value rows per warp
constexpr int MAXT = 16;

__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
  return x;
}

// 4 consecutive BF16 (8 bytes) -> floats
__device__ __forceinline__ void load4(const __nv_bfloat16* __restrict__ p, float (&d)[4]) {
  const uint2 w = *reinterpret_cast<const uint2*>(p);
  const __nv_bfloat162* e = reinterpret_cast<const __nv_bfloat162*>(&w);
  const float2 a = __bfloat1622float2(e[0]), b = __bfloat1622float2(e[1]);
  d[0] = a.x; d[1] = a.y; d[2] = b.x; d[3] = b.y;
}

__device__ __forceinline__ void store4(__nv_bfloat16* __restrict__ p, const float (&s)[4]) {
  uint2 w;
  __nv_bfloat162* e = reinterpret_cast<__nv_bfloat162*>(&w);
  e[0] = __floats2bfloat162_rn(s[0], s[1]);
  e[1] = __floats2bfloat162_rn(s[2], s[3]);
  *reinterpret_cast<uint2*>(p) = w;
}

__device__ __forceinline__ void load4(const float* __restrict__ p, float (&d)[4]) {
  const float4 w = *reinterpret_cast<const float4*>(p);
  d[0] = w.x; d[1] = w.y; d[2] = w.z; d[3] = w.w;
}

__device__ __forceinline__ void store4(float* __restrict__ p, const float (&s)[4]) {
  *reinterpret_cast<float4*>(p) = make_float4(s[0], s[1], s[2], s[3]);
}

__device__ __forceinline__ float to_state(float x, const __nv_bfloat16*) { return bf(__float2bfloat16(x)); }
__device__ __forceinline__ float to_state(float x, const float*) { return x; }

// Sum of 8 per-row partials over the 32 lanes (reduce-scatter butterfly, 9 shuffles).  Lane l ends with the full
// sum of row ri = 4*bit4 + 2*bit3 + bit2 of l (replicated over bits 1 and 0).
__device__ __forceinline__ float rows_sum(float (&p)[ROWS], int lane) {
  const bool b4 = lane & 16, b3 = lane & 8, b2 = lane & 4;
  float q4[4], q2[2];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float send = b4 ? p[k] : p[k + 4];
    const float keep = b4 ? p[k + 4] : p[k];
    q4[k] = keep + __shfl_xor_sync(0xffffffffu, send, 16);
  }
#pragma unroll
  for (int k = 0; k < 2; ++k) {
    const float send = b3 ? q4[k] : q4[k + 2];
    const float keep = b3 ? q4[k + 2] : q4[k];
    q2[k] = keep + __shfl_xor_sync(0xffffffffu, send, 8);
  }
  const float send = b2 ? q2[0] : q2[1];
  float r = (b2 ? q2[1] : q2[0]) + __shfl_xor_sync(0xffffffffu, send, 4);
  r += __shfl_xor_sync(0xffffffffu, r, 2);
  r += __shfl_xor_sync(0xffffffffu, r, 1);
  return r;
}

__device__ __forceinline__ int row_of_lane(int lane) { return ((lane >> 4) & 1) * 4 + ((lane >> 3) & 1) * 2 + ((lane >> 2) & 1); }
__device__ __forceinline__ int lane_of_row(int i) { return ((i >> 2) << 4) | (((i >> 1) & 1) << 3) | ((i & 1) << 2); }

__device__ __forceinline__ float decay_of(float a_raw, float bias, float neg_a) {
  const float x = a_raw + bias;
  return expf(neg_a * (x <= 20.f ? logf(1.f + expf(x)) : x));
}

__device__ __forceinline__ float sigmoid(float x) { return 1.f / (1.f + expf(-x)); }

__device__ __forceinline__ void l2_normalize(float (&x)[4], float mul) {
  const float ss = warp_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]);
  const float r = rsqrtf(ss + 1e-6f) * mul;
#pragma unroll
  for (int i = 0; i < 4; ++i) x[i] *= r;
}

// One token on this warp's 8 rows: h = decay h; u = (v - h.k) beta; h += u k.  v_row = v of row row_of_lane(lane).
__device__ __forceinline__ void delta_step(float (&h)[ROWS][4], const float (&k)[4], float v_row, float decay,
                                           float beta, int lane) {
  float p[ROWS];
#pragma unroll
  for (int i = 0; i < ROWS; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) h[i][j] *= decay;
    p[i] = h[i][0] * k[0] + h[i][1] * k[1] + h[i][2] * k[2] + h[i][3] * k[3];
  }
  const float u_mine = (v_row - rows_sum(p, lane)) * beta;
#pragma unroll
  for (int i = 0; i < ROWS; ++i) {
    const float u = __shfl_sync(0xffffffffu, u_mine, lane_of_row(i));
#pragma unroll
    for (int j = 0; j < 4; ++j) h[i][j] += u * k[j];
  }
}

// Two row sums at once (h.k and h.q): 16 partials, 9 shuffle levels with two independent chains.
__device__ __forceinline__ void rows_sum2(float (&p)[ROWS], float (&q)[ROWS], int lane, float& rp, float& rq) {
  rp = rows_sum(p, lane);
  rq = rows_sum(q, lane);
}

template <bool SINGLE_SLOT, typename S>
__global__ void __launch_bounds__(THREADS, 1) gdn_deferred_kernel(
    const __nv_bfloat16* __restrict__ qkv, long stride_qkv,
    const __nv_bfloat16* __restrict__ a, long stride_a,
    const __nv_bfloat16* __restrict__ b, long stride_b,
    const float* __restrict__ a_log, const float* __restrict__ dt_bias,
    S* __restrict__ state, long stride_slot,
    const int* __restrict__ state_indices, int num_indices,
    const int* __restrict__ cu_seqlens, const int* __restrict__ num_accepted,
    const __nv_bfloat16* __restrict__ z, long stride_z, const float* __restrict__ norm_w, float eps, int sigmoid_gate,
    __nv_bfloat16* __restrict__ out,
    __nv_bfloat16* __restrict__ side_k, __nv_bfloat16* __restrict__ side_v,
    __nv_bfloat16* __restrict__ side_a, __nv_bfloat16* __restrict__ side_b,
    int* __restrict__ book, const int* __restrict__ epoch_ptr,
    int H, int HV, int maxt, float scale) {
  // Inputs of every token staged once (the sequential token loop then only reads shared memory):
  // current tokens: normalized q / k (fp32), v (fp32); replayed tokens: normalized k, v.  o_s aliases the replay keys.
  __shared__ __align__(16) float qs[MAXT][K];
  __shared__ __align__(16) float ks[MAXT][K];
  __shared__ __align__(16) float rk[MAXT][K];
  __shared__ float vs[MAXT][V];
  __shared__ float rv[MAXT][V];
  __shared__ float dec[MAXT], bet[MAXT], kq[MAXT], rdec[MAXT], rbet[MAXT];
  float (*o_s)[V] = rk;   // replay keys are dead once the replay is done

  const int hv = blockIdx.x;
  const int G = HV / H;
  const int h = hv / G;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int r0 = warp * ROWS;
  const int my_row = r0 + row_of_lane(lane);
  const int k0 = lane * 4;

  const long bos = cu_seqlens[0];
  const int T = cu_seqlens[1] - static_cast<int>(bos);
  if (T <= 0) return;
  const int accepted = num_accepted[0];
  const int cur_epoch = *epoch_ptr;
  int* bk = book + hv * 4;
  const long base = bk[0];
  const int n_prev = bk[1];
  bool valid = (bk[2] == cur_epoch) && (n_prev > 0) && (base > 0);
  const int par = valid ? bk[3] : 0;
  long dst;
  if (SINGLE_SLOT) {
    dst = state_indices[0];
    valid = valid && (base == dst);
  } else {
    dst = (accepted > 0 && accepted <= num_indices) ? state_indices[accepted - 1] : 0;
  }
  if (dst <= 0) {  // padding / dummy run: zero outputs, invalidate this program's side buffers
    for (int i = tid; i < T * V; i += THREADS) out[((bos + i / V) * HV + hv) * V + (i % V)] = __float2bfloat16(0.f);
    __syncthreads();
    if (tid == 0) bk[1] = 0;
    return;
  }
  const int n_rec = valid ? min(accepted, n_prev) : 0;
  const float neg_a = -expf(a_log[hv]);
  const float bias = dt_bias[hv];
  const int w_par = 1 - par;

  // ---- stage: one warp per token (current: warps 0..T-1 own token warp; replayed: token warp too) ----------------
  for (int t = warp; t < T; t += THREADS / 32) {
    const long r = bos + t;
    const __nv_bfloat16* p = qkv + r * stride_qkv;
    const long woff = static_cast<long>(w_par) * maxt + t;
    float qv[4], kv[4];
    load4(p + h * K + k0, qv);
    load4(p + (H + h) * K + k0, kv);
    if ((hv % G) == 0)   // one CTA per k-head saves the raw key for the next step's replay
      *reinterpret_cast<uint2*>(side_k + woff * (H * K) + h * K + k0) =
          *reinterpret_cast<const uint2*>(p + (H + h) * K + k0);
    const float rq = rsqrtf(warp_sum(qv[0] * qv[0] + qv[1] * qv[1] + qv[2] * qv[2] + qv[3] * qv[3]) + 1e-6f) * scale;
    const float rk_ = rsqrtf(warp_sum(kv[0] * kv[0] + kv[1] * kv[1] + kv[2] * kv[2] + kv[3] * kv[3]) + 1e-6f);
    float dot = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      qv[j] *= rq;
      kv[j] *= rk_;
      dot += qv[j] * kv[j];
    }
    *reinterpret_cast<float4*>(&qs[t][k0]) = make_float4(qv[0], qv[1], qv[2], qv[3]);
    *reinterpret_cast<float4*>(&ks[t][k0]) = make_float4(kv[0], kv[1], kv[2], kv[3]);
    dot = warp_sum(dot);
    // the value slice of this head: 128 values, 4 per lane
    const __nv_bfloat16* pv = p + 2 * H * K + hv * V;
    float vv[4];
    load4(pv + k0, vv);
#pragma unroll
    for (int j = 0; j < 4; ++j) vs[t][k0 + j] = vv[j];
    *reinterpret_cast<uint2*>(side_v + woff * (HV * V) + hv * V + k0) = *reinterpret_cast<const uint2*>(pv + k0);
    if (lane == 0) {
      const __nv_bfloat16 a_raw = a[r * stride_a + hv], b_raw = b[r * stride_b + hv];
      side_a[woff * HV + hv] = a_raw;
      side_b[woff * HV + hv] = b_raw;
      dec[t] = decay_of(bf(a_raw), bias, neg_a);
      bet[t] = sigmoid(bf(b_raw));
      kq[t] = dot;
    }
  }
  for (int t = warp; t < n_rec; t += THREADS / 32) {
    const long off = static_cast<long>(par) * maxt + t;
    float kv[4];
    load4(side_k + off * (H * K) + h * K + k0, kv);
    const float rk_ = rsqrtf(warp_sum(kv[0] * kv[0] + kv[1] * kv[1] + kv[2] * kv[2] + kv[3] * kv[3]) + 1e-6f);
    *reinterpret_cast<float4*>(&rk[t][k0]) = make_float4(kv[0] * rk_, kv[1] * rk_, kv[2] * rk_, kv[3] * rk_);
    float vv[4];
    load4(side_v + off * (HV * V) + hv * V + k0, vv);
#pragma unroll
    for (int j = 0; j < 4; ++j) rv[t][k0 + j] = vv[j];
    if (lane == 0) {
      rdec[t] = decay_of(bf(side_a[off * HV + hv]), bias, neg_a);
      rbet[t] = sigmoid(bf(side_b[off * HV + hv]));
    }
  }

  float hs[ROWS][4];
  {
    const S* src = state + (valid ? base : dst) * stride_slot + (static_cast<long>(hv) * V + r0) * K + k0;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) load4(src + i * K, hs[i]);
  }
  __syncthreads();

  // ---- replay of the previous step's accepted tokens ------------------------------------------------------------
  for (int t = 0; t < n_rec; ++t) {
    const float4 k4 = *reinterpret_cast<const float4*>(&rk[t][k0]);
    const float kk[4] = {k4.x, k4.y, k4.z, k4.w};
    const float d = rdec[t];
    float p[ROWS];
#pragma unroll
    for (int i = 0; i < ROWS; ++i) p[i] = hs[i][0] * kk[0] + hs[i][1] * kk[1] + hs[i][2] * kk[2] + hs[i][3] * kk[3];
    const float u_mine = (rv[t][my_row] - d * rows_sum(p, lane)) * rbet[t];
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
      const float u = __shfl_sync(0xffffffffu, u_mine, lane_of_row(i));
#pragma unroll
      for (int j = 0; j < 4; ++j) hs[i][j] = hs[i][j] * d + u * kk[j];
    }
  }
  if (valid) {
    S* dptr = state + dst * stride_slot + (static_cast<long>(hv) * V + r0) * K + k0;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
      for (int j = 0; j < 4; ++j) hs[i][j] = to_state(hs[i][j], dptr);   // the reference stores it and reloads it
      store4(dptr + i * K, hs[i]);
    }
  }
  __syncthreads();   // o_s aliases the replay keys

  // ---- current tokens ------------------------------------------------------------------------------------------
  for (int t = 0; t < T; ++t) {
    const float4 k4 = *reinterpret_cast<const float4*>(&ks[t][k0]);
    const float4 q4 = *reinterpret_cast<const float4*>(&qs[t][k0]);
    const float kk[4] = {k4.x, k4.y, k4.z, k4.w};
    const float qq[4] = {q4.x, q4.y, q4.z, q4.w};
    const float d = dec[t];
    float pk[ROWS], pq[ROWS];
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
      pk[i] = hs[i][0] * kk[0] + hs[i][1] * kk[1] + hs[i][2] * kk[2] + hs[i][3] * kk[3];
      pq[i] = hs[i][0] * qq[0] + hs[i][1] * qq[1] + hs[i][2] * qq[2] + hs[i][3] * qq[3];
    }
    float sk, sq;
    rows_sum2(pk, pq, lane, sk, sq);
    const float u_mine = (vs[t][my_row] - d * sk) * bet[t];
    const float o = d * sq + u_mine * kq[t];       // (d h + u k^T) . q
    if ((lane & 3) == 0) o_s[t][my_row] = bf(__float2bfloat16(o));   // rounded like the unfused path
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
      const float u = __shfl_sync(0xffffffffu, u_mine, lane_of_row(i));
#pragma unroll
      for (int j = 0; j < 4; ++j) hs[i][j] = hs[i][j] * d + u * kk[j];
    }
  }
  __syncthreads();

  if (warp < T) {   // gated RMSNorm over V: one warp per token
    const int t = warp;
    const long r = bos + t;
    float ov[V / 32];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < V / 32; ++j) {
      ov[j] = o_s[t][lane + 32 * j];
      ss += ov[j] * ov[j];
    }
    const float rstd = rsqrtf(warp_sum(ss) / V + eps);
#pragma unroll
    for (int j = 0; j < V / 32; ++j) {
      const int v = lane + 32 * j;
      const float zf = bf(z[r * stride_z + hv * V + v]);
      const float gate = sigmoid_gate ? sigmoid(zf) : zf * sigmoid(zf);
      out[(r * HV + hv) * V + v] = __float2bfloat16(ov[j] * rstd * norm_w[v] * gate);
    }
  }
  __syncthreads();
  if (tid == 0) {
    bk[0] = static_cast<int>(dst);
    bk[1] = T;
    bk[2] = cur_epoch;
    bk[3] = w_par;
  }
}

}  // namespace

extern "C" int kai_gdn_deferred(
    const void* qkv, long stride_qkv, const void* a, long stride_a, const void* b, long stride_b,
    const void* a_log, const void* dt_bias, void* state, long stride_slot, const void* state_indices, int num_indices,
    const void* cu_seqlens, const void* num_accepted, const void* z, long stride_z, const void* norm_w, float eps,
    int sigmoid_gate, void* out, void* side_k, void* side_v, void* side_a, void* side_b, void* book,
    const void* epoch, int H, int HV, int maxt, float scale, int single_slot, int state_fp32, void* stream) {
  if (maxt > MAXT || HV % H) return static_cast<int>(cudaErrorInvalidValue);
  auto s = static_cast<cudaStream_t>(stream);
#define KAI_GDN_ARGS                                                                                           \
  static_cast<const __nv_bfloat16*>(qkv), stride_qkv, static_cast<const __nv_bfloat16*>(a), stride_a,           \
      static_cast<const __nv_bfloat16*>(b), stride_b, static_cast<const float*>(a_log),                         \
      static_cast<const float*>(dt_bias), static_cast<STATE_T*>(state), stride_slot,                            \
      static_cast<const int*>(state_indices), num_indices, static_cast<const int*>(cu_seqlens),                 \
      static_cast<const int*>(num_accepted), static_cast<const __nv_bfloat16*>(z), stride_z,                     \
      static_cast<const float*>(norm_w), eps, sigmoid_gate, static_cast<__nv_bfloat16*>(out),                    \
      static_cast<__nv_bfloat16*>(side_k), static_cast<__nv_bfloat16*>(side_v),                                 \
      static_cast<__nv_bfloat16*>(side_a), static_cast<__nv_bfloat16*>(side_b), static_cast<int*>(book),         \
      static_cast<const int*>(epoch), H, HV, maxt, scale
  if (state_fp32) {
#define STATE_T float
    if (single_slot)
      gdn_deferred_kernel<true, float><<<HV, THREADS, 0, s>>>(KAI_GDN_ARGS);
    else
      gdn_deferred_kernel<false, float><<<HV, THREADS, 0, s>>>(KAI_GDN_ARGS);
#undef STATE_T
  } else {
#define STATE_T __nv_bfloat16
    if (single_slot)
      gdn_deferred_kernel<true, __nv_bfloat16><<<HV, THREADS, 0, s>>>(KAI_GDN_ARGS);
    else
      gdn_deferred_kernel<false, __nv_bfloat16><<<HV, THREADS, 0, s>>>(KAI_GDN_ARGS);
#undef STATE_T
  }
#undef KAI_GDN_ARGS
  return static_cast<int>(cudaGetLastError());
}
