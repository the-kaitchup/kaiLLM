// SPDX-License-Identifier: Apache-2.0
// Unselected packed W8/W6A16 experiment. Original GGUF bytes stay resident.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>
#include <climits>

namespace wmma = nvcuda::wmma;

__device__ __forceinline__ float read_half(const uint8_t* p) {
  return __half2float(__ushort_as_half(*reinterpret_cast<const uint16_t*>(p)));
}

template <int Q> __device__ __forceinline__ float decode(
    const uint8_t* w, int n, int k, int width) {
  if constexpr (Q == 8) {
    const auto* b = w + (size_t(n) * (width / 32) + k / 32) * 34;
    return __fmul_rn(float(reinterpret_cast<const int8_t*>(b + 2)[k % 32]),
                     read_half(b));
  } else {
    const auto* b = w + (size_t(n) * (width / 256) + k / 256) * 210;
    const int p = k % 256, half = p / 128, group = (p % 128) / 32;
    const int lane = p % 32;
    const int low = b[half * 64 + (group & 1) * 32 + lane];
    const int high = b[128 + half * 32 + lane];
    const int code = (((group < 2 ? low : low >> 4) & 15) |
                      (((high >> (group * 2)) & 3) << 4)) - 32;
    const int s = reinterpret_cast<const int8_t*>(b + 192)
        [half * 8 + group * 2 + lane / 16];
    return __fmul_rn(float(code), __fmul_rn(float(s), read_half(b + 208)));
  }
}


__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a,
                                        uint32_t b0, uint32_t b1) {
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
template <bool OutBF16> __device__ __forceinline__ void store(
    void* y, int sample, int col, int m, int n, float value) {
  if (sample < m && col < n) {
    const size_t offset = size_t(sample) * n + col;
    if constexpr (OutBF16)
      static_cast<__nv_bfloat16*>(y)[offset] = __float2bfloat16_rn(value);
    else static_cast<float*>(y)[offset] = value;
  }
}
template <int Q, bool OutBF16, bool Small>
__global__ __launch_bounds__(64) void canonical_register_mma(
    const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ packed,
    void* __restrict__ y, int m, int n, int k, int splits) {
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  const int out = (int(blockIdx.x) * 2 + warp) * 16;
  if (out >= n) return;  // Whole-warp rejection; this kernel has no barriers.
  const int sample0 = int(blockIdx.y) * 16 + lane / 4, sample1 = sample0 + 8;
  const int column = (lane % 4) * 2;
  float total0[4] = {}, total1[4] = {};
  if (splits > 1) y = static_cast<float*>(y) + size_t(blockIdx.z) * m * n;
  const int chunk = ((k / 256 + splits - 1) / splits) * 16;
  const int begin = int(blockIdx.z) * chunk, end = min(k / 16, begin + chunk);
  for (int base = begin; base < end; base += 16) {
    float c0[4] = {}, c1[4] = {};
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const int tk = base + i;
      if (tk >= end) break;
      const int col = tk * 16 + column;
      uint32_t a[4] = {};
      if (sample0 < m) {
        const auto* px = reinterpret_cast<const uint32_t*>(x + size_t(sample0)*k+col);
        a[0] = __ldg(px); a[2] = __ldg(px+4);
      }
      if constexpr (!Small) {
        if (sample1 < m) {
          const auto* px = reinterpret_cast<const uint32_t*>(x+size_t(sample1)*k+col);
          a[1] = __ldg(px); a[3] = __ldg(px+4);
        }
      }
      const size_t tile = size_t(out / 16) * (k / 16) + tk;
      const uint8_t* data = packed + tile * 256;
      const auto* codes = reinterpret_cast<const uint16_t*>(data + lane * 6);
      const uint64_t word = uint64_t(codes[0]) | (uint64_t(codes[1]) << 16)
                            | (uint64_t(codes[2]) << 32);
      const auto* scale = reinterpret_cast<const float*>(data + 192);
      const float d0 = scale[lane / 4], d1 = scale[lane / 4 + 8];
      uint32_t b[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int c0v = int((word >> (12*j)) & 63) - 32;
        const int c1v = int((word >> (12*j+6)) & 63) - 32;
        const float d = (j & 1) ? d1 : d0;
        const uint32_t lo = __bfloat16_as_ushort(
            __float2bfloat16_rn(__fmul_rn(float(c0v),d)));
        const uint32_t hi = __bfloat16_as_ushort(
            __float2bfloat16_rn(__fmul_rn(float(c1v),d)));
        b[j] = lo | (hi << 16);
      }
      const uint32_t b0=b[0], b1=b[1], b2=b[2], b3=b[3];
      mma_bf16(c0,a,b0,b2); mma_bf16(c1,a,b1,b3);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      total0[i] = __fadd_rn(total0[i],c0[i]);
      total1[i] = __fadd_rn(total1[i],c1[i]);
    }
  }
  const int r0 = out + (lane % 4) * 2, r1 = r0 + 8;
  store<OutBF16>(y,sample0,r0,m,n,total0[0]);
  store<OutBF16>(y,sample0,r0+1,m,n,total0[1]);
  store<OutBF16>(y,sample0,r1,m,n,total1[0]);
  store<OutBF16>(y,sample0,r1+1,m,n,total1[1]);
  if constexpr (!Small) {
    store<OutBF16>(y,sample1,r0,m,n,total0[2]);
    store<OutBF16>(y,sample1,r0+1,m,n,total0[3]);
    store<OutBF16>(y,sample1,r1,m,n,total1[2]);
    store<OutBF16>(y,sample1,r1+1,m,n,total1[3]);
  }
}

template<bool BF16> __global__ void finish_q6_partials(
    const float* partial,void* y,int count,int splits) {
  const int i=int(blockIdx.x)*128+threadIdx.x;
  if(i>=count)return;
  float value=0;
  for(int s=0;s<splits;++s)value=__fadd_rn(value,partial[size_t(s)*count+i]);
  if constexpr(BF16)static_cast<__nv_bfloat16*>(y)[i]=__float2bfloat16_rn(value);
  else static_cast<float*>(y)[i]=value;
}
cudaError_t dispatch_q6(const void* x,const uint8_t* w,void* y,float* partial,
    int m,int n,int k,int output_bf16,int splits,cudaStream_t stream) {
  const dim3 grid((unsigned(n)+31)/32,(unsigned(m)+15)/16,splits);
  void* target=splits==1?y:partial;
#define LAUNCH(O,S) canonical_register_mma<14,O,S><<<grid,64,0,stream>>>( \
  static_cast<const __nv_bfloat16*>(x),w,target,m,n,k,splits)
  if(m<=8) {if(output_bf16&&splits==1)LAUNCH(true,true);else LAUNCH(false,true);}
  else {if(output_bf16&&splits==1)LAUNCH(true,false);else LAUNCH(false,false);}
#undef LAUNCH
  auto e=cudaGetLastError();if(e!=cudaSuccess)return e;
  if(splits>1) {
    if(output_bf16)finish_q6_partials<true><<<(size_t(m)*n+127)/128,128,0,stream>>>(partial,y,m*n,splits);
    else finish_q6_partials<false><<<(size_t(m)*n+127)/128,128,0,stream>>>(partial,y,m*n,splits);
  }
  return cudaGetLastError();
}
__device__ __forceinline__ int q6_code(const uint8_t* w,int row,int col,int n,int k) {
  if(row>=n)return 0;
  const uint8_t* b=w+(size_t(row)*(k/256)+col/256)*210;
  const int p=col%256, half=p/128, group=(p%128)/32, lane=p%32;
  const int low=b[half*64+(group&1)*32+lane],high=b[128+half*32+lane];
  return ((group<2?low:low>>4)&15)|(((high>>(2*group))&3)<<4);
}
__device__ __forceinline__ float q6_scale(const uint8_t* w,int row,int col,int n,int k) {
  if(row>=n)return 0.0f;
  const uint8_t* b=w+(size_t(row)*(k/256)+col/256)*210;
  const int scale=reinterpret_cast<const int8_t*>(b+192)[(col%256)/16];
  return __fmul_rn(float(scale),read_half(b+208));
}
__global__ void pack_q6_tiles(const uint8_t* w,uint8_t* cache,int n,int k) {
  const int lane=threadIdx.x%32;
  const size_t tile=size_t(blockIdx.x)*4+threadIdx.x/32;
  const size_t total=((uint64_t(n)+15)/16)*(k/16);
  if(tile>=total)return;
  const int out=int(tile/(k/16))*16,col=int(tile%(k/16))*16+(lane%4)*2;
  const int n0=out+lane/4,n1=n0+8;
  uint64_t word=0;
#pragma unroll
  for(int j=0;j<4;++j) {
    const int row=(j&1)?n1:n0,offset=(j>=2)?8:0;
    word|=uint64_t(q6_code(w,row,col+offset,n,k))<<(12*j);
    word|=uint64_t(q6_code(w,row,col+offset+1,n,k))<<(12*j+6);
  }
  auto* codes=reinterpret_cast<uint16_t*>(cache+tile*256+lane*6);
  codes[0]=uint16_t(word);codes[1]=uint16_t(word>>16);codes[2]=uint16_t(word>>32);
  if(lane%4==0) {
    auto* scale=reinterpret_cast<float*>(cache+tile*256+192);
    scale[lane/4]=q6_scale(w,n0,col,n,k);
    scale[lane/4+8]=q6_scale(w,n1,col,n,k);
  }
}
extern "C" int gguf_q6_tile_pack(const uint8_t* w,uint8_t* cache,int n,int k,
                                 uint64_t wire_bytes,uint64_t cache_bytes,void* stream) {
  const uint64_t tiles=((uint64_t(n)+15)/16)*(k/16);
  if(!w||!cache||n<1||k<1||k%256||wire_bytes<uint64_t(n)*(k/256)*210||
      cache_bytes<tiles*256||(reinterpret_cast<uintptr_t>(w)&1))return int(cudaErrorInvalidValue);
  pack_q6_tiles<<<(tiles+3)/4,128,0,static_cast<cudaStream_t>(stream)>>>(w,cache,n,k);
  return int(cudaGetLastError());
}

extern "C" int gguf_q6_tile_run(const void* x,const uint8_t* cache,void* y,float* partial,
    int m,int n,int k,int out_bf16,int splits,uint64_t cache_bytes,
    uint64_t partial_bytes,void* stream) {
  if(!x||!cache||!y||m<1||m>32||n<1||k<1||uint64_t(m)*n>INT_MAX||k%256||out_bf16<0||out_bf16>1||
      splits<1||splits>8||(splits&(splits-1))||
      cache_bytes<((uint64_t(n)+15)/16)*(k/16)*256||
      (splits>1&&(!partial||partial_bytes<uint64_t(m)*n*splits*4)))
    return int(cudaErrorInvalidValue);
  return int(dispatch_q6(x,cache,y,partial,m,n,k,out_bf16,splits,static_cast<cudaStream_t>(stream)));
}
extern "C" int gguf_q6_tile_abi(){return 2;}
