// SPDX-License-Identifier: Apache-2.0 AND MIT
// GGML decode/arithmetic: Copyright (c) 2023-2026 The ggml authors.
// See LLAMA_CPP_LICENSE.txt. Retains two native logical-warp chains per lane.
#pragma once
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/vecdotq.cuh"
#include <type_traits>

namespace current_native_heads {
struct DecodedQ5 {
    int vl[2];
    int vh[2];
    uint16_t aux[2];
    half2 dm;
};

__device__ __forceinline__ DecodedQ5 decode_weight(const block_q5_K * b, int iqs) {
    DecodedQ5 result;
    const int offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
    const int * ql = reinterpret_cast<const int *>(b->qs + 16 * offset + 4 * ((iqs / 2) % 4));
    const int * qh = reinterpret_cast<const int *>(b->qh + 4 * ((iqs / 2) % 4));
    result.vl[0] = ql[0];
    result.vl[1] = ql[4];
    result.vh[0] = qh[0] >> offset;
    result.vh[1] = qh[4] >> offset;
    const uint16_t * scales = reinterpret_cast<const uint16_t *>(b->scales);
    const int j = offset / 2, jm = j & 1;
    const uint32_t s0 = scales[jm], s2 = scales[jm + 2], s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    result.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) |
        ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    result.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) |
        ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    result.dm = b->dm;
    return result;
}

__device__ __forceinline__ float native_dot(
        const DecodedQ5 & weight, const block_q8_1 * activation, int iqs) {
    int u[2 * QR5_K];
    float d8[QR5_K];
    const int offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
    #pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1 * b = activation + offset + i;
        d8[i] = __low2float(b->ds);
        const int * q8 = reinterpret_cast<const int *>(b->qs) + ((iqs / 2) % 4);
        u[2*i] = q8[0];
        u[2*i + 1] = q8[4];
    }
    const auto * scales = reinterpret_cast<const uint8_t *>(weight.aux);
    // This is the original native FP expression, including its integer sums,
    // two-group scaling and FMA contraction under the original compiler flags.
    return vec_dot_q5_K_q8_1_impl_vmmq(
        weight.vl, weight.vh, u, scales, scales + 2, weight.dm, d8);
}


// Each physical warp owns a subset of columns and both native logical warps.
// The native two accumulation chains stay separate until their original add.
template<int Cols, int Groups>
__launch_bounds__(32 * (8/Cols) * Groups, 1)
__global__ void q5_column_warps(
        const block_q5_K * __restrict__ weight,
        const block_q8_1 * __restrict__ activation,
        float * __restrict__ output, int64_t output_stride, int k, int padded_k) {
    constexpr int rows = 2;
    const int row = rows * (blockIdx.x * Groups + threadIdx.z);
    const int first_col = Cols * threadIdx.y;
    const int blocks = k / QK_K;
    const int iqs = 2 * (threadIdx.x % 16);
    float sums[2][Cols][rows] = {{{0.0f}}};
    #pragma unroll
    for (int logical_warp = 0; logical_warp < 2; ++logical_warp) {
        for (int block = logical_warp * 2 + threadIdx.x / 16; block < blocks; block += 4) {
            DecodedQ5 decoded[rows];
            #pragma unroll
            for (int i = 0; i < rows; ++i) decoded[i] = decode_weight(weight + (row+i)*blocks+block, iqs);
            #pragma unroll
            for (int j = 0; j < Cols; ++j) {
                const auto * x = activation + (first_col+j)*(padded_k/QK8_1) + block*(QK_K/QK8_1);
                #pragma unroll
                for (int i = 0; i < rows; ++i) sums[logical_warp][j][i] += native_dot(decoded[i], x, iqs);
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < Cols; ++j) {
        #pragma unroll
        for (int i = 0; i < rows; ++i) {
            sums[0][j][i] += sums[1][j][i];
            sums[0][j][i] = warp_reduce_sum<32>(sums[0][j][i]);
            if (threadIdx.x == i) output[int64_t(first_col+j)*output_stride+row+i] = sums[0][j][i];
        }
    }
}


struct DecodedQ6 {
    int values[2];
    int scales[2];
    float d;
};

__device__ __forceinline__ DecodedQ6 decode_weight(const block_q6_K * b, int iqs) {
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));
    const int vl = get_int_b2(b->ql, iqs);
    const int vh = get_int_b2(b->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;
    DecodedQ6 out;
    #pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;
        out.values[i] = __vsub4((vil | vih), 0x20202020);
        out.scales[i] = b->scales[scale_offset+4*i];
    }
    out.d = __half2float(b->d);
    return out;
}

__device__ __forceinline__ float native_dot(
        const DecodedQ6 & w, const block_q8_1 * activation, int iqs) {
    const int offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    float sumf = 0.0f;
    #pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const auto & b = activation[offset + 2*i];
        const int u = get_int_b4(b.qs, iqs % QI8_1);
        const float d8 = __low2float(b.ds);
        sumf += d8 * (ggml_cuda_dp4a(w.values[i], u, 0) * w.scales[i]);
    }
    return w.d * sumf;
}


// Each physical warp owns a subset of columns and both native logical warps.
// The native two accumulation chains stay separate until their original add.
template<int Cols, int Groups>
__launch_bounds__(32 * (8/Cols) * Groups, 1)
__global__ void q6_column_warps(
        const block_q6_K * __restrict__ weight,
        const block_q8_1 * __restrict__ activation,
        float * __restrict__ output, int64_t output_stride, int k, int padded_k) {
    constexpr int rows = 2;
    const int row = rows * (blockIdx.x * Groups + threadIdx.z);
    const int first_col = Cols * threadIdx.y;
    const int blocks = k / QK_K;
    const int iqs = threadIdx.x;
    float sums[2][Cols][rows] = {{{0.0f}}};
    #pragma unroll
    for (int logical_warp = 0; logical_warp < 2; ++logical_warp) {
        for (int block = logical_warp; block < blocks; block += 2) {
            DecodedQ6 decoded[rows];
            #pragma unroll
            for (int i = 0; i < rows; ++i) decoded[i] = decode_weight(weight + (row+i)*blocks+block, iqs);
            #pragma unroll
            for (int j = 0; j < Cols; ++j) {
                const auto * x = activation + (first_col+j)*(padded_k/QK8_1) + block*(QK_K/QK8_1);
                #pragma unroll
                for (int i = 0; i < rows; ++i) sums[logical_warp][j][i] += native_dot(decoded[i], x, iqs);
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < Cols; ++j) {
        #pragma unroll
        for (int i = 0; i < rows; ++i) {
            sums[0][j][i] += sums[1][j][i];
            sums[0][j][i] = warp_reduce_sum<32>(sums[0][j][i]);
            if (threadIdx.x == i) output[int64_t(first_col+j)*output_stride+row+i] = sums[0][j][i];
        }
    }
}


inline bool eligible(int cc, ggml_type type, int64_t n, int64_t k, int64_t m) {
    return cc == 860 && m == 8 && n == 248320 &&
        ((type == GGML_TYPE_Q5_K && k == 1024) ||
         (type == GGML_TYPE_Q6_K && k == 2048));
}
inline void run(ggml_type type, const void * weights, const void * q8,
                float * output, int n, int k, int64_t output_stride,
                cudaStream_t stream) {
    const auto * x = static_cast<const block_q8_1 *>(q8);
    const int padded = GGML_PAD(k, MATRIX_ROW_PADDING);
    if (type == GGML_TYPE_Q5_K) {
        q5_column_warps<8,2><<<n/4,dim3(32,1,2),0,stream>>>(
            static_cast<const block_q5_K *>(weights),x,output,output_stride,k,padded);
    } else {
        q6_column_warps<8,4><<<n/8,dim3(32,1,4),0,stream>>>(
            static_cast<const block_q6_K *>(weights),x,output,output_stride,k,padded);
    }
}

namespace packed_q5 {
struct alignas(4) ScaleQ5 { half2 dm; uint8_t qh[32], scales[16], qs[128]; };
static_assert(sizeof(ScaleQ5)==180);
struct DecodedQ5 { int vl[2], vh[2]; uint16_t aux[2]; half2 dm; };
struct alignas(16) LaneQ5 { int4 u; float2 d8; int2 sums; };
static_assert(sizeof(LaneQ5)==32);
template<bool CG, typename T>
__device__ __forceinline__ T read_weight(const T * p) {
    if constexpr (CG) return __ldcg(p);
    else return *p;
}


template<typename Weight, bool CG>
__device__ __forceinline__ DecodedQ5 decode_weight(const Weight * b,int iqs) {
    DecodedQ5 out;
    const int offset=2*((iqs/2)/4);
    const int * ql=reinterpret_cast<const int *>(b->qs+16*offset+4*((iqs/2)%4));
    const int * qh=reinterpret_cast<const int *>(b->qh+4*((iqs/2)%4));
    out.vl[0]=read_weight<CG>(ql); out.vl[1]=read_weight<CG>(ql+4);
    out.vh[0]=read_weight<CG>(qh)>>offset; out.vh[1]=read_weight<CG>(qh+4)>>offset;
    if constexpr(std::is_same_v<Weight,ScaleQ5>) {
        out.aux[0]=read_weight<CG>(reinterpret_cast<const uint16_t *>(b->scales+offset));
        out.aux[1]=read_weight<CG>(reinterpret_cast<const uint16_t *>(b->scales+8+offset));
    } else {
        const auto * scales=reinterpret_cast<const uint16_t *>(b->scales);
        const int j=offset/2,jm=j&1;
        const uint32_t s0=read_weight<CG>(scales+jm),s2=read_weight<CG>(scales+jm+2),s4=read_weight<CG>(scales+jm+4);
        const uint32_t hi=uint32_t(-int32_t(j>=2));
        out.aux[0]=uint16_t(((s0&0x3f3f)&~hi)|((((s4>>0)&0x0f0f)|((s0&0xc0c0)>>2))&hi));
        out.aux[1]=uint16_t(((s2&0x3f3f)&~hi)|((((s4>>4)&0x0f0f)|((s2&0xc0c0)>>2))&hi));
    }
    const uint32_t dm=read_weight<CG>(reinterpret_cast<const uint32_t *>(&b->dm));
    out.dm=__halves2half2(__ushort_as_half(dm&0xffff),__ushort_as_half(dm>>16));
    return out;
}

__global__ void pack_activation(const block_q8_1 * input,LaneQ5 * output,int blocks,int padded) {
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=blocks*8*16) return;
    const int lane=index%16,col=(index/16)%8,block=index/(16*8),offset=2*(lane/4);
    const auto * x=input+col*(padded/QK8_1)+block*8;
    LaneQ5 out;
    out.u=make_int4(get_int_b4(x[offset].qs,lane%4),get_int_b4(x[offset].qs,lane%4+4),
        get_int_b4(x[offset+1].qs,lane%4),get_int_b4(x[offset+1].qs,lane%4+4));
    out.d8=make_float2(__low2float(x[offset].ds),__low2float(x[offset+1].ds));
    out.sums=make_int2(ggml_cuda_dp4a(0x01010101,out.u.x,ggml_cuda_dp4a(0x01010101,out.u.y,0)),
        ggml_cuda_dp4a(0x01010101,out.u.z,ggml_cuda_dp4a(0x01010101,out.u.w,0)));
    output[index]=out;
}

__device__ __forceinline__ float dot(const DecodedQ5 & w,const LaneQ5 & x) {
    const auto * sc=reinterpret_cast<const uint8_t *>(w.aux);
    float sumd=0.0f,summ=0.0f;
    const int u[4]={x.u.x,x.u.y,x.u.z,x.u.w};
    const float d8[2]={x.d8.x,x.d8.y};
    const int sums[2]={x.sums.x,x.sums.y};
    #pragma unroll
    for(int i=0;i<2;++i) {
        const int v0=((w.vl[0]>>(4*i))&0x0f0f0f0f)|(((w.vh[0]>>i)<<4)&0x10101010);
        const int v1=((w.vl[1]>>(4*i))&0x0f0f0f0f)|(((w.vh[1]>>i)<<4)&0x10101010);
        const int dot1=ggml_cuda_dp4a(v0,u[2*i],ggml_cuda_dp4a(v1,u[2*i+1],0));
        sumd+=d8[i]*(dot1*sc[i]);
        summ+=d8[i]*(sums[i]*sc[i+2]);
    }
    const float2 dm=__half22float2(w.dm);
    return dm.x*sumd-dm.y*summ;
}
template<typename Weight, bool CG, int Groups>
__launch_bounds__(32*Groups,1)
__global__ void matmul(const Weight * __restrict__ weights,
        const LaneQ5 * __restrict__ activation, float * __restrict__ output, int64_t output_stride, int k) {
    const int row=4*(blockIdx.x*Groups+threadIdx.z), lane=threadIdx.x, blocks=k/256;
    float sums[2][8][4] = {{{0.0f}}};
    #pragma unroll
    for (int logical=0; logical<2; ++logical) {
        for (int block=logical*2+lane/16; block<blocks; block+=4) {
            DecodedQ5 w[4];
            #pragma unroll
            for (int i=0;i<4;++i) w[i]=decode_weight<Weight,CG>(weights+(row+i)*blocks+block,2*(lane%16));
            #pragma unroll
            for (int j=0;j<8;++j) {
                const LaneQ5 x=activation[(block*8+j)*16+lane%16];
                #pragma unroll
                for (int i=0;i<4;++i) sums[logical][j][i]+=dot(w[i],x);
            }
        }
    }
    #pragma unroll
    for (int j=0;j<8;++j) {
        #pragma unroll
        for (int i=0;i<4;++i) {
            sums[0][j][i]+=sums[1][j][i];
            sums[0][j][i]=warp_reduce_sum<32>(sums[0][j][i]);
            if(lane==i) output[int64_t(j)*output_stride+row+i]=sums[0][j][i];
        }
    }
}

// Exact Q5_K K1024 specialization: each logical warp has one block per lane.
template<int Rows,int Groups>
__launch_bounds__(32*Groups,1)
__global__ void matmul_k1024(const block_q5_K * __restrict__ weights,const LaneQ5 * __restrict__ activation,float * __restrict__ output,int64_t stride) {
    const int row=Rows*(blockIdx.x*Groups+threadIdx.z),lane=threadIdx.x;
    float sums[2][8][Rows]={{{0.0f}}};
    #pragma unroll
    for(int logical=0;logical<2;++logical) {
        const int block=logical*2+lane/16;
        DecodedQ5 decoded[Rows];
        #pragma unroll
        for(int i=0;i<Rows;++i) {
            const int idx=(row+i)*4+block;
            decoded[i]=decode_weight<block_q5_K,false>(weights+idx,2*(lane%16));
        }
        #pragma unroll
        for(int j=0;j<8;++j) {
            const LaneQ5 x=activation[(block*8+j)*16+lane%16];
            #pragma unroll
            for(int i=0;i<Rows;++i)sums[logical][j][i]+=dot(decoded[i],x);
        }
    }
    #pragma unroll
    for(int j=0;j<8;++j) {
        #pragma unroll
        for(int i=0;i<Rows;++i) {
            sums[0][j][i]+=sums[1][j][i];
            const float value=warp_reduce_sum<32>(sums[0][j][i]);
            if(lane==i)output[int64_t(j)*stride+row+i]=value;
        }
    }
}

} // namespace packed_q5
namespace packed_q6 {
struct alignas(16) PaddedQ6 {
    uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; half d; uint8_t pad[14];
};
static_assert(sizeof(PaddedQ6) == 224);
struct DecodedQ6 { int values[2], scales[2]; float d; };

template<bool CG, typename T>
__device__ __forceinline__ T read_weight(const T * p) {
    if constexpr (CG) return __ldcg(p);
    else return *p;
}

template<typename Weight, bool CG>
__device__ __forceinline__ int word(const uint8_t * p, int index) {
    if constexpr (std::is_same_v<Weight, PaddedQ6>) {
        return read_weight<CG>(reinterpret_cast<const int *>(p) + index);
    } else {
        const auto * q = reinterpret_cast<const uint16_t *>(p) + 2*index;
        return int(read_weight<CG>(q)) | (int(read_weight<CG>(q+1)) << 16);
    }
}

template<typename Weight, bool CG>
__device__ __forceinline__ DecodedQ6 decode_weight(const Weight * b, int iqs) {
    const int scale_offset = 8*(iqs/16) + (iqs%16)/4;
    const int shift = 2*((iqs%16)/8);
    const int vl = word<Weight, CG>(b->ql, iqs);
    const int vh = word<Weight, CG>(b->qh, 8*(iqs/16)+iqs%8) >> shift;
    DecodedQ6 out;
    #pragma unroll
    for (int i=0; i<2; ++i) {
        const int low = (vl >> (4*i)) & 0x0f0f0f0f;
        const int high = ((vh >> (4*i)) << 4) & 0x30303030;
        out.values[i] = __vsub4(low | high, 0x20202020);
        out.scales[i] = read_weight<CG>(b->scales + scale_offset + 4*i);
    }
    const uint16_t d = read_weight<CG>(reinterpret_cast<const uint16_t *>(&b->d));
    out.d = __half2float(__ushort_as_half(d));
    return out;
}

// Duplicate integer codes and expand already-rounded native FP16 scales to F32.
// No activation quantization or floating-point product is changed.
__global__ void pack_activation(const block_q8_1 * input, int4 * output, int blocks, int padded) {
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= blocks*8*32) return;
    const int lane = index % 32, col = (index/32)%8, block = index/(32*8);
    const int offset = 4*(lane/16)+(lane%16)/8;
    const auto * x = input + col*(padded/QK8_1)+block*8;
    output[index] = make_int4(get_int_b4(x[offset].qs, lane%8),
        get_int_b4(x[offset+2].qs, lane%8),
        __float_as_int(__low2float(x[offset].ds)),
        __float_as_int(__low2float(x[offset+2].ds)));
}

__device__ __forceinline__ float dot(const DecodedQ6 & w, const int4 & x) {
    float sum = 0.0f;
    sum += __int_as_float(x.z) * (ggml_cuda_dp4a(w.values[0], x.x, 0) * w.scales[0]);
    sum += __int_as_float(x.w) * (ggml_cuda_dp4a(w.values[1], x.y, 0) * w.scales[1]);
    return w.d * sum;
}

template<typename Weight, bool CG, int Groups>
__launch_bounds__(32*Groups,1)
__global__ void matmul(const Weight * __restrict__ weights,
        const int4 * __restrict__ activation, float * __restrict__ output, int64_t output_stride, int k) {
    const int row=4*(blockIdx.x*Groups+threadIdx.z), lane=threadIdx.x, blocks=k/256;
    float sums[2][8][4] = {{{0.0f}}};
    #pragma unroll
    for (int logical=0; logical<2; ++logical) {
        for (int block=logical; block<blocks; block+=2) {
            DecodedQ6 w[4];
            #pragma unroll
            for (int i=0;i<4;++i) w[i]=decode_weight<Weight,CG>(weights+(row+i)*blocks+block,lane);
            #pragma unroll
            for (int j=0;j<8;++j) {
                const int4 x=activation[(block*8+j)*32+lane];
                #pragma unroll
                for (int i=0;i<4;++i) sums[logical][j][i]+=dot(w[i],x);
            }
        }
    }
    #pragma unroll
    for (int j=0;j<8;++j) {
        #pragma unroll
        for (int i=0;i<4;++i) {
            sums[0][j][i]+=sums[1][j][i];
            sums[0][j][i]=warp_reduce_sum<32>(sums[0][j][i]);
            if(lane==i) output[int64_t(j)*output_stride+row+i]=sums[0][j][i];
        }
    }
}

// Exact Q6_K K2048 specialization: unroll each native four-block FP chain.
template<int Rows,int Groups>
__launch_bounds__(32*Groups,1)
__global__ void matmul_k2048(const block_q6_K * __restrict__ weights,const int4 * __restrict__ activation,float * __restrict__ output,int64_t stride) {
    const int row=Rows*(blockIdx.x*Groups+threadIdx.z),lane=threadIdx.x;
    float sums[2][8][Rows]={{{0.0f}}};
    #pragma unroll
    for(int logical=0;logical<2;++logical) {
        #pragma unroll 4
        for(int block=logical;block<8;block+=2) {
            DecodedQ6 w[Rows];
            #pragma unroll
            for(int i=0;i<Rows;++i)w[i]=decode_weight<block_q6_K,false>(weights+(row+i)*8+block,lane);
            #pragma unroll
            for(int j=0;j<8;++j) {
                const int4 x=activation[(block*8+j)*32+lane];
                #pragma unroll
                for(int i=0;i<Rows;++i)sums[logical][j][i]+=dot(w[i],x);
            }
        }
    }
    #pragma unroll
    for(int j=0;j<8;++j) {
        #pragma unroll
        for(int i=0;i<Rows;++i) {
            sums[0][j][i]+=sums[1][j][i];
            const float value=warp_reduce_sum<32>(sums[0][j][i]);
            if(lane==i)output[int64_t(j)*stride+row+i]=value;
        }
    }
}

} // namespace packed_q6

// Per-call scratch only: duplicate integer codes and expand the native FP16
// activation scales once. Original Q8 quantization and all FP products remain
// unchanged. Both layouts need 4096 bytes for each 256 input elements.
inline size_t activation_bytes(int64_t k) { return size_t(k) * 16; }

inline void run_packed(ggml_type type, const void * weights, const void * q8,
                void * activation_scratch, float * output, int n, int k,
                int64_t output_stride, cudaStream_t stream) {
    const auto * x = static_cast<const block_q8_1 *>(q8);
    const int padded = GGML_PAD(k, MATRIX_ROW_PADDING), blocks = k / QK_K;
    if (type == GGML_TYPE_Q5_K) {
        auto * packed = static_cast<packed_q5::LaneQ5 *>(activation_scratch);
        packed_q5::pack_activation<<<(blocks*8*16+255)/256,256,0,stream>>>(x,packed,blocks,padded);
        packed_q5::matmul_k1024<4,1><<<n/4,dim3(32,1,1),0,stream>>>(
            static_cast<const block_q5_K *>(weights),packed,output,output_stride);
    } else {
        auto * packed = static_cast<int4 *>(activation_scratch);
        packed_q6::pack_activation<<<(blocks*8*32+255)/256,256,0,stream>>>(x,packed,blocks,padded);
        packed_q6::matmul_k2048<4,1><<<n/4,dim3(32,1,1),0,stream>>>(
            static_cast<const block_q6_K *>(weights),packed,output,output_stride);
    }
}
} // namespace current_native_heads
