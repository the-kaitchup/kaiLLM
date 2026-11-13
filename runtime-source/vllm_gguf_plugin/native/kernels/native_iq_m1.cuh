// SPDX-License-Identifier: Apache-2.0 AND MIT
// Native dot/quantization/reduction arithmetic: GGML authors, 2023-2026.
// See LLAMA_CPP_LICENSE.txt. SM86 M1 K1024 IQ specialization.
#pragma once
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/vecdotq.cuh"
namespace gguf_native_iq_m1 {
template<ggml_type Type, int Rows, int Groups, bool FuseQuant>
__launch_bounds__(32*Groups, 1)
__global__ void kernel(const void * __restrict__ weights,
        const block_q8_1 * __restrict__ activation, const float * __restrict__ input,
        float * __restrict__ output, int n) {
    // Native M1 has FOUR logical warps and blocks_per_iter=16. K1024
    // has only four IQ blocks, all belonging to logical warp0. Preserve
    // the three empty warp additions before the unchanged XOR tree.
    constexpr int k=1024, blocks=k/QK_K;
    const int lane=threadIdx.x, group=threadIdx.y;
    const int row=(blockIdx.x*Groups+group)*Rows;
    __shared__ block_q8_1 shared_activation[FuseQuant ? k/QK8_1 : 1];
    if constexpr (FuseQuant) {
        // Same warp-wise Q8_1 operations as native quantize_q8_1.
        for(int block=group;block<k/QK8_1;block+=Groups) {
            const float xi=input[block*QK8_1+lane];
            const float amax=warp_reduce_max<QK8_1>(fabsf(xi));
            const float sum=warp_reduce_sum<QK8_1>(xi);
            const float d=amax/127.0f;
            const int8_t q=amax==0.0f?0:roundf(xi/d);
            shared_activation[block].qs[lane]=q;
            if(lane==0) shared_activation[block].ds=make_half2(d,sum);
        }
        __syncthreads();
        activation=shared_activation;
    }
    const int block=lane/8, iqs=2*(lane%8);
    float sums[Rows]={0.0f};
    #pragma unroll
    for(int i=0;i<Rows;++i) {
        if(row+i<n) {
            if constexpr(Type==GGML_TYPE_IQ2_S)
                sums[i]+=vec_dot_iq2_s_q8_1(weights,activation+block*8,(row+i)*blocks+block,iqs);
            else if constexpr(Type==GGML_TYPE_IQ3_S)
                sums[i]+=vec_dot_iq3_s_q8_1(weights,activation+block*8,(row+i)*blocks+block,iqs);
            else
                sums[i]+=vec_dot_iq3_xxs_q8_1(weights,activation+block*8,(row+i)*blocks+block,iqs);
        }
    }
    #pragma unroll
    for(int i=0;i<Rows;++i) {
        sums[i]+=0.0f;
        sums[i]+=0.0f;
        sums[i]+=0.0f;
        sums[i]=warp_reduce_sum<32>(sums[i]);
        if(lane==i && row+i<n) output[row+i]=sums[i];
    }
}


template<ggml_type Type,int Rows,int Groups>
inline void launch(const void * weights,const void * q8,float * output,int n,cudaStream_t stream) {
    kernel<Type,Rows,Groups,false><<<n/(Rows*Groups),dim3(32,Groups),0,stream>>>(
        weights,static_cast<const block_q8_1 *>(q8),nullptr,output,n);
}
inline bool eligible(int cc,ggml_type type,int64_t n,int64_t k,int64_t m) {
    return cc==860 && m==1 && k==1024 &&
        ((type==GGML_TYPE_IQ2_S && (n==512 || n==3584 || n==4096)) ||
         (type==GGML_TYPE_IQ3_XXS && (n==512 || n==2048 || n==3584 || n==4096)) ||
         (type==GGML_TYPE_IQ3_S && n==512));
}
inline void run(ggml_type type,const void * weights,const void * q8,float * output,
                int n,int k,int64_t output_stride,cudaStream_t stream) {
    (void)k; (void)output_stride; // M1: each original matrix writes one row.
    if(type==GGML_TYPE_IQ2_S) {
        if(n==512) launch<GGML_TYPE_IQ2_S,1,2>(weights,q8,output,n,stream);
        else if(n==3584) launch<GGML_TYPE_IQ2_S,1,8>(weights,q8,output,n,stream);
        else launch<GGML_TYPE_IQ2_S,1,4>(weights,q8,output,n,stream);
    } else if(type==GGML_TYPE_IQ3_S) {
        launch<GGML_TYPE_IQ3_S,1,2>(weights,q8,output,n,stream);
    } else {
        if(n==512) launch<GGML_TYPE_IQ3_XXS,1,4>(weights,q8,output,n,stream);
        else if(n==3584) launch<GGML_TYPE_IQ3_XXS,4,2>(weights,q8,output,n,stream);
        else launch<GGML_TYPE_IQ3_XXS,2,4>(weights,q8,output,n,stream);
    }
}
} // namespace gguf_native_iq_m1
