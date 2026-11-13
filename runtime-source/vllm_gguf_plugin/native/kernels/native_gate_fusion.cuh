#pragma once
// SPDX-License-Identifier: Apache-2.0 AND MIT
// Native RMS/unary arithmetic: Copyright (c) 2023-2026 The ggml authors.
// See LLAMA_CPP_LICENSE.txt for upstream license terms.
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/unary.cuh"
#include "ggml-cuda/quantize.cuh"
#include <cstdint>

static __device__ __forceinline__ float torch_mul(float a, float b) {
    float result;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
    return result;
}

__global__ void native_swiglu_kernel(
        const float * input, float * output, int64_t width, int64_t rows,
        int64_t input_stride) {
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * width) return;
    const int64_t row = index / width, column = index % width;
    const float gate = input[row * input_stride + column];
    const float up = input[row * input_stride + width + column];
    output[index] = torch_mul(ggml_cuda_op_silu_single(gate), up);
}

__global__ void native_gated_rms_kernel(
        const float * input, const float * weight, const float * gate,
        float * output, int ncols, int64_t gate_rows_per_batch,
        int64_t gate_batch_stride, int64_t gate_row_stride, float eps) {
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;
    const float * x = input + row * ncols;
    const float * z = gate + (row/gate_rows_per_batch)*gate_batch_stride
                          + (row%gate_rows_per_batch)*gate_row_stride;
    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += 256) {
        const float xi = x[col];
        tmp += xi * xi;
    }
    extern __shared__ float scratch[];
    tmp = block_reduce<block_reduce_method::SUM, 256>(tmp, scratch);
    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);
    for (int col = tid; col < ncols; col += 256) {
        const float norm = scale * x[col] * weight[col];
        output[row * ncols + col] = torch_mul(
            norm, ggml_cuda_op_silu_single(z[col]));
    }
}

__global__ void native_swiglu_q8_kernel(
        const float * input, block_q8_1 * output, int64_t width,
        int64_t input_stride, int64_t padded_width) {
    const int64_t col = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= padded_width) return;
    const int64_t row = blockIdx.y;
    const int64_t linear = row * padded_width + col;
    const int64_t block = linear / QK8_1, lane = linear % QK8_1;
    float xi = 0.0f;
    if (col < width) {
        const float gate = input[row * input_stride + col];
        const float up = input[row * input_stride + width + col];
        xi = torch_mul(ggml_cuda_op_silu_single(gate), up);
    }
    float amax = fabsf(xi), sum = xi;
    amax = warp_reduce_max<QK8_1>(amax);
    sum = warp_reduce_sum<QK8_1>(sum);
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
    output[block].qs[lane] = q;
    if (lane == 0) output[block].ds = make_half2(d, sum);
}

__global__ void native_gated_rms_q8_kernel(
        const float * input, const float * weight, const float * gate,
        block_q8_1 * output, int ncols, int64_t gate_batch_stride,
        int64_t gate_row_stride, float eps) {
    const int64_t row = blockIdx.x;
    const int tid = threadIdx.x;
    const float * x = input + row * ncols;
    const float * z = gate + (row/16)*gate_batch_stride + (row%16)*gate_row_stride;
    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += 256) {
        const float xi = x[col];
        tmp += xi * xi;
    }
    extern __shared__ float scratch[];
    tmp = block_reduce<block_reduce_method::SUM, 256>(tmp, scratch);
    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);
    for (int col = tid; col < ncols; col += 256) {
        const float norm = scale * x[col] * weight[col];
        const float xi = torch_mul(norm,
            ggml_cuda_op_silu_single(z[col]));
        float amax = fabsf(xi), sum = xi;
        amax = warp_reduce_max<QK8_1>(amax);
        sum = warp_reduce_sum<QK8_1>(sum);
        const float d = amax / 127.0f;
        const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
        const int64_t block = (row*ncols+col)/QK8_1;
        const int lane = col%QK8_1;
        output[block].qs[lane] = q;
        if (lane == 0) output[block].ds = make_half2(d,sum);
    }
}

extern "C" int native_swiglu_run(
        const float * input, float * output, int64_t width, int64_t rows,
        int64_t input_stride, uintptr_t stream) {
    if (!input || !output || width <= 0 || rows <= 0 ||
            input_stride < 2*width || rows > INT32_MAX/width)
        return int(cudaErrorInvalidValue);
    native_swiglu_kernel<<<(rows*width+255)/256,256,0,
        reinterpret_cast<cudaStream_t>(stream)>>>(input,output,width,rows,input_stride);
    return int(cudaPeekAtLastError());
}

extern "C" int native_gated_rms_run(
        const float * input, const float * weight, const float * gate,
        float * output, int64_t rows, int64_t gate_rows_per_batch,
        int64_t gate_batch_stride, int64_t gate_row_stride, float eps,
        uintptr_t stream) {
    if (!input || !weight || !gate || !output || rows <= 0 ||
            rows > INT32_MAX || eps < 0 || gate_rows_per_batch <= 0 ||
            gate_row_stride < 128 || gate_batch_stride < 0)
        return int(cudaErrorInvalidValue);
    native_gated_rms_kernel<<<rows,256,32*sizeof(float),
        reinterpret_cast<cudaStream_t>(stream)>>>(input,weight,gate,output,128,
            gate_rows_per_batch,gate_batch_stride,gate_row_stride,eps);
    return int(cudaPeekAtLastError());
}

extern "C" int native_swiglu_q8_run(
        const float * input, void * output, int64_t width, int64_t rows,
        int64_t input_stride, int64_t padded_width, uintptr_t stream) {
    if (!input || !output || width <= 0 || rows <= 0 || rows > 8 ||
            input_stride < 2*width || padded_width < width || padded_width % 512)
        return int(cudaErrorInvalidValue);
    native_swiglu_q8_kernel<<<dim3((padded_width+255)/256,rows),256,0,
        reinterpret_cast<cudaStream_t>(stream)>>>(input,
        static_cast<block_q8_1 *>(output),width,input_stride,padded_width);
    return int(cudaPeekAtLastError());
}

extern "C" int native_q8_reference_run(
        const float * input, void * output, int64_t width, int64_t rows,
        int64_t padded_width, uintptr_t stream) {
    if (!input || !output || width <= 0 || rows <= 0 ||
            padded_width < width || padded_width % 512)
        return int(cudaErrorInvalidValue);
    quantize_row_q8_1_cuda(input,nullptr,output,GGML_TYPE_Q8_0,
        width,width,rows*width,rows*width,padded_width,rows,1,1,
        reinterpret_cast<cudaStream_t>(stream));
    return int(cudaPeekAtLastError());
}

extern "C" int native_gated_rms_q8_run(
        const float * input, const float * weight, const float * gate,
        void * output, int64_t tokens, int64_t gate_batch_stride,
        int64_t gate_row_stride, float eps, uintptr_t stream) {
    if (!input || !weight || !gate || !output || tokens <= 0 ||
            tokens > 8 || eps < 0 || gate_row_stride < 128 ||
            gate_batch_stride < 16*gate_row_stride)
        return int(cudaErrorInvalidValue);
    native_gated_rms_q8_kernel<<<tokens*16,256,32*sizeof(float),
        reinterpret_cast<cudaStream_t>(stream)>>>(
            input,weight,gate,static_cast<block_q8_1 *>(output),128,
            gate_batch_stride,gate_row_stride,eps);
    return int(cudaPeekAtLastError());
}
