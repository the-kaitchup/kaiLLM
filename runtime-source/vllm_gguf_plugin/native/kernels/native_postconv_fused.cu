// SPDX-License-Identifier: Apache-2.0 AND MIT
// Normalization/unary arithmetic adapted from llama.cpp norm.cu, unary.cu,
// and common.cuh. Copyright (c) 2023-2026 The ggml authors.
// See LLAMA_CPP_LICENSE.txt for the upstream notice and permission terms.
#include "ggml-cuda/common.cuh"
#include <cstdint>

// One grid combines the two independent native RMS kernels and V/g/beta work.
// Each Q/K row retains the original 256-thread scalar reduction, including
// inactive zero lanes. Pass ncols/eps/scale at runtime as the native API does.
__global__ void native_postconv_kernel(
        const float * conv, const float * a, const float * b,
        const float * original_a, const float * dt,
        float * q, float * k, float * v, float * g, float * beta,
        int64_t conv_stride, int64_t a_row, int64_t a_col,
        int64_t b_row, int64_t b_col, int ncols, int heads,
        float eps, float scale_out, bool normalize, bool output_g_exp) {
    const int row = blockIdx.x;
    const int head = row % heads;
    const int token = row / heads;
    const int kind = blockIdx.y;
    const int tid = threadIdx.x;
    const float * x = conv + token * conv_stride + (kind * heads + head) * ncols;
    float * dst = (kind == 0 ? q : kind == 1 ? k : v) + int64_t(row) * ncols;
    if (kind == 2) {
        for (int col = tid; col < ncols; col += 256) dst[col] = x[col];
        if (tid == 0) {
            // Existing reference executes these two Torch elementwise ops
            // across native unary-call boundaries. Keep independent F32
            // rounding, including subnormals, rather than fusing through them.
            float z;
            asm("add.rn.f32 %0, %1, %2;" : "=f"(z) :
                "f"(a[token * a_row + head * a_col]), "f"(dt[head]));
            const float sp = z > 20.0f ? z : logf(1.0f + expf(z));
            float gate;
            asm("mul.rn.f32 %0, %1, %2;" : "=f"(gate) :
                "f"(sp), "f"(original_a[head]));
            g[row] = output_g_exp ? expf(gate) : gate;
            const float bv = b[token * b_row + head * b_col];
            beta[row] = 1.0f / (1.0f + expf(-bv));
        }
        return;
    }
    if (!normalize) {
        for (int col = tid; col < ncols; col += 256) dst[col] = x[col];
        return;
    }
    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += 256) {
        const float xi = x[col];
        tmp += xi * xi;
    }
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, 256>(tmp, s_sum);
    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);
    for (int col = tid; col < ncols; col += 256)
        dst[col] = scale_out * (scale * x[col]);
}

extern "C" int native_postconv_run(
        const float * conv, const float * a, const float * b,
        const float * original_a, const float * dt,
        float * q, float * k, float * v, float * g, float * beta,
        int64_t tokens, int64_t conv_stride, int64_t a_row, int64_t a_col,
        int64_t b_row, int64_t b_col, float eps, float scale,
        int normalize, int output_g_exp, uintptr_t stream) {
    if (tokens <= 0 || tokens > INT32_MAX / 16 || conv_stride < 6144 ||
            a_row < 0 || a_col < 0 || b_row < 0 || b_col < 0)
        return int(cudaErrorInvalidValue);
    native_postconv_kernel<<<dim3(tokens * 16, 3), 256, 32 * sizeof(float),
            reinterpret_cast<cudaStream_t>(stream)>>>(
        conv, a, b, original_a, dt, q, k, v, g, beta,
        conv_stride, a_row, a_col, b_row, b_col, 128, 16,
        eps, scale, normalize != 0, output_g_exp != 0);
    return int(cudaPeekAtLastError());
}
