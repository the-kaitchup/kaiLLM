// SPDX-License-Identifier: Apache-2.0
// Diagnostic wrappers around the exact locally built GGML normalization and
// unary kernels. No GGML tensor storage is allocated or copied.
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/unary.cuh"
#include "ggml-cuda/ssm-conv.cuh"
#include <stdexcept>
#include <unordered_map>
#include "native_gate_fusion.cuh"

namespace {
thread_local std::string error_message;
struct Context {
    ggml_backend_cuda_context cuda;
    std::mutex mutex;
    explicit Context(int device) : cuda(device) {}
};
Context & context_for(int device) {
    static std::mutex mutex;
    static std::unordered_map<int, std::unique_ptr<Context>> contexts;
    std::lock_guard<std::mutex> lock(mutex);
    auto & context = contexts[device];
    if (!context) context = std::make_unique<Context>(device);
    return *context;
}
ggml_tensor header(int64_t columns, int64_t rows, void * data) {
    ggml_tensor result{};
    result.type = GGML_TYPE_F32;
    result.ne[0] = columns;
    result.ne[1] = rows;
    result.ne[2] = result.ne[3] = 1;
    result.nb[0] = sizeof(float);
    result.nb[1] = columns * sizeof(float);
    result.nb[2] = result.nb[3] = rows * result.nb[1];
    result.data = data;
    return result;
}
struct Stream {
    ggml_backend_cuda_context & context;
    Stream(ggml_backend_cuda_context & ctx, uintptr_t stream) : context(ctx) {
        GGML_ASSERT(context.streams[context.device][0] == nullptr);
        context.streams[context.device][0] = stream ? reinterpret_cast<cudaStream_t>(stream) : cudaStreamLegacy;
    }
    ~Stream() { context.streams[context.device][0] = nullptr; }
};
}
extern "C" const char * current_norm_error() { return error_message.c_str(); }
extern "C" int current_norm_prepare(int device) {
    try { context_for(device); return 0; }
    catch (const std::exception & error) { error_message = error.what(); return -1; }
}
extern "C" int current_norm_run(
        void * input, void * weight, void * output, int64_t columns,
        int64_t rows, float epsilon, float scale, int operation, int device,
        uintptr_t stream) {
    try {
        auto & context = context_for(device);
        std::lock_guard<std::mutex> lock(context.mutex);
        Stream borrowed(context.cuda, stream);
        auto x = header(columns, rows, input);
        auto y = header(columns, rows, output);
        y.src[0] = &x;
        if (operation == 0) {
            y.op = GGML_OP_RMS_NORM;
            memcpy(y.op_params, &epsilon, sizeof(float));
            if (weight) {
                auto w = header(columns, 1, weight);
                auto mul = header(columns, rows, output);
                mul.op = GGML_OP_MUL;
                mul.src[0] = &y; mul.src[1] = &w;
                ggml_cuda_op_rms_norm_fused(context.cuda, &y, &mul);
            } else if (scale != 1.0f) {
                auto scaled = header(columns, rows, output);
                scaled.op = GGML_OP_SCALE;
                scaled.src[0] = &y;
                memcpy(scaled.op_params, &scale, sizeof(float));
                ggml_cuda_op_rms_norm_scale_fused(context.cuda, &y, &scaled);
            } else {
                ggml_cuda_op_rms_norm(context.cuda, &y);
            }
        } else {
            y.op = GGML_OP_UNARY;
            if (operation == 1) ggml_cuda_op_silu(context.cuda, &y);
            else if (operation == 2) ggml_cuda_op_sigmoid(context.cuda, &y);
            else if (operation == 3) ggml_cuda_op_softplus(context.cuda, &y);
            else if (operation == 4) ggml_cuda_op_exp(context.cuda, &y);
            else throw std::runtime_error("Unknown diagnostic unary operation");
        }
        auto error = cudaGetLastError();
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
        return 0;
    } catch (const std::exception & error) {
        error_message = error.what(); return -1;
    }
}

extern "C" int current_conv_run(
        void * input, void * weight, void * output, int64_t channels,
        int64_t tokens, int width, int use_silu, int device, uintptr_t stream) {
    try {
        auto & context = context_for(device);
        std::lock_guard<std::mutex> lock(context.mutex);
        Stream borrowed(context.cuda, stream);
        auto x = header(tokens + width - 1, channels, input);
        auto w = header(width, channels, weight);
        auto y = header(channels, tokens, output);
        y.op = GGML_OP_SSM_CONV;
        y.src[0] = &x; y.src[1] = &w;
        auto activated = header(channels, tokens, output);
        activated.op = GGML_OP_UNARY;
        activated.src[0] = &y;
        ggml_cuda_op_ssm_conv(context.cuda, &y, nullptr, use_silu ? &activated : nullptr);
        auto error = cudaGetLastError();
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
        return 0;
    } catch (const std::exception & error) {
        error_message = error.what(); return -1;
    }
}

// Preserve SSM convolution's four ordered multiply-adds and native SiLU, but
// read each request's state directly. Each channel owns its three cache cells.
template <bool decode>
__global__ void current_conv_indexed_kernel(
        const float * input, const float * weight, float * state, float * output,
        const int64_t * slots, const int32_t * starts, const bool * has_state,
        int channels, int tokens, int cache_slots, int64_t input_stride,
        int64_t state_stride0, int64_t state_stride1, int64_t state_stride2,
        bool use_silu) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    const int request = blockIdx.y;
    if (channel >= channels) return;
    const int begin = decode ? request : starts[request];
    const int end = decode ? request + 1 : starts[request + 1];
    if (begin < 0 || end > tokens || begin >= end) return;
    const int64_t slot = slots[request];
    if (slot <= 0 || slot >= cache_slots) {
        for (int token = begin; token < end; ++token)
            output[int64_t(token) * channels + channel] = 0.0f;
        return;
    }
    float * cache = state + slot * state_stride0 + channel * state_stride1;
    const bool initialized = decode || has_state[request];
    float previous0 = initialized ? cache[0] : 0.0f;
    float previous1 = initialized ? cache[state_stride2] : 0.0f;
    float previous2 = initialized ? cache[2 * state_stride2] : 0.0f;
    const float w0 = weight[channel * 4];
    const float w1 = weight[channel * 4 + 1];
    const float w2 = weight[channel * 4 + 2];
    const float w3 = weight[channel * 4 + 3];
    for (int token = begin; token < end; ++token) {
        const float current = input[int64_t(token) * input_stride + channel];
        float sum = 0.0f;
        sum += previous0 * w0;
        sum += previous1 * w1;
        sum += previous2 * w2;
        sum += current * w3;
        sum += 0.0f;  // Native's absent-bias addition, including signed zero.
        output[int64_t(token) * channels + channel] =
            use_silu ? ggml_cuda_op_silu_single(sum) : sum;
        previous0 = previous1;
        previous1 = previous2;
        previous2 = current;
    }
    cache[0] = previous0;
    cache[state_stride2] = previous1;
    cache[2 * state_stride2] = previous2;
}

extern "C" int current_conv_indexed_run(
        const float * input, const float * weight, float * state, float * output,
        const int64_t * slots, const int32_t * starts, const bool * has_state,
        int channels, int tokens, int requests, int cache_slots,
        int64_t input_stride, int64_t state_stride0, int64_t state_stride1,
        int64_t state_stride2, int decode, int use_silu, uintptr_t stream) {
    if (channels <= 0 || tokens < 0 || requests <= 0 || cache_slots <= 0 ||
        (decode && tokens != requests) || (!decode && (!starts || !has_state)))
        return static_cast<int>(cudaErrorInvalidValue);
    dim3 grid((channels + 127) / 128, requests);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (decode) {
        current_conv_indexed_kernel<true><<<grid, 128, 0, cuda_stream>>>(
            input, weight, state, output, slots, starts, has_state, channels,
            tokens, cache_slots, input_stride, state_stride0, state_stride1,
            state_stride2, use_silu);
    } else {
        current_conv_indexed_kernel<false><<<grid, 128, 0, cuda_stream>>>(
            input, weight, state, output, slots, starts, has_state, channels,
            tokens, cache_slots, input_stride, state_stride0, state_stride1,
            state_stride2, use_silu);
    }
    return static_cast<int>(cudaPeekAtLastError());
}
