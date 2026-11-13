// SPDX-License-Identifier: Apache-2.0
// F32 butterflies with explicit arithmetic rounding.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace {
__device__ __forceinline__ float input_value(const void * x, int64_t index, int dtype) {
    if (dtype == 1) return __bfloat162float(static_cast<const __nv_bfloat16 *>(x)[index]);
    if (dtype == 2) return __half2float(static_cast<const __half *>(x)[index]);
    return static_cast<const float *>(x)[index];
}

__device__ __forceinline__ void output_value(void * y, int64_t index, int dtype,
                                           float value, const void * global,
                                           bool post, int global_dtype) {
    if (post && global_dtype == 1) {
        const double scaled = __dmul_rn(static_cast<double>(value),
                                       *static_cast<const double *>(global));
        if (dtype == 1) static_cast<__nv_bfloat16 *>(y)[index] = __double2bfloat16(scaled);
        else if (dtype == 2) static_cast<__half *>(y)[index] = __double2half(scaled);
        else static_cast<float *>(y)[index] = __double2float_rn(scaled);
    } else {
        if (post) value = __fmul_rn(value, *static_cast<const float *>(global));
        if (dtype == 1) static_cast<__nv_bfloat16 *>(y)[index] = __float2bfloat16_rn(value);
        else if (dtype == 2) static_cast<__half *>(y)[index] = __float2half_rn(value);
        else static_cast<float *>(y)[index] = value;
    }
}

template<int P, int THREADS>
__global__ void power_hadamard_kernel(const void * x, const float * scale,
                                     const void * global, void * y,
                                     int n, int input_dtype, int output_dtype,
                                     int global_dtype, int pre_mode, bool post,
                                     float normalizer) {
    constexpr int ITEMS = (P + THREADS - 1) / THREADS;
    __shared__ float shared[P];
    const int tid = threadIdx.x;
    const int factor = blockIdx.y;
    const int64_t base = int64_t(blockIdx.x) * n + factor * P;
    float values[ITEMS];
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        const int index = tid + j * THREADS;
        const float raw = index < P ? input_value(x, base + index, input_dtype) : 0.0f;
        const float gain = pre_mode && index < P ? scale[factor * P + index] : 1.0f;
        float value = pre_mode ? __fmul_rn(raw, gain) : raw;
        if (pre_mode == 2 && P > 1) {
            const float other = __shfl_xor_sync(0xffffffffu, value, 1);
            value = __fmaf_rn((index & 1) ? -raw : raw, gain, other);
        }
        values[j] = value;
    }
    #pragma unroll
    for (int step = 1; step < 32 && step < P; step *= 2) {
        if (step == 1 && pre_mode == 2) continue;
        #pragma unroll
        for (int j = 0; j < ITEMS; ++j) {
            const float other = __shfl_xor_sync(0xffffffffu, values[j], step);
            values[j] = (tid & step) ? __fsub_rn(other, values[j])
                                    : __fadd_rn(values[j], other);
        }
    }
    #pragma unroll
    for (int step = 32; step < THREADS && step < P; step *= 2) {
        #pragma unroll
        for (int j = 0; j < ITEMS; ++j) {
            const int index = tid + j * THREADS;
            if (index < P) shared[index] = values[j];
        }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < ITEMS; ++j) {
            const int index = tid + j * THREADS;
            const float other = index < P ? shared[index ^ step] : 0.0f;
            values[j] = (tid & step) ? __fsub_rn(other, values[j])
                                    : __fadd_rn(values[j], other);
        }
        if (step * 2 < THREADS && step * 2 < P) __syncthreads();
    }
    #pragma unroll
    for (int step = THREADS; step < P; step *= 2) {
        #pragma unroll
        for (int j = 0; j < ITEMS; ++j) {
            if ((j & (step / THREADS)) == 0) {
                const int other = j ^ (step / THREADS);
                const float a = values[j], b = values[other];
                values[j] = __fadd_rn(a, b);
                values[other] = __fsub_rn(a, b);
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        const int index = tid + j * THREADS;
        if (index >= P) continue;
        float value = __fmul_rn(values[j], normalizer);
        if (post) value = __fmul_rn(value, scale[factor * P + index]);
        output_value(y, base + index, output_dtype, value, global, post, global_dtype);
    }
}

template<int THREADS>
cudaError_t dispatch(const void * x, const float * scale, const void * global,
                     void * y, int n, int p, int batch, int input_dtype,
                     int output_dtype, int global_dtype, int pre_mode, bool post,
                     float normalizer, cudaStream_t stream) {
    const dim3 grid(batch, n / p);
    #define POWER_CASE(P) case P: power_hadamard_kernel<P, THREADS><<<grid, THREADS, 0, stream>>>( \
        x, scale, global, y, n, input_dtype, output_dtype, global_dtype, pre_mode, post, normalizer); break;
    switch (p) {
        POWER_CASE(1) POWER_CASE(2) POWER_CASE(4) POWER_CASE(8)
        POWER_CASE(16) POWER_CASE(32) POWER_CASE(64) POWER_CASE(128)
        POWER_CASE(256) POWER_CASE(512) POWER_CASE(1024) POWER_CASE(2048)
        default: return cudaErrorInvalidValue;
    }
    #undef POWER_CASE
    return cudaGetLastError();
}
}  // namespace

extern "C" int power_hadamard_run(const void * x, const float * scale,
                                  const void * global, void * y, int n, int p,
                                  int batch, int input_dtype, int output_dtype,
                                  int global_dtype, int threads, int pre_mode,
                                  int post, float normalizer, uintptr_t stream) {
    if (!x || !scale || !global || !y || n < 1 || p < 1 || n % p ||
        batch < 1 || input_dtype < 0 || input_dtype > 2 || output_dtype < 0 ||
        output_dtype > 2 || global_dtype < 0 || global_dtype > 1 ||
        pre_mode < 0 || pre_mode > 2 || (post && (pre_mode || n != p)))
        return int(cudaErrorInvalidValue);
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    #define THREAD_CASE(T) case T: return int(dispatch<T>(x, scale, global, y, n, p, batch, \
        input_dtype, output_dtype, global_dtype, pre_mode, post, normalizer, cuda_stream));
    switch (threads) {
        THREAD_CASE(32) THREAD_CASE(64) THREAD_CASE(128) THREAD_CASE(256)
        default: return int(cudaErrorInvalidValue);
    }
    #undef THREAD_CASE
}

extern "C" const char * power_hadamard_error(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}

// Independent SU transforms in one launch; raw X is shared, arithmetic is not.
template<int P, int THREADS>
__global__ void power_hadamard_grouped_input_kernel(const void * x, const float * scale,
                                     const void * global, void * y,
                                     int n, int batch, int input_dtype, int output_dtype,
                                     int global_dtype, int pre_mode, bool post,
                                     float normalizer) {
    constexpr int ITEMS = (P + THREADS - 1) / THREADS;
    __shared__ float shared[P];
    const int tid = threadIdx.x;
    const int factor = blockIdx.y;
    const int64_t base = int64_t(blockIdx.x) * n;
    const int64_t destination = (int64_t(factor) * batch + blockIdx.x) * n;
    float values[ITEMS];
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        const int index = tid + j * THREADS;
        const float raw = index < P ? input_value(x, base + index, input_dtype) : 0.0f;
        const float gain = pre_mode && index < P ? scale[factor * P + index] : 1.0f;
        float value = pre_mode ? __fmul_rn(raw, gain) : raw;
        if (pre_mode == 2 && P > 1) {
            const float other = __shfl_xor_sync(0xffffffffu, value, 1);
            value = __fmaf_rn((index & 1) ? -raw : raw, gain, other);
        }
        values[j] = value;
    }
    #pragma unroll
    for (int step = 1; step < 32 && step < P; step *= 2) {
        if (step == 1 && pre_mode == 2) continue;
        #pragma unroll
        for (int j = 0; j < ITEMS; ++j) {
            const float other = __shfl_xor_sync(0xffffffffu, values[j], step);
            values[j] = (tid & step) ? __fsub_rn(other, values[j])
                                    : __fadd_rn(values[j], other);
        }
    }
    #pragma unroll
    for (int step = 32; step < THREADS && step < P; step *= 2) {
        #pragma unroll
        for (int j = 0; j < ITEMS; ++j) {
            const int index = tid + j * THREADS;
            if (index < P) shared[index] = values[j];
        }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < ITEMS; ++j) {
            const int index = tid + j * THREADS;
            const float other = index < P ? shared[index ^ step] : 0.0f;
            values[j] = (tid & step) ? __fsub_rn(other, values[j])
                                    : __fadd_rn(values[j], other);
        }
        if (step * 2 < THREADS && step * 2 < P) __syncthreads();
    }
    #pragma unroll
    for (int step = THREADS; step < P; step *= 2) {
        #pragma unroll
        for (int j = 0; j < ITEMS; ++j) {
            if ((j & (step / THREADS)) == 0) {
                const int other = j ^ (step / THREADS);
                const float a = values[j], b = values[other];
                values[j] = __fadd_rn(a, b);
                values[other] = __fsub_rn(a, b);
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < ITEMS; ++j) {
        const int index = tid + j * THREADS;
        if (index >= P) continue;
        float value = __fmul_rn(values[j], normalizer);
        if (post) value = __fmul_rn(value, scale[factor * P + index]);
        output_value(y, destination + index, output_dtype, value, global, post, global_dtype);
    }
}


extern "C" int power_hadamard_grouped_input_run(
    const void* x, const float* scales, float* output,
    int groups, int batch, int input_dtype, float normalizer, uintptr_t stream) {
    if (!x || !scales || !output || groups < 1 || groups > 65535 || batch < 1 ||
        input_dtype < 0 || input_dtype > 2 ||
        uint64_t(groups) * uint64_t(batch) * 2048 * 4 > 4194304ULL ||
        (reinterpret_cast<uintptr_t>(x) & 15) ||
        (reinterpret_cast<uintptr_t>(scales) & 15) ||
        (reinterpret_cast<uintptr_t>(output) & 15))
        return int(cudaErrorInvalidValue);
    const dim3 grid(batch, groups);
    power_hadamard_grouped_input_kernel<2048, 256><<<grid, 256, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        x, scales, scales, output, 2048, batch, input_dtype, 0, 0, 2, false,
        normalizer);
    return int(cudaGetLastError());
}

// Component-only large-CTA resource. Existing exports and device math above
// remain untouched. This export is not selected by the serving runtime.
extern "C" int power_hadamard_grouped_input_1024_run(
    const void* x, const float* scales, float* output,
    int groups, int batch, int input_dtype, float normalizer, uintptr_t stream) {
    if (!x || !scales || !output || groups < 1 || groups > 65535 ||
        batch < 1 || batch > 8 || input_dtype < 0 || input_dtype > 1 ||
        uint64_t(groups) * uint64_t(batch) * 2048 * 4 > 4194304ULL ||
        (reinterpret_cast<uintptr_t>(x) & 15) ||
        (reinterpret_cast<uintptr_t>(scales) & 15) ||
        (reinterpret_cast<uintptr_t>(output) & 15))
        return int(cudaErrorInvalidValue);
    const dim3 grid(batch, groups);
    power_hadamard_grouped_input_kernel<2048, 1024><<<grid, 1024, 0,
        reinterpret_cast<cudaStream_t>(stream)>>>(
        x, scales, scales, output, 2048, batch, input_dtype, 0, 0, 2, false,
        normalizer);
    return int(cudaGetLastError());
}
