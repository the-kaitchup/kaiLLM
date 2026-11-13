// SPDX-License-Identifier: Apache-2.0
// Diagnostic bridge to the locally built current GGML CUDA backend. No copied
// quantization arithmetic: GGML builds and executes the MUL_MAT node itself.
#include <cuda_runtime_api.h>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {
thread_local std::string last_error;

void check_cuda(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

struct DeviceGuard {
    int previous;
    explicit DeviceGuard(int device) {
        check_cuda(cudaGetDevice(&previous));
        check_cuda(cudaSetDevice(device));
    }
    ~DeviceGuard() { cudaSetDevice(previous); }
};

struct Backend {
    ggml_backend_t backend;
    std::mutex mutex;
    explicit Backend(int device) : backend(ggml_backend_cuda_init(device)) {
        if (!backend) throw std::runtime_error("ggml_backend_cuda_init failed");
    }
    ~Backend() { ggml_backend_free(backend); }
};

std::shared_ptr<Backend> get_backend(int device) {
    static std::mutex mutex;
    static std::unordered_map<int, std::weak_ptr<Backend>> backends;
    std::lock_guard<std::mutex> lock(mutex);
    auto result = backends[device].lock();
    if (!result) {
        result = std::make_shared<Backend>(device);
        backends[device] = result;
    }
    return result;
}

ggml_context * make_context() {
    // A small graph plus two or three tensor headers; all data are allocated
    // separately by the CUDA backend.
    ggml_init_params params = {
        ggml_graph_overhead_custom(16, false) + 8 * ggml_tensor_overhead(),
        nullptr,
        true,
    };
    auto * context = ggml_init(params);
    if (!context) throw std::runtime_error("ggml_init failed");
    return context;
}

struct Graph {
    ggml_context * context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
    ~Graph() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (context) ggml_free(context);
    }
};

struct Linear {
    int device;
    int64_t n, k;
    std::shared_ptr<Backend> backend;
    ggml_context * context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * weight = nullptr;
    std::map<std::pair<int64_t, int64_t>, std::unique_ptr<Graph>> graphs;

    Linear(int device_, int64_t n_, int64_t k_)
        : device(device_), n(n_), k(k_), backend(get_backend(device_)) {}

    ~Linear() {
        // cudaFree synchronizes outstanding work. Calls finish all D2D copies
        // before returning; handles can therefore release their storage here.
        cudaSetDevice(device);
        graphs.clear();
        if (buffer) ggml_backend_buffer_free(buffer);
        if (context) ggml_free(context);
    }

    Graph & get_graph(int64_t m, int64_t channels = 1) {
        auto & result = graphs[{m, channels}];
        if (result) return *result;
        auto next = std::make_unique<Graph>();
        next->context = make_context();
        next->input = channels == 1
            ? ggml_new_tensor_2d(next->context, GGML_TYPE_F32, k, m)
            : ggml_new_tensor_3d(next->context, GGML_TYPE_F32, k, m, channels);
        ggml_set_name(next->input, "bridge_input_f32");
        ggml_set_input(next->input);
        next->output = ggml_mul_mat(next->context, weight, next->input);
        ggml_set_name(next->output, "bridge_mul_mat_f32");
        ggml_set_output(next->output);
        next->graph = ggml_new_graph_custom(next->context, 16, false);
        ggml_build_forward_expand(next->graph, next->output);
        next->buffer = ggml_backend_alloc_ctx_tensors(next->context, backend->backend);
        if (!next->buffer) throw std::runtime_error("GGML graph CUDA allocation failed");
        if (!ggml_backend_supports_op(backend->backend, next->output)) {
            throw std::runtime_error("Current GGML CUDA does not support this MUL_MAT");
        }
        result = std::move(next);
        return *result;
    }
};
}  // namespace

extern "C" const char * current_ggml_error() { return last_error.c_str(); }

extern "C" void * current_ggml_create(
    const void * data, size_t bytes, int quant_type, int64_t n, int64_t k,
    int device, bool source_cuda, uintptr_t torch_stream
) {
    try {
        last_error.clear();
        if (!data || n <= 0 || k <= 0) throw std::runtime_error("Invalid matrix arguments");
        if (quant_type != 0 && quant_type != 1 && quant_type != 8 &&
            quant_type != 12 && quant_type != 13 && quant_type != 14 &&
            quant_type != 18 && quant_type != 21 && quant_type != 22 && quant_type != 23) {
            throw std::runtime_error("Unsupported current GGML oracle type");
        }
        DeviceGuard guard(device);
        auto linear = std::make_unique<Linear>(device, n, k);
        std::lock_guard<std::mutex> lock(linear->backend->mutex);
        linear->context = make_context();
        linear->weight = ggml_new_tensor_2d(
            linear->context, static_cast<ggml_type>(quant_type), k, n);
        if (ggml_nbytes(linear->weight) != bytes) {
            throw std::runtime_error("GGUF byte count disagrees with shape/type");
        }
        ggml_set_name(linear->weight, "bridge_weight");
        linear->buffer = ggml_backend_alloc_ctx_tensors(linear->context, linear->backend->backend);
        if (!linear->buffer) throw std::runtime_error("GGML weight CUDA allocation failed");
        ggml_backend_buffer_set_usage(linear->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        if (source_cuda) {
            const auto stream = reinterpret_cast<cudaStream_t>(torch_stream);
            check_cuda(cudaMemcpyAsync(linear->weight->data, data, bytes, cudaMemcpyDeviceToDevice, stream));
            check_cuda(cudaStreamSynchronize(stream));
        } else {
            ggml_backend_tensor_set(linear->weight, data, 0, bytes);
        }
        return linear.release();
    } catch (const std::exception & error) {
        last_error = error.what();
        return nullptr;
    }
}

static int forward_impl(
    void * handle, const float * x, float * y, int64_t m, int64_t channels,
    uintptr_t torch_stream
) {
    try {
        last_error.clear();
        if (!handle || !x || !y || m <= 0 || channels <= 0)
            throw std::runtime_error("Invalid forward arguments");
        auto & linear = *static_cast<Linear *>(handle);
        DeviceGuard guard(linear.device);
        std::lock_guard<std::mutex> lock(linear.backend->mutex);
        auto & graph = linear.get_graph(m, channels);
        const auto stream = reinterpret_cast<cudaStream_t>(torch_stream);
        // Respect the caller's producer stream, then transfer ownership of the
        // dependency to GGML's private stream without any host tensor copies.
        check_cuda(cudaMemcpyAsync(graph.input->data, x, ggml_nbytes(graph.input), cudaMemcpyDeviceToDevice, stream));
        check_cuda(cudaStreamSynchronize(stream));
        const auto status = ggml_backend_graph_compute(linear.backend->backend, graph.graph);
        if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("GGML CUDA graph execution failed");
        // graph_compute synchronizes the backend. Complete the outgoing copy
        // too, so another caller cannot overwrite this cached graph output.
        check_cuda(cudaMemcpyAsync(y, graph.output->data, ggml_nbytes(graph.output), cudaMemcpyDeviceToDevice, stream));
        check_cuda(cudaStreamSynchronize(stream));
        return 0;
    } catch (const std::exception & error) {
        last_error = error.what();
        return -1;
    }
}

extern "C" int current_ggml_forward(
    void * handle, const float * x, float * y, int64_t m, uintptr_t torch_stream
) {
    return forward_impl(handle, x, y, m, 1, torch_stream);
}

extern "C" int current_ggml_forward_channels(
    void * handle, const float * x, float * y, int64_t tokens, int64_t channels,
    uintptr_t torch_stream
) {
    return forward_impl(handle, x, y, tokens, channels, torch_stream);
}

extern "C" void current_ggml_destroy(void * handle) {
    delete static_cast<Linear *>(handle);
}
