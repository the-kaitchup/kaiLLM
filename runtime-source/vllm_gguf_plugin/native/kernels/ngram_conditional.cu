// SPDX-License-Identifier: Apache-2.0
// ABI2: GPU ngram miss/hit selects retained Torch model child graphs.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <new>

struct NgramConditionalGraph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
};

__global__ void prepare_ngram_branch(cudaGraphConditionalHandle condition,
                                    const int32_t* valid_count,
                                    const int32_t* seq_len4,
                                    int32_t* seq_len1,
                                    int64_t* branch_counts) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const unsigned int hit = *valid_count != 0;
        // The outer fixed-query4 metadata includes three lookahead positions.
        // Only the separate query1 view changes; query4 metadata stays intact.
        *seq_len1 = *seq_len4 - 3;
        atomicAdd(reinterpret_cast<unsigned long long*>(branch_counts) + hit, 1ULL);
        cudaGraphSetConditional(condition, hit);
    }
}

static int fail(cudaError_t error, const char* operation,
                char* message, size_t size) {
    if (message && size) {
        std::snprintf(message, size, "%s: %s", operation, cudaGetErrorString(error));
    }
    return static_cast<int>(error);
}

static int validate_child(cudaGraph_t graph, char* message, size_t size,
                          unsigned int depth = 0) {
    if (depth > 16) {
        return fail(cudaErrorNotSupported, "conditional child nesting exceeds16", message, size);
    }
    size_t count = 0;
    cudaError_t status = cudaGraphGetNodes(graph, nullptr, &count);
    if (status != cudaSuccess) return fail(status, "cudaGraphGetNodes", message, size);
    auto* nodes = new (std::nothrow) cudaGraphNode_t[count];
    if (!nodes && count) {
        return fail(cudaErrorMemoryAllocation, "child node inspection", message, size);
    }
    status = cudaGraphGetNodes(graph, nodes, &count);
    if (status != cudaSuccess) {
        delete[] nodes;
        return fail(status, "cudaGraphGetNodes", message, size);
    }
    int result = 0;
    for (size_t i = 0; i < count && result == 0; ++i) {
        cudaGraphNodeType type;
        status = cudaGraphNodeGetType(nodes[i], &type);
        if (status != cudaSuccess) {
            result = fail(status, "cudaGraphNodeGetType", message, size);
            break;
        }
        switch (type) {
            case cudaGraphNodeTypeKernel:
            case cudaGraphNodeTypeMemset:
            case cudaGraphNodeTypeEmpty:
                break;
            case cudaGraphNodeTypeGraph: {
                cudaGraph_t child;
                status = cudaGraphChildGraphNodeGetGraph(nodes[i], &child);
                result = status == cudaSuccess
                    ? validate_child(child, message, size, depth + 1)
                    : fail(status, "cudaGraphChildGraphNodeGetGraph", message, size);
                break;
            }
            default:
                // Captured Torch memcpy nodes crashed conditional instantiation
                // on the validated CUDA13 target. Require kernel buffer copies.
                if (message && size) {
                    std::snprintf(message, size,
                        "unsupported conditional child node type%d; use kernel buffer copies",
                        static_cast<int>(type));
                }
                result = static_cast<int>(cudaErrorNotSupported);
        }
    }
    delete[] nodes;
    return result;
}

extern "C" int yaqa_ngram_conditional_abi_version() { return 2; }

extern "C" int yaqa_ngram_conditional_create(
    void* raw_m1, void* raw_m4, const int32_t* valid_count,
    const int32_t* seq_len4, int32_t* seq_len1, int64_t* branch_counts,
    void** output, char* message, size_t size) {
    if (output) *output = nullptr;
    if (!raw_m1 || !raw_m4 || raw_m1 == raw_m4 || !valid_count ||
        !seq_len4 || !seq_len1 || seq_len4 == seq_len1 ||
        !branch_counts || !output ||
        reinterpret_cast<uintptr_t>(valid_count) % alignof(int32_t) ||
        reinterpret_cast<uintptr_t>(seq_len4) % alignof(int32_t) ||
        reinterpret_cast<uintptr_t>(seq_len1) % alignof(int32_t) ||
        reinterpret_cast<uintptr_t>(branch_counts) % alignof(int64_t)) {
        return fail(cudaErrorInvalidValue, "invalid graph/control pointers", message, size);
    }
    int inspected = validate_child(reinterpret_cast<cudaGraph_t>(raw_m1), message, size);
    if (inspected) return inspected;
    inspected = validate_child(reinterpret_cast<cudaGraph_t>(raw_m4), message, size);
    if (inspected) return inspected;
    auto* context = new (std::nothrow) NgramConditionalGraph;
    if (!context) {
        return fail(cudaErrorMemoryAllocation, "host allocation", message, size);
    }
    cudaError_t status;
#define CHECK(call) do { status = (call); if (status != cudaSuccess) { \
    fail(status, #call, message, size); goto error; } } while (0)
    {
        cudaGraphConditionalHandle condition;
        cudaGraphNode_t setter, conditional, child;
        CHECK(cudaGraphCreate(&context->graph, 0));
        CHECK(cudaGraphConditionalHandleCreate(
            &condition, context->graph, 0, cudaGraphCondAssignDefault));
        void* arguments[] = {
            &condition, &valid_count, &seq_len4, &seq_len1, &branch_counts,
        };
        cudaKernelNodeParams setter_params{};
        setter_params.func = reinterpret_cast<void*>(prepare_ngram_branch);
        setter_params.gridDim = dim3(1);
        setter_params.blockDim = dim3(1);
        setter_params.kernelParams = arguments;
        CHECK(cudaGraphAddKernelNode(
            &setter, context->graph, nullptr, 0, &setter_params));
        cudaGraphNodeParams branch{};
        branch.type = cudaGraphNodeTypeConditional;
        branch.conditional.handle = condition;
        branch.conditional.type = cudaGraphCondTypeIf;
        branch.conditional.size = 2;
        CHECK(cudaGraphAddNode(
            &conditional, context->graph, &setter, nullptr, 1, &branch));
        CHECK(cudaGraphAddChildGraphNode(
            &child, branch.conditional.phGraph_out[0], nullptr, 0,
            reinterpret_cast<cudaGraph_t>(raw_m4)));
        CHECK(cudaGraphAddChildGraphNode(
            &child, branch.conditional.phGraph_out[1], nullptr, 0,
            reinterpret_cast<cudaGraph_t>(raw_m1)));
        CHECK(cudaGraphInstantiate(
            &context->executable, context->graph, nullptr, nullptr, 0));
    }
    *output = context;
    return static_cast<int>(cudaSuccess);
error:
    if (context->executable) cudaGraphExecDestroy(context->executable);
    if (context->graph) cudaGraphDestroy(context->graph);
    delete context;
    return static_cast<int>(status);
#undef CHECK
}

extern "C" int yaqa_ngram_conditional_launch(
    void* handle, void* raw_stream, char* message, size_t size) {
    if (!handle) return fail(cudaErrorInvalidValue, "null graph", message, size);
    auto* context = static_cast<NgramConditionalGraph*>(handle);
    cudaError_t status = cudaGraphLaunch(
        context->executable, reinterpret_cast<cudaStream_t>(raw_stream));
    return status == cudaSuccess ? 0 : fail(status, "cudaGraphLaunch", message, size);
}

extern "C" int yaqa_ngram_conditional_destroy(
    void* handle, char* message, size_t size) {
    if (!handle) return 0;
    auto* context = static_cast<NgramConditionalGraph*>(handle);
    cudaError_t first = cudaGraphExecDestroy(context->executable);
    cudaError_t second = cudaGraphDestroy(context->graph);
    delete context;
    if (first != cudaSuccess) {
        return fail(first, "cudaGraphExecDestroy", message, size);
    }
    return second == cudaSuccess ? 0 : fail(second, "cudaGraphDestroy", message, size);
}
