// SPDX-License-Identifier: Apache-2.0
// Reuse the diagnostic context and stream helpers in this separate library.
// The exported norm entry points are unused here; only rope_run is called.
#include "current_ggml_norm.cu"
#include "ggml-cuda/rope.cuh"

extern "C" int current_rope_run(
        void * input, void * positions, void * output, int tokens, int heads,
        int dim, int rotary, float base, int context_length, int device,
        uintptr_t stream) {
    try {
        auto & context = context_for(device);
        std::lock_guard<std::mutex> lock(context.mutex);
        Stream borrowed(context.cuda, stream);
        auto x = header(dim, heads, input);
        x.ne[2] = tokens;
        x.nb[3] = x.nb[2] * tokens;
        auto y = x;
        y.data = output;
        auto p = header(tokens * 4, 1, positions);
        p.type = GGML_TYPE_I32;
        y.op = GGML_OP_ROPE;
        y.src[0] = &x; y.src[1] = &p;
        y.op_params[1] = rotary;
        y.op_params[2] = GGML_ROPE_TYPE_IMROPE;
        y.op_params[4] = context_length;
        float values[] = {base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f};
        memcpy(y.op_params + 5, values, sizeof(values));
        // Text positions are identical along all three axes. These sections
        // are the source Qwen3.5-0.8B metadata and are checked by Python.
        int sections[] = {11, 11, 10, 0};
        memcpy(y.op_params + 11, sections, sizeof(sections));
        ggml_cuda_op_rope(context.cuda, &y);
        auto error = cudaGetLastError();
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
        return 0;
    } catch (const std::exception & error) {
        error_message = error.what(); return -1;
    }
}
