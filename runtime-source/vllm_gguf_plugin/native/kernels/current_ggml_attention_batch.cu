// SPDX-License-Identifier: Apache-2.0
// Diagnostic uniform-decode FA using llama.cpp's true multi-stream topology.
// Reuse the existing borrowed-stream context and Torch-owned scratch arena.
#include "current_ggml_attention.cu"

extern "C" int current_attention_batch_run(
    const float * q, const void * k, const void * v, const void * mask,
    float * output, void * scratch, size_t scratch_bytes,
    int device, int requests, int physical_kv, int logical_kv,
    float scale, uintptr_t stream
) {
    try {
        last_error.clear();
        const auto & info = ggml_cuda_info();
        if (device < 0 || device >= info.device_count ||
            info.devices[device].physical_device != device ||
            !q || !k || !v || !mask || !output || !scratch ||
            requests <= 0 || requests > 65535 || logical_kv <= 0 ||
            logical_kv % 256 || physical_kv < logical_kv || physical_kv % 256) {
            throw std::runtime_error("Native batched FA expects one query per stream and padded FP16 KV");
        }
        ggml_cuda_set_device(device);
        auto & context = get_context(device);
        std::lock_guard<std::mutex> lock(context.mutex);
        BorrowedStorage storage(context, scratch, scratch_bytes, stream);

        // Torch Q [B,1,8,256] -> GGML [256,1,8,B]. Physical KV
        // storage may exceed the common logical span exposed to attention.
        auto qt = headed(GGML_TYPE_F32, const_cast<float *>(q), 1, 8, 256);
        auto kt = headed(GGML_TYPE_F16, const_cast<void *>(k), logical_kv, 2, 256);
        auto vt = headed(GGML_TYPE_F16, const_cast<void *>(v), logical_kv, 2, 256);
        qt.ne[3] = kt.ne[3] = vt.ne[3] = requests;
        kt.nb[3] = vt.nb[3] = size_t(physical_kv) * 2 * 256 * sizeof(half);

        ggml_tensor mt{};
        mt.type = GGML_TYPE_F16;
        mt.data = const_cast<void *>(mask);
        mt.ne[0] = logical_kv;
        // The KV-max scan reads every row in its query tile, even when Q
        // has only one active row. Guard the largest native query tile with
        // -inf rows; this source revision's graph constructor does not pad it.
        mt.ne[1] = 64;
        mt.ne[2] = 1;
        mt.ne[3] = requests;
        mt.nb[0] = sizeof(half);
        mt.nb[1] = size_t(logical_kv) * sizeof(half);
        mt.nb[2] = mt.nb[3] = mt.nb[1] * mt.ne[1];

        ggml_tensor dst{};
        dst.type = GGML_TYPE_F32;
        dst.data = output;
        dst.ne[0] = 256;
        dst.ne[1] = 8;
        dst.ne[2] = 1;
        dst.ne[3] = requests;
        dst.nb[0] = sizeof(float);
        dst.nb[1] = 256 * sizeof(float);
        dst.nb[2] = dst.nb[3] = 8 * dst.nb[1];
        dst.op = GGML_OP_FLASH_ATTN_EXT;
        dst.src[0] = &qt;
        dst.src[1] = &kt;
        dst.src[2] = &vt;
        dst.src[3] = &mt;
        std::memcpy(dst.op_params, &scale, sizeof(scale));
        ggml_prec_set_acc(&dst, GGML_PREC_F32);
        if (!ggml_cuda_flash_attn_ext_supported(device, &dst)) {
            throw std::runtime_error("Local GGML backend rejects batched attention shape");
        }
        ggml_cuda_flash_attn_ext(context.cuda, &dst);
        const auto status = cudaGetLastError();
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        return 0;
    } catch (const std::exception & error) {
        last_error = error.what();
        return -1;
    }
}

// Separate ABI: callers of the established one-query decode export retain
// exactly the same tensor descriptions and native dispatch as before.
extern "C" int current_attention_batch_prefill_run(
    const float * q, const void * k, const void * v, const void * mask,
    float * output, void * scratch, size_t scratch_bytes,
    int device, int requests, int queries, int physical_kv, int logical_kv,
    float scale, uintptr_t stream
) {
    try {
        last_error.clear();
        const auto & info = ggml_cuda_info();
        if (device < 0 || device >= info.device_count ||
            info.devices[device].physical_device != device ||
            !q || !k || !v || !mask || !output || !scratch ||
            requests <= 1 || requests > 65535 || queries <= 1 ||
            queries > logical_kv || logical_kv <= 0 || logical_kv % 256 ||
            physical_kv < logical_kv || physical_kv % 256) {
            throw std::runtime_error("Native batched prefill expects B>1, uniform Q>1 and padded FP16 KV");
        }
        ggml_cuda_set_device(device);
        auto & context = get_context(device);
        std::lock_guard<std::mutex> lock(context.mutex);
        BorrowedStorage storage(context, scratch, scratch_bytes, stream);

        // Torch Q/out [B,Q,8,256], KV [B,physical_K,2,256].
        // Native query tiles are at most 64 rows; the KV-max scan also reads
        // the last tile's inactive query rows, which must contain -infinity.
        const int64_t mask_queries = (int64_t(queries) + 63) / 64 * 64;
        auto qt = headed(GGML_TYPE_F32, const_cast<float *>(q), queries, 8, 256);
        auto kt = headed(GGML_TYPE_F16, const_cast<void *>(k), logical_kv, 2, 256);
        auto vt = headed(GGML_TYPE_F16, const_cast<void *>(v), logical_kv, 2, 256);
        qt.ne[3] = kt.ne[3] = vt.ne[3] = requests;
        kt.nb[3] = vt.nb[3] = size_t(physical_kv) * 2 * 256 * sizeof(half);

        ggml_tensor mt{};
        mt.type = GGML_TYPE_F16;
        mt.data = const_cast<void *>(mask);
        mt.ne[0] = logical_kv;
        mt.ne[1] = mask_queries;
        mt.ne[2] = 1;
        mt.ne[3] = requests;
        mt.nb[0] = sizeof(half);
        mt.nb[1] = size_t(logical_kv) * sizeof(half);
        mt.nb[2] = mt.nb[3] = mt.nb[1] * mask_queries;

        ggml_tensor dst{};
        dst.type = GGML_TYPE_F32;
        dst.data = output;
        dst.ne[0] = 256;
        dst.ne[1] = 8;
        dst.ne[2] = queries;
        dst.ne[3] = requests;
        dst.nb[0] = sizeof(float);
        dst.nb[1] = 256 * sizeof(float);
        dst.nb[2] = 8 * dst.nb[1];
        dst.nb[3] = size_t(queries) * dst.nb[2];
        dst.op = GGML_OP_FLASH_ATTN_EXT;
        dst.src[0] = &qt;
        dst.src[1] = &kt;
        dst.src[2] = &vt;
        dst.src[3] = &mt;
        std::memcpy(dst.op_params, &scale, sizeof(scale));
        ggml_prec_set_acc(&dst, GGML_PREC_F32);
        if (!ggml_cuda_flash_attn_ext_supported(device, &dst)) {
            throw std::runtime_error("Local GGML backend rejects batched prefill shape");
        }
        ggml_cuda_flash_attn_ext(context.cuda, &dst);
        const auto status = cudaGetLastError();
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        return 0;
    } catch (const std::exception & error) {
        last_error = error.what();
        return -1;
    }
}
