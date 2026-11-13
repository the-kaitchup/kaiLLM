// SPDX-License-Identifier: Apache-2.0
// Diagnostic use of the exact local GGML FlashAttention implementation.
// All data/scratch belong to Torch, and launches borrow its current stream.
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/fattn.cuh"

#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace {
thread_local std::string last_error;

struct Arena final : ggml_cuda_pool {
    char * base = nullptr;
    size_t capacity = 0;
    size_t used = 0;
    void * alloc(size_t bytes, size_t * actual) override {
        *actual = (bytes + 255) & ~size_t(255);
        if (*actual > capacity - used) {
            throw std::runtime_error("Native attention scratch arena is too small");
        }
        void * pointer = base + used;
        used += *actual;
        return pointer;
    }
    void free(void * pointer, size_t bytes) override {
        GGML_ASSERT(bytes <= used && pointer == base + used - bytes);
        used -= bytes;
    }
};

struct Context {
    ggml_backend_cuda_context cuda;
    std::mutex mutex;
    explicit Context(int device) : cuda(device) {
        cuda.pools[device][0] = std::make_unique<Arena>();
    }
};

Context & get_context(int device) {
    static std::mutex mutex;
    static std::unordered_map<int, std::unique_ptr<Context>> contexts;
    std::lock_guard<std::mutex> lock(mutex);
    auto & result = contexts[device];
    if (!result) result = std::make_unique<Context>(device);
    return *result;
}

struct BorrowedStorage {
    Context & context;
    Arena & arena;
    BorrowedStorage(Context & context_, void * scratch, size_t bytes, uintptr_t stream)
        : context(context_), arena(static_cast<Arena &>(*context.cuda.pools[context.cuda.device][0])) {
        GGML_ASSERT(arena.used == 0 && context.cuda.streams[context.cuda.device][0] == nullptr);
        arena.base = static_cast<char *>(scratch);
        arena.capacity = bytes;
        context.cuda.streams[context.cuda.device][0] = stream ? reinterpret_cast<cudaStream_t>(stream) : cudaStreamLegacy;
    }
    ~BorrowedStorage() {
        GGML_ASSERT(arena.used == 0);
        arena.base = nullptr;
        arena.capacity = 0;
        context.cuda.streams[context.cuda.device][0] = nullptr;
    }
};

// Torch storage [tokens, heads, dim], GGML view [dim, tokens, heads, 1].
ggml_tensor headed(ggml_type type, void * data, int tokens, int heads, int dim) {
    ggml_tensor t{};
    t.type = type;
    t.data = data;
    t.ne[0] = dim;
    t.ne[1] = tokens;
    t.ne[2] = heads;
    t.ne[3] = 1;
    t.nb[0] = ggml_type_size(type);
    t.nb[1] = dim * heads * t.nb[0];
    t.nb[2] = dim * t.nb[0];
    t.nb[3] = tokens * t.nb[1];
    return t;
}
}  // namespace

extern "C" const char * current_attention_error() { return last_error.c_str(); }

static int attention_run(
    const float * q, const void * k, const void * v, const void * mask,
    float * output, void * scratch, size_t scratch_bytes,
    int device, int queries, int kv_tokens, int heads, int kv_heads, int dim,
    float scale, uintptr_t stream, int mask_stride
) {
    try {
        last_error.clear();
        const auto & info = ggml_cuda_info();
        if (device < 0 || device >= info.device_count ||
            info.devices[device].physical_device != device ||
            !q || !k || !v || !mask || !output || !scratch ||
            queries <= 0 || kv_tokens <= 0 || kv_tokens % 256 ||
            queries > kv_tokens || mask_stride < kv_tokens ||
            mask_stride % 256 || heads != 8 || kv_heads != 2 || dim != 256) {
            throw std::runtime_error("Native attention diagnostic expects B1 Qwen3.5 8/2 heads, D256, KV padded to256");
        }
        ggml_cuda_set_device(device);
        auto & context = get_context(device);
        std::lock_guard<std::mutex> lock(context.mutex);
        BorrowedStorage storage(context, scratch, scratch_bytes, stream);
        auto qt = headed(GGML_TYPE_F32, const_cast<float *>(q), queries, heads, dim);
        auto kt = headed(GGML_TYPE_F16, const_cast<void *>(k), kv_tokens, kv_heads, dim);
        auto vt = headed(GGML_TYPE_F16, const_cast<void *>(v), kv_tokens, kv_heads, dim);
        ggml_tensor mt{};
        mt.type = GGML_TYPE_F16;
        mt.data = const_cast<void *>(mask);
        mt.ne[0] = kv_tokens;
        mt.ne[1] = queries;
        mt.ne[2] = mt.ne[3] = 1;
        mt.nb[0] = sizeof(half);
        mt.nb[1] = mask_stride * sizeof(half);
        mt.nb[2] = mt.nb[3] = mt.nb[1] * queries;
        ggml_tensor dst{};
        dst.type = GGML_TYPE_F32;
        dst.data = output;
        dst.ne[0] = dim;
        dst.ne[1] = heads;
        dst.ne[2] = queries;
        dst.ne[3] = 1;
        dst.nb[0] = sizeof(float);
        dst.nb[1] = dim * sizeof(float);
        dst.nb[2] = heads * dst.nb[1];
        dst.nb[3] = queries * dst.nb[2];
        dst.op = GGML_OP_FLASH_ATTN_EXT;
        dst.src[0] = &qt;
        dst.src[1] = &kt;
        dst.src[2] = &vt;
        dst.src[3] = &mt;
        std::memcpy(dst.op_params, &scale, sizeof(scale));
        if (!ggml_cuda_flash_attn_ext_supported(device, &dst)) {
            throw std::runtime_error("Local GGML backend rejects attention shape");
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

extern "C" int current_attention_run(
    const float * q, const void * k, const void * v, const void * mask,
    float * output, void * scratch, size_t scratch_bytes,
    int device, int queries, int kv_tokens, int heads, int kv_heads, int dim,
    float scale, uintptr_t stream
) {
    return attention_run(q, k, v, mask, output, scratch, scratch_bytes,
                         device, queries, kv_tokens, heads, kv_heads, dim,
                         scale, stream, kv_tokens);
}

static int attention_many_run(
    const float * q, const void * k, const void * v, const void * mask,
    float * output, void * scratch, size_t scratch_bytes,
    int device, int requests, int max_queries, int max_kv, int mask_columns,
    const int32_t * query_counts, const int32_t * kv_tokens,
    float scale, uintptr_t stream
) {
    // The arrays above are CPU launch metadata, not device tensors. Preserve
    // each B1 launch's tile/stream-K partition. A single native 4-D batch would
    // change that partition through ntiles_dst *= batch_size.
    if (requests <= 0 || max_queries <= 0 || max_kv <= 0 ||
        max_kv % 256 || mask_columns <= 0 || mask_columns % 256 ||
        !query_counts || !kv_tokens) {
        last_error = "Invalid batched native attention launch metadata";
        return -1;
    }
    const size_t query_stride = size_t(max_queries) * 8 * 256;
    const size_t kv_stride = size_t(max_kv) * 2 * 256;
    const size_t mask_stride = size_t(max_queries) * mask_columns;
    for (int request = 0; request < requests; ++request) {
        const int nq = query_counts[request];
        const int nk = kv_tokens[request];
        if (nq <= 0 || nq > max_queries || nk <= 0 || nk > max_kv ||
            nk > mask_columns || nk % 256) {
            last_error = "Batched native attention request exceeds allocated shape";
            return -1;
        }
        const int status = attention_run(
            q + request * query_stride,
            static_cast<const half *>(k) + request * kv_stride,
            static_cast<const half *>(v) + request * kv_stride,
            static_cast<const half *>(mask) + request * mask_stride,
            output + request * query_stride, scratch, scratch_bytes,
            device, nq, nk, 8, 2, 256, scale, stream, mask_columns);
        if (status) return status;
        // Every request uses the same stream. Its scratch use finishes before
        // the next request overwrites it; no synchronization is necessary.
    }
    return 0;
}

extern "C" int current_attention_many_run(
    const float * q, const void * k, const void * v, const void * mask,
    float * output, void * scratch, size_t scratch_bytes,
    int device, int requests, int max_queries, int max_kv,
    const int32_t * query_counts, const int32_t * kv_tokens,
    float scale, uintptr_t stream
) {
    return attention_many_run(q, k, v, mask, output, scratch, scratch_bytes,
        device, requests, max_queries, max_kv, max_kv, query_counts, kv_tokens,
        scale, stream);
}

extern "C" int current_attention_many_strided_run(
    const float * q, const void * k, const void * v, const void * mask,
    float * output, void * scratch, size_t scratch_bytes,
    int device, int requests, int max_queries, int max_kv, int mask_columns,
    const int32_t * query_counts, const int32_t * kv_tokens,
    float scale, uintptr_t stream
) {
    return attention_many_run(q, k, v, mask, output, scratch, scratch_bytes,
        device, requests, max_queries, max_kv, mask_columns, query_counts,
        kv_tokens, scale, stream);
}
