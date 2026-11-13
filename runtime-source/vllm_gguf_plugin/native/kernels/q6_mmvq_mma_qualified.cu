// SPDX-License-Identifier: MIT
// GGML layout/launch portions: Copyright (c) 2023-2026 The ggml authors.
// See LLAMA_CPP_LICENSE.txt for the upstream notice and permission terms.
// Experimental Q6_K MMQ using the exact current MMVQ Q8_1 activation blocks.
// The native direct bridge is deliberately unchanged and remains the oracle.
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/quantize.cuh"

#include <climits>
#include <stdexcept>
#include <unordered_map>

namespace {
thread_local std::string last_error;
size_t aligned(size_t n) { return (n + 255) & ~size_t(255); }

struct Arena final : ggml_cuda_pool {
    char * base = nullptr;
    size_t capacity = 0;
    size_t used = 0;
    void * alloc(size_t bytes, size_t * actual) override {
        *actual = aligned(bytes);
        if (*actual > capacity - used) throw std::runtime_error("Q6 MMA scratch exhausted");
        void * result = base + used;
        used += *actual;
        return result;
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

Context & context_for(int device) {
    static std::mutex mutex;
    static std::unordered_map<int, std::unique_ptr<Context>> contexts;
    std::lock_guard<std::mutex> lock(mutex);
    auto & result = contexts[device];
    if (!result) result = std::make_unique<Context>(device);
    return *result;
}

void validate(int device, int64_t n, int64_t k, int64_t m) {
    const auto & info = ggml_cuda_info();
    if (device < 0 || device >= info.device_count ||
        info.devices[device].physical_device != device) {
        throw std::runtime_error("Q6 MMA requires a real CUDA device");
    }
    if (n <= 0 || k <= 0 || m <= 0 || k % QK_K ||
        n > INT_MAX || k > INT_MAX || m > INT_MAX) {
        throw std::runtime_error("Invalid Q6 MMA dimensions");
    }
}

struct Sizes { size_t q8, mmq, fixup; };
Sizes sizes(int device, int64_t n, int64_t k, int64_t m) {
    const auto & dev = ggml_cuda_info().devices[device];
    const bool fallback = n % 128 != 0;
    const size_t q8 = m * GGML_PAD(k, MATRIX_ROW_PADDING) / QK8_1 * sizeof(block_q8_1);
    int best_j = 0, best_tiles = INT_MAX;
    for (int j = 8; j <= 128 && best_tiles > 1; j += 8) {
        const auto config = ggml_cuda_mmq_get_config(GGML_TYPE_Q6_K, j, fallback, dev.cc);
        if (config.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(config, dev.cc) > dev.smpbo) continue;
        const int tiles = (m + j - 1) / j;
        if (tiles < best_tiles) { best_j = j; best_tiles = tiles; }
    }
    if (!best_j) throw std::runtime_error("No Q6 MMQ tile supports this shape/device");
    const auto config = ggml_cuda_mmq_get_config(GGML_TYPE_Q6_K, best_j, fallback, dev.cc);
    // Native dispatch never chooses MMQ below eight rows, where J_max returns
    // zero. Forced MMA still loads a complete J=8 tile, so reserve its guard
    // explicitly instead of relying on unrelated Torch allocator padding.
    const int guard = std::max(config.J, ggml_cuda_mmq_get_J_max(GGML_TYPE_Q6_K, fallback, dev.cc, m));
    const size_t mmq = q8 + guard * sizeof(block_q8_1_mmq);
    size_t fixup = 0;
    if (config.stream_k) {
        const int64_t tiles = ((n + config.I - 1) / config.I) * best_tiles;
        const int64_t waves = (tiles + dev.nsm - 1) / dev.nsm;
        const int64_t efficiency = 100 * tiles / (dev.nsm * waves);
        const int64_t blocks = GGML_CUDA_CC_IS_NVIDIA(dev.cc) && efficiency >= 90 ? tiles : dev.nsm;
        if (tiles % blocks) fixup = blocks * config.J * config.I * sizeof(float);
    }
    return {q8, mmq, fixup};
}

// Four row-major Q8_1 blocks become one transposed MMQ block. Codes remain
// bit-identical, and each FP16 scale is widened exactly to the D4 FP32 field.
// Invalid MMQ guard blocks are zeroed by this same kernel.
__global__ void repack_q8(const block_q8_1 * src, uint32_t * dst,
                        int64_t m, int64_t padded_k, int64_t total_words) {
    const int64_t word = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (word >= total_words) return;
    const int64_t block = word / 36;
    const int lane = word % 36;
    if (block >= m * padded_k / QK8_1_MMQ) { dst[word] = 0; return; }
    const int64_t row = block % m, kb = block / m;
    if (lane < 4) {
        dst[word] = __float_as_uint(__low2float(src[row * padded_k / QK8_1 + kb * 4 + lane].ds));
    } else {
        const int iq = lane - 4;
        const auto & q = src[row * padded_k / QK8_1 + kb * 4 + iq / 8];
        dst[word] = reinterpret_cast<const uint32_t *>(q.qs)[iq % 8];
    }
}

void pack(const float * x, void * q8, void * mmq, int64_t k, int64_t m,
          size_t mmq_bytes, cudaStream_t stream) {
    const int64_t padded = GGML_PAD(k, MATRIX_ROW_PADDING);
    quantize_row_q8_1_cuda(x, nullptr, q8, GGML_TYPE_Q6_K,
                          k, k, m*k, m*k, padded, m, 1, 1, stream);
    const int64_t words = mmq_bytes / sizeof(uint32_t);
    repack_q8<<<(words + 255) / 256, 256, 0, stream>>>
        (static_cast<block_q8_1 *>(q8), static_cast<uint32_t *>(mmq), m, padded, words);
}

struct Borrow {
    Context & ctx;
    Arena & arena;
    Borrow(Context & c, void * scratch, size_t bytes, cudaStream_t stream)
        : ctx(c), arena(static_cast<Arena &>(*c.cuda.pools[c.cuda.device][0])) {
        GGML_ASSERT(arena.used == 0 && ctx.cuda.streams[ctx.cuda.device][0] == nullptr);
        arena.base = static_cast<char *>(scratch);
        arena.capacity = bytes;
        ctx.cuda.streams[ctx.cuda.device][0] = stream;
    }
    ~Borrow() {
        GGML_ASSERT(arena.used == 0);
        arena.base = nullptr; arena.capacity = 0;
        ctx.cuda.streams[ctx.cuda.device][0] = nullptr;
    }
};

void check_launch() {
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
} // namespace

extern "C" const char * q6_mma_error() { return last_error.c_str(); }
extern "C" int q6_mma_prepare(int device) {
    try {
        last_error.clear(); validate(device, 128, 256, 1); context_for(device); return 0;
    } catch (const std::exception & e) { last_error = e.what(); return -1; }
}
extern "C" size_t q6_mma_workspace(int device, int64_t n, int64_t k, int64_t m) {
    try {
        last_error.clear(); validate(device, n, k, m);
        const auto s = sizes(device, n, k, m);
        return aligned(s.q8) + aligned(s.mmq) + aligned(s.fixup);
    } catch (const std::exception & e) { last_error = e.what(); return 0; }
}
extern "C" int q6_mma_pack(const float * x, void * q8, void * mmq,
                           int device, int64_t k, int64_t m, size_t mmq_bytes, uintptr_t stream_raw) {
    try {
        last_error.clear(); validate(device, 128, k, m); ggml_cuda_set_device(device);
        if (!x || !q8 || !mmq || mmq_bytes % sizeof(block_q8_1_mmq) ||
            mmq_bytes < size_t(m * GGML_PAD(k, MATRIX_ROW_PADDING) / QK8_1_MMQ * sizeof(block_q8_1_mmq)))
            throw std::runtime_error("Invalid activation buffers");
        pack(x, q8, mmq, k, m, mmq_bytes, stream_raw ? reinterpret_cast<cudaStream_t>(stream_raw) : cudaStreamLegacy);
        check_launch(); return 0;
    } catch (const std::exception & e) { last_error = e.what(); return -1; }
}
extern "C" int q6_mma_run(const void * weight, const float * x, float * y,
                          void * scratch, size_t bytes, int device,
                          int64_t n, int64_t k, int64_t m, uintptr_t stream_raw) {
    try {
        last_error.clear(); validate(device, n, k, m);
        const auto s = sizes(device, n, k, m);
        if (!weight || !x || !y || !scratch || bytes < aligned(s.q8) + aligned(s.mmq) + aligned(s.fixup))
            throw std::runtime_error("Invalid Q6 MMA storage");
        auto & ctx = context_for(device);
        std::lock_guard<std::mutex> lock(ctx.mutex);
        ggml_cuda_set_device(device);
        const cudaStream_t stream = stream_raw ? reinterpret_cast<cudaStream_t>(stream_raw) : cudaStreamLegacy;
        Borrow borrow(ctx, scratch, bytes, stream);
        ggml_cuda_pool_alloc<char> q8(ctx.cuda.pool(), s.q8);
        ggml_cuda_pool_alloc<char> mmq(ctx.cuda.pool(), s.mmq);
        pack(x, q8.ptr, mmq.ptr, k, m, s.mmq, stream);
        const int64_t sy = m * GGML_PAD(k, MATRIX_ROW_PADDING) * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const mmq_args args = {
            static_cast<const char *>(weight), GGML_TYPE_Q6_K, reinterpret_cast<const int *>(mmq.ptr), nullptr, nullptr, y, nullptr,
            k, n, m, k/QK_K, m, n,
            1, 1, n*k/QK_K, sy, n*m,
            1, 1, n*k/QK_K, sy, n*m,
            m, m};
        mul_mat_q_case<GGML_TYPE_Q6_K, GGML_PREC_Q8>(ctx.cuda, args, stream);
        check_launch(); return 0;
    } catch (const std::exception & e) { last_error = e.what(); return -1; }
}
