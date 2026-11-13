// SPDX-License-Identifier: Apache-2.0 AND MIT
// GGML argument-layout portions: Copyright (c) 2023-2026 The ggml authors.
// See LLAMA_CPP_LICENSE.txt for the upstream notice and permission terms.
// Experimental direct launch of the exact locally built GGML CUDA kernels.
// Tensor storage and scratch belong to Torch. No CUDA allocation/copy/sync is
// performed by a launch; the current Torch stream is borrowed temporarily.
#include "ggml-backend-impl.h"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/mmf.cuh"
#include "ggml-cuda/mmvf.cuh"
#include "ggml-cuda/quantize.cuh"
#include "iq_native_m8.cuh"
#include "native_iq_m1.cuh"
#include "native_q4_m1.cuh"
#include "native_head_m1.cuh"
#include "ssm_channel_interleave.cuh"
#include "native_ssm_fused_channels.cuh"
#include "native_head_m8.cuh"
#include "native_body_m8.cuh"

#include <array>
#include <climits>
#include <stdexcept>
#include <unordered_map>

namespace {
thread_local std::string last_error;

size_t align_size(size_t bytes) { return (bytes + 255) & ~size_t(255); }

struct Arena final : ggml_cuda_pool {
    char * base = nullptr;
    size_t capacity = 0;
    size_t used = 0;
    void * alloc(size_t bytes, size_t * actual) override {
        *actual = align_size(bytes);
        if (*actual > capacity - used) {
            throw std::runtime_error("Torch GGML scratch arena is too small");
        }
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

void validate(int device, int type, int64_t n, int64_t k, int64_t m) {
    const auto & info = ggml_cuda_info();
    if (device < 0 || device >= info.device_count ||
        info.devices[device].physical_device != device) {
        throw std::runtime_error("Direct bridge requires a real CUDA device index");
    }
    const bool q3_head = type == GGML_TYPE_Q3_K &&
        info.devices[device].cc == 1200 && n == 128000 && k == 2048 &&
        m > 0;
    if (type != GGML_TYPE_Q8_0 && !q3_head && type != GGML_TYPE_Q4_K &&
        type != GGML_TYPE_Q5_K && type != GGML_TYPE_Q6_K &&
        type != GGML_TYPE_IQ2_S && type != GGML_TYPE_IQ3_S &&
        type != GGML_TYPE_IQ3_XXS && type != GGML_TYPE_IQ4_XS) {
        throw std::runtime_error("Unsupported direct GGML quantization type");
    }
    if (n <= 0 || k <= 0 || m <= 0 || k % ggml_blck_size(ggml_type(type))) {
        throw std::runtime_error("Invalid direct GGML matrix shape");
    }
    if (n > INT_MAX || k > INT_MAX || m > INT_MAX) {
        throw std::runtime_error("Direct bridge shape exceeds CUDA kernel index range");
    }
}

size_t workspace_size(int device, ggml_type type, int64_t n, int64_t k, int64_t m,
                      int64_t channels = 1) {
    const auto & dev = ggml_cuda_info().devices[device];
    const int64_t padded_k = GGML_PAD(k, MATRIX_ROW_PADDING);
    size_t bytes = channels * m * padded_k * sizeof(block_q8_1) / QK8_1;
    if (ggml_cuda_should_use_mmvq(type, dev.cc, m)) {
        return align_size(bytes) + (current_native_heads::eligible(dev.cc, type, n, k, m)
            ? align_size(current_native_heads::activation_bytes(k)) : 0);
    }
    if (!ggml_cuda_should_use_mmq(type, dev.cc, m, 0)) {
        throw std::runtime_error("Current GGML would not select MMVQ/MMQ for this shape");
    }
    const bool fallback = n % 128 != 0;
    bytes += ggml_cuda_mmq_get_J_max(type, fallback, dev.cc, m) * sizeof(block_q8_1_mmq);

    // Match the library's host tile selection to size its temporary stream-K
    // fixup buffer; all arithmetic kernels remain inside the shared library.
    int best_j = 0;
    int best_tiles = INT_MAX;
    for (int j = 8; j <= 128 && best_tiles > 1; j += 8) {
        const auto config = ggml_cuda_mmq_get_config(type, j, fallback, dev.cc);
        if (config.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(config, dev.cc) > dev.smpbo) continue;
        const int tiles = (m + j - 1) / j;
        if (tiles < best_tiles) {
            best_j = j;
            best_tiles = tiles;
        }
    }
    if (!best_j) throw std::runtime_error("No current MMQ tile supports this shape/device");
    const auto config = ggml_cuda_mmq_get_config(type, best_j, fallback, dev.cc);
    size_t fixup = 0;
    if (config.stream_k) {
        const int64_t tiles = ((n + config.I - 1) / config.I) * best_tiles * channels;
        const int64_t waves = (tiles + dev.nsm - 1) / dev.nsm;
        const int64_t efficiency = 100 * tiles / (dev.nsm * waves);
        const int64_t blocks = GGML_CUDA_CC_IS_NVIDIA(dev.cc) && efficiency >= 90 ? tiles : dev.nsm;
        if (tiles % blocks) fixup = blocks * config.J * config.I * sizeof(float);
    }
    return align_size(bytes) + align_size(fixup);
}

ggml_tensor tensor_header(ggml_type type, int64_t width, int64_t rows, void * data) {
    ggml_tensor result{};
    result.type = type;
    result.ne[0] = width;
    result.ne[1] = rows;
    result.ne[2] = result.ne[3] = 1;
    result.nb[0] = ggml_type_size(type);
    result.nb[1] = ggml_row_size(type, width);
    result.nb[2] = result.nb[3] = result.nb[1] * rows;
    result.data = data;
    return result;
}

struct BorrowedStorage {
    Context & context;
    Arena & arena;
    BorrowedStorage(Context & context_, void * scratch, size_t bytes, uintptr_t stream)
        : context(context_), arena(static_cast<Arena &>(*context.cuda.pools[context.cuda.device][0])) {
        GGML_ASSERT(arena.used == 0 && context.cuda.streams[context.cuda.device][0] == nullptr);
        arena.base = static_cast<char *>(scratch);
        arena.capacity = bytes;
        // ctx.stream() treats nullptr as an instruction to create its own
        // stream; the explicit CUDA legacy sentinel denotes Torch stream 0.
        context.cuda.streams[context.cuda.device][0] = stream ? reinterpret_cast<cudaStream_t>(stream) : cudaStreamLegacy;
    }
    ~BorrowedStorage() {
        GGML_ASSERT(arena.used == 0);
        arena.base = nullptr;
        arena.capacity = 0;
        context.cuda.streams[context.cuda.device][0] = nullptr;
    }
};
}  // namespace

extern "C" const char * current_direct_error() { return last_error.c_str(); }

extern "C" int current_direct_prepare(int device) {
    try {
        last_error.clear();
        validate(device, GGML_TYPE_Q6_K, 1, 256, 1);
        context_for(device);
        return 0;
    } catch (const std::exception & error) {
        last_error = error.what();
        return -1;
    }
}

extern "C" size_t current_direct_workspace(int device, int type, int64_t n, int64_t k, int64_t m) {
    try {
        last_error.clear();
        validate(device, type, n, k, m);
        return workspace_size(device, ggml_type(type), n, k, m);
    } catch (const std::exception & error) {
        last_error = error.what();
        return 0;
    }
}

extern "C" int current_direct_run(
    const void * packed, const float * input, float * output, void * scratch,
    size_t scratch_bytes, int device, int type, int64_t n, int64_t k, int64_t m,
    uintptr_t stream
) {
    try {
        last_error.clear();
        validate(device, type, n, k, m);
        if (!packed || !input || !output || !scratch ||
            scratch_bytes < workspace_size(device, ggml_type(type), n, k, m)) {
            throw std::runtime_error("Invalid direct GGML pointers or scratch capacity");
        }
        auto & context = context_for(device);
        std::lock_guard<std::mutex> lock(context.mutex);
        ggml_cuda_set_device(device);
        BorrowedStorage storage(context, scratch, scratch_bytes, stream);
        auto weight = tensor_header(ggml_type(type), k, n, const_cast<void *>(packed));
        auto x = tensor_header(GGML_TYPE_F32, k, m, const_cast<float *>(input));
        auto y = tensor_header(GGML_TYPE_F32, n, m, output);
        ggml_backend_buffer weight_buffer{};
        weight_buffer.usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        weight.buffer = &weight_buffer;
        y.op = GGML_OP_MUL_MAT;
        y.src[0] = &weight;
        y.src[1] = &x;
        const int cc = ggml_cuda_info().devices[device].cc;
        if (gguf_native_iq_m1::eligible(cc, weight.type, n, k, m) ||
            gguf_native_q4_m1::eligible(cc, weight.type, n, k, m) ||
            gguf_native_head_m1::eligible(cc, weight.type, n, k, m) ||
            current_iq_m8::eligible(cc, weight.type, n, k, m) ||
            current_native_heads::eligible(cc, weight.type, n, k, m) ||
            gguf_native_body::eligible(cc, weight.type, n, k, m)) {
            // Use the original native quantizer and exact lane/reduction tree.
            // Scratch already contains room for the native MMVQ Q8_1 blocks.
            const cudaStream_t cuda_stream = context.cuda.stream();
            const int64_t padded = GGML_PAD(k, MATRIX_ROW_PADDING);
            quantize_row_q8_1_cuda(input, nullptr, scratch, weight.type,
                k, k, m*k, m*k, padded, m, 1, 1, cuda_stream);
            if (gguf_native_iq_m1::eligible(cc, weight.type, n, k, m)) {
                gguf_native_iq_m1::run(weight.type, packed, scratch, output, n, k, n, cuda_stream);
            } else if (gguf_native_q4_m1::eligible(cc, weight.type, n, k, m)) {
                gguf_native_q4_m1::run(packed, scratch, output, cuda_stream);
            } else if (gguf_native_head_m1::eligible(cc, weight.type, n, k, m)) {
                gguf_native_head_m1::run(packed, scratch, output, n, cuda_stream);
            } else if (current_native_heads::eligible(cc, weight.type, n, k, m)) {
                void * activation = static_cast<char *>(scratch) +
                    align_size(m * padded * sizeof(block_q8_1) / QK8_1);
                current_native_heads::run_packed(
                    weight.type, packed, scratch, activation, output, n, k, n, cuda_stream);
            } else if (gguf_native_body::eligible(cc, weight.type, n, k, m)) {
                gguf_native_body::run(weight.type, packed, scratch, output, n, k, n, cuda_stream);
            } else {
                current_iq_m8::run(weight.type, packed, scratch, output, n, n, cuda_stream);
            }
        } else if (ggml_cuda_should_use_mmvq(weight.type, cc, m)) {
            ggml_cuda_mul_mat_vec_q(context.cuda, &weight, &x, nullptr, &y);
        } else {
            ggml_cuda_mul_mat_q(context.cuda, &weight, &x, nullptr, &y);
        }
        const auto status = cudaGetLastError();
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        return 0;
    } catch (const std::exception & error) {
        last_error = error.what();
        return -1;
    }
}

namespace {
void validate_channels(int device, int type, int64_t n, int64_t k,
                       int64_t tokens, int64_t channels) {
    validate(device, type, n, k, tokens);
    if (channels <= 0 || channels > 65535 || tokens > INT_MAX / channels) {
        throw std::runtime_error("Invalid GGML token/channel dimensions");
    }
}
}  // namespace

extern "C" size_t current_channels_workspace(
    int device, int type, int64_t n, int64_t k, int64_t tokens, int64_t channels
) {
    try {
        last_error.clear();
        validate_channels(device, type, n, k, tokens, channels);
        return workspace_size(device, ggml_type(type), n, k, tokens, channels);
    } catch (const std::exception & error) {
        last_error = error.what();
        return 0;
    }
}

// Qwen3.5 SSM-out retains [K, tokens_per_request, requests] in llama.cpp.
// In particular dispatch and the MMQ tile width use tokens_per_request, not
// the flattened token count. Weights broadcast over the request channels.
extern "C" int current_channels_run(
    const void * packed, const float * input, float * output, void * scratch,
    size_t scratch_bytes, int device, int type, int64_t n, int64_t k,
    int64_t tokens, int64_t channels, uintptr_t stream
) {
    try {
        last_error.clear();
        validate_channels(device, type, n, k, tokens, channels);
        if (!packed || !input || !output || !scratch ||
            scratch_bytes < workspace_size(device, ggml_type(type), n, k, tokens, channels)) {
            throw std::runtime_error("Invalid channel GGML pointers or scratch capacity");
        }
        auto & context = context_for(device);
        std::lock_guard<std::mutex> lock(context.mutex);
        ggml_cuda_set_device(device);
        BorrowedStorage storage(context, scratch, scratch_bytes, stream);
        auto weight = tensor_header(ggml_type(type), k, n, const_cast<void *>(packed));
        auto x = tensor_header(GGML_TYPE_F32, k, tokens, const_cast<float *>(input));
        auto y = tensor_header(GGML_TYPE_F32, n, tokens, output);
        x.ne[2] = y.ne[2] = channels;
        x.nb[3] *= channels;
        y.nb[3] *= channels;
        ggml_backend_buffer weight_buffer{};
        weight_buffer.usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        weight.buffer = &weight_buffer;
        y.op = GGML_OP_MUL_MAT;
        y.src[0] = &weight;
        y.src[1] = &x;
        const int cc = ggml_cuda_info().devices[device].cc;
        if (gguf_native_ssm_fused::selected(cc, weight.type, n, k, tokens, channels)) {
            gguf_native_ssm_fused::run(
                packed, input, output, scratch, weight.type, int(n), int(channels), context.cuda.stream());
        } else if (gguf_ssm_channel_interleave::selected(cc, weight.type, n, k, tokens, channels)) {
            gguf_ssm_channel_interleave::run(
                packed, input, output, scratch, int(n), int(k), int(channels), context.cuda.stream());
        } else if (ggml_cuda_should_use_mmvq(weight.type, cc, tokens)) {
            ggml_cuda_mul_mat_vec_q(context.cuda, &weight, &x, nullptr, &y);
        } else {
            ggml_cuda_mul_mat_q(context.cuda, &weight, &x, nullptr, &y);
        }
        const auto status = cudaGetLastError();
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        return 0;
    } catch (const std::exception & error) {
        last_error = error.what();
        return -1;
    }
}

// Gate/normalization fusion may produce the unchanged native row-major Q8_1
// bytes directly. SSM channels still use the original T1 four-warp tree;
// treating them as a flat M8 projection would change floating-point results.
extern "C" int current_prequant_run(
    const void * packed, const void * q8, float * output,
    int device, int type, int64_t n, int64_t k, int64_t m,
    int channels, uintptr_t stream_raw
) {
    try {
        last_error.clear();
        validate(device, type, n, k, m);
        const int cc = ggml_cuda_info().devices[device].cc;
        if (!packed || !q8 || !output || cc != 860 || m > 8 ||
            (channels != 0 && channels != 1) ||
            !ggml_cuda_should_use_mmvq(ggml_type(type), cc, channels ? 1 : m)) {
            throw std::runtime_error("Unsupported native prequantized projection");
        }
        auto & context = context_for(device);
        std::lock_guard<std::mutex> lock(context.mutex);
        ggml_cuda_set_device(device);
        const cudaStream_t stream = stream_raw ? reinterpret_cast<cudaStream_t>(stream_raw) : cudaStreamLegacy;
        if (channels && m > 1) {
            if (!gguf_native_ssm_fused::selected(cc, ggml_type(type), n, k, 1, m)) {
                throw std::runtime_error("Unsupported prequantized SSM shape");
            }
            gguf_native_ssm_fused::run_q8(packed, q8, output, ggml_type(type), int(n), int(m), stream);
        } else if (gguf_native_iq_m1::eligible(cc, ggml_type(type), n, k, m)) {
            gguf_native_iq_m1::run(ggml_type(type), packed, q8, output, n, k, n, stream);
        } else if (gguf_native_q4_m1::eligible(cc, ggml_type(type), n, k, m)) {
            gguf_native_q4_m1::run(packed, q8, output, stream);
        } else if (gguf_native_head_m1::eligible(cc, ggml_type(type), n, k, m)) {
            gguf_native_head_m1::run(packed, q8, output, n, stream);
        } else if (current_native_heads::eligible(cc, ggml_type(type), n, k, m)) {
            current_native_heads::run(ggml_type(type), packed, q8, output, n, k, n, stream);
        } else if (gguf_native_body::eligible(cc, ggml_type(type), n, k, m)) {
            gguf_native_body::run(ggml_type(type), packed, q8, output, n, k, n, stream);
        } else if (current_iq_m8::eligible(cc, ggml_type(type), n, k, m)) {
            current_iq_m8::run(ggml_type(type), packed, q8, output, n, n, stream);
        } else {
            auto weight = tensor_header(ggml_type(type), k, n, const_cast<void *>(packed));
            auto input = tensor_header(GGML_TYPE_F32, k, m, nullptr);
            auto destination = tensor_header(GGML_TYPE_F32, n, m, output);
            ggml_cuda_op_mul_mat_vec_q(context.cuda, &weight, &input, &destination,
                static_cast<const char *>(packed), nullptr, static_cast<const char *>(q8), output,
                0, n, m, GGML_PAD(k, MATRIX_ROW_PADDING), stream);
        }
        const auto status = cudaGetLastError();
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        return 0;
    } catch (const std::exception & error) {
        last_error = error.what();
        return -1;
    }
}

namespace {
// Shared activation layouts preserve native quantization and original matrix
// dimensions. MMVQ uses row-major half ds for all supported weight types;
// MMQ uses transposed FP32 D4 or half DS4 and must never mix the two.
int shared_layout(int device, ggml_type type, int64_t m) {
    const int cc = ggml_cuda_info().devices[device].cc;
    if (ggml_cuda_should_use_mmvq(type, cc, m)) return 0;
    return mmq_get_q8_1_ds_layout(type) == MMQ_Q8_1_DS_LAYOUT_D4 ? 1 : 2;
}

struct SharedSizes {
    std::array<size_t, 3> quant{};
    std::array<ggml_type, 3> representative{};
    size_t fixup = 0;
    int64_t output_rows = 0;
    size_t prefix() const { return align_size(quant[0]) + align_size(quant[1]) + align_size(quant[2]); }
    size_t total() const { return prefix() + fixup; }
};

SharedSizes shared_sizes(int device, const int * types, const int64_t * rows,
                        int count, int64_t k, int64_t m) {
    if (!types || !rows || count < 1 || count > 32)
        throw std::runtime_error("Expected between one and 32 original GGUF matrices");
    SharedSizes result;
    for (int i = 0; i < count; ++i) {
        validate(device, types[i], rows[i], k, m);
        const auto type = ggml_type(types[i]);
        const int layout = shared_layout(device, type, m);
        size_t bytes = m * GGML_PAD(k, MATRIX_ROW_PADDING) * sizeof(block_q8_1) / QK8_1;
        if (layout) {
            const int cc = ggml_cuda_info().devices[device].cc;
            bytes += ggml_cuda_mmq_get_J_max(type, rows[i] % 128 != 0, cc, m) * sizeof(block_q8_1_mmq);
        }
        result.quant[layout] = std::max(result.quant[layout], bytes);
        result.representative[layout] = type;
        result.fixup = std::max(result.fixup, workspace_size(device, type, rows[i], k, m) - align_size(bytes));
        if (!layout && rows[i] % 4) {
            // Native MMVQ bounds row stores against its destination stride.
            // An incomplete row group cannot write directly into a larger
            // merged stride: its padding would overwrite the next slice/row.
            result.fixup = std::max(result.fixup, align_size(m * rows[i] * sizeof(float)));
        }
        result.output_rows += rows[i];
        if (result.output_rows > INT_MAX)
            throw std::runtime_error("Grouped output width exceeds CUDA index range");
    }
    return result;
}

void shared_mmq(ggml_backend_cuda_context & context, const mmq_args & args, cudaStream_t stream) {
    // These are the same exported instantiations invoked by native MMQ.
    switch (args.type_x) {
        case GGML_TYPE_Q8_0: mul_mat_q_case<GGML_TYPE_Q8_0>(context, args, stream); break;
        case GGML_TYPE_Q3_K: mul_mat_q_case<GGML_TYPE_Q3_K>(context, args, stream); break;
        case GGML_TYPE_Q4_K: mul_mat_q_case<GGML_TYPE_Q4_K>(context, args, stream); break;
        case GGML_TYPE_Q5_K: mul_mat_q_case<GGML_TYPE_Q5_K>(context, args, stream); break;
        case GGML_TYPE_Q6_K: mul_mat_q_case<GGML_TYPE_Q6_K>(context, args, stream); break;
        case GGML_TYPE_IQ2_S: mul_mat_q_case<GGML_TYPE_IQ2_S>(context, args, stream); break;
        case GGML_TYPE_IQ3_S: mul_mat_q_case<GGML_TYPE_IQ3_S>(context, args, stream); break;
        case GGML_TYPE_IQ3_XXS: mul_mat_q_case<GGML_TYPE_IQ3_XXS>(context, args, stream); break;
        case GGML_TYPE_IQ4_XS: mul_mat_q_case<GGML_TYPE_IQ4_XS>(context, args, stream); break;
        default: throw std::runtime_error("Unsupported shared MMQ matrix type");
    }
}
}  // namespace

extern "C" int current_shared_layout(int device, int type, int64_t m) {
    try {
        last_error.clear();
        validate(device, type, type == GGML_TYPE_Q3_K ? 128000 : 128,
                 type == GGML_TYPE_Q3_K ? 2048 : 256, m);
        return shared_layout(device, ggml_type(type), m);
    } catch (const std::exception & error) { last_error = error.what(); return -1; }
}

extern "C" size_t current_shared_workspace(int device, const int * types, const int64_t * rows,
                                           int count, int64_t k, int64_t m) {
    try {
        last_error.clear(); return shared_sizes(device, types, rows, count, k, m).total();
    } catch (const std::exception & error) { last_error = error.what(); return 0; }
}

extern "C" int current_shared_run(const void * const * weights, const int * types, const int64_t * rows,
                                   int count, const float * x, float * y, void * scratch,
                                   size_t scratch_bytes, int device, int64_t k, int64_t m, uintptr_t stream_raw) {
    try {
        last_error.clear();
        const auto sizes = shared_sizes(device, types, rows, count, k, m);
        if (!weights || !x || !y || !scratch || scratch_bytes < sizes.total())
            throw std::runtime_error("Invalid shared GGML tensor or workspace storage");
        for (int i = 0; i < count; ++i) if (!weights[i]) throw std::runtime_error("Null packed matrix");
        auto & context = context_for(device);
        std::lock_guard<std::mutex> lock(context.mutex);
        ggml_cuda_set_device(device);
        const cudaStream_t stream = stream_raw ? reinterpret_cast<cudaStream_t>(stream_raw) : cudaStreamLegacy;
        BorrowedStorage storage(context, static_cast<char *>(scratch) + sizes.prefix(), sizes.fixup, stream_raw);
        std::array<char *, 3> quant{};
        size_t offset = 0;
        const int64_t padded = GGML_PAD(k, MATRIX_ROW_PADDING);
        for (int layout = 0; layout < 3; ++layout) {
            if (!sizes.quant[layout]) continue;
            quant[layout] = static_cast<char *>(scratch) + offset;
            offset += align_size(sizes.quant[layout]);
            const quantize_cuda_t quantize = layout ? quantize_mmq_q8_1_cuda : quantize_row_q8_1_cuda;
            quantize(x, nullptr, quant[layout], sizes.representative[layout],
                     k, k, m*k, m*k, padded, m, 1, 1, stream);
        }
        const int64_t sy = m * padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t total = sizes.output_rows;
        int64_t first_row = 0;
        for (int i = 0; i < count; ++i) {
            const auto type = ggml_type(types[i]);
            const int64_t n = rows[i], row_blocks = k / ggml_blck_size(type);
            const int layout = shared_layout(device, type, m);
            float * output = y + first_row;
            if (!layout && gguf_native_iq_m1::eligible(
                    ggml_cuda_info().devices[device].cc, type, n, k, m)) {
                gguf_native_iq_m1::run(type, weights[i], quant[layout], output, n, k, total, stream);
            } else if (!layout && gguf_native_q4_m1::eligible(
                    ggml_cuda_info().devices[device].cc, type, n, k, m)) {
                gguf_native_q4_m1::run(weights[i], quant[layout], output, stream);
            } else if (!layout && gguf_native_head_m1::eligible(
                    ggml_cuda_info().devices[device].cc, type, n, k, m)) {
                gguf_native_head_m1::run(weights[i], quant[layout], output, n, stream);
            } else if (!layout && current_native_heads::eligible(
                    ggml_cuda_info().devices[device].cc, type, n, k, m)) {
                current_native_heads::run_packed(type, weights[i], quant[layout],
                    static_cast<char *>(scratch) + sizes.prefix(), output, n, k, total, stream);
            } else if (!layout && gguf_native_body::eligible(
                    ggml_cuda_info().devices[device].cc, type, n, k, m)) {
                gguf_native_body::run(type, weights[i], quant[layout], output, n, k, total, stream);
            } else if (!layout && current_iq_m8::eligible(
                    ggml_cuda_info().devices[device].cc, type, n, k, m)) {
                // The shared Q8 blocks are unchanged. Keep original N and
                // write into this matrix's slice using the merged row stride.
                current_iq_m8::run(type, weights[i], quant[layout], output, n, total, stream);
            } else if (!layout) {
                auto weight = tensor_header(type, k, n, const_cast<void *>(weights[i]));
                auto input = tensor_header(GGML_TYPE_F32, k, m, const_cast<float *>(x));
                const bool compact = n % 4 != 0;
                float * target = compact
                    ? reinterpret_cast<float *>(static_cast<char *>(scratch) + sizes.prefix()) : output;
                auto destination = tensor_header(GGML_TYPE_F32, compact ? n : total, m, target);
                ggml_cuda_op_mul_mat_vec_q(context.cuda, &weight, &input, &destination,
                    static_cast<const char *>(weights[i]), x, quant[layout], target,
                    0, n, m, padded, stream);
                if (compact) {
                    const auto copy_status = cudaMemcpy2DAsync(output, total * sizeof(float),
                        target, n * sizeof(float), n * sizeof(float), m, cudaMemcpyDeviceToDevice, stream);
                    if (copy_status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(copy_status));
                }
            } else {
                const mmq_args args = {
                    static_cast<const char *>(weights[i]), type, reinterpret_cast<const int *>(quant[layout]),
                    nullptr, nullptr, output, nullptr,
                    k, n, m, row_blocks, m, total,
                    1, 1, n*row_blocks, sy, total*m,
                    1, 1, n*row_blocks, sy, total*m, m, m};
                shared_mmq(context.cuda, args, stream);
            }
            first_row += n;
        }
        const auto status = cudaGetLastError();
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        return 0;
    } catch (const std::exception & error) { last_error = error.what(); return -1; }
}

namespace {
void check_cublas(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("GGML float cuBLAS status " + std::to_string(int(status)));
    }
}

int float_route(int device, int64_t n, int64_t k, int64_t m) {
    validate(device, GGML_TYPE_Q8_0, 1, 32, 1);
    if (n <= 0 || k <= 0 || m <= 0 || n > INT_MAX || k > INT_MAX || m > INT_MAX) {
        throw std::runtime_error("Invalid FP32 matrix dimensions");
    }
    const auto & dev = ggml_cuda_info().devices[device];
    const auto weight = tensor_header(GGML_TYPE_F32, k, n, nullptr);
    if (ggml_cuda_should_use_mmvf(weight.type, dev.cc, weight.ne, weight.nb, m)) return 1;
    const auto input = tensor_header(GGML_TYPE_F32, k, m, nullptr);
    if (n == 1 && m > MMVF_MAX_BATCH_SIZE &&
        ggml_cuda_should_use_mmvf(input.type, dev.cc, input.ne, input.nb, 1)) return 2;
    if (ggml_cuda_should_use_mmf(weight.type, dev.cc, dev.warp_size, weight.ne, weight.nb, m, false)) return 3;
    // This bridge only reproduces the default contiguous F32 cuBLAS branch.
    // Do not silently ignore an environment override to F16/BF16 arithmetic.
    const char * precision = getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
    if (precision) {
        std::string value(precision);
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
        if (value != "auto" && value != "f32" && value != "fp32") {
            throw std::runtime_error("Float bridge requires default/F32 cuBLAS compute type");
        }
    }
    return 4;
}

size_t float_workspace(int device, int route) {
    if (route != 4) return 0;
    return ggml_cuda_info().devices[device].cc >= GGML_CUDA_CC_HOPPER ? 32*1024*1024 : 4*1024*1024;
}
}  // namespace

extern "C" int current_float_prepare(int device) {
    try {
        last_error.clear();
        float_route(device, 16, 1024, 1);
        auto & context = context_for(device);
        std::lock_guard<std::mutex> lock(context.mutex);
        ggml_cuda_set_device(device);
        auto & handle = context.cuda.cublas_handles[device][0];
        if (!handle) {
            check_cublas(cublasCreate(&handle));
            check_cublas(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
        }
        return 0;
    } catch (const std::exception & error) {
        last_error = error.what();
        return -1;
    }
}

extern "C" int current_float_route(int device, int64_t n, int64_t k, int64_t m) {
    try {
        last_error.clear();
        return float_route(device, n, k, m);
    } catch (const std::exception & error) {
        last_error = error.what();
        return -1;
    }
}

extern "C" size_t current_float_workspace(int device, int route) {
    return float_workspace(device, route);
}

extern "C" int current_float_run(
    const float * weight_data, const float * input, float * output,
    void * scratch, size_t scratch_bytes, int device, int64_t n, int64_t k, int64_t m,
    uintptr_t stream
) {
    try {
        last_error.clear();
        const int route = float_route(device, n, k, m);
        if (!weight_data || !input || !output || scratch_bytes < float_workspace(device, route)) {
            throw std::runtime_error("Invalid float GGML pointers/workspace");
        }
        auto & context = context_for(device);
        std::lock_guard<std::mutex> lock(context.mutex);
        ggml_cuda_set_device(device);
        BorrowedStorage storage(context, scratch, scratch_bytes, stream);
        auto weight = tensor_header(GGML_TYPE_F32, k, n, const_cast<float *>(weight_data));
        auto x = tensor_header(GGML_TYPE_F32, k, m, const_cast<float *>(input));
        auto y = tensor_header(GGML_TYPE_F32, n, m, output);
        y.op = GGML_OP_MUL_MAT;
        y.src[0] = &weight;
        y.src[1] = &x;
        if (route == 1) {
            ggml_cuda_mul_mat_vec_f(context.cuda, &weight, &x, nullptr, &y);
        } else if (route == 2) {
            auto vector_output = tensor_header(GGML_TYPE_F32, m, 1, output);
            ggml_cuda_mul_mat_vec_f(context.cuda, &x, &weight, nullptr, &vector_output);
        } else if (route == 3) {
            ggml_cuda_mul_mat_f(context.cuda, &weight, &x, nullptr, &y);
        } else {
            auto handle = context.cuda.cublas_handles[device][0];
            if (!handle) throw std::runtime_error("Initialize float GGML module before launch/capture");
            // Exact dense F32 branch of ggml_cuda_mul_mat_cublas_impl: same
            // handle math mode, workspace size, transpose, dimensions, strides,
            // alpha and beta. Its wrapper itself is a private static symbol.
            check_cublas(cublasSetStream(handle, context.cuda.stream()));
            check_cublas(cublasSetWorkspace(handle, scratch, scratch_bytes));
            const float alpha = 1.0f;
            const float beta = 0.0f;
            check_cublas(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                n, m, k, &alpha, weight_data, k, input, k, &beta, output, n));
        }
        const auto status = cudaGetLastError();
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        return 0;
    } catch (const std::exception & error) {
        last_error = error.what();
        return -1;
    }
}

namespace {
// Preserve the original F16 GGUF storage and the native type-dependent
// dispatch. Converting the weights to F32 changes both MMVF selection and the
// cuBLAS compute/output dtype, even when the stored values remain exact.
int half_route(int device, int64_t n, int64_t k, int64_t m) {
    validate(device, GGML_TYPE_Q8_0, 1, 32, 1);
    if (n <= 0 || k <= 0 || m <= 0 || n > INT_MAX || k > INT_MAX || m > INT_MAX) {
        throw std::runtime_error("Invalid F16 matrix dimensions");
    }
    const auto & dev = ggml_cuda_info().devices[device];
    const auto weight = tensor_header(GGML_TYPE_F16, k, n, nullptr);
    if (ggml_cuda_should_use_mmvf(weight.type, dev.cc, weight.ne, weight.nb, m)) return 1;
    if (ggml_cuda_should_use_mmf(weight.type, dev.cc, dev.warp_size, weight.ne, weight.nb, m, false)) return 3;
    if (!GGML_CUDA_CC_IS_NVIDIA(dev.cc) || !fast_fp16_hardware_available(dev.cc)) {
        throw std::runtime_error("F16 bridge cuBLAS branch requires NVIDIA fast-F16 hardware");
    }
    const char * precision = getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
    if (precision) {
        std::string value(precision);
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
        if (value != "auto" && value != "f16" && value != "fp16") {
            throw std::runtime_error("F16 bridge requires default/F16 cuBLAS compute type");
        }
    }
    return 4;
}

size_t half_workspace(int device, int64_t n, int64_t k, int64_t m) {
    if (half_route(device, n, k, m) != 4) return 0;
    return float_workspace(device, 4) + align_size(m*k*sizeof(half)) + align_size(m*n*sizeof(half));
}
}  // namespace

extern "C" int current_half_route(int device, int64_t n, int64_t k, int64_t m) {
    try {
        last_error.clear();
        return half_route(device, n, k, m);
    } catch (const std::exception & error) {
        last_error = error.what();
        return -1;
    }
}

extern "C" size_t current_half_workspace(int device, int64_t n, int64_t k, int64_t m) {
    try {
        last_error.clear();
        return half_workspace(device, n, k, m);
    } catch (const std::exception & error) {
        last_error = error.what();
        return 0;
    }
}

extern "C" int current_half_run(
    const void * weight_data, const float * input, float * output,
    void * scratch, size_t scratch_bytes, int device, int64_t n, int64_t k, int64_t m,
    uintptr_t stream
) {
    try {
        last_error.clear();
        const int route = half_route(device, n, k, m);
        if (!weight_data || !input || !output || scratch_bytes < half_workspace(device, n, k, m)) {
            throw std::runtime_error("Invalid half GGML pointers/workspace");
        }
        auto & context = context_for(device);
        std::lock_guard<std::mutex> lock(context.mutex);
        ggml_cuda_set_device(device);
        BorrowedStorage storage(context, scratch, scratch_bytes, stream);
        auto weight = tensor_header(GGML_TYPE_F16, k, n, const_cast<void *>(weight_data));
        auto x = tensor_header(GGML_TYPE_F32, k, m, const_cast<float *>(input));
        auto y = tensor_header(GGML_TYPE_F32, n, m, output);
        y.op = GGML_OP_MUL_MAT;
        y.src[0] = &weight;
        y.src[1] = &x;
        if (route == 1) {
            ggml_cuda_mul_mat_vec_f(context.cuda, &weight, &x, nullptr, &y);
        } else if (route == 3) {
            ggml_cuda_mul_mat_f(context.cuda, &weight, &x, nullptr, &y);
        } else {
            auto handle = context.cuda.cublas_handles[device][0];
            if (!handle) throw std::runtime_error("Initialize half GGML module before launch/capture");
            const size_t cublas_bytes = float_workspace(device, 4);
            auto * input_half = reinterpret_cast<half *>(static_cast<char *>(scratch) + cublas_bytes);
            auto * output_half = reinterpret_cast<half *>(reinterpret_cast<char *>(input_half) + align_size(m*k*sizeof(half)));
            const auto cuda_stream = context.cuda.stream();
            ggml_get_to_fp16_cuda(GGML_TYPE_F32)(input, input_half, m*k, cuda_stream);
            check_cublas(cublasSetStream(handle, cuda_stream));
            check_cublas(cublasSetWorkspace(handle, scratch, cublas_bytes));
            // Match ggml_cuda_mul_mat_cublas_impl<F16>: Volta keeps F32
            // output/accumulation; other tested NVIDIA architectures use F16
            // compute and output, followed by the native F16->F32 conversion.
            if (ggml_cuda_info().devices[device].cc == GGML_CUDA_CC_VOLTA) {
                const float alpha = 1.0f, beta = 0.0f;
                check_cublas(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    n, m, k, &alpha, weight_data, CUDA_R_16F, k,
                    input_half, CUDA_R_16F, k, &beta, output, CUDA_R_32F, n,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            } else {
                const half alpha = __float2half(1.0f), beta = __float2half(0.0f);
                check_cublas(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    n, m, k, &alpha, weight_data, CUDA_R_16F, k,
                    input_half, CUDA_R_16F, k, &beta, output_half, CUDA_R_16F, n,
                    CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                ggml_get_to_fp32_cuda(GGML_TYPE_F16)(output_half, output, m*n, cuda_stream);
            }
        }
        const auto status = cudaGetLastError();
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        return 0;
    } catch (const std::exception & error) {
        last_error = error.what();
        return -1;
    }
}
