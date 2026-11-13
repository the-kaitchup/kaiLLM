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
#include "kai_dual_mmvq.cuh"
#include "kai_prepare_ext.cuh"

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
    // Every listed type has GGML MMVQ and MMQ kernels with exported MMQ
    // instantiations; the legacy 32-element formats share the generic paths.
    if (type != GGML_TYPE_Q4_0 && type != GGML_TYPE_Q4_1 &&
        type != GGML_TYPE_Q5_0 && type != GGML_TYPE_Q5_1 &&
        type != GGML_TYPE_Q8_0 && type != GGML_TYPE_Q2_K &&
        type != GGML_TYPE_Q3_K && type != GGML_TYPE_Q4_K &&
        type != GGML_TYPE_Q5_K && type != GGML_TYPE_Q6_K &&
        type != GGML_TYPE_IQ2_XXS && type != GGML_TYPE_IQ2_XS &&
        type != GGML_TYPE_IQ2_S && type != GGML_TYPE_IQ3_XXS &&
        type != GGML_TYPE_IQ3_S && type != GGML_TYPE_IQ1_S &&
        type != GGML_TYPE_IQ4_NL && type != GGML_TYPE_IQ4_XS) {
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
        case GGML_TYPE_Q4_0: mul_mat_q_case<GGML_TYPE_Q4_0>(context, args, stream); break;
        case GGML_TYPE_Q4_1: mul_mat_q_case<GGML_TYPE_Q4_1>(context, args, stream); break;
        case GGML_TYPE_Q5_0: mul_mat_q_case<GGML_TYPE_Q5_0>(context, args, stream); break;
        case GGML_TYPE_Q5_1: mul_mat_q_case<GGML_TYPE_Q5_1>(context, args, stream); break;
        case GGML_TYPE_Q8_0: mul_mat_q_case<GGML_TYPE_Q8_0>(context, args, stream); break;
        case GGML_TYPE_Q2_K: mul_mat_q_case<GGML_TYPE_Q2_K>(context, args, stream); break;
        case GGML_TYPE_Q3_K: mul_mat_q_case<GGML_TYPE_Q3_K>(context, args, stream); break;
        case GGML_TYPE_Q4_K: mul_mat_q_case<GGML_TYPE_Q4_K>(context, args, stream); break;
        case GGML_TYPE_IQ2_XXS: mul_mat_q_case<GGML_TYPE_IQ2_XXS>(context, args, stream); break;
        case GGML_TYPE_IQ2_XS: mul_mat_q_case<GGML_TYPE_IQ2_XS>(context, args, stream); break;
        case GGML_TYPE_IQ1_S: mul_mat_q_case<GGML_TYPE_IQ1_S>(context, args, stream); break;
        case GGML_TYPE_IQ4_NL: mul_mat_q_case<GGML_TYPE_IQ4_NL>(context, args, stream); break;
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
        last_error.clear(); validate(device, type, 128, 256, m);
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

// ---------------------------------------------------------------------------
// Fused activation preparation + MMVQ (kai_dual_mmvq.cuh). Activation storage:
// [pass1: m*(k/32) block_q8_1][pass2 (optional): same][inverse: m floats].
namespace {
struct FusedLayout {
    block_q8_1 * y1;
    block_q8_1 * y2;
    float * inverse;
};

FusedLayout fused_layout(void * activation, int64_t m, int64_t k, int two_pass) {
    char * base = static_cast<char *>(activation);
    const size_t pass = kai_dual::pass_bytes(m, k);
    return {reinterpret_cast<block_q8_1 *>(base),
            two_pass ? reinterpret_cast<block_q8_1 *>(base + pass) : nullptr,
            reinterpret_cast<float *>(base + pass * (two_pass ? 2 : 1))};
}

bool fused_shape_ok(int type, int64_t n, int64_t k, int64_t m) {
    return kai_dual::type_supported(type) && n > 0 && n <= INT_MAX && k > 0 && k <= INT_MAX &&
           m >= 1 && m <= kai_dual::MAX_COLS && k % QK8_1 == 0 && k / QK8_1 <= kai_dual::PREP_THREADS &&
           k % ggml_blck_size(ggml_type(type)) == 0;
}
}  // namespace

extern "C" int kai_fused_supported(int type, int64_t n, int64_t k, int64_t m) {
    return fused_shape_ok(type, n, k, m) ? 1 : 0;
}

extern "C" size_t kai_fused_activation_bytes(int64_t m, int64_t k, int two_pass) {
    return kai_dual::pass_bytes(m, k) * (two_pass ? 2 : 1) + kai_dual::align256(size_t(m) * sizeof(float));
}

extern "C" int kai_fused_prepare(const void * x, int x_dtype, int64_t m, int64_t k, int64_t stride_x,
                                 void * activation, int two_pass, uintptr_t stream_raw) {
    try {
        last_error.clear();
        if (!x || !activation || m < 1 || m > INT_MAX || k <= 0 || k % QK8_1 || k > INT_MAX || stride_x < k) {
            throw std::runtime_error("Invalid fused activation preparation arguments");
        }
        const cudaStream_t stream = stream_raw ? reinterpret_cast<cudaStream_t>(stream_raw) : cudaStreamLegacy;
        const auto layout = fused_layout(activation, m, k, two_pass);
        switch (x_dtype) {
            case 0: kai_dual::prepare(static_cast<const float *>(x), m, k, stride_x, layout.y1, layout.y2, layout.inverse, stream); break;
            case 1: kai_dual::prepare(static_cast<const nv_bfloat16 *>(x), m, k, stride_x, layout.y1, layout.y2, layout.inverse, stream); break;
            case 2: kai_dual::prepare(static_cast<const half *>(x), m, k, stride_x, layout.y1, layout.y2, layout.inverse, stream); break;
            default: throw std::runtime_error("Fused activations must be F32, BF16 or F16");
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
template <typename F>
int fused_entry(F && body) {
    try {
        last_error.clear();
        body();
        const auto status = cudaGetLastError();
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        return 0;
    } catch (const std::exception & error) {
        last_error = error.what();
        return -1;
    }
}
cudaStream_t stream_of(uintptr_t raw) { return raw ? reinterpret_cast<cudaStream_t>(raw) : cudaStreamLegacy; }
}  // namespace

// Single-pass activation of x[..., view]: stored column c reads x column ((c/s)%a)*b*s + (c/(a*s))*s + c%s.
extern "C" int kai_fused_prepare_view(const void * x, int x_dtype, int64_t m, int64_t k, int64_t stride_x,
                                      void * activation, int64_t a, int64_t b, int64_t s, uintptr_t stream_raw) {
    return fused_entry([&] {
        if (!x || !activation || m < 1 || m > kai_dual::MAX_COLS || k <= 0 || k % QK8_1 || stride_x < k ||
            s <= 0 || s % QK8_1 || a <= 0 || b <= 0 || a * b * s != k) {
            throw std::runtime_error("Invalid fused view preparation arguments");
        }
        const auto layout = fused_layout(activation, m, k, 0);
        const int threads = kai_dual::prep_threads(k);
        const cudaStream_t stream = stream_of(stream_raw);
        switch (x_dtype) {
            case 0: kai_dual::prepare_view<float><<<unsigned(m), threads, 0, stream>>>(static_cast<const float *>(x), stride_x, int(k), int(a), int(b), int(s), layout.y1, layout.inverse); break;
            case 1: kai_dual::prepare_view<nv_bfloat16><<<unsigned(m), threads, 0, stream>>>(static_cast<const nv_bfloat16 *>(x), stride_x, int(k), int(a), int(b), int(s), layout.y1, layout.inverse); break;
            case 2: kai_dual::prepare_view<half><<<unsigned(m), threads, 0, stream>>>(static_cast<const half *>(x), stride_x, int(k), int(a), int(b), int(s), layout.y1, layout.inverse); break;
            default: throw std::runtime_error("Fused activations must be F32, BF16 or F16");
        }
    });
}

// Single-pass activation of silu(gate) * up from a merged [m, 2k] gate_up row (k = down-projection input).
extern "C" int kai_fused_prepare_swiglu(const void * gate_up, int dtype, int64_t m, int64_t k, int64_t stride,
                                        void * activation, uintptr_t stream_raw) {
    return fused_entry([&] {
        if (!gate_up || !activation || m < 1 || m > kai_dual::MAX_COLS || k <= 0 || k % QK8_1 || stride < 2 * k) {
            throw std::runtime_error("Invalid fused SwiGLU preparation arguments");
        }
        const auto layout = fused_layout(activation, m, k, 0);
        const int threads = kai_dual::prep_threads(k);
        const cudaStream_t stream = stream_of(stream_raw);
        switch (dtype) {
            case 0: kai_dual::prepare_swiglu<float><<<unsigned(m), threads, 0, stream>>>(static_cast<const float *>(gate_up), stride, int(k), layout.y1, layout.inverse); break;
            case 1: kai_dual::prepare_swiglu<nv_bfloat16><<<unsigned(m), threads, 0, stream>>>(static_cast<const nv_bfloat16 *>(gate_up), stride, int(k), layout.y1, layout.inverse); break;
            case 2: kai_dual::prepare_swiglu<half><<<unsigned(m), threads, 0, stream>>>(static_cast<const half *>(gate_up), stride, int(k), layout.y1, layout.inverse); break;
            default: throw std::runtime_error("Fused activations must be F32, BF16 or F16");
        }
    });
}

// Gemma RMSNorm (optional residual add into an F32 residual stream) writing BF16 y and y's activation.
extern "C" int kai_fused_norm_prepare(const void * x, int x_dtype, int64_t stride_x, const void * residual,
                                      int residual_dtype, int64_t stride_r, float * residual_out,
                                      const float * weight32, float eps, void * y, int64_t m, int64_t k,
                                      void * activation, uintptr_t stream_raw) {
    return fused_entry([&] {
        if (!x || !y || !weight32 || !activation || m < 1 || m > kai_dual::MAX_COLS || k <= 0 || k % QK8_1 ||
            stride_x < k || (residual && (!residual_out || stride_r < k))) {
            throw std::runtime_error("Invalid fused norm preparation arguments");
        }
        const auto layout = fused_layout(activation, m, k, 0);
        const int threads = kai_dual::prep_threads(k);
        const cudaStream_t stream = stream_of(stream_raw);
        auto * out = static_cast<nv_bfloat16 *>(y);
        const int key = x_dtype * 4 + (residual ? residual_dtype : 0);
        switch (key) {
            case 1 * 4 + 0: kai_dual::norm_prepare<nv_bfloat16, float><<<unsigned(m), threads, 0, stream>>>(static_cast<const nv_bfloat16 *>(x), stride_x, static_cast<const float *>(residual), stride_r, residual_out, weight32, eps, out, int(k), layout.y1, layout.inverse); break;
            case 1 * 4 + 1: kai_dual::norm_prepare<nv_bfloat16, nv_bfloat16><<<unsigned(m), threads, 0, stream>>>(static_cast<const nv_bfloat16 *>(x), stride_x, static_cast<const nv_bfloat16 *>(residual), stride_r, residual_out, weight32, eps, out, int(k), layout.y1, layout.inverse); break;
            case 0 * 4 + 0: kai_dual::norm_prepare<float, float><<<unsigned(m), threads, 0, stream>>>(static_cast<const float *>(x), stride_x, static_cast<const float *>(residual), stride_r, residual_out, weight32, eps, out, int(k), layout.y1, layout.inverse); break;
            case 0 * 4 + 1: kai_dual::norm_prepare<float, nv_bfloat16><<<unsigned(m), threads, 0, stream>>>(static_cast<const float *>(x), stride_x, static_cast<const nv_bfloat16 *>(residual), stride_r, residual_out, weight32, eps, out, int(k), layout.y1, layout.inverse); break;
            default: throw std::runtime_error("Fused norm inputs must be BF16 or F32 (residual BF16 or F32)");
        }
    });
}

extern "C" int kai_fused_mmvq(const void * weight, int type, int64_t n, int64_t k, int64_t m,
                              const void * activation, int two_pass, void * dst, int dst_bf16,
                              int64_t stride_dst, uintptr_t stream_raw) {
    try {
        last_error.clear();
        if (!weight || !activation || !dst || !fused_shape_ok(type, n, k, m) || stride_dst < n || stride_dst > INT_MAX) {
            throw std::runtime_error("Invalid fused MMVQ arguments");
        }
        const cudaStream_t stream = stream_raw ? reinterpret_cast<cudaStream_t>(stream_raw) : cudaStreamLegacy;
        const auto layout = fused_layout(const_cast<void *>(activation), m, k, two_pass);
        kai_dual::mmvq(type, weight, layout.y1, layout.y2, layout.inverse, dst, dst_bf16 != 0,
                       int(k), int(n), int(m), int(stride_dst), stream);
        const auto status = cudaGetLastError();
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        return 0;
    } catch (const std::exception & error) {
        last_error = error.what();
        return -1;
    }
}

// ---------------------------------------------------------------------------
// Fused MMQ projection group (batches > 8 and prefill): exact per-row power-of-two
// normalization + GGML MMQ activation quantization (bit-identical to
// quantize_mmq_q8_1 on the normalized row), one GGML MMQ launch per matrix into a
// merged F32 tile, and one epilogue that undoes the normalization and stores
// BF16/F32. Replaces per-matrix guards, quantization, repairs, casts and concats.
namespace kai_mmq {
using kai_dual::load_value;
using kai_dual::store_value;
constexpr int SCALE_THREADS = 256;

template <typename src_t>
__launch_bounds__(SCALE_THREADS, 1)
static __global__ void row_scale(const src_t * __restrict__ x, const int64_t stride_x, const int k,
                                 float * __restrict__ scale, float * __restrict__ inverse) {
    const src_t * xr = x + int64_t(blockIdx.x) * stride_x;
    constexpr int vec = 16 / sizeof(src_t);
    float amax = 0.0f;
    if (reinterpret_cast<uintptr_t>(xr) % 16 == 0 && k % vec == 0) {
        const int4 * xv = reinterpret_cast<const int4 *>(xr);
#pragma unroll 4
        for (int i = threadIdx.x; i < k / vec; i += SCALE_THREADS) {
            const int4 packed = xv[i];
            const src_t * values = reinterpret_cast<const src_t *>(&packed);
#pragma unroll
            for (int j = 0; j < vec; ++j) {
                amax = fmaxf(amax, fabsf(load_value(values + j)));
            }
        }
    } else {
        for (int i = threadIdx.x; i < k; i += SCALE_THREADS) {
            amax = fmaxf(amax, fabsf(load_value(xr + i)));
        }
    }
    amax = warp_reduce_max<WARP_SIZE>(amax);
    __shared__ float warp_max[SCALE_THREADS / WARP_SIZE];
    const int warp = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    if (lane == 0) {
        warp_max[warp] = amax;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = 0.0f;
#pragma unroll
        for (int w = 0; w < SCALE_THREADS / WARP_SIZE; ++w) {
            m = fmaxf(m, warp_max[w]);
        }
        const int biased = int((__float_as_uint(m) >> 23) & 255u);
        int scaled_exponent = 127;
        if (biased != 0 && biased != 255) {
            scaled_exponent = min(max(260 - biased, 1), 253);
        }
        scale[blockIdx.x] = __uint_as_float(uint32_t(scaled_exponent) << 23);
        inverse[blockIdx.x] = __uint_as_float(uint32_t(254 - scaled_exponent) << 23);
    }
}

// GGML quantize_mmq_q8_1 (single channel, no scatter) on the normalized row.
template <mmq_q8_1_ds_layout ds_layout, typename src_t>
static __global__ void quantize_scaled(const src_t * __restrict__ x, const float * __restrict__ scale,
                                       void * __restrict__ vy, const int64_t ne00, const int64_t s01,
                                       const int64_t ne0, const int ne1) {
    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;
    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;
    if (i0 >= ne0) {
        return;
    }
    const int row = blockIdx.x;
    const float sc = scale[row];
    const src_t * xr = x + int64_t(row) * s01;
    float4 xi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (i0 < ne00) {
        xi.x = load_value(xr + i0 + 0) * sc;
        xi.y = load_value(xr + i0 + 1) * sc;
        xi.z = load_value(xr + i0 + 2) * sc;
        xi.w = load_value(xr + i0 + 3) * sc;
    }
    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;
    const int64_t k_block = i0 / QK8_1_MMQ;
    const int64_t iqs     = i0 % QK8_1_MMQ;

    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }
    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);
    const float d = 1.0f / d_inv;
    // As current GGML quantize_mmq_q8_1: the stored sums are those of the quantized values.
    float sum = 0.0f;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        int sum_q = int(q.x) + int(q.y) + int(q.z) + int(q.w);
#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            sum_q += __shfl_xor_sync(0xFFFFFFFF, sum_q, offset, WARP_SIZE);
        }
        sum = d * float(sum_q);
    }

    const int64_t ib = k_block*ne1 + row;
    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;
    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 == 0 && iqs < 96) {
            y[ib].d2s6[2 + iqs/16] = sum;
            if (iqs % 64 == 0) {
                y[ib].d2s6[iqs/64] = d;
            }
        }
    } else if (iqs % 32 == 0) {
        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
            y[ib].ds4[iqs/32] = make_half2(d, sum);
        } else {
            y[ib].d4[iqs/32] = d;
        }
    }
}

// dst[row*stride_dst + col] = src[row*n + col] * inverse[row]; n % 4 == 0.
template <typename dst_t>
static __global__ void scale_store(const float * __restrict__ src, const float * __restrict__ inverse,
                                   dst_t * __restrict__ dst, const int64_t m, const int64_t n,
                                   const int64_t stride_dst) {
    const int64_t i = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if (i >= m * n) {
        return;
    }
    const int64_t row = i / n, col = i % n;
    const float4 v = *reinterpret_cast<const float4 *>(src + i);
    const float s = inverse[row];
    dst_t * d = dst + row * stride_dst + col;
    d[0] = store_value<dst_t>(v.x * s);
    d[1] = store_value<dst_t>(v.y * s);
    d[2] = store_value<dst_t>(v.z * s);
    d[3] = store_value<dst_t>(v.w * s);
}

int layout_index(ggml_type type) {
    const auto layout = mmq_get_q8_1_ds_layout(type);
    return layout == MMQ_Q8_1_DS_LAYOUT_D4 ? 0 : layout == MMQ_Q8_1_DS_LAYOUT_DS4 ? 1 : 2;
}

struct Sizes {
    size_t quant[3] = {0, 0, 0};
    size_t fixup = 0;
    int64_t total_rows = 0;
    size_t scales() const { return align_size(0); }
};

struct Plan {
    size_t offset_scale, offset_inverse, offset_quant[3], offset_out, offset_fixup, fixup, total;
    int64_t total_rows;
    bool has[3];
};

Plan plan(int device, const int * types, const int64_t * rows, int count, int64_t k, int64_t m, bool direct_f32) {
    if (!types || !rows || count < 1 || count > 32) {
        throw std::runtime_error("Expected between one and 32 original GGUF matrices");
    }
    const int cc = ggml_cuda_info().devices[device].cc;
    const int64_t padded = GGML_PAD(k, MATRIX_ROW_PADDING);
    size_t quant[3] = {0, 0, 0};
    bool has[3] = {false, false, false};
    size_t fixup = 0;
    int64_t total_rows = 0;
    for (int i = 0; i < count; ++i) {
        validate(device, types[i], rows[i], k, m);
        const auto type = ggml_type(types[i]);
        if (ggml_cuda_should_use_mmvq(type, cc, m) || !ggml_cuda_should_use_mmq(type, cc, m, 0)) {
            throw std::runtime_error("Fused MMQ requires a batch that GGML runs with MMQ");
        }
        const size_t base = m * padded * sizeof(block_q8_1) / QK8_1;
        const size_t bytes = base + ggml_cuda_mmq_get_J_max(type, rows[i] % 128 != 0, cc, m) * sizeof(block_q8_1_mmq);
        const int l = layout_index(type);
        quant[l] = std::max(quant[l], bytes);
        has[l] = true;
        fixup = std::max(fixup, workspace_size(device, type, rows[i], k, m) - align_size(bytes));
        total_rows += rows[i];
    }
    if (total_rows % 4 || total_rows > INT_MAX) {
        throw std::runtime_error("Fused MMQ output width must be a multiple of four");
    }
    Plan p{};
    size_t offset = 0;
    p.offset_scale = offset;   offset += align_size(m * sizeof(float));
    p.offset_inverse = offset; offset += align_size(m * sizeof(float));
    for (int l = 0; l < 3; ++l) {
        p.offset_quant[l] = offset;
        offset += align_size(quant[l]);
        p.has[l] = has[l];
    }
    p.offset_out = offset;
    if (!direct_f32) {
        offset += align_size(size_t(m) * total_rows * sizeof(float));
    }
    p.offset_fixup = offset;
    p.fixup = align_size(fixup);
    offset += p.fixup;
    p.total = offset;
    p.total_rows = total_rows;
    return p;
}

template <typename src_t>
void quantize_all(const Plan & p, char * base, const src_t * x, int64_t stride_x, int64_t k, int64_t m,
                  cudaStream_t stream) {
    float * scale = reinterpret_cast<float *>(base + p.offset_scale);
    float * inverse = reinterpret_cast<float *>(base + p.offset_inverse);
    row_scale<src_t><<<unsigned(m), SCALE_THREADS, 0, stream>>>(x, stride_x, int(k), scale, inverse);
    const int64_t padded = GGML_PAD(k, MATRIX_ROW_PADDING);
    const dim3 grid(unsigned(m), unsigned((padded + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ)), 1);
    if (p.has[0]) quantize_scaled<MMQ_Q8_1_DS_LAYOUT_D4, src_t><<<grid, CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 0, stream>>>(
        x, scale, base + p.offset_quant[0], k, stride_x, padded, int(m));
    if (p.has[1]) quantize_scaled<MMQ_Q8_1_DS_LAYOUT_DS4, src_t><<<grid, CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 0, stream>>>(
        x, scale, base + p.offset_quant[1], k, stride_x, padded, int(m));
    if (p.has[2]) quantize_scaled<MMQ_Q8_1_DS_LAYOUT_D2S6, src_t><<<grid, CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 0, stream>>>(
        x, scale, base + p.offset_quant[2], k, stride_x, padded, int(m));
}
}  // namespace kai_mmq

extern "C" size_t kai_fused_mmq_workspace(int device, const int * types, const int64_t * rows, int count,
                                          int64_t k, int64_t m, int direct_f32) {
    try {
        last_error.clear();
        return kai_mmq::plan(device, types, rows, count, k, m, direct_f32 != 0).total;
    } catch (const std::exception & error) {
        last_error = error.what();
        return 0;
    }
}

extern "C" int kai_fused_mmq_run(const void * const * weights, const int * types, const int64_t * rows, int count,
                                 const void * x, int x_dtype, int64_t stride_x, int64_t k, int64_t m,
                                 void * dst, int dst_bf16, int64_t stride_dst, void * scratch, size_t scratch_bytes,
                                 int device, uintptr_t stream_raw) {
    try {
        last_error.clear();
        const bool direct_f32 = !dst_bf16 && stride_dst == 0;
        const auto p = kai_mmq::plan(device, types, rows, count, k, m, direct_f32);
        if (!weights || !x || !dst || !scratch || scratch_bytes < p.total || k % 4 || stride_x < k) {
            throw std::runtime_error("Invalid fused MMQ pointers, strides or scratch capacity");
        }
        for (int i = 0; i < count; ++i) if (!weights[i]) throw std::runtime_error("Null packed matrix");
        auto & context = context_for(device);
        std::lock_guard<std::mutex> lock(context.mutex);
        ggml_cuda_set_device(device);
        const cudaStream_t stream = stream_raw ? reinterpret_cast<cudaStream_t>(stream_raw) : cudaStreamLegacy;
        char * base = static_cast<char *>(scratch);
        switch (x_dtype) {
            case 0: kai_mmq::quantize_all(p, base, static_cast<const float *>(x), stride_x, k, m, stream); break;
            case 1: kai_mmq::quantize_all(p, base, static_cast<const nv_bfloat16 *>(x), stride_x, k, m, stream); break;
            case 2: kai_mmq::quantize_all(p, base, static_cast<const half *>(x), stride_x, k, m, stream); break;
            default: throw std::runtime_error("Fused MMQ activations must be F32, BF16 or F16");
        }
        float * out = direct_f32 ? static_cast<float *>(dst) : reinterpret_cast<float *>(base + p.offset_out);
        BorrowedStorage storage(context, base + p.offset_fixup, p.fixup, stream_raw);
        const int64_t padded = GGML_PAD(k, MATRIX_ROW_PADDING);
        const int64_t sy = m * padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t total = p.total_rows;
        int64_t first_row = 0;
        for (int i = 0; i < count; ++i) {
            const auto type = ggml_type(types[i]);
            const int64_t n = rows[i], row_blocks = k / ggml_blck_size(type);
            const int * quant = reinterpret_cast<const int *>(base + p.offset_quant[kai_mmq::layout_index(type)]);
            const mmq_args args = {
                static_cast<const char *>(weights[i]), type, quant, nullptr, nullptr, out + first_row, nullptr,
                k, n, m, row_blocks, m, total,
                1, 1, n*row_blocks, sy, total*m,
                1, 1, n*row_blocks, sy, total*m, m, m};
            shared_mmq(context.cuda, args, stream);
            first_row += n;
        }
        const float * inverse = reinterpret_cast<const float *>(base + p.offset_inverse);
        const int64_t elements = m * total;
        const unsigned blocks = unsigned((elements / 4 + 255) / 256);
        if (dst_bf16) {
            kai_mmq::scale_store<nv_bfloat16><<<blocks, 256, 0, stream>>>(out, inverse, static_cast<nv_bfloat16 *>(dst),
                                                                       m, total, stride_dst ? stride_dst : total);
        } else {
            kai_mmq::scale_store<float><<<blocks, 256, 0, stream>>>(out, inverse, static_cast<float *>(dst),
                                                                 m, total, stride_dst ? stride_dst : total);
        }
        const auto status = cudaGetLastError();
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        return 0;
    } catch (const std::exception & error) {
        last_error = error.what();
        return -1;
    }
}
