// SPDX-License-Identifier: Apache-2.0 AND MIT
// GGML layout/launch portions: Copyright (c) 2023-2026 The ggml authors.
// See LLAMA_CPP_LICENSE.txt for the upstream notice and permission terms.
// Separate experimental library; the existing direct/shared binaries remain
// unchanged. Reuse their host validation, context, and native fallback code.
#include "current_ggml_direct.cu"
#ifdef GGUF_AFFINE_MMA
#include "q45_affine_mma.cuh"
#endif

namespace {
bool body_shape_policy(ggml_type type, int64_t n, int64_t k, int64_t m) {
#ifdef GGUF_AFFINE_MMA
    if (m == 8 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)
            && n >= 128 && n % 128 == 0 && k > 0 && k % 256 == 0) return true;
#endif
    return m == 8 && type == GGML_TYPE_Q6_K &&
        ((n == 6144 && k == 1024) || (n == 1024 && k == 3584));
}

bool body_mma_enabled(int device, ggml_type type, int64_t n, int64_t k, int64_t m) {
    return body_shape_policy(type, n, k, m) &&
        ggml_cuda_should_use_mmvq(type, ggml_cuda_info().devices[device].cc, m);
}

struct BodySizes {
    SharedSizes native;
    size_t mma = 0, affine = 0, fixup = 0;
    size_t prefix() const { return native.prefix() + align_size(mma) + align_size(affine); }
    size_t total() const { return prefix() + fixup; }
};

BodySizes body_sizes(int device, const int * types, const int64_t * rows,
                     int count, int64_t k, int64_t m) {
    BodySizes result;
    result.native = shared_sizes(device, types, rows, count, k, m);
    result.fixup = result.native.fixup;
    const auto & dev = ggml_cuda_info().devices[device];
    for (int i = 0; i < count; ++i) {
        const auto type = ggml_type(types[i]);
        if (!body_mma_enabled(device, type, rows[i], k, m)) continue;
        const bool fallback = rows[i] % 128 != 0;
        int best_j = 0, best_tiles = INT_MAX;
        for (int j = 8; j <= 128 && best_tiles > 1; j += 8) {
            const auto config = ggml_cuda_mmq_get_config(type, j, fallback, dev.cc);
            if (config.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(config, dev.cc) > dev.smpbo) continue;
            const int tiles = (m + j - 1) / j;
            if (tiles < best_tiles) { best_j = j; best_tiles = tiles; }
        }
        if (!best_j) throw std::runtime_error("No body MMA tile supports this device/shape");
        const auto config = ggml_cuda_mmq_get_config(type, best_j, fallback, dev.cc);
        const int guard = std::max(config.J, ggml_cuda_mmq_get_J_max(type, fallback, dev.cc, m));
        const size_t bytes = m * GGML_PAD(k, MATRIX_ROW_PADDING) * sizeof(block_q8_1) / QK8_1
            + guard * sizeof(block_q8_1_mmq);
        if (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)
            result.affine = std::max(result.affine, bytes);
        else result.mma = std::max(result.mma, bytes);
        if (config.stream_k) {
            const int64_t tiles = ((rows[i] + config.I - 1) / config.I) * best_tiles;
            const int64_t waves = (tiles + dev.nsm - 1) / dev.nsm;
            const int64_t efficiency = 100 * tiles / (dev.nsm * waves);
            const int64_t blocks = GGML_CUDA_CC_IS_NVIDIA(dev.cc) && efficiency >= 90 ? tiles : dev.nsm;
            if (tiles % blocks)
                result.fixup = std::max(result.fixup, align_size(blocks * config.J * config.I * sizeof(float)));
        }
    }
    return result;
}

// Codes are copied, not requantized. Widening native half scales into D4 is
// exact. The complete small-M load guard is zeroed by the same launch.
template<bool AFFINE = false>
__global__ void body_repack_q8(const block_q8_1 * src, uint32_t * dst,
                              int64_t m, int64_t padded_k, int64_t total_words) {
    const int64_t word = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (word >= total_words) return;
    const int64_t block = word / 36;
    const int lane = word % 36;
    if (block >= m * padded_k / QK8_1_MMQ) { dst[word] = 0; return; }
    const int64_t row = block % m, kb = block / m;
    if (lane < 4) {
        const auto & q = src[row * padded_k / QK8_1 + kb * 4 + lane];
        if constexpr (AFFINE) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                sum = __dp4a(reinterpret_cast<const int *>(q.qs)[i], 0x01010101, sum);
            dst[word] = (*reinterpret_cast<const uint32_t *>(&q.ds) & 0xffff)
                | (uint32_t(uint16_t(int16_t(sum))) << 16);
        } else dst[word] = __float_as_uint(__low2float(q.ds));
    } else {
        const int iq = lane - 4;
        const auto & q = src[row * padded_k / QK8_1 + kb * 4 + iq / 8];
        dst[word] = reinterpret_cast<const uint32_t *>(q.qs)[iq % 8];
    }
}
}  // namespace

extern "C" int current_body_shape_policy(int type, int64_t n, int64_t k, int64_t m) {
    return body_shape_policy(ggml_type(type), n, k, m);
}

extern "C" int current_body_mma_enabled(int device, int type, int64_t n, int64_t k, int64_t m) {
    try {
        last_error.clear(); validate(device, type, n, k, m);
        return body_mma_enabled(device, ggml_type(type), n, k, m);
    } catch (const std::exception & error) { last_error = error.what(); return -1; }
}

extern "C" size_t current_body_workspace(int device, const int * types, const int64_t * rows,
                                         int count, int64_t k, int64_t m) {
    try {
        last_error.clear(); return body_sizes(device, types, rows, count, k, m).total();
    } catch (const std::exception & error) { last_error = error.what(); return 0; }
}

extern "C" int current_body_run(const void * const * weights, const int * types, const int64_t * rows,
                                 int count, const float * x, float * y, void * scratch,
                                 size_t bytes, int device, int64_t k, int64_t m, uintptr_t stream_raw) {
    try {
        last_error.clear();
        const auto sizes = body_sizes(device, types, rows, count, k, m);
        if (!sizes.mma && !sizes.affine)
            return current_shared_run(weights, types, rows, count, x, y, scratch, bytes, device, k, m, stream_raw);
        if (!weights || !x || !y || !scratch || bytes < sizes.total())
            throw std::runtime_error("Invalid body MMA storage");
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
            if (!sizes.native.quant[layout]) continue;
            quant[layout] = static_cast<char *>(scratch) + offset;
            offset += align_size(sizes.native.quant[layout]);
            const quantize_cuda_t quantize = layout ? quantize_mmq_q8_1_cuda : quantize_row_q8_1_cuda;
            quantize(x, nullptr, quant[layout], sizes.native.representative[layout],
                     k, k, m*k, m*k, padded, m, 1, 1, stream);
        }
        auto * mma = static_cast<char *>(scratch) + sizes.native.prefix();
        auto * affine = mma + align_size(sizes.mma);
        if (sizes.mma) {
            const int64_t words = sizes.mma / sizeof(uint32_t);
            body_repack_q8<false><<<(words + 255) / 256, 256, 0, stream>>>
                (reinterpret_cast<const block_q8_1 *>(quant[0]), reinterpret_cast<uint32_t *>(mma), m, padded, words);
        }
        if (sizes.affine) {
            const int64_t words = sizes.affine / sizeof(uint32_t);
            body_repack_q8<true><<<(words + 255) / 256, 256, 0, stream>>>
                (reinterpret_cast<const block_q8_1 *>(quant[0]), reinterpret_cast<uint32_t *>(affine), m, padded, words);
        }
        const int64_t sy = m * padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t total = sizes.native.output_rows;
        int64_t first_row = 0;
        for (int i = 0; i < count; ++i) {
            const auto type = ggml_type(types[i]);
            const int64_t n = rows[i], row_blocks = k / ggml_blck_size(type);
            const int layout = shared_layout(device, type, m);
            const bool forced = body_mma_enabled(device, type, n, k, m);
            float * output = y + first_row;
            if (!layout && !forced) {
                auto weight = tensor_header(type, k, n, const_cast<void *>(weights[i]));
                auto input = tensor_header(GGML_TYPE_F32, k, m, const_cast<float *>(x));
                auto destination = tensor_header(GGML_TYPE_F32, total, m, output);
                ggml_cuda_op_mul_mat_vec_q(context.cuda, &weight, &input, &destination,
                    static_cast<const char *>(weights[i]), x, quant[0], output, 0, n, m, padded, stream);
            } else {
                const bool is_affine = forced && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K);
                const mmq_args args = {
                    static_cast<const char *>(weights[i]), type,
                    reinterpret_cast<const int *>(forced ? (is_affine ? affine : mma) : quant[layout]),
                    nullptr, nullptr, output, nullptr,
                    k, n, m, row_blocks, m, total,
                    1, 1, n*row_blocks, sy, total*m,
                    1, 1, n*row_blocks, sy, total*m, m, m};
#ifdef GGUF_AFFINE_MMA
                if (is_affine) {
                    if (type == GGML_TYPE_Q4_K)
                        launch_mul_mat_q<GGML_TYPE_Q4_K, 8, false>(context.cuda, args, stream);
                    else launch_mul_mat_q<GGML_TYPE_Q5_K, 8, false>(context.cuda, args, stream);
                } else
#endif
                shared_mmq(context.cuda, args, stream);
            }
            first_row += n;
        }
        const auto status = cudaGetLastError();
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        return 0;
    } catch (const std::exception & error) { last_error = error.what(); return -1; }
}
