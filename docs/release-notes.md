# Unreleased

- Linux x86-64 installs from a source checkout use its serving code with a
  checksum-verified binary and dependency base.
- The default LFM3 chat configuration supports OpenAI-compatible function
  calling. Streamed and complete responses expose thinking separately from
  the final answer.
- With the bundled CUDA 13.0 and a GCC 16 or newer default, the launcher can
  select an installed compatible compiler pair. The installer and
  `kaillm doctor` report the configuration.
- Local GGUFs with unsupported architecture or Inkling layout fail during
  header inspection, before the engine starts.
- Added an [Inkling source installer for DGX Spark](../README.md#dgx-spark), an
  optional DSpark-5 draft cache window, and opt-in parallel projections
  for single-request Dondon XS plain decoding.
- Added opt-in Inkling Chimachima XS verification on RTX 5090
  (`KAILLM_INKLING_CHIMA_XS_ROW_STABLE_DS5=1`) and Dondon M prefill and
  verification kernels on RTX PRO 6000
  (`KAILLM_INKLING_DONDON_IQ_PREFILL=1` and `KAILLM_INKLING_DONDON_IQ_VERIFY=1`).
  These options target a single request with BF16 KV, DSpark-5, a 73,728-token
  context, and 512-token prefill chunks.

# kaiLLM 0.1.0rc6

ITQ tensors at 3.5 and 4 bits per weight, for the v2.0 LFM3 chimachima checkpoints.

- kaiPTQ checkpoints can now contain ITQ tensors at 3.5 and 4 bits per weight:
  2 weights per step, at 7 or 8 bits per step. The v2.0 750 MB and 1000 MB
  checkpoints in
  [kaitchup/LFM3-1.7B-chimachima-kaiPTQ](https://huggingface.co/kaitchup/LFM3-1.7B-chimachima-kaiPTQ)
  use them and need this release. The kernels are prebuilt for compute
  capability 8.6 and 12.0.
- Everything else is unchanged from 0.1.0rc5.

# kaiLLM 0.1.0rc5

Qwen3.8 27B kaiPTQ checkpoints, which also run on the RTX 3060, and DSpine speculative decoding.

- kaiPTQ checkpoints can now contain ITQ tensors: trellis codes, as in QTIP,
  decoded with an integer table, from 1.25 to 3 bits per weight. Each step
  decodes 4 weights from a 12-bit table, or 2 weights from a 10-bit table of
  int8 pairs. The decode, GEMV and prefill kernels are prebuilt for compute
  capability 8.6 and 12.0, so the first server start doesn't compile them.
- Qwen3.8 kaiPTQ checkpoints run on the RTX 3060 (compute capability 8.6).
  Plain Qwen3.5 and Qwen3.8 GGUFs are still qualified on Blackwell only.
- The first one, Q2_K_XXS in
  [kaitchup/Qwen3.8-27B-chimachima-kaiPTQ](https://huggingface.co/kaitchup/Qwen3.8-27B-chimachima-kaiPTQ),
  is 6.73 GB, the same file size as unsloth's UD-IQ1_M. Its KL divergence to
  the BF16 model on held-out reasoning traces and code is 57% lower. On the
  RTX 3060, kaiLLM runs it 14% faster at batch 1 and 34% faster at batch 8
  than llama.cpp runs UD-IQ1_M, and prefills 83% faster. On the RTX Pro 6000,
  it is 3% slower at batch 1, but 84% faster at batch 8, 132% faster at
  batch 16, and it prefills 54% faster.
- DSpine speculative decoding with `--dspine`, one sequence at a time:
  kaiLLM serves a DSpine drafter next to a Qwen3.8 27B GGUF. The chimachima
  repository has a 4-bit copy of Tencent's drafter, and a drafter in Tencent's
  own format is converted once to the same 4-bit weights. In a chat with
  thinking, it makes the Q2_K_XXS checkpoint 1.85× faster on the RTX Pro 6000
  and 1.92× faster on the RTX 3060. On GPUs with 16 GB or less, `--dspine`
  defaults to a 1 GiB KV cache and 512-token prefill chunks: with the larger
  defaults, the server went past 12 GB and decoding got about 4x slower.
- The installer now works from a clone of the repository: it imported the
  clone's kaillm folder instead of the installed package and failed at its
  last step.
- MTP speculative decoding now uses the same verification paths as DSpine:
  split-KV attention sized for each verification length, and vLLM's fused GDN
  path up to 16 tokens. I didn't measure the 0.1.0rc4 MTP figures again with
  this release.
- unsloth's UD-IQ1_M runs, but kaiLLM's kernels for the IQ1_M and IQ2 formats
  aren't optimized yet: llama.cpp is much faster with it.

# kaiLLM 0.1.0rc4

The weights now stay quantized in VRAM, as with llama.cpp.

- The serving defaults no longer keep any persistent dequantized or repacked
  copy of the weights: no BF16 prefill cache, no Marlin repack, no F16 copy of
  Q6_K weights and no CPU RAM cache. Weights are dequantized on the fly, inside
  the kernels or one matrix at a time for prefill. The weights loaded in VRAM
  are within 1% of the GGUF file size on the A40 and RTX 5090. With 0.1.0rc3,
  the 500 MB chimachima checkpoint took 4.88 GiB of VRAM.
- YAQA prefill without the BF16 cache: kaiLLM decodes the YAQA weights of one
  matrix at a time, then multiplies them with cuBLAS (FP16 inputs, FP32
  output). The kaiPTQ checkpoints themselves are unchanged, but their prefill is
  slower than with 0.1.0rc3.
- RTX 3060 and A40 (compute capability 8.6): the resident small-batch decode
  path for YAQA is off, since it needed the full dequantized weight cache.
- The short-context speed figures were measured again with this release, against llama.cpp 93cf736, on the RTX 3060,
  A40, RTX 5090 and RTX Pro 6000. On the RTX 3060, this llama.cpp is much
  faster at batch 8 and 16 than the build I used for the earlier figures, so
  the gains listed below for 0.1.0rc3 are smaller against it.

# kaiLLM 0.1.0rc3

Highly experimental GGUF and mixed YAQA/GGUF serving built on vLLM, using parts
of llama.cpp/ggml. Optimized for LFM3 on RTX 3060, A40 and RTX 5090, and for
Qwen3.8 27B on RTX Pro 6000.

- Qwen3.5/Qwen3.8 GGUF support (Gated DeltaNet + attention), qualified on
  Blackwell (compute capability 12.0). The tokenizer, config and chat template
  come from the GGUF itself. GGUFs with MTP layers use MTP speculative decoding
  by default, with 4 draft tokens up to 8 concurrent sequences; `--spec-tokens`
  turns speculation on or off at any capacity.
- Qwen3.8 27B (unsloth UD-Q4_K_XL and UD-Q2_K_XL) on an RTX Pro 6000: faster
  than llama.cpp at batches 1, 2, 8 and 16, with and without speculative
  decoding.
- suisui on the RTX 3060: +9.6% to +15.4% at batch 1 and +116% to +125% at batch 16
  compared with llama.cpp.
- Shared GGUF paths: every GGML format with a Q8_1 dot product, including the
  legacy Q4_0/Q4_1/Q5_0/Q5_1 formats, runs through fused native decode kernels
  (one activation preparation per projection group) and fused MMQ for larger
  batches and prefill. Activation block sums now use the quantized values, as
  in current llama.cpp, which lowers the quantization error of Q4_0, Q4_1, Q5_x,
  Q4_K and Q5_K matrices.
- Lower CPU overhead when pinned host memory is unavailable.

The llama.cpp benchmark baseline was not optimized. See the README for
the measured gains and slowdowns.

Qwen3.5/Qwen3.8 on Ampere GPUs, and RTX 5080 and RTX 3090 qualification are planned.
