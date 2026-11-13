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
- All the speed figures in the README and in docs/speed-results.md were
  measured again with this release, against llama.cpp 93cf736, on the RTX 3060,
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
- Lower CPU overhead under WSL, where vLLM cannot use pinned host memory.

The llama.cpp benchmark baseline was not optimized. See the README for
the measured gains and slowdowns.

Qwen3.5/Qwen3.8 on Ampere GPUs, and RTX 5080 and RTX 3090 qualification are planned.
