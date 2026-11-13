![kaiLLM banner](docs/images/kaillm-banner.png)
[Support (buy me GPU hours)](https://buymeacoffee.com/kaitchup) · [Subscribe to The Kaitchup](https://kaitchup.substack.com/subscribe)

# kaiLLM



**Highly experimental. Optimized for LFM3 and Qwen3.8 27B.** Support for other
models (and Qwen4 27B, if this model exists) is planned.

kaiLLM is an inference engine built on top of [vLLM](https://github.com/vllm-project/vllm),
using parts of [llama.cpp](https://github.com/ggml-org/llama.cpp) for GGUF support.
It serves GGUF checkpoints, kaiPTQ checkpoints (YAQA tensors mixed with GGUF tensors), and more,
through vLLM's batching scheduler and OpenAI-compatible API.

The goal is to support a wider range of complex quantization formats than
llama.cpp and vLLM currently support, preserve more accuracy at lower precision,
and make inference faster, especially at high concurrency. Performance depends
on the checkpoint, GPU and batch size; this is an ongoing project.

The focus is local/workstation GPUs. Current optimization and
testing cover **RTX 3060, RTX 5090, and RTX Pro 6000**. **RTX 5080 and RTX 3090**
are on the roadmap. AMD GPUs could be too... I only own an RTX 3060.  If you want to help expand support,
please consider [supporting GPU time](https://buymeacoffee.com/kaitchup).

## Models

Tested models. Other GGUFs with the same architectures (LFM3,
Qwen3.5 and Qwen3.8) might work too, but I haven't tested them.

### LFM3

| Model | Checkpoints | Format |
|---|---|---|
| [kaitchup/LFM3-1.7B-suisui-GGUF](https://huggingface.co/kaitchup/LFM3-1.7B-suisui-GGUF) | 1.00, 1.25 and 1.50 GB | Mixed GGUF tensors |
| [kaitchup/LFM3-1.7B-chimachima-kaiPTQ](https://huggingface.co/kaitchup/LFM3-1.7B-chimachima-kaiPTQ) | 500, 750, 1000, 1250 and 1500 MB tiers | kaiPTQ (mixed YAQA and GGUF tensors) |
| [LiquidAI/LFM3-1.7B-GGUF](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF) | QAD-Q4_0 and Q8_0 | GGUF tensors |
| [AtomicChat/LFM3-1.7B-GGUF](https://huggingface.co/AtomicChat/LFM3-1.7B-GGUF) | AD-IQ3_S, AD-Q5_K_M and AD-Q6_K |   GGUF tensors |

kaiPTQ checkpoints are stored as .gguf files, but they aren't ordinary GGUFs:
llama.cpp and other GGUF runtimes can't load them.

### Qwen3.8

| Model | Checkpoints | Format |
|---|---|---|
| [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | UD-Q4_K_XL and UD-Q2_K_XL |   GGUF tensors, with the MTP layer |

In this release, Qwen3.5 and Qwen3.8 GGUFs only run on Blackwell GPUs (compute
capability 12.0).

## Install

Requirements: Linux + CUDA 13.

```bash
git clone https://github.com/the-kaitchup/kaiLLM.git
cd kaiLLM
bash install.sh
```

The installer creates an isolated Python environment and installs the pinned
runtime.
The first server start may take several minutes to prepare kernels and CUDA
graphs.

## Serve a model

This command downloads the 1 GB suisui checkpoint if needed and starts a local
OpenAI-compatible server:

```bash
kaillm serve kaitchup/LFM3-1.7B-suisui-GGUF:LFM3-1.7B-suisui-Q4_0_XXS.gguf
```

A local file works too:

```bash
kaillm serve /path/to/model.gguf
```

Qwen3.8 27B works the same way:

```bash
kaillm serve unsloth/Qwen3.8-27B-GGUF:Qwen3.8-27B-UD-Q4_K_XL.gguf
```

The tokenizer, GPU profile and eligible quantization paths are selected
automatically from the GGUF. LFM3 GGUFs use the LFM3 tokenizer. Qwen3.5 and
Qwen3.8 GGUFs use their own tokenizer and chat template.

The server listens at `http://127.0.0.1:8000` and uses the model
name `kaillm`. It defaults to eight concurrent sequences. To change the serving
capacity, for example:

```bash
kaillm serve /path/to/model.gguf --max-num-seqs 16
```

You decide when to use speculative decoding. `--spec-tokens` sets the number of
draft tokens, at any serving capacity, and `--spec-tokens 0` turns speculation
off. GGUFs with MTP layers, like the unsloth Qwen3.8 27B GGUFs, draft with their
MTP layer. GGUFs without MTP layers, like all the LFM3 GGUFs above, use n-gram
drafting: the draft tokens are copied from earlier text in the context that
matches the last generated tokens. For example, MTP with 4 draft tokens at 16
concurrent sequences:

```bash
kaillm serve unsloth/Qwen3.8-27B-GGUF:Qwen3.8-27B-UD-Q4_K_XL.gguf --max-num-seqs 16 --spec-tokens 4
```

Without `--spec-tokens`, kaiLLM picks a default from the serving capacity:

- GGUFs with MTP layers: MTP with 4 draft tokens up to 8 concurrent sequences,
  no speculation above.
- LFM3 GGUFs: n-gram drafting with 3 draft tokens up to 4 concurrent sequences,
  fewer up to 8, no speculation above.
- Qwen3.5 and Qwen3.8 GGUFs without MTP layers: no speculation.

Send a request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"kaillm","messages":[{"role":"user","content":"What is 6 + 7? Answer with the number only."}],"temperature":1.0,"top_p":0.95,"top_k":20,"max_tokens":4096,"chat_template_kwargs":{"enable_thinking":true}}'
```

## Speed

Unless stated otherwise, the figures below were obtained without speculative decoding.

For a single sequence, kaiLLM can still be much slower than llama.cpp, especially on the RTX 5090. Most of the optimization work has focused on batching.

**The llama.cpp baseline was not optimized.** All the comparisons use llama.cpp master (93cf736).

Generation: 512 input / 256 output tokens. Prefill: 2048 input tokens. Rates are batch totals in tokens/s.

**These numbers aren't realistic for real use cases.** They are promising, but better benchmarks are necessary and planned.

### Qwen3.8 27B: kaiLLM vs llama.cpp (same GGUF)

I optimized kaiLLM for Qwen3.8 27B on a rented RTX Pro 6000 (Blackwell, 96 GB). Both engines ran the same unsloth GGUFs in the same session. Each figure is the median of 3 to 5 runs. Rates are kaiLLM / llama.cpp in tokens/s.

Without speculative decoding:

| Checkpoint | Batch 1 | Batch 2 | Batch 8 | Batch 16 | Prefill |
|---|---:|---:|---:|---:|---:|
| UD-Q4_K_XL | 67.8 / 66.4 (**+2.1%**) | 122.6 / 116.8 (**+5.0%**) | 344.0 / 260.8 (**+31.9%**) | 502.3 / 353.3 (**+42.2%**) | 4,736 / 3,198 (**+48.1%**) |
| UD-Q2_K_XL | 94.8 / 92.7 (**+2.3%**) | 162.6 / 153.8 (**+5.7%**) | 355.5 / 273.9 (**+29.8%**) | 508.1 / 353.0 (**+43.9%**) | 4,466 / 3,001 (**+48.8%**) |


With MTP speculative decoding, each engine at its best draft length (2, 3 or 4) for each batch size:

| Checkpoint | Batch 1 | Batch 2 | Batch 8 | Batch 16 |
|---|---:|---:|---:|---:|
| UD-Q4_K_XL | 140.3 (n=4) / 122.3 (n=4) (**+14.7%**) | 223.9 (n=3) / 187.4 (n=2) (**+19.5%**) | 544.4 (n=3) / 316.5 (n=2) (**+72.0%**) | 620.6 (n=2) / 396.8 (n=3) (**+56.4%**) |
| UD-Q2_K_XL | 153.6 (n=4) / 137.3 (n=3) (**+11.9%**) | 254.2 (n=2) / 210.4 (n=2) (**+20.8%**) | 593.8 (n=3) / 347.3 (n=3) (**+71.0%**) | 631.0 (n=2) / 382.3 (n=2) (**+65.1%**) |


### chimachima kaiPTQ (kaiLLM) vs LiquidAI Q4_K_M (llama.cpp)

llama.cpp can't run kaiPTQ checkpoints, so the baseline is LiquidAI's Q4_K_M GGUF.

| GPU | Checkpoint | Batch 1 | Batch 8 | Batch 16 | Prefill |
|---|---|---:|---:|---:|---:|
| RTX 3060 | chimachima 500 MB | 147.8 / 194.6 (-24.0%) | 584.3 / 582.3 (**+0.3%**) | 819.9 / 954.4 (-14.1%) | 4,273 / 7,917 (-46.0%) |
| RTX 3060 | chimachima 750 MB | 155.8 / 194.6 (-19.9%) | 587.4 / 582.3 (**+0.9%**) | 799.8 / 954.4 (-16.2%) | 4,235 / 7,917 (-46.5%) |
| RTX 3060 | chimachima 1000 MB | 166.0 / 194.6 (-14.7%) | 585.3 / 582.3 (**+0.5%**) | 806.0 / 954.4 (-15.5%) | 4,246 / 7,917 (-46.4%) |
| RTX 3060 | chimachima 1250 MB | 169.4 / 194.6 (-12.9%) | 617.3 / 582.3 (**+6.0%**) | 1,013 / 954.4 (**+6.2%**) | 6,073 / 7,917 (-23.3%) |
| RTX 3060 | chimachima 1500 MB | 167.5 / 194.6 (-13.9%) | 610.5 / 582.3 (**+4.8%**) | 1,175 / 954.4 (**+23.1%**) | 7,824 / 7,917 (-1.2%) |
| A40 | chimachima 500 MB | 233.0 / 300.0 (-22.3%) | 1,040 / 1,106 (-6.0%) | 1,607 / 1,557 (**+3.2%**) | 11,021 / 15,561 (-29.2%) |
| A40 | chimachima 750 MB | 221.7 / 300.0 (-26.1%) | 1,047 / 1,106 (-5.3%) | 1,568 / 1,557 (**+0.7%**) | 10,983 / 15,561 (-29.4%) |
| A40 | chimachima 1000 MB | 241.1 / 300.0 (-19.6%) | 1,039 / 1,106 (-6.1%) | 1,553 / 1,557 (-0.3%) | 11,087 / 15,561 (-28.8%) |
| A40 | chimachima 1250 MB | 253.4 / 300.0 (-15.5%) | 1,193 / 1,106 (**+7.9%**) | 1,775 / 1,557 (**+14.0%**) | 14,158 / 15,561 (-9.0%) |
| A40 | chimachima 1500 MB | 261.6 / 300.0 (-12.8%) | 1,229 / 1,106 (**+11.1%**) | 1,838 / 1,557 (**+18.0%**) | 16,165 / 15,561 (**+3.9%**) |
| RTX 5090 | chimachima 500 MB | 461.3 / 630.0 (-26.8%) | 2,051 / 2,100 (-2.3%) | 2,872 / 2,746 (**+4.6%**) | 30,814 / 31,900 (-3.4%) |
| RTX 5090 | chimachima 750 MB | 426.3 / 630.0 (-32.3%) | 1,926 / 2,100 (-8.3%) | 2,844 / 2,746 (**+3.6%**) | 30,381 / 31,900 (-4.8%) |
| RTX 5090 | chimachima 1000 MB | 475.9 / 630.0 (-24.5%) | 1,971 / 2,100 (-6.2%) | 2,926 / 2,746 (**+6.6%**) | 30,764 / 31,900 (-3.6%) |
| RTX 5090 | chimachima 1250 MB | 446.5 / 630.0 (-29.1%) | 2,472 / 2,100 (**+17.7%**) | 3,550 / 2,746 (**+29.3%**) | 39,675 / 31,900 (**+24.4%**) |
| RTX 5090 | chimachima 1500 MB | 437.7 / 630.0 (-30.5%) | 2,974 / 2,100 (**+41.6%**) | 4,369 / 2,746 (**+59.1%**) | 46,679 / 31,900 (**+46.3%**) |

### suisui: kaiLLM vs llama.cpp (same GGUF)


| GPU | Checkpoint | Batch 1 | Batch 2 | Batch 8 | Batch 16 | Prefill |
|---|---|---:|---:|---:|---:|---:|
| RTX 3060 | suisui Q4_0_XXS | 223.9 / 207.1 (**+8.1%**) | 402.3 / 363.5 (**+10.7%**) | 780.9 / 636.4 (**+22.7%**) | 1,407 / 1,008 (**+39.5%**) | 8,836 / 8,261 (**+7.0%**) |
| RTX 3060 | suisui Q4_0_XS | 196.9 / 174.2 (**+13.0%**) | 353.2 / 311.7 (**+13.3%**) | 783.0 / 636.1 (**+23.1%**) | 1,320 / 947.4 (**+39.3%**) | 8,970 / 8,147 (**+10.1%**) |
| RTX 3060 | suisui Q8_0_S | 168.6 / 152.4 (**+10.6%**) | 315.8 / 275.6 (**+14.6%**) | 764.2 / 624.9 (**+22.3%**) | 1,245 / 910.8 (**+36.6%**) | 8,914 / 8,313 (**+7.2%**) |
| A40 | suisui Q4_0_XXS | 321.2 / 329.0 (-2.4%) | 596.7 / 579.6 (**+3.0%**) | 1,467 / 1,223 (**+19.9%**) | 2,198 / 1,620 (**+35.6%**) | 17,499 / 15,375 (**+13.8%**) |
| A40 | suisui Q4_0_XS | 296.7 / 289.5 (**+2.5%**) | 533.5 / 506.1 (**+5.4%**) | 1,449 / 1,209 (**+19.9%**) | 2,069 / 1,552 (**+33.3%**) | 17,961 / 16,264 (**+10.4%**) |
| A40 | suisui Q8_0_S | 261.1 / 259.0 (**+0.8%**) | 475.9 / 461.5 (**+3.1%**) | 1,383 / 1,152 (**+20.1%**) | 1,948 / 1,474 (**+32.2%**) | 17,564 / 16,293 (**+7.8%**) |
| RTX 5090 | suisui Q4_0_XXS | 441.7 / 636.7 (-30.6%) | 898.5 / 1,075 (-16.4%) | 3,012 / 2,279 (**+32.1%**) | 4,891 / 2,806 (**+74.3%**) | 53,788 / 37,511 (**+43.4%**) |
| RTX 5090 | suisui Q4_0_XS | 463.5 / 571.5 (-18.9%) | 899.8 / 985.8 (-8.7%) | 3,044 / 2,219 (**+37.2%**) | 4,788 / 2,757 (**+73.7%**) | 55,408 / 35,789 (**+54.8%**) |
| RTX 5090 | suisui Q8_0_S | 459.8 / 533.1 (-13.7%) | 942.1 / 849.4 (**+10.9%**) | 2,946 / 2,144 (**+37.4%**) | 4,161 / 2,630 (**+58.2%**) | 56,181 / 32,847 (**+71.0%**) |

### Other GGUFs: kaiLLM vs llama.cpp (same GGUF)

These checkpoints use kaiLLM's shared GGUF kernels, without further optimization
for the individual checkpoints.

| GPU | Checkpoint | Batch 1 | Batch 8 | Batch 16 | Prefill |
|---|---|---:|---:|---:|---:|
| RTX 3060 | AtomicChat AD-IQ3_S | 210.5 / 187.0 (**+12.6%**) | 764.8 / 612.5 (**+24.9%**) | 1,356 / 979.5 (**+38.4%**) | 8,574 / 8,142 (**+5.3%**) |
| RTX 3060 | LiquidAI QAD-Q4_0 | 220.5 / 204.3 (**+7.9%**) | 806.1 / 656.4 (**+22.8%**) | 1,417 / 1,011 (**+40.1%**) | 8,923 / 8,304 (**+7.5%**) |
| RTX 3060 | AtomicChat AD-Q5_K_M | 188.7 / 176.0 (**+7.2%**) | 647.5 / 555.9 (**+16.5%**) | 1,273 / 936.5 (**+35.9%**) | 8,081 / 7,663 (**+5.5%**) |
| RTX 3060 | AtomicChat AD-Q6_K | 163.6 / 154.5 (**+5.9%**) | 647.3 / 577.7 (**+12.0%**) | 1,171 / 881.2 (**+32.9%**) | 8,040 / 7,641 (**+5.2%**) |
| RTX 3060 | LiquidAI Q8_0 | 146.7 / 138.3 (**+6.1%**) | 732.3 / 581.3 (**+26.0%**) | 1,153 / 845.3 (**+36.4%**) | 8,815 / 8,298 (**+6.2%**) |
| A40 | AtomicChat AD-IQ3_S | 331.2 / 316.8 (**+4.5%**) | 1,514 / 1,165 (**+30.0%**) | 2,199 / 1,573 (**+39.8%**) | 18,714 / 16,037 (**+16.7%**) |
| A40 | LiquidAI QAD-Q4_0 | 330.3 / 321.4 (**+2.8%**) | 1,534 / 1,218 (**+26.0%**) | 2,162 / 1,541 (**+40.3%**) | 18,234 / 15,699 (**+16.1%**) |
| A40 | AtomicChat AD-Q5_K_M | 286.2 / 265.2 (**+7.9%**) | 1,319 / 1,038 (**+27.1%**) | 1,986 / 1,442 (**+37.7%**) | 17,170 / 14,685 (**+16.9%**) |
| A40 | AtomicChat AD-Q6_K | 247.7 / 247.5 (**+0.1%**) | 1,286 / 1,028 (**+25.1%**) | 1,832 / 1,351 (**+35.7%**) | 17,433 / 15,216 (**+14.6%**) |
| A40 | LiquidAI Q8_0 | 225.7 / 211.6 (**+6.7%**) | 1,310 / 992.3 (**+32.0%**) | 1,812 / 1,281 (**+41.4%**) | 17,917 / 15,606 (**+14.8%**) |
| RTX 5090 | AtomicChat AD-IQ3_S | 460.4 / 614.4 (-25.1%) | 2,622 / 2,143 (**+22.4%**) | 4,481 / 2,902 (**+54.4%**) | 56,525 / 37,255 (**+51.7%**) |
| RTX 5090 | LiquidAI QAD-Q4_0 | 454.4 / 641.7 (-29.2%) | 2,804 / 2,535 (**+10.6%**) | 4,675 / 2,979 (**+56.9%**) | 52,507 / 28,458 (**+84.5%**) |
| RTX 5090 | AtomicChat AD-Q5_K_M | 485.1 / 560.8 (-13.5%) | 3,010 / 1,920 (**+56.8%**) | 4,409 / 2,702 (**+63.2%**) | 47,061 / 35,601 (**+32.2%**) |
| RTX 5090 | AtomicChat AD-Q6_K | 474.3 / 516.9 (-8.2%) | 2,938 / 2,026 (**+45.1%**) | 4,022 / 2,643 (**+52.1%**) | 49,160 / 30,349 (**+62.0%**) |
| RTX 5090 | LiquidAI Q8_0 | 452.5 / 497.3 (-9.0%) | 2,966 / 2,116 (**+40.2%**) | 4,188 / 2,700 (**+55.1%**) | 53,215 / 27,689 (**+92.2%**) |

[All recorded kaiLLM speeds and batches](docs/speed-results.md).

## Optimization coverage

The matrix distinguishes checkpoints covered by the optimization work from
checkpoints benchmarked with the existing implementation. Suisui was optimized on
the RTX 3060, and Qwen3.8 27B on the RTX Pro 6000.

| Model checkpoint | RTX 3060 | A40 | RTX 5090 |
|---|---|---|---|
| Chimachima 500 MB | Optimized | Optimized | Optimized |
| Chimachima 750 MB | Optimized | Optimized | Optimized |
| Chimachima 1000 MB | Optimized | Optimized | Optimized |
| Chimachima 1250 MB | Optimized | Optimized | Optimized |
| Chimachima 1500 MB | Optimized | Optimized | Optimized |
| AtomicChat AD-IQ3_S | Shared kernel coverage | Shared kernel coverage | Shared kernel coverage |
| LiquidAI QAD-Q4_0 | Shared kernel coverage | Shared kernel coverage | Shared kernel coverage |
| AtomicChat AD-Q5_K_M | Shared kernel coverage | Shared kernel coverage | Shared kernel coverage |
| AtomicChat AD-Q6_K | Shared kernel coverage | Shared kernel coverage | Shared kernel coverage |
| LiquidAI Q8_0 | Shared kernel coverage | Shared kernel coverage | Shared kernel coverage |
| Suisui Q4_0_XXS | Optimized | Benchmark only | Benchmark only |
| Suisui Q4_0_XS | Optimized | Benchmark only | Benchmark only |
| Suisui Q8_0_S | Optimized | Benchmark only | Benchmark only |

| Model checkpoint | RTX Pro 6000 |
|---|---|
| Qwen3.8 27B UD-Q4_K_XL | Optimized |
| Qwen3.8 27B UD-Q2_K_XL | Optimized |

**Optimized** marks checkpoint/GPU combinations used to tune and validate the
kernels and serving profiles. Tuning used ngram speculative decoding with
3 speculative tokens per step at batches 1 and 2, 1 at batch 8, and speculation
disabled at batch 16. The speed comparisons above use speculation disabled in
both engines, except for the Qwen3.8 MTP table.

**Shared kernel coverage** marks public GGUFs measured with the improved format
paths. **Benchmark only** marks tests performed without further tuning.

Profiles are selected by GPU architecture, tensor format, shape and serving
capacity. Performance at other models, batches and GPUs still needs testing.

## Quantization support

- Load GGUF files directly, including mixed tensor formats.
- Load kaiPTQ checkpoints, which combine GGUF and YAQA tensors in one file.
- Support K-quants, IQ formats and YAQA QTIP bit widths, with fallback paths
  for formats without an optimized native kernel.
- Use separate prefill and decode paths, CUDA graphs and automatic hardware
  profiles for batching.

The packaged reader recognizes 35 storage types. Fallback reconstruction
supports QTIP 1–8 bits, ternary 1.6-bit and scalar-int4. Optimized GPU coverage is
a subset of that format coverage.

## Roadmap

- Broader model support, and Qwen3.5/Qwen3.8 on Ampere GPUs (RTX 3060, RTX 3090, A40).
- RTX 5080 and RTX 3090 qualification and optimization.
- More native paths for kaiPTQ checkpoints and better concurrency.

## License and acknowledgments

kaiLLM uses **Apache-2.0**. See [LICENSE](LICENSE).
It builds on vLLM and uses parts of llama.cpp/ggml. Third-party components retain
their original licenses and notices; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Thanks to the vLLM, llama.cpp/ggml, PyTorch, Humming and model-author communities.

[Support the project](https://buymeacoffee.com/kaitchup) · [Get updates in the newsletter](https://kaitchup.substack.com/subscribe)
