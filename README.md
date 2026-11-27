![kaiLLM banner](docs/images/kaillm-banner.png)

# kaiLLM

kaiLLM is an experimental inference engine for GGUF and kaiPTQ models on
NVIDIA GPUs. It uses vLLM for batching, parts of llama.cpp and ggml for GGUF
inference, and exposes an OpenAI-compatible API. [vLLM](https://github.com/vllm-project/vllm) ·
[llama.cpp](https://github.com/ggml-org/llama.cpp)

[Support GPU testing](https://buymeacoffee.com/kaitchup) ·
[The Kaitchup newsletter](https://kaitchup.substack.com/subscribe)

## Updates

- **November 27, 2026 · Available on `main`.** Inkling DSpark and expert-kernel
  improvements on RTX 5090, RTX PRO 6000, and DGX Spark, plus LFM3 tool calling
  and separate reasoning output. The Linux x86-64 installer now uses the
  serving code from `main`. [Release notes](docs/release-notes.md#november-27-2026)
- **0.1.0rc6.** Added 3.5- and 4-bit ITQ tensors for LFM3 Chimachima v2.
  [Release notes](docs/release-notes.md#kaillm-010rc6)
- **0.1.0rc5.** Added Qwen3.8 27B kaiPTQ support on RTX 3060 and DSpine
  speculative decoding. [Release notes](docs/release-notes.md#kaillm-010rc5)
- **0.1.0rc4.** Kept weights in their quantized form in VRAM during serving.
  [Release notes](docs/release-notes.md#kaillm-010rc4)
- **0.1.0rc3.** Added Qwen3.5 and Qwen3.8 GGUF serving with native MTP on
  Blackwell. [Release notes](docs/release-notes.md#kaillm-010rc3)

## Models

I focused mainly on GPU coverage. Speed optimizations are ongoing. If you
have a favorite AI agent, it can help make kaiLLM faster.

Each GPU link opens the model's measurements and configuration limits in
the speed report. Each listed GPU has at least one completed kaiLLM run for
that family. Rates apply to the benchmarked checkpoint, batch, and build.
[Full speed report](speed.md) · [Checkpoint files](speed.md#checkpoint-files)

| Model | Checkpoints timed in kaiLLM | GPUs with kaiLLM measurements |
|---|---|---|
| **LFM3 1.7B** | [LiquidAI/LFM3-1.7B-GGUF](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/tree/cd0c74c575b5c416f3011b5b4c298077f26ca8c2) (QAD-Q4_0) · [kaitchup/LFM3-1.7B-suisui-GGUF](https://huggingface.co/kaitchup/LFM3-1.7B-suisui-GGUF/tree/6fcfa6348c8399bd436b7ccd8a0c5e6c9dfe765b) (Q4_0_XS) · [kaitchup/LFM3-1.7B-chimachima-kaiPTQ](https://huggingface.co/kaitchup/LFM3-1.7B-chimachima-kaiPTQ/tree/f9963f2a326732d3aa3202c316ebe9e7784999ab) (ITQ 750 MB) | [RTX 3090](speed.md#gpu-3090-1-model-lfm) · [RTX 5070 Ti](speed.md#gpu-5070-ti-1-model-lfm) · [RTX 5080](speed.md#gpu-5080-1-model-lfm) · [RTX 5090](speed.md#gpu-5090-1-model-lfm) · [RTX PRO 6000](speed.md#gpu-pro-6000-1-model-lfm) |
| **DaisyRose 27B** | [kaitchup/DaisyRose-27B-dondon-GGUF](https://huggingface.co/kaitchup/DaisyRose-27B-dondon-GGUF/tree/3592e073925dbdbc65a340fca851fc01b8d2c409) (IQ2_M) · [kaitchup/DaisyRose-27B-chimachima-kaiPTQ](https://huggingface.co/kaitchup/DaisyRose-27B-chimachima-kaiPTQ/tree/6c1a63358c57f5baf2730574ba418195eb7edfa8) (Q2_K_XS) | [RTX 3090](speed.md#gpu-3090-1-model-daisy) · [RTX 5090](speed.md#gpu-5090-1-model-daisy) · [RTX PRO 6000](speed.md#gpu-pro-6000-1-model-daisy) |
| **Qwen3.8 27B** | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/4af51b3542f775b0a8e4b310bfb3f17e01c306c9) (UD-Q2_K_XL) · [kaitchup/Qwen3.8-27B-chimachima-kaiPTQ](https://huggingface.co/kaitchup/Qwen3.8-27B-chimachima-kaiPTQ/tree/aa5396581a0d09201312c0262dcfa2d0e1871eba) (Q2_K_XXS) | [RTX 3090](speed.md#gpu-3090-1-model-qwen) · [RTX 5080](speed.md#gpu-5080-1-model-qwen) · [RTX 5090](speed.md#gpu-5090-1-model-qwen) · [RTX PRO 6000](speed.md#gpu-pro-6000-1-model-qwen) |
| **Inkling 2 Mini** | [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/8078d149286eec7193795413b9e0643da6d11fa1) (UD-Q2_K_XL) · [ggml-org/Inkling-2-Mini-GGUF](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/tree/0bdc580634ceac4c4d921f452777bf44228de9af) (IQ4_NL) · [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q2_K_XS, Q3_K_S, Q3_K_M) · [kaitchup/Inkling-2-Mini-chimachima-kaiPTQ](https://huggingface.co/kaitchup/Inkling-2-Mini-chimachima-kaiPTQ/tree/1b27e927c15a6cde7f717926a25b2ce831d08fc1) (Q2_K_XS, Q3_K_S, Q3_K_M) | [RTX 3090](speed.md#gpu-3090-1-model-inkling) · [RTX 5070 Ti](speed.md#gpu-5070-ti-1-model-inkling) · [RTX 5080](speed.md#gpu-5080-1-model-inkling) · [RTX 5090](speed.md#gpu-5090-1-model-inkling) · [RTX PRO 6000](speed.md#gpu-pro-6000-1-model-inkling) · [DGX Spark](speed.md#gpu-dgx-spark-1-model-inkling) |
| **Hy4 Flash** | [Xylolabs/Hy4-flash-GGUF](https://huggingface.co/Xylolabs/Hy4-flash-GGUF/tree/cc17e8a68a84db84a367f56fd42fbd40b54b28e2) (IQ2_M-E4) | [RTX PRO 6000](speed.md#gpu-pro-6000-1-model-hy4) |

Earlier measurements cover LFM3 on RTX 3060 and A40 and Qwen3.8 Chimachima
on RTX 3060. [Earlier short-context benchmarks](speed.md#earlier-short-context-benchmarks)

kaiLLM supports standard GGUF types such as K-quants and IQ formats, plus
kaiPTQ checkpoints that mix GGUF, YAQA, QTIP, or ITQ tensors in a `.gguf` file.
kaiPTQ checkpoints require a compatible runtime.

## Speed

kaiLLM prefill / decode in tokens/s for batch-one code with 8,192 input and
65,536 output tokens.

| GPU | LFM3 1.7B QAD-Q4_0 | Inkling 2 Mini Chimachima Q3_K_S |
|---|---:|---:|
| [DGX Spark](speed.md#gpu-dgx-spark-1) | — | 697.6 / 49.4 |
| [RTX 5080](speed.md#gpu-5080-1) | 35,509.3 / 366.0 | 1,144.6 / 86.5 |

LFM3 uses greedy plain decoding. Inkling uses DSpark-5 with model-recommended
sampling. LFM3 has no DGX Spark measurement.

### Inkling Dondon

kaiLLM throughput in tokens/s for code runs with model-recommended sampling.

| GPU | Dondon GGUF | Batch | Mode | Prefill | Decode |
|---|---|---:|---|---:|---:|
| [RTX PRO 6000](speed.md#gpu-pro-6000-1-model-inkling) | Q3_K_M | 1 | Plain | 2,055.0 | 82.1 |
| RTX PRO 6000 | Q3_K_M | 1 | DSpark-5 | 533.8 | 47.3 |
| RTX PRO 6000 | Q3_K_M | 8 | Plain | 994.3 | 105.8 |
| [RTX 5090](speed.md#gpu-5090-1-model-inkling) | Q3_K_S | 1 | Plain | 2,231.4 | 85.0 |
| RTX 5090 | Q3_K_S | 1 | DSpark-3 | 497.0 | 51.7 |
| RTX 5090 | Q3_K_M | 1 | Plain | 1,993.6 | 98.0 |
| [DGX Spark](speed.md#gpu-dgx-spark-1-model-inkling) | Q2_K_XS | 1 | Plain | 641.3 | 36.1 |
| DGX Spark | Q2_K_XS | 1 | DSpark-5 | 153.7 | 21.9 |

The plain and DSpark-5 runs used separate DGX Spark systems and kaiLLM builds.
On prose, DSpark-5 reached 7.0 decode tok/s.

### Inkling Chimachima

Both are code runs with model-recommended sampling.

- **RTX 5090 · batch 1 · DSpark-5.** Chimachima Q3_K_M reached **2,565.6
  prefill and 105.8 decode tok/s**.
  [Results](speed.md#gpu-5090-1-model-inkling)
- **RTX PRO 6000 · batch 8 · plain.** Chimachima Q3_K_M reached **4,130.3
  prefill and 209.4 decode tok/s**.
  [Results](speed.md#gpu-pro-6000-1-model-inkling)

## Install

### Linux x86-64

Install Python 3.9 or newer, a C++ compiler, and an NVIDIA driver compatible
with CUDA 13.
Install from `main` to use the current serving code with a checksum-verified
binary and dependency base.

```bash
git clone --branch main https://github.com/the-kaitchup/kaiLLM.git
cd kaiLLM
bash install.sh
~/.local/bin/kaillm doctor
```

The installer prepares an isolated Python 3.12 environment for the checked-out
source. The first server start may take several minutes to prepare kernels
and CUDA graphs. The launcher lives at `~/.local/bin/kaillm`.
`kaillm --version` also shows the checkout fingerprint alongside the package version.

Ordinary Qwen3.5/Qwen3.8 GGUFs use compute capability 12.0 by default.
The Qwen3.8 27B Chimachima Q2_K_XXS also runs on the 12 GB RTX 3060.

### DGX Spark

Fixed-loop Inkling runs on one GB10 with a source build. Use Linux AArch64,
CPython 3.12, a CUDA toolkit with `sm_121` and NVRTC support, and a Python
environment with CUDA-enabled PyTorch and native vLLM built from
`f3871c3cbf8a608f7ee833ec4c2c5277f77bee41`. Run the installer in that environment.

```bash
git clone --branch main https://github.com/the-kaitchup/kaiLLM.git
cd kaiLLM
bash install.sh
~/.local/share/kaillm/spark/bin/kaillm doctor
```

### CUDA host compiler on rolling Linux

The bundled CUDA 13.0 compiler supports GCC through version 15. If a build
reports `unsupported GNU version`, select an installed compatible GCC and G++
pair before starting kaiLLM. [NVIDIA CUDA 13.0 compiler requirements](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-installation-guide-linux/index.html#host-compiler-support-policy)

```bash
export NVCC_CCBIN=/usr/bin/g++-14 CC=/usr/bin/gcc-14
export CXX=/usr/bin/g++-14 CUDAHOSTCXX=/usr/bin/g++-14
kaillm doctor
```

## Serve a model

Pass a Hugging Face repository and GGUF filename or a local path.

```bash
# Hugging Face checkpoint
kaillm serve kaitchup/LFM3-1.7B-suisui-GGUF:LFM3-1.7B-suisui-Q4_0_XS.gguf
# Local checkpoint
kaillm serve /path/to/model.gguf
```

The server defaults to `127.0.0.1:8000`, model name `kaillm`, eight concurrent
sequence slots, and an 8,192-token context. Use `--device`, `--max-num-seqs`,
and `--max-model-len` to select the GPU, concurrency, and context size.
Keep all shards of a split GGUF in the same directory and serve the first
local shard.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"kaillm","messages":[{"role":"user","content":"What is 6 + 7?"}],"max_tokens":256}'
```

### Speculative decoding

| Target | Drafter | Default |
|---|---|---|
| LFM3 | Built-in n-gram | Draft length depends on configured concurrency |
| GGUF with a native MTP head | Built-in MTP | Four draft tokens through eight concurrent slots, off above eight |
| Qwen3.8 27B | Optional DSpine | Seven draft tokens with one sequence |
| Inkling 2 Mini on a qualified build | Optional DSpark | Five draft tokens |

Set `--spec-tokens 0` to disable automatic n-gram or MTP drafting. Plain
serving uses no `--dspine` or `--dspark` flag. For an explicit drafter,
`--spec-tokens` controls the number of draft tokens.

With DSpine on GPUs with 16 GB or less, the launcher defaults to a 1 GiB KV
cache and 512-token prefill chunks.

```bash
kaillm serve kaitchup/Qwen3.8-27B-chimachima-kaiPTQ:Qwen3.8-27B-chimachima-Q2_K_XXS.gguf --max-num-seqs 1 \
  --dspine kaitchup/Qwen3.8-27B-chimachima-kaiPTQ:dspine-b16-int4
```

```bash
kaillm serve /path/to/Inkling.gguf --max-num-seqs 1 --dspark /path/to/drafter.gguf
```

### LFM3 tools and reasoning

The current source accepts function schemas in `tools` and returns structured
`tool_calls` with the standard LFM3 chat template. Thinking is returned
separately from final content in both streaming and complete responses. Read
`reasoning` or `reasoning_content`. Set
`chat_template_kwargs.enable_thinking=false` for direct answers or
`include_reasoning=false` to omit thinking from the response. Disable automatic
tool choice with `--no-enable-auto-tool-choice`. For custom templates, select
matching parsers via `--tool-call-parser` and `--reasoning-parser`.
[Release notes](docs/release-notes.md#november-27-2026)

## License and acknowledgments

kaiLLM uses Apache-2.0. Third-party components retain their own licenses.
[License](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md)

Thanks to the vLLM, llama.cpp/ggml, PyTorch, Humming, and model-author communities.
