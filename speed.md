# kaiLLM prefill and decode speed

**8,192 input tokens and 65,536 output tokens per request.** Prefill and decode are aggregate tokens/s across the batch.

[Prose source (Moby-Dick)](https://raw.githubusercontent.com/GITenberg/Moby-Dick--Or-The-Whale_2701/bdf1948e6cd00963730971e5624e764a35f238c3/2701-0.txt) · [Code source (CPython)](https://raw.githubusercontent.com/python/cpython/60403a5409ff2c3f3b07dd2ca91a7a3e096839c7/Lib/ast.py) · [Benchmark code](benchmarks/long_context/) · [Checkpoints](#checkpoint-files) · [Timing data](benchmarks/long_context/results.jsonl) · [Availability](#availability)

[1 × RTX 3090](#gpu-3090-1) · [4 × RTX 3090](#gpu-3090-4) · [1 × RTX 5070 Ti](#gpu-5070-ti-1) · [1 × RTX 5080](#gpu-5080-1) · [4 × RTX 5080](#gpu-5080-4) · [1 × RTX 5090](#gpu-5090-1) · [2 × RTX 5090](#gpu-5090-2) · [1 × RTX PRO 6000](#gpu-pro-6000-1) · [1 × DGX Spark](#gpu-dgx-spark-1)

Red marks a kaiLLM rate when llama.cpp was more than 10% faster on the matched workload (**not optimized**). A dash indicates a missing measurement. † marks output that differed from a plain run on the same engine with the same sampler.

**Archived greedy** selects the highest-probability token at each step (temperature 0). **Model-recommended** samples using the model publisher’s settings. Sampling can change the generated text and can affect draft acceptance and decode speed.

<a id="gpu-3090-1"></a>

## 1 × RTX 3090

<a id="gpu-3090-1-model-lfm"></a>

### LFM3 1.7B

#### [LiquidAI/LFM3-1.7B-GGUF](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/tree/cd0c74c575b5c416f3011b5b4c298077f26ca8c2) (QAD-Q4_0)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain | prose | 20898.7 | 5336.8 | 291.2 | 206.5 | Unrecorded | Archived greedy |
| 1 | plain | code | 20911.8 | 4079.5 | 288.6 | 207.5 | Unrecorded | Archived greedy |
| 2 | plain | prose | 20955.9 | 5207.6 | 525.8 | 222.8 | Unrecorded | Archived greedy |
| 2 | plain | code | 20767.6 | 3652.8 | 522.8 | 224.7 | Unrecorded | Archived greedy |
| 8 | ngram1 | prose | 21460.3 | — | 1681.2 | — | Unrecorded | Archived greedy |
| 8 | ngram1 | code | 21190.4 | — | 1609.2 | — | Unrecorded | Archived greedy |

#### [kaitchup/LFM3-1.7B-suisui-GGUF](https://huggingface.co/kaitchup/LFM3-1.7B-suisui-GGUF/tree/6fcfa6348c8399bd436b7ccd8a0c5e6c9dfe765b) (Q4_0_XS)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 4 | ngram3 | prose | 21307.1 | 5134.6 | 2297.3 | 388.8 | Unrecorded | Archived greedy |
| 4 | ngram3 | code | 21027.1 | 4077.3 | 2136.5 | 288.0 | Unrecorded | Archived greedy |
| 16 | plain | prose | — | 5903.8 | — | 279.8 | Unrecorded | Archived greedy |
| 16 | plain | code | — | 5616.1 | — | 280.8 | Unrecorded | Archived greedy |

<a id="gpu-3090-1-model-daisy"></a>

### DaisyRose 27B

#### [kaitchup/DaisyRose-27B-dondon-GGUF](https://huggingface.co/kaitchup/DaisyRose-27B-dondon-GGUF/tree/3592e073925dbdbc65a340fca851fc01b8d2c409) (IQ2_M)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain | prose | 1365.9 | 1293.1 | 41.2 | 33.1 | Unrecorded | Archived greedy |
| 1 | plain | code | 1354.6 | 1294.3 | 41.3 | 33.2 | Unrecorded | Archived greedy |
| 1 | mtp5 | prose | 1340.2 | 1075.4 | 104.2† | 102.0† | Unrecorded | Archived greedy |
| 1 | mtp5 | code | 1333.5 | 1013.4 | 112.7† | 101.3† | Unrecorded | Archived greedy |

<a id="gpu-3090-1-model-qwen"></a>

### Qwen3.8 27B

#### [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/4af51b3542f775b0a8e4b310bfb3f17e01c306c9) (UD-Q2_K_XL)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain | prose | 1487.3 | 1418.1 | 43.6 | 34.5 | Unrecorded | Archived greedy |
| 1 | plain | code | 1466.1 | — | 43.6 | — | Unrecorded | Archived greedy |

#### [kaitchup/Qwen3.8-27B-chimachima-kaiPTQ](https://huggingface.co/kaitchup/Qwen3.8-27B-chimachima-kaiPTQ/tree/aa5396581a0d09201312c0262dcfa2d0e1871eba) (Q2_K_XXS)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain | prose | 2322.2 | Unsupported GGUF | 47.9 | Unsupported GGUF | Unrecorded | Archived greedy |
| 1 | plain | code | 2284.3 | Unsupported GGUF | 47.9 | Unsupported GGUF | Unrecorded | Archived greedy |

<a id="gpu-3090-1-model-inkling"></a>

### Inkling 2 Mini

#### [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/8078d149286eec7193795413b9e0643da6d11fa1) (UD-Q2_K_XL)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | — | 889.1 | — | 50.1 | llama.cpp default (ub 512) | Archived greedy |

#### [ggml-org/Inkling-2-Mini-GGUF](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/tree/0bdc580634ceac4c4d921f452777bf44228de9af) (IQ4_NL)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | 1884.3 | 812.2 | $\color{red}{39.8}$ | 49.6 | kaiLLM Default (prefill eager, decode graphs, eager head) · `3815707a` · llama.cpp default (ub 256) | Archived greedy |
| 1 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | 1866.1 | 812.2 | 51.5 | 49.6 | kaiLLM Default (prefill eager, decode graphs, eager head) · `e7ebc58f` · llama.cpp default (ub 256) | Archived greedy |
| 1 | plain (earlier kaiLLM · Humming unrecorded · [Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | $\color{red}{159.2}$ | 812.2 | $\color{red}{24.1}$ | 49.6 | kaiLLM Unrecorded · llama.cpp default (ub 256) | Archived greedy |
| 1 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | 1840.8 | 811.2 | $\color{red}{39.7}$ | 49.8 | kaiLLM Default (prefill eager, decode graphs, eager head) · `3815707a` · llama.cpp default (ub 256) | Archived greedy |
| 1 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | 1834.2 | 811.2 | 51.6 | 49.8 | kaiLLM Default (prefill eager, decode graphs, eager head) · `e7ebc58f` · llama.cpp default (ub 256) | Archived greedy |
| 1 | plain (earlier kaiLLM · Humming unrecorded · [Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | $\color{red}{158.8}$ | 811.2 | $\color{red}{24.1}$ | 49.8 | kaiLLM Unrecorded · llama.cpp default (ub 256) | Archived greedy |
| 1 | dspark5 ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | — | 622.6 | — | 61.5 | llama.cpp default (ub 128) | Archived greedy |
| 1 | dspark5 (deterministic kaiLLM · Inkling PR) | prose | 161.5 | — | 25.5 | — | Eager | Archived greedy |
| 1 | dspark5 ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | — | 640.8 | — | 86.2 | llama.cpp default (ub 128) | Archived greedy |
| 1 | dspark5 (deterministic kaiLLM · Inkling PR) | code | 161.3 | — | 35.0 | — | Eager | Archived greedy |

<details>
<summary><strong>DSpark decode comparisons</strong></summary>

| Engine | Batch | Domain | Baseline | Mode | Baseline decode | Measured decode | Change | Sampling |
|---|---:|---|---|---|---:|---:|---:|---|
| kaiLLM | 1 | code | plain | dspark5 | 51.6 | 35.0 | -32.1% | Archived greedy |
| kaiLLM | 1 | prose | plain | dspark5 | 51.5 | 25.5 | -50.5% | Archived greedy |

</details>

#### [kaitchup/Inkling-2-Mini-chimachima-kaiPTQ](https://huggingface.co/kaitchup/Inkling-2-Mini-chimachima-kaiPTQ/tree/1b27e927c15a6cde7f717926a25b2ce831d08fc1) (Q2_K_XS)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain | prose | 1177.3 | Unsupported GGUF | 44.3 | Unsupported GGUF | Full graphs | Archived greedy |
| 1 | plain | code | 1173.0 | Unsupported GGUF | 44.3 | Unsupported GGUF | Full graphs | Archived greedy |
| 1 | plain (deterministic kaiLLM 5b14819) | code | 1277.9 | Unsupported GGUF | 43.4 | Unsupported GGUF | Full graphs | Model-recommended |
| 1 | dspark3 | prose | 935.2 | Unsupported GGUF | 20.9 | Unsupported GGUF | Full graphs · 512-token prefill chunks | Model-recommended |
| 1 | dspark3 | code | 923.6 | Unsupported GGUF | 55.3 | Unsupported GGUF | Full graphs · 512-token prefill chunks | Model-recommended |
| 1 | dspark5 (deterministic kaiLLM 5b14819) | prose | 933.9 | Unsupported GGUF | 21.6 | Unsupported GGUF | Full graphs · 512-token prefill chunks | Model-recommended |
| 1 | dspark5 (deterministic kaiLLM 5b14819) | code | 920.9 | Unsupported GGUF | 51.8† | Unsupported GGUF | Full graphs · 512-token prefill chunks | Model-recommended |
| 2 | plain | code | 1280.5 | Unsupported GGUF | 44.5 | Unsupported GGUF | Full graphs · `86ac55c5` | Model-recommended |
| 2 | plain | code | 1280.3 | Unsupported GGUF | 28.3 | Unsupported GGUF | Full graphs · `c6f22a6f` | Model-recommended |
| 4 | plain | code | 942.6 | Unsupported GGUF | 51.1 | Unsupported GGUF | Full graphs · 512-token prefill chunks | Model-recommended |

<details>
<summary><strong>Batch-one performance versus closest-size GGUF</strong></summary>

| Mode | Domain | Closest-size llama.cpp checkpoint | Prefill vs reference | Decode vs reference | Sampling |
|---|---|---|---:|---:|---|
| dspark5 | code | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_S) | 101.4% | 87.4% | Model-recommended |
| dspark5 | prose | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_S) | 103.0% | 110.9% | Model-recommended |
| plain | code | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q2_K_XS) | 132.5% | 85.6% | Archived greedy |
| plain | prose | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q2_K_XS) | 131.8% | 85.2% | Archived greedy |

</details>

<details>
<summary><strong>DSpark decode comparisons</strong></summary>

| Engine | Batch | Domain | Baseline | Mode | Baseline decode | Measured decode | Change | Sampling |
|---|---:|---|---|---|---:|---:|---:|---|
| kaiLLM | 1 | code | plain | dspark3 | 43.4 | 55.3 | +27.4% | Model-recommended |
| kaiLLM | 1 | code | plain | dspark5 | 43.4 | 51.8 | +19.2% | Model-recommended |
| kaiLLM | 1 | code | dspark5 | dspark3 | 51.8 | 55.3 | +6.9% | Model-recommended |
| kaiLLM | 1 | prose | dspark5 | dspark3 | 21.6 | 20.9 | -3.4% | Model-recommended |

</details>

#### [kaitchup/Inkling-2-Mini-chimachima-kaiPTQ](https://huggingface.co/kaitchup/Inkling-2-Mini-chimachima-kaiPTQ/tree/1b27e927c15a6cde7f717926a25b2ce831d08fc1) (Q3_K_S)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain | prose | 1640.1 | Unsupported GGUF | 46.6 | Unsupported GGUF | Full graphs | Archived greedy |
| 1 | plain | code | 1630.0 | Unsupported GGUF | 46.7 | Unsupported GGUF | Full graphs | Archived greedy |
| 1 | plain (deterministic kaiLLM 7b9466d) | code | 1747.2 | Unsupported GGUF | 46.6 | Unsupported GGUF | Full graphs | Model-recommended |
| 1 | dspark3 | prose | 1181.9 | Unsupported GGUF | 26.2 | Unsupported GGUF | Full graphs · 512-token prefill chunks | Model-recommended |
| 1 | dspark3 | code | 1162.0 | Unsupported GGUF | 52.1 | Unsupported GGUF | Full graphs · 512-token prefill chunks | Model-recommended |
| 1 | dspark5 (deterministic kaiLLM 7b9466d) | prose | 1181.4 | Unsupported GGUF | 17.9 | Unsupported GGUF | Full graphs · 512-token prefill chunks | Model-recommended |
| 1 | dspark5 (deterministic kaiLLM 7b9466d) | code | 1163.5 | Unsupported GGUF | 46.5† | Unsupported GGUF | Full graphs · 512-token prefill chunks | Model-recommended |
| 2 | plain | code | 1751.6 | Unsupported GGUF | 46.4 | Unsupported GGUF | Full graphs · `86ac55c5` | Model-recommended |
| 2 | plain | code | 1750.5 | Unsupported GGUF | 29.2 | Unsupported GGUF | Full graphs · `c6f22a6f` | Model-recommended |
| 4 | plain | code | 1195.9 | Unsupported GGUF | 53.4 | Unsupported GGUF | Full graphs · 512-token prefill chunks | Model-recommended |

<details>
<summary><strong>Batch-one performance versus closest-size GGUF</strong></summary>

| Mode | Domain | Closest-size llama.cpp checkpoint | Prefill vs reference | Decode vs reference | Sampling |
|---|---|---|---:|---:|---|
| dspark5 | code | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_S) | 128.1% | 78.5% | Model-recommended |
| dspark5 | prose | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_S) | 130.3% | 92.0% | Model-recommended |
| plain | code | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_S) | 192.9% | 95.6% | Archived greedy |
| plain | prose | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_S) | 189.3% | 95.4% | Archived greedy |

</details>

<details>
<summary><strong>DSpark decode comparisons</strong></summary>

| Engine | Batch | Domain | Baseline | Mode | Baseline decode | Measured decode | Change | Sampling |
|---|---:|---|---|---|---:|---:|---:|---|
| kaiLLM | 1 | code | plain | dspark3 | 46.6 | 52.1 | +11.7% | Model-recommended |
| kaiLLM | 1 | code | plain | dspark5 | 46.6 | 46.5 | -0.3% | Model-recommended |
| kaiLLM | 1 | code | dspark5 | dspark3 | 46.5 | 52.1 | +12.0% | Model-recommended |
| kaiLLM | 1 | prose | dspark5 | dspark3 | 17.9 | 26.2 | +46.1% | Model-recommended |

</details>

#### [kaitchup/Inkling-2-Mini-chimachima-kaiPTQ](https://huggingface.co/kaitchup/Inkling-2-Mini-chimachima-kaiPTQ/tree/1b27e927c15a6cde7f717926a25b2ce831d08fc1) (Q3_K_M)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain | prose | 1345.8 | Unsupported GGUF | 44.2 | Unsupported GGUF | Full graphs | Model-recommended |
| 1 | plain | code | 1342.2 | Unsupported GGUF | 44.2 | Unsupported GGUF | Full graphs | Model-recommended |

<details>
<summary><strong>Batch-one performance versus closest-size GGUF</strong></summary>

| Mode | Domain | Closest-size llama.cpp checkpoint | Prefill vs reference | Decode vs reference | Sampling |
|---|---|---|---:|---:|---|
| plain | code | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_M) | 157.2% | 96.6% | Model-recommended |
| plain | prose | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_M) | 156.8% | 96.4% | Model-recommended |

</details>

#### [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q2_K_XS)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | 975.2 | 802.2 | 46.4 | 41.0 | kaiLLM Full graphs · llama.cpp default (ub 512) | Archived greedy |
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | 977.2 | 802.8 | 46.4 | 38.2 | kaiLLM Full graphs · llama.cpp default (ub 512) | Model-recommended |
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | 995.2 | 893.3 | 47.7 | 51.9 | kaiLLM Full graphs · llama.cpp default (ub 512) | Archived greedy |
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | 970.8 | 803.1 | 46.4 | 41.0 | kaiLLM Full graphs · llama.cpp default (ub 512) | Archived greedy |
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | 973.9 | 804.1 | 46.4 | 38.2 | kaiLLM Full graphs · llama.cpp default (ub 512) | Model-recommended |
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | 991.5 | 885.2 | 47.7 | 51.8 | kaiLLM Full graphs · llama.cpp default (ub 512) | Archived greedy |
| 1 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | — | 875.9 | — | 67.1† | llama.cpp default (ub 512) | Archived greedy |
| 1 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | — | 782.9 | — | 41.2† | llama.cpp default (ub 512) | Model-recommended |
| 1 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | — | 868.0 | — | 94.1† | llama.cpp default (ub 512) | Archived greedy |
| 2 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | 975.2 | 777.8 | $\color{red}{23.6}$ | 60.1 | kaiLLM Full graphs · llama.cpp default (ub 512) | Archived greedy |
| 2 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | 978.1 | 778.4 | $\color{red}{23.5}$ | 60.1 | kaiLLM Full graphs · llama.cpp default (ub 512) | Archived greedy |

<details>
<summary><strong>DSpark decode comparisons</strong></summary>

| Engine | Batch | Domain | Baseline | Mode | Baseline decode | Measured decode | Change | Sampling |
|---|---:|---|---|---|---:|---:|---:|---|
| llama.cpp | 1 | code | plain | dspark5 | 51.8 | 94.1 | +81.8% | Archived greedy |
| llama.cpp | 1 | code | plain | dspark5 | 38.2 | 41.2 | +7.8% | Model-recommended |
| llama.cpp | 1 | prose | plain | dspark5 | 51.9 | 67.1 | +29.1% | Archived greedy |

</details>

#### [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_S)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | $\color{red}{251.7}$ | 866.4 | $\color{red}{19.7}$ | 48.8 | kaiLLM Full graphs · llama.cpp default (ub 512) | Archived greedy |
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | $\color{red}{249.9}$ | 845.1 | $\color{red}{19.7}$ | 48.9 | kaiLLM Full graphs · llama.cpp default (ub 512) | Archived greedy |
| 1 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | — | 849.2 | — | 44.5† | llama.cpp default (ub 512) | Archived greedy |
| 1 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | — | 907.0 | — | 19.5 | llama.cpp default (ub 512) | Model-recommended |
| 1 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | — | 846.3 | — | 63.0† | llama.cpp default (ub 512) | Archived greedy |
| 1 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | — | 908.0 | — | 59.2 | llama.cpp default (ub 512) | Model-recommended |

<details>
<summary><strong>DSpark decode comparisons</strong></summary>

| Engine | Batch | Domain | Baseline | Mode | Baseline decode | Measured decode | Change | Sampling |
|---|---:|---|---|---|---:|---:|---:|---|
| llama.cpp | 1 | code | plain | dspark5 | 48.9 | 63.0 | +28.9% | Archived greedy |
| llama.cpp | 1 | prose | plain | dspark5 | 48.8 | 44.5 | -9.0% | Archived greedy |

</details>

#### [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_M)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | $\color{red}{264.0}$ | 870.4 | $\color{red}{19.6}$ | 48.1 | kaiLLM Full graphs · llama.cpp default (ub 512) | Archived greedy |
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | $\color{red}{341.0}$ | 858.3 | $\color{red}{27.6}$ | 45.8 | kaiLLM Full graphs · `a59d68f0` · llama.cpp default (ub 512) | Model-recommended |
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | $\color{red}{265.7}$ | 858.3 | $\color{red}{19.7}$ | 45.8 | kaiLLM Full graphs · `c6f22a6f` · llama.cpp default (ub 512) | Model-recommended |
| 1 | plain | code | 333.1 | — | 27.4 | — | Full graphs | Archived greedy |
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | — | 861.3 | — | 47.9 | llama.cpp default (ub 512) | Archived greedy |
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | $\color{red}{340.9}$ | 853.6 | $\color{red}{27.6}$ | 45.8 | kaiLLM Full graphs · `a59d68f0` · llama.cpp default (ub 512) | Model-recommended |
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | $\color{red}{265.2}$ | 853.6 | $\color{red}{19.7}$ | 45.8 | kaiLLM Full graphs · `c6f22a6f` · llama.cpp default (ub 512) | Model-recommended |
| 1 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | — | 859.2 | — | 12.0† | llama.cpp default (ub 512) | Model-recommended |
| 1 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | — | 834.7 | — | 33.0† | llama.cpp default (ub 512) | Model-recommended |
| 2 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | — | 778.7 | — | 61.3 | llama.cpp default (ub 512) | Model-recommended |
| 2 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | $\color{red}{633.3}$ | 908.9 | $\color{red}{18.9}$ | 63.9 | kaiLLM Full graphs · llama.cpp default (ub 512) | Model-recommended |

<details>
<summary><strong>DSpark decode comparisons</strong></summary>

| Engine | Batch | Domain | Baseline | Mode | Baseline decode | Measured decode | Change | Sampling |
|---|---:|---|---|---|---:|---:|---:|---|
| llama.cpp | 1 | code | plain | dspark5 | 45.8 | 33.0 | -28.0% | Model-recommended |
| llama.cpp | 1 | prose | plain | dspark5 | 45.8 | 12.0 | -73.7% | Model-recommended |

</details>

<a id="gpu-3090-4"></a>

## 4 × RTX 3090

No full-length benchmark results.

<a id="gpu-5070-ti-1"></a>

## 1 × RTX 5070 Ti

<a id="gpu-5070-ti-1-model-lfm"></a>

### LFM3 1.7B

#### [LiquidAI/LFM3-1.7B-GGUF](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/tree/cd0c74c575b5c416f3011b5b4c298077f26ca8c2) (QAD-Q4_0)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain | prose | 28068.2 | 22181.7 | 357.0 | 212.6 | kaiLLM Full graphs · llama.cpp Forced graphs | Archived greedy |
| 1 | plain | code | 27786.0 | 24483.9 | 364.8 | 211.2 | kaiLLM Full graphs · llama.cpp Forced graphs | Archived greedy |
| 1 | ngram3 | prose | — | 23301.0 | — | 237.2† | Forced graphs | Archived greedy |
| 1 | ngram3 | code | — | 22684.8 | — | 220.9 | Forced graphs | Archived greedy |
| 2 | plain | prose | 28388.0 | 20238.0 | 593.6 | 153.9 | kaiLLM Full graphs · llama.cpp Forced graphs | Archived greedy |
| 2 | plain | code | 28397.1 | 23082.6 | 593.4 | 159.7 | kaiLLM Full graphs · llama.cpp Forced graphs | Archived greedy |
| 2 | ngram3 ([llama.cpp f4f6f2e](https://github.com/ggml-org/llama.cpp/commit/f4f6f2e7db76e98c95cc9b7fc608ea669c806044)) | prose | — | 25542.7 | — | 205.7 | llama.cpp default (ub 512) | Model-recommended |
| 2 | ngram3 ([llama.cpp f4f6f2e](https://github.com/ggml-org/llama.cpp/commit/f4f6f2e7db76e98c95cc9b7fc608ea669c806044)) | code | — | 25336.0 | — | 179.6 | llama.cpp default (ub 512) | Model-recommended |
| 4 | plain ([llama.cpp f4f6f2e](https://github.com/ggml-org/llama.cpp/commit/f4f6f2e7db76e98c95cc9b7fc608ea669c806044)) | prose | 28799.9 | 27151.9 | 857.2 | 225.8 | kaiLLM CUDA graphs captured at startup · llama.cpp default (ub 512) | Archived greedy |
| 4 | plain ([llama.cpp f4f6f2e](https://github.com/ggml-org/llama.cpp/commit/f4f6f2e7db76e98c95cc9b7fc608ea669c806044)) | code | 28361.3 | 26871.0 | 858.2 | 226.6 | kaiLLM CUDA graphs captured at startup · llama.cpp default (ub 512) | Archived greedy |
| 8 | plain ([llama.cpp f4f6f2e](https://github.com/ggml-org/llama.cpp/commit/f4f6f2e7db76e98c95cc9b7fc608ea669c806044)) | code | — | 25093.9 | — | 162.9 | llama.cpp default (ub 512) | Model-recommended |

#### [kaitchup/LFM3-1.7B-chimachima-kaiPTQ](https://huggingface.co/kaitchup/LFM3-1.7B-chimachima-kaiPTQ/tree/f9963f2a326732d3aa3202c316ebe9e7784999ab) (ITQ 750 MB)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain | prose | 17774.2 | Unsupported GGUF | 324.9 | Unsupported GGUF | Full graphs | Archived greedy |
| 1 | plain | code | 17848.8 | Unsupported GGUF | 280.2 | Unsupported GGUF | Default (prefill piecewise graphs, decode graphs, eager head) · `d0f8e74a` | Model-recommended |

#### [kaitchup/LFM3-1.7B-suisui-GGUF](https://huggingface.co/kaitchup/LFM3-1.7B-suisui-GGUF/tree/6fcfa6348c8399bd436b7ccd8a0c5e6c9dfe765b) (Q4_0_XS)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain | prose | 27783.7 | 25162.0 | 338.9 | 203.3 | kaiLLM Full graphs · llama.cpp Forced graphs | Archived greedy |
| 1 | plain | code | 27274.5 | 25123.5 | 342.8 | 207.8 | kaiLLM Full graphs · llama.cpp Forced graphs | Archived greedy |
| 1 | ngram3 ([llama.cpp f4f6f2e](https://github.com/ggml-org/llama.cpp/commit/f4f6f2e7db76e98c95cc9b7fc608ea669c806044)) | prose | — | 26519.0 | — | 203.9 | llama.cpp default (ub 512) | Model-recommended |
| 1 | ngram3 ([llama.cpp f4f6f2e](https://github.com/ggml-org/llama.cpp/commit/f4f6f2e7db76e98c95cc9b7fc608ea669c806044)) | code | — | 26014.2 | — | 220.1 | llama.cpp default (ub 512) | Model-recommended |
| 2 | plain | prose | 28265.0 | — | 539.5 | — | Full graphs | Archived greedy |
| 2 | plain ([llama.cpp f4f6f2e](https://github.com/ggml-org/llama.cpp/commit/f4f6f2e7db76e98c95cc9b7fc608ea669c806044)) | prose | — | 27760.8 | — | 199.8 | llama.cpp default (ub 512) | Archived greedy |
| 2 | plain | code | 28330.4 | — | 539.8 | — | Full graphs | Archived greedy |
| 2 | plain ([llama.cpp f4f6f2e](https://github.com/ggml-org/llama.cpp/commit/f4f6f2e7db76e98c95cc9b7fc608ea669c806044)) | code | — | 27379.4 | — | 194.7 | llama.cpp default (ub 512) | Archived greedy |
| 2 | ngram3 ([llama.cpp f4f6f2e](https://github.com/ggml-org/llama.cpp/commit/f4f6f2e7db76e98c95cc9b7fc608ea669c806044)) | prose | — | 25755.4 | — | 154.3 | llama.cpp default (ub 512) | Model-recommended |
| 2 | ngram3 ([llama.cpp f4f6f2e](https://github.com/ggml-org/llama.cpp/commit/f4f6f2e7db76e98c95cc9b7fc608ea669c806044)) | code | — | 25464.1 | — | 163.6 | llama.cpp default (ub 512) | Model-recommended |
| 4 | plain ([llama.cpp f4f6f2e](https://github.com/ggml-org/llama.cpp/commit/f4f6f2e7db76e98c95cc9b7fc608ea669c806044)) | prose | 28364.5 | 27214.1 | 802.2 | 222.9 | kaiLLM CUDA graphs captured at startup · llama.cpp default (ub 512) | Archived greedy |
| 4 | plain ([llama.cpp f4f6f2e](https://github.com/ggml-org/llama.cpp/commit/f4f6f2e7db76e98c95cc9b7fc608ea669c806044)) | code | 28231.2 | 26764.5 | 802.7 | 223.6 | kaiLLM CUDA graphs captured at startup · llama.cpp default (ub 512) | Archived greedy |
| 8 | plain ([llama.cpp f4f6f2e](https://github.com/ggml-org/llama.cpp/commit/f4f6f2e7db76e98c95cc9b7fc608ea669c806044)) | prose | 29173.2 | 27029.1 | 936.9 | 233.6 | kaiLLM CUDA graphs captured at startup · llama.cpp default (ub 512) | Archived greedy |
| 8 | plain ([llama.cpp f4f6f2e](https://github.com/ggml-org/llama.cpp/commit/f4f6f2e7db76e98c95cc9b7fc608ea669c806044)) | code | 28898.5 | 26974.7 | 934.2 | 235.7 | kaiLLM CUDA graphs captured at startup · llama.cpp default (ub 512) | Archived greedy |

<a id="gpu-5070-ti-1-model-inkling"></a>

### Inkling 2 Mini

#### [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/8078d149286eec7193795413b9e0643da6d11fa1) (UD-Q2_K_XL)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | 1254.0 | 1252.1 | 65.5 | 65.5 | kaiLLM Default (prefill eager, decode graphs, eager head) · `ba2b758b` · llama.cpp default (ub 256) | Archived greedy |
| 1 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | $\color{red}{274.5}$ | 1252.1 | $\color{red}{26.6}$ | 65.5 | kaiLLM CUDA graphs captured at startup · llama.cpp default (ub 256) | Archived greedy |
| 1 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | $\color{red}{283.3}$ | 1224.5 | $\color{red}{24.9}$ | 63.3 | kaiLLM Default (prefill eager, decode graphs, eager head) · `a8d3a360` · llama.cpp default (ub 256) | Archived greedy |
| 1 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | 1280.9 | 1224.5 | 66.5 | 63.3 | kaiLLM Default (prefill eager, decode graphs, eager head) · `ba2b758b` · llama.cpp default (ub 256) | Archived greedy |
| 1 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | 1246.5 | 1244.1 | 65.5 | 65.5 | kaiLLM Default (prefill eager, decode graphs, eager head) · `ba2b758b` · llama.cpp default (ub 256) | Archived greedy |
| 1 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | $\color{red}{274.6}$ | 1244.1 | $\color{red}{26.6}$ | 65.5 | kaiLLM CUDA graphs captured at startup · llama.cpp default (ub 256) | Archived greedy |
| 1 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | $\color{red}{284.1}$ | 1216.5 | $\color{red}{24.9}$ | 63.3 | kaiLLM Default (prefill eager, decode graphs, eager head) · `a8d3a360` · llama.cpp default (ub 256) | Archived greedy |
| 1 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | 1280.4 | 1216.5 | 66.5 | 63.3 | kaiLLM Default (prefill eager, decode graphs, eager head) · `ba2b758b` · llama.cpp default (ub 256) | Archived greedy |
| 1 | dspark5 ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | — | 990.8 | — | 81.8† | llama.cpp default (ub 128) · 4/8 greedy control texts matched | Archived greedy |
| 1 | dspark5 ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | — | 1000.5 | — | 123.9† | llama.cpp default (ub 128) · 4/8 greedy control texts matched | Archived greedy |
| 2 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | — | 1009.7 | — | 94.7 | llama.cpp default (ub 128) | Archived greedy |
| 2 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | — | 995.1 | — | 94.8 | llama.cpp default (ub 128) | Archived greedy |

<details>
<summary>Configuration limits (8)</summary>

| Engine | Mode | Batch | Observation |
|---|---|---:|---|
| llama.cpp default | dspark5 | 4 | The 9,248 MiB GPU KV allocation and at least 8,469 MiB of resident model weights exceed the 16,303 MiB device capacity. |
| llama.cpp default | dspark5 | 8 | The 18,496 MiB GPU KV allocation exceeds the 16,303 MiB device capacity. |
| llama.cpp default | dspark5 | 16 | The 36,992 MiB GPU KV allocation exceeds the 16,303 MiB device capacity. |
| llama.cpp default | dspark5 | 32 | The 73,984 MiB GPU KV allocation exceeds the 16,303 MiB device capacity. |
| llama.cpp default | plain | 4 | The 9,216 MiB GPU KV allocation and at least 8,469 MiB of resident model weights exceed the 16,303 MiB device capacity. |
| llama.cpp default | plain | 8 | The 18,432 MiB GPU KV allocation exceeds the 16,303 MiB device capacity. |
| llama.cpp default | plain | 16 | The 36,864 MiB GPU KV allocation exceeds the 16,303 MiB device capacity. |
| llama.cpp default | plain | 32 | The 73,728 MiB GPU KV allocation exceeds the 16,303 MiB device capacity. |

</details>

<details>
<summary><strong>DSpark decode comparisons</strong></summary>

| Engine | Batch | Domain | Baseline | Mode | Baseline decode | Measured decode | Change | Sampling |
|---|---:|---|---|---|---:|---:|---:|---|
| llama.cpp | 1 | code | plain | dspark5 | 65.5 | 123.9 | +89.2% | Archived greedy |
| llama.cpp | 1 | prose | plain | dspark5 | 65.5 | 81.8 | +24.8% | Archived greedy |

</details>

#### [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q2_K_XS)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | — | 1218.8 | — | 59.2 | llama.cpp default (ub 256) | Model-recommended |
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | 1390.2 | 1211.7 | 62.1 | 59.4 | kaiLLM Full graphs · llama.cpp default (ub 256) | Model-recommended |
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | $\color{red}{316.2}$ | 1209.5 | $\color{red}{25.5}$ | 59.2 | kaiLLM Full graphs · `c6f22a6f` · llama.cpp default (ub 256) | Model-recommended |
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | $\color{red}{316.4}$ | 1209.5 | 61.8 | 59.2 | kaiLLM Full graphs · `e1448455` · llama.cpp default (ub 256) | Model-recommended |
| 1 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | — | 1178.3 | — | 29.1† | llama.cpp default (ub 256) | Model-recommended |
| 1 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | — | 1176.1 | — | 64.8† | llama.cpp default (ub 256) | Model-recommended |

<details>
<summary><strong>DSpark decode comparisons</strong></summary>

| Engine | Batch | Domain | Baseline | Mode | Baseline decode | Measured decode | Change | Sampling |
|---|---:|---|---|---|---:|---:|---:|---|
| llama.cpp | 1 | code | plain | dspark5 | 59.2 | 64.8 | +9.6% | Model-recommended |
| llama.cpp | 1 | prose | plain | dspark5 | 59.2 | 29.1 | -50.8% | Model-recommended |

</details>

<a id="gpu-5080-1"></a>

## 1 × RTX 5080

<a id="gpu-5080-1-model-lfm"></a>

### LFM3 1.7B

#### [LiquidAI/LFM3-1.7B-GGUF](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/tree/cd0c74c575b5c416f3011b5b4c298077f26ca8c2) (QAD-Q4_0)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain | prose | 35520.7 | 32601.1 | 366.0 | 352.9 | Unrecorded | Archived greedy |
| 1 | plain | code | 35509.3 | 32566.5 | 366.0 | 354.2 | Unrecorded | Archived greedy |
| 1 | ngram3 | prose | 35356.5 | 32688.4 | 1355.9 | 534.4† | Unrecorded | Archived greedy |
| 1 | ngram3 | code | 35439.6 | 32520.1 | 1187.5† | 459.4† | Unrecorded | Archived greedy |
| 2 | plain | prose | 36105.4 | 32007.9 | 590.0 | 245.3 | Unrecorded | Archived greedy |
| 2 | plain | code | 35919.8 | 31828.1 | 590.6 | 245.6 | Unrecorded | Archived greedy |
| 2 | ngram3 | prose | 36092.2 | 31904.7 | 2047.2 | 510.3† | Unrecorded | Archived greedy |
| 2 | ngram3 | code | 36047.5 | 31927.0 | 1748.6† | 413.5† | Unrecorded | Archived greedy |
| 4 | plain | prose | 36438.0 | 31777.2 | 883.3 | 274.9 | Unrecorded | Archived greedy |
| 4 | plain | code | 36258.5 | 31719.4 | 881.4 | 276.3 | Unrecorded | Archived greedy |
| 4 | ngram3 | prose | 36571.0 | — | 2392.4 | — | Unrecorded | Archived greedy |
| 4 | ngram3 | code | 36478.8 | — | 2584.8† | — | Unrecorded | Archived greedy |
| 8 | plain | prose | 36520.3 | 31559.3 | 1122.2 | 289.6 | Unrecorded | Archived greedy |
| 8 | plain | code | 36156.5 | 31480.2 | 1121.8 | 288.4 | Unrecorded | Archived greedy |
| 8 | ngram1 (llama.cpp map-k) | prose | 36429.9 | 31496.3 | 2026.5† | 310.3† | Unrecorded | Archived greedy |
| 8 | ngram1 | code | 36216.8 | — | 1991.6† | — | Unrecorded | Archived greedy |

<details>
<summary>Configuration limits (1)</summary>

| Engine | Mode | Batch | Observation |
|---|---|---:|---|
| llama.cpp | plain | 32 | Memory allocation failed at the requested context |

</details>

#### [kaitchup/LFM3-1.7B-chimachima-kaiPTQ](https://huggingface.co/kaitchup/LFM3-1.7B-chimachima-kaiPTQ/tree/f9963f2a326732d3aa3202c316ebe9e7784999ab) (ITQ 750 MB)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain | prose | 21915.6 | Unsupported GGUF | 326.8 | Unsupported GGUF | Unrecorded | Archived greedy |
| 1 | plain | code | 21869.9 | Unsupported GGUF | 326.8 | Unsupported GGUF | Unrecorded | Archived greedy |
| 1 | ngram3 | prose | 21858.9 | Unsupported GGUF | 1173.6 | Unsupported GGUF | Unrecorded | Archived greedy |
| 1 | ngram3 | code | 21855.1 | Unsupported GGUF | 1072.8† | Unsupported GGUF | Unrecorded | Archived greedy |
| 2 | plain | prose | 22252.7 | Unsupported GGUF | 529.2 | Unsupported GGUF | Unrecorded | Archived greedy |
| 2 | plain | code | 22289.1 | Unsupported GGUF | 529.2 | Unsupported GGUF | Unrecorded | Archived greedy |
| 2 | ngram3 | prose | 22239.5 | Unsupported GGUF | 1867.7† | Unsupported GGUF | Unrecorded | Archived greedy |
| 2 | ngram3 | code | 22376.0 | Unsupported GGUF | 1852.4† | Unsupported GGUF | Unrecorded | Archived greedy |
| 4 | plain | prose | 22538.8 | Unsupported GGUF | 800.3 | Unsupported GGUF | Default (prefill piecewise graphs, decode full graphs + piecewise graphs, eager head) · `d0f8e74a` | Archived greedy |
| 4 | plain | code | 22502.2 | Unsupported GGUF | 798.8 | Unsupported GGUF | Default (prefill piecewise graphs, decode full graphs + piecewise graphs, eager head) · `d0f8e74a` | Archived greedy |

#### [kaitchup/LFM3-1.7B-suisui-GGUF](https://huggingface.co/kaitchup/LFM3-1.7B-suisui-GGUF/tree/6fcfa6348c8399bd436b7ccd8a0c5e6c9dfe765b) (Q4_0_XS)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain | prose | 35111.7 | 32097.0 | 339.5 | 327.3 | Unrecorded | Archived greedy |
| 1 | plain | code | 35123.0 | 32245.6 | 339.3 | 327.7 | Unrecorded | Archived greedy |
| 1 | ngram3 | prose | 35250.0 | 32050.0 | 1220.4 | 593.5 | Unrecorded | Archived greedy |
| 1 | ngram3 | code | 35213.8 | 32214.8 | 1158.6† | 463.7† | Unrecorded | Archived greedy |
| 2 | plain | prose | 35857.3 | 31700.2 | 539.4 | 239.7 | Unrecorded | Archived greedy |
| 2 | plain | code | 35861.4 | 31674.2 | 539.5 | 240.7 | Unrecorded | Archived greedy |
| 2 | ngram3 | prose | 36233.8 | 31627.3 | 1919.6† | 551.6† | Unrecorded | Archived greedy |
| 2 | ngram3 | code | 35768.8 | 31554.7 | 1759.7† | 424.9† | Unrecorded | Archived greedy |
| 4 | plain | prose | 36115.8 | 31433.4 | 825.9 | 271.4 | Unrecorded | Archived greedy |
| 4 | plain | code | 35994.4 | 31386.3 | 824.5 | 272.5 | Unrecorded | Archived greedy |
| 4 | ngram3 | prose | 36154.5 | 31338.6 | 2723.1† | 518.8† | Unrecorded | Archived greedy |
| 4 | ngram3 | code | 36064.4 | 31233.2 | 2525.8† | 365.2† | Unrecorded | Archived greedy |

<details>
<summary>Configuration limits (1)</summary>

| Engine | Mode | Batch | Observation |
|---|---|---:|---|
| llama.cpp | plain | 16 | Memory allocation failed at the requested context |

</details>

<a id="gpu-5080-1-model-daisy"></a>

### DaisyRose 27B

#### [bartowski/RadixArk_DaisyRose-27B-GGUF](https://huggingface.co/bartowski/RadixArk_DaisyRose-27B-GGUF/tree/4514fec3c6573951bce8acc7f850d9eb496301c7) (IQ2_M)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain | prose | — | 1959.7 | — | 43.3 | Unrecorded | Archived greedy |
| 1 | plain | code | — | 1956.7 | — | 43.3 | Unrecorded | Archived greedy |

<details>
<summary>Configuration limits (3)</summary>

| Engine | Mode | Batch | Observation |
|---|---|---:|---|
| llama.cpp | mtp5 | 1 | Memory allocation failed at the requested context |
| llama.cpp | mtp5 | 2 | Memory allocation failed at the requested context |
| llama.cpp | plain | 2 | Memory allocation failed at the requested context |

</details>

#### [kaitchup/DaisyRose-27B-chimachima-kaiPTQ](https://huggingface.co/kaitchup/DaisyRose-27B-chimachima-kaiPTQ/tree/6c1a63358c57f5baf2730574ba418195eb7edfa8) (Q2_K_XS)

<details>
<summary>Configuration limits (2)</summary>

| Engine | Mode | Batch | Observation |
|---|---|---:|---|
| kaiLLM | dspine15 | 1 | Memory allocation failed at the requested context |
| kaiLLM | dspine7 | 1 | Memory allocation failed at the requested context |

</details>

#### [kaitchup/DaisyRose-27B-dondon-GGUF](https://huggingface.co/kaitchup/DaisyRose-27B-dondon-GGUF/tree/3592e073925dbdbc65a340fca851fc01b8d2c409) (IQ2_M)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain | prose | — | 1942.0 | — | 42.8 | Unrecorded | Archived greedy |
| 1 | plain | code | — | 1937.9 | — | 42.8 | Unrecorded | Archived greedy |

<details>
<summary>Configuration limits (3)</summary>

| Engine | Mode | Batch | Observation |
|---|---|---:|---|
| llama.cpp | mtp5 | 1 | Memory allocation failed at the requested context |
| llama.cpp | mtp5 | 2 | Memory allocation failed at the requested context |
| llama.cpp | plain | 2 | Memory allocation failed at the requested context |

</details>

<a id="gpu-5080-1-model-qwen"></a>

### Qwen3.8 27B

#### [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/4af51b3542f775b0a8e4b310bfb3f17e01c306c9) (UD-Q2_K_XL)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain | prose | — | 2177.4 | — | 43.5 | Unrecorded | Archived greedy |
| 1 | plain | code | — | 2168.4 | — | 43.5 | Unrecorded | Archived greedy |

<details>
<summary>Configuration limits (2)</summary>

| Engine | Mode | Batch | Observation |
|---|---|---:|---|
| llama.cpp | mtp5 | 2 | Memory allocation failed at the requested context |
| llama.cpp | plain | 2 | Memory allocation failed at the requested context |

</details>

#### [kaitchup/Qwen3.8-27B-chimachima-kaiPTQ](https://huggingface.co/kaitchup/Qwen3.8-27B-chimachima-kaiPTQ/tree/aa5396581a0d09201312c0262dcfa2d0e1871eba) (Q2_K_XXS)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain | prose | 1682.2 | Unsupported GGUF | 68.5 | Unsupported GGUF | Unrecorded | Archived greedy |
| 1 | plain | code | 1679.7 | Unsupported GGUF | 68.5 | Unsupported GGUF | Unrecorded | Archived greedy |

<details>
<summary>Configuration limits (2)</summary>

| Engine | Mode | Batch | Observation |
|---|---|---:|---|
| kaiLLM | dspine15 | 1 | Memory allocation failed at the requested context |
| kaiLLM | dspine7 | 1 | Memory allocation failed at the requested context |

</details>

<a id="gpu-5080-1-model-inkling"></a>

### Inkling 2 Mini

#### [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/8078d149286eec7193795413b9e0643da6d11fa1) (UD-Q2_K_XL)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | 1567.4 | 1418.8 | $\color{red}{63.8}$ | 70.9 | kaiLLM Default (prefill eager, decode graphs, eager head) · `4c744485` · llama.cpp default (ub 256) | Archived greedy |
| 1 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | $\color{red}{316.7}$ | 1418.8 | $\color{red}{27.2}$ | 70.9 | kaiLLM Default (prefill eager, decode graphs, eager head) · `a8d3a360` · llama.cpp default (ub 256) | Archived greedy |
| 1 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | 1567.3 | 1418.8 | 72.5 | 70.9 | kaiLLM Default (prefill eager, decode graphs, eager head) · `a989472f` · llama.cpp default (ub 256) | Archived greedy |
| 1 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | 1556.5 | 1413.9 | $\color{red}{63.8}$ | 70.8 | kaiLLM Default (prefill eager, decode graphs, eager head) · `4c744485` · llama.cpp default (ub 256) | Archived greedy |
| 1 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | $\color{red}{316.9}$ | 1413.9 | $\color{red}{27.2}$ | 70.8 | kaiLLM Default (prefill eager, decode graphs, eager head) · `a8d3a360` · llama.cpp default (ub 256) | Archived greedy |
| 1 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | 1556.1 | 1413.9 | 72.5 | 70.8 | kaiLLM Default (prefill eager, decode graphs, eager head) · `a989472f` · llama.cpp default (ub 256) | Archived greedy |
| 1 | dspark5 ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | — | 1139.3 | — | 94.4† | llama.cpp default (ub 128) | Archived greedy |
| 1 | dspark5 ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | — | 1132.3 | — | 144.4† | llama.cpp default (ub 128) | Archived greedy |
| 2 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | 1572.8 | 1150.5 | 115.2 | 107.0 | kaiLLM Default (prefill eager, decode graphs + eager, eager head) · `67251f9b` · B2 verified on workload (1.65 nominal slots) · llama.cpp default (ub 128) | Archived greedy |
| 2 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | 1561.8 | 1143.7 | 114.9 | 106.7 | kaiLLM Default (prefill eager, decode graphs + eager, eager head) · `67251f9b` · B2 verified on workload (1.65 nominal slots) · llama.cpp default (ub 128) | Archived greedy |

<details>
<summary>Configuration limits (1)</summary>

| Engine | Mode | Batch | Observation |
|---|---|---:|---|
| llama.cpp default | dspark5 | 2 | On the tested 16,303 MiB board, allocation failed for target compute at ub512/256, draft weights at ub128, draft KV at ub64. |

</details>

<details>
<summary><strong>DSpark decode comparisons</strong></summary>

| Engine | Batch | Domain | Baseline | Mode | Baseline decode | Measured decode | Change | Sampling |
|---|---:|---|---|---|---:|---:|---:|---|
| llama.cpp | 1 | code | plain | dspark5 | 70.8 | 144.4 | +103.8% | Archived greedy |
| llama.cpp | 1 | prose | plain | dspark5 | 70.9 | 94.4 | +33.1% | Archived greedy |

</details>

#### [kaitchup/Inkling-2-Mini-chimachima-kaiPTQ](https://huggingface.co/kaitchup/Inkling-2-Mini-chimachima-kaiPTQ/tree/1b27e927c15a6cde7f717926a25b2ce831d08fc1) (Q2_K_XS)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain (earlier kaiLLM · deterministic Humming) | code | 1823.5 | Unsupported GGUF | 61.2 | Unsupported GGUF | Full graphs | Model-recommended |
| 1 | dspark5 (deterministic kaiLLM 46d0dc4) | prose | 1376.5 | Unsupported GGUF | 28.9 | Unsupported GGUF | Full graphs · 512-token prefill chunks | Model-recommended |
| 1 | dspark5 (deterministic kaiLLM 46d0dc4) | code | 1363.3 | Unsupported GGUF | 74.8 | Unsupported GGUF | Full graphs · 512-token prefill chunks | Model-recommended |
| 2 | plain | code | 1384.0 | Unsupported GGUF | 38.4 | Unsupported GGUF | Full graphs · 512-token prefill chunks | Model-recommended |

<details>
<summary><strong>Batch-one performance versus closest-size GGUF</strong></summary>

| Mode | Domain | Closest-size llama.cpp checkpoint | Prefill vs reference | Decode vs reference | Sampling |
|---|---|---|---:|---:|---|
| dspark5 | code | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q2_K_XS) | 99.3% | 82.3% | Model-recommended |
| dspark5 | prose | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q2_K_XS) | 100.1% | 67.2% | Model-recommended |
| plain | code | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q2_K_XS) | 129.2% | 98.6% | Model-recommended |

</details>

<details>
<summary><strong>DSpark decode comparisons</strong></summary>

| Engine | Batch | Domain | Baseline | Mode | Baseline decode | Measured decode | Change | Sampling |
|---|---:|---|---|---|---:|---:|---:|---|
| kaiLLM | 1 | code | plain | dspark5 | 61.2 | 74.8 | +22.2% | Model-recommended |

</details>

#### [kaitchup/Inkling-2-Mini-chimachima-kaiPTQ](https://huggingface.co/kaitchup/Inkling-2-Mini-chimachima-kaiPTQ/tree/1b27e927c15a6cde7f717926a25b2ce831d08fc1) (Q3_K_S)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain (earlier kaiLLM · deterministic Humming) | code | 2169.8 | Unsupported GGUF | 64.8 | Unsupported GGUF | Full graphs · 1,024-token prefill chunks | Model-recommended |
| 1 | dspark5 (deterministic kaiLLM 16b936a) | prose | 1167.1 | Unsupported GGUF | 31.3 | Unsupported GGUF | Full graphs · `16b936a9` · 256-token prefill chunks | Model-recommended |
| 1 | dspark5 (deterministic kaiLLM 3116655) | prose | 568.6 | Unsupported GGUF | 37.8 | Unsupported GGUF | Full graphs · `31166554` · 64-token prefill chunks | Model-recommended |
| 1 | dspark5 (deterministic kaiLLM 16b936a) | code | 1144.6 | Unsupported GGUF | 86.5 | Unsupported GGUF | Full graphs · `16b936a9` · 256-token prefill chunks | Model-recommended |
| 1 | dspark5 (deterministic kaiLLM 3116655) | code | 591.7 | Unsupported GGUF | 63.3 | Unsupported GGUF | Full graphs · `31166554` · 64-token prefill chunks | Model-recommended |
| 2 | plain | code | 617.7 | Unsupported GGUF | 38.8 | Unsupported GGUF | Full graphs · 64-token prefill chunks | Model-recommended |

<details>
<summary><strong>Batch-one performance versus closest-size GGUF</strong></summary>

| Mode | Domain | Closest-size llama.cpp checkpoint | Prefill vs reference | Decode vs reference | Sampling |
|---|---|---|---:|---:|---|
| dspark5 | code | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_S) | 106.6% | 111.5% | Model-recommended |
| dspark5 | prose | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_S) | 107.9% | 58.2% | Model-recommended |
| plain | code | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_S) | 154.4% | 107.0% | Model-recommended |

</details>

<details>
<summary><strong>DSpark decode comparisons</strong></summary>

| Engine | Batch | Domain | Baseline | Mode | Baseline decode | Measured decode | Change | Sampling |
|---|---:|---|---|---|---:|---:|---:|---|
| kaiLLM | 1 | code | plain | dspark5 | 64.8 | 86.5 | +33.5% | Model-recommended |

</details>

#### [kaitchup/Inkling-2-Mini-chimachima-kaiPTQ](https://huggingface.co/kaitchup/Inkling-2-Mini-chimachima-kaiPTQ/tree/1b27e927c15a6cde7f717926a25b2ce831d08fc1) (Q3_K_M)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain | code | 570.6 | Unsupported GGUF | 62.8 | Unsupported GGUF | Full graphs · 64-token prefill chunks | Model-recommended |

<details>
<summary><strong>Batch-one performance versus closest-size GGUF</strong></summary>

| Mode | Domain | Closest-size llama.cpp checkpoint | Prefill vs reference | Decode vs reference | Sampling |
|---|---|---|---:|---:|---|
| plain | code | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_M) | 51.5% | 107.1% | Model-recommended |

</details>

#### [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q2_K_XS)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | 1681.1 | 1410.9 | 67.4 | 62.1 | kaiLLM Full graphs · llama.cpp default (ub 256) | Model-recommended |
| 1 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | — | 1374.5 | — | 43.0 | llama.cpp default (ub 256) | Model-recommended |
| 1 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | — | 1372.8 | — | 90.9† | llama.cpp default (ub 256) | Model-recommended |

<details>
<summary><strong>DSpark decode comparisons</strong></summary>

| Engine | Batch | Domain | Baseline | Mode | Baseline decode | Measured decode | Change | Sampling |
|---|---:|---|---|---|---:|---:|---:|---|
| llama.cpp | 1 | code | plain | dspark5 | 62.1 | 90.9 | +46.4% | Model-recommended |

</details>

#### [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_S)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | $\color{red}{925.1}$ | 1405.3 | $\color{red}{48.9}$ | 60.6 | kaiLLM Full graphs · `8a0cf56f` · llama.cpp default (ub 256) | Model-recommended |
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | $\color{red}{1250.2}$ | 1405.3 | 70.2 | 60.6 | kaiLLM Full graphs · `bdaaef83` · llama.cpp default (ub 256) | Model-recommended |
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | 1307.3 | 1405.3 | 70.3 | 60.6 | kaiLLM Full graphs · `eee78c36` · llama.cpp default (ub 256) | Model-recommended |
| 1 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | — | 1081.4 | — | 53.7 | llama.cpp default (ub 128) | Model-recommended |
| 1 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | — | 1074.0 | — | 77.6† | llama.cpp default (ub 128) | Model-recommended |

<details>
<summary><strong>DSpark decode comparisons</strong></summary>

| Engine | Batch | Domain | Baseline | Mode | Baseline decode | Measured decode | Change | Sampling |
|---|---:|---|---|---|---:|---:|---:|---|
| llama.cpp | 1 | code | plain | dspark5 | 60.6 | 77.6 | +28.0% | Model-recommended |

</details>

#### [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_M)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | $\color{red}{387.2}$ | 1107.6 | $\color{red}{28.8}$ | 58.7 | kaiLLM Full graphs · llama.cpp default (ub 128) | Model-recommended |

<a id="gpu-5080-4"></a>

## 4 × RTX 5080

<a id="gpu-5080-4-model-hy4"></a>

### Hy4 Flash

#### [kaitchup/Hy4-flash-dondon-GGUF](https://huggingface.co/kaitchup/Hy4-flash-dondon-GGUF/tree/ee58ad539c09c0b2b290a06f17c99e638a0db139) (Q2_K_M)

<details>
<summary>Configuration limits (1)</summary>

| Engine | Mode | Batch | Observation |
|---|---|---:|---|
| llama.cpp default | plain | 1 | One GPU failed to initialize CUDA before and after a container reboot. |

</details>

<a id="gpu-5090-1"></a>

## 1 × RTX 5090

<a id="gpu-5090-1-model-lfm"></a>

### LFM3 1.7B

#### [LiquidAI/LFM3-1.7B-GGUF](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/tree/cd0c74c575b5c416f3011b5b4c298077f26ca8c2) (QAD-Q4_0)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain | prose | 53571.1 | 34327.9 | 559.4 | 412.5 | Unrecorded | Archived greedy |
| 1 | plain | code | 53218.8 | 33115.7 | 559.8 | 419.0 | Unrecorded | Archived greedy |
| 1 | ngram3 | prose | 52814.2 | 34260.5 | 2016.7 | 480.3† | Unrecorded | Archived greedy |
| 1 | ngram3 | code | 51734.5 | 33584.8 | 1912.6 | 451.9† | Unrecorded | Archived greedy |
| 2 | plain | prose | 58611.0 | 32404.8 | 922.9 | 344.4 | Unrecorded | Archived greedy |
| 2 | plain | code | 56988.6 | 32880.5 | 924.2 | 343.7 | Unrecorded | Archived greedy |
| 2 | ngram3 | prose | 56198.7 | 32353.2 | 3155.0 | 407.8† | Unrecorded | Archived greedy |
| 2 | ngram3 | code | 58024.8 | 32158.6 | 2917.5† | 331.5† | Unrecorded | Archived greedy |
| 4 | plain | prose | 58631.7 | 32206.9 | 1417.8 | 381.7 | Unrecorded | Archived greedy |
| 4 | plain | code | 59293.6 | 32048.1 | 1415.9 | 385.5 | Unrecorded | Archived greedy |
| 4 | ngram3 | prose | 59427.2 | 31964.1 | 3418.9† | 395.9† | Unrecorded | Archived greedy |
| 4 | ngram3 | code | 58170.5 | 32184.9 | 3603.2† | 310.6† | Unrecorded | Archived greedy |

#### [kaitchup/LFM3-1.7B-chimachima-kaiPTQ](https://huggingface.co/kaitchup/LFM3-1.7B-chimachima-kaiPTQ/tree/f9963f2a326732d3aa3202c316ebe9e7784999ab) (ITQ 750 MB)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain | prose | 37731.9 | Unsupported GGUF | 450.9 | Unsupported GGUF | Unrecorded | Archived greedy |
| 1 | plain | code | 37592.6 | Unsupported GGUF | 451.1 | Unsupported GGUF | Unrecorded | Archived greedy |
| 1 | ngram3 | prose | 38786.6 | Unsupported GGUF | 1603.8 | Unsupported GGUF | Unrecorded | Archived greedy |
| 1 | ngram3 | code | 37272.3 | Unsupported GGUF | 1599.9† | Unsupported GGUF | Unrecorded | Archived greedy |
| 2 | plain | prose | 39879.6 | Unsupported GGUF | 753.9 | Unsupported GGUF | Unrecorded | Archived greedy |
| 2 | plain | code | 39793.9 | Unsupported GGUF | 753.8 | Unsupported GGUF | Unrecorded | Archived greedy |
| 2 | ngram3 | prose | 40613.0 | Unsupported GGUF | 1574.0† | Unsupported GGUF | Unrecorded | Archived greedy |
| 2 | ngram3 | code | 38943.9 | Unsupported GGUF | 2670.0† | Unsupported GGUF | Unrecorded | Archived greedy |
| 4 | plain | prose | 41032.6 | Unsupported GGUF | 1194.8 | Unsupported GGUF | Unrecorded | Archived greedy |
| 4 | plain | code | 40494.9 | Unsupported GGUF | 1193.8 | Unsupported GGUF | Unrecorded | Archived greedy |
| 4 | ngram3 | prose | 40770.4 | Unsupported GGUF | 4490.3† | Unsupported GGUF | Unrecorded | Archived greedy |
| 4 | ngram3 | code | 41304.5 | Unsupported GGUF | 4363.2† | Unsupported GGUF | Unrecorded | Archived greedy |

#### [kaitchup/LFM3-1.7B-suisui-GGUF](https://huggingface.co/kaitchup/LFM3-1.7B-suisui-GGUF/tree/6fcfa6348c8399bd436b7ccd8a0c5e6c9dfe765b) (Q4_0_XS)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain | prose | 52662.8 | 33857.7 | 523.8 | 394.2 | Unrecorded | Archived greedy |
| 1 | plain | code | 52829.1 | 34539.5 | 524.0 | 414.8 | Unrecorded | Archived greedy |
| 1 | ngram3 | prose | 51606.1 | 34036.5 | 1807.1† | 525.7 | Unrecorded | Archived greedy |
| 1 | ngram3 | code | 51467.9 | 33956.2 | 1804.7† | 423.2† | Unrecorded | Archived greedy |
| 2 | plain | prose | 57605.2 | 33287.1 | 834.7 | 339.5 | Unrecorded | Archived greedy |
| 2 | plain | code | 56770.6 | 32735.7 | 834.4 | 341.0 | Unrecorded | Archived greedy |
| 2 | ngram3 | prose | 57189.5 | 32566.9 | 2912.6† | 414.2† | Unrecorded | Archived greedy |
| 2 | ngram3 | code | 57069.6 | 32249.8 | 2689.3† | 322.5† | Unrecorded | Archived greedy |
| 4 | plain | prose | 58750.6 | 32425.0 | 1311.3 | 377.3 | Unrecorded | Archived greedy |
| 4 | plain | code | 58544.6 | 32178.8 | 1309.7 | 379.5 | Unrecorded | Archived greedy |
| 4 | ngram3 | prose | 58920.4 | 32120.1 | 3985.3† | 407.2† | Unrecorded | Archived greedy |
| 4 | ngram3 | code | 57635.3 | 32029.3 | 3754.2† | 365.3† | Unrecorded | Archived greedy |

<a id="gpu-5090-1-model-daisy"></a>

### DaisyRose 27B

#### [kaitchup/DaisyRose-27B-chimachima-kaiPTQ](https://huggingface.co/kaitchup/DaisyRose-27B-chimachima-kaiPTQ/tree/6c1a63358c57f5baf2730574ba418195eb7edfa8) (Q2_K_XS)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain | prose | 3097.6 | Unsupported GGUF | 94.9 | Unsupported GGUF | Unrecorded | Archived greedy |
| 1 | plain | code | 3063.4 | Unsupported GGUF | 94.9 | Unsupported GGUF | Unrecorded | Archived greedy |
| 1 | dspine7 | prose | 3068.7 | Unsupported GGUF | 411.5† | Unsupported GGUF | Unrecorded | Archived greedy |
| 1 | dspine7 | code | 3039.5 | Unsupported GGUF | 412.9† | Unsupported GGUF | Unrecorded | Archived greedy |
| 1 | dspine15 | prose | 3073.1 | Unsupported GGUF | 237.4† | Unsupported GGUF | Unrecorded | Archived greedy |
| 1 | dspine15 | code | 3072.5 | Unsupported GGUF | 241.7† | Unsupported GGUF | Unrecorded | Archived greedy |
| 2 | plain | prose | 3104.7 | Unsupported GGUF | 146.7 | Unsupported GGUF | Unrecorded | Archived greedy |
| 2 | plain | code | 3086.5 | Unsupported GGUF | 146.7 | Unsupported GGUF | Unrecorded | Archived greedy |

<a id="gpu-5090-1-model-qwen"></a>

### Qwen3.8 27B

#### [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/4af51b3542f775b0a8e4b310bfb3f17e01c306c9) (UD-Q2_K_XL)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain | prose | 4085.3 | 3474.2 | 96.2 | 72.5 | Unrecorded | Archived greedy |
| 1 | plain | code | 4026.7 | 3478.9 | 96.1 | 72.4 | Unrecorded | Archived greedy |
| 1 | mtp5 | prose | 4006.5 | 3252.8 | 275.6† | 208.4† | Unrecorded | Archived greedy |
| 1 | mtp5 | code | 3959.1 | 3239.6 | 292.8† | 204.2† | Unrecorded | Archived greedy |
| 1 | dspine7 | prose | 4035.9 | — | 294.3† | — | Unrecorded | Archived greedy |
| 1 | dspine7 | code | 3982.3 | — | 296.2† | — | Unrecorded | Archived greedy |
| 1 | dspine15 | prose | 4032.7 | — | 155.0† | — | Unrecorded | Archived greedy |
| 1 | dspine15 | code | 4017.8 | — | 221.5† | — | Unrecorded | Archived greedy |
| 2 | plain | prose | 4087.2 | 3471.6 | 153.6 | 81.1 | Unrecorded | Archived greedy |
| 2 | plain | code | 4052.3 | 3441.0 | 153.5 | 81.1 | Unrecorded | Archived greedy |
| 2 | mtp5 | prose | 4006.2 | 3195.4 | 447.8† | 236.7† | Unrecorded | Archived greedy |
| 2 | mtp5 | code | 3959.0 | 3191.4 | 472.0† | 256.4† | Unrecorded | Archived greedy |

#### [kaitchup/Qwen3.8-27B-chimachima-kaiPTQ](https://huggingface.co/kaitchup/Qwen3.8-27B-chimachima-kaiPTQ/tree/aa5396581a0d09201312c0262dcfa2d0e1871eba) (Q2_K_XXS)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain | prose | 3092.8 | Unsupported GGUF | 110.5 | Unsupported GGUF | Unrecorded | Archived greedy |
| 1 | plain | code | 3068.2 | Unsupported GGUF | 110.5 | Unsupported GGUF | Unrecorded | Archived greedy |
| 1 | dspine7 | prose | 3072.5 | Unsupported GGUF | 464.3† | Unsupported GGUF | Unrecorded | Archived greedy |
| 1 | dspine7 | code | 3034.8 | Unsupported GGUF | 464.9† | Unsupported GGUF | Unrecorded | Archived greedy |
| 1 | dspine15 | prose | 3061.1 | Unsupported GGUF | 192.0† | Unsupported GGUF | Unrecorded | Archived greedy |
| 1 | dspine15 | code | 3060.3 | Unsupported GGUF | 242.7† | Unsupported GGUF | Unrecorded | Archived greedy |
| 2 | plain | prose | 3088.2 | Unsupported GGUF | 168.0 | Unsupported GGUF | Unrecorded | Archived greedy |
| 2 | plain | code | 3076.6 | Unsupported GGUF | 168.0 | Unsupported GGUF | Unrecorded | Archived greedy |

<a id="gpu-5090-1-model-inkling"></a>

### Inkling 2 Mini

#### [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/8078d149286eec7193795413b9e0643da6d11fa1) (UD-Q2_K_XL)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | 3137.2 | 1989.6 | 89.9 | 82.5 | kaiLLM Default (prefill eager, decode graphs, eager head) · `800ab62d` · llama.cpp default (ub 512) | Archived greedy |
| 1 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | $\color{red}{521.2}$ | 1989.6 | $\color{red}{30.8}$ | 82.5 | kaiLLM Default (prefill eager, decode graphs, eager head) · `a8d3a360` · llama.cpp default (ub 512) | Archived greedy |
| 1 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | 3112.1 | 2005.0 | 89.9 | 82.5 | kaiLLM Default (prefill eager, decode graphs, eager head) · `800ab62d` · llama.cpp default (ub 512) | Archived greedy |
| 1 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | $\color{red}{522.6}$ | 2005.0 | $\color{red}{30.8}$ | 82.5 | kaiLLM Default (prefill eager, decode graphs, eager head) · `a8d3a360` · llama.cpp default (ub 512) | Archived greedy |
| 1 | dspark5 ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | — | 1931.9 | — | 41.2 | llama.cpp default (ub 512) | Model-recommended |
| 1 | dspark5 ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | — | 1917.6 | — | 82.2 | llama.cpp default (ub 512) | Model-recommended |
| 2 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | — | 1925.7 | — | 118.2 | llama.cpp default (ub 512) | Model-recommended |

#### [ggml-org/Inkling-2-Mini-GGUF](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/tree/0bdc580634ceac4c4d921f452777bf44228de9af) (IQ4_NL)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain (Inkling PR) | prose | — | 2010.7 | — | 81.2 | Unrecorded | Archived greedy |
| 1 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | 3251.2 | 2038.0 | $\color{red}{65.9}$ | 80.2 | kaiLLM Default (prefill eager, decode graphs, eager head) · `328c2ce9` · llama.cpp default (ub 512) | Archived greedy |
| 1 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | 3243.3 | 2038.0 | 98.3 | 80.2 | kaiLLM Default (prefill eager, decode graphs, eager head) · `6758422b` · llama.cpp default (ub 512) | Archived greedy |
| 1 | plain (earlier kaiLLM · Humming unrecorded · [Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | $\color{red}{1094.3}$ | 2038.0 | $\color{red}{62.1}$ | 80.2 | kaiLLM Decode-only capture · llama.cpp default (ub 512) | Archived greedy |
| 1 | plain (earlier kaiLLM · Humming unrecorded) | prose | $\color{red}{1049.9}$ | 1996.8 | $\color{red}{61.8}$ | 80.8 | Unrecorded | Archived greedy |
| 1 | plain (earlier kaiLLM · deterministic Humming · [Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | $\color{red}{1067.3}$ | 2038.0 | $\color{red}{46.2}$ | 80.2 | kaiLLM Full graphs · llama.cpp default (ub 512) | Archived greedy |
| 1 | plain (earlier kaiLLM · deterministic Humming) | prose | $\color{red}{1067.3}$ | 2052.8 | $\color{red}{46.2}$ | 80.5 | kaiLLM Full graphs · llama.cpp Forced graphs | Archived greedy |
| 1 | plain (Inkling PR) | code | — | 1997.0 | — | 81.2 | Unrecorded | Archived greedy |
| 1 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | 3231.4 | 2018.0 | $\color{red}{65.9}$ | 80.1 | kaiLLM Default (prefill eager, decode graphs, eager head) · `328c2ce9` · llama.cpp default (ub 512) | Archived greedy |
| 1 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | 3224.5 | 2018.0 | 98.3 | 80.1 | kaiLLM Default (prefill eager, decode graphs, eager head) · `6758422b` · llama.cpp default (ub 512) | Archived greedy |
| 1 | plain (earlier kaiLLM · Humming unrecorded) | code | $\color{red}{1035.5}$ | 1996.6 | $\color{red}{61.9}$ | 81.3 | Unrecorded | Archived greedy |
| 1 | plain (earlier kaiLLM · deterministic Humming · [Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | $\color{red}{1066.8}$ | 2018.0 | $\color{red}{46.2}$ | 80.1 | kaiLLM Full graphs · llama.cpp default (ub 512) | Archived greedy |
| 1 | plain (earlier kaiLLM · deterministic Humming) | code | $\color{red}{1066.8}$ | 2042.5 | $\color{red}{46.2}$ | 80.4 | kaiLLM Full graphs · llama.cpp Forced graphs | Archived greedy |
| 1 | dspark5 (Inkling PR) | prose | — | 1959.6 | — | 129.5† | Unrecorded | Archived greedy |
| 1 | dspark5 (Inkling PR) | prose | — | 1980.0 | — | 129.3† | Forced graphs | Archived greedy |
| 1 | dspark5 ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | — | 1973.0 | — | 128.0† | llama.cpp default (ub 512) | Archived greedy |
| 1 | dspark5 (Inkling PR) | code | — | 1947.3 | — | 154.1† | Unrecorded | Archived greedy |
| 1 | dspark5 (Inkling PR) | code | — | 1965.6 | — | 151.6† | Forced graphs | Archived greedy |
| 1 | dspark5 ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | — | 1957.8 | — | 151.4† | llama.cpp default (ub 512) | Archived greedy |
| 2 | plain (Inkling PR) | prose | — | 1982.5 | — | 123.9 | Forced graphs | Archived greedy |
| 2 | plain (Inkling PR) | prose | $\color{red}{1093.1}$ | 1985.8 | $\color{red}{101.5}$ | 123.2 | kaiLLM Decode-only capture · llama.cpp Unrecorded | Archived greedy |
| 2 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | 3259.0 | 1986.9 | 145.6 | 123.0 | kaiLLM Default (prefill eager, decode graphs + eager, eager head) · `609c1611` · llama.cpp default (ub 512) | Archived greedy |
| 2 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | $\color{red}{1093.1}$ | 1986.9 | $\color{red}{101.5}$ | 123.0 | kaiLLM Decode-only capture · llama.cpp default (ub 512) | Archived greedy |
| 2 | plain (Inkling PR) | code | — | 1960.2 | — | 123.2 | Forced graphs | Archived greedy |
| 2 | plain (Inkling PR) | code | $\color{red}{1087.5}$ | 1964.2 | $\color{red}{101.2}$ | 122.9 | kaiLLM Decode-only capture · llama.cpp Unrecorded | Archived greedy |
| 2 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | 3244.0 | 1961.2 | 144.7 | 122.2 | kaiLLM Default (prefill eager, decode graphs + eager, eager head) · `609c1611` · llama.cpp default (ub 512) | Archived greedy |
| 2 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | $\color{red}{1087.5}$ | 1961.2 | $\color{red}{101.2}$ | 122.2 | kaiLLM Decode-only capture · llama.cpp default (ub 512) | Archived greedy |
| 2 | dspark5 (Inkling PR) | prose | — | 1911.2 | — | 140.7† | Unrecorded | Archived greedy |
| 2 | dspark5 ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | — | 1917.9 | — | 140.1† | llama.cpp default (ub 512) | Archived greedy |
| 2 | dspark5 (Inkling PR) | code | — | 1912.4 | — | 187.0† | Unrecorded | Archived greedy |
| 2 | dspark5 ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | — | 1888.2 | — | 189.8† | llama.cpp default (ub 512) | Archived greedy |
| 4 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | $\color{red}{1053.8}$ | 1793.3 | $\color{red}{127.0}$ | 157.8 | kaiLLM Default (prefill eager, decode graphs + eager, eager head) · `6758422b` · llama.cpp default (ub 256) | Archived greedy |
| 4 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | 3261.3 | 1793.3 | 211.7 | 157.8 | kaiLLM Default (prefill eager, decode graphs + eager, eager head) · `724c29d5` · llama.cpp default (ub 256) | Archived greedy |
| 4 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | $\color{red}{1051.7}$ | 1770.4 | $\color{red}{124.6}$ | 156.2 | kaiLLM Default (prefill eager, decode graphs + eager, eager head) · `6758422b` · llama.cpp default (ub 256) | Archived greedy |
| 4 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | 3240.1 | 1770.4 | 210.5 | 156.2 | kaiLLM Default (prefill eager, decode graphs + eager, eager head) · `724c29d5` · llama.cpp default (ub 256) | Archived greedy |

At batch 4, kaiLLM `724c29d5` reproduced the full prose output of `6758422b` for 1 of 4 requests (Archived greedy). At batch 4, kaiLLM `724c29d5` reproduced the full code output of `6758422b` for 2 of 4 requests (Archived greedy). With matching request order, a fresh greedy control agreed on all 16,384 token decisions and top-eight scores.

<details>
<summary><strong>DSpark decode comparisons</strong></summary>

| Engine | Batch | Domain | Baseline | Mode | Baseline decode | Measured decode | Change | Sampling |
|---|---:|---|---|---|---:|---:|---:|---|
| llama.cpp | 1 | code | plain | dspark5 | 81.3 | 154.1 | +89.7% | Archived greedy |
| llama.cpp | 1 | code | plain | dspark5 | 80.4 | 151.6 | +88.4% | Archived greedy |
| llama.cpp | 1 | prose | plain | dspark5 | 81.2 | 129.5 | +59.5% | Archived greedy |
| llama.cpp | 1 | prose | plain | dspark5 | 80.5 | 129.3 | +60.6% | Archived greedy |
| llama.cpp | 2 | code | plain | dspark5 | 123.2 | 189.8 | +54.0% | Archived greedy |
| llama.cpp | 2 | prose | plain | dspark5 | 123.9 | 140.7 | +13.5% | Archived greedy |

</details>

#### [kaitchup/Inkling-2-Mini-chimachima-kaiPTQ](https://huggingface.co/kaitchup/Inkling-2-Mini-chimachima-kaiPTQ/tree/1b27e927c15a6cde7f717926a25b2ce831d08fc1) (Q2_K_XS)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain | prose | 3084.5 | Unsupported GGUF | 66.5 | Unsupported GGUF | Default (prefill eager, decode graphs, eager head) · `800ab62d` | Archived greedy |
| 1 | plain | code | 3067.2 | Unsupported GGUF | 66.5 | Unsupported GGUF | Default (prefill eager, decode graphs, eager head) · `800ab62d` | Archived greedy |
| 1 | dspark5 (deterministic kaiLLM 62cb750) | prose | 2200.1 | Unsupported GGUF | 29.9 | Unsupported GGUF | Full graphs · 512-token prefill chunks | Model-recommended |
| 1 | dspark5 (deterministic kaiLLM 62cb750) | code | 2169.7 | Unsupported GGUF | 78.7 | Unsupported GGUF | Full graphs · 512-token prefill chunks | Model-recommended |
| 2 | plain | prose | 3089.9 | Unsupported GGUF | 112.4 | Unsupported GGUF | Default (prefill eager, decode graphs + eager, eager head) · `800ab62d` | Archived greedy |
| 2 | plain | code | 3103.5 | Unsupported GGUF | 112.7 | Unsupported GGUF | Default (prefill eager, decode graphs + eager, eager head) · `800ab62d` | Archived greedy |
| 4 | plain | prose | 2891.2 | Unsupported GGUF | 177.8 | Unsupported GGUF | Default (prefill eager, decode graphs + eager, eager head) · `800ab62d` | Archived greedy |
| 4 | plain | prose | 2899.5 | Unsupported GGUF | 176.8 | Unsupported GGUF | Default (prefill eager, decode graphs + eager, eager head) · `800ab62d` | Model-recommended |
| 4 | plain | code | 2910.9 | Unsupported GGUF | 177.6 | Unsupported GGUF | Default (prefill eager, decode graphs + eager, eager head) · `800ab62d` | Archived greedy |
| 4 | plain | code | 2916.5 | Unsupported GGUF | 177.4 | Unsupported GGUF | Default (prefill eager, decode graphs + eager, eager head) · `800ab62d` | Model-recommended |

<details>
<summary><strong>Batch-one performance versus closest-size GGUF</strong></summary>

| Mode | Domain | Closest-size llama.cpp checkpoint | Prefill vs reference | Decode vs reference | Sampling |
|---|---|---|---:|---:|---|
| dspark5 | code | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_M) | 113.5% | 127.7% | Model-recommended |
| dspark5 | prose | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_M) | 114.6% | 73.4% | Model-recommended |
| plain | code | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q2_K_XS) | 153.8% | 79.1% | Archived greedy |
| plain | prose | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q2_K_XS) | 153.3% | 79.1% | Archived greedy |

</details>

#### [kaitchup/Inkling-2-Mini-chimachima-kaiPTQ](https://huggingface.co/kaitchup/Inkling-2-Mini-chimachima-kaiPTQ/tree/1b27e927c15a6cde7f717926a25b2ce831d08fc1) (Q3_K_S)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain (earlier kaiLLM · deterministic Humming) | prose | 4034.4 | Unsupported GGUF | 70.5 | Unsupported GGUF | Full graphs | Archived greedy |
| 1 | plain (earlier kaiLLM · deterministic Humming) | code | 4046.8 | Unsupported GGUF | 70.6 | Unsupported GGUF | Full graphs | Archived greedy |
| 1 | plain (earlier kaiLLM · deterministic Humming) | code | 4040.7 | Unsupported GGUF | 70.6 | Unsupported GGUF | Full graphs | Model-recommended |
| 1 | dspark3 | prose | 2756.6 | Unsupported GGUF | 33.4 | Unsupported GGUF | Full graphs · 512-token prefill chunks | Model-recommended |
| 1 | dspark3 | code | 2719.5 | Unsupported GGUF | 86.1 | Unsupported GGUF | Full graphs · 512-token prefill chunks | Model-recommended |
| 1 | dspark5 (deterministic kaiLLM e4b7dab) | prose | 2769.0 | Unsupported GGUF | 40.3 | Unsupported GGUF | Full graphs · 512-token prefill chunks | Model-recommended |
| 1 | dspark5 (deterministic kaiLLM e4b7dab) | code | 2733.9 | Unsupported GGUF | 82.3 | Unsupported GGUF | Full graphs · 512-token prefill chunks | Model-recommended |
| 2 | plain | code | 4070.3 | Unsupported GGUF | 74.5 | Unsupported GGUF | Full graphs | Model-recommended |
| 4 | plain | code | 4078.5 | Unsupported GGUF | 76.4 | Unsupported GGUF | Full graphs | Model-recommended |

<details>
<summary><strong>Batch-one performance versus closest-size GGUF</strong></summary>

| Mode | Domain | Closest-size llama.cpp checkpoint | Prefill vs reference | Decode vs reference | Sampling |
|---|---|---|---:|---:|---|
| dspark3 | code | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_S) | 143.1% | 84.9% | Model-recommended |
| dspark3 | prose | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_S) | 145.3% | 55.2% | Model-recommended |
| dspark5 | code | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_M) | 143.0% | 133.6% | Model-recommended |
| dspark5 | prose | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_M) | 144.2% | 98.9% | Model-recommended |
| plain | code | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_S) | 206.8% | 83.1% | Archived greedy |
| plain | code | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_S) | 206.6% | 88.2% | Model-recommended |
| plain | prose | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_S) | 205.0% | 83.3% | Archived greedy |

</details>

<details>
<summary><strong>DSpark decode comparisons</strong></summary>

| Engine | Batch | Domain | Baseline | Mode | Baseline decode | Measured decode | Change | Sampling |
|---|---:|---|---|---|---:|---:|---:|---|
| kaiLLM | 1 | code | plain | dspark3 | 70.6 | 86.1 | +22.0% | Model-recommended |
| kaiLLM | 1 | code | plain | dspark5 | 70.6 | 82.3 | +16.6% | Model-recommended |
| kaiLLM | 1 | code | dspark5 | dspark3 | 82.3 | 86.1 | +4.7% | Model-recommended |
| kaiLLM | 1 | prose | dspark5 | dspark3 | 40.3 | 33.4 | -17.1% | Model-recommended |

</details>

#### [kaitchup/Inkling-2-Mini-chimachima-kaiPTQ](https://huggingface.co/kaitchup/Inkling-2-Mini-chimachima-kaiPTQ/tree/1b27e927c15a6cde7f717926a25b2ce831d08fc1) (Q3_K_M)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain (earlier kaiLLM · deterministic Humming) | prose | 3671.8 | Unsupported GGUF | 69.1 | Unsupported GGUF | Full graphs | Archived greedy |
| 1 | plain (earlier kaiLLM · deterministic Humming) | code | 3661.9 | Unsupported GGUF | 69.2 | Unsupported GGUF | Full graphs | Archived greedy |
| 1 | plain (earlier kaiLLM · deterministic Humming) | code | 3668.4 | Unsupported GGUF | 69.2 | Unsupported GGUF | Full graphs | Model-recommended |
| 1 | dspark3 | prose | 2583.7 | Unsupported GGUF | 59.1 | Unsupported GGUF | Full graphs · 512-token prefill chunks | Model-recommended |
| 1 | dspark3 | code | 2554.4 | Unsupported GGUF | 77.7 | Unsupported GGUF | Full graphs · 512-token prefill chunks | Model-recommended |
| 1 | dspark5 (deterministic kaiLLM 0c59ea4) | prose | 2600.1 | Unsupported GGUF | 30.4 | Unsupported GGUF | Full graphs · `0c59ea4e` · 512-token prefill chunks | Model-recommended |
| 1 | dspark5 (deterministic kaiLLM ea990bc) | prose | 2592.0 | Unsupported GGUF | 31.1 | Unsupported GGUF | Full graphs · `ea990bcb` · 512-token prefill chunks | Model-recommended |
| 1 | dspark5 (deterministic kaiLLM 0c59ea4) | code | 2564.0 | Unsupported GGUF | 103.6 | Unsupported GGUF | Full graphs · `0c59ea4e` · 512-token prefill chunks | Model-recommended |
| 1 | dspark5 (deterministic kaiLLM ea990bc) | code | 2565.6 | Unsupported GGUF | 105.8 | Unsupported GGUF | Full graphs · `ea990bcb` · 512-token prefill chunks | Model-recommended |
| 2 | plain | code | 3680.1 | Unsupported GGUF | 69.6 | Unsupported GGUF | Full graphs | Model-recommended |
| 4 | plain | code | 3701.0 | Unsupported GGUF | 76.2 | Unsupported GGUF | Full graphs · `c6f22a6f` | Model-recommended |
| 4 | plain | code | 3695.8 | Unsupported GGUF | 125.3 | Unsupported GGUF | Full graphs · `c87ed8e0` | Model-recommended |

<details>
<summary><strong>Batch-one performance versus closest-size GGUF</strong></summary>

| Mode | Domain | Closest-size llama.cpp checkpoint | Prefill vs reference | Decode vs reference | Sampling |
|---|---|---|---:|---:|---|
| dspark3 | code | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_S) | 134.4% | 76.5% | Model-recommended |
| dspark3 | prose | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_S) | 136.2% | 97.7% | Model-recommended |
| dspark5 | code | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_M) | 134.2% | 171.6% | Model-recommended |
| dspark5 | prose | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_M) | 135.0% | 76.2% | Model-recommended |
| plain | code | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_M) | 185.7% | 83.6% | Archived greedy |
| plain | code | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_M) | 186.3% | 88.9% | Model-recommended |
| plain | prose | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_M) | 186.0% | 83.6% | Archived greedy |

</details>

<details>
<summary><strong>DSpark decode comparisons</strong></summary>

| Engine | Batch | Domain | Baseline | Mode | Baseline decode | Measured decode | Change | Sampling |
|---|---:|---|---|---|---:|---:|---:|---|
| kaiLLM | 1 | code | plain | dspark3 | 69.2 | 77.7 | +12.3% | Model-recommended |
| kaiLLM | 1 | code | plain | dspark5 | 69.2 | 105.8 | +52.9% | Model-recommended |
| kaiLLM | 1 | code | dspark5 | dspark3 | 105.8 | 77.7 | -26.6% | Model-recommended |
| kaiLLM | 1 | prose | dspark5 | dspark3 | 31.1 | 59.1 | +90.3% | Model-recommended |

</details>

#### [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q2_K_XS)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | 3173.2 | 2011.6 | 89.0 | 84.1 | kaiLLM Default (prefill eager, decode graphs, eager head) · `41776a15` · llama.cpp default (ub 512) | Archived greedy |
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | 3174.3 | 2011.6 | $\color{red}{75.8}$ | 84.1 | kaiLLM Default (prefill eager, decode graphs, eager head) · `426c7f23` · llama.cpp default (ub 512) | Archived greedy |
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | 3145.9 | 1994.8 | 89.0 | 84.1 | kaiLLM Default (prefill eager, decode graphs, eager head) · `41776a15` · llama.cpp default (ub 512) | Archived greedy |
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | 3146.4 | 1994.8 | $\color{red}{75.9}$ | 84.1 | kaiLLM Default (prefill eager, decode graphs, eager head) · `426c7f23` · llama.cpp default (ub 512) | Archived greedy |
| 1 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | — | 1919.7 | — | 115.1† | llama.cpp default (ub 512) | Archived greedy |
| 1 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | — | 1893.5 | — | 130.1† | llama.cpp default (ub 512) | Archived greedy |
| 2 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | 3183.5 | 1954.9 | 121.4 | 127.9 | kaiLLM Default (prefill eager, decode graphs + eager, eager head) · `e66200d7` · llama.cpp default (ub 512) | Archived greedy |
| 2 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | 3153.2 | 1944.4 | 121.3 | 127.7 | kaiLLM Default (prefill eager, decode graphs + eager, eager head) · `e66200d7` · llama.cpp default (ub 512) | Archived greedy |
| 2 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | — | 1815.8 | — | 149.1 | llama.cpp default (ub 512) | Archived greedy |
| 2 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | — | 1806.0 | — | 194.5 | llama.cpp default (ub 512) | Archived greedy |
| 4 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | $\color{red}{595.7}$ | 1868.8 | $\color{red}{86.2}$ | 170.0 | kaiLLM Default (prefill eager, decode graphs + eager, eager head) · `800ab62d` · llama.cpp default (ub 512) | Archived greedy |
| 4 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | 2975.0 | 1869.8 | 183.5 | 139.6 | kaiLLM Default (prefill eager, decode graphs + eager, eager head) · `1c797924` · llama.cpp default (ub 512) | Model-recommended |
| 4 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | $\color{red}{599.1}$ | 1868.3 | $\color{red}{85.7}$ | 169.2 | kaiLLM Default (prefill eager, decode graphs + eager, eager head) · `800ab62d` · llama.cpp default (ub 512) | Archived greedy |
| 4 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | 2945.8 | 1859.2 | 183.1 | 139.3 | kaiLLM Default (prefill eager, decode graphs + eager, eager head) · `1c797924` · llama.cpp default (ub 512) | Model-recommended |
| 4 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | — | 1803.4 | — | 53.3† | llama.cpp default (ub 512) | Model-recommended |
| 4 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | — | 1793.1 | — | 121.0† | llama.cpp default (ub 512) | Model-recommended |

<details>
<summary><strong>DSpark decode comparisons</strong></summary>

| Engine | Batch | Domain | Baseline | Mode | Baseline decode | Measured decode | Change | Sampling |
|---|---:|---|---|---|---:|---:|---:|---|
| llama.cpp | 1 | code | plain | dspark5 | 84.1 | 130.1 | +54.7% | Archived greedy |
| llama.cpp | 1 | prose | plain | dspark5 | 84.1 | 115.1 | +36.7% | Archived greedy |
| llama.cpp | 4 | code | plain | dspark5 | 139.3 | 121.0 | -13.1% | Model-recommended |
| llama.cpp | 4 | prose | plain | dspark5 | 139.6 | 53.3 | -61.8% | Model-recommended |

</details>

#### [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_S)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain (earlier kaiLLM · deterministic Humming · [llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | $\color{red}{528.2}$ | 1968.2 | $\color{red}{30.8}$ | 84.7 | kaiLLM Full graphs · llama.cpp default (ub 512) | Archived greedy |
| 1 | plain (earlier kaiLLM · deterministic Humming · [llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | 2245.1 | 1966.1 | 85.0 | 79.9 | kaiLLM Full graphs · llama.cpp default (ub 512) | Model-recommended |
| 1 | plain (earlier kaiLLM · deterministic Humming · [llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | $\color{red}{528.0}$ | 1956.8 | $\color{red}{30.8}$ | 85.0 | kaiLLM Full graphs · llama.cpp default (ub 512) | Archived greedy |
| 1 | plain (earlier kaiLLM · deterministic Humming · [llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | 2231.4 | 1955.4 | 85.0 | 80.0 | kaiLLM Full graphs · llama.cpp default (ub 512) | Model-recommended |
| 1 | dspark3 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | $\color{red}{502.1}$ | 1896.6 | $\color{red}{22.6}$ | 60.5† | kaiLLM Full graphs · 512-token prefill chunks · llama.cpp default (ub 512) | Model-recommended |
| 1 | dspark3 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | $\color{red}{497.0}$ | 1899.9 | $\color{red}{51.7}$ | 101.5† | kaiLLM Full graphs · 512-token prefill chunks · llama.cpp default (ub 512) | Model-recommended |
| 1 | dspark5 (deterministic kaiLLM 3f45daf · Inkling PR · [llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | $\color{red}{500.4}$ | 1900.4 | $\color{red}{58.7}$ | 131.2† | kaiLLM Full graphs · 512-token prefill chunks · llama.cpp default (ub 512) | Archived greedy |
| 1 | dspark5 (deterministic kaiLLM 3f45daf · Inkling PR · [llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | $\color{red}{495.6}$ | 1892.1 | $\color{red}{85.4}$ | 167.2† | kaiLLM Full graphs · 512-token prefill chunks · llama.cpp default (ub 512) | Archived greedy |
| 2 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | — | 1920.1 | — | 129.1 | llama.cpp default (ub 512) | Archived greedy |
| 2 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | 1796.2 | 1919.1 | $\color{red}{68.6}$ | 128.7 | kaiLLM Full graphs · llama.cpp default (ub 512) | Archived greedy |
| 2 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | 2157.5 | 1914.6 | $\color{red}{78.2}$ | 117.5 | kaiLLM Full graphs · `4dc683e8` · llama.cpp default (ub 512) | Model-recommended |
| 2 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | 2159.1 | 1914.6 | $\color{red}{72.6}$ | 117.5 | kaiLLM Full graphs · `5c55c760` · llama.cpp default (ub 512) | Model-recommended |
| 2 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | — | 1836.3 | — | 148.0† | llama.cpp default (ub 512) | Archived greedy |
| 2 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | — | 1832.2 | — | 185.7† | llama.cpp default (ub 512) | Archived greedy |
| 4 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | $\color{red}{539.2}$ | 1895.3 | $\color{red}{73.2}$ | 148.2 | kaiLLM Full graphs · llama.cpp default (ub 512) | Model-recommended |

<details>
<summary><strong>DSpark decode comparisons</strong></summary>

| Engine | Batch | Domain | Baseline | Mode | Baseline decode | Measured decode | Change | Sampling |
|---|---:|---|---|---|---:|---:|---:|---|
| kaiLLM | 1 | code | plain | dspark3 | 85.0 | 51.7 | -39.2% | Model-recommended |
| kaiLLM | 1 | code | plain | dspark5 | 30.8 | 85.4 | +176.9% | Archived greedy |
| kaiLLM | 1 | prose | plain | dspark3 | 85.0 | 22.6 | -73.4% | Model-recommended |
| kaiLLM | 1 | prose | plain | dspark5 | 30.8 | 58.7 | +90.5% | Archived greedy |
| llama.cpp | 1 | code | plain | dspark3 | 80.0 | 101.5 | +26.8% | Model-recommended |
| llama.cpp | 1 | code | plain | dspark5 | 85.0 | 167.2 | +96.8% | Archived greedy |
| llama.cpp | 1 | prose | plain | dspark3 | 79.9 | 60.5 | -24.4% | Model-recommended |
| llama.cpp | 1 | prose | plain | dspark5 | 84.7 | 131.2 | +55.0% | Archived greedy |
| llama.cpp | 2 | code | plain | dspark5 | 128.7 | 185.7 | +44.3% | Archived greedy |
| llama.cpp | 2 | prose | plain | dspark5 | 129.1 | 148.0 | +14.7% | Archived greedy |

</details>

#### [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_M)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | 1885.2 | 1974.6 | 81.8 | 82.7 | kaiLLM Full graphs · `b9e3d1a0` · llama.cpp default (ub 512) | Archived greedy |
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | $\color{red}{578.3}$ | 1974.6 | $\color{red}{32.0}$ | 82.7 | kaiLLM Full graphs · `c6f22a6f` · llama.cpp default (ub 512) | Archived greedy |
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | 1880.0 | 1987.9 | 81.8 | 78.0 | kaiLLM Full graphs · `21259a7e` · llama.cpp default (ub 512) | Model-recommended |
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | 2003.7 | 1987.9 | 98.1 | 78.0 | kaiLLM Full graphs · `7417185d` · llama.cpp default (ub 512) | Model-recommended |
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | 1872.1 | 1971.7 | 81.8 | 82.8 | kaiLLM Full graphs · `b9e3d1a0` · llama.cpp default (ub 512) | Archived greedy |
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | $\color{red}{572.7}$ | 1971.7 | $\color{red}{32.0}$ | 82.8 | kaiLLM Full graphs · `c6f22a6f` · llama.cpp default (ub 512) | Archived greedy |
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | 1871.4 | 1968.7 | 81.7 | 77.9 | kaiLLM Full graphs · `21259a7e` · llama.cpp default (ub 512) | Model-recommended |
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | 1993.6 | 1968.7 | 98.0 | 77.9 | kaiLLM Full graphs · `7417185d` · llama.cpp default (ub 512) | Model-recommended |
| 1 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | — | 1901.9 | — | 120.6† | llama.cpp default (ub 512) | Archived greedy |
| 1 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | — | 1920.0 | — | 40.7† | llama.cpp default (ub 512) | Model-recommended |
| 1 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | — | 1903.3 | — | 134.3† | llama.cpp default (ub 512) | Archived greedy |
| 1 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | — | 1911.9 | — | 61.6† | llama.cpp default (ub 512) | Model-recommended |
| 2 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | — | 1928.8 | — | 127.0 | llama.cpp default (ub 512) | Archived greedy |
| 2 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | $\color{red}{592.5}$ | 1928.7 | $\color{red}{66.3}$ | 116.3 | kaiLLM Full graphs · llama.cpp default (ub 512) | Model-recommended |
| 2 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | — | 1922.1 | — | 126.8 | llama.cpp default (ub 512) | Archived greedy |
| 2 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | $\color{red}{593.3}$ | 1919.7 | $\color{red}{66.3}$ | 115.7 | kaiLLM Full graphs · `0fad3c94` · llama.cpp default (ub 512) | Model-recommended |
| 2 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | 1778.8 | 1919.7 | $\color{red}{67.0}$ | 115.7 | kaiLLM Full graphs · `1d2d0305` · llama.cpp default (ub 512) | Model-recommended |
| 2 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | — | 1845.0 | — | 40.4† | llama.cpp default (ub 512) | Model-recommended |
| 2 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | — | 1844.0 | — | 98.1† | llama.cpp default (ub 512) | Model-recommended |
| 4 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | — | 1900.9 | — | 137.3 | llama.cpp default (ub 512) | Model-recommended |

<details>
<summary><strong>DSpark decode comparisons</strong></summary>

| Engine | Batch | Domain | Baseline | Mode | Baseline decode | Measured decode | Change | Sampling |
|---|---:|---|---|---|---:|---:|---:|---|
| llama.cpp | 1 | code | plain | dspark5 | 82.8 | 134.3 | +62.3% | Archived greedy |
| llama.cpp | 1 | code | plain | dspark5 | 77.9 | 61.6 | -20.9% | Model-recommended |
| llama.cpp | 1 | prose | plain | dspark5 | 82.7 | 120.6 | +45.8% | Archived greedy |
| llama.cpp | 1 | prose | plain | dspark5 | 78.0 | 40.7 | -47.8% | Model-recommended |
| llama.cpp | 2 | code | plain | dspark5 | 115.7 | 98.1 | -15.2% | Model-recommended |
| llama.cpp | 2 | prose | plain | dspark5 | 116.3 | 40.4 | -65.2% | Model-recommended |

</details>

<a id="gpu-5090-2"></a>

## 2 × RTX 5090

<a id="gpu-5090-2-model-hy4"></a>

### Hy4 Flash

#### [kaitchup/Hy4-flash-dondon-GGUF](https://huggingface.co/kaitchup/Hy4-flash-dondon-GGUF/tree/ee58ad539c09c0b2b290a06f17c99e638a0db139) (Q2_K_M)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain ([Hy4 PR 0a054d0](https://github.com/ggml-org/llama.cpp/commit/0a054d0b02454c998e068381d15f3f6615bdd987)) | prose | — | 1443.0 | — | 20.3 | llama.cpp default (ub 512) | Archived greedy |
| 1 | plain ([Hy4 PR 0a054d0](https://github.com/ggml-org/llama.cpp/commit/0a054d0b02454c998e068381d15f3f6615bdd987)) | code | — | 1414.1 | — | 20.0 | llama.cpp default (ub 512) | Archived greedy |

<a id="gpu-pro-6000-1"></a>

## 1 × RTX PRO 6000

<a id="gpu-pro-6000-1-model-lfm"></a>

### LFM3 1.7B

#### [LiquidAI/LFM3-1.7B-GGUF](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/tree/cd0c74c575b5c416f3011b5b4c298077f26ca8c2) (QAD-Q4_0)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 32 | plain | prose | 65970.5 | 34098.0 | 2435.6 | 483.3 | Unrecorded | Archived greedy |
| 32 | plain | code | 62457.6 | 34256.5 | 2441.6 | 480.8 | Unrecorded | Archived greedy |

<a id="gpu-pro-6000-1-model-daisy"></a>

### DaisyRose 27B

#### [bartowski/RadixArk_DaisyRose-27B-GGUF](https://huggingface.co/bartowski/RadixArk_DaisyRose-27B-GGUF/tree/4514fec3c6573951bce8acc7f850d9eb496301c7) (IQ2_M)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain | prose | — | 3539.9 | — | 74.3 | Unrecorded | Archived greedy |
| 1 | plain | code | — | 3435.6 | — | 74.3 | Unrecorded | Archived greedy |

#### [kaitchup/DaisyRose-27B-chimachima-kaiPTQ](https://huggingface.co/kaitchup/DaisyRose-27B-chimachima-kaiPTQ/tree/6c1a63358c57f5baf2730574ba418195eb7edfa8) (Q2_K_XS)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain | prose | 4285.9 | Unsupported GGUF | 96.4 | Unsupported GGUF | Unrecorded | Archived greedy |
| 1 | plain | code | 4081.2 | Unsupported GGUF | 96.3 | Unsupported GGUF | Unrecorded | Archived greedy |
| 1 | dspine7 | prose | 4314.2 | Unsupported GGUF | 122.0† | Unsupported GGUF | Unrecorded | Archived greedy |
| 1 | dspine7 | code | 4038.1 | Unsupported GGUF | 416.6† | Unsupported GGUF | Unrecorded | Archived greedy |
| 1 | dspine15 | prose | 4224.5 | Unsupported GGUF | 261.9† | Unsupported GGUF | Unrecorded | Archived greedy |
| 1 | dspine15 | code | 4217.6 | Unsupported GGUF | 245.2† | Unsupported GGUF | Unrecorded | Archived greedy |

#### [kaitchup/DaisyRose-27B-dondon-GGUF](https://huggingface.co/kaitchup/DaisyRose-27B-dondon-GGUF/tree/3592e073925dbdbc65a340fca851fc01b8d2c409) (IQ2_M)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain | prose | 3937.0 | 3389.1 | 94.0 | 72.9 | Unrecorded | Archived greedy |
| 1 | plain | code | 3730.1 | 3388.0 | 93.8 | 72.8 | Unrecorded | Archived greedy |
| 1 | dspine7 | prose | 3894.4 | — | 286.5† | — | Unrecorded | Archived greedy |
| 1 | dspine7 | code | 3724.1 | — | 284.8† | — | Unrecorded | Archived greedy |
| 1 | dspine15 | prose | 3891.1 | — | 230.0 | — | Unrecorded | Archived greedy |
| 1 | dspine15 | code | 3837.9 | — | 219.3† | — | Unrecorded | Archived greedy |

<a id="gpu-pro-6000-1-model-qwen"></a>

### Qwen3.8 27B

#### [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/4af51b3542f775b0a8e4b310bfb3f17e01c306c9) (UD-Q2_K_XL)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | mtp5 | prose | 4315.7 | — | 269.8 | — | Unrecorded | Archived greedy |
| 1 | mtp5 | code | 4089.9 | — | 291.3 | — | Unrecorded | Archived greedy |

<a id="gpu-pro-6000-1-model-inkling"></a>

### Inkling 2 Mini

#### [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/8078d149286eec7193795413b9e0643da6d11fa1) (UD-Q2_K_XL)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | 3280.5 | 2246.6 | 109.6 | 85.1 | kaiLLM Default (prefill eager, decode graphs, eager head) · `ed9235e9` · llama.cpp default (ub 512) | Archived greedy |
| 1 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | 3210.8 | 2242.7 | 109.6 | 85.1 | kaiLLM Default (prefill eager, decode graphs, eager head) · `ed9235e9` · llama.cpp default (ub 512) | Archived greedy |
| 1 | dspark5 ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | — | 2211.9 | — | 124.8† | llama.cpp default (ub 512) | Archived greedy |
| 1 | dspark5 ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | — | 2203.5 | — | 184.4† | llama.cpp default (ub 512) | Archived greedy |
| 2 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | — | 2198.6 | — | 133.7 | llama.cpp default (ub 512) | Archived greedy |
| 2 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | — | 2176.6 | — | 133.6 | llama.cpp default (ub 512) | Archived greedy |
| 4 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | — | 2171.3 | — | 172.0 | llama.cpp default (ub 512) | Archived greedy |
| 4 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | — | 2162.9 | — | 171.7 | llama.cpp default (ub 512) | Archived greedy |

<details>
<summary><strong>DSpark decode comparisons</strong></summary>

| Engine | Batch | Domain | Baseline | Mode | Baseline decode | Measured decode | Change | Sampling |
|---|---:|---|---|---|---:|---:|---:|---|
| llama.cpp | 1 | code | plain | dspark5 | 85.1 | 184.4 | +116.7% | Archived greedy |
| llama.cpp | 1 | prose | plain | dspark5 | 85.1 | 124.8 | +46.6% | Archived greedy |

</details>

#### [ggml-org/Inkling-2-Mini-GGUF](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/tree/0bdc580634ceac4c4d921f452777bf44228de9af) (IQ4_NL)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | 4807.2 | 2279.0 | 84.4 | 81.5 | kaiLLM Default (prefill eager, decode graphs, eager head) · `ffe8fc5e` · llama.cpp default (ub 512) | Archived greedy |
| 1 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | 4743.7 | 2267.6 | 84.4 | 81.5 | kaiLLM Default (prefill eager, decode graphs, eager head) · `ffe8fc5e` · llama.cpp default (ub 512) | Archived greedy |
| 1 | dspark5 ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | — | 2247.5 | — | 128.8† | llama.cpp default (ub 512) | Archived greedy |
| 1 | dspark5 ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | — | 2231.0 | — | 177.0† | llama.cpp default (ub 512) | Archived greedy |
| 2 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | 4875.1 | 2228.0 | 149.2 | 127.5 | kaiLLM Default (prefill eager, decode graphs + eager, eager head) · `bd300e88` · llama.cpp default (ub 512) | Archived greedy |
| 2 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | 4785.3 | 2209.5 | 148.2 | 127.0 | kaiLLM Default (prefill eager, decode graphs + eager, eager head) · `bd300e88` · llama.cpp default (ub 512) | Archived greedy |
| 4 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | prose | 4864.9 | 2197.2 | 212.4 | 165.7 | kaiLLM Default (prefill eager, decode graphs + eager, eager head) · `5ee051c5` · llama.cpp default (ub 512) | Archived greedy |
| 4 | plain ([Inkling PR e4b1cb5](https://github.com/ggml-org/llama.cpp/commit/e4b1cb5318f1ac55d148be228c6347c6a94cc4c6)) | code | 4767.0 | 2178.7 | 211.1 | 164.0 | kaiLLM Default (prefill eager, decode graphs + eager, eager head) · `5ee051c5` · llama.cpp default (ub 512) | Archived greedy |

<details>
<summary><strong>DSpark decode comparisons</strong></summary>

| Engine | Batch | Domain | Baseline | Mode | Baseline decode | Measured decode | Change | Sampling |
|---|---:|---|---|---|---:|---:|---:|---|
| llama.cpp | 1 | code | plain | dspark5 | 81.5 | 177.0 | +117.2% | Archived greedy |
| llama.cpp | 1 | prose | plain | dspark5 | 81.5 | 128.8 | +57.9% | Archived greedy |

</details>

#### [kaitchup/Inkling-2-Mini-chimachima-kaiPTQ](https://huggingface.co/kaitchup/Inkling-2-Mini-chimachima-kaiPTQ/tree/1b27e927c15a6cde7f717926a25b2ce831d08fc1) (Q3_K_M)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain (earlier kaiLLM · deterministic Humming) | code | 4150.0 | Unsupported GGUF | 69.3 | Unsupported GGUF | Full graphs | Model-recommended |
| 1 | dspark5 (deterministic kaiLLM 5e81374) | prose | 2794.8 | Unsupported GGUF | 52.3 | Unsupported GGUF | Full graphs · 512-token prefill chunks | Model-recommended |
| 1 | dspark5 (deterministic kaiLLM 5e81374) | code | 2794.1 | Unsupported GGUF | 93.4 | Unsupported GGUF | Full graphs · 512-token prefill chunks | Model-recommended |
| 8 | plain | code | 4130.3 | Unsupported GGUF | 209.4 | Unsupported GGUF | Full graphs · `2028e6a3` | Model-recommended |
| 8 | plain | code | 4128.0 | Unsupported GGUF | 148.9 | Unsupported GGUF | Full graphs · `c6f22a6f` | Model-recommended |

<details>
<summary><strong>Batch-one performance versus closest-size GGUF</strong></summary>

| Mode | Domain | Closest-size llama.cpp checkpoint | Prefill vs reference | Decode vs reference | Sampling |
|---|---|---|---:|---:|---|
| dspark5 | code | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_M) | 127.7% | 98.0% | Model-recommended |
| dspark5 | prose | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_M) | 127.3% | 138.6% | Model-recommended |
| plain | code | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_M) | 183.7% | 87.2% | Model-recommended |

</details>

<details>
<summary><strong>DSpark decode comparisons</strong></summary>

| Engine | Batch | Domain | Baseline | Mode | Baseline decode | Measured decode | Change | Sampling |
|---|---:|---|---|---|---:|---:|---:|---|
| kaiLLM | 1 | code | plain | dspark5 | 69.3 | 93.4 | +34.8% | Model-recommended |

</details>

#### [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_M)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain (earlier kaiLLM · deterministic Humming · [llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | 2055.0 | 2258.6 | 82.1 | 79.5 | kaiLLM Full graphs · `784dde45` · llama.cpp default (ub 512) | Model-recommended |
| 1 | plain (earlier kaiLLM · deterministic Humming · [llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | $\color{red}{1997.8}$ | 2258.6 | 82.1 | 79.5 | kaiLLM Full graphs · `789a1e59` · llama.cpp default (ub 512) | Model-recommended |
| 1 | dspark5 (deterministic kaiLLM c18a38c · Inkling PR · [llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | $\color{red}{528.7}$ | 2196.0 | $\color{red}{16.7}$ | 37.7 | kaiLLM Full graphs · 512-token prefill chunks · llama.cpp default (ub 512) | Model-recommended |
| 1 | dspark5 (deterministic kaiLLM c18a38c · Inkling PR · [llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | $\color{red}{533.8}$ | 2187.4 | $\color{red}{47.3}$ | 95.3† | kaiLLM Full graphs · 512-token prefill chunks · llama.cpp default (ub 512) | Model-recommended |
| 2 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | $\color{red}{643.5}$ | 2217.2 | $\color{red}{30.1}$ | 120.9 | kaiLLM Full graphs · llama.cpp default (ub 512) | Model-recommended |
| 8 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | $\color{red}{994.3}$ | 2149.9 | $\color{red}{105.8}$ | 189.6 | kaiLLM Full graphs · `98a70af8` · llama.cpp default (ub 512) | Model-recommended |
| 8 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | $\color{red}{630.1}$ | 2149.9 | $\color{red}{88.3}$ | 189.6 | kaiLLM Full graphs · `c6f22a6f` · llama.cpp default (ub 512) | Model-recommended |
| 8 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | $\color{red}{630.2}$ | 2149.9 | $\color{red}{106.7}$ | 189.6 | kaiLLM Full graphs · `ccf04fd4` · llama.cpp default (ub 512) | Model-recommended |
| 8 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | — | 2066.2 | — | 64.0 | llama.cpp default (ub 512) | Model-recommended |
| 8 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | — | 2070.1 | — | 149.5† | llama.cpp default (ub 512) | Model-recommended |

<details>
<summary><strong>DSpark decode comparisons</strong></summary>

| Engine | Batch | Domain | Baseline | Mode | Baseline decode | Measured decode | Change | Sampling |
|---|---:|---|---|---|---:|---:|---:|---|
| kaiLLM | 1 | code | plain | dspark5 | 82.1 | 47.3 | -42.4% | Model-recommended |
| llama.cpp | 1 | code | plain | dspark5 | 79.5 | 95.3 | +19.9% | Model-recommended |
| llama.cpp | 8 | code | plain | dspark5 | 189.6 | 149.5 | -21.1% | Model-recommended |

</details>

<a id="gpu-pro-6000-1-model-hy4"></a>

### Hy4 Flash

#### [Xylolabs/Hy4-flash-GGUF](https://huggingface.co/Xylolabs/Hy4-flash-GGUF/tree/cc17e8a68a84db84a367f56fd42fbd40b54b28e2) (IQ2_M-E4)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain | prose | 194.7 | — | 5.3 | — | Unrecorded | Archived greedy |
| 1 | plain ([Hy4 PR 0a054d0](https://github.com/ggml-org/llama.cpp/commit/0a054d0b02454c998e068381d15f3f6615bdd987)) | prose | — | 1702.5 | — | 41.9 | llama.cpp default (ub 512) | Archived greedy |
| 1 | plain | code | 195.9 | — | 5.4 | — | Unrecorded | Archived greedy |
| 1 | plain ([Hy4 PR 0a054d0](https://github.com/ggml-org/llama.cpp/commit/0a054d0b02454c998e068381d15f3f6615bdd987)) | code | — | 1711.5 | — | 42.1 | llama.cpp default (ub 512) | Archived greedy |

#### [kaitchup/Hy4-flash-dondon-GGUF](https://huggingface.co/kaitchup/Hy4-flash-dondon-GGUF/tree/ee58ad539c09c0b2b290a06f17c99e638a0db139) (Q2_K_M)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain ([Hy4 PR 0a054d0](https://github.com/ggml-org/llama.cpp/commit/0a054d0b02454c998e068381d15f3f6615bdd987)) | code | — | 2071.3 | — | 49.7 | llama.cpp default (ub 512) | Model-recommended |

<a id="gpu-dgx-spark-1"></a>

## 1 × DGX Spark

<a id="gpu-dgx-spark-1-model-inkling"></a>

### Inkling 2 Mini

#### [kaitchup/Inkling-2-Mini-chimachima-kaiPTQ](https://huggingface.co/kaitchup/Inkling-2-Mini-chimachima-kaiPTQ/tree/1b27e927c15a6cde7f717926a25b2ce831d08fc1) (Q2_K_XS)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain (deterministic kaiLLM 7871156) | code | 563.1 | Unsupported GGUF | 32.9 | Unsupported GGUF | Full graphs | Model-recommended |
| 1 | dspark5 (deterministic kaiLLM 7871156) | prose | 549.2 | Unsupported GGUF | 30.0 | Unsupported GGUF | Full graphs | Model-recommended |
| 1 | dspark5 (deterministic kaiLLM 7871156) | code | 556.1 | Unsupported GGUF | 40.6† | Unsupported GGUF | Full graphs | Model-recommended |

<details>
<summary><strong>Batch-one performance versus closest-size GGUF</strong></summary>

| Mode | Domain | Closest-size llama.cpp checkpoint | Prefill vs reference | Decode vs reference | Sampling |
|---|---|---|---:|---:|---|
| dspark5 | code | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q2_K_XS) | 113.5% | 332.8% | Model-recommended |
| dspark5 | prose | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q2_K_XS) | 112.2% | 274.7% | Model-recommended |
| plain | code | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_S) | 115.0% | 102.3% | Model-recommended |

</details>

<details>
<summary><strong>DSpark decode comparisons</strong></summary>

| Engine | Batch | Domain | Baseline | Mode | Baseline decode | Measured decode | Change | Sampling |
|---|---:|---|---|---|---:|---:|---:|---|
| kaiLLM | 1 | code | plain | dspark5 | 32.9 | 40.6 | +23.3% | Model-recommended |

</details>

#### [kaitchup/Inkling-2-Mini-chimachima-kaiPTQ](https://huggingface.co/kaitchup/Inkling-2-Mini-chimachima-kaiPTQ/tree/1b27e927c15a6cde7f717926a25b2ce831d08fc1) (Q3_K_S)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain (deterministic kaiLLM f1e1514) | code | 715.7 | Unsupported GGUF | 33.2 | Unsupported GGUF | Full graphs | Model-recommended |
| 1 | dspark3 | prose | 698.6 | Unsupported GGUF | 45.1 | Unsupported GGUF | Full graphs | Model-recommended |
| 1 | dspark3 | code | 699.6 | Unsupported GGUF | 43.7 | Unsupported GGUF | Full graphs | Model-recommended |
| 1 | dspark5 (deterministic kaiLLM e1ef426) | prose | 696.5 | Unsupported GGUF | 20.3 | Unsupported GGUF | Full graphs · `e1ef426d` | Model-recommended |
| 1 | dspark5 (deterministic kaiLLM f1e1514) | prose | 698.2 | Unsupported GGUF | 20.3 | Unsupported GGUF | Full graphs · `f1e15143` | Model-recommended |
| 1 | dspark5 (deterministic kaiLLM e1ef426) | code | 697.6 | Unsupported GGUF | 49.4 | Unsupported GGUF | Full graphs · `e1ef426d` | Model-recommended |
| 1 | dspark5 (deterministic kaiLLM f1e1514) | code | 699.6 | Unsupported GGUF | 49.3† | Unsupported GGUF | Full graphs · `f1e15143` | Model-recommended |

<details>
<summary><strong>Batch-one performance versus closest-size GGUF</strong></summary>

| Mode | Domain | Closest-size llama.cpp checkpoint | Prefill vs reference | Decode vs reference | Sampling |
|---|---|---|---:|---:|---|
| dspark5 | code | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_S) | 143.5% | 511.3% | Model-recommended |
| dspark5 | prose | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_S) | 143.5% | 202.5% | Model-recommended |
| plain | code | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_S) | 146.1% | 103.2% | Model-recommended |

</details>

<details>
<summary><strong>DSpark decode comparisons</strong></summary>

| Engine | Batch | Domain | Baseline | Mode | Baseline decode | Measured decode | Change | Sampling |
|---|---:|---|---|---|---:|---:|---:|---|
| kaiLLM | 1 | code | plain | dspark3 | 33.2 | 43.7 | +31.7% | Model-recommended |
| kaiLLM | 1 | code | plain | dspark5 | 33.2 | 49.4 | +48.8% | Model-recommended |
| kaiLLM | 1 | code | dspark5 | dspark3 | 49.4 | 43.7 | -11.5% | Model-recommended |
| kaiLLM | 1 | prose | dspark5 | dspark3 | 20.3 | 45.1 | +122.0% | Model-recommended |

</details>

#### [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q2_K_XS)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | 649.7 | 495.7 | 33.5 | 35.0 | kaiLLM Default (prefill eager, decode graphs, eager head) · `47daa6ef` · llama.cpp default (ub 512) | Archived greedy |
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | 648.6 | 495.7 | 36.2 | 35.0 | kaiLLM Default (prefill eager, decode graphs, eager head) · `c1db9916` · llama.cpp default (ub 512) | Archived greedy |
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | 647.0 | 494.7 | 36.2 | 33.8 | kaiLLM Default (prefill eager, decode graphs, eager head) · `c1db9916` · llama.cpp default (ub 512) | Model-recommended |
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | 639.5 | 493.1 | 33.6 | 34.9 | kaiLLM Default (prefill eager, decode graphs, eager head) · `47daa6ef` · llama.cpp default (ub 512) | Archived greedy |
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | 640.5 | 493.1 | 36.2 | 34.9 | kaiLLM Default (prefill eager, decode graphs, eager head) · `c1db9916` · llama.cpp default (ub 512) | Archived greedy |
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | 641.3 | 493.3 | 36.1 | 34.1 | kaiLLM Default (prefill eager, decode graphs, eager head) · `c1db9916` · llama.cpp default (ub 512) | Model-recommended |
| 1 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | — | 488.2 | — | 14.5† | llama.cpp default (ub 512) | Model-recommended |
| 1 | dspark5 (deterministic kaiLLM be9d799 · Inkling PR · [llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | $\color{red}{152.6}$ | 489.5 | $\color{red}{7.0}$ | 10.9 | kaiLLM Full graphs · llama.cpp default (ub 512) | Model-recommended |
| 1 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | — | 487.2 | — | 38.2† | llama.cpp default (ub 512) | Model-recommended |
| 1 | dspark5 (deterministic kaiLLM be9d799 · Inkling PR · [llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | $\color{red}{153.7}$ | 490.1 | 21.9 | 12.2 | kaiLLM Full graphs · llama.cpp default (ub 512) | Model-recommended |

<details>
<summary><strong>DSpark decode comparisons</strong></summary>

| Engine | Batch | Domain | Baseline | Mode | Baseline decode | Measured decode | Change | Sampling |
|---|---:|---|---|---|---:|---:|---:|---|
| llama.cpp | 1 | code | plain | dspark5 | 34.1 | 38.2 | +12.1% | Model-recommended |
| llama.cpp | 1 | prose | plain | dspark5 | 33.8 | 14.5 | -57.2% | Model-recommended |

</details>

#### [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_S)

| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | plain ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | $\color{red}{141.3}$ | 489.7 | $\color{red}{17.5}$ | 32.1 | kaiLLM Full graphs · llama.cpp default (ub 512) | Model-recommended |
| 1 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | prose | — | 485.5 | — | 10.0 | llama.cpp default (ub 512) | Model-recommended |
| 1 | dspark5 ([llama.cpp b5ed9f2](https://github.com/unslothai/llama.cpp/commit/b5ed9f2845358124d1cbce91f475011047515e74)) | code | — | 486.0 | — | 9.7† | llama.cpp default (ub 512) | Model-recommended |

<details>
<summary><strong>DSpark decode comparisons</strong></summary>

| Engine | Batch | Domain | Baseline | Mode | Baseline decode | Measured decode | Change | Sampling |
|---|---:|---|---|---|---:|---:|---:|---|
| llama.cpp | 1 | code | plain | dspark5 | 32.1 | 9.7 | -70.0% | Model-recommended |

</details>

## Earlier checkpoint and GPU measurements

| Scope | Checkpoint | GPU | Engine | Batch | Mode | Domain | Prefill | Decode | GPU execution | Sampling |
|---|---|---|---|---:|---|---|---:|---:|---|---|
| Earlier checkpoint | [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/45fd08764f2b0bf16b03a5483a80e5c7aaec8772) (UD-Q2_K_XL) | 1 × RTX 5080 | kaiLLM | 1 | plain | code | 317.9 | 27.2 | Decode-only capture | Archived greedy |
| Earlier checkpoint | [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/45fd08764f2b0bf16b03a5483a80e5c7aaec8772) (UD-Q2_K_XL) | 1 × RTX 5080 | kaiLLM | 1 | plain | prose | 318.0 | 27.2 | Decode-only capture | Archived greedy |
| Retired GPU | [LiquidAI/LFM3-1.7B-GGUF](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/tree/cd0c74c575b5c416f3011b5b4c298077f26ca8c2) (QAD-Q4_0) | 1 × RTX 3060 | kaiLLM | 1 | plain | code | 8084.9 | 167.8 | Unrecorded | Archived greedy |
| Retired GPU | [LiquidAI/LFM3-1.7B-GGUF](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/tree/cd0c74c575b5c416f3011b5b4c298077f26ca8c2) (QAD-Q4_0) | 1 × RTX 3060 | kaiLLM | 1 | plain | prose | 8236.8 | 168.0 | Unrecorded | Archived greedy |

## DSpark5 throughput · 8,192-token prompt checked

Each kaiLLM build matched ordinary decoding on four 48-token outputs after 8,192-token prompts. Three shorter prose cases diverged.

| Checkpoint | GPU | Batch | Domain | Engine | Prefill tok/s | Decode tok/s | Draft tokens accepted/proposed | CUDA graphs | Full text matching ordinary kaiLLM runs | Sampling |
|---|---|---:|---|---|---:|---:|---:|---|---|---|
| [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/8078d149286eec7193795413b9e0643da6d11fa1) (UD-Q2_K_XL) | 1 × RTX 5070 Ti | 1 | prose | kaiLLM `c6f22a6f` | $\color{red}{256.0}$ | $\color{red}{51.7}$ | 52,339/65,976 (79.3%) | Verified | `ba2b758b` no, `fb0fbe68` no | Archived greedy |
| [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/8078d149286eec7193795413b9e0643da6d11fa1) (UD-Q2_K_XL) | 1 × RTX 5070 Ti | 1 | prose | llama.cpp default `e4b1cb53` (ub128) | 990.8 | 81.8 | 48,851/83,415 (58.6%) | Upstream default | — | Archived greedy |
| [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/8078d149286eec7193795413b9e0643da6d11fa1) (UD-Q2_K_XL) | 1 × RTX 5070 Ti | 1 | prose | kaiLLM `fcebd21c` · packed expert gates | 1257.2 | 98.9 | 52,339/65,976 (79.3%) | Verified | `ba2b758b` no, `fb0fbe68` no | Archived greedy |
| [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/8078d149286eec7193795413b9e0643da6d11fa1) (UD-Q2_K_XL) | 1 × RTX 5070 Ti | 1 | code | kaiLLM `c6f22a6f` | $\color{red}{256.3}$ | $\color{red}{54.5}$ | 53,807/58,638 (91.8%) | Verified | `ba2b758b` no, `fb0fbe68` no | Archived greedy |
| [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/8078d149286eec7193795413b9e0643da6d11fa1) (UD-Q2_K_XL) | 1 × RTX 5070 Ti | 1 | code | llama.cpp default `e4b1cb53` (ub128) | 1000.5 | 123.9 | 54,308/56,135 (96.7%) | Upstream default | — | Archived greedy |
| [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/8078d149286eec7193795413b9e0643da6d11fa1) (UD-Q2_K_XL) | 1 × RTX 5070 Ti | 1 | code | kaiLLM `fcebd21c` · packed expert gates | 1250.8 | $\color{red}{110.9}$ | 53,807/58,638 (91.8%) | Verified | `ba2b758b` no, `fb0fbe68` no | Archived greedy |

## Availability

See configuration limits under each checkpoint and GPU.

## Checkpoint files

<details>
<summary>Hugging Face revisions and filenames</summary>

| Family | Checkpoint | Format | GGUF files |
|---|---|---|---|
| LFM3 1.7B | [LiquidAI/LFM3-1.7B-GGUF](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/tree/cd0c74c575b5c416f3011b5b4c298077f26ca8c2) (QAD-Q4_0) | ordinary | `LFM3-1.7B-QAD-Q4_0.gguf` |
| LFM3 1.7B | [kaitchup/LFM3-1.7B-chimachima-kaiPTQ](https://huggingface.co/kaitchup/LFM3-1.7B-chimachima-kaiPTQ/tree/f9963f2a326732d3aa3202c316ebe9e7784999ab) (ITQ 750 MB) | chimachima | `LFM3-1.7B-chimachima-750MB.gguf` |
| LFM3 1.7B | [kaitchup/LFM3-1.7B-suisui-GGUF](https://huggingface.co/kaitchup/LFM3-1.7B-suisui-GGUF/tree/6fcfa6348c8399bd436b7ccd8a0c5e6c9dfe765b) (Q4_0_XS) | suisui | `LFM3-1.7B-suisui-Q4_0_XS.gguf` |
| DaisyRose 27B | [bartowski/RadixArk_DaisyRose-27B-GGUF](https://huggingface.co/bartowski/RadixArk_DaisyRose-27B-GGUF/tree/4514fec3c6573951bce8acc7f850d9eb496301c7) (IQ2_M) | ordinary | `RadixArk_DaisyRose-27B-IQ2_M.gguf` |
| DaisyRose 27B | [kaitchup/DaisyRose-27B-chimachima-kaiPTQ](https://huggingface.co/kaitchup/DaisyRose-27B-chimachima-kaiPTQ/tree/6c1a63358c57f5baf2730574ba418195eb7edfa8) (Q2_K_XS) | chimachima | `DaisyRose-27B-chimachima-kaiPTQ-Q2_K_XS.gguf` |
| DaisyRose 27B | [kaitchup/DaisyRose-27B-dondon-GGUF](https://huggingface.co/kaitchup/DaisyRose-27B-dondon-GGUF/tree/3592e073925dbdbc65a340fca851fc01b8d2c409) (IQ2_M) | dondon | `DaisyRose-27B-dondon-IQ2_M.gguf` |
| Qwen3.8 27B | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/4af51b3542f775b0a8e4b310bfb3f17e01c306c9) (UD-Q2_K_XL) | ordinary | `Qwen3.8-27B-UD-Q2_K_XL.gguf` |
| Qwen3.8 27B | [kaitchup/Qwen3.8-27B-chimachima-kaiPTQ](https://huggingface.co/kaitchup/Qwen3.8-27B-chimachima-kaiPTQ/tree/aa5396581a0d09201312c0262dcfa2d0e1871eba) (Q2_K_XXS) | chimachima | `Qwen3.8-27B-chimachima-Q2_K_XXS.gguf` |
| Inkling 2 Mini | [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/8078d149286eec7193795413b9e0643da6d11fa1) (UD-Q2_K_XL) | ordinary | `Inkling-2-Mini-UD-Q2_K_XL.gguf` |
| Inkling 2 Mini | [ggml-org/Inkling-2-Mini-GGUF](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/tree/0bdc580634ceac4c4d921f452777bf44228de9af) (IQ4_NL) | ordinary | `Inkling-2-Mini-IQ4_NL.gguf` |
| Inkling 2 Mini | [kaitchup/Inkling-2-Mini-chimachima-kaiPTQ](https://huggingface.co/kaitchup/Inkling-2-Mini-chimachima-kaiPTQ/tree/1b27e927c15a6cde7f717926a25b2ce831d08fc1) (Q2_K_XS) | chimachima | `Inkling-2-Mini-chimachima-kaiPTQ-Q2_K_XS.gguf` |
| Inkling 2 Mini | [kaitchup/Inkling-2-Mini-chimachima-kaiPTQ](https://huggingface.co/kaitchup/Inkling-2-Mini-chimachima-kaiPTQ/tree/1b27e927c15a6cde7f717926a25b2ce831d08fc1) (Q3_K_S) | chimachima | `Inkling-2-Mini-chimachima-kaiPTQ-Q3_K_S.gguf` |
| Inkling 2 Mini | [kaitchup/Inkling-2-Mini-chimachima-kaiPTQ](https://huggingface.co/kaitchup/Inkling-2-Mini-chimachima-kaiPTQ/tree/1b27e927c15a6cde7f717926a25b2ce831d08fc1) (Q3_K_M) | chimachima | `Inkling-2-Mini-chimachima-kaiPTQ-Q3_K_M.gguf` |
| Inkling 2 Mini | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q2_K_XS) | dondon | `Inkling-2-Mini-dondon-Q2_K_XS.gguf` |
| Inkling 2 Mini | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_S) | dondon | `Inkling-2-Mini-dondon-Q3_K_S.gguf` |
| Inkling 2 Mini | [kaitchup/Inkling-2-Mini-dondon-GGUF](https://huggingface.co/kaitchup/Inkling-2-Mini-dondon-GGUF/tree/1268c29b21d41c668a5b11044e9e9e2c6f0c2b75) (Q3_K_M) | dondon | `Inkling-2-Mini-dondon-Q3_K_M.gguf` |
| Hy4 Flash | [Xylolabs/Hy4-flash-GGUF](https://huggingface.co/Xylolabs/Hy4-flash-GGUF/tree/cc17e8a68a84db84a367f56fd42fbd40b54b28e2) (IQ2_M-E4) | ordinary | `Hy4-flash-IQ2_M-E4-00001-of-00002.gguf`<br>`Hy4-flash-IQ2_M-E4-00002-of-00002.gguf` |
| Hy4 Flash | [kaitchup/Hy4-flash-dondon-GGUF](https://huggingface.co/kaitchup/Hy4-flash-dondon-GGUF/tree/ee58ad539c09c0b2b290a06f17c99e638a0db139) (Q2_K_M) | dondon | `Hy4-flash-dondon-Q2_K_M-00001-of-00002.gguf`<br>`Hy4-flash-dondon-Q2_K_M-00002-of-00002.gguf` |

</details>

## Earlier short-context benchmarks

The earlier short-context measurements below used plain decoding unless otherwise indicated.

For a single sequence, kaiLLM can still be much slower than llama.cpp, especially on the RTX 5090. Most of the optimization work has focused on batching.

**The llama.cpp baseline was not optimized.** All the comparisons use llama.cpp master (93cf736).

Generation: 512 input / 256 output tokens. Prefill: 2048 input tokens. Rates are batch totals in tokens/s.

### Qwen3.8 27B: kaiLLM vs llama.cpp (same GGUF)

I optimized kaiLLM for Qwen3.8 27B on a rented RTX Pro 6000 (Blackwell, 96 GB). Both engines ran the same unsloth GGUFs in the same session. Each figure is the median of 3 to 5 runs. Rates are kaiLLM / llama.cpp in tokens/s.

Without speculative decoding:

| Checkpoint | Batch 1 | Batch 2 | Batch 8 | Batch 16 | Prefill |
|---|---:|---:|---:|---:|---:|
| UD-Q4_K_XL | 67.8 / 66.4 (**+2.1%**) | 122.6 / 116.8 (**+5.0%**) | 344.0 / 260.8 (**+31.9%**) | 502.3 / 353.3 (**+42.2%**) | 4,736 / 3,198 (**+48.1%**) |
| UD-Q2_K_XL | 94.8 / 92.7 (**+2.3%**) | 162.6 / 153.8 (**+5.7%**) | 355.5 / 273.9 (**+29.8%**) | 508.1 / 353.0 (**+43.9%**) | 4,466 / 3,001 (**+48.8%**) |
| UD-IQ1_M | 81.4 / 109.2 (-25.5%) | 115.0 / 172.4 (-33.3%) | 168.5 / 287.2 (-41.3%) | 138.6 / 333.9 (-58.5%) | 549 / 3,091 (-82.2%) |

UD-IQ1_M was measured with kaiLLM 0.1.0rc5. kaiLLM's kernels for the IQ1_M and IQ2 formats aren't optimized yet: llama.cpp is much faster with this file.


With MTP speculative decoding, each engine at its best draft length (2, 3 or 4) for each batch size:

| Checkpoint | Batch 1 | Batch 2 | Batch 8 | Batch 16 |
|---|---:|---:|---:|---:|
| UD-Q4_K_XL | 140.3 (n=4) / 122.3 (n=4) (**+14.7%**) | 223.9 (n=3) / 187.4 (n=2) (**+19.5%**) | 544.4 (n=3) / 316.5 (n=2) (**+72.0%**) | 620.6 (n=2) / 396.8 (n=3) (**+56.4%**) |
| UD-Q2_K_XL | 153.6 (n=4) / 137.3 (n=3) (**+11.9%**) | 254.2 (n=2) / 210.4 (n=2) (**+20.8%**) | 593.8 (n=3) / 347.3 (n=3) (**+71.0%**) | 631.0 (n=2) / 382.3 (n=2) (**+65.1%**) |


### chimachima kaiPTQ (kaiLLM) vs LiquidAI Q4_K_M (llama.cpp)

llama.cpp can't run kaiPTQ checkpoints, so the baseline is LiquidAI's Q4_K_M GGUF. These rows were measured with kaiLLM 0.1.0rc4 and the first versions of the 500 MB, 750 MB and 1000 MB checkpoints.

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

### Qwen3.8 27B kaiPTQ (kaiLLM) vs UD-IQ1_M (llama.cpp)

llama.cpp can't run kaiPTQ checkpoints, so the baseline is unsloth's UD-IQ1_M GGUF, which has the same file size. Measured with kaiLLM 0.1.0rc5. On the RTX 3060, both engines served 16 sequences with a shorter context than in the other tables, to fit in 12 GB next to the desktop. Each figure is the median of 3 to 5 runs.

| GPU | Checkpoint | Batch 1 | Batch 2 | Batch 8 | Batch 16 | Prefill |
|---|---|---:|---:|---:|---:|---:|
| RTX 3060 | chimachima Q2_K_XXS / UD-IQ1_M | 26.7 / 23.5 (**+13.6%**) | 37.9 / 34.6 (**+9.5%**) | 69.7 / 51.9 (**+34.3%**) | 68.3 / 70.8 (-3.5%) | 966 / 527 (**+83.3%**) |
| RTX Pro 6000 | chimachima Q2_K_XXS / UD-IQ1_M | 106.3 / 109.2 (-2.7%) | 184.4 / 172.4 (**+7.0%**) | 528.6 / 287.2 (**+84.1%**) | 774.5 / 333.9 (**+132.0%**) | 4,772 / 3,091 (**+54.4%**) |

With DSpine (`--dspine`, one sequence), in a chat with thinking enabled (temperature 1.0, top-p 0.95, top-k 20; 16 everyday prompts of up to 1,024 generated tokens), kaiLLM generates the chimachima checkpoint's tokens 1.85× faster on the RTX Pro 6000 (207.8 instead of 112.1 tokens/s) and 1.92× faster on the RTX 3060 (52.3 instead of 27.2 tokens/s).

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
