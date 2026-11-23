# kaiLLM prefill and decode speed

Each request uses **8,192 input tokens and 65,536 output tokens**. Rates are aggregate tokens per second at the listed concurrency. A red kaiLLM value marks a workload as **not optimized** when llama.cpp was more than 10% faster on the same checkpoint, GPU, workload, batch, and decoding mode.

[Protocol, commands and prompt sources](benchmarks/long_context/README.md) · [Pinned checkpoints](benchmarks/long_context/manifest.json) · [Timing data](benchmarks/long_context/results.jsonl)

## Results

| GPU | Checkpoint | Mode | Batch | Input | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode |
|---|---|---|---:|---|---:|---:|---:|---:|
| GeForce RTX 5090 | lfm-chima | ngram3 | 1 | code | 37272.3 | — | 1599.9† | — |
| GeForce RTX 5090 | lfm-chima | ngram3 | 1 | prose | 38786.6 | — | 1603.8 | — |
| GeForce RTX 5090 | lfm-chima | plain | 1 | code | 37592.6 | — | 451.1 | — |
| GeForce RTX 5090 | lfm-chima | plain | 1 | prose | 37731.9 | — | 450.9 | — |
| GeForce RTX 5090 | lfm-qad | ngram3 | 1 | code | 51734.5 | — | 1912.6 | — |
| GeForce RTX 5090 | lfm-qad | ngram3 | 1 | prose | 52814.2 | — | 2016.7 | — |
| GeForce RTX 5090 | lfm-qad | plain | 1 | code | 56235.5 | 33115.7 | 559.8 | 419.0 |
| GeForce RTX 5090 | lfm-qad | plain | 1 | prose | 59933.2 | 34327.9 | 560.0 | 412.5 |
| GeForce RTX 5090 | lfm-suisui | ngram3 | 1 | code | 51467.9 | — | 1804.7† | — |
| GeForce RTX 5090 | lfm-suisui | ngram3 | 1 | prose | 51606.1 | 34036.5 | 1807.1† | 525.7 |
| GeForce RTX 5090 | lfm-suisui | plain | 1 | code | 52829.1 | 34539.5 | 524.0 | 414.8 |
| GeForce RTX 5090 | lfm-suisui | plain | 1 | prose | 53587.7 | 33857.7 | 523.8 | 394.2 |
| RTX PRO 6000 Blackwell Workstation Edition | daisy-rco | dspine15 | 1 | code | 4181.5 | — | 245.2 | — |
| RTX PRO 6000 Blackwell Workstation Edition | daisy-rco | dspine15 | 1 | prose | 4224.5 | — | 261.9† | — |
| RTX PRO 6000 Blackwell Workstation Edition | daisy-rco | dspine7 | 1 | code | 4038.1 | — | 416.6† | — |
| RTX PRO 6000 Blackwell Workstation Edition | daisy-rco | dspine7 | 1 | prose | 4314.2 | — | 122.0† | — |
| RTX PRO 6000 Blackwell Workstation Edition | daisy-rco | plain | 1 | code | 4077.4 | — | 96.4 | — |
| RTX PRO 6000 Blackwell Workstation Edition | daisy-rco | plain | 1 | prose | 4333.7 | — | 96.5 | — |

† The generated text differed from that engine’s plain run on at least one request. All scored runs completed the specified output length. A dash means there is no matched measurement in that cell. kaiPTQ/chimachima requires kaiLLM. DSpine uses batch 1.

## Hardware coverage

The campaign covers the RTX 3060, 3090, 5080, 5090, RTX PRO 6000, and GB10.
