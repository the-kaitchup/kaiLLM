# Long-context throughput benchmark

The workload is an 8,192-token prompt followed by exactly 65,536 generated tokens per request. A batch of 32 completes 2,097,152 generated tokens in one domain. The harness runs raw completions, greedy decoding, seed 42, EOS ignored, prefix caching disabled, and a 73,728-token context per active sequence. It rejects early termination, truncated inputs and incomplete batches.

## Checkpoints and prompts

The manifest pins the Hugging Face repository, commit, filename, byte count and SHA-256 for every GGUF shard. The campaign resolves the latest main branch at the start, then downloads the pinned objects from Hugging Face on the benchmark machine. Ordinary GGUFs cover QAD Q4_0, IQ2_M, dynamic mixed K/IQ and MoE formats. They accompany LFM3 suisui and chimachima, DaisyRose dondon and RCO chimachima, Qwen chimachima, and Hy4 dondon.

The public prose input comes from *Moby-Dick*, starting with chapter one. The code input comes from CPython's abstract-syntax-tree module. Each family uses its native tokenizer at the pinned revision, without added special tokens. Thirty-two deterministic source windows provide distinct request prefixes. A separate window primes the server. Batch 2 uses the first two timed windows from batch 32. The raw input token IDs and SHA-256 match across engines, quantizations of a family, and GPUs. [Prose source](https://github.com/GITenberg/Moby-Dick--Or-The-Whale_2701/blob/bdf1948e6cd00963730971e5624e764a35f238c3/2701-0.txt) · [Code source](https://github.com/python/cpython/blob/60403a5409ff2c3f3b07dd2ca91a7a3e096839c7/Lib/ast.py)

## Hardware

The local RTX 3060 runs when idle. Rental targets are the RTX 3090, 5080, 5090, RTX PRO 6000, and GB10. The PRO 6000 and GB10 use one device each. Other targets can use two or four devices when the model requires it.

## Timing

Requests start concurrently on the same host as the server. After an untimed warm-up, three independent one-output-token batches measure prompt processing. Prefill throughput is the input-token count across the batch divided by their median wall time. A separate full-length batch uses the same input. Decode throughput is the remaining 65,535 tokens per request divided by full wall time minus the measured prefill time. This estimates the sustained generation phase with identical client-side timing for both engines, including scheduling and HTTP overhead. The timing archive includes the three prefill durations, full duration, input hashes and generated-text hashes. The code/prose labels identify the prompt source. Fixed-length greedy continuations can repeat.

Batches are 1, 2, 4, 8, 16 and 32. Only full-context batches that fit count. kaiLLM's startup capacity and llama.cpp's allocated context determine feasibility. Both engines keep the model body on GPU and use an unquantized 16-bit attention KV cache. Hy4's large Engram lookup table stays on the host.

| Target | Speculative settings | Plain comparison |
|---|---|---|
| LFM3 donor, suisui and chimachima | N-gram with 3 draft tokens at batches 1, 2 and 4, 1 draft token at batch 8, and drafting off at batches 16 and 32 | Same checkpoint and batch |
| Qwen3.8-family checkpoint with a native MTP head | 5 draft tokens | Same checkpoint and batch |
| DaisyRose and Qwen3.8 with Tencent DSpine | 7 and 15 draft tokens at batch 1 | Same checkpoint and GPU |
| Inkling 2 Mini with its DSpark GGUF | 5 draft tokens in llama.cpp | Same checkpoint and batch |

LFM3 follows the merged serving policy. kaiLLM searches matching suffixes of 2–8 tokens. The pinned llama.cpp implementation takes a single lookup length, set to 2, and the same draft-token limit for each batch. A plain row at batch 16 or 32 also represents LFM3’s default. [LFM3 serving policy](../../runtime-source/vllm_gguf_plugin/serving_policy.py)

## Reproduction

Use the pinned kaiLLM binary release in the checkout's `release.json` and build the manifest's llama.cpp commit with CUDA for the target GPU. The runtime assembly overlays the merged Python and CUDA sources onto the installed release in a separate directory. Run the following from the repository root. Set `KAI_PY` to the release environment's Python and `LLAMA_SERVER` to the built executable. The prompt preparation environment needs `tokenizers` and `huggingface_hub`. Access-controlled model downloads require an authorized Hugging Face token.

```bash
B=benchmarks/long_context
python "$B/weights.py" fetch lfm-qad "$PWD/work/weights"
python "$B/benchmark.py" prompts lfm "$PWD/work/banks/lfm.json"
"$KAI_PY" "$B/runtime.py" "$PWD/work/runtime"
for engine in kaiLLM llama.cpp; do
  "$KAI_PY" "$B/worker.py" --engine "$engine" --mode plain --batch 1 \
    --checkpoint lfm-qad --weights "$PWD/work/weights" --banks "$PWD/work/banks" \
    --python "$KAI_PY" --runtime "$PWD/work/runtime" --llama "$LLAMA_SERVER" \
    --out "$PWD/work/results" --kai-commit "$(git rev-parse HEAD)"
done
python "$B/report.py" "$B/results.jsonl" speed.md "$PWD/work/results"
```

Run `python "$B/matrix.py"` to list the full cell matrix. Optional checkpoint keys and `--batches` select a subset. Each cell runs both prompt domains. Use the same entry points for every GPU and model. Pass `--devices 0 1` or `--devices 0 1 2 3` for multi-GPU runs. The worker records the launch command, runtime revision, GPU count and model, memory, power limit and driver. The report pairs only identical model and prompt revisions, physical GPU sets, power limits, drivers, modes and batch sizes. A red kaiLLM metric marks a llama.cpp lead above 10% using unrounded rates.

The timing archive identifies the equivalent public runtime source for each run. Revisions were remapped after a repository history cleanup. Executable runtime code and measured rates are unchanged. LFM3 n-gram measurements use the shipped ABI2 verifier identities in commit `3bb3967`.
