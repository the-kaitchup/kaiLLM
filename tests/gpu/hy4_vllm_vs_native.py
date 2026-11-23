"""GPU check of kaiLLM's Hy4 path through the real vLLM engine (scheduler, model runner, CUDA kernels).

    KAILLM_HY4_ROOT="$PWD" PYTHONPATH="$PWD/tests/gpu/site" \
    python tests/gpu/hy4_vllm_vs_native.py <model.gguf> (--ref <cpu.npz> | --dump <dump_logits>) \
        [--dtype bfloat16] [--windows 4 --len 256] [--tokens tokens.bin | --text corpus.txt]
        [--max-batched N] [--decode 32] [--ngl 999]

Reference, one of:
  --ref  logits saved by tests/test_hy4_vllm_cpu.py --save (float32 CPU path, itself checked against
         transformers: tests/test_hy4_vllm_vs_hf_cpu.py). Use it for the toy, whose 4-head indexer
         has near-tie group picks that llama.cpp's BF16 pooled keys flip.
  --dump llama.cpp's dump_logits on the same GGUF (real models). Runs before vLLM starts, so the two
         never hold the model on the GPU together; HY4_NGL=<--ngl> offloads it with a CUDA build.

Checks:
  prefill  teacher-forced prompt logprobs of every window, all windows submitted together (batching);
           --max-batched below the window length forces chunked prefill.
  decode   (--decode G) greedy generation of G tokens after the first half of each window, compared
           with the teacher-forced logprobs of the same final sequence (decode path vs prefill path).
Metrics: EAR@10 = reference mass on kaiLLM's top-10 / reference top-10 mass; top-1 agreement;
max |logprob difference| over kaiLLM's top-10 tokens.
"""
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def native_logits(dump, gguf, ids, ngl):
    with tempfile.TemporaryDirectory() as td:
        t, o = Path(td) / "t.i32", Path(td) / "o.f32"
        np.asarray(ids, dtype=np.int32).tofile(t)
        env = dict(os.environ, HY4_NGL=str(ngl))
        subprocess.run([str(dump), str(gguf), str(t), str(o), str(os.cpu_count() or 16)], check=True, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return np.fromfile(o, dtype=np.float32).reshape(len(ids), -1)


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def score(prompt_logprobs, ref_logits, per_position=None):
    """prompt_logprobs[i] (token i given < i) vs reference row i - 1."""
    ear, agree, dmax, n = 0.0, 0, 0.0, 0
    ref_lp = log_softmax(ref_logits)
    for i in range(1, ref_logits.shape[0] + 1):
        lp = prompt_logprobs[i] if i < len(prompt_logprobs) else None
        if not lp:
            continue
        top = sorted(lp.items(), key=lambda kv: -kv[1].logprob)[:10]
        ids = [t for t, _ in top]
        r = ref_lp[i - 1]
        p = np.exp(r)
        e = p[ids].sum() / np.sort(p)[-10:].sum()
        d = max(abs(v.logprob - r[t]) for t, v in top)
        ear += e
        agree += int(ids[0] == int(r.argmax()))
        dmax = max(dmax, d)
        n += 1
        if per_position is not None:
            srt = np.sort(r)
            per_position.append({"pos": i - 1, "ear10": float(e), "top1": int(ids[0] == int(r.argmax())),
                                 "dmax": float(d), "ref_gap": float(srt[-1] - srt[-2]),
                                 "ref_top1_p": float(np.exp(srt[-1]))})
    return {"positions": n, "ear10": ear / max(n, 1), "top1": agree / max(n, 1), "max_dlogprob": dmax}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf", type=Path)
    ap.add_argument("--ref", type=Path, default=None)
    ap.add_argument("--dump", type=Path, default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--len", type=int, default=256)
    ap.add_argument("--tokens", type=Path, default=None, help="input file containing int32 token IDs")
    ap.add_argument("--text", type=Path, default=None)
    ap.add_argument("--max-batched", type=int, default=None)
    ap.add_argument("--decode", type=int, default=0)
    ap.add_argument("--ngl", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.6)
    ap.add_argument("--per-position", type=Path, default=None, help="write per-position prefill metrics (JSON)")
    a = ap.parse_args()
    assert (a.ref is None) != (a.dump is None), "pass exactly one of --ref / --dump"

    if a.ref is not None:
        z = np.load(a.ref)
        wins = [z["seq_a"].tolist(), z["seq_b"].tolist()]
        refs = [z["single"], z["single_b"]]
    else:
        if a.tokens is not None:
            ids = np.fromfile(a.tokens, dtype=np.int32).tolist()
        else:
            from transformers import AutoTokenizer
            from vllm_gguf_plugin.hy4_metadata import prepare_hy4_metadata
            tok = AutoTokenizer.from_pretrained(prepare_hy4_metadata(a.gguf))
            text = a.text.read_text() if a.text else Path(__file__).read_text() * 50
            ids = tok(text, add_special_tokens=False)["input_ids"]
        wins = [ids[w * a.len:(w + 1) * a.len] for w in range(a.windows)]
        refs = [native_logits(a.dump, a.gguf, w, a.ngl) for w in wins]   # before vLLM takes the GPU

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    longest = max(len(w) for w in wins)
    kw = {} if a.max_batched is None else {"max_num_batched_tokens": a.max_batched, "enable_chunked_prefill": True}
    llm = LLM(model=str(a.gguf), dtype=a.dtype, enforce_eager=True, max_model_len=longest + max(a.decode, 1) + 8,
              gpu_memory_utilization=a.gpu_mem, max_num_seqs=max(4, len(wins)), **kw)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=10, temperature=0.0)
    outs = llm.generate([TokensPrompt(prompt_token_ids=w) for w in wins], sp, use_tqdm=False)
    rows, pp = [], []
    for k, (out, ref) in enumerate(zip(outs, refs)):
        mine = [] if a.per_position else None
        rows.append({"check": "prefill", "window": k, **score(out.prompt_logprobs, ref[: len(wins[k]) - 1], mine)})
        print(json.dumps(rows[-1]))
        if mine is not None:
            pp.append(mine)
    if a.per_position:
        a.per_position.write_text(json.dumps(pp))

    if a.decode:
        half = [w[: len(w) // 2] for w in wins]
        gen = llm.generate([TokensPrompt(prompt_token_ids=h) for h in half],
                           SamplingParams(max_tokens=a.decode, temperature=0.0, logprobs=10), use_tqdm=False)
        full = [h + list(g.outputs[0].token_ids) for h, g in zip(half, gen)]
        tf = llm.generate([TokensPrompt(prompt_token_ids=f) for f in full], sp, use_tqdm=False)
        for k, (g, t, h) in enumerate(zip(gen, tf, half)):
            dmax, agree, n = 0.0, 0, 0
            for j, step in enumerate(g.outputs[0].logprobs):        # step j predicts full[len(h) + j]
                ref = t.prompt_logprobs[len(h) + j]
                if not ref:
                    continue
                top = sorted(step.items(), key=lambda kv: -kv[1].logprob)[:10]
                common = [(tid, v.logprob, ref[tid].logprob) for tid, v in top if tid in ref]
                dmax = max([dmax] + [abs(x - y) for _, x, y in common])
                agree += int(top[0][0] == max(ref.items(), key=lambda kv: kv[1].logprob)[0])
                n += 1
            rows.append({"check": "decode-vs-prefill", "window": k, "steps": n, "top1": agree / max(n, 1),
                         "max_dlogprob_common_top10": dmax})
            print(json.dumps(rows[-1]))
    pre = [r for r in rows if r["check"] == "prefill"]
    print(json.dumps({"mean_ear10": float(np.mean([r["ear10"] for r in pre])),
                      "mean_top1": float(np.mean([r["top1"] for r in pre]))}))


if __name__ == "__main__":
    main()
