# SPDX-License-Identifier: Apache-2.0
"""Throughput of an OpenAI-compatible server: N concurrent completions of M tokens each (ignore_eos), reporting
aggregate decode tok/s and per-request latency; or one long-context request (--needle N): a passphrase at the start
of an N-token prompt, asked for at the end (greedy chat, streamed: time to first token = prefill, then decode speed
at that context).  Standard library only.

    python tests/gpu/inkling_serve_bench.py --url http://127.0.0.1:8801 --model inkling --concurrency 32 --tokens 256
    python tests/gpu/inkling_serve_bench.py --url http://127.0.0.1:8801 --needle 100000 --tokens 400
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
import urllib.request


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def filler(n_tokens: int, seed: int = 0) -> str:
    """~n_tokens of varied plain sentences (about 1.3 tokens per word)."""
    rng = random.Random(seed)
    subjects = ["The river", "A small village", "The old library", "Every winter", "The committee", "A traveler",
                "The northern road", "The museum", "Our neighbor", "The harbor"]
    verbs = ["remembered", "described", "measured", "ignored", "painted", "repaired", "counted", "visited",
             "discussed", "recorded"]
    objects = ["the quiet hills", "seven lanterns", "a broken clock", "the spring market", "forty letters",
               "the iron bridge", "a field of barley", "the evening bells", "an unfinished map", "the copper roofs"]
    words, out = 0, []
    while words < n_tokens / 1.3:
        s = f"{rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(objects)} in year {rng.randint(1200, 1999)}."
        out.append(s)
        words += len(s.split())
    return " ".join(out)


def needle(a):
    secret = "violet-otter-42"
    content = (f"Remember this: the secret passphrase is '{secret}'.\n\n" + filler(a.needle) +
               "\n\nWhat is the secret passphrase mentioned at the very beginning? Reply with the passphrase only.")
    body = {"model": a.model, "messages": [{"role": "user", "content": content}], "max_tokens": a.tokens,
            "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    first = None
    text, usage, chunks = [], None, 0
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                delta = ch.get("delta", {})
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or delta.get("reasoning") or "")
                if piece:
                    if first is None:
                        first = time.time()
                    chunks += 1
                    text.append(piece)
    end = time.time()
    out = "".join(text)
    comp = usage["completion_tokens"] if usage else chunks
    res = {"needle_prompt_tokens": usage["prompt_tokens"] if usage else None, "completion_tokens": comp,
           "ttft_s": round(first - t0, 2) if first else None,
           "prefill_tok_s": round(usage["prompt_tokens"] / (first - t0), 1) if usage and first else None,
           "decode_tok_s": round((comp - 1) / (end - first), 2) if first and end > first and comp > 1 else None,
           "found": secret in out, "answer_tail": out[-200:]}
    print(json.dumps(res), flush=True)


def throughput(a):
    prompt = " ".join(["The history of science is a long story of patient observation."] * max(1, a.prompt_words // 10))
    out, lock = [], threading.Lock()

    def one(i):
        t0 = time.time()
        r = post(a.url + "/v1/completions", {"model": a.model, "prompt": f"{i}: {prompt}", "max_tokens": a.tokens,
                                             "temperature": 1.0, "ignore_eos": True})
        dt = time.time() - t0
        with lock:
            out.append((r["usage"]["completion_tokens"], r["usage"]["prompt_tokens"], dt))

    t0 = time.time()
    th = [threading.Thread(target=one, args=(i,)) for i in range(a.concurrency)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.time() - t0
    gen = sum(o[0] for o in out)
    print(json.dumps({"concurrency": a.concurrency, "tokens_per_request": a.tokens, "prompt_tokens": out[0][1],
                      "generated": gen, "wall_s": round(wall, 2), "aggregate_tok_s": round(gen / wall, 1),
                      "per_request_tok_s": round(sum(o[0] / o[2] for o in out) / len(out), 2)}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8801")
    ap.add_argument("--model", default="inkling")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--prompt-words", type=int, default=64)
    ap.add_argument("--needle", type=int, default=0, help="long-context request of about this many prompt tokens")
    a = ap.parse_args()
    (needle if a.needle else throughput)(a)


if __name__ == "__main__":
    main()
