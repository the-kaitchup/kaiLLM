# SPDX-License-Identifier: Apache-2.0
"""Greedy completions of fixed prompts from an OpenAI-compatible server (sent concurrently, so they decode as one
batch), with the chosen tokens' log-probs; and a comparison of two such runs (e.g. eager vs CUDA graphs).

    python tests/gpu/inkling_serve_compare.py collect --url http://127.0.0.1:8801 --out eager.json [--tokens 96]
    python tests/gpu/inkling_serve_compare.py compare eager.json graphs.json

compare: per prompt, the identical token prefix and the largest |delta log-prob| on it (greedy decoding: tokens stay
identical until a near tie flips).  Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import urllib.request

PROMPTS = [
    "The three primary colors are",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Question: What is the capital of France?\nAnswer:",
    "The derivative of x^3 + 2x with respect to x is",
    "In 1905, Albert Einstein published four papers that",
    "Translate to French: 'The weather is nice today.'\nFrench:",
    "A prime number is a natural number greater than 1 that",
    "Photosynthesis converts light energy into",
]


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def collect(a):
    out = [None] * len(PROMPTS)

    def one(i):
        r = post(a.url + "/v1/completions", {"model": a.model, "prompt": PROMPTS[i], "max_tokens": a.tokens,
                                             "temperature": 0.0, "logprobs": 1, "ignore_eos": True,
                                             "return_tokens_as_token_ids": True})
        lp = r["choices"][0]["logprobs"]
        out[i] = {"prompt": PROMPTS[i], "text": r["choices"][0]["text"], "tokens": lp["tokens"],
                  "logprobs": lp["token_logprobs"]}

    th = [threading.Thread(target=one, args=(i,)) for i in range(len(PROMPTS))]
    for t in th:
        t.start()
    for t in th:
        t.join()
    json.dump(out, open(a.out, "w"), indent=1)
    for o in out:
        print(repr(o["text"][:100]))


def compare(a):
    A, B = json.load(open(a.a)), json.load(open(a.b))
    same_all, worst = 0, 0.0
    for x, y in zip(A, B):
        n = 0
        while n < min(len(x["tokens"]), len(y["tokens"])) and x["tokens"][n] == y["tokens"][n]:
            n += 1
        d = max((abs(p - q) for p, q in zip(x["logprobs"][:n], y["logprobs"][:n])), default=0.0)
        worst = max(worst, d)
        same_all += n == len(x["tokens"])
        print(f"{x['prompt'][:40]!r:44s} identical prefix {n:4d} / {len(x['tokens'])}  max |dlogprob| {d:.2e}")
    print(f"identical completions {same_all} / {len(A)}, max |dlogprob| on the common prefixes {worst:.2e}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--url", default="http://127.0.0.1:8801")
    c.add_argument("--model", default="inkling")
    c.add_argument("--tokens", type=int, default=96)
    c.add_argument("--out", required=True)
    d = sub.add_parser("compare")
    d.add_argument("a")
    d.add_argument("b")
    a = ap.parse_args()
    (collect if a.cmd == "collect" else compare)(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
