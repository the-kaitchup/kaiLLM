# SPDX-License-Identifier: Apache-2.0
"""Greedy (temperature 0) chat completions of fixed prompts, one request at a time (batch 1: the same kernels every
step), thinking on (reasoning_effort high), special tokens kept, token ids recorded; and a token-by-token comparison
of two such runs (e.g. KAILLM_INKLING_COMPILE=1 vs 0): first divergence position and the text around it.
Standard library only.

    python3 tests/gpu/inkling_greedy_compare.py run --url http://127.0.0.1:8801 --prompts p.json --out a.json [--max-tokens 8192]
    python3 tests/gpu/inkling_greedy_compare.py compare a.json b.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request


def post(url, body, timeout=7200):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run(a):
    prompts = json.load(open(a.prompts))
    out = []
    for p in prompts:
        t0 = time.time()
        r = post(a.url + "/v1/chat/completions", {
            "model": a.model, "messages": [{"role": "user", "content": p["prompt"]}], "max_tokens": a.max_tokens,
            "temperature": 0.0, "logprobs": True, "top_logprobs": 0, "skip_special_tokens": False,
            "return_tokens_as_token_ids": True, "chat_template_kwargs": {"reasoning_effort": "high"}})
        ch = r["choices"][0]
        content = (ch.get("logprobs") or {}).get("content") or []
        toks = [int(t["token"].split(":", 1)[1]) for t in content]
        # each token's bytes (hex): exact text windows around a divergence
        pieces = [bytes(t.get("bytes") or []).hex() for t in content]
        out.append({"dataset_index": p.get("dataset_index"), "text": ch["message"]["content"], "tokens": toks,
                    "bytes": pieces, "finish_reason": ch.get("finish_reason"), "seconds": round(time.time() - t0, 1)})
        print(f"{p.get('dataset_index')}: {len(toks)} tokens, {ch.get('finish_reason')}, {time.time() - t0:.0f} s",
              flush=True)
    json.dump(out, open(a.out, "w"))


def answer(text):
    """The last `ANSWER: X` letter of a completion (the GPQA protocol's rule), or None."""
    m = re.findall(r"ANSWER\s*:\s*\(?([A-D])\b", text or "")
    return m[-1] if m else None


def compare(a):
    A, B = json.load(open(a.a)), json.load(open(a.b))
    for x, y in zip(A, B):
        n = min(len(x["tokens"]), len(y["tokens"]))
        d = next((i for i in range(n) if x["tokens"][i] != y["tokens"][i]), None)
        same = d is None and len(x["tokens"]) == len(y["tokens"])
        where = "identical" if same else (f"first divergence at token {d}" if d is not None else
                                         f"identical over {n}, lengths {len(x['tokens'])} / {len(y['tokens'])}")
        print(f"question {x['dataset_index']}: {len(x['tokens'])} / {len(y['tokens'])} tokens "
              f"({x['finish_reason']} / {y['finish_reason']}; answers {answer(x['text'])} / {answer(y['text'])}): "
              f"{where}")
        if d is not None:
            def window(r, a0, b0):
                if r.get("bytes"):
                    return bytes.fromhex("".join(r["bytes"][a0:b0])).decode("utf-8", "replace")
                frac = len(r["text"]) / max(len(r["tokens"]), 1)            # older runs: approximate
                return r["text"][int(a0 * frac):int(b0 * frac)]
            print(f"    before:  {window(x, max(0, d - 40), d)!r}")
            print(f"    A after: {window(x, d, d + 60)!r}")
            print(f"    B after: {window(y, d, d + 60)!r}")
            print(f"    A end:   {x['text'][-240:]!r}")
            print(f"    B end:   {y['text'][-240:]!r}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", default="http://127.0.0.1:8801")
    r.add_argument("--model", default="inkling")
    r.add_argument("--prompts", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--max-tokens", type=int, default=8192)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    (run if a.cmd == "run" else compare)(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
