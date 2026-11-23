# SPDX-License-Identifier: Apache-2.0
"""Greedy continuation of a recorded completion's prefix (inkling_greedy_compare.py run output) on another server /
build: the chat prompt (thinking on) + the recorded tokens up to --at, continued at temperature 0, compared token by
token with what the recorded run produced next.  Tells whether a behaviour seen late in one run (e.g. a repetition
loop) belongs to that trajectory (any build continues it) or to the build that produced it.  Standard library only.

    python3 tests/gpu/inkling_greedy_prefix.py --url http://127.0.0.1:8801 --prompts p.json --run on.json \
        --question 149 --at 3400 [--tokens 400]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8801")
    ap.add_argument("--model", default="inkling")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--question", type=int, required=True)
    ap.add_argument("--at", type=int, nargs="+", required=True)
    ap.add_argument("--tokens", type=int, default=400)
    a = ap.parse_args()
    prompt = next(p for p in json.load(open(a.prompts)) if p.get("dataset_index") == a.question)
    rec = next(r for r in json.load(open(a.run)) if r.get("dataset_index") == a.question)
    tok = post(a.url + "/tokenize", {"model": a.model, "messages": [{"role": "user", "content": prompt["prompt"]}],
                                     "add_generation_prompt": True,
                                     "chat_template_kwargs": {"reasoning_effort": "high"}})
    ids = tok["tokens"]
    for at in a.at:
        r = post(a.url + "/v1/completions", {"model": a.model, "prompt": ids + rec["tokens"][:at],
                                              "max_tokens": a.tokens, "temperature": 0.0, "logprobs": 0,
                                              "skip_special_tokens": False, "return_tokens_as_token_ids": True})
        got = [int(t.split(":", 1)[1]) for t in r["choices"][0]["logprobs"]["tokens"]]
        want = rec["tokens"][at:at + len(got)]
        same = next((i for i in range(min(len(got), len(want))) if got[i] != want[i]), min(len(got), len(want)))
        print(f"q{a.question} prompt {len(ids)} + {at} recorded tokens: continuation matches the recorded run for "
              f"{same} of {len(got)} tokens; text: {r['choices'][0]['text'][:200]!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
