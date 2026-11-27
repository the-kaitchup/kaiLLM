"""Compare next-token distributions at fixed public prose and code prefixes."""
import argparse
import json
import math
from pathlib import Path
import urllib.request

from benchmark import hash_tokens, verify_prompt_bank, write_json


def predict(engine, url, prompt, *, output_tokens=8, timeout=240):
    payload = dict(prompt=prompt, temperature=0, seed=42, ignore_eos=True)
    if engine == "llama.cpp":
        path = "/completion"
        payload.update(n_predict=output_tokens, n_probs=8, cache_prompt=False, return_tokens=True)
    else:
        path = "/v1/completions"
        payload.update(model="bench", max_tokens=output_tokens, logprobs=8, return_token_ids=True, return_tokens_as_token_ids=True)
    request = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.load(response)
    if engine == "llama.cpp":
        if data["tokens_evaluated"] != len(prompt):
            raise ValueError("Reference truncated the input")
        tokens = data["tokens"]
        top = [{str(item["id"]): item["logprob"] for item in step["top_logprobs"]}
               for step in data["completion_probabilities"]]
    else:
        if data["usage"]["prompt_tokens"] != len(prompt):
            raise ValueError("Candidate truncated the input")
        choice, = data["choices"]
        tokens = choice["token_ids"]
        top = [{str(int(key.removeprefix("token_id:"))): value for key, value in step.items()}
               for step in choice["logprobs"]["top_logprobs"]]
    if len(tokens) != output_tokens or len(top) != output_tokens or any(str(token) not in scores for token, scores in zip(tokens, top)):
        raise ValueError("Incomplete greedy token or log-probability capture")
    return {"prompt_sha256": hash_tokens(prompt), "tokens": tokens, "top_logprobs": top}


def capture(engine, url, bank, family, prefix_sizes=(32, 127, 257, 511)):
    rows = []
    for domain in ("prose", "code"):
        verify_prompt_bank(bank, family, domain)
        for i in range(3):
            source = bank["domains"][domain]["rows"][i]
            for size in prefix_sizes:
                if not 1 <= size <= len(source):
                    raise ValueError("Validation prefix is outside the prompt bank")
                rows.append({"domain": domain, "prompt_tokens": size, **predict(engine, url, source[:size])})
    return {"engine": engine, "rows": rows}


def compare(reference, candidate):
    left = reference["rows"]
    right = candidate["rows"]
    if not left or len(left) != len(right):
        raise ValueError("Both engines must cover the same prefixes")
    matched = 0
    first = 0
    compared = 0
    differences = []
    for ref, test in zip(left, right):
        if (ref["domain"], ref["prompt_sha256"]) != (test["domain"], test["prompt_sha256"]):
            raise ValueError("The two captures contain different prompts")
        for step, (a, b) in enumerate(zip(ref["tokens"], test["tokens"], strict=True)):
            compared += 1
            if a == b:
                matched += 1
                first += step == 0
                continue
            # Later positions have different preceding tokens and cannot be compared.
            scores = ref["top_logprobs"][step]
            gap = scores[str(a)] - scores[str(b)] if str(b) in scores else None
            if gap is not None and not math.isfinite(gap):
                raise ValueError("Invalid reference log probability")
            differences.append({"domain": ref["domain"], "prompt_sha256": ref["prompt_sha256"], "step": step,
                                "reference_token": a, "candidate_token": b, "reference_logprob_gap": gap})
            break
    return {"prefixes": len(left), "first_tokens_matching": first, "positions_with_identical_context": compared,
            "greedy_tokens_matching": matched, "first_differences": differences}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("engine", choices=("kaiLLM", "llama.cpp"))
    cap.add_argument("url")
    cap.add_argument("family")
    cap.add_argument("bank", type=Path)
    cap.add_argument("output", type=Path)
    diff = sub.add_parser("compare")
    diff.add_argument("reference", type=Path)
    diff.add_argument("candidate", type=Path)
    args = parser.parse_args()
    if args.action == "capture":
        write_json(args.output, capture(args.engine, args.url, json.loads(args.bank.read_bytes()), args.family))
    else:
        print(json.dumps(compare(json.loads(args.reference.read_bytes()), json.loads(args.candidate.read_bytes())), indent=2))
