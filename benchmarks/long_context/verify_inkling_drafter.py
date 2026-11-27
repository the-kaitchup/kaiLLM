"""Verify that Inkling block decoding preserves greedy target tokens after rejection."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request


def counters(url, *, required=False):
    with urllib.request.urlopen(url + "/metrics", timeout=10) as response:
        metrics = response.read().decode().splitlines()
    result = {}
    for name in ("draft_tokens", "accepted_tokens"):
        metric = "vllm:spec_decode_num_" + name + "_total"
        rows = [line for line in metrics if line.startswith(metric + "{") or line.startswith(metric + " ")]
        if required and not rows:
            raise ValueError(f"Missing speculative token counter: {metric}")
        result[name] = sum(int(float(line.split("}", 1)[-1].removeprefix(metric).split()[0])) for line in rows)
    return result


SLOTS = (1, 2, 4, 8, 16, 32)


def request_labels(slots):
    if slots in (1, 2):
        return ("prose", "code")
    return tuple(f"{domain}-{i}" for domain in ("prose", "code") for i in range(slots))


def capture(url, bank, model, concurrency=2, prompt_length=128, *, schedule=None, native=False):
    if concurrency not in SLOTS:
        raise ValueError("Unsupported Inkling slot count")
    wide = concurrency > 2
    if wide and schedule is None:
        raise ValueError("Wide Inkling checks require physical model-forward observations")
    domains = ("prose", "code")
    prompts = {}
    for domain in domains:
        rows = bank["domains"][domain]["rows"]
        for i in range(concurrency if wide else 1):
            label = f"{domain}-{i}" if wide else domain
            prompts[label] = rows[i][:prompt_length]
            if len(prompts[label]) != prompt_length:
                raise ValueError("Incomplete qualifying prompt for " + label)
    before = counters(url, required=native)

    def generate(domain):
        payload = dict(model=model, prompt=prompts[domain], max_tokens=48, temperature=0,
                       seed=42, ignore_eos=True, return_token_ids=True)
        req = urllib.request.Request(url + "/v1/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as response:
            result = json.load(response)
        ids = result["choices"][0]["token_ids"]
        if len(ids) != 48 or result["usage"]["prompt_tokens"] != prompt_length:
            raise ValueError("Incomplete correctness sample for " + domain)
        return ids

    outputs = {}
    width_by_domain = {}
    with ThreadPoolExecutor(concurrency) as pool:
        if wide:
            from check_inkling_draft_overlap import observed_steps, overlap
            from verify_batch_schedule import prompt_hash
            marker = schedule / "recording"
            trace = schedule / "steps.jsonl"
            for domain in domains:
                labels = [f"{domain}-{i}" for i in range(concurrency)]
                hashes = [prompt_hash(prompts[label]) for label in labels]
                offset = trace.stat().st_size if trace.exists() else 0
                marker.touch(exist_ok=False)
                try:
                    outputs.update(zip(labels, pool.map(generate, labels), strict=True))
                finally:
                    marker.unlink(missing_ok=True)
                events = observed_steps(trace, offset)
                width_by_domain[domain] = {
                    "prompt_sha256": hashes, "prompt_length": prompt_length, "events": events,
                    "overlap": overlap(events, hashes, [prompt_length] * concurrency, native),
                }
        else:
            outputs.update(zip(domains, pool.map(generate, domains), strict=True))
    after = counters(url, required=native)
    return {"client_concurrency": concurrency,
            "prompt_hash": hashlib.sha256(json.dumps(prompts, sort_keys=True).encode()).hexdigest(),
            "output_ids": outputs, "counts": {name: after[name] - before[name] for name in before},
            **({"physical_schedule":width_by_domain} if wide else {})}


def check_width(run, slots, native):
    from check_inkling_draft_overlap import overlap
    schedule = run.get("physical_schedule", {})
    if set(schedule) != {"prose", "code"}:
        raise ValueError("Missing physical model-forward observations for this batch")
    for evidence in schedule.values():
        length = evidence["prompt_length"]
        if len(evidence["prompt_sha256"]) != slots:
            raise ValueError("The observed request count differs from the slot count")
        found = overlap(evidence["events"], evidence["prompt_sha256"], [length] * slots, native)
        key = "joint_five_proposal_steps" if native else "joint_steps"
        if found != evidence["overlap"] or (length == 128 and found[key] < 1):
            raise ValueError("The short calibration never exercised the requested decode width")
    return {k:v["prompt_sha256"] for k,v in schedule.items()}


def compare(plain, draft, forced):
    concurrency = plain.get("client_concurrency", 2)
    if concurrency not in SLOTS or any(run.get("client_concurrency", 2) != concurrency for run in (draft, forced)):
        raise ValueError("The three controls need the same client concurrency")
    for label, run in (("plain", plain), ("DSpark", draft), ("forced rejection", forced)):
        outputs = run["output_ids"]
        if set(outputs) != set(request_labels(concurrency)) or any(len(ids) != 48 for ids in outputs.values()):
            raise ValueError(label + " must contain 48 output token IDs for each domain")
    if concurrency > 2:
        fingerprints = check_width(plain, concurrency, False)
        if any(check_width(run, concurrency, True) != fingerprints for run in (draft, forced)):
            raise ValueError("Physical model-forward prompts differ between the control and verifier")
    if plain["counts"] != {"draft_tokens": 0, "accepted_tokens": 0}:
        raise ValueError("The plain control reported speculative tokens")
    for label, run in (("DSpark", draft), ("forced rejection", forced)):
        if run["prompt_hash"] != plain["prompt_hash"]:
            raise ValueError(label + " received different prompts")
        for domain, expected in plain["output_ids"].items():
            actual = run["output_ids"][domain]
            if actual != expected:
                index = next(i for i, (a, b) in enumerate(zip(actual, expected, strict=True)) if a != b)
                raise ValueError(f"{label}, {domain}, token {index}: got {actual[index]}, expected {expected[index]}")
    proposed, accepted = (draft["counts"][key] for key in ("draft_tokens", "accepted_tokens"))
    if not 0 < accepted < proposed:
        raise ValueError(f"Ordinary DSpark proposed {proposed} and accepted {accepted} tokens")
    if forced["counts"]["draft_tokens"] <= 0 or forced["counts"]["accepted_tokens"] != 0:
        raise ValueError(f"Forced rejection counters: {forced['counts']}")
    result = {"matched_tokens": {d: len(ids) for d, ids in plain["output_ids"].items()},
              "dspark": draft["counts"], "forced_rejection": forced["counts"]}
    if concurrency != 2:
        result["client_concurrency"] = concurrency
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_subparsers(dest="mode", required=True)
    grab = modes.add_parser("capture")
    grab.add_argument("--url", required=True)
    grab.add_argument("--model", required=True)
    grab.add_argument("--bank", required=True, type=Path)
    grab.add_argument("--concurrency", type=int, choices=SLOTS, default=2)
    grab.add_argument("--schedule", type=Path, help="Serving process schedule observer directory for wide batches")
    grab.add_argument("--native", action="store_true", help="Require DSpark counters and five-proposal forwards")
    grab.add_argument("--out", required=True, type=Path)
    verify = modes.add_parser("compare")
    for name in ("plain", "draft", "forced"):
        verify.add_argument("--" + name, required=True, type=Path)
    args = cli.parse_args()
    if args.mode == "capture":
        doc = capture(args.url.rstrip("/"), json.loads(args.bank.read_text()), args.model, args.concurrency, schedule=args.schedule, native=args.native)
        args.out.write_text(json.dumps(doc, indent=2) + "\n")
    else:
        print(json.dumps(compare(*(json.loads(p.read_text()) for p in (args.plain, args.draft, args.forced))), indent=2))
