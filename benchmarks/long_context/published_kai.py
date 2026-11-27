"""Import full-length plain timings from the unmodified public kaiLLM client and release."""

import csv
from datetime import datetime
import json
from pathlib import Path
import re

from benchmark import MANIFEST, OUTPUT, PROMPT, llama_revision, verify_prompt_bank
from devices import physical_id
from published_llama import CLIENT, FILES, checksum, input_manifest, sha, timing_signature


SOURCE = "e227679a7b8ad99e4b39f36831e4c2c9f2e5682a52a7da5d574de3d029cdd150"
RELEASE = "84cfbe974004c0822bc905c970357d742693a40beba6f0ea046cc414a8400875"
MODELS = {"lfm-qad", "lfm-suisui", "inkling-udq2"}


def require(condition, reason):
    if not condition:
        raise ValueError("Published kaiLLM benchmark: " + reason)


def verified(row):
    proof = row.get("published_kai") or {}
    return (
        row.get("engine") == "kaiLLM"
        and row.get("mode") == "plain"
        and row.get("checkpoint") in MODELS
        and (row["checkpoint"] != "inkling-udq2" or checksum(proof.get("input_manifest_sha256")))
        and row.get("kai_source") == CLIENT
        and row.get("engine_revision") == SOURCE
        and row.get("kai_enforce_eager") is not True
        and len(row.get("gpu", [])) == 1
        and proof.get("source_lock") == SOURCE
        and proof.get("release_sha256") == RELEASE
        and proof.get("hardware_id") == row.get("hardware_id")
        and proof.get("measured_gguf_sha256") == row.get("sha256")
        and proof.get("timing_signature") == timing_signature(row)
        and proof.get("graph_capture_logged") is True
        and proof.get("phase_replay_counts") is None
        and proof.get("preemptions") == 0
        and type(proof.get("full_wave_samples")) is int
        and proof["full_wave_samples"] >= 2
        and proof.get("max_active_requests") == row.get("batch")
        and all(
            checksum(proof.get(k))
            for k in (
                "receipt_sha256",
                "process_samples_sha256",
                "launch_preview_sha256",
                "server_log_sha256",
            )
        )
        and not any(
            k in row
            for k in ("cuda_graphs", "kai_default_dispatch", "target_head_direct", "target_head_cuda_graphs")
        )
    )


def control_metrics(text):
    totals = {}
    needs = ("num_preemptions_total", "generation_tokens_total", "num_requests_running")
    for line in text.splitlines():
        matched = re.fullmatch(r"vllm:(\w+)(?:\{([^}]*)\})?\s+([\d.eE+\-]+)", line.strip())
        if matched and matched[1] in needs:
            totals[matched[1]] = totals.get(matched[1], 0.0) + float(matched[3])
    require(all(k in totals and totals[k] >= 0 for k in needs), "missing scheduler metrics")
    return totals


def full_samples(samples, domain, begin, end, uuid, batch):
    selected = []
    identities = set()
    for item in samples:
        reading = item["reading"]
        header, body = reading.split("\ndevice\n", 1)
        when, state = header.split("\n", 1)
        when = int(when)
        state = json.loads(state)
        if not (begin <= when <= end and state.get("stage") == domain and state.get("state") == "running"):
            continue
        _, body = body.split("\ngpu_processes\n", 1)
        drivers, body = body.split("\nowned_by_public_worker\n", 1)
        owned, metrics = body.split("\nmetrics\n", 1)
        devices = list(csv.reader(drivers.splitlines(), skipinitialspace=True))
        require(devices and all(d[0] == uuid for d in devices), "unexpected GPU process owner")
        tree = [tuple(map(int, line.split())) for line in owned.splitlines()]
        require(
            all(len(row) == 3 and min(row) > 0 for row in tree)
            and {x[0] for x in tree} == {int(d[1]) for d in devices},
            "a concurrent GPU process is outside the measured public worker",
        )
        identities.add(tuple(sorted(tree)))
        counters = control_metrics(metrics)
        require(counters["num_preemptions_total"] == 0, "GPU scheduler preempted a request")
        selected.append((when, counters))
    selected.sort(key=lambda x: x[0])
    require(
        len(selected) >= 2 and len(identities) == 1, "missing continuous full-generation process evidence"
    )
    generated = [c["generation_tokens_total"] for _, c in selected]
    active = max(c["num_requests_running"] for _, c in selected)
    require(
        all(a <= b for a, b in zip(generated, generated[1:]))
        and generated[-1] > generated[0]
        and active == batch,
        "generation stalled, restarted or did not run the requested batch",
    )
    return len(selected), identities.pop()


def records(folder, compact):
    def read(name):
        return json.loads((folder / name).read_bytes())

    require((folder / "exit").read_text().strip() == "0", "worker did not complete")
    require(
        (folder / "source-head").read_text().strip() == CLIENT
        and sha((folder / "runtime-source-lock.json").read_bytes()) == SOURCE,
        "source checkout or runtime differs from the published engine",
    )
    require(
        read("public-client.json") == {"repo": "the-kaitchup/kaiLLM", "revision": CLIENT, "files": FILES},
        "public client hashes differ",
    )
    release = read("public-release.json")
    require(release.get("version") == "0.1.0rc6" and release.get("sha256") == RELEASE, "unknown release")
    (gpu,) = csv.reader((folder / "physical-gpu.csv").read_text().splitlines(), skipinitialspace=True)
    samples_raw = (folder / "process-samples.jsonl").read_bytes()
    samples = [json.loads(l) for l in samples_raw.splitlines()]
    preview_raw = (folder / "launch-preview.json").read_bytes()
    preview = json.loads(preview_raw)
    require(
        preview.get("runtime_started") is False
        and isinstance(preview.get("command"), list)
        and not any(s == "--enforce-eager" or s == "--enforce-eager=True" for s in preview["command"]),
        "public graph policy differs",
    )
    (log,) = [path for path in folder.glob("*.log") if path.name != "launcher.log"]
    log_bytes = log.read_bytes()
    require(
        re.search(rb"Graph capturing finished in [\d.]+ secs", log_bytes),
        "server did not record completing CUDA graph capture",
    )
    result = []
    worker_ids = set()
    for domain in ("prose", "code"):
        (rawfile,) = folder.glob("*-" + domain + ".json")
        data = rawfile.read_bytes()
        timing = json.loads(data)
        row = compact(timing)
        key = row["checkpoint"]
        require(key in MODELS and row["engine"] == "kaiLLM" and row["mode"] == "plain", "unexpected workload")
        model = MANIFEST["models"][key]
        download = read("download.json")
        require(
            download == {"source": "huggingface.co", **{k: model[k] for k in ("repo", "revision", "files")}},
            "public GGUF revision differs",
        )
        bank = read("prompt-bank.json")
        verify_prompt_bank(bank, model["family"], domain)
        require(
            timing["prompt_hashes"] == bank["domains"][domain]["hashes"][: row["batch"]],
            "unexpected prompt windows",
        )
        (device,) = timing["environment"]["devices"]
        uuid = device["uuid"]
        require(
            (
                device["name"],
                uuid,
                int(device["memory.total"]),
                device["driver_version"],
                float(device["power.limit"]),
            )
            == (gpu[0], gpu[1], int(gpu[2]), gpu[3], float(gpu[4])),
            "different physical device, driver or board power",
        )
        argv = timing["environment"]["server_argv"]
        (filename,) = model["files"]
        require(
            argv[1].endswith("/runtime/deployment/serve.py")
            and argv[2].endswith("/" + model["repo"] + "/" + model["revision"] + "/" + filename),
            "unverified model or runtime",
        )
        require(
            argv[3:]
            == [
                "--max-num-seqs",
                str(row["batch"]),
                "--max-model-len",
                str(PROMPT + OUTPUT),
                "--kv-cache-memory-bytes",
                str(row["kai_kv"]["bytes_per_device"]),
                "--gpu-memory-utilization",
                "0.94",
                "--max-num-batched-tokens",
                "2048",
                "--port",
                "18951",
                "--served-model-name",
                "bench",
                "--spec-tokens",
                "0",
                "--extra-vllm-args",
                json.dumps(["--enable-prompt-tokens-details", "--enable-per-request-metrics"]),
            ],
            "nonstandard released benchmark command",
        )
        end = datetime.fromisoformat(row["timestamp_utc"]).timestamp()
        n, identity = full_samples(
            samples, domain, end + 1 - timing["full"]["seconds"], end, uuid, row["batch"]
        )
        worker_ids.add(identity)
        proof = dict(
            source_lock=SOURCE,
            release_sha256=RELEASE,
            hardware_id=physical_id([uuid]),
            measured_gguf_sha256=row["sha256"],
            timing_signature=timing_signature(row),
            graph_capture_logged=True,
            phase_replay_counts=None,
            preemptions=0,
            full_wave_samples=n,
            max_active_requests=row["batch"],
            receipt_sha256=sha(data),
            process_samples_sha256=sha(samples_raw),
            launch_preview_sha256=sha(preview_raw),
            server_log_sha256=sha(log_bytes),
        )
        if model["family"] == "inkling":
            require(preview["environment"].get("KAILLM_HUMMING_DETERMINISTIC", "0") == "0"
                    and not row.get("kai_humming_deterministic"), "unexpected public Inkling reduction flag")
            proof["input_manifest_sha256"] = input_manifest(folder, key, llama_revision(model))
        row["published_kai"] = proof
        require(verified(row), "invalid released runtime or timing signature")
        result.append(row)
    require(len(worker_ids) == 1, "server changed between domains")
    return result
