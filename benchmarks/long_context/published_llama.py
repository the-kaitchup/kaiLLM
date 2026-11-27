"""Verify an upstream llama.cpp plain run made with the published 8K/64K timing client."""

import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re

from benchmark import MANIFEST, OUTPUT, PROMPT, dspark_asset, llama_revision, model_at_revision, verify_prompt_bank
from devices import physical_id
from llama_default import MICROBATCHES, compute_oom, counts, valid_attempts


CLIENT = "fb0fbe6861d1f72fccc2386bfdf56b10804f768b"
FILES = {
    "benchmark.py": "4ec80a88d6f35e0890acc9904e9c868950ad24b112d82ea9e79a5c53a7eb499d",
    "worker.py": "71faf0da75390c2b9634e01059308dd7bb79b0a86044a4453c584719af7eb8a3",
    "manifest.json": "b867707fb58e398ca4f7159f3650b76d610a0738a58d0cd572ec6dca1a355ec6",
}
CORE = {"libllama", "libggml", "libggml-base", "libggml-cpu", "libggml-cuda"}


def require(ok, reason):
    if not ok:
        raise ValueError("Published llama.cpp reference: " + reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def checksum(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def timing_signature(row):
    data = {
        key: row.get(key)
        for key in (
            "protocol",
            "checkpoint",
            "model_revision",
            "sha256",
            "domain",
            "batch",
            "mode",
            "hardware_id",
            "driver",
            "power_limit_w",
            "llama_source",
            "requested_context_per_slot",
            "prompt_set_sha256",
            "input_tokens_per_request",
            "output_tokens_per_request",
            "prefill_seconds",
            "full_seconds",
            "timestamp_utc",
        )
    }
    return sha(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def libraries(certificate, binary, upstream):
    """Read the clean Git revision, CUDA build options and resolved file identities."""
    lines = certificate.splitlines()
    require(
        lines[0] == upstream
        and any(s == "GGML_CUDA:BOOL=ON" for s in lines)
        and any(s == "GGML_CUDA_GRAPHS:BOOL=ON" for s in lines),
        "unverified upstream CUDA build",
    )
    files = {}
    for i, line in enumerate(lines[1:], 1):
        if re.fullmatch(r"(?:GGML_CUDA(?:_GRAPHS)?|CMAKE_CUDA_ARCHITECTURES):\w+=.+", line):
            continue
        if re.fullmatch(r"[0-9a-f]{64}  /.+", line):
            continue
        parts = line.split(maxsplit=3)
        require(len(parts) == 4, "unrecognized Git status or file stat")
        device, inode, size, name = parts
        path = Path(name)
        require(
            path.parent == Path(binary).parent and path.name not in files and int(size) > 0,
            "unexpected binary path",
        )
        digest, hashed_path = lines[i + 1].split(maxsplit=1)
        require(
            checksum(digest) and hashed_path == name,
            "compiled library checksum mismatch",
        )
        dev = int(device, 16)
        files[path.name] = (os.major(dev), os.minor(dev), int(inode), digest)
    require(
        Path(binary).name in files and CORE <= {s.partition(".so")[0] for s in files},
        "incomplete CUDA binary inventory",
    )
    return files


def process(reading, command, uuid, files):
    """Match the actual Linux process, loaded library inodes and physical CUDA owner."""
    reading = reading.split("\nterminal=", 1)[0]
    command_block, rest = reading.split("\npolicy_override_count\n", 1)
    when, pid, ticks, *actual = command_block.splitlines()
    require(
        pid.startswith("pid=") and ticks.startswith("start_ticks=") and actual == command,
        "unexpected running command",
    )
    pid = int(pid.removeprefix("pid="))
    start = int(ticks.removeprefix("start_ticks="))
    require(pid > 0 and start > 0, "invalid process identity")
    count, rest = rest.split("\nlibrary_maps\n", 1)
    mapped, rest = rest.split("\ngpu_processes\n", 1)
    owners, state = rest.split("\nstatus\n", 1)
    require(
        count == "0" and [s.strip() for s in owners.split(",")] == [uuid, str(pid)],
        "CUDA process ownership or policy changed",
    )
    actual_libs = {}
    for line in mapped.splitlines():
        device, inode, name = line.split(maxsplit=2)
        path = Path(name)
        major, minor = map(lambda n: int(n, 16), device.split(":"))
        require(
            path.parent == Path(command[0]).parent and path.name in files,
            "unexpected loaded DSO",
        )
        require(
            files[path.name][:3] == (major, minor, int(inode)),
            "mapped DSO differs from the checked file",
        )
        actual_libs[path.name] = files[path.name][3]
    require(
        CORE <= {s.partition(".so")[0] for s in actual_libs},
        "missing mapped engine or CUDA backend",
    )
    stamp = json.loads(state)
    return int(when), pid, start, actual_libs, stamp


def startup_attempts(folder):
    """Check each larger upstream compute-buffer failure before admitting a smaller microbatch."""
    selected = folder / "selected-ub"
    if not selected.exists():
        return 512, None
    size = int(selected.read_text().strip())
    require(size in MICROBATCHES, "unknown physical microbatch")
    attempts = []
    for previous in MICROBATCHES[: MICROBATCHES.index(size)]:
        raw = (folder / f"startup-ub{previous}.log").read_bytes()
        requested = compute_oom(raw.decode(errors="replace"))
        require(
            requested,
            "upstream failed for a reason other than CUDA compute-buffer exhaustion",
        )
        attempts.append(
            dict(
                microbatch=previous,
                status="compute-buffer-oom",
                requested_bytes=requested,
                log_sha256=sha(raw),
            )
        )
    require((folder / f"startup-ub{size}.log").is_file(), "missing selected server log")
    attempts.append(dict(microbatch=size, status="ready"))
    require(valid_attempts(attempts, size), "unexpected microbatch fallback order")
    return size, attempts


def input_manifest(folder, model_key, upstream):
    """Verify that the published input manifest was refreshed only with pinned public model and build metadata."""
    path = folder / "public-input-manifest.json"
    if not path.exists():
        return None
    raw = (folder / "original-manifest.json").read_bytes()
    require(sha(raw) == FILES["manifest.json"], "modified original published manifest")
    expected = json.loads(raw)
    model = MANIFEST["models"][model_key]
    require(model["family"] == "inkling", "unexpected public input metadata override")
    if model_key not in expected["models"]:
        # The frozen client indexes these public metadata fields to fetch and time a new target.
        expected["models"][model_key] = {k: model[k] for k in ("family", "kind")}
    expected["models"][model_key].update({k: model[k] for k in ("repo", "revision", "files")})
    draft = dspark_asset(model)[1]
    expected["dspark"] = {k: draft[k] for k in ("repo", "revision", "files")}
    expected["llama_cpp"]["revision"] = upstream
    derived = path.read_bytes()
    require(
        json.loads(derived) == expected,
        "an unrelated entry changed in the public input manifest",
    )
    return sha(derived)


def native_draft(folder, row, full):
    """Check the draft artifact, eight greedy output hashes, and measured full-wave proposals."""
    draft = dspark_asset(MANIFEST["models"][row["checkpoint"]])[1]
    wanted = {"source": "huggingface.co", **{key: draft[key] for key in ("repo", "revision", "files")}}
    require(json.loads((folder / "drafter-download.json").read_bytes()) == wanted, "wrong HF drafter")
    controls = []
    for name in ("plain", "dspark5"):
        raw = (folder / f"parity-{name}.jsonl").read_bytes()
        rows = [json.loads(s) for s in raw.splitlines()]
        expected = [(domain, n) for domain in ("prose", "code") for n in (128, 511, 2048, 8192)]
        require(
            [(x[0], x[1]) for x in rows] == expected
            and all(
                len(x) == 5
                and checksum(x[2])
                and type(x[3]) is int
                and type(x[4]) is int
                and 0 <= x[4] <= x[3]
                for x in rows
            ),
            "invalid upstream greedy output checks",
        )
        controls.append(rows)
    plain, native = controls
    require(
        all(x[:2] == y[:2] for x, y in zip(plain, native))
        and sum(x[3] for x in plain) == 0
        and sum(x[3] for x in native) > 0,
        "public DSpark control inputs differ or proposal verification was inactive",
    )
    (digest,) = [v["sha256"] for v in draft["files"].values()]
    return dict(
        measured_drafter_sha256=digest,
        greedy_checked_tokens=8 * 48,
        greedy_matching_outputs=sum(x[2] == y[2] for x, y in zip(plain, native)),
        greedy_controls_sha256=sha(json.dumps(controls, sort_keys=True).encode()),
        speculative=counts(full, "dspark5"),
    )


def native_proof_matches(row, proof):
    if row.get("mode") == "plain":
        return not any(
            key in proof for key in ("speculative", "greedy_checked_tokens", "measured_drafter_sha256")
        )
    if row.get("mode") != "dspark5":
        return False
    try:
        model = model_at_revision(row["checkpoint"], row["model_revision"])
    except (KeyError, ValueError):
        return False
    if model["family"] != "inkling":
        return False
    draft = dspark_asset(model)[1]
    counters = proof.get("speculative") or {}
    values = [counters.get(k) for k in ("proposed", "accepted", "rejected")]
    return (
        proof.get("measured_drafter_sha256") == next(iter(draft["files"].values()))["sha256"]
        and row.get("drafter") == {k: draft[k] for k in ("repo", "revision")}
        and checksum(proof.get("greedy_controls_sha256"))
        and proof.get("greedy_checked_tokens") == 384
        and type(proof.get("greedy_matching_outputs")) is int
        and 0 <= proof["greedy_matching_outputs"] <= 8
        and counters.get("scope") == "full-wave"
        and all(type(v) is int and v >= 0 for v in values)
        and values[0] > 0
        and values[1] + values[2] == values[0]
    )


def verified(row, proof):
    """Check the shareable receipt against its timing row without exposing process or hardware IDs."""
    if not isinstance(proof, dict) or proof.get("published_client") != CLIENT:
        return False
    libraries = proof.get("mapped_dso_sha256")
    model = MANIFEST["models"].get(row.get("checkpoint"))
    if model and model.get("llama_revision_override") and proof.get("upstream") != llama_revision(model):
        return False
    return (
        proof.get("policy") == "default"
        and proof.get("upstream") == row.get("llama_source")
        and row.get("kai_source") == CLIENT
        and row.get("engine") == "llama.cpp"
        and native_proof_matches(row, proof)
        and proof.get("measured_gguf_sha256") == row.get("sha256")
        and bool(row.get("sha256"))
        and proof.get("hardware_id") == row.get("hardware_id")
        and proof.get("device_count") == len(row.get("gpu", [])) == 1
        and proof.get("domain") == row.get("domain")
        and proof.get("batch") == row.get("batch")
        and proof.get("completed_utc") == row.get("timestamp_utc")
        and (
            (proof.get("startup_attempts") is None and proof.get("microbatch") == 512)
            or (
                checksum(proof.get("input_manifest_sha256"))
                and valid_attempts(proof.get("startup_attempts"), proof.get("microbatch"))
            )
        )
        and proof.get("timing_signature") == timing_signature(row)
        and proof.get("observer_configured") is False
        and proof.get("environment_overrides") == []
        and type(proof.get("full_wave_samples")) is int
        and proof["full_wave_samples"] > 0
        and all(
            checksum(proof.get(k))
            for k in (
                "executable_sha256",
                "timing_receipt_sha256",
                "process_samples_sha256",
                "client_receipt_sha256",
                "build_receipt_sha256",
            )
        )
        and isinstance(libraries, dict)
        and all(checksum(v) for v in libraries.values())
        and CORE <= {s.partition(".so")[0] for s in libraries}
        and all(
            k not in row
            for k in (
                "cuda_graphs",
                "target_head_cuda_graphs",
                "drafter_cuda_graphs",
                "llama_graph_launches_by_wave",
            )
        )
    )


def records(folder, compact):
    from llama_default import validate_command

    def read(name):
        return json.loads((folder / name).read_bytes())

    require((folder / "exit").read_text().strip() == "0", "incomplete public worker")
    client = read("public-client.json")
    build = read("public-build.json")
    download = read("download.json")
    require(
        client == {"repo": "the-kaitchup/kaiLLM", "revision": CLIENT, "files": FILES},
        "incorrect published client files",
    )
    bank = read("prompt-bank.json")
    (gpu,) = csv.reader((folder / "physical-gpu.csv").read_text().splitlines(), skipinitialspace=True)
    readings = (folder / "process-samples.jsonl").read_bytes()
    results = []
    identities = set()
    size, attempts = startup_attempts(folder)
    for domain in ("prose", "code"):
        (path,) = folder.glob("*-" + domain + ".json")
        raw = path.read_bytes()
        timing = json.loads(raw)
        row = compact(timing, published_microbatch=size)
        model = MANIFEST["models"][row["checkpoint"]]
        upstream = llama_revision(model)
        input_sha = input_manifest(folder, row["checkpoint"], upstream)
        require(
            model["family"] != "inkling" or (input_sha and attempts),
            "missing current public Inkling inputs or startup history",
        )
        (fn,) = model["files"]
        require(
            download
            == {
                "source": "huggingface.co",
                **{k: model[k] for k in ("repo", "revision", "files")},
            },
            "unverified Hub download",
        )
        verify_prompt_bank(bank, model["family"], domain)
        require(
            timing["domain"] == domain
            and timing["prompt_hashes"] == bank["domains"][domain]["hashes"][: row["batch"]],
            "wrong token windows",
        )
        (dev,) = timing["environment"]["devices"]
        uuid = dev["uuid"]
        require(
            (
                dev["name"],
                uuid,
                int(dev["memory.total"]),
                dev["driver_version"],
                float(dev["power.limit"]),
            )
            == (gpu[0], gpu[1], int(gpu[2]), gpu[3], float(gpu[4])),
            "different physical board or hardware settings",
        )
        require(
            row["llama_source"] == upstream == build["upstream"] and row["kai_source"] == CLIENT,
            "unexpected source revision",
        )
        command = timing["environment"]["server_argv"]
        binary = command[0]
        require(
            str(binary).endswith("/" + upstream + "/build/bin/llama-server")
            and command[2].endswith("/" + model["repo"] + "/" + model["revision"] + "/" + fn),
            "wrong binary or checkpoint path",
        )
        validate_command(command, row["mode"], row["requested_context_per_slot"], size)
        draft_args = ["none"]
        if row["mode"] == "dspark5":
            draft = dspark_asset(model)[1]
            directory = Path(command[2]).parents[len(Path(model["repo"]).parts) + 1]
            (filename,) = draft["files"]
            draft_file = directory / draft["repo"] / draft["revision"] / filename
            draft_args = ["draft-dspark", "--model-draft", str(draft_file), "--spec-draft-n-max", "5"]
        require(
            command[1] == "-m"
            and command[3:6] == ["--host", "127.0.0.1", "--port"]
            and command[7:]
            == [
                "--ctx-size",
                str((PROMPT + OUTPUT + (6 if row["mode"] == "dspark5" else 0)) * row["batch"]),
                "--parallel",
                str(row["batch"]),
                "-ngl",
                "999",
                "--fit",
                "off",
                "--flash-attn",
                "on",
                "--cache-type-k",
                "bf16",
                "--cache-type-v",
                "bf16",
                "--cache-ram",
                "0",
                "--no-context-shift",
                "-b",
                "2048",
                "-ub",
                str(size),
                "--spec-type",
                *draft_args,
            ],
            "nonstandard stock workload or policy override",
        )
        files = libraries((folder / "public-binaries.txt").read_text(), binary, upstream)
        require(
            files["llama-server"][3] == build["public_executable_sha256"],
            "binary changed after public build",
        )
        end = datetime.fromisoformat(row["timestamp_utc"]).timestamp()
        start = end + 1 - timing["full"]["seconds"]
        observed = 0
        loaded = None
        for line in readings.splitlines():
            sample = json.loads(line)
            when, pid, ticks, dsos, stamp = process(sample["reading"], command, uuid, files)
            identities.add((pid, ticks, tuple(sorted(dsos.items()))))
            if stamp.get("stage") == domain and stamp.get("state") == "running" and start <= when <= end:
                observed += 1
                loaded = dsos
        require(
            observed > 0 and len(identities) == 1,
            "missing full-wave process sample or restarted server",
        )
        proof = dict(
            policy="default",
            upstream=upstream,
            published_client=CLIENT,
            device_count=1,
            hardware_id=physical_id([uuid]),
            domain=domain,
            batch=row["batch"],
            completed_utc=row["timestamp_utc"],
            microbatch=size,
            measured_gguf_sha256=timing["sha256"],
            environment_overrides=[],
            observer_configured=False,
            executable_sha256=files["llama-server"][3],
            mapped_dso_sha256=loaded,
            timing_receipt_sha256=sha(raw),
            timing_signature=timing_signature(row),
            process_samples_sha256=sha(readings),
            full_wave_samples=observed,
            client_receipt_sha256=sha((folder / "public-client.json").read_bytes()),
            build_receipt_sha256=sha((folder / "public-build.json").read_bytes()),
        )
        if row["mode"] == "dspark5":
            proof.update(native_draft(folder, row, timing["full"]))
        if input_sha:
            proof.update(input_manifest_sha256=input_sha, startup_attempts=attempts)
        require(verified(row, proof), "invalid shareable reference evidence")
        row["llama_default"] = proof
        results.append(row)
    return results
