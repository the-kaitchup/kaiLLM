"""Ingest separately reported DSpark timings with a prompt-length-specific accuracy check."""
import hashlib
import json
import math
import re
import statistics

from benchmark import OUTPUT, PROMPT

LABEL = "DSpark5 throughput · 8,192-token prompt checked"
# Exact SHA-256 of the four-way numerical scope receipt, including hardware, sources,
# target/drafter hashes, 192 checked token IDs and the three shorter prose failures.
SCOPE_SHA256 = "cf31d0860383a804f8b7170d5e1b242da93c923ca05eadb179bd250b9b881081"


# Each additional placement needs a reviewed same-GPU accuracy receipt.
PACKED_EXPERT_SCOPE = "de4191a4a847653827da7602bcd5062f772a11ad9646cef89dfce5cd2ab7ce95"
ADDITIONAL_SCOPES = {PACKED_EXPERT_SCOPE: "Three shorter prose cases diverged."}
EXPERT_FLAGS = ("KAILLM_INKLING_5070_NATIVE_PREFILL", "KAILLM_INKLING_5070_NATIVE_VERIFY_ROWS")


def scope_note(fingerprint):
    if fingerprint == SCOPE_SHA256:
        return "Three shorter prose cases diverged."
    return ADDITIONAL_SCOPES.get(fingerprint)


def require(condition, message):
    if not condition:
        raise ValueError("Prefix-checked DSpark throughput: " + message)


def check_scope(raw, compact):
    environment = raw["environment"]
    scope = environment["numerical_scope"]
    serialized = (json.dumps(scope, indent=2, ensure_ascii=False) + "\n").encode()
    fingerprint = hashlib.sha256(serialized).hexdigest()
    require(scope_note(fingerprint), "unrecognized numerical scope")
    device, = environment["devices"]
    require(compact["engine"] == "kaiLLM" and compact["mode"] == "dspark5" and compact["batch"] == 1
            and compact["checkpoint"] == "inkling-udq2" and compact["domain"] in ("prose", "code")
            and compact["requested_context_per_slot"] == PROMPT + OUTPUT
            and compact.get("kai_humming_deterministic") is True
            and scope["gpu_uuid"] == device["uuid"] and environment["gpu_names"] == [device["name"]]
            and scope["runtime"] == {"revision": compact["kai_source"], "source_lock": compact["engine_revision"]}
            and environment.get("measured_gguf_sha256") == compact["sha256"] == [scope["target_sha256"]]
            and environment.get("measured_drafter_sha256") == [compact["drafter"]["sha256"]] == [scope["drafter_sha256"]],
            "source, hardware, checkpoint or drafter differs from numerical controls")
    counts = raw.get("kai_dspark") or {}
    proposed, accepted, rejected = (counts.get(k) for k in ("proposed", "accepted", "rejected"))
    require(counts.get("scope") == "full-wave" and all(type(n) is int for n in (proposed, accepted, rejected))
            and proposed > 0 and min(accepted, rejected) >= 0 and proposed == accepted + rejected,
            "missing full-generation proposal counts")
    for wave in [*raw["prefill"], raw["full"]]:
        require(wave["requests"][0]["cache_hits"] == 0 and math.isfinite(wave["seconds"]) and wave["seconds"] > 0,
                "incomplete wall time or cache evidence")
    require(raw["full"]["seconds"] > statistics.median(compact["prefill_seconds"]), "invalid decode interval")
    require(re.fullmatch(r"[0-9a-f]{64}", compact["generated_text_hashes"][0]), "missing full-output digest")
    return fingerprint


def preemptions(path):
    samples = re.findall(r"^vllm:num_preemptions_total(\{[^}]*\})?\s+([0-9.eE+-]+)$", path.read_text(), re.M)
    counts = {labels: float(value) for labels, value in samples}
    require(counts and len(counts) == len(samples) and all(math.isfinite(v) and v >= 0 for v in counts.values()),
            "missing or invalid preemption counters")
    return counts


def packed_tile_map(kind):
    prefill = [8, 64, 16, 4] if kind in (16, 22) else [8, 32, 32, 4]
    return {**{str(i): [1, 16, 64, 4] for i in range(1, 7)}, "2048": prefill}


def packed_expert_evidence(path, row, gpu, expected_hash):
    data = path.read_bytes()
    proof = json.loads(data)
    require(proof.get("source") == row["kai_source"] and proof.get("source_lock") == row["engine_revision"]
            and proof.get("physical_gpu") == gpu and proof.get("serving_task_descendant") is True
            and proof.get("active_cuda_process") is True and proof.get("selector_environment") == dict.fromkeys(EXPERT_FLAGS, "1"),
            "active GPU process did not attest both packed expert gates")
    digest = hashlib.sha256(data).hexdigest()
    require(digest == expected_hash, "timed run and resident CUDA process proofs differ")
    expected_operator = "c8e509a6bde9c9035b65a3f56e4ef9ea9406b23837ccc90fef6d724bd4a81aae"
    kinds = (16, 17, 19, 22)
    require(proof.get("selected_tiles") == {str(k): packed_tile_map(k) for k in kinds}
            and proof.get("operator_source_sha256") == expected_operator,
            "timed model did not select the measured tiles for all four expert types")
    hashes = proof.get("stage_receipts_sha256", {})
    require(set(hashes) == {f"stage-{k}.json" for k in kinds}, "missing resident expert tensors")
    for kind in kinds:
        name = f"stage-{kind}.json"
        tensor = (path.parent / "selector-witness" / name).read_bytes()
        require(hashlib.sha256(tensor).hexdigest() == hashes[name], "resident tensor receipt changed")
        stage = json.loads(tensor)
        require(stage.get("kind") == kind and stage.get("selected") == packed_tile_map(kind)
                and stage.get("flags") == dict.fromkeys(EXPERT_FLAGS, "1")
                and stage.get("source_file_sha256") == expected_operator
                and stage.get("method") == "dflash" and stage.get("k") == 5
                and stage.get("draft_architectures") == ["DFlashInklingDSparkDraftModel"],
                "resident expert-stage selection disagrees with the launch")
    captures = proof.get("capture_receipts_sha256", {})
    require(set(captures) == {f"capture-{kind}-{width}.json" for kind in kinds for width in (6, 2048)},
            "missing CUDA graph capture for verifier or prefill")
    for kind in kinds:
        for width in (6, 2048):
            name = f"capture-{kind}-{width}.json"
            raw = (path.parent / "selector-witness" / name).read_bytes()
            capture = json.loads(raw)
            require(hashlib.sha256(raw).hexdigest() == captures[name] and capture.get("capturing") is True
                    and capture.get("kind") == kind and capture.get("rows") == width
                    and capture.get("tile") == packed_tile_map(kind)[str(width)],
                    "packed expert kernel did not execute while recording the target graph")
    return digest


def read_run(folder, get_record, rates):
    result = []
    for domain in ("prose", "code"):
        path = folder / (domain + ".json")
        if not path.is_file():
            continue
        data = path.read_bytes()
        raw = json.loads(data)
        row = get_record(raw)
        require(scope_note(row.get("numerical_scope_sha256")) and row["domain"] == domain,
                "missing control receipt or incorrect domain")
        require(preemptions(folder / f"metrics-before-{domain}.txt") == preemptions(folder / f"metrics-after-{domain}.txt"),
                "preemption during timed wave")
        pp, tg = rates(row)
        require(tg > 0 and math.isclose(pp, raw["prefill_tps"], rel_tol=1e-10)
                and math.isclose(tg, raw["decode_tps"], rel_tol=1e-10), "reported rates differ from wall time")
        if row["numerical_scope_sha256"] == PACKED_EXPERT_SCOPE:
            row["packed_expert_proof_sha256"] = packed_expert_evidence(
                folder / "selector-environment.json", row, raw["environment"]["devices"][0]["uuid"],
                raw["environment"].get("packed_expert_proof_sha256"))
        row["raw_sha256"] = hashlib.sha256(data).hexdigest()
        result.append(row)
    require(result, "no completed domain")
    return result
