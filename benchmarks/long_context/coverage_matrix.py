"""Count full-length benchmark cells and deduplicate retries from the timing archive."""
from __future__ import annotations

import json
import kai_default
import published_kai
import empirical_capacity
import stock_kv_floor
from collections import Counter
from pathlib import Path

from benchmark import MANIFEST, current_limit_build, current_weights, full_graph_runtime, kai_loader_runtime, timing_domains
from cuda_graphs import gpu_draft_mode
from matrix import DGX_SPARK, cells
from historical_timings import previously_accepted
from llama_default import verified as default_llama_verified

TARGETS = ("3090", "5070 Ti", "5080", "5090", "PRO 6000", DGX_SPARK)
ARCHIVED_GPUS = ("3060",)


# Physical capacity of the cards in this campaign.
VRAM_GIB = {"3090": 24, "5070 Ti": 16, "5080": 16, "5090": 32, "PRO 6000": 96}


PROVISIONAL = {(r["hardware_id"], r["timestamp_utc"]): r["reason"]
               for r in MANIFEST.get("provisional_timings", ())}

INACTIVE_NGRAM = "The pinned llama.cpp ngram-simple implementation returned an empty draft"
NO_NGRAM_PROPOSALS = "llama.cpp recorded zero proposed tokens in the full generation"
MISSING_NGRAM_COUNTS = "llama.cpp full-generation proposal counts are missing"


_OBSERVATIONS = json.loads(Path(__file__).with_name("historical_graph_observations.json").read_text())
HISTORICAL_GRAPH_MODE = {(r["hardware_id"], r["engine"], r["engine_revision"], r["checkpoint"], r["mode"], r["batch"], stamp): r["observed"]
                         for r in _OBSERVATIONS for stamp in r["timestamps_utc"]}


def historical_graph_mode(row):
    return HISTORICAL_GRAPH_MODE.get(tuple(row.get(k) for k in (
        "hardware_id", "engine", "engine_revision", "checkpoint", "mode", "batch", "timestamp_utc")))


DRAFT_FIELDS = ("proposals", "completed_proposals", "graph_replays")
GRAPH_WAVES = {"prefill": ("prefill-1", "prefill-2", "prefill-3", "full"), "decode": ("full",)}


def graph_phase_verified(row, phase):
    if 'target_head_direct' in row or 'llama_default' in row:
        return False
    observed = historical_graph_mode(row)
    if row.get("kai_enforce_eager") is True or observed == "eager" or (phase == "prefill" and observed == "decode_only"):
        return False
    gpu_count = len(row["gpu"])
    if not phase_replays_verified(row.get("cuda_graphs"), phase, gpu_count):
        return False
    if row.get("engine") != "kaiLLM":
        return True
    proof = row.get("target_head_cuda_graphs")
    if not isinstance(proof, dict):
        return False
    stages = proof.get("pipeline_stages")
    if type(stages) is not int or stages < 1 or gpu_count % stages:
        return False
    head_count = gpu_count // stages
    expected = list(range(gpu_count - head_count, gpu_count))
    ranks = proof.get("device_ranks")
    if ranks != expected or any(type(i) is not int for i in ranks):
        return False
    return phase_replays_verified(proof, phase, head_count)


def phase_replays_verified(evidence, phase, devices):
    wave = evidence.get(phase) if isinstance(evidence, dict) else None
    if not isinstance(wave, dict) or wave.get("scope") != "timed-wave":
        return False
    replay = wave.get("graph_replays")
    if (type(replay) is not int or replay <= 0
            or any(type(wave.get(key)) is not int or wave[key] != 0 for key in ("eager_steps", "unclassified_steps"))):
        return False
    ranks = wave.get("per_wave_by_rank")
    if not isinstance(ranks, dict) or set(ranks) != set(GRAPH_WAVES[phase]):
        return False
    return (devices > 0 and all(isinstance(values, list) and len(values) == devices
                                and all(type(n) is int and n > 0 for n in values) for values in ranks.values())
            and sum(sum(values) for values in ranks.values()) == replay)


def draft_graphs_verified(row):
    proof = row.get("drafter_cuda_graphs")
    if not isinstance(proof, dict) or proof.get("scope") != "timed-wave":
        return False
    if any(type(proof.get(k)) is not int or proof[k] != 0 for k in ("eager_steps", "unclassified_steps")):
        return False
    ranks = proof.get("device_ranks")
    if (not isinstance(ranks, list) or not ranks or any(type(i) is not int or not 0 <= i < len(row["gpu"]) for i in ranks)
            or sorted(set(ranks)) != ranks):
        return False
    waves = proof.get("per_wave_by_rank")
    if not isinstance(waves, dict) or set(waves) != set(GRAPH_WAVES["prefill"]):
        return False
    totals = Counter()
    for label, values in waves.items():
        if not isinstance(values, list) or len(values) != len(ranks):
            return False
        for observed in values:
            if not isinstance(observed, dict) or any(type(observed.get(k)) is not int or observed[k] < 0 for k in DRAFT_FIELDS):
                return False
            proposals, completed, replays = (observed[k] for k in DRAFT_FIELDS)
            if row.get("engine") == "kaiLLM" and row.get("mode") in ("ngram1", "ngram3") and replays != proposals:
                return False
            if completed != proposals or replays < proposals or (proposals == 0 and (replays or label == "full")):
                return False
            totals.update({key: observed[key] for key in DRAFT_FIELDS})
    return all(type(proof.get(key)) is int and proof[key] == totals[key] for key in DRAFT_FIELDS)


def graphs_verified(row):
    mode = row.get("mode")
    if row.get("engine") == "kaiLLM" and mode in ("ngram1", "ngram3"):
        counts = row.get("kai_ngram") or {}
        proposed, accepted, rejected = (counts.get(k) for k in ("proposed", "accepted", "rejected"))
        requested = 3 if mode == "ngram3" else 1
        if (counts.get("lookup_min") != 2 or counts.get("lookup_max") != 8 or counts.get("draft_tokens") != requested
                or counts.get("scope") != "full-wave" or any(type(n) is not int for n in (proposed, accepted, rejected))
                or proposed <= 0 or accepted < 0 or rejected < 0 or accepted + rejected != proposed):
            return False
    return (all(graph_phase_verified(row, phase) for phase in GRAPH_WAVES)
            and (not gpu_draft_mode(row.get("engine"), mode) or draft_graphs_verified(row)))


def direct_head_verified(row):
    proof = row.get('target_head_direct')
    if (not isinstance(proof, dict) or row.get('engine') != 'kaiLLM' or row.get('mode') != 'plain'
            or row.get('kai_enforce_eager') or 'target_head_cuda_graphs' in row or 'drafter_cuda_graphs' in row
            or row.get('measured_gguf_sha256') != row.get('sha256') or not row.get('sha256')
            or row.get('kai_humming_deterministic') is not True):
        return False
    pin = full_graph_runtime(row['checkpoint'], 'plain', row.get('kai_source'), output_head='direct')
    if pin != (row.get('kai_source'), row.get('engine_revision')):
        return False
    stages = proof.get('pipeline_stages')
    count = len(row['gpu'])
    if type(stages) is not int or stages < 1 or not count or count % stages:
        return False
    ranks = proof.get('device_ranks')
    if (ranks != list(range(count-count//stages, count)) or any(type(n) is not int for n in ranks)):
        return False
    backbone = row.get('cuda_graphs')
    for phase, labels in GRAPH_WAVES.items():
        direct = proof.get(phase)
        if (not phase_replays_verified(backbone, phase, count) or not isinstance(direct, dict)
                or direct.get('execution') != 'direct' or direct.get('scope') != 'timed-wave'):
            return False
        calls = direct.get('per_wave_by_rank')
        if not isinstance(calls, dict) or set(calls) != set(labels):
            return False
        for label in labels:
            expected = [backbone[phase]['per_wave_by_rank'][label][rank] for rank in ranks]
            if calls[label] != expected or any(type(n) is not int for n in calls[label]):
                return False
        if type(direct.get('completed_calls')) is not int or direct['completed_calls'] != sum(sum(x) for x in calls.values()):
            return False
    return True


def timing_accepted(row):
    if "numerical_scope_sha256" in row:
        return False
    if ('kai_empirical_capacity' in row or row.get('engine') == 'kaiLLM') and not empirical_capacity.capacity_valid(row):
        return False
    if provisional_reason(row):
        return False
    if 'published_kai' in row:
        return published_kai.verified(row)
    if 'kai_default_dispatch' in row:
        return kai_default.verified(row, MANIFEST)
    if 'llama_default' in row:
        return default_llama_verified(row)
    if 'target_head_direct' in row:
        return direct_head_verified(row)
    return previously_accepted(row) or graphs_verified(row)


def provisional_reason(row):
    if row.get("engine") == "llama.cpp" and row.get("mode") == "ngram1":
        setup = row.get("llama_ngram")
        # Early B8 receipts used ngram-simple, N=2, M=1. This upstream build
        # returns an empty draft whenever the available continuation is shorter than N.
        if setup is None or (setup["method"] == "ngram-simple" and setup["draft_tokens"] < setup["lookup_tokens"]):
            return INACTIVE_NGRAM
        if "proposed" not in setup:
            return MISSING_NGRAM_COUNTS
    if row.get("llama_ngram", {}).get("proposed") == 0:
        return NO_NGRAM_PROPOSALS
    return PROVISIONAL.get((row.get("hardware_id"), row.get("timestamp_utc")))


def loader_exclusion(key, engine):
    model = MANIFEST["models"][key]
    if engine == "kaiLLM" and (full_graph_runtime(key, "plain") or kai_loader_runtime(model)):
        return None
    if engine == "kaiLLM" and model.get("gdn_layout") == "fused_qkv_gate_beta_alpha":
        return "Fused DaisyRose donor layout"
    if engine == "llama.cpp" and model["kind"] == "chimachima":
        return "No kaiPTQ tensor loader"
    return None


def known_exclusion(gpu, count, key, engine, batch=None):
    loader = loader_exclusion(key, engine)
    if loader:
        return loader
    if gpu in VRAM_GIB and MANIFEST["models"][key].get("transformer_block_bytes", 0) > count * VRAM_GIB[gpu] * 2**30:
        return "Transformer blocks exceed GPU memory"
    if engine == "llama.cpp" and batch is not None and count == 1 and gpu in VRAM_GIB:
        required = stock_kv_floor.minimum_bytes(key, batch)
        if required is not None and required > VRAM_GIB[gpu] * 2**30:
            return "Minimum llama.cpp GPU allocation exceeds VRAM"
    return None


def device_group(names):
    if not names:
        raise ValueError("Missing GPU inventory")
    groups = {label for label in (*TARGETS, *ARCHIVED_GPUS) if all(label.upper() in name.upper() for name in names)}
    if len(groups) != 1:
        raise ValueError(f"Unrecognized GPU inventory: {names}")
    return groups.pop(), len(names)


def active_rows(rows):
    """Keep only the active GPU families and current GGUF bytes in reported coverage."""
    return [row for row in rows if device_group(row["gpu"])[0] in TARGETS and current_weights(row)]


def read_assignments(archive):
    path = archive.with_name("multi_gpu_cells.json")
    return json.loads(path.read_text()) if path.exists() else []


def minimum_gpu_assignments():
    assignments = set()
    for gpu in ("3090", "5080", "5090"):
        capacity = VRAM_GIB[gpu] * 2**30
        for key, model in MANIFEST["models"].items():
            blocks = model.get("transformer_block_bytes", 0)
            minimum = next((n for n in (1, 2, 4) if blocks <= n * capacity), None)
            if minimum in (2, 4):
                assignments.update((gpu, minimum, *row) for row in cells([key]))
    return assignments


def group_counts(planned, excluded, done, group):
    totals, unavailable, filled = (Counter(map(group, cells)) for cells in (planned, excluded, done))
    return {key: {"filled": filled[key], "unavailable": unavailable[key],
                  "remaining": total - unavailable[key] - filled[key]} for key, total in totals.items()}


def observation_board(row):
    """A measured observation can remain visible on one board while another completes the cell."""
    return row.get("hardware_id", "") if row.get("observation_only") else ""


def active_limits(records, limits):
    def configuration(row):
        return (*device_group(row['gpu']), row['checkpoint'], row['engine'], row['mode'], row['batch'])

    # Capacity is shared across prompt domains. A complete run under an accepted
    # alternative dispatch proves the batch fits even if the first allocation failed.
    recovered = set()
    for row in active_rows(records):
        if not provisional_reason(row) and (
                default_llama_verified(row) or kai_default.verified(row, MANIFEST) or direct_head_verified(row)):
            recovered.add((configuration(row), ""))
            recovered.add((configuration(row), row["hardware_id"]))
    return [row for row in active_rows(limits) if current_limit_build(row)
            and (configuration(row), observation_board(row)) not in recovered]


def counts(records, limits, extra=()):
    records = active_rows(records)
    limits = active_limits(records, limits)
    planned = {(gpu, 1, key, engine, mode, batch, domain)
               for gpu in TARGETS for key, engine, mode, batch in cells(MANIFEST["models"], gpu=gpu)
               for domain in timing_domains(mode)}
    # Include required multi-GPU tests even when the rental is still pending.
    assigned = minimum_gpu_assignments().union(tuple(row) for row in extra)
    for r in list(records) + list(limits):
        gpu, n = device_group(r["gpu"])
        if n > 1:
            assigned.add((gpu, n, r["checkpoint"], r["engine"], r["mode"], r["batch"]))
    for gpu, n, key, engine, mode, batch in assigned:
        if gpu not in ("3090", "5080", "5090") or n not in (2, 4) or (key, engine, mode, batch) not in set(cells([key])):
            raise ValueError(f"Invalid multi-GPU assignment: {(gpu, n, key, engine, mode, batch)}")
        planned.update((gpu, n, key, engine, mode, batch, domain) for domain in timing_domains(mode))

    def cell(row):
        return (*device_group(row["gpu"]), row["checkpoint"], row["engine"], row["mode"], row["batch"], row["domain"])

    done = {cell(row) for row in records if timing_accepted(row)}
    # Keep already measured plain prose cells in coverage without scheduling any more.
    planned.update(c for c in done if c[4] == "plain" and c[-1] == "prose" and (*c[:-1], "code") in planned)
    excluded = {c: reason for c in planned if (reason := known_exclusion(*c[:4], batch=c[5]))}
    for r in limits:
        if r.get('observation_only'):
            continue
        group = device_group(r["gpu"])
        for domain in timing_domains(r["mode"]):
            marked = (*group, r["checkpoint"], r["engine"], r["mode"], r["batch"], domain)
            if marked in planned:
                excluded[marked] = "Measured configuration limit"
    records = [row for row in records if cell(row) in planned]
    historical = {cell(r) for r in records if previously_accepted(r) and not provisional_reason(r)}
    graphed = {cell(r) for r in records if graphs_verified(r) and not provisional_reason(r)}
    direct = {cell(r) for r in records if direct_head_verified(r) and not provisional_reason(r)}
    stock = {cell(r) for r in records if default_llama_verified(r) and not provisional_reason(r)}
    released = {cell(r) for r in records if published_kai.verified(r) and not provisional_reason(r)}
    normal = {cell(r) for r in records if kai_default.verified(r, MANIFEST) and not provisional_reason(r)}
    if done - planned:
        raise ValueError(f"Timing records outside the declared matrix: {done-planned}")
    for cell in done:
        excluded.pop(cell, None)
    gpu_totals = group_counts(planned, excluded, done, lambda cell: cell[:2])
    family_totals = group_counts(planned, excluded, done, lambda cell: MANIFEST["models"][cell[2]]["family"])
    inkling_releases = {
        cell for cell in planned
        if MANIFEST["models"][cell[2]]["family"] == "inkling"
        and MANIFEST["models"][cell[2]].get("benchmark_gpus")
    }
    release_counts = group_counts(
        inkling_releases, set(excluded) & inkling_releases, done & inkling_releases,
        lambda cell: (cell[0], MANIFEST["models"][cell[2]]["kind"]),
    )
    hardware = [{"gpu": gpu, "count": n, **gpu_totals[gpu, n]}
                for gpu, n in sorted(gpu_totals, key=lambda g: (TARGETS.index(g[0]), g[1]))]
    eligible = len(planned) - len(excluded)
    excluded_by_engine = {}
    for cell, reason in excluded.items():
        excluded_by_engine.setdefault(reason, Counter())[cell[3]] += 1
    return {"planned":len(planned), "unavailable":len(excluded), "eligible":eligible,
            "filled":len(done), "historical_cells":len(historical), "graph_verified_cells":len(graphed),
            "published_kai_cells":len(released), "kai_default_cells":len(normal), "llama_default_cells":len(stock), "direct_head_cells":len(direct), "historical_graph_overlap":len(historical & graphed), "remaining":eligible-len(done), "percent":100*len(done)/eligible if eligible else 0,
            "excluded_by_reason":dict(sorted(Counter(excluded.values()).items())),
            "excluded_by_engine":excluded_by_engine,
            "by_gpu":{f"{n}× {gpu}": t["filled"] for (gpu,n), t in sorted(gpu_totals.items()) if t["filled"]},
            "hardware":hardware, "families":family_totals,
            "published_inkling": [{"gpu": gpu, "kind": kind, **total}
                                  for (gpu, kind), total in sorted(release_counts.items())]}
