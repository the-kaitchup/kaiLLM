"""Rebuild the GPU-first and model-first speed reports from completed runs. Never match across GPUs or prompt banks."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics

import benchmark
import llama_graph
import llama_kv_limits
import kai_default
import empirical_capacity
import published_kai
import prefix_checked
import llama_default
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from urllib.parse import quote

from benchmark import (BATCHES, DSPARK_TOKENS, MANIFEST, OUTPUT, PROMPT, PROTOCOL, current_limit_build, current_weights, full_graph_runtime, inkling_fused_q8,
                       dspark_asset, dspark_digest, inkling_draft_runtime, inkling_native_placement, placement_resources, kai_loader_runtime, llama_revision, loader_variant, model_at_revision,
                       verify_weight_digests)
from devices import is_gb10, physical_id
from matrix import DGX_SPARK
from historical_timings import previously_accepted, receipt_hash
from cuda_graphs import output_head_devices, pipeline_stages
from coverage_matrix import (DRAFT_FIELDS, GRAPH_WAVES, TARGETS, active_limits, active_rows, counts,
                             device_group, direct_head_verified, graph_phase_verified, graphs_verified,
                             historical_graph_mode, loader_exclusion, observation_board, provisional_reason,
                             read_assignments, timing_accepted)
from worker import startup_failure, startup_tail


EARLIER_BENCHMARKS = "\n## Earlier short-context benchmarks\n"


def digest(items):
    return hashlib.sha256(json.dumps(items, separators=(",", ":")).encode()).hexdigest()


def sampling_key(row):
    """Match the target distribution and seed. Source is documentation provenance."""
    if "sampling" not in row:
        return ()
    return tuple(sorted((k, v) for k, v in row["sampling"].items() if k != "source"))


def sampling_label(row):
    return "Model-recommended" if "sampling" in row else "Archived greedy"


def validate_sampling(row, family):
    if "sampling" not in row:
        return
    profile = row["sampling"]
    if (not isinstance(profile, dict) or profile != benchmark.sampling_for(family)
            or any(isinstance(value, bool) for value in profile.values())):
        raise ValueError("Timing sampler differs from the model-recommended protocol")


def checkpoint_label(key, revision=None):
    model = model_at_revision(key, revision) if revision is not None else MANIFEST["models"][key]
    return f"[{model['repo']}]({checkpoint_url(model)}) ({model['gguf_type']})"


def checkpoint_url(model):
    return f"https://huggingface.co/{model['repo']}/tree/{model['revision']}"


def disclosure(title, body):
    return ["<details>", f"<summary><strong>{escape(title)}</strong></summary>", "", *body, "", "</details>", ""]


def collapsed_section(lines):
    return lines[:2] + disclosure("Show table", lines[2:]) if lines else []


def compact_graph_proof(raw, devices):
    """Preserve physical-rank coverage without publishing the raw CUDA device identifiers."""
    result = {}
    uuids = [dev["uuid"] for dev in devices]
    for phase, labels in GRAPH_WAVES.items():
        wave = raw.get(phase) if isinstance(raw, dict) else None
        if not isinstance(wave, dict):
            continue
        summary = {key: wave[key] for key in ("graph_replays", "eager_steps", "unclassified_steps") if type(wave.get(key)) is int}
        if wave.get("scope") == "timed-wave":
            summary["scope"] = "timed-wave"
        measured = wave.get("per_wave_by_gpu")
        if (isinstance(measured, dict) and set(measured) == set(labels) and len(set(uuids)) == len(uuids)
                and all(isinstance(values, dict) and set(values) == set(uuids) for values in measured.values())):
            summary["per_wave_by_rank"] = {label: [measured[label][uuid] for uuid in uuids] for label in labels
                                           if all(type(n) is int for n in measured[label].values())}
        result[phase] = summary
    return result


def reference_launches(intervals, ndev):
    """Retain newly captured, updated, and unchanged graph launches for each physical rank."""
    if set(intervals) != set(GRAPH_WAVES["prefill"]):
        raise ValueError("Incomplete llama.cpp graph waves")
    result = {}
    for wave, data in intervals.items():
        version = data["version"]
        phases = llama_graph.PHASES[:4] if version == 1 else llama_graph.PHASES
        if version not in (1, 2) or data["device_count"] != ndev or set(data["phases"]) != set(phases):
            raise ValueError("llama.cpp graph layout differs from the measured hardware")
        result[wave] = {}
        for phase, rows in data["phases"].items():
            counts = []
            if len(rows) != ndev:
                raise ValueError("Missing llama.cpp graph device")
            for row in rows:
                events = llama_graph.EVENTS[:5] if version == 1 else llama_graph.EVENTS
                if any(type(row.get(k)) is not int or row[k] < 0 for k in events):
                    raise ValueError("Invalid llama.cpp graph counters")
                captured, reused = row["capture_launches"], row["replays"]
                if (row["rejected"] or row["captures"] != captured or captured + reused < row["batches"]
                        or row.get("updates", 0) > captured or (phase == "unattributed" and any(row[k] for k in events))):
                    raise ValueError("Incomplete llama.cpp graph execution")
                item = {"launches_after_capture": captured, "unchanged_replays": reused}
                if version == 2:
                    item["updated_executable_launches"] = row["updates"]
                counts.append(item)
            if any(item["launches_after_capture"] or item["unchanged_replays"] for item in counts):
                result[wave][phase] = counts
    return result


def compact_head_graph_proof(raw, devices, head_uuids, stages):
    try:
        head_devices = output_head_devices(devices, stages)
    except ValueError:
        return {}
    size = len(head_devices)
    if head_uuids != [dev["uuid"] for dev in head_devices]:
        return {}
    return {**compact_graph_proof(raw, head_devices), "pipeline_stages": stages,
            "device_ranks": list(range(len(devices)-size, len(devices)))}


def compact_direct_head_proof(raw, devices, head_uuids, stages):
    heads = output_head_devices(devices, stages)
    uuids = [d['uuid'] for d in heads]
    if head_uuids != uuids or not isinstance(raw, dict) or set(raw) != set(GRAPH_WAVES):
        raise ValueError('Direct head evidence differs from the final pipeline stage or measured phases')
    result = {'pipeline_stages': stages, 'device_ranks': list(range(len(devices)-len(heads), len(devices)))}
    for phase, labels in GRAPH_WAVES.items():
        obs = raw[phase]
        calls = obs.get('per_wave_by_gpu') if isinstance(obs, dict) else None
        if (not isinstance(calls, dict) or set(calls) != set(labels) or
                any(not isinstance(device_calls, dict) or set(device_calls) != set(uuids) for device_calls in calls.values())):
            raise ValueError('Missing measured direct output-head calls for a wave or device')
        result[phase] = {key: obs.get(key) for key in ('scope', 'execution', 'completed_calls')}
        result[phase]['per_wave_by_rank'] = {label: [calls[label][uid] for uid in uuids] for label in labels}
    return result


def compact_draft_graph_proof(raw, devices, drafter_uuids):
    if not isinstance(raw, dict) or not isinstance(drafter_uuids, list):
        return {}
    uuids = [dev["uuid"] for dev in devices]
    if (not drafter_uuids or any(type(uuid) is not str or uuid not in uuids for uuid in drafter_uuids)
            or len(set(drafter_uuids)) != len(drafter_uuids)):
        return {}
    result = {key: raw[key] for key in (*DRAFT_FIELDS, "eager_steps", "unclassified_steps") if type(raw.get(key)) is int}
    if raw.get("scope") == "timed-wave":
        result["scope"] = "timed-wave"
    result["device_ranks"] = [uuids.index(uuid) for uuid in drafter_uuids]
    wave_results = {}
    waves = raw.get("per_wave_by_gpu")
    if isinstance(waves, dict) and set(waves) == set(GRAPH_WAVES["prefill"]):
        for label, values in waves.items():
            if (isinstance(values, dict) and set(values) == set(drafter_uuids)
                    and all(isinstance(v, dict) for v in values.values())):
                wave_results[label] = [{k: values[uuid][k] for k in DRAFT_FIELDS if type(values[uuid].get(k)) is int}
                                       for uuid in drafter_uuids]
    result["per_wave_by_rank"] = wave_results
    return result


def get_record(r, *, published_microbatch=None):
    key = r["checkpoint"]
    m = model_at_revision(key, r["model_revision"])
    if (r["protocol"], r["prompt_tokens"], r["generated_tokens"]) != (PROTOCOL, PROMPT, OUTPUT):
        raise ValueError(f"Invalid protocol for {key}")
    validate_sampling(r, m["family"])
    verify_weight_digests(m, r["sha256"])
    n = r["batch"]
    if n not in BATCHES or len(r["prompt_hashes"]) != n or len(set(r["prompt_hashes"])) != n:
        raise ValueError("Invalid prompt batch")
    if len(r["prefill"]) != 3:
        raise ValueError("Expected three prefill probes")
    for wave, length in [(w, 1) for w in r["prefill"]] + [(r["full"], OUTPUT)]:
        if len(wave["requests"]) != n or any((x["prompt_tokens"], x["output_tokens"]) != (PROMPT, length)
                or x["cache_hits"] not in (0, None) for x in wave["requests"]):
            raise ValueError("Incomplete request or cached prefix")
    env = r["environment"]
    dev = env["devices"]
    argv = env["server_argv"]
    hardware_id = physical_id(d["uuid"] for d in dev)
    batch_tokens = 2048
    if r["engine"] == "llama.cpp":
        microbatch = published_microbatch if published_microbatch is not None else 512
        if env.get('llama_policy') == 'default':
            proof = r.get('llama_default') or {}
            microbatch = proof.get('microbatch')
            if (env.get('llama_startup_attempts') != proof.get('startup_attempts')
                    or not llama_default.valid_attempts(proof.get('startup_attempts'), microbatch)):
                raise ValueError('Unverified upstream llama.cpp microbatch fallback')
        batch_args = {"--parallel": n, "-b": 2048, "-ub": microbatch}
    else:
        option = "--max-num-batched-tokens"
        limit = argv[argv.index(option) + 1] if option in argv else None
        for placement in m.get("kai_prefill_chunk_placements", ()):
            if (r["engine"] == "kaiLLM" and len(dev) == 1 and limit == str(placement["tokens"])
                    and (hardware_id, env["kai_source"], env["engine_revision"], r["mode"], n, r["domain"]) == (
                        placement["hardware_id"], placement["source_revision"], placement["source_lock"],
                        placement["mode"], placement["batch"], placement["domain"])
                    and env.get("cuda_graph_capture_sizes_by_gpu") == {dev[0]["uuid"]: placement["capture_sizes"]}):
                batch_tokens = placement["tokens"]
                break
        batch_args = {"--max-num-seqs": n, option: batch_tokens}
    if any(flag not in argv or argv[argv.index(flag) + 1] != str(value) for flag, value in batch_args.items()):
        raise ValueError("Server request slots or token batch limits differ from the benchmark settings")
    if m.get("kai_pipeline_runtime") and len(dev) > 1 and r["engine"] == "kaiLLM":
        if "--pipeline-parallel-size" not in argv or int(argv[argv.index("--pipeline-parallel-size") + 1]) != len(dev):
            raise ValueError("Hy4 pipeline stage count differs from the physical device count")
    if m["family"] == "hy4" and r["engine"] == "llama.cpp":
        required = {"--override-tensor": r"^per_layer_token_embd\.weight$=CPU", "--load-mode": "mmap", "--lazy-mode": "on"}
        if any(option not in argv or argv[argv.index(option) + 1] != value for option, value in required.items()):
            raise ValueError("Hy4 llama.cpp timing requires the pinned CPU Engram offload settings")
    option = "--ctx-size" if r["engine"] == "llama.cpp" else "--max-model-len"
    context = int(argv[argv.index(option) + 1])
    gpu = env["gpu_names"]
    if (hardware_id in MANIFEST.get("dgx_spark_hardware", ()) and len(dev) == 1 and is_gb10(dev[0])
            and gpu == [dev[0]["name"]]):
        gpu = [DGX_SPARK]
    record = {"protocol": PROTOCOL, "gpu": gpu, "hardware_id": hardware_id,
            "driver": [d["driver_version"] for d in dev], "power_limit_w": [d["power.limit"] for d in dev],
            "checkpoint": key, "model_revision": r["model_revision"], "sha256": r["sha256"], "engine": r["engine"],
            "kai_source": env["kai_source"], "llama_source": None if "numerical_scope" in env else env["llama_source"],
            "engine_revision": env["engine_revision"], "mode": r["mode"], "batch": n, "domain": r["domain"],
            "requested_context_per_slot": context // n if r["engine"] == "llama.cpp" else context,
            "prompt_set_sha256": digest(r["prompt_hashes"]), "input_tokens_per_request": PROMPT,
            "output_tokens_per_request": OUTPUT, "prefill_seconds": [w["seconds"] for w in r["prefill"]],
            "full_seconds": r["full"]["seconds"], "generated_text_hashes": [x["output_sha256"] for x in r["full"]["requests"]],
            "timestamp_utc": r["timestamp_utc"]}
    if "sampling" in r:
        record["sampling"] = r["sampling"].copy()
    if batch_tokens != 2048:
        record["kai_prefill_chunk_tokens"] = batch_tokens
    if env.get('kai_policy') == 'default' or 'kai_default_dispatch' in r:
        if (env.get('kai_policy') != 'default' or 'kai_default_dispatch' not in r
                or env.get('measured_gguf_sha256') != r['sha256']):
            raise ValueError('Missing normal-dispatch proof or actual GGUF byte hashes')
        config_mode = kai_default.launch(env['effective_command'], m['family'])
        if m['family'] == 'lfm':
            if env.get('kai_humming_deterministic') is not False:
                raise ValueError('Ordinary LFM3 comparison must retain the original GEMM selection')
            record['kai_default_config_mode'] = config_mode
            record['kai_humming_deterministic'] = False
        record['measured_gguf_sha256'] = env['measured_gguf_sha256']
        record['kai_default_dispatch'] = kai_default.compact(r['kai_default_dispatch'], [d['uuid'] for d in dev])
    if 'kai_empirical_capacity' in r or 'empirical_capacity_harness' in env:
        proof = r.get('kai_empirical_capacity')
        if (not isinstance(proof,dict) or env.get('empirical_capacity_harness') != proof.get('harness_sha256')
                or env.get('kai_policy') != 'default' or 'kai_default_dispatch' not in record):
            raise ValueError('Missing empirical capacity source or ordinary dispatch evidence')
        empirical_capacity.check_requests(r)
        empirical_capacity.validate_launch(argv,env['effective_command'])
        record['kai_empirical_capacity'] = empirical_capacity.validate_evidence(proof, r['kai_default_dispatch']['full'])
    if "cuda_graphs" in r:
        record["cuda_graphs"] = compact_graph_proof(r["cuda_graphs"], dev)
    if r["engine"] == "llama.cpp" and "llama_cuda_graph_intervals" in r:
        record["llama_graph_launches_by_wave"] = reference_launches(r["llama_cuda_graph_intervals"], len(dev))
    if "target_head_cuda_graphs" in r and r["engine"] == "kaiLLM":
        pp = pipeline_stages(argv)
        record["target_head_cuda_graphs"] = compact_head_graph_proof(
            r["target_head_cuda_graphs"], dev, env.get("cuda_graph_head_devices"), pp)
    if env.get('benchmark_output_head') == 'direct' or 'target_head_direct' in r:
        if (env.get('benchmark_output_head') != 'direct' or 'target_head_direct' not in r
                or env.get('cuda_graph_head_devices') != [] or env.get('cuda_graph_drafter_devices') != []
                or env.get('measured_gguf_sha256') != r['sha256']):
            raise ValueError('Direct head route or rehashed checkpoint differs from the measured deployment')
        record['measured_gguf_sha256'] = env['measured_gguf_sha256']
        record['target_head_direct'] = compact_direct_head_proof(
            r['target_head_direct'], dev, env.get('direct_head_devices'), pipeline_stages(argv))
    if "drafter_cuda_graphs" in r:
        record["drafter_cuda_graphs"] = compact_draft_graph_proof(r["drafter_cuda_graphs"], dev, env.get("cuda_graph_drafter_devices"))
    if r["engine"] == "kaiLLM":
        extra = json.loads(argv[argv.index("--extra-vllm-args") + 1]) if "--extra-vllm-args" in argv else []
        eager_flags = [arg.replace('_', '-').split('=', 1) for arg in (*argv, *extra)]
        if env.get("kai_enforce_eager") is True or any(parts[0] == '--enforce-eager' and
                (len(parts) == 1 or parts[1].lower() not in ('false', '0')) for parts in eager_flags):
            record["kai_enforce_eager"] = True
    if r["engine"] == "kaiLLM":
        size = int(argv[argv.index("--kv-cache-memory-bytes") + 1])
        capacity = env["max_concurrency"]
        record["kai_kv"] = {"bytes_per_device": size, "full_context_sequences": capacity}
        if not empirical_capacity.capacity_valid(record):
            raise ValueError("Invalid full-context KV allocation or empirical capacity evidence")
    if r["engine"] == "llama.cpp" and r["mode"].startswith("ngram"):
        method = argv[argv.index("--spec-type") + 1]
        timings = [request["server_timings"] for request in r["full"]["requests"]]
        for timing in timings:
            proposed = timing.get("draft_n", 0)
            accepted = timing.get("draft_n_accepted", 0)
            if type(proposed) is not int or type(accepted) is not int or not 0 <= accepted <= proposed:
                raise ValueError("Invalid llama.cpp n-gram draft counts for a request")
        record["llama_ngram"] = {
            "method": method,
            "lookup_tokens": int(argv[argv.index(f"--spec-{method}-size-n") + 1]),
            "draft_tokens": int(argv[argv.index(f"--spec-{method}-size-m") + 1]),
            "proposed": sum(t.get("draft_n", 0) for t in timings),
            "accepted": sum(t.get("draft_n_accepted", 0) for t in timings),
        }
        if method == "ngram-map-k":
            record["llama_ngram"]["min_hits"] = int(argv[argv.index("--spec-ngram-map-k-min-hits") + 1])
        setup = record["llama_ngram"]
        expected_method = "ngram-map-k" if r["mode"] == "ngram1" else "ngram-simple"
        if (setup["lookup_tokens"] != 2 or setup["draft_tokens"] != int(r["mode"].removeprefix("ngram")) or
                method not in ("ngram-simple", expected_method) or (method == "ngram-map-k" and setup["min_hits"] != 1)):
            raise ValueError("llama.cpp n-gram settings differ from the recorded benchmark policy")
    if r["engine"] == "kaiLLM" and r["mode"] in ("ngram1", "ngram3") and "kai_ngram" in r:
        policy = {"lookup_min": 2, "lookup_max": 8, "draft_tokens": int(r["mode"].removeprefix("ngram"))}
        for name, flag in (("lookup_min", "--lookup-min"), ("lookup_max", "--lookup-max"), ("draft_tokens", "--spec-tokens")):
            if flag not in argv or argv[argv.index(flag) + 1] != str(policy[name]) or r["kai_ngram"].get(name) != policy[name]:
                raise ValueError("GPU n-gram launch parameters differ from the recorded lookup and draft lengths")
        record["kai_ngram"] = {**policy, **{key: r["kai_ngram"].get(key) for key in ("scope", "proposed", "accepted", "rejected")}}
    if env.get("kai_ampere_qwen_opt_in"):
        record["kai_ampere_qwen_opt_in"] = True
    if 'kai_inkling_fused_q8' in env:
        if env['kai_inkling_fused_q8'] is not True:
            raise ValueError('Unexpected Inkling grouped-projection setting')
        record['kai_inkling_fused_q8'] = True
    if env.get("kai_humming_deterministic") or (r["engine"] == "kaiLLM" and m.get("kai_dspark_batches")
                                                and "kai_humming_deterministic" in env):
        record["kai_humming_deterministic"] = env["kai_humming_deterministic"]
    dspark = r['mode'] in DSPARK_TOKENS
    if r["mode"].startswith("dspine") or dspark:
        native = dspark and r["engine"] == "kaiLLM"
        flag = ("--dspark" if native else "--model-draft") if dspark else "--dspine"
        path = Path(argv[argv.index(flag) + 1])
        ref = dspark_asset(m)[1] if dspark else MANIFEST["dspine"]
        suffix = Path(ref["repo"]) / ref["revision"]
        if dspark:
            filename, = ref["files"]
            suffix /= filename
        if path.parts[-len(suffix.parts):] != suffix.parts:
            raise ValueError(f"Drafter in the measured server command differs from the pinned {r['mode']} checkpoint")
        record["drafter"] = {"repo": ref["repo"], "revision": ref["revision"]}
        if native:
            if "--spec-tokens" not in argv or argv[argv.index("--spec-tokens") + 1] != str(DSPARK_TOKENS[r['mode']]):
                raise ValueError("Inkling draft-token limit differs from the benchmark mode")
            record["drafter"].update(filename=filename, sha256=ref["files"][filename]["sha256"])
            record["kai_dspark"] = r.get("kai_dspark")
    placement = (inkling_native_placement(m, n, record['gpu'], record['kai_source'], hardware_id=record['hardware_id'],
                                          draft_tokens=DSPARK_TOKENS.get(record['mode'], 5))
                 if record['engine'] == 'kaiLLM' and (record['mode'] == 'plain' or dspark) else None)
    if placement:
        record.update(kai_inkling_prefill_bias_mib=env.get('kai_inkling_prefill_bias_mib'),
                      kai_inkling_qualification_sha256=env.get('kai_inkling_qualification_sha256'),
                      kai_inkling_serial_verify_post=env.get('kai_inkling_serial_verify_post', False),
                      kai_inkling_row_stable_dspark=env.get('kai_inkling_row_stable_dspark', False),
                      kai_inkling_serial_verify_route=env.get('kai_inkling_serial_verify_route', False),
                      kai_inkling_serial_verify_bias=env.get('kai_inkling_serial_verify_bias', False),
                      kai_inkling_serial_verify_combine=env.get('kai_inkling_serial_verify_combine', False),
                      kai_inkling_verify_combine_fma4=env.get('kai_inkling_verify_combine_fma4',''),
                      measured_gguf_sha256=env.get('measured_gguf_sha256'))
        if 'kai_inkling_itq_moe_decode_rows' in env:
            record['kai_inkling_itq_moe_decode_rows'] = env['kai_inkling_itq_moe_decode_rows']
        if 'kai_inkling_packed_verify_attention' in env:
            record['kai_inkling_packed_verify_attention'] = env['kai_inkling_packed_verify_attention']
        if 'kai_inkling_itq_moe_shared_workspace' in env:
            record['kai_inkling_itq_moe_shared_workspace'] = env['kai_inkling_itq_moe_shared_workspace']
        if dspark:
            record['measured_drafter_sha256']=env.get('measured_drafter_sha256')
        if placement_resources(placement,record['mode'])[1]:
            record['kai_inkling_capture_sizes'] = env['cuda_graph_capture_sizes_by_gpu'].get(dev[0]['uuid'])
    if 'kai_default_dispatch' in record and not kai_default.verified(record, MANIFEST):
        raise ValueError('Normal kaiLLM dispatch failed its measured per-wave or runtime checks')
    if 'target_head_direct' in record and not direct_head_verified(record):
        raise ValueError('Invalid source, weights, backbone or direct output-head evidence')
    if 'llama_default' in r or env.get('llama_policy') == 'default':
        if ('llama_default' not in r or env.get('llama_policy') != 'default' or r['engine'] != 'llama.cpp'
                or env.get('llama_process') != r['llama_default'].get('process')
                or env.get('measured_gguf_sha256') != r['sha256'] or env.get('cuda_graph_head_devices') != []
                or env.get('cuda_graph_drafter_devices') != [] or env.get('cuda_graph_capture_sizes_by_gpu') != {}):
            raise ValueError('Missing or conflicting default-policy llama.cpp provenance')
        record['llama_default'] = llama_default.compact(r['llama_default'], [d['uuid'] for d in dev], r['sha256'])
        if record['llama_default']['speculative'] != llama_default.counts(r['full'], r['mode']):
            raise ValueError('llama.cpp default speculative counts differ from the server responses')
        if dspark:
            sha = dspark_digest(m)
            if env.get('measured_drafter_sha256') != sha:
                raise ValueError('Default-policy llama.cpp DSpark head was not rehashed')
            record['llama_default']['measured_drafter_sha256'] = sha
        llama_default.validate_command(argv, r['mode'], context//n, microbatch)
        if not llama_default.verified(record):
            raise ValueError('llama.cpp default receipt did not verify its binary, libraries, and execution policy')
    if "numerical_scope" in env:
        record["numerical_scope_sha256"] = prefix_checked.check_scope(r, record)
    elif native_inkling_record(record):
        verify_kai_native_record(record)
    return record


def native_inkling_record(row):
    if row["engine"] != "kaiLLM":
        return False
    model = model_at_revision(row["checkpoint"], row["model_revision"])
    pin = inkling_draft_runtime(model, row["batch"], len(row["gpu"]), gpu_names=row["gpu"], revision=row["kai_source"],
                                hardware_id=row.get("hardware_id"), draft_tokens=DSPARK_TOKENS.get(row['mode'], 5))
    return row['mode'] in DSPARK_TOKENS or (pin is not None and row["kai_source"] == pin[0])


def verify_kai_native_record(row):
    """Check the exact target, batch, runtime, Humming setting and drafter counters."""
    model = model_at_revision(row["checkpoint"], row["model_revision"])
    draft_tokens = DSPARK_TOKENS.get(row['mode'])
    pin = inkling_draft_runtime(model, row["batch"], len(row["gpu"]), gpu_names=row["gpu"], revision=row["kai_source"],
                                hardware_id=row.get("hardware_id"), draft_tokens=draft_tokens or 5)
    if ((row["kai_source"], row["engine_revision"]) != pin
            or row["mode"] != "plain" and draft_tokens is None
            or row.get("kai_humming_deterministic") is not True
            or row["llama_source"] != llama_revision(model)):
        raise ValueError("Inkling native timings require the qualified single-slot runtime and deterministic Humming")
    placement = inkling_native_placement(model, row["batch"], row["gpu"], row["kai_source"],
                                        hardware_id=row.get("hardware_id"), draft_tokens=draft_tokens or 5)
    if placement:
        expected_kv, expected_sizes = placement_resources(placement, row['mode'])
        expected_weights = [f['sha256'] for f in model['files'].values()]
        if (row.get('hardware_id') != placement['hardware_id']
                or row.get('kai_inkling_prefill_bias_mib') != placement.get('prefill_bias_mib')
                or row.get('kai_kv',{}).get('bytes_per_device') != expected_kv * 2**20
                or row.get('kai_inkling_capture_sizes') != expected_sizes
                or row.get('kai_inkling_qualification_sha256') != placement.get('qualification_sha256')
                or row.get('kai_inkling_row_stable_dspark', False) != (
                    draft_tokens is not None and placement.get('row_stable_dspark', False))
                or row.get('kai_inkling_itq_moe_decode_rows') != (
                    placement.get('itq_moe_decode_rows') if draft_tokens is not None else None)
                or row.get('kai_inkling_packed_verify_attention', False) is not (
                    row['mode']=='dspark5' and placement.get('packed_verify_attention',False))
                or row.get('kai_inkling_itq_moe_shared_workspace', False) is not (
                    draft_tokens is not None and placement.get('itq_moe_share_prefill_workspace',False))
                or row.get('kai_inkling_verify_combine_fma4','') != placement.get('verify_combine_fma4','')
                or any(row.get('kai_inkling_'+key,False) is not placement.get(key,False)
                       for key in ('serial_verify_post','serial_verify_bias','serial_verify_route','serial_verify_combine'))
                or row.get('measured_gguf_sha256') != expected_weights or row.get('sha256') != expected_weights
                or not graphs_verified(row)):
            raise ValueError('Inkling timing differs from its qualified device, source, memory, weights or graph policy')
    if row["mode"] == "plain":
        if "kai_dspark" in row or "drafter" in row:
            raise ValueError("A plain Inkling control cannot contain drafter evidence")
        return
    counts = row.get("kai_dspark") or {}
    numbers = [counts.get(k) for k in ("proposed", "accepted", "rejected")]
    if (counts.get("scope") not in ("full-wave", "observed-request-interval")
            or any(type(n) is not int for n in numbers)
            or numbers[0] <= 0 or min(numbers[1:]) < 0 or numbers[0] != sum(numbers[1:])):
        raise ValueError("Native DSpark needs measured proposed, accepted and rejected counts with their scope")
    observed = counts.get("observed_generated_tokens")
    if observed is not None and (counts["scope"] != "observed-request-interval"
                                 or type(observed) is not int or not 1 <= observed <= OUTPUT):
        raise ValueError("Observed generation coverage must fit one native timed request")
    ref = dspark_asset(model)[1]
    filename, = ref["files"]
    if placement and (row.get('measured_drafter_sha256') != ref['files'][filename]['sha256'] or numbers[1] <= 0):
        raise ValueError('The qualified native timing must verify its draft head and observed acceptance')
    expected = dict(repo=ref["repo"], revision=ref["revision"], filename=filename,
                    sha256=ref["files"][filename]["sha256"])
    if row.get("drafter") != expected:
        raise ValueError("Inkling native drafter filename or digest differs from the pinned GGUF")


def rates(row):
    prefill = statistics.median(row["prefill_seconds"])
    return row["batch"] * PROMPT / prefill, row["batch"] * (OUTPUT - 1) / (row["full_seconds"] - prefill)


def matched_performance(pairs):
    # Use the most recent kaiLLM measurement of each physical workload, independent
    # of its speed. If that measurement has both references, use upstream defaults.
    latest = {}
    for pair in pairs:
        if "kaiLLM" not in pair or "llama.cpp" not in pair:
            continue
        kai, reference = pair["kaiLLM"], pair["llama.cpp"]
        if sampling_key(kai) != sampling_key(reference):
            continue
        key = (measured_workload(kai), kai["model_revision"], tuple(kai["sha256"]),
               reference["llama_source"])
        rank = (kai.get("timestamp_utc", ""), llama_default.verified(reference), reference.get("timestamp_utc", ""))
        if key not in latest or rank > latest[key][0]:
            latest[key] = (rank, pair)
    groups = defaultdict(list)
    for _, pair in latest.values():
        kai, stock = pair["kaiLLM"], pair["llama.cpp"]
        groups[sampling_label(kai)].append((kai, rates(kai), rates(stock)))
    if not groups:
        return []
    lines = ["## Matched engine performance", ""]
    for label in ("Model-recommended", "Archived greedy"):
        if label in groups:
            lines.extend(matched_summary(groups[label], label))
    return lines


def matched_summary(comparisons, label):
    lines = [f"### {label}", "",
             f"{len(comparisons)} matched {'pair shares' if len(comparisons) == 1 else 'pairs share'} the same hardware, checkpoint and workload settings.",
             "", "| Phase | kaiLLM faster | Tied | llama.cpp faster | llama.cpp lead above 10% |",
             "|---|---:|---:|---:|---:|"]
    largest_leads = []
    for phase, name in enumerate(("Prefill", "Decode")):
        wins = sum(kai[phase] > llama[phase] for _, kai, llama in comparisons)
        ties = sum(kai[phase] == llama[phase] for _, kai, llama in comparisons)
        losses = len(comparisons) - wins - ties
        above_threshold = sum(llama[phase] > 1.10 * kai[phase] for _, kai, llama in comparisons)
        lines.append(f"| {name} | {wins} | {ties} | {losses} | {above_threshold} |")
        if losses:
            row, kai, llama = max(comparisons, key=lambda pair: pair[2][phase] / pair[1][phase])
            gpu, count = device_group(row["gpu"])
            power_limits = [str(w).removesuffix(".00") for w in row["power_limit_w"]]
            power = (power_limits[0] if len(set(power_limits)) == 1 else "/".join(power_limits)) + " W"
            largest_leads.append(
                f"- {name} on {gpu_label(gpu, count)} ({power}), batch {row['batch']}, {row['mode']}, {row['domain']}, "
                f"kaiLLM `{row['kai_source'][:8]}`. "
                f"kaiLLM {kai[phase]:,.1f} vs. llama.cpp {llama[phase]:,.1f} tokens/s "
                f"({(llama[phase] / kai[phase] - 1):.1%} llama.cpp lead). {checkpoint_label(row['checkpoint'])}")
    if largest_leads:
        lines.extend(["", "Largest observed llama.cpp lead by phase", "", *largest_leads])
    return [*lines, ""]


def confirmed_limit(stamp, log=None):
    # A different reference execution policy can have a different memory footprint.
    if stamp.get('llama_policy') == 'default':
        return None
    if stamp.get("state") != "failed" or not stamp.get("gpu_names"):
        return None
    if not current_limit_build(stamp):
        return None
    reason = stamp["reason"]
    if log is not None and stamp.get("stage") == "startup" and log.is_file():
        # Historical workers used a short log tail. Match the original receipt
        # before attributing a log retained from a possibly superseded attempt.
        full = log.read_text(errors="replace")
        if reason == startup_tail(full) and abs(log.stat().st_mtime - stamp["time"]) <= 60:
            reason = startup_failure(full)
    match = re.search(r"Full-context KV capacity [\d.]+ sequences < batch \d+", reason)
    if match:
        # A small configured KV pool can produce this before exhausting device memory.
        return None
    if re.search(r"out of memory|cannot allocate memory|failed to allocate.*buffer|no available memory for the cache blocks", reason, re.I):
        detail = "Memory allocation failed at the requested context"
    elif reason.startswith("Unknown GGUF architecture in the measured engine:"):
        detail = reason
    elif "No kaiLLM release native catalog for compute capability" in reason:
        detail = reason
    elif reason in ("The pinned kaiLLM loader requires separate QKV and gate tensors for this GGUF",
                    "GGUF has no native MTP tensor head", "kaiLLM Inkling has no DSpark verifier",
                    "The merged adapter has no native draft verifier for this family"):
        detail = reason
    else:
        return None
    limit = {"gpu": stamp["gpu_names"], **{k: stamp[k] for k in ("checkpoint", "engine", "mode", "batch")}, "reason": detail}
    if "model_revision" in stamp:
        limit["model_revision"] = stamp["model_revision"]
    if stamp["engine"] == "llama.cpp" and "llama_source" in stamp:
        limit["llama_source"] = stamp["llama_source"]
    if stamp["engine"] == "kaiLLM":
        model = MANIFEST["models"][stamp["checkpoint"]]
        devices = len(stamp["gpu_names"])
        if (full_graph_runtime(stamp["checkpoint"], stamp["mode"], batch=stamp["batch"], gpu_names=stamp["gpu_names"]) or
                inkling_draft_runtime(model, stamp["batch"], devices, gpu_names=stamp["gpu_names"]) or kai_loader_runtime(model, devices)):
            limit["kai_source"] = stamp["kai_source"]
    return limit


FAMILIES = {"lfm": "LFM3 1.7B", "daisy": "DaisyRose 27B", "qwen": "Qwen3.8 27B",
            "inkling": "Inkling 2 Mini", "hy4": "Hy4 Flash"}
MODES = ("plain", "ngram1", "ngram3", "mtp5", *DSPARK_TOKENS, "dspine7", "dspine15")


def execution_label(row):
    if published_kai.verified(row):
        return "CUDA graphs captured at startup"
    if kai_default.verified(row, MANIFEST):
        waves = row['kai_default_dispatch']
        phases = []
        for phase in kai_default.PHASES:
            direct = any(r[phase]['direct'] for ranks in waves.values() for r in ranks)
            graph = any(r[phase]['full'] for ranks in waves.values() for r in ranks)
            label = ('graphs + eager' if graph and direct else 'graphs' if graph else 'eager')
            if any(r[phase].get('piecewise', 0) for ranks in waves.values() for r in ranks):
                label = ' + '.join(name for name, active in (('full graphs', graph), ('piecewise graphs', True), ('eager', direct)) if active)
            phases.append(f'{phase} {label}')
        label = 'Default (' + ', '.join(phases) + ', eager head) · `' + row['kai_source'][:8] + '`'
        if 'kai_empirical_capacity' in row:
            label += f" · B2 verified on workload ({row['kai_kv']['full_context_sequences']:.2f} nominal slots)"
        return label
    if llama_default.verified(row):
        proof = row['llama_default']
        label = f"llama.cpp default (ub {proof['microbatch']})"
        if proof.get('published_client'):
            if 'greedy_matching_outputs' in proof:
                label += f" · {proof['greedy_matching_outputs']}/8 greedy control texts matched"
            return label
        attempts = proof.get('startup_attempts') or []
        if (any(a['status'] == 'prefill-graph-oom' for a in attempts)
                and attempts[-1].get('admission') != 'separate-process'):
            label += ' · capacity probe on timed process'
        return label
    if direct_head_verified(row):
        return 'Graphed backbone, eager head'
    if graphs_verified(row):
        return "Forced graphs" if row["engine"] == "llama.cpp" else "Full graphs"
    if row.get("kai_enforce_eager") or historical_graph_mode(row) == "eager":
        return "Eager"
    if historical_graph_mode(row) == "decode_only":
        return "Decode-only capture"
    return "Unrecorded"


def archived_checkpoint_table(records):
    lines = []
    for row in sorted(records, key=lambda r: (r["checkpoint"], r["engine"], r["batch"], r["domain"])):
        model = model_at_revision(row["checkpoint"], row["model_revision"])
        gpu, count = device_group(row["gpu"])
        provenance = "Earlier checkpoint" if not current_weights(row) else "Retired GPU"
        link = f'[{model["repo"]}](https://huggingface.co/{model["repo"]}/tree/{model["revision"]}) ({model["gguf_type"]})'
        pp, tg = rates(row)
        lines.append(f'| {provenance} | {link} | {gpu_label(gpu, count)} | {row["engine"]} | {row["batch"]} | '
                     f'{row["mode"]} | {row["domain"]} | {pp:.1f} | {tg:.1f} | {execution_label(row)} | {sampling_label(row)} |')
    if not lines:
        return []
    return ["## Earlier checkpoint and GPU measurements", "",
            "| Scope | Checkpoint | GPU | Engine | Batch | Mode | Domain | Prefill | Decode | GPU execution | Sampling |",
            "|---|---|---|---|---:|---|---|---:|---:|---|---|", *lines, ""]


def result_block(samples, observations):
    lines = []
    if samples:
        lines.extend(["| Batch | Mode | Domain | kaiLLM prefill | llama.cpp prefill | kaiLLM decode | llama.cpp decode | GPU execution | Sampling |",
                      "|---:|---|---|---:|---:|---:|---:|---|---|"])
        for batch, mode, domain, label, nums in sorted(
                samples, key=lambda r: (r[0], MODES.index(r[1]) if r[1] in MODES else len(MODES),
                                        r[1], r[2] != "prose", r[2], r[3])):
            lines.append("| " + " | ".join(map(str, [batch, label, domain, *nums])) + " |")
        lines.append("")
    if observations:
        lines.extend(["<details>", f"<summary>Configuration limits ({len(observations)})</summary>", "",
                      "| Engine | Mode | Batch | Observation |", "|---|---|---:|---|"])
        for limit in sorted(observations, key=lambda r: (r["engine"], r["mode"], r["batch"])):
            engine = 'llama.cpp default' if limit.get('llama_policy') == 'default' else limit['engine']
            lines.append(f"| {engine} | {limit['mode']} | {limit['batch']} | {limit['reason']} |")
        lines.extend(["", "</details>", ""])
    return lines


def measured_workload(row):
    return (tuple(row["gpu"]), row["hardware_id"], tuple(row["driver"]), tuple(row["power_limit_w"]),
            row["checkpoint"], row["mode"], row["batch"], row["domain"], row["prompt_set_sha256"], sampling_key(row))


def same_default_workload(target, reference, *, check_sampling=True):
    # A historical kaiLLM process may predate the current llama.cpp pin. Only
    # the actual reference executed llama.cpp; both runs must share these inputs.
    # Sampler is the final element. A known prompt alone can authenticate a
    # diagnostic input while its sampled ordinary comparison is still pending.
    return (measured_workload(target)[:-1] == measured_workload(reference)[:-1]
            and (not check_sampling or sampling_key(target) == sampling_key(reference))
            and target["sha256"] == reference["sha256"]
            and target["protocol"] == reference["protocol"]
            and target["input_tokens_per_request"] == reference["input_tokens_per_request"]
            and target["output_tokens_per_request"] == reference["output_tokens_per_request"])


def closest_inkling_references(records):
    """Compare supported Inkling weights at the nearest size on the same batch-one workload."""
    targets = {}
    references = defaultdict(list)
    sizes = {
        key: sum(file["bytes"] for file in model["files"].values())
        for key, model in MANIFEST["models"].items() if model["family"] == "inkling"
    }
    for row in sorted(records, key=lambda row: row.get("timestamp_utc", ""), reverse=True):
        model, mode = row["checkpoint"], row["mode"]
        if (model not in sizes or row["batch"] != 1 or (mode != "plain" and mode not in DSPARK_TOKENS)
                or not row.get("hardware_id") or not row.get("driver") or not row.get("power_limit_w")
                or (row.get("protocol"), row.get("input_tokens_per_request"), row.get("output_tokens_per_request")) != (PROTOCOL, PROMPT, OUTPUT)):
            continue
        draft = row.get("measured_drafter_sha256") or row.get("llama_default", {}).get("measured_drafter_sha256")
        if mode in DSPARK_TOKENS and not draft:
            continue
        workload = measured_workload(row)
        # The reference has different GGUF bytes. All other workload fields still match.
        key = (*workload[:4], *workload[5:], draft)
        unavailable = loader_exclusion(model, "llama.cpp")
        if row["engine"] == "kaiLLM" and unavailable and timing_accepted(row):
            targets.setdefault((model, key), row)
        elif row["engine"] == "llama.cpp" and not unavailable and llama_default.verified(row):
            references[key].append(row)

    comparisons = defaultdict(list)
    for (model, key), target in targets.items():
        reference = min(references[key], key=lambda row: abs(sizes[row["checkpoint"]] - sizes[model]), default=None)
        if reference:
            group = (*device_group(target["gpu"]), model)
            comparisons[group].append((target, reference))
    return comparisons


def closest_reference_block(pairs):
    lines = ["| Mode | Domain | Closest-size llama.cpp checkpoint | Prefill vs reference | Decode vs reference | Sampling |",
             "|---|---|---|---:|---:|---|"]
    for target, reference in sorted(pairs, key=lambda pair: (pair[0]["mode"], pair[0]["domain"],
                                                            sampling_label(pair[0]), pair[0]["hardware_id"])):
        prefill, decode = (f"{a / b:.1%}" for a, b in zip(rates(target), rates(reference), strict=True))
        link = checkpoint_label(reference["checkpoint"], reference["model_revision"])
        lines.append(f"| {target['mode']} | {target['domain']} | {link} | {prefill} | {decode} | {sampling_label(target)} |")
    return disclosure("Batch-one performance versus closest-size GGUF", lines)


def inkling_draft_comparisons(records):
    """Compare draft lengths and plain decoding on matching Inkling workloads."""
    cohorts = defaultdict(dict)
    for row in records:
        model = MANIFEST['models'][row['checkpoint']]
        mode = row['mode']
        if (model['family'] != 'inkling' or (mode != 'plain' and mode not in DSPARK_TOKENS)
                or not row.get('hardware_id') or not row.get('driver') or not row.get('power_limit_w')
                or not current_weights(row) or not timing_accepted(row)
                or (row.get('protocol'), row.get('input_tokens_per_request'), row.get('output_tokens_per_request')) != (PROTOCOL, PROMPT, OUTPUT)):
            continue
        workload = measured_workload(row)
        key = (row['engine'], *workload[:5], *workload[6:], tuple(row['sha256']))
        previous = cohorts[key].get(mode)
        if previous is None or rates(row)[1] > rates(previous)[1]:
            cohorts[key][mode] = row
    comparisons = defaultdict(list)
    for entries in cohorts.values():
        plain = entries.get('plain')
        if plain is not None:
            group = (*device_group(plain['gpu']), plain['checkpoint'])
            for mode in DSPARK_TOKENS:
                if mode in entries:
                    comparisons[group].append((plain, entries[mode]))
        if 'dspark5' in entries and 'dspark3' in entries:
            baseline = entries['dspark5']
            group = (*device_group(baseline['gpu']), baseline['checkpoint'])
            comparisons[group].append((baseline, entries['dspark3']))
    return comparisons


def draft_comparison_block(pairs):
    lines = ['| Engine | Batch | Domain | Baseline | Mode | Baseline decode | Measured decode | Change | Sampling |',
             '|---|---:|---|---|---|---:|---:|---:|---|']
    for baseline, draft in sorted(pairs, key=lambda pair: (pair[0]['batch'], pair[0]['engine'], pair[0]['domain'],
                                                           pair[0]['mode'] != 'plain', pair[1]['mode'],
                                                           sampling_label(pair[0]), pair[0]['hardware_id'])):
        ordinary, speculative = rates(baseline)[1], rates(draft)[1]
        lines.append(f"| {draft['engine']} | {draft['batch']} | {draft['domain']} | {baseline['mode']} | {draft['mode']} | {ordinary:.1f} | "
                     f"{speculative:.1f} | {speculative / ordinary - 1:+.1%} | {sampling_label(draft)} |")
    return disclosure('DSpark decode comparisons', lines)


def output_agreement_notes(records, runtimes):
    """Compare each registered tuning run to its preceding ordinary run on the same physical workload."""
    controls = {pin["revision"]: pin["output_control_revision"] for pin in runtimes if "output_control_revision" in pin}
    sources = set(controls) | set(controls.values())
    baselines = {}
    notes = defaultdict(list)
    for row in sorted((r for r in records if r.get("kai_source") in sources and r.get("engine") == "kaiLLM"),
                      key=lambda r: r["timestamp_utc"]):
        source = row["kai_source"]
        key = (measured_workload(row), tuple(row["sha256"]), row["protocol"],
               row["input_tokens_per_request"], row["output_tokens_per_request"])
        hashes = row.get("generated_text_hashes")
        if not isinstance(hashes, list) or len(hashes) != row["batch"]:
            continue
        base = baselines.get((controls.get(source), key))
        if base is not None:
            matches = sum(x == y for x, y in zip(hashes, base, strict=True))
            group = (*device_group(row["gpu"]), row["checkpoint"])
            notes[group].append(f"At batch {row['batch']}, kaiLLM `{source[:8]}` reproduced the full {row['domain']} "
                                f"output of `{controls[source][:8]}` for {matches} of {row['batch']} requests ({sampling_label(row)}).")
        baselines[source, key] = hashes
    for pin in runtimes:
        validation = pin.get("matched_request_order_check")
        if validation:
            groups = {(*device_group(row["gpu"]), row["checkpoint"]) for row in records
                      if row.get("kai_source") == pin["revision"]}
            for group in groups:
                notes[group].append(f"With matching request order, a fresh greedy control agreed on all "
                                    f"{validation['tokens']:,} token decisions and top-eight scores.")
    return notes


def kai_rate(value, reference, missing="—"):
    if value is None:
        return missing
    number = f"{value:.1f}"
    return "$\\color{red}{" + number + "}$" if reference is not None and reference > 1.10 * value else number


def prefix_checked_section(diagnostics, records):
    if not diagnostics:
        return []
    notes = {}
    for row in diagnostics:
        note = prefix_checked.scope_note(row.get("numerical_scope_sha256"))
        prefix_checked.require(note, "unknown archived scope")
        notes[row["kai_source"]] = note
    introduction = ("Four 48-token outputs matched ordinary decoding after 8,192-token prompts." if len(notes) == 1 else
                    "Each kaiLLM build matched ordinary decoding on four 48-token outputs after 8,192-token prompts.")
    message = (next(iter(notes.values())) if len(set(notes.values())) == 1 else
               " ".join(f"`{source[:8]}` — {note}" for source, note in sorted(notes.items())))
    lines = ["## " + prefix_checked.LABEL, "",
             f"{introduction} {message}", "",
             "| Checkpoint | GPU | Batch | Domain | Engine | Prefill tok/s | Decode tok/s | Draft tokens accepted/proposed | CUDA graphs | Full text matching ordinary kaiLLM runs | Sampling |",
             "|---|---|---:|---|---|---:|---:|---:|---|---|---|"]
    printed_references = set()
    for row in sorted(diagnostics, key=lambda r: (r["checkpoint"], r["gpu"], r["batch"], r["domain"] != "prose", r["timestamp_utc"])):
        control_key = {**row, "mode": "plain"}
        controls = [r for r in records if r.get("engine") == "kaiLLM" and r["mode"] == "plain"
                    and r["model_revision"] == row["model_revision"] and timing_accepted(r)
                    and same_default_workload(control_key, r, check_sampling=False) and len(r.get("generated_text_hashes", [])) == 1]
        # A published full-length control also identifies the exact canonical prompt for this domain.
        prefix_checked.require(controls, "missing an ordinary full-length control on this board and prompt")
        controls = [r for r in controls if sampling_key(r) == sampling_key(row)]
        checks = ", ".join(f"`{r['kai_source'][:8]}` " +
                           ("yes" if row["generated_text_hashes"] == r["generated_text_hashes"] else "no")
                           for r in sorted(controls, key=lambda r: (r["kai_source"], r["timestamp_utc"]))) or "Awaiting same-sampler plain timing"
        spec_refs = [r for r in records if r.get("engine") == "llama.cpp" and r["mode"] == "dspark5"
                     and r["model_revision"] == row["model_revision"] and same_default_workload(row, r)
                     and llama_default.verified(r) and r["llama_default"].get("measured_drafter_sha256") == row["drafter"]["sha256"]]
        ref = max(spec_refs, key=lambda r: r["timestamp_utc"], default=None)
        ref_rates = rates(ref) if ref else (None, None)
        gpu, count = device_group(row["gpu"])
        start = f"| {checkpoint_label(row['checkpoint'], row['model_revision'])} | {gpu_label(gpu, count)} | {row['batch']} | {row['domain']} |"
        kai = [kai_rate(v, c) for v, c in zip(rates(row), ref_rates)]
        build = f"kaiLLM `{row['kai_source'][:8]}`"
        if row["numerical_scope_sha256"] == prefix_checked.PACKED_EXPERT_SCOPE:
            prefix_checked.require(re.fullmatch("[0-9a-f]{64}", row.get("packed_expert_proof_sha256", "")),
                                   "missing packed expert selector receipt")
            build += " · packed expert gates"
        entries = [(build, kai, row["kai_dspark"],
                    "Verified" if graphs_verified(row) else "Unverified", checks)]
        ref_key = (measured_workload(ref), ref["timestamp_utc"], ref["llama_source"]) if ref else None
        if ref and ref_key not in printed_references:
            printed_references.add(ref_key)
            policy = ref["llama_default"]
            entries.append((f"llama.cpp default `{ref['llama_source'][:8]}` (ub{policy['microbatch']})",
                            [f"{v:.1f}" for v in ref_rates], policy["speculative"], "Upstream default", "—"))
        for engine, (pp, tg), counts, graph, matching in entries:
            accepted, proposed = counts["accepted"], counts["proposed"]
            lines.append(f"{start} {engine} | {pp} | {tg} | {accepted:,}/{proposed:,} ({accepted/proposed:.1%}) | {graph} | {matching} | {sampling_label(row)} |")
    return [*lines, ""]


def gpu_label(gpu, count):
    return f"{count} × {DGX_SPARK}" if gpu == DGX_SPARK else f"{count} × RTX {gpu}"


def tables(records, limits=(), assignments=(), unavailable_rentals=(), gpu_filename="speed.md", *, prefix_timings=()):
    records = list(records)
    historical_appendix = archived_checkpoint_table([r for r in records if previously_accepted(r) and
                                                     (device_group(r["gpu"])[0] not in TARGETS or not current_weights(r))])
    records = active_rows(records)
    limits = {(*r["gpu"], r["checkpoint"], r["engine"], r["mode"], r["batch"], r.get('llama_policy'), observation_board(r)): r
              for r in active_limits(records, limits)}.values()
    rows = defaultdict(dict)
    plain = {}
    def workload(r):
        spec = MANIFEST["models"][r["checkpoint"]]
        inkling_target = spec.get("kai_dspark_batches") or spec.get('kai_dspark_placements')
        engine_revision = r.get("engine_revision") if r["engine"] == "kaiLLM" and inkling_target else None
        return (previously_accepted(r), "llama_default" in r, engine_revision, r["hardware_id"], tuple(r["driver"]), tuple(r["power_limit_w"]), r["checkpoint"],
                r["batch"], r["domain"], r["prompt_set_sha256"], r["engine"], r["llama_source"], sampling_key(r))
    def paired_workload(r):
        return (*measured_workload(r), r["llama_source"])

    current_graph_sources = {}
    direct_sources = {}
    for r in records:
        if direct_head_verified(r) and not provisional_reason(r):
            direct_sources[paired_workload(r)] = (r['kai_source'], r['engine_revision'])
        if r["engine"] == "kaiLLM" and graphs_verified(r) and not provisional_reason(r):
            pin = full_graph_runtime(r["checkpoint"], r["mode"], batch=r["batch"], gpu_names=r["gpu"], hardware_id=r.get("hardware_id"), model_revision=r.get("model_revision"))
            if pin is not None and r["kai_source"] == pin[0]:
                current_graph_sources[paired_workload(r)] = pin

    strict_references = {}
    for r in records:
        if r['engine'] == 'llama.cpp' and graphs_verified(r) and not provisional_reason(r) and not previously_accepted(r):
            key = paired_workload(r)
            if key not in strict_references or r['timestamp_utc'] > strict_references[key]['timestamp_utc']:
                strict_references[key] = r

    stock_refs = {}
    for r in records:
        if not timing_accepted(r):
            continue
        if 'llama_default' in r:
            w = paired_workload(r)
            if w not in stock_refs or r['timestamp_utc'] > stock_refs[w]['timestamp_utc']:
                stock_refs[w] = r
            if r['mode'] == 'plain':
                plain[workload(r)] = r['generated_text_hashes']
            continue
        model = MANIFEST["models"][r["checkpoint"]]
        devices = len(r["gpu"])
        loader_pin = kai_loader_runtime(model, devices)
        if r["llama_source"] == MANIFEST["llama_cpp"]["inkling_revision"]:
            loader_pin = inkling_draft_runtime(model, r["batch"], devices, gpu_names=r["gpu"],
                                              hardware_id=r["hardware_id"]) or loader_pin
        if r["engine"] == "llama.cpp" and not previously_accepted(r):
            matched = paired_workload(r)
            loader_pin = current_graph_sources.get(matched, direct_sources.get(matched, loader_pin))
        comparison_source = loader_pin[0] if loader_pin and r["engine"] == "llama.cpp" else r["kai_source"]
        key = (*measured_workload(r), previously_accepted(r), comparison_source, r["llama_source"])
        prior = rows[key].get(r["engine"])
        if prior is not None:
            # Retain every receipt in the archive. For repeated measurements of
            # identical bytes, prefer the current pin, then the latest timestamp.
            current = model["revision"]
            if ((prior["model_revision"] == current, prior["timestamp_utc"]) >=
                    (r["model_revision"] == current, r["timestamp_utc"])):
                continue
        rows[key][r["engine"]] = r
        if direct_head_verified(r) and paired_workload(r) in strict_references:
            rows[key]['llama.cpp'] = strict_references[paired_workload(r)]
        if r["mode"] == "plain":
            plain[workload(r)] = r["generated_text_hashes"]
    # Preserve strict and historical pairs and label the extra default control
    # independently. Reuse it only on identical physical/driver/prompt/model slots.
    original = list(rows.items())
    for matched, ref in stock_refs.items():
        targets = [(key, pair) for key, pair in original if 'kaiLLM' in pair and same_default_workload(pair['kaiLLM'], ref)]
        if targets:
            for key, pair in targets:
                if 'llama.cpp' not in pair:
                    rows.pop(key, None)
                rows[(*key[:-2], key[-2] + '/llama-default', ref['llama_source'])] = {'kaiLLM': pair['kaiLLM'], 'llama.cpp': ref}
        else:
            rows[(*matched[:-1], False, '/llama-default', matched[-1])] = {'llama.cpp': ref}
    measurements = defaultdict(list)
    limits_by_model = defaultdict(list)
    for limit in limits:
        limits_by_model[(*device_group(limit["gpu"]), limit["checkpoint"])].append(limit)
    references = {MANIFEST["llama_cpp"]["inkling_revision"]: "Inkling PR", MANIFEST["llama_cpp"]["hy4_revision"]: "Hy4 PR"}
    graph_builds = defaultdict(set)
    for key, pair in rows.items():
        kai = pair.get("kaiLLM")
        if kai and kai.get("kai_source") and execution_label(kai) == "Full graphs":
            graph_builds[(*key[:-2], key[-1])].add(kai["kai_source"])
    for key, pair in sorted(rows.items()):
        gpus, _, _, _, model, mode, batch, domain, *_ = key
        x = rates(pair["kaiLLM"]) if "kaiLLM" in pair else (None, None)
        y = rates(pair["llama.cpp"]) if "llama.cpp" in pair else (None, None)
        kai_missing = "Unsupported GGUF" if loader_exclusion(model, "kaiLLM") else "—"
        llama_missing = "Unsupported GGUF" if loader_exclusion(model, "llama.cpp") else "—"
        nums = []
        for metric, (kai, llama) in enumerate(zip(x, y)):
            value = kai_rate(kai, llama, kai_missing)
            values = [value, llama_missing if llama is None else f"{llama:.1f}"]
            if metric == 1 and mode != "plain":
                for i, engine in enumerate(("kaiLLM", "llama.cpp")):
                    r = pair.get(engine)
                    if r and workload(r) in plain and r["generated_text_hashes"] != plain[workload(r)]:
                        values[i] += "†"
            nums.extend(values)
        execution = {engine: execution_label(row) for engine, row in pair.items()}
        if execution.get("kaiLLM") == "Full graphs" and len(graph_builds[(*key[:-2], key[-1])]) > 1:
            execution["kaiLLM"] += f" · `{pair['kaiLLM']['kai_source'][:8]}`"
        chunk_tokens = pair.get("kaiLLM", {}).get("kai_prefill_chunk_tokens")
        if chunk_tokens:
            execution["kaiLLM"] += f" · {chunk_tokens:,}-token prefill chunks"
        if len(set(execution.values())) == 1:
            nums.append(next(iter(execution.values())))
        else:
            nums.append(" · ".join(execution[engine] if execution[engine].startswith('llama.cpp default') else f"{engine} {execution[engine]}"
                                   for engine in ("kaiLLM","llama.cpp")))
        reference_name = references.get(key[-1]) if llama_missing != "Unsupported GGUF" else None
        label = f"{mode} ({reference_name})" if reference_name else mode
        kai_row = pair.get("kaiLLM", {})
        spec = MANIFEST["models"][model]
        current_pin = inkling_draft_runtime(spec, batch, len(gpus), gpu_names=gpus, hardware_id=key[1])
        native_pin = inkling_draft_runtime(spec, batch, len(gpus), gpu_names=gpus,
                                           revision=kai_row.get('kai_source'),hardware_id=kai_row.get('hardware_id'))
        if current_pin and kai_row and "kai_default_dispatch" not in kai_row:
            if native_pin and kai_row["kai_source"] == native_pin[0]:
                reference = "" if llama_missing == "Unsupported GGUF" else " · Inkling PR"
                source = " " + native_pin[0][:7] if kai_row.get('kai_inkling_qualification_sha256') else ""
                label = f"{mode} (deterministic kaiLLM{source}{reference})"
            elif direct_head_verified(kai_row):
                label = 'plain (deterministic Humming)'
            elif mode == "plain":
                recorded = kai_row.get("kai_humming_deterministic")
                setting = {None: "Humming unrecorded", False: "default Humming", True: "deterministic Humming"}[recorded]
                label = f"plain (earlier kaiLLM · {setting})"
        if mode == "ngram1" and "llama.cpp" in pair:
            label = "ngram1 (llama.cpp map-k)"
        if "llama_default" in pair.get("llama.cpp", {}):
            revision = pair["llama.cpp"]["llama_source"]
            upstream = MANIFEST["models"][model].get("llama_repository_override", "ggml-org/llama.cpp")
            pin = f"[{reference_name or 'llama.cpp'} {revision[:7]}](https://github.com/{upstream}/commit/{revision})"
            if reference_name and reference_name in label:
                label = label.replace(reference_name, pin)
            elif label.endswith(")"):
                label = f"{label[:-1]} · {pin})"
            else:
                label = f"{label} ({pin})"
        nums.append(sampling_label(next(iter(pair.values()))))
        measurements[(*device_group(gpus), model)].append((batch, mode, domain, label, nums))

    progress = counts(records, limits, assignments)
    gpu_link = quote(gpu_filename)
    intro = [
        "**8,192 input tokens and 65,536 output tokens per request.** Prefill and decode are aggregate tokens/s across the batch.", "",
    ]
    sources = MANIFEST["sources"]
    links = (f"[Prose source (Moby-Dick)]({sources['prose']['url']}) · "
             f"[Code source (CPython)]({sources['code']['url']}) · "
             "[Benchmark code](benchmarks/long_context/) · "
             "[Checkpoints](#checkpoint-files) · [Timing data](benchmarks/long_context/results.jsonl) · [Availability](#availability)")
    by_gpu = ["# kaiLLM prefill and decode speed", "", *intro, links, ""]
    by_model = ["# kaiLLM prefill and decode speed by model", "", *intro, f"[Browse by GPU]({gpu_link}) · " + links, ""]
    sections = []
    for group in progress["hardware"]:
        gpu, count = group["gpu"], group["count"]
        title = gpu_label(gpu, count)
        anchor = f"gpu-{gpu.lower().replace(' ', '-')}-{count}"
        sections.append((gpu, count, title, anchor))
    by_gpu.append(" · ".join(f"[{title}](#{anchor})" for _, _, title, anchor in sections))
    by_model.append(" · ".join(f"[{name}](#model-{family})" for family, name in FAMILIES.items()))
    legend = ["", "Red marks a kaiLLM rate when llama.cpp was more than 10% faster on the matched workload "
              "(**not optimized**). A dash indicates a missing measurement. "
              "† marks output that differed from a plain run on the same engine with the same sampler.", "",
              "**Archived greedy** selects the highest-probability token at each step (temperature 0). "
              "**Model-recommended** samples using the model publisher’s settings. "
              "Sampling can change the generated text and can affect draft acceptance and decode speed.", ""]
    by_gpu.extend(legend)
    by_model.extend(legend)
    by_model.extend(collapsed_section(matched_performance(rows.values())))
    checkpoints = {family: [key for key, model in MANIFEST["models"].items() if model["family"] == family]
                   for family in FAMILIES}
    blocks = {group: result_block(measurements.get(group, ()), limits_by_model.get(group, ()))
              for group in measurements.keys() | limits_by_model.keys()}
    agreement = output_agreement_notes(records, MANIFEST["kaillm"].get("default_dispatch_runtimes", []))
    for group, notes in agreement.items():
        if group in blocks:
            blocks[group].extend([" ".join(dict.fromkeys(notes)), ""])
    for group, pairs in closest_inkling_references(records).items():
        if group in blocks:
            blocks[group].extend(closest_reference_block(pairs))
    for group, pairs in inkling_draft_comparisons(records).items():
        if group in blocks:
            blocks[group].extend(draft_comparison_block(pairs))
    for gpu, count, title, anchor in sections:
        by_gpu.extend([f'<a id="{anchor}"></a>', "", f"## {title}", ""])
        has_data = False
        for family, keys in checkpoints.items():
            observed = [key for key in keys if (gpu, count, key) in blocks]
            if observed:
                has_data = True
                by_gpu.extend([f'<a id="{anchor}-model-{family}"></a>', "",
                               f"### {FAMILIES[family]}", ""])
                for key in observed:
                    by_gpu.extend([f"#### {checkpoint_label(key)}", "", *blocks[gpu, count, key]])
        if not has_data:
            by_gpu.extend(["No full-length benchmark results.", ""])
    for family, keys in checkpoints.items():
        by_model.extend([f'<a id="model-{family}"></a>', "", f"## {FAMILIES[family]}", ""])
        for key in keys:
            model = MANIFEST["models"][key]
            title = f"{model['repo']} ({model['gguf_type']})"
            body = [f"[{title}]({checkpoint_url(model)})", ""]
            observed = [(gpu_title, blocks[gpu, count, key]) for gpu, count, gpu_title, _ in sections
                        if (gpu, count, key) in blocks]
            if not observed:
                body.extend(["Full-length timings pending.", ""])
            for gpu_title, block in observed:
                # The outer checkpoint toggle reveals the timing and limit tables together.
                body.extend([f"#### {gpu_title}", "", *("<details open>" if line == "<details>" else line for line in block)])
            anchor = re.sub(r"[^\w -]", "", title.lower()).replace(" ", "-")
            by_model.extend([f'<a id="{anchor}"></a>', "", *disclosure(title, body)])

    by_gpu.extend(historical_appendix)
    by_model.extend(collapsed_section(historical_appendix))

    diagnostics = prefix_checked_section(prefix_timings, records)
    lines = ["## Availability", "", "See configuration limits under each checkpoint and GPU.", ""]
    if "Fused DaisyRose donor layout" in progress["excluded_by_reason"]:
        lines.extend([f"{checkpoint_label('daisy-iq2m')} uses a fused tensor layout supported by llama.cpp. "
                      "The benchmarked kaiLLM loader requires separate tensors.", ""])
    if unavailable_rentals:
        lines.extend(["| Pending rental | Last check |", "|---|---|"])
        for item in unavailable_rentals:
            lines.append(f"| {item['count']} × {item['gpu']} | No qualifying Vast.ai offer on {item['checked_utc']} |")
        lines.append("")

    files_section = ["## Checkpoint files", "", "<details>", "<summary>Hugging Face revisions and filenames</summary>", "",
                     "| Family | Checkpoint | Format | GGUF files |", "|---|---|---|---|"]
    for family, family_title in FAMILIES.items():
        for key, model in MANIFEST["models"].items():
            if model["family"] == family:
                files = "<br>".join(f"`{name}`" for name in model["files"])
                files_section.append(f"| {family_title} | {checkpoint_label(key)} | {model['kind']} | {files} |")
    files_section.extend(["", "</details>", ""])
    return ("\n".join(by_gpu + diagnostics + lines + files_section),
            "\n".join(by_model + collapsed_section(diagnostics) + collapsed_section(lines) + files_section))


def table(records, limits=(), assignments=(), unavailable_rentals=()):
    """Return the original GPU-first Markdown view."""
    return tables(records, limits, assignments, unavailable_rentals)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("directories", nargs="*", type=Path)
    parser.add_argument("--source-map", type=Path, help="Optional mapping from recorded commits to verified source-equivalent public revisions")
    parser.add_argument("--diagnostics", type=Path, help="Saved startup logs grouped by result directory name")
    parser.add_argument('--stock-kv-probes', type=Path, action='append', default=[],
                        help='Directory containing original llama.cpp KV probe receipts, matching logs and a physical GPU inventory')
    parser.add_argument('--published-llama-run', type=Path, action='append', default=[],
                        help='Completed upstream llama.cpp run with its binaries, process samples, model and prompt receipts')
    parser.add_argument('--published-kai-run', type=Path, action='append', default=[],
                        help='Completed plain LFM run using the kaiLLM release')
    parser.add_argument("--prefix-checked-run", type=Path, action="append", default=[],
                        help="Separately display a DSpark run with a pinned 8,192-token accuracy check")
    a = parser.parse_args()
    model_report = a.report.with_name("speed-by-model.md")
    if a.report == model_report:
        parser.error("The GPU-first output must have a different filename from speed-by-model.md")
    source_map = json.loads(a.source_map.read_text()) if a.source_map else {}
    records = {}
    prompt_sets = {}
    def add(r):
        if DGX_SPARK in r["gpu"] and (r["gpu"] != [DGX_SPARK] or r["hardware_id"] not in MANIFEST.get("dgx_spark_hardware", ())):
            raise ValueError("Unrecognized physical DGX Spark")
        if "numerical_scope_sha256" in r:
            raise ValueError("Read prefix-checked timings using --prefix-checked-run")
        model = model_at_revision(r["checkpoint"], r["model_revision"])
        validate_sampling(r, model["family"])
        if "sampling" in r and previously_accepted(r):
            raise ValueError("An archived greedy measurement cannot be relabeled as sampled")
        r["kai_source"] = source_map.get(r["kai_source"], r["kai_source"])
        if 'llama_default' in r and not llama_default.verified(r):
            raise ValueError('Invalid default-policy llama.cpp process or proposal evidence')
        if r.get('engine') == 'kaiLLM' and not empirical_capacity.capacity_valid(r):
            raise ValueError('Invalid archived KV allocation or empirical B2 proof')
        if 'published_kai' in r and not published_kai.verified(r):
            raise ValueError('Invalid public kaiLLM deployment or workload')
        native = native_inkling_record(r)
        route = 'direct' if 'target_head_direct' in r else 'graph'
        if route == 'direct' and not direct_head_verified(r):
            raise ValueError('Invalid source, weights, backbone or direct output-head evidence')
        graph_pin = full_graph_runtime(r["checkpoint"], r["mode"], r["kai_source"], output_head=route, batch=r["batch"], gpu_names=r["gpu"], hardware_id=r.get("hardware_id"), model_revision=r.get("model_revision")) if r["engine"] == "kaiLLM" else None
        if 'kai_default_dispatch' in r:
            if not kai_default.verified(r, MANIFEST):
                raise ValueError('Normal kaiLLM dispatch does not match the source or full-length workload')
            graph_pin = kai_default.runtime(MANIFEST, r['checkpoint'], r['mode'], r['kai_source'])
        deterministic_humming = (native or (graph_pin is not None and model["family"] == "inkling")
                                 or (r["engine"] == "kaiLLM" and model.get("kai_humming_deterministic", False)))
        # The unchanged public release is an independent ordinary baseline.
        if 'published_kai' in r and model["family"] == "inkling":
            deterministic_humming = False
        if r.get("kai_humming_deterministic", False) is not deterministic_humming:
            raise ValueError("Humming reduction setting differs from the checkpoint's timing protocol")
        fused_q8 = inkling_fused_q8(r['checkpoint'], r['mode'], r['batch'], r['gpu'], graph_pin)
        if r.get('kai_inkling_fused_q8', False) is not fused_q8:
            raise ValueError("Inkling grouped projections differ from the recorded source and GPU")
        expected = (PROTOCOL, model["revision"], PROMPT, OUTPUT)
        if (r["protocol"], r["model_revision"], r["input_tokens_per_request"], r["output_tokens_per_request"]) != expected:
            raise ValueError(f"Archived workload differs from the pinned protocol or checkpoint: {r['checkpoint']}")
        workload = (model["family"], r["batch"], r["domain"])
        if prompt_sets.setdefault(workload, r["prompt_set_sha256"]) != r["prompt_set_sha256"]:
            raise ValueError(f"Prompt token IDs differ for {workload}")
        llama = MANIFEST["llama_cpp"]
        allowed = {llama_revision(model)}
        if "llama_revision_override" not in model:
            allowed.add(llama["revision"])
        if r["mode"] in DSPARK_TOKENS:
            allowed = {llama_revision(model)} if model["family"] == "inkling" else set()
            ref = dspark_asset(model)[1]
            expected_drafter = {field: ref[field] for field in ("repo", "revision")}
            if not native and r.get("drafter") != expected_drafter:
                raise ValueError("DSpark drafter differs from the pinned reference")
        recorded_llama = r["llama_source"]
        if recorded_llama not in allowed:
            raise ValueError("llama.cpp source differs from the pinned revision")
        if native:
            verify_kai_native_record(r)
        elif graph_pin is not None:
            if (r["kai_source"], r["engine_revision"]) != graph_pin:
                raise ValueError("kaiLLM full-graph runtime differs from the pinned model source")
        elif r["engine"] == "kaiLLM":
            variant = loader_variant(model, len(r["gpu"])) or ("lfm_ngram" if r["mode"].startswith("ngram") else "default")
            kai = MANIFEST["kaillm"]
            if (not published_kai.verified(r) and (r["kai_source"] != kai[variant + "_revision"] or
                    r["engine_revision"] not in kai["runtime_locks"][variant].values())):
                raise ValueError(f"kaiLLM runtime differs from the pinned {variant} sources")
        elif r["engine"] != "llama.cpp" or recorded_llama[:7] not in r["engine_revision"]:
            raise ValueError("Unrecognized engine or llama.cpp binary revision")
        k = tuple(r[x] for x in ("hardware_id", "checkpoint", "engine", "mode", "batch", "domain", "kai_source", "llama_source", "timestamp_utc", "model_revision"))
        if k in records:
            previous = records[k]
            # An older compact receipt can be enriched from the same immutable run.
            for field in ("sha256", "kai_kv", "drafter", "kai_ampere_qwen_opt_in", "llama_ngram", "kai_humming_deterministic", "kai_inkling_fused_q8",
                          "kai_dspark", "kai_inkling_capture_sizes", "kai_inkling_prefill_bias_mib", "kai_inkling_serial_verify_post", "kai_inkling_serial_verify_bias", "kai_inkling_serial_verify_route", "kai_inkling_serial_verify_combine", "kai_inkling_verify_combine_fma4", "kai_inkling_qualification_sha256", "kai_inkling_row_stable_dspark", "kai_inkling_itq_moe_decode_rows", "kai_inkling_itq_moe_shared_workspace", "kai_inkling_packed_verify_attention", "measured_drafter_sha256", "kai_ngram", "cuda_graphs", "llama_graph_launches_by_wave", "llama_default", "published_kai", "kai_default_dispatch", "kai_empirical_capacity", "target_head_cuda_graphs", "target_head_direct", "measured_gguf_sha256", "drafter_cuda_graphs", "kai_enforce_eager", "kai_prefill_chunk_tokens"):
                if field in r:
                    previous.setdefault(field, r[field])
                elif field in previous:
                    r[field] = previous[field]
            if previous != r:
                raise ValueError("Conflicting receipts for the same measurement timestamp")
        verify_weight_digests(model, r.get("sha256"))
        records[k] = r
    for root in a.directories:
        for domain in ("prose", "code"):
            for f in root.rglob(f"*-{domain}.json"):
                raw = json.loads(f.read_text())
                if "numerical_scope" not in raw.get("environment", {}):
                    add(get_record(raw))
    for folder in a.published_kai_run:
        for row in published_kai.records(folder, get_record):
            add(row)
    if a.published_llama_run:
        from published_llama import records as published_records
        for folder in a.published_llama_run:
            for row in published_records(folder, get_record):
                add(row)
    if a.archive.exists():
        for line in a.archive.read_text().splitlines():
            if line.strip():
                existing = json.loads(line)
                existing.setdefault("requested_context_per_slot", PROMPT + OUTPUT)
                add(existing)
    ordered = [records[k] for k in sorted(records)]
    index = a.archive.with_name("limits.json")
    def limit_key(r):
        runtime = r.get("kai_source") if r["engine"] == "kaiLLM" else r.get("llama_source")
        return (tuple(r["gpu"]), r["checkpoint"], r["engine"], r["mode"], r["batch"], runtime or "",
                r.get('llama_policy',''), observation_board(r), r.get("model_revision") or "")
    limits = {limit_key(r): r for r in json.loads(index.read_text())} if index.exists() else {}
    measured_limits = {}
    for root in a.directories:
        for f in root.rglob("*.status.json"):
            stamp = json.loads(f.read_text())
            name = f.name.removesuffix(".status.json") + ".log"
            log = f.with_name(name)
            if not log.is_file() and a.diagnostics:
                log = a.diagnostics / f.parent.name / name
            r = confirmed_limit(stamp, log)
            if r:
                key = limit_key(r)
                # Older workers omitted the HF revision. Keep an existing
                # versioned observation when the same failure is collected again.
                if not r.get("model_revision") and any(
                        previous[:-1] == key[:-1] and current_limit_build(stored)
                        and all(stored.get(field) == value for field, value in r.items()) for previous, stored in limits.items()):
                    continue
                measured_limits[key] = r
            elif (stamp.get("reason") == "Insufficient VRAM after weights and working memory"
                  or stamp.get("reason", "").startswith("Full-context KV capacity ")):
                # The old preflight reserve was a heuristic, without an engine allocation.
                limits.pop(limit_key({**stamp, "gpu": stamp["gpu_names"]}), None)
    for folder in a.stock_kv_probes:
        for entry in llama_kv_limits.measured_limits(folder, ordered):
            measured_limits[limit_key(entry)] = entry
    limits.update(measured_limits)
    for r in ordered:
        if current_limit_build(r) and graphs_verified(r):
            cell = limit_key(r)[:5]
            for key in list(limits):
                if key[:5] == cell and current_limit_build(limits[key]) and not observation_board(limits[key]):
                    del limits[key]
    limits = [limits[k] for k in sorted(limits)]
    index.write_text(json.dumps(limits, indent=2) + "\n")
    a.archive.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in ordered))
    market = a.archive.with_name("hardware_availability.json")
    unavailable_rentals = json.loads(market.read_text()) if market.is_file() else []
    diagnostic_file = a.archive.with_name("prefix_checked_timings.json")
    diagnostics = json.loads(diagnostic_file.read_text()) if diagnostic_file.is_file() else []
    by_digest = {row["raw_sha256"]: row for row in diagnostics}
    for folder in a.prefix_checked_run:
        for row in prefix_checked.read_run(folder, get_record, rates):
            if row != by_digest.setdefault(row["raw_sha256"], row):
                raise ValueError("Conflicting records for a single prefix-checked timing receipt")
    diagnostics = [by_digest[key] for key in sorted(by_digest)]
    gpu_view, model_view = tables(ordered, limits, read_assignments(a.archive), unavailable_rentals,
                                  a.report.name, prefix_timings=diagnostics)
    if a.prefix_checked_run:
        diagnostic_file.write_text(json.dumps(diagnostics, indent=2) + "\n")
    # The earlier measurements are maintained alongside the generated tables in speed.md.
    history = Path(__file__).resolve().parents[2] / "speed.md"
    if history.is_file():
        _, heading, appendix = history.read_text().partition(EARLIER_BENCHMARKS)
        gpu_view += heading + appendix
    a.report.write_text(gpu_view)
    model_report.write_text(model_view)


if __name__ == "__main__":
    main()
