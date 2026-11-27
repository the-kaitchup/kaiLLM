"""Matched prefill and decode throughput for an already running local server."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import statistics
import struct
import threading
import time
import urllib.request
from pathlib import Path

import hybrid_execution
import empirical_capacity
import kai_default
from cuda_graphs import draft_proof_record, gpu_draft_mode, output_head_devices, pipeline_stages, proof_record, snapshot, verify_draft_interval, verify_head_interval, verify_interval

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / "manifest.json").read_text())


PROMPT = 8192
OUTPUT = 65536
BATCHES = (1, 2, 4, 8, 16, 32)
DSPARK_TOKENS = {"dspark3": 3, "dspark5": 5}
DRAFT_TOKENS = {"plain": 0, "mtp5": 5, **DSPARK_TOKENS, "dspine7": 7, "dspine15": 15,
                "ngram3": 3, "ngram1": 1}
PROTOCOL = "8k64k-v1"

SAMPLING_PROFILE = "model-recommended"
SAMPLING_PROFILES = (SAMPLING_PROFILE, "greedy")
MODEL_SAMPLING = {
    "lfm": (0.1, 1.0, 50, 1.1, "https://huggingface.co/LiquidAI/LFM3-1.7B#sampling-parameters"),
    "daisy": (1.0, 0.95, 20, 1.0, "https://huggingface.co/RadixArk/DaisyRose-27B#how-we-evaluated"),
    "qwen": (1.0, 0.95, 20, 1.0, "https://huggingface.co/Qwen/Qwen3.8-27B#best-practices"),
    "inkling": (1.0, 1.0, 0, 1.0, "https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/main/generation_config.json"),
    "hy4": (0.9, 1.0, 0, 1.0, "https://huggingface.co/tencent/Hy4-flash/blob/main/generation_config.json"),
}


def timing_domains(mode):
    return ("prose", "code") if DRAFT_TOKENS[mode] else ("code",)


def sampling_for(family, profile=SAMPLING_PROFILE):
    if profile == "greedy":
        return None
    if profile != SAMPLING_PROFILE:
        raise ValueError(f"Unknown sampling profile: {profile}")
    temperature, top_p, top_k, repeat, source = MODEL_SAMPLING[family]
    return {"profile": SAMPLING_PROFILE, "source": source, "seed": 42, "temperature": temperature,
            "top_p": top_p, "top_k": top_k, "min_p": 0.0, "repetition_penalty": repeat,
            "presence_penalty": 0.0, "frequency_penalty": 0.0}


def sampling_payload(engine, sampling):
    settings = {key: value for key, value in sampling.items() if key not in ("profile", "source")}
    if engine == "llama.cpp":
        settings["repeat_penalty"] = settings.pop("repetition_penalty")
        if settings["repeat_penalty"] != 1.0:
            # Match the checkpoint's prompt-and-generation repetition penalty throughout the workload.
            settings["repeat_last_n"] = PROMPT + OUTPUT
    return settings


def model_at_revision(checkpoint, revision):
    model = MANIFEST["models"][checkpoint]
    if revision == model["revision"]:
        return model
    try:
        return MANIFEST["model_history"][checkpoint][revision]
    except KeyError:
        raise ValueError(f"Unknown Hugging Face revision for {checkpoint}: {revision}") from None


def dspark_asset(model):
    """Resolve the pinned DSpark artifact published for this specific target."""
    key = model.get("dspark")
    if key is None:
        return "dspark", MANIFEST["dspark"]
    draft = MANIFEST["drafters"][key]
    if draft["repo"] == model["repo"] and draft["revision"] != model["revision"]:
        draft = MANIFEST["drafter_history"][key][model["revision"]]
    return key, draft


def dspark_digest(model):
    weight, = dspark_asset(model)[1]["files"].values()
    return weight["sha256"]


def verify_weight_digests(model, hashes):
    if hashes != [file["sha256"] for file in model["files"].values()]:
        raise ValueError(f"Weight digests differ from {model['repo']} at {model['revision']}")


def current_weights(row):
    """A historical revision qualifies only when every shard matches the current one."""
    key = row["checkpoint"]
    current = MANIFEST["models"][key]
    revision = row.get("model_revision")
    if revision is None:
        # Early capacity stamps lacked a model revision. They are only unambiguous
        # when all recorded revisions of this checkpoint used the same GGUFs.
        if "domain" in row:
            return False
        revisions = MANIFEST.get("model_history", {}).get(key, {}).values()
        return all(model["files"] == current["files"] for model in revisions)
    if revision != current["revision"] and revision not in MANIFEST.get("model_history", {}).get(key, {}):
        return False
    model = model_at_revision(key, revision)
    if "domain" in row or "sha256" in row:
        verify_weight_digests(model, row.get("sha256"))
    return model["files"] == current["files"]


def llama_revision(model):
    if isinstance(model, dict):
        if model.get("llama_revision_override"):
            return model["llama_revision_override"]
        model = model["family"]
    pins = MANIFEST["llama_cpp"]
    return pins.get(model + "_revision", pins["revision"])


def loader_variant(model, device_count=1):
    if device_count > 1 and model.get("kai_pipeline_runtime"):
        return model["kai_pipeline_runtime"]
    return model.get("kai_loader_runtime")


def runtime_pin(variant):
    if variant is None:
        return None
    pins = MANIFEST["kaillm"]
    return pins[variant + "_revision"], pins["runtime_locks"][variant]["source"]


def kai_loader_runtime(model, device_count=1):
    """Return the source pin for this GGUF and device layout."""
    return runtime_pin(loader_variant(model, device_count))


def full_graph_runtime(checkpoint, mode, revision=None, *, output_head="graph", batch=None, gpu_names=(), hardware_id=None, model_revision=None):
    """Select a locked graph build for this model and mode; retain older measurement pins."""
    catalog = 'direct_head_runtimes' if output_head == 'direct' else 'full_graph_runtimes'
    hybrid_execution.head_policy('kaiLLM', mode, output_head)
    if (mode == 'plain' or mode in DSPARK_TOKENS) and output_head == 'graph' and batch is not None:
        target = model_at_revision(checkpoint, model_revision) if model_revision is not None else MANIFEST['models'][checkpoint]
        placement = inkling_native_placement(target, batch, gpu_names, revision,
                                            hardware_id=hardware_id, draft_tokens=DSPARK_TOKENS.get(mode, 5))
        if placement:
            return placement['revision'], placement['source']
    for runtime in MANIFEST["kaillm"].get(catalog, ()):
        if (checkpoint in runtime["models"] and mode in runtime["modes"]
                and (revision is None or revision == runtime["revision"])
                and batch == runtime.get("batch", batch)
                and hardware_id == runtime.get("hardware_id", hardware_id)):
            return runtime["revision"], runtime["source"]
    return None


def inkling_fused_q8(checkpoint, mode, batch, gpu_names, pin):
    """Apply projection grouping only to the measured single-GPU serving configuration."""
    if pin is None or batch != 1 or mode != 'plain' or len(gpu_names) != 1:
        return False
    catalog = MANIFEST['kaillm']
    entries = (*catalog.get('default_dispatch_runtimes', ()), *catalog.get('full_graph_runtimes', ()))
    return any((entry['revision'], entry['source']) == pin and checkpoint in entry['models']
               and gpu_names[0] in entry.get('inkling_fused_q8', ()) for entry in entries)


def inkling_native_placement(model, batch, gpu_names, revision=None, *, hardware_id=None, draft_tokens=5):
    """A native graph qualification applies to its measured hardware and slot count."""
    if len(gpu_names) != 1:
        return None
    for placement in reversed(model.get('kai_dspark_placements', ())):
        if (placement.get('model_revision', model['revision']) == model['revision']
                and placement.get('draft_tokens', 5) == draft_tokens
                and ('dspark' not in model or placement.get('drafter_sha256') == dspark_digest(model))
                and (hardware_id is None or hardware_id == placement['hardware_id'])
                and placement['batch'] == batch and gpu_names[0] in (placement['gpu'], placement['matrix_gpu'])
                and (revision is None or revision == placement['revision'])):
            return placement
    return None


def inkling_draft_pair(model, batch, devices=1, *, gpu_names=()):
    return devices == 1 and (("dspark" not in model and batch in model.get("kai_dspark_batches", ()))
                             or inkling_native_placement(model, batch, gpu_names) is not None)


def inkling_draft_runtime(model, batch, devices=1, *, gpu_names=(), revision=None, hardware_id=None, draft_tokens=5):
    """Source revision and lock shared by a qualified native drafter and its plain control."""
    placement = (inkling_native_placement(model, batch, gpu_names, revision, hardware_id=hardware_id, draft_tokens=draft_tokens)
                 if devices == 1 else None)
    if placement:
        return placement['revision'], placement['source']
    if draft_tokens == 5 and inkling_draft_pair(model, batch, devices):
        pin = runtime_pin(model["kai_dspark_runtime"])
        if revision is None or revision == pin[0]:
            return pin
    return None


def placement_resources(placement, mode):
    """Read the ordinary or speculative pool and resident target graph sizes that passed qualification."""
    if mode == 'plain':
        return placement.get('plain_kv_mib', placement['kv_mib']), placement.get('plain_capture_sizes')
    return placement['kv_mib'], placement.get('capture_sizes')


def current_limit_build(row):
    if not current_weights(row):
        return False
    if row["engine"] != "llama.cpp":
        model = MANIFEST["models"][row["checkpoint"]]
        names = row.get("gpu") or row.get("gpu_names") or []
        pin = full_graph_runtime(row["checkpoint"], row.get("mode"), batch=row.get("batch"), gpu_names=names,
                                 hardware_id=row.get("hardware_id"), model_revision=row.get("model_revision"))
        if pin:
            return row.get("kai_source") == pin[0]
        devices = len(row.get("gpu") or row.get("gpu_names") or [])
        loader = inkling_draft_runtime(model, row.get("batch"), devices, gpu_names=names, hardware_id=row.get("hardware_id"),
                                       draft_tokens=DSPARK_TOKENS.get(row.get('mode'), 5)) or kai_loader_runtime(model, devices)
        return loader is None or row.get("kai_source") == loader[0]
    model = MANIFEST["models"][row["checkpoint"]]
    # A historical, unversioned failure cannot exclude a newly pinned reference.
    return row.get("llama_source", MANIFEST["llama_cpp"]["revision"]) == llama_revision(model)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def hash_tokens(ids):
    return hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()


def verify_prompt_bank(bank, family, domain):
    ref = MANIFEST["tokenizers"][family]
    if (bank["family"], bank["tokenizer_revision"], bank["tokenizer_sha256"]) != (
            family, ref["revision"], ref["tokenizer_sha256"]):
        raise ValueError("Prompt bank tokenizer differs from the manifest")
    data = bank["domains"][domain]
    if len(data["rows"]) != 33 or len(data["hashes"]) != 33:
        raise ValueError("Expected 32 timed prompt windows and one warmup window")
    fingerprints = []
    for row, claimed in zip(data["rows"], data["hashes"], strict=True):
        if len(row) != PROMPT or hash_tokens(row) != claimed:
            raise ValueError("Prompt bank integrity check failed")
        fingerprints.append(bytes.fromhex(claimed))
    if hashlib.sha256(b"".join(fingerprints)).hexdigest() != ref["prompt_bank_sha256"][domain]:
        raise ValueError("Prompt windows differ from the pinned benchmark inputs")


def prompt_bank(family, dest):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    ref = MANIFEST["tokenizers"][family]
    path = hf_hub_download(ref["repo"], "tokenizer.json", revision=ref["revision"])
    tokenizer = Tokenizer.from_file(path)
    bank = {"family": family, "tokenizer_revision": ref["revision"],
            "tokenizer_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), "domains": {}}
    for domain, source in MANIFEST["sources"].items():
        with urllib.request.urlopen(source["url"], timeout=120) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != source["sha256"]:
            raise ValueError(f"{domain} source hash mismatch")
        passage = content.decode("utf-8-sig")
        if domain == "prose":
            chapter = "CHAPTER 1. Loomings."
            # The second occurrence starts the novel after the table of contents.
            at = passage.upper().rfind(chapter.upper())
            if at < 0:
                raise ValueError("First chapter missing")
            passage = passage[at:]
        ids = tokenizer.encode(passage, add_special_tokens=False).ids
        usable = len(ids) - PROMPT
        if usable < 1024:
            raise ValueError(f"Insufficient distinct windows in {domain}")
        # Index 32 is disjoint from the timed batch by starting position and prefix.
        rows = [ids[i * usable // 32:i * usable // 32 + PROMPT] for i in range(33)]
        if len({tuple(row[:64]) for row in rows}) != 33:
            raise ValueError(f"Duplicate prompt prefixes in {domain}")
        bank["domains"][domain] = {"hashes": [hash_tokens(row) for row in rows], "rows": rows}
        verify_prompt_bank(bank, family, domain)
    write_json(dest, bank)


def completion(engine, url, ids, count, timeout, *, return_token_ids=False, sampling=None):
    if engine == "kaiLLM":
        endpoint = "/v1/completions"
        payload = {"model": "bench", "prompt": ids, "max_tokens": count, "temperature": 0,
                   "seed": 42, "ignore_eos": True}
        if return_token_ids:
            payload["return_token_ids"] = True
    else:
        endpoint = "/completion"
        payload = {"prompt": ids, "n_predict": count, "temperature": 0, "seed": 42,
                   "ignore_eos": True, "cache_prompt": False}
    if sampling is not None:
        settings = sampling_payload(engine, sampling)
        payload.update(settings)
    req = urllib.request.Request(url.rstrip("/") + endpoint, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        raw = json.load(res)
    if engine == "kaiLLM":
        choice, = raw["choices"]
        u = raw["usage"]
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
        received = (u["prompt_tokens"], u["completion_tokens"])
        if choice["finish_reason"] != "length":
            raise ValueError(f"Unexpected finish reason: {choice['finish_reason']}")
        output = choice["text"]
        timings = raw.get("metrics") or {}
    else:
        received = raw["tokens_evaluated"], raw["tokens_predicted"]
        timings = raw["timings"]
        cached = timings.get("cache_n", 0)
        if sampling is not None:
            effective = raw.get("generation_settings", {})
            for key, value in settings.items():
                actual = effective.get(key)
                if type(actual) not in (int, float) or not math.isclose(actual, value, rel_tol=1e-6, abs_tol=1e-7):
                    raise ValueError(f"llama.cpp reported sampling {key}={actual!r}; expected {value}")
        # llama.cpp also sets truncated when the final requested token lands exactly on
        # the context boundary. Distinguish that stop from an early cut or dropped input.
        if timings["prompt_n"] != len(ids) or raw.get("stop_type") != "limit":
            raise ValueError("llama.cpp reused input or stopped before the requested length")
        if raw.get("truncated") and (count != OUTPUT or received != (len(ids), OUTPUT)):
            raise ValueError("llama.cpp reached the context boundary prematurely")
        output = raw["content"]
    if received != (len(ids), count) or cached not in (None, 0):
        raise ValueError(f"Expected {(len(ids), count)} with empty cache, got {received}, cached={cached}")
    result = {"output_tokens": count, "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
              "prompt_tokens": len(ids), "cache_hits": cached, "server_timings": timings}
    if return_token_ids:
        tokens = choice["token_ids"] if engine == "kaiLLM" else None
        if not isinstance(tokens, list) or len(tokens) != count or any(type(token) is not int for token in tokens):
            raise ValueError("Qualification requires all generated token IDs")
        result["token_ids"] = tokens
    return result


def wave(engine, url, prompts, count, timeout, *, return_token_ids=False, sampling=None):
    gate = threading.Barrier(len(prompts) + 1)
    def one(ids):
        gate.wait(timeout=30)
        return completion(engine, url, ids, count, timeout, return_token_ids=return_token_ids, sampling=sampling)
    with concurrent.futures.ThreadPoolExecutor(len(prompts)) as pool:
        pending = [pool.submit(one, ids) for ids in prompts]
        start = time.monotonic()
        gate.wait(timeout=30)
        results = [f.result() for f in pending]
        elapsed = time.monotonic() - start
    return {"seconds": elapsed, "requests": results}


def run(args):
    bank = json.loads(args.prompts.read_text())
    env = json.loads(args.environment.read_text())
    model = MANIFEST["models"][args.checkpoint]
    sampling = sampling_for(model["family"], getattr(args, "sampling_profile", SAMPLING_PROFILE))
    if args.output.exists() and json.loads(args.output.read_bytes()).get("sampling") != sampling:
        raise ValueError("Use a fresh output path to preserve the timing with a different sampler")
    verify_prompt_bank(bank, model["family"], args.domain)
    if not env.get("gpu_names") or not env.get("engine_revision") or not env.get("server_argv"):
        raise ValueError("Missing hardware, engine revision or launch configuration")
    data = bank["domains"][args.domain]
    batch = args.batch
    rows = data["rows"][:batch]
    uuids = [device['uuid'] for device in env['devices']]
    graph_file = getattr(args, 'graph_dir', None)
    stock = env.get('llama_policy') == 'default'
    if stock and args.engine != 'llama.cpp':
        raise ValueError('Incorrect engine for the default llama.cpp measurement')
    normal = env.get('kai_policy') == 'default'
    if normal and (args.engine != 'kaiLLM' or args.mode != 'plain' or len(uuids) != 1):
        raise ValueError('Unexpected deployment for a normal kaiLLM dispatch control')
    empirical = 'empirical_capacity_harness' in env
    if empirical:
        empirical_capacity.scope(args.engine, args.mode, batch, len(uuids), normal)
        harness = empirical_capacity.source_lock()
        if harness != env['empirical_capacity_harness']:
            raise ValueError('Empirical capacity worker differs from recorded source')
        empirical_capacity.validate_launch(env['server_argv'], env['effective_command'])
    gpu_draft = not stock and gpu_draft_mode(args.engine, args.mode)
    draft_uuids = env.get('cuda_graph_drafter_devices', [])
    reference_graph_waves = {}
    checked_waves = []
    head_waves = []
    direct_head = hybrid_execution.head_policy(args.engine, args.mode, env.get('benchmark_output_head', 'graph')) == 'direct'
    if direct_head:
        verify_weight_digests(model, env.get('measured_gguf_sha256'))
        pin = full_graph_runtime(args.checkpoint, args.mode, env.get('kai_source'), output_head='direct')
        if pin != (env.get('kai_source'), env.get('engine_revision')):
            raise ValueError('Direct head was not served by its registered runtime')
    if normal:
        verify_weight_digests(model, env['measured_gguf_sha256'])
        if kai_default.runtime(MANIFEST, args.checkpoint, args.mode, env['kai_source']) != (env['kai_source'], env['engine_revision']):
            raise ValueError('Incorrect runtime for the normal dispatch measurement')
        kai_default.launch(env['effective_command'], model['family'])
        def read_graphs():
            return kai_default.read(graph_file, uuids, model['family'])
        def target_interval(label, before, after, phases):
            return kai_default.interval(before, after, phases)
        prior = read_graphs()
    elif args.engine == 'kaiLLM':
        def read_graphs():
            return hybrid_execution.read_direct(graph_file, uuids) if direct_head else snapshot(graph_file, uuids)
        draft_interval = verify_draft_interval
        head_uuids = env.get('direct_head_devices' if direct_head else 'cuda_graph_head_devices', [])
        def target_interval(label, before, after, phases):
            backbone = verify_interval(before, after, phases)
            head = (hybrid_execution.direct_interval(before, after, phases, head_uuids, backbone) if direct_head
                    else verify_head_interval(before, after, phases, head_uuids))
            if label != 'warmup':
                head_waves.append((label, head))
            return backbone
        prior = read_graphs()
        expected = output_head_devices(uuids, pipeline_stages(env['server_argv']))
        observed = (hybrid_execution.direct_devices(prior, uuids, env['server_argv']) if direct_head
                    else [uid for uid in uuids if prior[uid]['head_sizes']])
        if head_uuids != expected or head_uuids != observed or (direct_head and env.get('cuda_graph_head_devices')):
            raise ValueError('Target output-head devices differ from the observed graph inventory')
        actual_draft = [uid for uid in uuids if prior[uid]['draft_captures']]
        if actual_draft != draft_uuids or bool(actual_draft) != gpu_draft:
            raise ValueError('GPU proposer graph inventory does not match requested deployment')
    elif stock:
        import llama_default
        verify_weight_digests(model, env.get('measured_gguf_sha256'))
        witness = env['llama_process']
        microbatch = int(env['server_argv'][env['server_argv'].index('-ub') + 1])
        attempts = env['llama_startup_attempts']
        if not llama_default.valid_attempts(attempts, microbatch):
            raise ValueError('Unverified llama.cpp microbatch fallback')
        def read_graphs():
            return llama_default.process(witness['pid'], env['server_argv'], env['llama_source'], uuids)
        def target_interval(label, before, after, phases):
            if before != witness or after != witness:
                raise ValueError('llama.cpp process, loaded libraries, or upstream execution policy changed within a wave')
            if label != 'warmup':
                checked_waves.append(label)
        prior = read_graphs()
        if prior != witness:
            raise ValueError('llama.cpp launch provenance changed before first request')
    else:
        import llama_graph
        draft_interval = llama_graph.verify_draft_interval
        if (gpu_draft and draft_uuids != uuids) or (not gpu_draft and draft_uuids):
            raise ValueError('llama.cpp neural draft rank configuration does not match the selected devices')
        def read_graphs():
            return llama_graph.snapshot(graph_file)
        def target_interval(label, before, after, phases):
            counts = llama_graph.verify_interval(before, after, phases, uuids)
            if label != 'warmup':
                reference_graph_waves[label] = llama_graph.delta(before, after)
            return {phase: {uid: row['graph_launches'] for uid, row in devices.items()}
                    for phase, devices in counts.items()}
        prior = read_graphs()
    # A separate untimed request primes prefill and decode kernels without sharing a timed prefix.
    if empirical:
        initial = empirical_capacity.snapshot(args.url)
    completion(args.engine, args.url, data["rows"][32], 32, args.timeout, sampling=sampling)
    if empirical:
        counts = empirical_capacity.settled(args.url, initial, 1, 32)
        scheduler_waves = {}
    current = read_graphs()
    target_interval('warmup', prior, current, ('prefill', 'decode'))
    if gpu_draft:
        draft_interval(prior, current, draft_uuids, True)
    prefill, graph_waves, draft_waves = [], [], []
    for i in range(3):
        prior = current
        prefill.append(wave(args.engine, args.url, rows, 1, args.timeout, sampling=sampling))
        current = read_graphs()
        if empirical:
            after_counts = empirical_capacity.settled(args.url, counts, 2, 1)
            scheduler_waves[f'prefill-{i+1}'] = {'before':counts,'after':after_counts}
            counts = after_counts
        graph_waves.append((f'prefill-{i+1}', target_interval(f'prefill-{i+1}', prior, current, ('prefill',))))
        if gpu_draft:
            draft_waves.append((f'prefill-{i+1}', draft_interval(prior, current, draft_uuids, False)))
    count_key = None
    if args.engine == "kaiLLM":
        count_key = "kai_dspark" if args.mode in DSPARK_TOKENS else {"ngram1": "kai_ngram", "ngram3": "kai_ngram"}.get(args.mode)
    if count_key:
        from verify_inkling_drafter import counters
        before = counters(args.url.rstrip("/"), required=True)
    full = wave(args.engine, args.url, rows, OUTPUT, args.timeout, sampling=sampling)
    end = read_graphs()
    graph_waves.append(('full', target_interval('full', current, end, ('prefill', 'decode'))))
    if empirical:
        after_counts = empirical_capacity.settled(args.url, counts, 2, OUTPUT)
        scheduler_waves['full'] = {'before':counts,'after':after_counts}
        dispatch = dict(graph_waves)['full']
        empirical_proof = {'protocol':empirical_capacity.CONTRACT, 'harness_sha256':harness, 'waves':scheduler_waves,
                           'overlap':empirical_capacity.overlap(dispatch)}
        empirical_capacity.validate_evidence(empirical_proof, dispatch)
    if gpu_draft:
        draft_waves.append(('full', draft_interval(current, end, draft_uuids, True)))
    if count_key:
        after = counters(args.url.rstrip("/"), required=True)
        proposed = after["draft_tokens"] - before["draft_tokens"]
        accepted = after["accepted_tokens"] - before["accepted_tokens"]
        if not 0 <= accepted <= proposed or proposed == 0:
            raise ValueError(f"{args.mode} requires a positive full-wave draft-token count and valid accepts")

    pp_seconds = statistics.median(x["seconds"] for x in prefill)
    decode_seconds = full["seconds"] - pp_seconds
    if not math.isfinite(decode_seconds) or min(pp_seconds, decode_seconds) <= 0:
        raise ValueError("Invalid timing interval")
    record = {"protocol": PROTOCOL, "checkpoint": args.checkpoint, "model_revision": model["revision"],
              "sha256": [f["sha256"] for f in model["files"].values()], "engine": args.engine,
              "mode": args.mode, "domain": args.domain, "batch": batch, "prompt_tokens": PROMPT,
              "generated_tokens": OUTPUT, "prompt_hashes": data["hashes"][:batch], "environment": env,
              "prefill_tps": batch * PROMPT / pp_seconds,
              "decode_tps": batch * (OUTPUT - 1) / decode_seconds, "prefill": prefill, "full": full,
              "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if sampling is not None:
        record["sampling"] = sampling
    if stock:
        record['llama_default'] = {'process': witness, 'wave_checks': checked_waves,
                                   'microbatch': microbatch, 'startup_attempts': attempts,
                                   'measured_gguf_sha256': env['measured_gguf_sha256'],
                                   'speculative': llama_default.counts(full, args.mode)}
    elif normal:
        record['kai_default_dispatch'] = dict(graph_waves)
        if empirical:
            empirical_capacity.check_requests(record)
            record['kai_empirical_capacity'] = empirical_proof
    else:
        record['cuda_graphs'] = proof_record(graph_waves)
    if direct_head:
        record["target_head_direct"] = hybrid_execution.direct_record(head_waves)
    elif head_waves:
        record["target_head_cuda_graphs"] = proof_record(head_waves)
    if reference_graph_waves:
        record["llama_cuda_graph_intervals"] = reference_graph_waves
    if gpu_draft:
        record["drafter_cuda_graphs"] = draft_proof_record(draft_waves)
    if count_key:
        record[count_key] = {"scope": "full-wave", "proposed": proposed, "accepted": accepted,
                                "rejected": proposed - accepted}
    if count_key == "kai_ngram":
        record[count_key].update(lookup_min=2, lookup_max=8, draft_tokens=DRAFT_TOKENS[args.mode])
    write_json(args.output, record)
    print(f"{args.engine} {args.checkpoint} {args.domain} B{batch}: "
          f"prefill {record['prefill_tps']:.1f}, decode {record['decode_tps']:.1f} tok/s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("family", choices=MANIFEST["tokenizers"])
    p.add_argument("output", type=Path)
    b = sub.add_parser("run")
    b.add_argument("--engine", required=True, choices=("kaiLLM", "llama.cpp"))
    b.add_argument("--checkpoint", required=True, choices=MANIFEST["models"])
    b.add_argument("--url", required=True)
    b.add_argument("--prompts", type=Path, required=True)
    b.add_argument("--domain", required=True, choices=("prose", "code"))
    b.add_argument("--sampling-profile", choices=SAMPLING_PROFILES, default=SAMPLING_PROFILE)
    b.add_argument("--batch", required=True, choices=BATCHES, type=int)
    b.add_argument("--mode", choices=DRAFT_TOKENS, required=True)
    b.add_argument("--environment", type=Path, required=True)
    b.add_argument("--graph-dir", type=Path, help="Counter directory for a strict graph run")
    b.add_argument("--timeout", type=int, default=43200)
    b.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    if a.action == "prompts":
        prompt_bank(a.family, a.output)
    else:
        run(a)
