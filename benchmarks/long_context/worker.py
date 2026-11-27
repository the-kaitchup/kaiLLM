"""Launch one engine configuration, check full-context capacity, and time the selected domains."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import hybrid_execution
import empirical_capacity
import kai_default
from devices import is_gb10, shared_memory_inventory
from benchmark import BATCHES, DRAFT_TOKENS, DSPARK_TOKENS, MANIFEST, OUTPUT, PROMPT, SAMPLING_PROFILE, SAMPLING_PROFILES, dspark_asset, dspark_digest, full_graph_runtime, inkling_draft_runtime, inkling_fused_q8, inkling_native_placement, placement_resources, kai_loader_runtime, llama_revision, run, sampling_for, timing_domains, verify_prompt_bank, wave, write_json
from cuda_graphs import ENV as GRAPH_DIR, NEURAL_MODES, gpu_draft_mode, output_head_devices, pipeline_stages, snapshot, validate_kai_launch

CONTEXT = PROMPT + OUTPUT
GIB = 1024 ** 3


def gpu_inventory():
    fields = "name,uuid,driver_version,compute_cap,memory.total,memory.free,power.limit"
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"], text=True)
    return [dict(zip(fields.split(","), (v.strip() for v in row), strict=True))
            for row in csv.reader(raw.splitlines())]


def selected_devices(inventory, indices, visible):
    if not visible:
        raise ValueError("No CUDA device is visible to the benchmark")
    if indices is None:
        match = next((d for d in inventory if d["uuid"] == visible[0]), None)
        if match is None:
            raise ValueError("First visible CUDA device is missing from the nvidia-smi inventory")
        return [shared_memory_inventory(match)]
    if len(set(indices)) != len(indices) or any(i < 0 or i >= len(inventory) for i in indices):
        raise ValueError("Select distinct GPUs from the nvidia-smi inventory")
    devices = [inventory[i] for i in indices]
    if any(d["uuid"] not in visible for d in devices):
        raise ValueError("Requested GPU is outside the process's visible CUDA devices")
    return [shared_memory_inventory(device) for device in devices]


def admission_fraction(devices):
    """Leave 256 MiB below observed free memory for CUDA initialization and display use."""
    free_ratio = min((float(d["memory.free"]) - 256) / float(d["memory.total"]) for d in devices)
    if free_ratio <= 0:
        raise ValueError("Insufficient free GPU memory for CUDA initialization")
    return min(0.94, math.floor(free_ratio * 10000) / 10000)


def kv_shortfall(logs):
    matches = re.findall(r"([\d.]+) GiB KV cache is needed, which is larger than the available "
                         r"KV cache memory \(([\d.]+) GiB\)", logs)
    return max((float(needed), float(available)) for needed, available in matches) if matches else None


def startup_tail(logs):
    lines = [s for s in logs[-12000:].splitlines() if re.search(r"error|failed|cache|context|memory", s, re.I)]
    return "Server startup failed: " + " ".join(lines[-4:])[-1200:]


def startup_failure(logs):
    arch = re.search(r"unknown model architecture: '([a-zA-Z0-9_.-]+)'", logs)
    if arch:
        return f"Unknown GGUF architecture in the measured engine: {arch[1]}"
    if re.search(r"CUDA (?:error: )?out of memory|torch\.OutOfMemoryError:|failed to allocate CUDA\d+ buffer", logs, re.I):
        return "Server startup failed: CUDA out of memory"
    shortage = kv_shortfall(logs)
    if shortage:
        return f"Engine requires {shortage[0]:g} GiB of KV for one full sequence; {shortage[1]:g} GiB was available"
    return startup_tail(logs)


def wait_server(proc, url, log):
    until = time.monotonic() + 1500
    while proc.poll() is None and time.monotonic() < until:
        try:
            with urllib.request.urlopen(url + "/health", timeout=3) as res:
                if res.status == 200:
                    return
        except OSError:
            pass
        time.sleep(3)
    raise RuntimeError(startup_failure(log.read_text(errors="replace")))


def capacity_estimate(logs):
    matches = re.finditer(r"(?:GPU KV cache size: ([\d,]+) tokens, )?Maximum concurrency for "
                          r"([\d,]+) tokens per request: (\d+(?:\.\d+)?)x", logs)
    estimates = []
    for match in matches:
        total, length, slots = match.groups()
        capacity = float(slots)
        context = int(length.replace(",", ""))
        if total and context:
            ratio = int(total.replace(",", "")) / context
            # Hybrid KV layouts can show a token count for a different group.
            if abs(ratio - capacity) <= .005000001:
                estimates.append((ratio, True))
                continue
        estimates.append((capacity, False))
    return min(estimates, default=(None, False), key=lambda entry: entry[0])


def reported_capacity(logs):
    return capacity_estimate(logs)[0]


def stop_server(proc):
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=50)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def next_kv_allocation(logs, current, batch, ceiling, oom):
    block = GIB // 16
    capacity, exact = capacity_estimate(logs)
    if oom:
        if capacity is None or capacity <= batch:
            return None
        # Allow room for graph capture while retaining the required sequence slots.
        target = current * batch * 11 / (capacity * 10)
        sized = math.ceil(target / block) * block
        if sized < current:
            return sized
        # A narrowly overprovisioned pool can still crowd out a CUDA graph.
        # Use the engine's token counts when its display rounds the slot ratio.
        precise = capacity if exact else capacity - .005
        if precise <= batch:
            return None
        granularity = GIB // 128
        minimum = math.ceil(current * batch / precise / granularity) * granularity + granularity
        return minimum if minimum < current else None
    shortage = kv_shortfall(logs)
    if shortage:
        needed, available = shortage
        # Account for KV bytes the engine deducted before its availability check.
        # vLLM reports both GiB quantities to two decimal places.
        target = (needed + .005) * GIB * batch + max(0, current - (available - .005) * GIB)
    elif capacity is not None and 0 < capacity < batch:
        target = current * batch / capacity
    else:
        return None
    sized = min(math.ceil(target / block) * block, int(ceiling // block) * block)
    return sized if sized > current else None


def start_server(cmd, env, url, log, batch, automatic_kv, ceiling):
    attempted = set()
    evidence_env = kai_default.ENV if kai_default.ENV in env else GRAPH_DIR
    graph_root = env.get(evidence_env)
    for attempt in range(3):
        if graph_root:
            directory = Path(graph_root) / f'attempt-{attempt+1}'
            directory.mkdir()
            env[evidence_env] = str(directory)
        with log.open("w") as stream:
            proc = subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            wait_server(proc, url, log)
            if automatic_kv:
                capacity = reported_capacity(log.read_text(errors="replace"))
                if capacity is not None and capacity < batch:
                    raise RuntimeError(f"Full-context KV capacity {capacity:g} sequences < batch {batch}")
            return proc
        except BaseException as exc:
            stop_server(proc)
            if attempt == 2 or not automatic_kv or not isinstance(exc, RuntimeError):
                raise
            index = cmd.index("--kv-cache-memory-bytes") + 1
            original = int(cmd[index])
            attempted.add(original)
            allocation = next_kv_allocation(log.read_text(errors="replace"), original, batch, ceiling,
                                            "CUDA out of memory" in str(exc))
            if allocation is None or allocation in attempted:
                raise
            log.replace(log.with_suffix(f".attempt{attempt + 1}.log"))
            cmd[index] = str(allocation)
            print(f"Retrying batch {batch} with {allocation / GIB:g} GiB of KV after {exc}", flush=True)


def keyname(model, engine, mode, batch):
    return f"{model}-{engine}-{mode}-b{batch}"


def serving_defaults():
    source = Path(__file__).resolve().parents[2] / "runtime-source/vllm_gguf_plugin/serving_defaults.py"
    spec = importlib.util.spec_from_file_location("bench_gguf_defaults", source)
    defaults = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(defaults)
    return defaults


def checkpoint_files(root, name, spec):
    directory = root / spec["repo"] / spec["revision"]
    expected = {key: spec[key] for key in ("repo", "revision", "files")}
    receipts = [directory / ("download-" + name + ".json"), directory / "download.json"]
    for receipt in receipts:
        if receipt.exists():
            data = json.loads(receipt.read_bytes())
            if data.get("source") == "huggingface.co" and all(data.get(k) == v for k, v in expected.items()):
                break
    else:
        raise ValueError("Verified Hugging Face receipt required for " + name)
    files = [directory / filename for filename in spec["files"]]
    if any(p.stat().st_size != spec["files"][p.name]["bytes"] for p in files):
        raise ValueError("Checkpoint size mismatch for " + name)
    return files


def main(a):
    from devices import physical_id, visible_cuda_uuids

    head_policy = hybrid_execution.head_policy(a.engine, a.mode, getattr(a, "output_head", "graph"))
    direct_head = head_policy == "direct"
    normal = getattr(a, 'kai_policy', 'strict') == 'default'
    if normal and (a.engine != 'kaiLLM' or a.mode != 'plain' or direct_head):
        raise ValueError('The normal kaiLLM dispatch control serves plain decoding with the usual output head')
    stock = getattr(a, 'llama_policy', 'strict') == 'default'
    if stock and a.engine != 'llama.cpp':
        raise ValueError('The default llama.cpp policy requires a llama.cpp reference run')
    devices = selected_devices(gpu_inventory(), a.devices, visible_cuda_uuids())
    empirical = getattr(a, 'empirical_b2_capacity', False)
    if empirical:
        empirical_capacity.scope(a.engine, a.mode, a.batch, len(devices), normal)
        if a.kv_gib is None or not math.isfinite(a.kv_gib) or a.kv_gib <= 0:
            raise ValueError('Select an explicit positive --kv-gib for empirical B2 capacity')
        harness = empirical_capacity.source_lock()
    spec = MANIFEST["models"][a.checkpoint]
    dspark = a.mode in DSPARK_TOKENS
    if dspark and spec['family'] != 'inkling':
        raise ValueError('DSpark requires an Inkling checkpoint')
    sampling_profile = getattr(a, "sampling_profile", SAMPLING_PROFILE)
    sampling = sampling_for(spec["family"], sampling_profile)
    if empirical and spec['family']=='lfm':
        raise ValueError('The ordinary LFM3 comparison requires nominal full-context capacity')
    files = checkpoint_files(a.weights, a.checkpoint, spec)
    a.out.mkdir(parents=True, exist_ok=True)
    key = keyname(a.checkpoint, a.engine, a.mode, a.batch)
    if empirical:
        key += f'-empirical-{round(a.kv_gib*GIB)}'
    if stock:
        key += '-llama-default'
    if normal:
        key += '-kai-default'
    scheduled_domains = timing_domains(a.mode)
    domains = dict.fromkeys(getattr(a, "domain", None) or scheduled_domains)
    if set(domains).difference(scheduled_domains):
        raise ValueError("Plain decoding timings use code only")
    outputs = {d: a.out / f"{key}-{d}.json" for d in domains}
    for domain, path in outputs.items():
        if not path.exists():
            continue
        old = json.loads(path.read_text())
        if old.get('sampling') != sampling:
            raise ValueError('Use a fresh output directory to preserve timings with other sampling settings')
        hardware = old.get('environment', {}).get('devices', [])
        if empirical != ('kai_empirical_capacity' in old):
            raise ValueError('Use a fresh empirical capacity result directory')
        if empirical:
            empirical_capacity.check_requests(old)
            proof = old['kai_empirical_capacity']
            old_env = old['environment']
            if old_env.get('empirical_capacity_harness') != harness or proof.get('harness_sha256') != harness:
                raise ValueError('Empirical capacity harness changed between attempts')
            empirical_capacity.validate_launch(old_env['server_argv'], old_env['effective_command'])
            actual_size = int(old_env['server_argv'][old_env['server_argv'].index('--kv-cache-memory-bytes')+1])
            if actual_size != round(a.kv_gib*GIB):
                raise ValueError('Empirical capacity retry used a different KV allocation')
            empirical_capacity.validate_evidence(proof, old['kai_default_dispatch']['full'])
        if ((a.engine == "kaiLLM" and a.mode.startswith("ngram") and not old.get("kai_ngram"))
                or (stock != ('llama_default' in old)) or (normal != ('kai_default_dispatch' in old))
                or (a.engine == 'kaiLLM' and old['environment'].get('kai_source') != a.kai_commit)
                or (not (stock or normal) and not old.get('cuda_graphs')) or (a.engine == 'kaiLLM' and not normal and not old.get('target_head_direct' if direct_head else 'target_head_cuda_graphs'))
                or old.get('environment', {}).get('benchmark_output_head', 'graph') != head_policy
                or (direct_head and (old.get('target_head_cuda_graphs') or old['environment'].get('kai_source') != a.kai_commit))
                or (not stock and gpu_draft_mode(a.engine, a.mode) and not old.get('drafter_cuda_graphs'))
                or (old.get('checkpoint'), old.get('engine'), old.get('mode'), old.get('batch'), old.get('domain'))
                    != (a.checkpoint, a.engine, a.mode, a.batch, domain)
                or old.get('model_revision') != spec['revision']
                or (dspark and 'dspark' in spec
                    and old['environment'].get('measured_drafter_sha256') != dspark_digest(spec))
                or [d.get('uuid') for d in hardware] != [d['uuid'] for d in devices]):
            raise ValueError('Use a fresh output directory to preserve an unmatched or incomplete historical timing')
    if all(p.exists() for p in outputs.values()):
        return
    measured_digests = hybrid_execution.verify_files(files, spec) if direct_head or stock or normal else None
    measured_draft = None
    log = a.out / (key + ".log")
    stamp = a.out / (key + ".status.json")
    env = os.environ.copy()
    env.pop(hybrid_execution.HEAD_ENV, None)
    env.pop(kai_default.ENV, None)
    env.pop(GRAPH_DIR, None)
    if normal:
        if len(devices) != 1:
            raise ValueError('This normal-dispatch control requires one GPU')
        for override in ('KAILLM_INKLING_FULL_GRAPHS','KAILLM_PREFILL_GRAPHS'):
            env.pop(override, None)
    if direct_head:
        env[hybrid_execution.HEAD_ENV] = head_policy
    # CUDA and nvidia-smi can enumerate the same cards in different orders.
    env["CUDA_VISIBLE_DEVICES"] = ",".join(d["uuid"] for d in devices)
    env["OMP_NUM_THREADS"] = str(min(8, max(1, len(os.sched_getaffinity(0)) // 2)))
    ampere_qwen = (a.engine == "kaiLLM" and spec["family"] in ("daisy", "qwen")
                   and spec["kind"] != "chimachima" and all(d["compute_cap"] == "8.6" for d in devices))
    if ampere_qwen:
        env["KAILLM_ALLOW_QWEN35_SM86"] = "1"
    gpu_names = [d['name'] for d in devices]
    board = physical_id(d['uuid'] for d in devices)
    placement = (inkling_native_placement(spec, a.batch, gpu_names, a.kai_commit, hardware_id=board,
                                          draft_tokens=DSPARK_TOKENS.get(a.mode, 5))
                 if a.engine == 'kaiLLM' and (a.mode == 'plain' or dspark) else None)
    native_pin = (inkling_draft_runtime(spec, a.batch, len(devices), gpu_names=gpu_names, revision=a.kai_commit, hardware_id=board,
                                       draft_tokens=DSPARK_TOKENS.get(a.mode, 5))
                  if a.engine == "kaiLLM" else None)
    env.pop('KAILLM_INKLING_PREFILL_BIAS_MIB', None)
    env.pop('KAILLM_INKLING_B1_COMPACT_GRAPHS', None)
    env.pop('KAILLM_INKLING_SERIAL_VERIFY_POST', None)
    env.pop('KAILLM_INKLING_SERIAL_VERIFY_BIAS', None)
    env.pop('KAILLM_INKLING_SERIAL_VERIFY_ROUTE', None)
    env.pop('KAILLM_INKLING_SERIAL_VERIFY_COMBINE', None)
    env.pop('KAILLM_INKLING_VERIFY_COMBINE_FMA4', None)
    env.pop('KAILLM_INKLING_CHIMA_XS_ROW_STABLE_DS5', None)
    env.pop('KAILLM_ITQ_MOE_DECODE_ROWS', None)
    env.pop('KAILLM_ITQ_MOE_REPORT_WORKSPACE', None)
    env.pop('KAILLM_ITQ_MOE_SHARE_PREFILL_WORKSPACE', None)
    env.pop('KAILLM_INKLING_PACKED_VERIFY_ATTN', None)
    env.pop('KAILLM_INKLING_PACKED_VERIFY_ATTN_RECEIPT', None)
    row_stable_dspark = bool(placement and dspark and placement.get('row_stable_dspark'))
    prefill_tokens = placement['prefill_tokens'] if row_stable_dspark else 2048
    itq_decode_rows = placement.get('itq_moe_decode_rows') if row_stable_dspark else None
    shared_itq_workspace = bool(row_stable_dspark and placement.get('itq_moe_share_prefill_workspace'))
    if itq_decode_rows is not None:
        expected_cutoff = 512 if shared_itq_workspace else 8 * prefill_tokens
        if type(itq_decode_rows) is not int or itq_decode_rows != expected_cutoff:
            raise ValueError('The ITQ prefill cutoff differs from its qualified geometry')
        env['KAILLM_ITQ_MOE_DECODE_ROWS'] = str(itq_decode_rows)
    if shared_itq_workspace:
        if prefill_tokens != 256:
            raise ValueError('The shared ITQ workspace requires its verified prefill placement')
        env['KAILLM_ITQ_MOE_SHARE_PREFILL_WORKSPACE'] = '1'
    packed_verify_attention = bool(row_stable_dspark and placement.get('packed_verify_attention'))
    if packed_verify_attention:
        env['KAILLM_INKLING_PACKED_VERIFY_ATTN'] = '1'
    if row_stable_dspark:
        env['KAILLM_INKLING_CHIMA_XS_ROW_STABLE_DS5'] = '1'
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    if placement:
        qualified_kv_mib, capture_sizes = placement_resources(placement, a.mode)
        if placement.get('prefill_bias_mib') is not None:
            env['KAILLM_INKLING_PREFILL_BIAS_MIB'] = str(placement['prefill_bias_mib'])
        if placement.get('serial_verify_combine'):
            env['KAILLM_INKLING_SERIAL_VERIFY_COMBINE'] = '1'
        if placement.get('verify_combine_fma4'):
            env['KAILLM_INKLING_VERIFY_COMBINE_FMA4'] = placement['verify_combine_fma4']
        if placement.get('serial_verify_route'):
            env['KAILLM_INKLING_SERIAL_VERIFY_ROUTE'] = '1'
        if placement.get('serial_verify_bias'):
            env['KAILLM_INKLING_SERIAL_VERIFY_BIAS'] = '1'
        if placement.get('serial_verify_post'):
            env['KAILLM_INKLING_SERIAL_VERIFY_POST'] = '1'
        if placement.get('compact_graphs'):
            env['KAILLM_INKLING_B1_COMPACT_GRAPHS'] = '1'
        measured_digests = hybrid_execution.verify_files(files, spec)
    graph_pin = full_graph_runtime(a.checkpoint, a.mode, a.kai_commit, output_head=head_policy, batch=a.batch, gpu_names=gpu_names, hardware_id=board) if a.engine == "kaiLLM" else None
    if normal:
        graph_pin = kai_default.runtime(MANIFEST, a.checkpoint, a.mode, a.kai_commit)
    deterministic_humming = (native_pin is not None or (graph_pin is not None and spec["family"] == "inkling")
                             or (a.engine == "kaiLLM" and spec.get("kai_humming_deterministic", False)))
    if deterministic_humming:
        env["KAILLM_HUMMING_DETERMINISTIC"] = "1"
    else:
        env.pop("KAILLM_HUMMING_DETERMINISTIC", None)
    fused_q8 = inkling_fused_q8(a.checkpoint, a.mode, a.batch, [d['name'] for d in devices], graph_pin)
    if fused_q8:
        env['KAILLM_INKLING_FUSED_Q8'] = '1'
    else:
        env.pop('KAILLM_INKLING_FUSED_Q8', None)
    stage = "preflight"
    proc = None
    llama_source = llama_revision(spec)
    try:
        tokens = DRAFT_TOKENS[a.mode]
        dspine = a.mode.startswith("dspine")
        ngram = a.mode.startswith("ngram")
        if ngram and (spec["family"] != "lfm" or tokens != max(0, min(3, 16 // a.batch - 1))):
            raise ValueError("LFM3 n-gram draft length must follow its batch-dependent serving policy")
        if a.engine == "kaiLLM" and graph_pin is None:
            raise ValueError("This model and mode require a registered full-graph kaiLLM runtime before serving")
        loader = kai_loader_runtime(spec, len(devices)) if a.engine == "kaiLLM" else None
        loader = graph_pin or native_pin or loader
        if loader:
            if a.runtime is None or (a.kai_commit, (a.runtime / "benchmark-source.sha256").read_text().strip()) != loader:
                raise ValueError("This GGUF requires its pinned kaiLLM loader revision and source lock")
        if not loader and a.engine == "kaiLLM" and spec["family"] in ("daisy", "qwen") and spec["kind"] != "chimachima":
            names = serving_defaults().read_header(files[0])[1]
            if "blk.0.attn_qkv_gate.weight" in names and "blk.0.attn_qkv.weight" not in names:
                raise ValueError("The pinned kaiLLM loader requires separate QKV and gate tensors for this GGUF")
        if a.mode == "mtp5" and not serving_defaults().inspect(files[0])["mtp_layers"]:
            raise ValueError("GGUF has no native MTP tensor head")
        if dspine or dspark:
            auxiliary, ref = ("dspine", MANIFEST["dspine"]) if dspine else dspark_asset(spec)
            directory = a.weights / ref["repo"] / ref["revision"]
            receipt = json.loads((directory / ("download-" + auxiliary + ".json")).read_text())
            if receipt != {"source": "huggingface.co", "repo": ref["repo"], "revision": ref["revision"], "files": ref["files"]}:
                raise ValueError("Drafter receipt differs from the pinned version")
            path = directory if dspine else directory / next(iter(ref["files"]))
            if a.drafter is not None and a.drafter.resolve() != path.resolve():
                raise ValueError("Drafter must match the pinned Hugging Face download")
            a.drafter = path
            if dspark:
                measured_draft, = hybrid_execution.verify_files([path], ref)
        if a.engine == "kaiLLM":
            if a.runtime is None or a.python is None:
                raise ValueError("Set --runtime and --python for the merged kaiLLM runtime")
            spark = (len(devices) == 1 and is_gb10(devices[0]) and spec["family"] == "inkling"
                     and (a.mode == "plain" or (dspark and placement is not None)))
            if not spark and any(d["compute_cap"] not in ("8.6", "12.0") for d in devices):
                raise ValueError("No kaiLLM release native catalog for compute capability " + devices[0]["compute_cap"])
            if ngram and serving_defaults().inspect(files[0])["speculation"] != "ngram":
                raise ValueError("GGUF does not select n-gram drafting")
            if dspine and (a.batch != 1 or spec["family"] not in ("qwen", "daisy")):
                raise ValueError("DSpine requires one slot and a Qwen3.8-family target")
            if dspark and not native_pin:
                raise ValueError("Native DSpark requires a qualified target, batch and GPU count")
            if ((spec["family"] == "inkling" and a.mode != "plain" and not dspark)
                    or (spec["family"] == "hy4" and a.mode != "plain")):
                raise ValueError("The merged adapter has no native draft verifier for this family")
            env["KAILLM_NATIVE_ASSETS"] = str(a.runtime / "runtime-native-cache/native-assets.json")
            free = min(float(d["memory.free"]) * 2**20 for d in devices)
            weight_bytes = sum(p.stat().st_size for p in files)
            if spec["family"] == "hy4":
                # Compute the resident size from the tensor directory; the table stays in host RAM.
                sys.path.insert(0, str(a.runtime / "gguf-reader"))
                import gguf
                table_bytes = sum(t.n_bytes for p in files for t in gguf.GGUFReader(str(p)).tensors
                                  if t.name == "per_layer_token_embd.weight")
                if not table_bytes:
                    raise ValueError("Missing Hy4 host Engram tensor")
                weight_bytes -= table_bytes
            reserve = max(3 * GIB, free * .10) + (3 * GIB if dspine else GIB if dspark else 0)
            if placement:
                qualified_kv = qualified_kv_mib * 2**20
                if a.kv_gib is not None and a.kv_gib * GIB != qualified_kv:
                    raise ValueError('Requested KV pool differs from the qualified native placement')
                kv = qualified_kv
            elif a.kv_gib is None:
                kv = max(GIB, int((free - weight_bytes / len(devices) - reserve) // GIB) * GIB)
            else:
                if not math.isfinite(a.kv_gib) or a.kv_gib <= 0:
                    raise ValueError("KV allocation must be a positive number of GiB")
                kv = round(a.kv_gib * GIB)
            fraction = admission_fraction(devices)
            if spark:
                fraction = min(.24, fraction)
                if kv + weight_bytes + reserve > free:
                    raise ValueError("Spark weights, KV pool and startup reserve exceed measured shared-memory headroom")
            extra = ["--enable-prompt-tokens-details", "--enable-per-request-metrics"]
            if row_stable_dspark:
                if capture_sizes != [*range(1, tokens + 2), prefill_tokens] or a.batch != 1:
                    raise ValueError('Row-stable Inkling requires every verifier width and its qualified prefill graph')
                graph = dict(mode=0, cudagraph_mode='FULL', cudagraph_capture_sizes=capture_sizes,
                             max_cudagraph_capture_size=prefill_tokens)
                extra += ['--no-async-scheduling', '--compilation-config', json.dumps(graph)]
            if len(devices) > 1:
                if spec["family"] != "hy4":
                    extra += ["--tensor-parallel-size", str(len(devices))]
                elif not spec.get("kai_pipeline_runtime"):
                    raise ValueError("Hy4 multi-GPU serving requires a qualified pipeline runtime")
            cmd = [str(a.python), str(a.runtime / "deployment/serve.py"), str(files[0]),
                   "--max-num-seqs", str(a.batch), "--max-model-len", str(CONTEXT),
                   "--kv-cache-memory-bytes", str(kv), "--gpu-memory-utilization", str(fraction),
                   "--max-num-batched-tokens", str(prefill_tokens), "--port", str(a.port), "--served-model-name", "bench",
                   "--spec-tokens", str(tokens), "--extra-vllm-args", json.dumps(extra)]
            if len(devices) > 1 and spec.get("kai_pipeline_runtime"):
                cmd += ["--pipeline-parallel-size", str(len(devices))]
            if ngram:
                cmd += ["--lookup-min", "2", "--lookup-max", "8"]
            if dspine:
                cmd += ["--dspine", str(a.drafter)]
            elif dspark:
                cmd += ["--dspark", str(a.drafter)]
            if spec["family"] == "hy4":
                # Hy4's model adapter creates the correct GGUF tokenizer. Override the generic LFM fallback.
                cmd += ["--tokenizer", str(files[0])]
            revision = (a.runtime / "benchmark-source.sha256").read_text().strip()
        else:
            if spec["kind"] == "chimachima":
                raise ValueError("llama.cpp has no kaiPTQ ITQ tensor loader")
            if a.llama is None:
                raise ValueError("Set --llama to the pinned llama-server executable")
            if dspine:
                raise ValueError("The Tencent kaiLLM DSpine fixture is not a llama.cpp draft GGUF")
            # A speculative verification step temporarily holds draft positions as well as the
            # accepted history. With an exact-capacity slot llama.cpp can stop a few tokens early.
            slot_tokens = CONTEXT + (tokens + 1 if tokens else 0)
            cmd = [str(a.llama), "-m", str(files[0]), "--host", "127.0.0.1", "--port", str(a.port),
                   "--ctx-size", str(slot_tokens * a.batch), "--parallel", str(a.batch), "-ngl", "999",
                   "--fit", "off", "--flash-attn", "on", "--cache-type-k", "bf16", "--cache-type-v", "bf16",
                   "--cache-ram", "0", "--no-context-shift", "-b", "2048", "-ub", "512"]
            if spec["family"] == "hy4":
                cmd += ["--override-tensor", r"^per_layer_token_embd\.weight$=CPU", "--load-mode", "mmap", "--lazy-mode", "on"]
            if a.mode == "mtp5":
                cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "5"]
            elif a.mode == "ngram1":
                # The stock simple lookup returns an empty draft when M < N.
                cmd += ["--spec-type", "ngram-map-k", "--spec-ngram-map-k-size-n", "2",
                        "--spec-ngram-map-k-size-m", "1", "--spec-ngram-map-k-min-hits", "1",
                        "--spec-draft-n-max", "1"]
            elif ngram:
                cmd += ["--spec-type", "ngram-simple", "--spec-ngram-simple-size-n", "2", "--spec-ngram-simple-size-m", str(tokens),
                        "--spec-draft-n-max", str(tokens)]
            elif dspark:
                if a.drafter is None:
                    raise ValueError("A pinned DSpark GGUF is required")
                cmd += ["--spec-type", "draft-dspark", "--model-draft", str(a.drafter), "--spec-draft-n-max", str(tokens)]
            else:
                cmd += ["--spec-type", "none"]
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", a.port))
        stage = "startup"
        url = f"http://127.0.0.1:{a.port}"
        if a.engine == 'kaiLLM':
            dispatch_dir = kai_default.ENV if normal else GRAPH_DIR
            env[dispatch_dir] = tempfile.mkdtemp(prefix=key+'.cuda-', dir=a.out)
            preview = subprocess.run([*cmd, '--dry-run', '--compute-capability', devices[0]['compute_cap']],
                                     env=env, check=True, capture_output=True, text=True)
            effective = json.loads(preview.stdout)['command']
            if normal:
                kai_default.launch(effective, spec['family'])
                if empirical:
                    empirical_capacity.validate_launch(cmd, effective)
            else:
                validate_kai_launch(effective)
        else:
            import llama_graph
            env.pop(GRAPH_DIR, None)
            if stock:
                import llama_default
                graph_file = None
                env = llama_default.prepare(a.llama, llama_source, len(devices), env)
            else:
                graph_root = Path(tempfile.mkdtemp(prefix=key+'.cuda-graphs-', dir=a.out))
                graph_file = graph_root / 'llama-graph.bin'
                env.update(llama_graph.prepare(a.llama, graph_file, llama_source, len(devices), base_env=env))
            revision = subprocess.check_output([str(a.llama), '--version'], env=env, stderr=subprocess.STDOUT, text=True).strip()
            if llama_source[:7] not in revision:
                raise ValueError('llama.cpp binary revision differs from manifest: ' + revision[-200:])
        ceiling = min(float(d["memory.free"]) for d in devices) * 2**20
        if a.engine == "kaiLLM" and spark:
            ceiling -= weight_bytes + reserve
        if stock:
            attempt_dir = Path(tempfile.mkdtemp(prefix=key+'.startup-', dir=a.out))
            # Readiness only covers model initialization. Some CUDA graphs first allocate on a real request.
            domain = next(domain for domain, output in outputs.items() if not output.exists())
            bank = json.loads((a.banks / (spec['family'] + '.json')).read_bytes())
            verify_prompt_bank(bank, spec['family'], domain)
            warmup = bank['domains'][domain]['rows'][32]
            # Each slot gets a distinct start. No timed prompt is used for this allocation probe.
            prompts = [warmup[i:] + warmup[:i] for i in range(a.batch)]
            proc, attempts = llama_default.start_with_microbatches(
                lambda argv: start_server(argv, env, url, log, a.batch, False, ceiling), stop_server, cmd, log, attempt_dir, write_json,
                probe=lambda: wave(a.engine, url, prompts, 32, a.timeout, sampling=sampling))
        else:
            proc = start_server(cmd, env, url, log, a.batch, a.engine == "kaiLLM" and a.kv_gib is None and not placement, ceiling)
        stage = "capacity"
        logs = log.read_text(errors="replace")
        captured, draft_devices, head_devices = {}, [], []
        cap = None
        if a.engine == "kaiLLM":
            graph_file = Path(env[dispatch_dir])
            uuids = [d['uuid'] for d in devices]
            if normal:
                kai_default.read(graph_file, uuids, spec['family'])
            else:
                captured = hybrid_execution.read_direct(graph_file, uuids) if direct_head else snapshot(graph_file, uuids)
                head_devices = [d['uuid'] for d in devices if captured[d['uuid']]['head_sizes']]
                if direct_head:
                    hybrid_execution.direct_devices(captured, uuids, cmd)
                elif head_devices != output_head_devices(uuids, pipeline_stages(cmd)):
                    raise ValueError('Every rank of the final pipeline stage must capture the vocabulary projection')
                draft_devices = [d['uuid'] for d in devices if captured[d['uuid']]['draft_captures']]
                if bool(draft_devices) != gpu_draft_mode(a.engine, a.mode):
                    raise ValueError('Requested draft mode differs from the observed GPU proposer graphs')
            if placement and capture_sizes:
                if captured[uuids[0]]['captures'] != capture_sizes:
                    raise ValueError('Resident native CUDA graphs differ from the qualified shape inventory')
            cap = reported_capacity(logs)
            if cap is None:
                raise ValueError("kaiLLM did not confirm full-context KV capacity")
            if empirical and cap < 1:
                raise ValueError(f"Empirical B2 requires one full nominal sequence; engine reported {cap}")
            if not empirical and cap < a.batch:
                raise ValueError(f"Full-context KV capacity {cap} sequences < batch {a.batch}")
        elif stock:
            witness = llama_default.process(proc.pid, cmd, llama_source, [d['uuid'] for d in devices])
        else:
            llama_graph.snapshot(graph_file)
            if a.mode in NEURAL_MODES:
                draft_devices = [d['uuid'] for d in devices]
        envfile = a.out / (key + ".environment.json")
        write_json(envfile, {"gpu_names": [d["name"] for d in devices], "devices": devices,
                             **({"benchmark_output_head": head_policy, "measured_gguf_sha256": measured_digests,
                                 "direct_head_devices": output_head_devices(uuids, pipeline_stages(cmd))} if direct_head else {}),
                             **({'llama_policy': 'default', 'llama_process': witness, 'llama_startup_attempts': attempts,
                                 'measured_gguf_sha256': measured_digests} if stock else {}),
                             **({'kai_policy': 'default', 'measured_gguf_sha256': measured_digests,
                                 'effective_command': effective} if normal else {}),
                             **({'empirical_capacity_harness':harness} if empirical else {}),
                             **({'kai_inkling_itq_moe_decode_rows': itq_decode_rows} if itq_decode_rows is not None else {}),
                             **({'kai_inkling_itq_moe_shared_workspace': True} if shared_itq_workspace else {}),
                             **({'kai_inkling_packed_verify_attention': True} if packed_verify_attention else {}),
                             **({'kai_inkling_prefill_bias_mib': placement.get('prefill_bias_mib'),
                                 'kai_inkling_qualification_sha256': placement.get('qualification_sha256'),
                                 'kai_inkling_serial_verify_combine': env.get('KAILLM_INKLING_SERIAL_VERIFY_COMBINE') == '1',
                                 'kai_inkling_verify_combine_fma4': env.get('KAILLM_INKLING_VERIFY_COMBINE_FMA4',''),
                                 'kai_inkling_serial_verify_route': env.get('KAILLM_INKLING_SERIAL_VERIFY_ROUTE') == '1',
                                 'kai_inkling_serial_verify_bias': env.get('KAILLM_INKLING_SERIAL_VERIFY_BIAS') == '1',
                                 'kai_inkling_serial_verify_post': env.get('KAILLM_INKLING_SERIAL_VERIFY_POST') == '1',
                                 'kai_inkling_row_stable_dspark': row_stable_dspark,
                                 'measured_gguf_sha256': measured_digests}
                                if placement else {}),
                             **({"measured_drafter_sha256": measured_draft} if measured_draft else {}),
                             "cuda_graph_drafter_devices": draft_devices,
                             "cuda_graph_head_devices": head_devices,
                             "cuda_graph_capture_sizes_by_gpu": {uid: r['captures'] for uid, r in captured.items()},
                             "engine_revision": revision, "llama_source": llama_source,
                             "kai_ampere_qwen_opt_in": ampere_qwen,
                             "kai_humming_deterministic": deterministic_humming,
                             **({"kai_inkling_fused_q8": True} if fused_q8 else {}),
                             "kai_source": a.kai_commit, "server_argv": cmd, "max_concurrency": cap})
        for domain, output in outputs.items():
            if output.exists():
                continue
            stage = domain
            write_json(stamp, {"state": "running", "stage": stage, "model_revision": spec["revision"], "time": time.time()})
            run(SimpleNamespace(prompts=a.banks / (spec["family"] + ".json"), environment=envfile,
                                checkpoint=a.checkpoint, domain=domain, batch=a.batch, engine=a.engine,
                                url=url, timeout=a.timeout, mode=a.mode, output=output, graph_dir=graph_file,
                                sampling_profile=sampling_profile))
        write_json(stamp, {"state": "complete", "model_revision": spec["revision"], "time": time.time()})
    except Exception as exc:
        write_json(stamp, {"state": "failed", **({"llama_policy":"default"} if stock else {}), "stage": stage, "reason": str(exc)[-1400:], "time": time.time(),
                           "gpu_names": [d["name"] for d in devices], "checkpoint": a.checkpoint,
                           "engine": a.engine, "mode": a.mode, "batch": a.batch, "llama_source": llama_source,
                           "kai_source": a.kai_commit, "model_revision": spec["revision"]})
        raise
    finally:
        stop_server(proc)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "banks", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("runtime", "python", "llama", "drafter"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--checkpoint", required=True, choices=MANIFEST["models"])
    p.add_argument("--engine", required=True, choices=("kaiLLM", "llama.cpp"))
    p.add_argument("--mode", required=True, choices=DRAFT_TOKENS)
    p.add_argument("--batch", required=True, type=int, choices=BATCHES)
    p.add_argument("--domain", action="append", choices=("prose", "code"), help="Select a domain. Plain mode uses code. Speculative modes default to both.")
    p.add_argument("--sampling-profile", choices=SAMPLING_PROFILES, default=SAMPLING_PROFILE)
    p.add_argument("--devices", type=int, nargs="+", help="GPU row indices in nvidia-smi (default: first visible CUDA device)")
    p.add_argument("--port", type=int, default=8876)
    p.add_argument('--llama-policy', choices=('strict', 'default'), default='strict',
                   help='Use the certified llama.cpp build with strict graphs or its upstream default execution policy')
    p.add_argument('--kai-policy', choices=('strict','default'), default='strict',
                   help='Measure the normal Inkling dispatch or use the full-graph control')
    p.add_argument("--output-head", choices=("graph", "direct"), default="graph",
                   help="Run only the vocabulary projection directly on an eligible kaiLLM plain build")
    p.add_argument("--kv-gib", type=float, help="Explicit GiB per device, including fractional allocations")
    p.add_argument("--empirical-b2-capacity", action="store_true",
                   help="Verify ordinary plain B2 on the explicit KV pool from the completed workload and physical dispatches")
    p.add_argument("--kai-commit", required=True)
    p.add_argument("--timeout", type=int, default=43200)
    main(p.parse_args())
