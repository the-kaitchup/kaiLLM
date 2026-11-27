"""Qualify full-context single-GPU llama.cpp KV allocation failures from original startup evidence."""
from collections import defaultdict
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

from benchmark import BATCHES, MANIFEST, OUTPUT, PROMPT, dspark_asset, llama_revision
from devices import physical_id
from llama_default import MICROBATCHES, compute_oom, validate_command
from llama_graph import BINARY_NAMES, SOURCE_LOCK, VERSION

MIB = 1 << 20


def digest(data):
    return hashlib.sha256(data).hexdigest()


def gpu_inventory(folder, records):
    names = {r['hardware_id']: r['gpu'][0] for r in records if len(r['gpu']) == 1}
    path = folder / 'physical-gpu.csv'
    devices = {}
    if path.is_file():
        for row in csv.reader(path.read_text().splitlines(), skipinitialspace=True):
            name, uuid, memory, driver, power = (x.strip() for x in row)
            devices[uuid] = (name, int(memory), driver, Decimal(power))
    return names, devices, digest(path.read_bytes()) if path.is_file() else None


def allocation(log, target):
    """Identify one CUDA allocation followed by an explicit KV failure from the intended target."""
    lines = log.splitlines()
    starts = [i for i,line in enumerate(lines) if 'load_model:' in line and 'loading model ' in line]
    if len(starts) != 1 or f"loading model '{target}'" not in lines[starts[0]]:
        return None
    sizes = [(i,int(m[1])) for i,line in enumerate(lines) if (m := re.search(r'failed to allocate CUDA0 buffer of size (\d+)\s*$',line))]
    if len(sizes) != 1:
        return None
    i, requested = sizes[0]
    if (i < starts[0]+2 or i+1 >= len(lines) or 'on device 0: cudaMalloc failed: out of memory' not in lines[i-1]
            or 'failed to initialize the context: failed to allocate buffer for kv cache' not in lines[i+1]):
        return None
    # Use one target allocation. A repeated model-buffer print must not count twice.
    # The printed MiB size is rounded, so use the smallest possible byte count.
    buffers = [m[1] for line in lines[starts[0]:i] if (m := re.search(r'\bCUDA0\s+model buffer size\s*=\s*(\d+\.\d+)\s*MiB\s*$',line))]
    resident = max((int((Decimal(s)-Decimal(1).scaleb(-len(s.partition('.')[2]))/2)*MIB) for s in buffers), default=0)
    return requested, max(0,resident)


def exact_model(given, expected):
    return given and all(given.get(k) == expected[k] for k in ('repo', 'revision', 'files'))


def source_receipts(folder, checkpoint, mode, upstream):
    model = MANIFEST['models'][checkpoint]
    build = json.loads((folder / f"public-build-{model['family']}.json").read_bytes())
    executable = build.get('public_executable_sha256', '')
    if build.get('upstream') != upstream or not re.fullmatch(r'[0-9a-f]{64}', executable):
        raise ValueError('The public executable certificate does not match the tested llama.cpp revision')
    names = [(checkpoint, model)]
    if mode == 'dspark5':
        names.append(dspark_asset(model))
    sources = {}
    for name, spec in names:
        raw = (folder / f'download-{name}.json').read_bytes()
        download = json.loads(raw)
        if download.get('source') != 'huggingface.co' or not exact_model(download, spec):
            raise ValueError('The physical download receipt differs from the pinned Hugging Face file')
        sources[name] = digest(raw)
    return {'executable_sha256': executable, 'download_receipts': sources}


def validate_model_path(argv, flag, spec):
    filename, = spec['files']
    value = argv[argv.index(flag) + 1]
    expected = f"/{spec['repo']}/{spec['revision']}/{filename}"
    if not value.endswith(expected):
        raise ValueError('KV probe model path differs from the pinned GGUF')
    return value


def staged_draft_observation(folder, inventory, inventory_hash):
    """Read a complete stock microbatch sweep ending in draft KV OOM on one physical board."""
    stamp_bytes = (folder / 'stock-draft-admission.json').read_bytes()
    stamp = json.loads(stamp_bytes)
    model = MANIFEST['models'][stamp['checkpoint']]
    draft_key, draft = dspark_asset(model)
    revision = llama_revision(model)
    cert = json.loads((folder / 'certified-upstream.json').read_bytes())
    if (model['family'] != 'inkling' or cert['protocol'] != VERSION or cert['upstream'] != revision
            or cert['source_sha256'] != json.loads(SOURCE_LOCK.read_bytes())[revision]
            or set(cert['binary_sha256']) != set(BINARY_NAMES)
            or any(not re.fullmatch(r'[0-9a-f]{64}', value) for value in cert['binary_sha256'].values())):
        raise ValueError('Incorrect stock draft build certificate')
    downloads = {}
    for key, spec in ((stamp['checkpoint'], model), (draft_key, draft)):
        raw = (folder / f'download-{key}.json').read_bytes()
        if json.loads(raw) != {'source': 'huggingface.co', **{k: spec[k] for k in ('repo', 'revision', 'files')}}:
            raise ValueError('Different target or draft GGUF')
        downloads[key] = digest(raw)
    uuid, batch, argv = stamp['uuid'], stamp['batch'], stamp['minimum_command']
    context = PROMPT + OUTPUT + 6
    validate_command(argv, 'dspark5', context, MICROBATCHES[-1])
    if (batch not in BATCHES or argv[argv.index('--ctx-size')+1] != str(batch*context)
            or argv[argv.index('--parallel')+1] != str(batch)):
        raise ValueError('Different stock batch or context')
    flags = {v for v in argv if v.startswith('-')}
    if flags != {'-m','--host','--port','--ctx-size','--parallel','-ngl','--fit','--flash-attn','--cache-type-k',
                 '--cache-type-v','--cache-ram','--no-context-shift','-b','-ub','--spec-type','--model-draft','--spec-draft-n-max'}:
        raise ValueError('Unexpected stock command override')
    target_path = validate_model_path(argv, '-m', model)
    draft_path = validate_model_path(argv, '--model-draft', draft)
    results = []
    for size in MICROBATCHES:
        raw = (folder / f'ub{size}.log').read_bytes()
        log = raw.decode(errors='replace')
        if digest(raw) != stamp['log_sha256'].get(str(size)) or f"loading model '{target_path}'" not in log:
            raise ValueError('Target or log digest mismatch during the stock draft sweep')
        loading = f"loading draft model '{draft_path}'"
        if loading not in log and (requested := compute_oom(log)):
            stage = 'target compute'
        else:
            aft = log.partition(loading)[2]
            match = re.search(r'on device 0: cudaMalloc failed: out of memory[^\n]*\n[^\n]*failed to allocate CUDA0 buffer of size (\d+)', aft)
            if not match:
                raise ValueError('No witnessed stock draft allocation failure')
            requested = int(match[1])
            if 'failed to allocate buffer for kv cache' in aft and 'failed to create MTP context' in aft:
                stage = 'draft KV'
            elif 'unable to allocate CUDA0 buffer' in aft and f"failed to load draft model, '{draft_path}'" in aft:
                stage = 'draft weights'
            else:
                raise ValueError('The stock draft failure does not identify the allocation')
        results.append(dict(microbatch=size, stage=stage, requested_bytes=requested, log_sha256=digest(raw)))
    if results[-1]['stage'] != 'draft KV':
        raise ValueError('Missing full-context draft KV failure at the smallest tested microbatch')
    name, total, driver, power = inventory[uuid]
    evidence = dict(context_per_slot=context, total_mib=total, driver=driver, power_limit_w=str(power),
                    inventory_sha256=inventory_hash, receipt_sha256=digest(stamp_bytes), witnesses=results,
                    executable_sha256=cert['binary_sha256']['llama-server'], download_receipts=downloads)
    by_stage = {}
    for run in results:
        by_stage.setdefault(run['stage'], []).append(str(run['microbatch']))
    detail = ', '.join(f"{stage} at ub{'/'.join(sizes)}" for stage, sizes in by_stage.items())
    return dict(gpu=[name], hardware_id=physical_id([uuid]), checkpoint=stamp['checkpoint'], engine='llama.cpp',
                mode='dspark5', batch=batch, llama_source=revision, llama_policy='default', model_revision=model['revision'],
                sha256=[v['sha256'] for v in model['files'].values()], observation_only=True, llama_draft_admission=evidence,
                reason=f'On the tested {total:,} MiB board, allocation failed for {detail}.')


def sample(path):
    receipt_bytes = path.read_bytes()
    receipt = json.loads(receipt_bytes)
    raw = path.with_suffix('.log').read_bytes()
    if digest(raw) != receipt['log_sha256']:
        raise ValueError(f'KV probe log digest mismatch for {path.name}')

    checkpoint = receipt['checkpoint']
    model = MANIFEST['models'].get(checkpoint)
    mode, batch, microbatch = receipt['mode'], receipt['batch'], receipt['microbatch']
    if (not model or mode not in ('plain', 'dspark5') or batch not in BATCHES or microbatch not in (512, 64)
            or not exact_model(receipt['model'], model) or receipt['upstream'] != llama_revision(model)):
        return None
    native = mode == 'dspark5'
    if (native and not exact_model(receipt['draft'], dspark_asset(model)[1])) or (not native and receipt.get('draft') is not None):
        raise ValueError('Incorrect draft assets for llama.cpp KV evidence')
    argv = receipt['argv']
    context = PROMPT + OUTPUT + (6 if native else 0)
    validate_command(argv, mode, context, microbatch)
    expected = {'-m', '--host', '--port', '--ctx-size', '--parallel', '-ngl', '--fit', '--flash-attn',
                '--cache-type-k', '--cache-type-v', '--cache-ram', '--no-context-shift', '-b', '-ub', '--spec-type'}
    if native:
        expected.update(('--model-draft', '--spec-draft-n-max'))
    if '-lv' in argv:
        expected.add('-lv')
    flags = [arg for arg in argv[1:] if arg.startswith('-')]
    binary_suffix = f"/{receipt['upstream']}/build/bin/llama-server"
    if (set(flags) != expected or len(flags) != len(expected) or not argv[0].endswith(binary_suffix)
            or str(batch) != argv[argv.index('--parallel') + 1] or str(batch * context) != argv[argv.index('--ctx-size') + 1]):
        raise ValueError('Different physical workload, offload or allocation settings')
    target = validate_model_path(argv, '-m', model)
    if native:
        validate_model_path(argv, '--model-draft', dspark_asset(model)[1])
    fields = re.fullmatch(r'(GPU-[0-9a-fA-F-]{36}),\s*(\d+) MiB,\s*(\d+) MiB,\s*([\d.]+),\s*([\d.]+) W', receipt['gpu'])
    if not fields:
        raise ValueError('Incomplete GPU inventory for measured KV failure')
    uuid, total, free, driver, power = fields.groups()
    if not 0 < int(free) <= int(total):
        raise ValueError('Invalid GPU memory counters')
    identity = (uuid, int(total), driver, Decimal(power), checkpoint, receipt['upstream'], mode, batch)
    data = allocation(raw.decode(errors='replace'), target) if receipt['outcome'] == 'cuda-oom' else None
    return identity, dict(microbatch=microbatch, outcome=receipt['outcome'], allocation=data,
                          log=digest(raw), receipt=digest(receipt_bytes))


def measured_limits(folder, records=()):
    """A physical limit requires the same KV bytes at both supported microbatch endpoints."""
    names, inventory, inventory_hash = gpu_inventory(folder, records)
    cases = defaultdict(list)
    for path in sorted(folder.glob('public-capacity-*.json')):
        record = sample(path)
        if record:
            key, run = record
            cases[key].append(run)

    limits = []
    if (folder / 'stock-draft-admission.json').is_file():
        limits.append(staged_draft_observation(folder, inventory, inventory_hash))
    for (uuid, total, driver, power, checkpoint, revision, mode, batch), runs in cases.items():
        if (any(r['outcome'] != 'cuda-oom' or r['allocation'] is None for r in runs)
                or {r['microbatch'] for r in runs} != {64, 512}):
            continue
        requests = {r['allocation'][0] for r in runs}
        if len(requests) != 1:
            continue
        requested, = requests
        resident = max(r['allocation'][1] for r in runs)
        board = physical_id([uuid])
        gpu = names.get(board)
        if uuid in inventory:
            label, capacity, version, watts = inventory[uuid]
            if (total, driver, power) != (capacity, version, watts) or (gpu and gpu != label):
                raise ValueError('Capacity experiment differs from the recorded physical inventory')
            gpu = label
        if gpu is None:
            continue
        sources = source_receipts(folder, checkpoint, mode, revision)
        # nvidia-smi displays an integer MiB capacity. Allow for one further MiB
        # when deciding whether the request exceeds total memory on this card.
        hard = requested + resident >= (total + 1) * MIB
        kv = f'{requested / MIB:,.2f}'.removesuffix('.00')
        if requested >= (total + 1) * MIB:
            observation = f'The {kv} MiB GPU KV allocation exceeds the {total:,} MiB device capacity.'
        elif hard:
            observation = (f'The {kv} MiB GPU KV allocation and at least {resident // MIB:,} MiB of resident '
                           f'model weights exceed the {total:,} MiB device capacity.')
        else:
            observation = (f'The {kv} MiB GPU KV allocation failed at microbatches 512 and 64. '
                           f'The measured allocation fits within the {total:,} MiB physical capacity on its own.')
        admission = dict(requested_bytes=requested, resident_min_bytes=resident, total_mib=total,
                         inventory_sha256=inventory_hash, driver=driver, power_limit_w=str(power),
                         context_per_slot=PROMPT + OUTPUT + (6 if mode == 'dspark5' else 0), **sources,
                         evidence=[{k: r[k] for k in ('microbatch', 'log', 'receipt')} for r in runs])
        model = MANIFEST['models'][checkpoint]
        limits.append(dict(gpu=[gpu], hardware_id=board, checkpoint=checkpoint, engine='llama.cpp', mode=mode,
                           batch=batch, llama_source=revision, llama_policy='default', model_revision=model['revision'],
                           sha256=[v['sha256'] for v in model['files'].values()], reason=observation,
                           observation_only=not hard, llama_kv_allocation=admission))
    return limits
