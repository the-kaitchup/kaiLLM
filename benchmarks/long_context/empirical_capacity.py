"""Prove simultaneous plain batch-two decoding with a workload-specific KV pool."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
CONTRACT = 'b2-8192-65536-v1'
LOCK = ROOT / 'empirical_capacity_sources.json'
FILES = ('worker.py', 'benchmark.py', 'kai_default.py', 'empirical_capacity.py')
PROMPT, OUTPUT = 8192, 65536
WAVES = ('prefill-1', 'prefill-2', 'prefill-3', 'full')
# Mixed prefill/decode dispatches are credited once in the observer's decode phase.
MIN_PAIRED_PERCENT = 99
METRICS = {
    'vllm:prompt_tokens_total': 'prompt_tokens',
    'vllm:generation_tokens_total': 'generated_tokens',
    'vllm:num_preemptions_total': 'preemptions',
    'vllm:prompt_tokens_cached_total': 'cached_tokens',
    'vllm:request_num_preemptions_sum': 'finished_preemptions',
    'vllm:request_num_preemptions_count': 'finished_requests',
    'vllm:prompt_tokens_by_source_total': 'local_compute',
    'vllm:request_success_total': 'length_finishes',
}
FIELDS = set(METRICS.values())
LABEL = re.compile(r'([a-z_][a-z0-9_]*)="([^"\\]*)"')


def scope(engine, mode, batch, devices, normal):
    if (engine, mode, batch, devices, normal) != ('kaiLLM', 'plain', 2, 1, True):
        raise ValueError('Empirical KV capacity requires ordinary single-GPU kaiLLM plain B2')


def validate_launch(outer, effective):
    def option(cmd, flag):
        if cmd.count(flag) != 1 or cmd.index(flag)+1 >= len(cmd):
            raise ValueError(f'Missing or repeated empirical capacity option {flag}')
        return cmd[cmd.index(flag)+1]
    expected = {'--max-num-seqs':'2', '--max-num-batched-tokens':'2048',
                '--max-model-len':'73728', '--served-model-name':'bench'}
    for command in (outer, effective):
        if any(option(command, key) != value for key, value in expected.items()):
            raise ValueError('Empirical capacity launch differs from the locked B2 workload')
    if (option(outer,'--spec-tokens') != '0' or '--speculative-config' in effective
            or '--no-enable-prefix-caching' not in effective or '--enable-prefix-caching' in effective
            or option(outer, '--kv-cache-memory-bytes') != option(effective,'--kv-cache-memory-bytes')):
        raise ValueError('Empirical capacity needs ordinary decode, no prefix reuse, and matching KV bytes')


def source_lock():
    files = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES}
    key = hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if catalog().get(key) != files:
        raise ValueError('Unregistered empirical capacity benchmark source')
    return key


def catalog():
    return json.loads(LOCK.read_bytes())


def parse(text):
    result = {}
    for line in text.splitlines():
        name = line.split('{',1)[0]
        if name not in METRICS:
            continue
        sample = re.fullmatch(r'[^{}]+\{([^{}]*)\}\s+([^\s]+)(?:\s+[0-9]+)?', line)
        if not sample:
            raise ValueError('Malformed empirical capacity metric')
        labels = {}
        for part in sample[1].split(','):
            m=LABEL.fullmatch(part)
            if not m or m[1] in labels:
                raise ValueError('Unexpected empirical capacity metric labels')
            labels[m[1]]=m[2]
        if (labels.pop('model_name', None), labels.pop('engine', None)) != ('bench', '0'):
            raise ValueError('Empirical capacity requires an isolated bench model on one engine')
        key = METRICS[name]
        if key in ('local_compute','length_finishes'):
            label, value = ('source','local_compute') if key=='local_compute' else ('finished_reason','length')
            if set(labels) != {label}:
                raise ValueError('Unexpected empirical capacity metric labels')
            if labels[label] != value:
                continue
        elif labels:
            raise ValueError('Unexpected empirical capacity metric labels')
        try:
            value = Decimal(sample[2])
            if not value.is_finite() or value<0 or value!=value.to_integral_value() or key in result:
                raise ValueError('Invalid or duplicate empirical capacity counter')
            result[key] = int(value)
        except InvalidOperation as exc:
            raise ValueError('Invalid empirical capacity counter') from exc
    if set(result)!=FIELDS:
        raise ValueError('Missing empirical capacity counters: '+', '.join(sorted(FIELDS-set(result))))
    return result


def snapshot(url):
    with urllib.request.urlopen(url.rstrip('/')+'/metrics',timeout=5) as r:
        return parse(r.read().decode())


def expected(batch, outputs):
    return {'prompt_tokens':batch*PROMPT, 'local_compute':batch*PROMPT, 'generated_tokens':batch*outputs,
            'preemptions':0, 'cached_tokens':0, 'finished_preemptions':0,
            'finished_requests':batch, 'length_finishes':batch}


def delta(before, after):
    if (not isinstance(before,dict) or not isinstance(after,dict) or set(before)!=FIELDS or set(after)!=FIELDS
            or any(type(n) is not int or n<0 for n in (*before.values(),*after.values()))):
        raise ValueError('Invalid empirical capacity snapshots')
    values = {field:after[field]-before[field] for field in FIELDS}
    if min(values.values())<0:
        raise ValueError('Serving metrics reset during the measured wave')
    return values


def settled(url, before, batch, outputs, *, timeout=15, read=snapshot):
    # Prometheus can lag the response. Poll read-only counters outside the timed interval.
    want=expected(batch,outputs)
    until=time.monotonic()+timeout
    while True:
        try:
            after=read(url)
            actual=delta(before,after)
            if actual==want:
                return after
            if any(n>want[k] for k,n in actual.items()):
                raise ValueError(f'Extra requests, preemptions, cache reuse or tokens during the wave: {actual}')
        except OSError:
            if time.monotonic()>=until:
                raise
        if time.monotonic()>=until:
            raise ValueError(f'Incomplete scheduler counters after the wave: {actual}; expected {want}')
        time.sleep(.1)


def overlap(dispatch, outputs=OUTPUT):
    if not isinstance(dispatch,dict) or len(dispatch)!=1:
        raise ValueError('Expected one target GPU in the full-wave dispatch counters')
    phases=next(iter(dispatch.values()))
    if not isinstance(phases,dict) or not isinstance(phases.get('decode'),dict):
        raise ValueError('Expected an observed decode phase')
    decode=phases['decode']
    steps=[decode.get(k) for k in ('full','direct','other','direct_head')]
    if any(type(v) is not int or v<0 for v in steps) or steps[2] or not steps[0]:
        raise ValueError('Unknown or missing decode work')
    dispatches=steps[0]+steps[1]
    if steps[3]!=dispatches:
        raise ValueError('Every decode-bearing forward must have a completed output head')
    # The first token of each plain request is produced by the terminal prefill.
    # A physical decode-bearing forward can advance one token for at most two requests.
    advances=2*(outputs-1)
    paired=advances-dispatches
    if paired>dispatches or 100*paired<MIN_PAIRED_PERCENT*dispatches:
        raise ValueError(f'{dispatches} physical decode dispatches for {advances} advances do not demonstrate near-full B2 overlap')
    return {'decode_dispatches':dispatches,'decode_token_advances':advances,
            'min_paired_decode_dispatches':paired,'max_other_decode_dispatches':dispatches-paired}


def check_requests(raw):
    hashes = raw.get('prompt_hashes')
    if (raw.get('prompt_tokens'),raw.get('generated_tokens'),raw.get('batch')) != (PROMPT,OUTPUT,2):
        raise ValueError('Incomplete empirical capacity workload')
    if not isinstance(hashes,list) or len(hashes)!=2 or not all(isinstance(h,str) and h for h in hashes) or hashes[0]==hashes[1]:
        raise ValueError('Empirical capacity requires two distinct prompt windows')
    prefill=raw.get('prefill')
    if not isinstance(prefill,list) or len(prefill)!=3:
        raise ValueError('Empirical capacity requires three prefill waves')
    for wave,count in [*((w,1) for w in prefill),(raw.get('full'),OUTPUT)]:
        requests=wave.get('requests') if isinstance(wave,dict) else None
        if (not isinstance(requests,list) or len(requests)!=2
                or any(not isinstance(x,dict) or x.get('prompt_tokens')!=PROMPT or x.get('output_tokens')!=count
                       or x.get('cache_hits') not in (0,None) for x in requests)):
            raise ValueError('Incomplete empirical capacity request')


def validate_evidence(raw, dispatch):
    if not isinstance(raw,dict) or set(raw)!= {'protocol','harness_sha256','waves','overlap'}:
        raise ValueError('Incomplete empirical capacity evidence')
    if raw['protocol']!=CONTRACT or raw['harness_sha256'] not in catalog() or not isinstance(raw['waves'],dict) or set(raw['waves'])!=set(WAVES):
        raise ValueError('Unknown empirical capacity source or workload')
    previous=None
    for name in WAVES:
        obs=raw['waves'][name]
        if not isinstance(obs,dict) or set(obs)!= {'before','after'}:
            raise ValueError('Missing scheduler interval')
        if previous is not None and obs['before']!=previous:
            raise ValueError('Missing scheduler interval between measured waves')
        if delta(obs['before'],obs['after'])!=expected(2,OUTPUT if name=='full' else 1):
            raise ValueError('Scheduler did not account for the full uncached B2 workload')
        previous=obs['after']
    computed=overlap(dispatch)
    if (not isinstance(raw['overlap'],dict) or set(raw['overlap'])!=set(computed)
            or any(type(raw['overlap'][k]) is not int or raw['overlap'][k]!=v for k,v in computed.items())):
        raise ValueError('Unverified B2 physical overlap')
    return raw


def capacity_valid(row):
    kv=row.get('kai_kv')
    tagged='kai_empirical_capacity' in row
    if kv is None:
        return not tagged and not (row.get('batch') == 2 and 'kai_default_dispatch' in row)
    if not isinstance(kv,dict):
        return False
    nominal=kv.get('full_context_sequences')
    if (type(nominal) not in (int,float) or not math.isfinite(nominal) or nominal<=0
            or type(kv.get('bytes_per_device')) is not int or kv['bytes_per_device']<=0):
        return False
    if not tagged:
        batch=row.get('batch',1)
        return type(batch) is int and batch>0 and nominal>=batch
    if nominal < 1:
        return False
    try:
        scope(row.get('engine'),row.get('mode'),row.get('batch'),len(row.get('gpu',())), 'kai_default_dispatch' in row)
        if (row.get('requested_context_per_slot'),row.get('input_tokens_per_request'),row.get('output_tokens_per_request')) != (PROMPT+OUTPUT,PROMPT,OUTPUT):
            return False
        ranks=row['kai_default_dispatch']['full']
        if not isinstance(ranks,list) or len(ranks)!=1:
            return False
        validate_evidence(row['kai_empirical_capacity'], {0:ranks[0]})
    except (ValueError, KeyError, TypeError):
        return False
    return True
