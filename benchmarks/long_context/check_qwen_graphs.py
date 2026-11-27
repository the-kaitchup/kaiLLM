"""Qualify one Qwen3.8 or DaisyRose mode against same-device direct and chunked GDN."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import subprocess
import sys
import threading

from benchmark import MANIFEST, full_graph_runtime, verify_prompt_bank, write_json
from cuda_graphs import (COUNTERS, ENV, PHASES, snapshot, validate_kai_launch,
                         verify_draft_interval, verify_head_interval, verify_interval)
from devices import visible_cuda_uuids
from verify_gguf_loader import predict
from verify_inkling_drafter import counters
from worker import admission_fraction, checkpoint_files, gpu_inventory, selected_devices, stop_server, wait_server


REVISION = 'bb8ca4b2cc82c07ed483bb33a6263e38582a04ce'
PREFIXES = (63, 64, 65, 2047, 2048, 2049, 8192)
MODES = {'plain': 0, 'mtp5': 5, 'dspine7': 7, 'dspine15': 15}
MODELS = {k: v for k, v in MANIFEST['models'].items() if v['family'] in ('qwen', 'daisy')}


def fingerprints(paths):
    files = {}
    for path in paths:
        with path.open('rb') as stream:
            files[path.name] = {'bytes': path.stat().st_size,
                                'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}
    return files


def verified_files(root, key, spec):
    paths = checkpoint_files(root, key, spec)
    if fingerprints(paths) != spec['files']:
        raise ValueError(f'Hugging Face checkpoint SHA-256 mismatch for {key}')
    return paths


def validate_mode(model, mode, batch):
    if mode == 'mtp5' and not model['native_mtp_layers']:
        raise ValueError('This checkpoint has no native MTP layer')
    if mode.startswith('dspine') and batch != 1:
        raise ValueError('The pinned DSpine runtime supports precisely one slot')


def option(argv, key, default=None):
    # The wrapper appends explicit vLLM arguments after its generated defaults.
    values = [argv[i+1] for i, entry in enumerate(argv) if entry == key]
    return values[-1] if values else default


def validate_launch(argv, batch, budget, mode, forced, ranks):
    validate_kai_launch(argv)
    if ('--language-model-only' not in argv or '--no-enable-prefix-caching' not in argv
            or option(argv, '--dtype') != 'bfloat16' or option(argv, '--kv-cache-dtype') != 'auto'
            or int(option(argv, '--max-num-seqs')) != batch
            or int(option(argv, '--max-num-batched-tokens')) != budget
            or int(option(argv, '--tensor-parallel-size', '1')) != ranks
            or int(option(argv, '--pipeline-parallel-size', '1')) != 1):
        raise ValueError('Effective serving configuration differs from the qualification settings')
    spec = json.loads(option(argv, '--speculative-config', '{}'))
    if mode == 'plain':
        if spec:
            raise ValueError('Expected plain target decoding')
    elif (spec.get('method') != ('mtp' if mode == 'mtp5' else 'dflash')
          or spec.get('num_speculative_tokens') != MODES[mode]
          or (spec.get('rejection_sample_method') == 'synthetic') != forced
          or (forced and spec.get('synthetic_acceptance_rates') != [0.] * MODES[mode])):
        raise ValueError('Effective speculative policy differs from the requested control')
    return spec


def token_delta(before, after, forced):
    proposed = after['draft_tokens'] - before['draft_tokens']
    accepted = after['accepted_tokens'] - before['accepted_tokens']
    if not 0 <= accepted <= proposed or proposed <= 0 or (forced and accepted):
        raise ValueError('Speculative counters do not establish the requested rejection behavior')
    return {'proposed': proposed, 'accepted': accepted, 'rejected': proposed - accepted}


class MixedSteps:
    """Observe both target phases outstanding on the same serial GPU worker step."""
    def __init__(self, directory, before):
        self.directory, self.before = directory, before
        self.seen = {}

    def __enter__(self):
        self.stack = ExitStack()
        self.done = threading.Event()
        try:
            maps = {}
            for uuid, state in self.before.items():
                path = self.directory / f"{uuid}.{state['pid']}.counts"
                f = self.stack.enter_context(path.open('rb'))
                maps[uuid] = self.stack.enter_context(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))
            self.thread = threading.Thread(target=self.poll, args=(maps,), daemon=True)
            self.thread.start()
        except BaseException:
            self.stack.close()
            raise
        return self

    def poll(self, maps):
        last = {}
        while not self.done.is_set() and len(self.seen) != len(maps):
            for uuid, mapping in maps.items():
                state = COUNTERS.unpack_from(mapping)
                p, pr, pe, pu, d, dr, de, du = state
                if (uuid not in self.seen and state == last.get(uuid) and p == pr + 1 and d == dr + 1
                        and p > self.before[uuid]['prefill']['scheduled'] and d > self.before[uuid]['decode']['scheduled']
                        and not (pe or pu or de or du)):
                    self.seen[uuid] = {'prefill': p, 'decode': d}
                last[uuid] = state
            self.done.wait(.0001)

    def __exit__(self, *exc):
        self.done.set()
        self.thread.join()
        self.stack.close()

    def finished(self, after):
        if (set(self.seen) != set(after) or
                any(count > after[uuid][phase]['graph_replays']
                    for uuid, phases in self.seen.items() for phase, count in phases.items())):
            raise ValueError('No completed mixed prefill/decode graph witnessed on every selected GPU')
        return self.seen


def capture_wave(url, prompts, output_lengths, timeout):
    start = threading.Barrier(len(prompts))
    def one(prompt, output_tokens):
        start.wait(timeout=timeout)
        return predict('kaiLLM', url, prompt, output_tokens=output_tokens, timeout=timeout)
    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        return list(executor.map(one, prompts, output_lengths))


def compare_scores(reference, candidate, max_logprob_delta=None):
    matched = positions = overlap = exact_sets = 0
    max_error = total_error = 0.
    first = None
    if not reference or reference.keys() != candidate.keys():
        raise ValueError('Expected the same qualification waves')
    for label in reference:
        if not reference[label]:
            raise ValueError('Empty qualification wave')
        for slot, (left, right) in enumerate(zip(reference[label], candidate[label], strict=True)):
            if not left['tokens'] or left['prompt_sha256'] != right['prompt_sha256']:
                raise ValueError('Missing or different prompt prefix')
            for row in (left, right):
                for token, scores in zip(row['tokens'], row['top_logprobs'], strict=True):
                    if len(scores) != 8 or str(token) not in scores or not all(map(math.isfinite, scores.values())):
                        raise ValueError('Incomplete or nonfinite top-eight log probabilities')
            for step, (a, b) in enumerate(zip(left['tokens'], right['tokens'], strict=True)):
                ref, got = left['top_logprobs'][step], right['top_logprobs'][step]
                shared = ref.keys() & got.keys()
                errors = [abs(ref[k] - got[k]) for k in shared]
                overlap += len(shared)
                exact_sets += len(shared) == 8
                max_error = max([max_error, *errors])
                total_error += sum(errors)
                positions += 1
                if a != b:
                    if first is None:
                        def gap(scores, x, y):
                            return scores[str(x)] - scores[str(y)] if str(y) in scores else None
                        first = {'wave': label, 'slot': slot, 'output_position': step, 'reference': a, 'candidate': b,
                                 'reference_logprob_advantage': gap(ref, a, b), 'candidate_logprob_advantage': gap(got, b, a)}
                    break
                matched += 1
    if not positions:
        raise ValueError('No common-prefix decisions captured')
    scores_pass = max_logprob_delta is None or (overlap > 0 and max_error <= max_logprob_delta)
    return {'passed': matched == positions and scores_pass, 'common_prefix_positions': positions, 'matching_greedy': matched,
            'matching_top8_sets': exact_sets, 'overlapping_top8_ids': overlap, 'max_abs_logprob_delta': max_error if overlap else None,
            'mean_abs_logprob_delta': total_error / overlap if overlap else None, 'first_difference': first}


def prepare_launch(args, name, target, draft, devices, directory):
    uuids = [d['uuid'] for d in devices]
    mode = args.mode if name in ('native', 'forced') else 'plain'
    k = MODES[mode]
    # vLLM requires scheduler_tokens >= batch. At B32 the generic FULL descriptors
    # have min(tokens, 32) slots, so all descriptors up to 32 still dispatch direct GDN.
    budget = max(16, args.batch) if name == 'direct' else 2048
    extra = ['--tensor-parallel-size', str(len(uuids))]
    if name == 'direct' and args.batch > 16:
        extra += ['--long-prefill-token-threshold', '16']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(uuids), KAILLM_HUMMING_DETERMINISTIC='1',
               KAILLM_QWEN35_FULL_GRAPHS='1', KAILLM_CACHE_DIR=str(args.out / 'derived-cache'),
               KAILLM_NATIVE_ASSETS=str(args.runtime / 'runtime-native-cache/native-assets.json'),
               OMP_NUM_THREADS=str(min(8, max(1, len(os.sched_getaffinity(0)) // 2))))
    env[ENV] = str(directory)
    if MODELS[args.checkpoint]['kind'] != 'chimachima' and all(d['compute_cap'] == '8.6' for d in devices):
        env['KAILLM_ALLOW_QWEN35_SM86'] = '1'
    cmd = [sys.executable, str(args.runtime / 'deployment/serve.py'), str(target), '--port', str(args.port),
           '--served-model-name', 'bench', '--spec-tokens', str(k), '--max-num-seqs', str(args.batch),
           '--max-model-len', '8256', '--max-num-batched-tokens', str(budget),
           '--kv-cache-memory-bytes', str(args.kv_mib << 20), '--gpu-memory-utilization', str(admission_fraction(devices))]
    if mode.startswith('dspine'):
        cmd += ['--dspine', str(draft)]

    def preview(forced):
        launch = [*cmd, '--extra-vllm-args', json.dumps(extra)]
        dry = subprocess.run([*launch, '--dry-run', '--compute-capability', devices[0]['compute_cap']],
                             env=env, check=True, capture_output=True, text=True)
        data = json.loads(dry.stdout)
        if (data['model_architecture'] != 'qwen35' or data['compute_capability'] != devices[0]['compute_cap']
                or not data['capture_sizes'] or max(data['capture_sizes']) != budget):
            raise ValueError('BB8 family, GPU architecture or graph capture budget differs from the requested control')
        effective = {key: val for key, val in env.items() if key not in data.get('cleared_environment_keys', ())}
        effective.update(data['environment'])
        expected = {'KAILLM_QWEN35_FULL_GRAPHS': '1', 'KAILLM_HUMMING_DETERMINISTIC': '1',
                    'KAILLM_GDN_SINGLE_SLOT': '0', 'KAILLM_GDN_DEFERRED': '0'}
        if any(effective.get(flag) != value for flag, value in expected.items()):
            raise ValueError('Deterministic graph-only Qwen configuration was overridden')
        spec = validate_launch(data['command'], args.batch, budget, mode, forced, len(uuids))
        return launch, data['command'], spec

    launch, argv, spec = preview(False)
    if name == 'forced':
        extra += ['--speculative-config', json.dumps(dict(spec, rejection_sample_method='synthetic',
                                                        synthetic_acceptance_rates=[0.] * k))]
        launch, argv, spec = preview(True)
    return launch, env, spec, {'argv': argv, 'batch': args.batch, 'gpu_uuids': uuids,
                               'deterministic_humming': True, 'scheduler_tokens': budget}


def run_control(args, name, target, draft, bank, devices):
    out = args.out / name
    directory = out / 'graphs'
    directory.mkdir(parents=True)
    launch, env, spec, settings = prepare_launch(args, name, target, draft, devices, directory)
    uuids = settings['gpu_uuids']
    k = spec.get('num_speculative_tokens', 0)
    write_json(out / 'launch.json', settings)
    url = f'http://127.0.0.1:{args.port}'
    log = out / 'server.log'
    with log.open('w') as stream:
        proc = subprocess.Popen(launch, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        wait_server(proc, url, log)
        derived = None
        if spec.get('method') == 'dflash':
            prepared = Path(spec['model'])
            shards = sorted(prepared.glob('*.safetensors'))
            if not shards or not prepared.is_relative_to(args.out / 'derived-cache'):
                raise ValueError('Expected freshly prepared DSpine weights inside this qualification directory')
            derived = fingerprints([*sorted(prepared.glob('*.json')), *shards])
        result = {}
        totals = {'proposed': 0, 'accepted': 0, 'rejected': 0}
        output = max(8, 2 * (MODES[args.mode] + 1))
        for domain in ('prose', 'code'):
            rows = bank['domains'][domain]['rows'][:args.batch]
            groups = [(str(size), [row[:size] for row in rows]) for size in PREFIXES]
            if args.batch > 1 and args.mode != 'plain':
                groups.append(('mixed', [rows[0][:64], *rows[1:]]))
            for suffix, prompts in groups:
                label = f'{domain}-{suffix}'
                before = snapshot(directory, uuids)
                if any(not s['head_sizes'] or bool(s['draft_captures']) != bool(k) for s in before.values()):
                    raise ValueError('Mismatch in active target vocabulary-head or drafter graph inventories')
                n0 = counters(url) if k else None
                watcher = None
                with ExitStack() as stack:
                    if suffix == 'mixed' and k:
                        watcher = stack.enter_context(MixedSteps(directory, before))
                    # Only the short request can run a decode step while long rows prefill.
                    lengths = [output] + [1] * (len(prompts) - 1) if suffix == 'mixed' else [output] * len(prompts)
                    result[label] = capture_wave(url, prompts, lengths, args.timeout)
                write_json(out / f'{label}.json', result[label])
                after = snapshot(directory, uuids)
                proof = {'target': verify_interval(before, after, PHASES),
                         'head': verify_head_interval(before, after, PHASES, uuids)}
                if k:
                    proof['draft'] = verify_draft_interval(before, after, uuids, require_proposal=True)
                    delta = token_delta(n0, counters(url, required=True), name == 'forced')
                    proof['tokens'] = delta
                    for field in totals:
                        totals[field] += delta[field]
                if watcher:
                    proof['mixed_steps'] = watcher.finished(after)
                write_json(out / f'{label}-evidence.json', proof)
        if name == 'native' and not (totals['accepted'] and totals['rejected']):
            raise ValueError('Ordinary speculation must exhibit both accepted and rejected proposals')
        return result, {'draft_tokens': totals if k else None, 'prepared_draft_files': derived}
    finally:
        stop_server(proc)


def main(args):
    model = MODELS[args.checkpoint]
    validate_mode(model, args.mode, args.batch)
    if args.max_logprob_delta is not None and (not math.isfinite(args.max_logprob_delta) or args.max_logprob_delta < 0):
        raise ValueError('The optional common-token score tolerance must be nonnegative and finite')
    if args.kv_mib <= 0 or args.timeout <= 0:
        raise ValueError('KV allocation and request timeout must be positive')
    pin = full_graph_runtime(args.checkpoint, 'plain', revision=REVISION)
    if not pin or (args.runtime / 'benchmark-source.sha256').read_text().strip() != pin[1]:
        raise ValueError('Qualification requires the exact BB8 graph runtime')
    target, = verified_files(args.weights, args.checkpoint, model)
    remote_draft = None
    draft = None
    if args.mode.startswith('dspine'):
        remote_draft = MANIFEST['dspine']
        source_files = verified_files(args.weights, 'dspine', remote_draft)
        draft = source_files[0].parent
        if set(draft.glob('*.safetensors')) != {f for f in source_files if f.suffix == '.safetensors'}:
            raise ValueError('Unexpected DSpine source shards outside the pinned HF revision')
        if (draft / 'generation_config.json').exists() and 'generation_config.json' not in remote_draft['files']:
            raise ValueError('Unpinned DSpine generation configuration in the verified weights directory')
    bank = json.loads((args.banks / (model['family'] + '.json')).read_bytes())
    for domain in ('prose', 'code'):
        verify_prompt_bank(bank, model['family'], domain)
    devices = selected_devices(gpu_inventory(), args.devices, visible_cuda_uuids())
    if (len(devices) not in (1, 2, 4) or len({d['compute_cap'] for d in devices}) != 1
            or devices[0]['compute_cap'] not in ('8.6', '12.0')):
        raise ValueError('Select one, two or four GPUs sharing a supported compute capability')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    keys = ('repo', 'revision', 'files')
    receipt = {'passed': False, 'engine': 'kaiLLM', 'mode': args.mode, 'checkpoint': args.checkpoint,
               'batch': args.batch, 'gpus': [{k: d[k] for k in ('name', 'uuid', 'driver_version', 'compute_cap')} for d in devices],
               'runtime_revision': pin[0], 'runtime_sha256': pin[1], 'target': {k: model[k] for k in keys},
               'drafter': remote_draft, 'prefixes': PREFIXES, 'tokenizer_revision': bank['tokenizer_revision'],
               'prompt_bank_sha256': MANIFEST['tokenizers'][model['family']]['prompt_bank_sha256'],
               'kv_bytes_per_device': args.kv_mib << 20, 'max_logprob_delta': args.max_logprob_delta,
               'controls': {}, 'comparisons': {}}
    dest = args.out / 'qualification.json'
    write_json(dest, receipt)
    try:
        captures = {}
        stages = ['direct', 'plain'] if args.mode == 'plain' else ['direct', 'plain', 'native', 'forced']
        for name in stages:
            captures[name], receipt['controls'][name] = run_control(args, name, target, draft, bank, devices)
            if name != 'direct':
                reference = 'direct' if name == 'plain' else 'plain'
                receipt['comparisons'][f'{reference}-{name}'] = compare_scores(captures[reference], captures[name], args.max_logprob_delta)
            write_json(dest, receipt)
        if (draft and receipt['controls']['native']['prepared_draft_files']
                != receipt['controls']['forced']['prepared_draft_files']):
            raise ValueError('Ordinary and forced DSpine loaded different prepared weights')
        if not all(c['passed'] for c in receipt['comparisons'].values()):
            raise ValueError('Common-prefix greedy or score parity failed; see qualification.json')
        receipt['passed'] = True
    except Exception as exc:
        receipt['error'] = str(exc)
        raise
    finally:
        write_json(dest, receipt)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('runtime', 'weights', 'banks', 'out'):
        p.add_argument('--' + key, required=True, type=Path)
    p.add_argument('--checkpoint', required=True, choices=MODELS)
    p.add_argument('--mode', required=True, choices=MODES)
    p.add_argument('--batch', required=True, choices=(1, 2, 4, 8, 16, 32), type=int)
    p.add_argument('--max-logprob-delta', type=float, help='Also require every overlapping top-eight score to meet this absolute tolerance')
    p.add_argument('--devices', type=int, nargs='+', default=[0])
    p.add_argument('--kv-mib', type=int, default=4096, help='KV allocation per GPU in MiB for the short controls')
    p.add_argument('--port', type=int, default=18889)
    p.add_argument('--timeout', type=int, default=600, help='Timeout in seconds per request')
    main(p.parse_args())
