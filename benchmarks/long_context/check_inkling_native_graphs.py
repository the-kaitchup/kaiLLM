"""Check a single-GPU Inkling DSpark target against independent plain and forced-rejection servers."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from benchmark import MANIFEST, PROMPT, dspark_asset, verify_prompt_bank, write_json
from cuda_graphs import ENV, PHASES, snapshot, validate_kai_launch, verify_draft_interval, verify_head_interval, verify_interval
from devices import visible_cuda_uuids
from verify_inkling_drafter import SLOTS, capture, check_width, compare, request_labels
from worker import CONTEXT, GIB, admission_fraction, checkpoint_files, gpu_inventory, selected_devices, stop_server, wait_server


def checkpoint_file(root, name, spec):
    path, = checkpoint_files(root, name, spec)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != spec['files'][path.name]['sha256']:
        raise ValueError(f'{name} bytes do not match its pinned Hugging Face SHA-256')
    return path


def full_admission(logs, slots):
    readings = re.findall(r'GPU KV cache size: ([0-9,]+) tokens, Maximum concurrency for '
                          r'([0-9,]+) tokens per request: ([0-9.]+)x', logs)
    if len(readings) != 1:
        raise ValueError('Expected one physical GPU KV capacity for full-context admission')
    total, length, shown = readings[0]
    tokens, context = (int(x.replace(',', '')) for x in (total, length))
    if context != CONTEXT or abs(tokens / context - float(shown)) > .005000001:
        raise ValueError('Capacity does not describe the requested full-length sequence')
    if tokens < context * slots:
        raise ValueError(f'Only {tokens} GPU KV tokens for {slots} slots of {context}')
    return {'tokens': tokens, 'context': context, 'slots': slots, 'displayed_concurrency': shown}


def qualifying_histories(bank, lengths, fresh_row=None, slots=1):
    for length in lengths:
        yield length, length, bank
    if fresh_row is None:
        return
    if type(fresh_row) is not int or not 1 <= fresh_row <= 32:
        raise ValueError('Fresh qualifying prompt must be one of the other 32 pinned public rows')
    selected = {'domains':{}}
    for domain in ('prose', 'code'):
        rows = bank['domains'][domain]['rows']
        if len(rows) <= fresh_row or any(rows[0][:length] == rows[fresh_row][:length] for length in lengths):
            raise ValueError(f'The {domain} fresh history repeats a checked prefix')
        count = slots if slots > 2 else 1
        if fresh_row + count > len(rows):
            raise ValueError('Insufficient distinct public rows for the requested qualification batch')
        selected['domains'][domain] = {'rows':rows[fresh_row:fresh_row+count]}
    for length in lengths:
        yield f'{length}-row-{fresh_row}', length, selected


def run_control(args, mode, target, draft, bank, devices, *, label=None):
    out = args.out / (label or mode)
    graph_dir = out / 'graphs'
    graph_dir.mkdir(parents=True)
    uuid = devices[0]['uuid']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, KAILLM_HUMMING_DETERMINISTIC='1',
               KAILLM_NATIVE_ASSETS=str(args.runtime / 'runtime-native-cache/native-assets.json'),
               OMP_NUM_THREADS=str(min(8, max(1, len(os.sched_getaffinity(0)) // 2))))
    env[ENV] = str(graph_dir)
    full = getattr(args, 'full_context', False)
    kv_gib = (args.plain_kv_gib if mode == 'plain' else args.native_kv_gib) if full else 1.5
    common_extra = ['--no-enable-prefix-caching', '--enable-prompt-tokens-details']
    cmd = [sys.executable, str(args.runtime / 'deployment/serve.py'), str(target),
           '--port', str(args.port), '--served-model-name', 'bench', '--spec-tokens', '0' if mode == 'plain' else '5',
           '--max-num-seqs', str(args.slots), '--max-model-len', str(CONTEXT if full else 4096), '--max-num-batched-tokens', '2048',
           '--kv-cache-memory-bytes', str(int(kv_gib * GIB)), '--gpu-memory-utilization', str(admission_fraction(devices))]
    if mode != 'plain':
        cmd += ['--dspark', str(draft)]

    def preview(extra):
        dry = subprocess.run([*cmd, '--extra-vllm-args', json.dumps(extra), '--dry-run'], env=env, text=True, capture_output=True, check=True)
        argv = json.loads(dry.stdout)['command']
        validate_kai_launch(argv)
        return argv
    argv = preview(common_extra)
    if mode == 'forced':
        spec = json.loads(argv[argv.index('--speculative-config') + 1])
        spec.update(rejection_sample_method='synthetic', synthetic_acceptance_rates=[0.] * 5)
        common_extra.extend(['--speculative-config', json.dumps(spec)])
        argv = preview(common_extra)
    cmd += ['--extra-vllm-args', json.dumps(common_extra)]
    write_json(out / 'launch.json', {'argv': argv, 'gpu': uuid, 'humming_deterministic': True,
                                    'runtime_lock': (args.runtime / 'benchmark-source.sha256').read_text().strip()})
    log = out / 'server.log'
    observed = getattr(args, 'check_overlap', False)
    if observed or args.slots > 2:
        observer = Path(__file__).with_name('verify_batch_schedule.py')
        cmd = [sys.executable, str(observer), 'serve', '--directory', str(out / 'schedule'), '--', *cmd]
    with log.open('w') as stream:
        proc = subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        url = f'http://127.0.0.1:{args.port}'
        wait_server(proc, url, log)
        if full:
            write_json(out / 'admission.json', full_admission(log.read_text(errors='replace'), args.slots))
        result = {}
        lengths = (128, 511, 2048, PROMPT) if full else (128, 511, 2048)
        for key, length, selected in qualifying_histories(bank, lengths, getattr(args, 'fresh_prompt_row', None), args.slots):
            before = snapshot(graph_dir, [uuid])
            if not before[uuid].get('head_sizes') or bool(before[uuid].get('draft_captures')) != (mode != 'plain'):
                raise ValueError('Required target output head or drafter graphs were not captured before the request')
            wide = {'schedule':out/'schedule', 'native':mode != 'plain'} if args.slots > 2 else {}
            sample = capture(url, selected, 'bench', args.slots, prompt_length=length, **wide)
            after = snapshot(graph_dir, [uuid])
            result[key] = sample
            write_json(out / f'outputs-{key}.json', sample)
            write_json(out / f'graph-proof-{key}.json', graph_proof(before, after, uuid, mode))
        if observed:
            import check_inkling_draft_overlap as overlap
            before = snapshot(graph_dir, [uuid])
            overlap.capture(SimpleNamespace(directory=out / 'schedule', bank=args.banks / 'inkling.json',
                                            out=out / 'overlap', mode=mode, url=url, model='bench', timeout=480))
            after = snapshot(graph_dir, [uuid])
            write_json(out / 'graph-proof-overlap.json', graph_proof(before, after, uuid, mode))
        return result
    finally:
        stop_server(proc)



def graph_proof(before, after, uuid, mode):
    proof = {'target': verify_interval(before, after, PHASES),
             'head': verify_head_interval(before, after, PHASES, [uuid])}
    if mode != 'plain':
        proof['drafter'] = verify_draft_interval(before, after, [uuid], require_proposal=True)
    return proof


def compare_plain_repeats(plain, repeated):
    if not plain or plain.keys() != repeated.keys():
        raise ValueError('The two plain servers did not run all the same prefixes')
    differences = []
    rows = 0
    for length, first in plain.items():
        second = repeated[length]
        if (first['prompt_hash'] != second['prompt_hash'] or first['client_concurrency'] != second['client_concurrency']
                or any(run['counts'] != {'draft_tokens':0, 'accepted_tokens':0} for run in (first, second))):
            raise ValueError('Plain repeat input or zero-draft control differs')
        slots = first['client_concurrency']
        if slots > 2 and (check_width(first, slots, False) != check_width(second, slots, False)):
            raise ValueError('Fresh plain controls ran different physical prompts')
        expected = set(request_labels(slots))
        if set(first['output_ids']) != expected or set(second['output_ids']) != expected:
            raise ValueError('Missing one or more qualifying request outputs')
        for domain in request_labels(slots):
            ids = first['output_ids'][domain]
            other = second['output_ids'][domain]
            if len(ids) != 48 or len(other) != len(ids):
                raise ValueError('Incomplete plain repeat output')
            rows += len(ids)
            for index, (a, b) in enumerate(zip(ids, other, strict=True)):
                if a != b:
                    differences.append({'prefix':length, 'domain':domain, 'first_index':index, 'expected':a, 'actual':b})
                    break
    return {'tokens_checked':rows, 'first_differences':differences}


def main(args):
    if getattr(args, 'check_overlap', False) and not (args.slots == 2 and args.full_context and args.repeat_plain):
        raise ValueError('Overlap qualification requires two full-context slots and an independent plain repeat')
    if args.slots > 2 and not (args.full_context and args.repeat_plain):
        raise ValueError('Wide-slot qualification requires verified full context and a fresh plain repeat')
    spec = MANIFEST['models'][args.checkpoint]
    if spec['family'] != 'inkling':
        raise ValueError('Expected an Inkling target')
    target = checkpoint_file(args.weights, args.checkpoint, spec)
    draft_key, draft_spec = dspark_asset(spec)
    draft = checkpoint_file(args.weights, draft_key, draft_spec)
    devices = selected_devices(gpu_inventory(), [args.device], visible_cuda_uuids())
    bank = json.loads((args.banks / 'inkling.json').read_bytes())
    args.out.mkdir(parents=True, exist_ok=False)
    for domain in ('prose', 'code'):
        verify_prompt_bank(bank, 'inkling', domain)
    inputs = {}
    for kind, path, publisher in [('target', target, spec), ('draft', draft, draft_spec)]:
        inputs[kind] = {'repo': publisher['repo'], 'revision': publisher['revision'],
                        'file': path.name, **publisher['files'][path.name]}
    write_json(args.out / 'inputs.json', inputs)
    plain = run_control(args, 'plain', target, draft, bank, devices)
    if args.repeat_plain:
        second = run_control(args, 'plain', target, draft, bank, devices, label='plain-repeat')
        agreement = compare_plain_repeats(plain, second)
        write_json(args.out / 'plain-repeat-comparison.json', agreement)
        if agreement['first_differences']:
            raise ValueError('Fresh plain baseline servers disagree; inspect plain-repeat-comparison.json')
    controls = [plain] + [run_control(args, mode, target, draft, bank, devices) for mode in ('native', 'forced')]
    verified = {length: compare(*(control[length] for control in controls)) for length in controls[0]}
    if getattr(args, 'check_overlap', False):
        import check_inkling_draft_overlap as overlap
        directories = [args.out / role / 'overlap' for role in ('plain', 'plain-repeat', 'native', 'forced')]
        write_json(args.out / 'overlap-comparison.json', overlap.compare(directories))
    write_json(args.out / 'comparison.json', verified)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runtime', 'weights', 'banks', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--checkpoint', choices=[key for key, model in MANIFEST['models'].items() if model['family'] == 'inkling'], required=True)
    parser.add_argument('--slots', type=int, choices=SLOTS, default=1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--port', type=int, default=18889)
    parser.add_argument('--fresh-prompt-row', type=int, choices=range(1, 33),
                        help='Also qualify independent histories from this verified public prompt-bank row')
    parser.add_argument('--repeat-plain', action='store_true',
                        help='Require two independent fresh plain servers to agree before starting the native controls')
    parser.add_argument('--check-overlap', action='store_true',
                        help='Verify simultaneous draft verification and mixed prompt/decode on all four validation servers')
    parser.add_argument('--full-context', action='store_true',
                        help='Require measured 73,728-token admission and add the 8,192-token prefix')
    parser.add_argument('--plain-kv-gib', type=float, default=3.0625, help='Plain pool with --full-context')
    parser.add_argument('--native-kv-gib', type=float, default=3.75, help='Native and forced pools with --full-context')
    args = parser.parse_args()
    if not all(0 < x < 1024 for x in (args.plain_kv_gib, args.native_kv_gib)):
        parser.error('Expected finite positive KV sizes below 1 TiB')
    main(args)
