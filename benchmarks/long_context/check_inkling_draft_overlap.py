"""Validate concurrent DSpark verification and mixed prefill/decode with two real Inkling requests."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from benchmark import MANIFEST, dspark_digest, verify_prompt_bank, write_json
from verify_batch_schedule import prompt_hash, request_scores, valid_scores
from verify_inkling_drafter import counters


CASES = {
    'joint': [('prose', 0, 511), ('code', 1, 511)],
    'code-prefill': [('prose', 1, 128), ('code', 0, 8192)],
    'prose-prefill': [('code', 1, 128), ('prose', 0, 8192)],
}
STEPS = 48


def observed_steps(trace, offset):
    if not trace.is_file():
        return []
    with trace.open('rb') as source:
        source.seek(offset)
        return [json.loads(line) for line in source if line.endswith(b'\n')]


def overlap(events, fingerprints, lengths, native):
    slots = len(fingerprints)
    if slots not in (2, 4, 8, 16, 32) or len(set(fingerprints)) != slots or len(lengths) != slots:
        raise ValueError('Expected a distinct prompt for each request slot')
    indices = {}
    remaining = list(lengths)
    joint = mixed = full_width = 0
    joint_token_counts = {}
    widths = [0] * slots
    for event in events:
        for request, digest in event['admissions']:
            if request in indices or digest not in fingerprints or fingerprints.index(digest) in indices.values():
                raise ValueError('Unexpected request admission')
            indices[request] = fingerprints.index(digest)
        work = dict(event['scheduled'])
        drafts = dict(event.get('draft_widths', []))
        if (len(work) != len(event['scheduled']) or len(drafts) != len(event.get('draft_widths', []))
                or set(drafts) - work.keys() or set(work) - indices.keys() or (not native and drafts)):
            raise ValueError('Unexpected request schedule or draft')
        decode, prefill = set(), set()
        for request, n in work.items():
            slot = indices[request]
            width = drafts.get(request, 0)
            if type(n) is not int or n < 1 or type(width) is not int or not 0 <= width <= min(5, n-1):
                raise ValueError('Invalid DSpark verification width')
            if remaining[slot]:
                if width:
                    raise ValueError('Draft verification began before the request completed its prompt')
                prefill.add(slot)
                remaining[slot] = max(0, remaining[slot] - n)
            elif (native and width) or (not native and n == 1):
                decode.add(slot)
                widths[slot] += width
        if len(decode) == slots:
            joint += 1
            scheduled = str(sum(work.values()))
            joint_token_counts[scheduled] = joint_token_counts.get(scheduled, 0) + 1
            full_width += len(drafts) == slots and all(drafts[request] == 5 and work[request] == 6 for request in drafts)
        mixed += bool(decode and prefill)
    if len(indices) != slots or any(remaining):
        raise ValueError('Missing or incomplete scheduled prompt')
    return dict(joint_steps=joint, mixed_steps=mixed, scheduled_draft_tokens=widths,
                joint_scheduled_token_counts=joint_token_counts, joint_five_proposal_steps=full_width)


def required_overlap(name, native):
    if name != 'joint':
        return 'mixed_steps'
    return 'joint_five_proposal_steps' if native else 'joint_steps'


def measured_case(args, name, bank):
    layout = CASES[name]
    prompts = [bank['domains'][domain]['rows'][row][:size] for domain, row, size in layout]
    fingerprints = list(map(prompt_hash, prompts))
    native = args.mode in ('native', 'forced')
    marker = args.directory / 'recording'
    trace = args.directory / 'steps.jsonl'
    offset = trace.stat().st_size if trace.exists() else 0
    before = counters(args.url, required=native)
    marker.touch(exist_ok=False)
    try:
        with ThreadPoolExecutor(2) as executor:
            def submit(tokens):
                return executor.submit(request_scores, args.url, args.model, [tokens], STEPS, args.timeout)
            first = submit(prompts[0])
            if name != 'joint':
                # Wait for the short request to finish prefill. Submit the long request while it generates.
                deadline = time.monotonic() + args.timeout
                while time.monotonic() < deadline:
                    events = observed_steps(trace, offset)
                    if len(events) >= 2:
                        break
                    if first.done():
                        first.result()
                        raise RuntimeError('First request finished before a second request could be staggered')
                    time.sleep(.002)
                else:
                    raise TimeoutError('No first decode step observed')
            second = submit(prompts[1])
            answers = [first.result()[0], second.result()[0]]
    finally:
        marker.unlink(missing_ok=True)
    after = counters(args.url, required=native)
    events = observed_steps(trace, offset)
    return dict(prompt_sha256=fingerprints, lengths=list(map(len, prompts)), answers=answers, events=events,
                counter_delta={k:after[k]-before[k] for k in before}, overlap=overlap(events,fingerprints,list(map(len,prompts)),native))


def validate_launch(mode, identity):
    argv = identity['arguments']
    def last(name):
        values = [argv[i+1] for i, flag in enumerate(argv[:-1]) if flag == name]
        return values[-1] if values else None

    if (last('--max-num-seqs') != '2' or last('--max-model-len') != '73728'
            or identity['humming_deterministic'] != '1'):
        raise ValueError('Expected a two-slot, full-length launch with deterministic Humming')
    spec = json.loads(last('--speculative-config') or 'null')
    if mode == 'plain':
        if spec:
            raise ValueError('Plain control has drafting enabled')
        return
    models = list(MANIFEST['models'].values())
    for archived in MANIFEST.get('model_history', {}).values():
        models.extend(archived.values())
    matches = {dspark_digest(model) for model in models if model['family'] == 'inkling'
               and [value['sha256'] for value in model['files'].values()] == [identity['checkpoint_sha256']]}
    if len(matches) != 1:
        raise ValueError('The observed target bytes do not identify a unique pinned Inkling drafter')
    digest, = matches
    if (not spec or spec.get('num_speculative_tokens') != 5 or spec.get('method') not in ('dspark','dflash')
            or not Path(spec.get('model','')).name.startswith(digest + '-')):
        raise ValueError('Expected the pinned DSpark GGUF and five verification positions')
    policy = spec.get('rejection_sample_method','standard')
    if mode == 'forced' and (policy != 'synthetic' or spec.get('synthetic_acceptance_rates') != [0]*5):
        raise ValueError('Forced control must reject every proposal')
    if mode == 'native' and policy != 'standard':
        raise ValueError('Native control must use its ordinary acceptance policy')


def capture(args):
    bank = json.loads(args.bank.read_bytes())
    for domain in ('prose', 'code'):
        verify_prompt_bank(bank, 'inkling', domain)
    identity = json.loads((args.directory / 'launch.json').read_bytes())
    validate_launch(args.mode, identity)
    server = json.loads((args.directory / 'server-start.json').read_bytes())
    if len(server.get('runtime_source_lock', '')) != 64:
        raise ValueError('Expected a source-locked validation server')
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / 'identity.json', dict(mode=args.mode, launch=identity, server=server))
    for name in CASES:
        for attempt in range(3):
            result = measured_case(args, name, bank)
            write_json(args.out / f'{name}-{attempt}.json', result)
            metric = required_overlap(name, args.mode in ('native', 'forced'))
            if result['overlap'][metric]:
                write_json(args.out / (name + '.json'), result)
                break
        else:
            raise ValueError('No observed two-slot ' + name + ' schedule')


def verify_proposals(mode, counts, widths):
    proposed, accepted = (counts[k] for k in ('draft_tokens', 'accepted_tokens'))
    if not 0 <= accepted <= proposed:
        raise ValueError('Invalid acceptance counts')
    if mode == 'plain':
        if proposed:
            raise ValueError('Plain control reported proposals')
        return
    if not proposed or not all(widths):
        raise ValueError('Both concurrent requests must reach the native verifier')
    if mode == 'native' and not 0 < accepted < proposed:
        raise ValueError('The native control must accept and reject proposals')
    if mode == 'forced' and accepted:
        raise ValueError('A proposal passed the forced-rejection control')


def compare(directories):
    modes = ('plain', 'plain', 'native', 'forced')
    provenance = [json.loads((d / 'identity.json').read_bytes()) for d in directories]
    if [p['mode'] for p in provenance] != list(modes):
        raise ValueError('Expected plain, plain repeat, native, and forced controls in order')
    for field in ('checkpoint_sha256', 'cuda_uuid', 'humming_deterministic'):
        if len({p['launch'][field] for p in provenance}) != 1:
            raise ValueError('Different ' + field + ' in the four controls')
    if (len({p['server']['id'] for p in provenance}) != 4
            or len({p['server']['runtime_source_lock'] for p in provenance}) != 1):
        raise ValueError('Expected four fresh servers using the same runtime source')
    for p in provenance:
        validate_launch(p['mode'], p['launch'])
    if provenance[0]['launch'] != provenance[1]['launch']:
        raise ValueError('The plain repeat has different launch arguments')

    totals = {}
    for name in CASES:
        records = [json.loads((d / (name + '.json')).read_bytes()) for d in directories]
        control = records[0]
        scores_changed = []
        for mode, record in zip(modes, records):
            if any(control[k] != record[k] for k in ('prompt_sha256', 'lengths')):
                raise ValueError('Inputs differ for ' + name)
            verified = overlap(record['events'], record['prompt_sha256'], record['lengths'], mode != 'plain')
            metric = required_overlap(name, mode in ('native', 'forced'))
            if verified != record['overlap'] or not verified[metric]:
                raise ValueError('Missing observed request overlap for ' + name)
            verify_proposals(mode, record['counter_delta'], verified['scheduled_draft_tokens'])
            if len(record['answers']) != 2:
                raise ValueError('Missing slot')
            changed = 0
            for ref, answer in zip(control['answers'], record['answers']):
                if (len(answer['tokens']) != STEPS or len(answer['scores']) != STEPS
                        or any(not valid_scores(t, s) for t, s in zip(answer['tokens'], answer['scores']))):
                    raise ValueError('Incomplete token or score capture')
                if ref['tokens'] != answer['tokens']:
                    raise ValueError('Greedy output differs in ' + mode + '/' + name)
                changed += sum(a != b for a, b in zip(ref['scores'], answer['scores']))
            scores_changed.append(changed)
        totals[name] = dict(matched_greedy_positions=2 * STEPS, changed_top_eight_positions=scores_changed,
                            overlaps=[r['overlap'] for r in records], counters=[r['counter_delta'] for r in records])
    return totals


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    subs = cli.add_subparsers(dest='action', required=True)
    cap = subs.add_parser('capture')
    for name in ('directory', 'bank', 'out'):
        cap.add_argument('--' + name, required=True, type=Path)
    cap.add_argument('--mode', required=True, choices=('plain', 'native', 'forced'))
    cap.add_argument('--url', required=True)
    cap.add_argument('--model', default='bench')
    cap.add_argument('--timeout', type=int, default=480)
    check = subs.add_parser('compare')
    check.add_argument('directories', type=Path, nargs=4, metavar='CONTROL_DIR')
    check.add_argument('--out', type=Path, required=True)
    args = cli.parse_args()
    if args.action == 'capture':
        capture(args)
    else:
        write_json(args.out, compare(args.directories))
