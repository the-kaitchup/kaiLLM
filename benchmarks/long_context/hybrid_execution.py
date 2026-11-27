"""Physical per-rank evidence for a graphed backbone with direct vocabulary projection."""
import hashlib
import json
from pathlib import Path

from cuda_graphs import HEAD_COUNTERS, PHASES, output_head_devices, pipeline_stages, snapshot

HEAD_ENV = 'KAILLM_BENCHMARK_HEAD_EXECUTION'


def head_policy(engine, mode, choice):
    if choice not in ('graph', 'direct') or (choice == 'direct' and (engine, mode) != ('kaiLLM', 'plain')):
        raise ValueError('Direct vocabulary projection is available for kaiLLM plain only')
    return choice


def verify_files(files, spec):
    digests = []
    for path in files:
        with path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        if digest != spec['files'][path.name]['sha256']:
            raise ValueError(f'Direct-head benchmark checkpoint digest differs for {path.name}')
        digests.append(digest)
    return digests


def read_direct(directory, uuids):
    # snapshot() verifies physical UUIDs, the FULL backbone, and its actual counters.
    state = snapshot(directory, uuids)
    for uid, item in state.items():
        path = Path(directory) / f"{uid}.{item['pid']}.json"
        meta = json.loads(path.read_bytes())
        if item['head_sizes'] or item['draft_captures']:
            raise ValueError('Direct-head plain run reported captured head or proposer graphs')
        route = meta.get('output_head_execution')
        if route is not None:
            if route != 'direct':
                raise ValueError('Unknown observed output-head execution route')
            values = HEAD_COUNTERS.unpack(path.with_suffix('.direct-head-counts').read_bytes())
            item['direct_head'] = dict(zip(PHASES, (values[:2], values[2:]), strict=True))
    return state


def direct_devices(state, uuids, argv):
    expected = output_head_devices(uuids, pipeline_stages(argv))
    if [uid for uid in uuids if 'direct_head' in state[uid]] != expected:
        raise ValueError('Direct target head must cover every rank of the final pipeline stage')
    return expected


def direct_interval(before, after, phases, devices, backbone):
    """Match each completed direct head call to its real terminal-rank backbone launch."""
    result = {phase: {} for phase in PHASES}
    for state in (before, after):
        if {uid for uid, r in state.items() if 'direct_head' in r} != set(devices) or not devices:
            raise ValueError('Direct target head changed participating ranks')
    for uid in devices:
        if before[uid]['pid'] != after[uid]['pid']:
            raise ValueError('Direct target head process changed during a wave')
        for phase in PHASES:
            p = before[uid]['direct_head'][phase]
            q = after[uid]['direct_head'][phase]
            attempted, completed = q[0] - p[0], q[1] - p[1]
            if (completed < 0 or completed != attempted or completed != backbone[phase][uid]
                    or (phase in phases and not completed)):
                raise ValueError(f'Unmatched direct target head execution during {phase} on {uid}')
            result[phase][uid] = completed
    return result


def direct_record(waves):
    result = {}
    for phase in PHASES:
        items = {label: counts[phase] for label, counts in waves if phase == 'prefill' or label == 'full'}
        result[phase] = {'scope': 'timed-wave', 'execution': 'direct',
                         'completed_calls': sum(sum(devices.values()) for devices in items.values()),
                         'per_wave_by_gpu': items}
    return result
