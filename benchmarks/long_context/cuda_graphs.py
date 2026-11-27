"""Verify actual per-GPU target graph replays outside benchmark timing intervals."""
import json
from pathlib import Path
import struct


ENV = 'KAILLM_BENCHMARK_GRAPH_DIR'
FIELDS = ('scheduled', 'graph_replays', 'eager_steps', 'unclassified_steps')
PHASES = ('prefill', 'decode')
COUNTERS = struct.Struct('<8Q')
DRAFT_FIELDS = ('proposals','completed_proposals','graph_replays','eager_steps','unclassified_steps')
DRAFT_COUNTERS = struct.Struct('<5Q')
HEAD_COUNTERS = struct.Struct('<4Q')
NEURAL_MODES = {'mtp5', 'dspark3', 'dspark5', 'dspine7', 'dspine15'}


def gpu_draft_mode(engine, mode):
    return mode in NEURAL_MODES or (engine == "kaiLLM" and mode in ("ngram1", "ngram3"))


def output_head_devices(uuids, stages):
    if type(stages) is not int or stages < 1 or not uuids or len(uuids) % stages:
        raise ValueError("Invalid GPU layout for the target output projection")
    return uuids[-(len(uuids) // stages):]


def pipeline_stages(command):
    extra = json.loads(command[command.index("--extra-vllm-args") + 1]) if "--extra-vllm-args" in command else []
    launch = [*command, *extra]
    return int(launch[launch.index("--pipeline-parallel-size") + 1]) if "--pipeline-parallel-size" in launch else 1


def validate_kai_launch(command):
    mode = None
    for index, token in enumerate(command):
        option, separator, inline = token.partition('=')
        option = option.replace('_', '-')
        if option == '--enforce-eager' and (not separator or inline.lower() not in ('false', '0')):
            raise ValueError('kaiLLM graph-only benchmarking cannot request eager model execution')
        if option in ('-cc', '--compilation-config', '-cc.cudagraph-mode', '--compilation-config.cudagraph-mode'):
            value = inline if separator else command[index+1]
            if option.endswith('.cudagraph-mode'):
                mode = value.strip('"')
            else:
                mode = json.loads(value).get('cudagraph_mode')
    if mode != 'FULL':
        raise ValueError('kaiLLM must select FULL CUDA graphs for both prefill and decode before benchmarking')


def snapshot(directory, uuids):
    if directory is None:
        raise ValueError('CUDA graph evidence directory is required before sending benchmark requests')
    result = {}
    for path in Path(directory).glob('*.json'):
        meta = json.loads(path.read_text())
        uuid = meta['uuid']
        if (meta['schema'] != 1 or meta['mode'] != 'FULL' or meta['enforce_eager'] is not False
                or not meta['captured_token_sizes'] or not meta['prefill_capture_sizes']
                or max(meta['prefill_capture_sizes']) < meta['max_prefill_tokens'] or uuid in result
                or path.stem != f"{uuid}.{meta['pid']}"):
            raise ValueError('Missing or ambiguous full CUDA graph capture on a benchmark device')
        values = COUNTERS.unpack(path.with_suffix('.counts').read_bytes())
        result[uuid] = {'pid': meta['pid'], 'captures': meta['captured_token_sizes'],
                        **{phase: dict(zip(FIELDS, values[i*4:i*4+4], strict=True)) for i, phase in enumerate(PHASES)}}
        head_sizes = meta.get('output_head_capture_sizes', [])
        if not isinstance(head_sizes, list) or any(type(size) is not int or size <= 0 for size in head_sizes):
            raise ValueError('Invalid target output-head graph inventory')
        result[uuid]['head_sizes'] = head_sizes
        if head_sizes:
            values = HEAD_COUNTERS.unpack(path.with_suffix('.head-counts').read_bytes())
            result[uuid]['head'] = {phase: {'scheduled': values[i*2], 'graph_replays': values[i*2+1],
                                            'eager_steps': 0, 'unclassified_steps': 0}
                                    for i, phase in enumerate(PHASES)}
        captures = meta.get('draft_graph_captures', 0)
        if type(captures) is not int or captures < 0:
            raise ValueError('Invalid draft graph inventory')
        result[uuid]['draft_captures'] = captures
        if captures:
            values = DRAFT_COUNTERS.unpack(path.with_suffix('.draft-counts').read_bytes())
            result[uuid]['draft'] = dict(zip(DRAFT_FIELDS, values, strict=True))
    if set(result) != set(uuids) or len(result) != len(uuids):
        raise ValueError('No positive CUDA graph capture proof from every selected GPU')
    return result


def verify_interval(before, after, required):
    if set(before) != set(after) or not before:
        raise ValueError('CUDA graph evidence lost a participating device')
    counts = {phase: {} for phase in PHASES}
    for uuid, first in before.items():
        last = after[uuid]
        if first['pid'] != last['pid'] or first['captures'] != last['captures']:
            raise ValueError('CUDA graph process or capture inventory changed within a wave')
        for phase in PHASES:
            diffs = {key: last[phase][key] - first[phase][key] for key in FIELDS}
            if (min(diffs.values()) < 0 or diffs['scheduled'] != diffs['graph_replays']
                    or diffs['eager_steps'] or diffs['unclassified_steps'] or (phase in required and not diffs['graph_replays'])):
                raise ValueError(f'Incomplete {phase} CUDA graph coverage on {uuid}')
            counts[phase][uuid] = diffs['graph_replays']
    return counts


def proof_record(intervals):
    # Each prefill probe and the full request's prefill are independently verified.
    proof = {}
    for phase in PHASES:
        rows = [(label, item[phase]) for label, item in intervals if phase == 'prefill' or label == 'full']
        proof[phase] = {'scope': 'timed-wave', 'graph_replays': sum(sum(row.values()) for _, row in rows),
                        'eager_steps': 0, 'unclassified_steps': 0,
                        'per_wave_by_gpu': {label: row for label, row in rows}}
    return proof


def verify_draft_interval(before, after, uuids, require_proposal):
    participants = {uuid for uuid, item in before.items() if item.get('draft_captures')}
    if not participants or participants != set(uuids) or len(uuids) != len(participants):
        raise ValueError('Missing or unexpected neural draft GPU')
    if participants != {uuid for uuid, item in after.items() if item.get('draft_captures')}:
        raise ValueError('Neural draft device changed within wave')
    counts = {}
    for uuid in uuids:
        first, last = before[uuid], after[uuid]
        if first['pid'] != last['pid'] or first['draft_captures'] != last['draft_captures']:
            raise ValueError('Neural draft server changed during a wave')
        diff = {k: last['draft'][k] - first['draft'][k] for k in DRAFT_FIELDS}
        if (min(diff.values()) < 0 or diff['proposals'] != diff['completed_proposals']
                or diff['graph_replays'] < diff['proposals'] or diff['eager_steps'] or diff['unclassified_steps']
                or bool(diff['graph_replays']) != bool(diff['proposals']) or (require_proposal and not diff['proposals'])):
            raise ValueError(f'Incomplete neural draft graph execution on {uuid}')
        counts[uuid] = {key: diff[key] for key in DRAFT_FIELDS[:3]}
    return counts


def draft_proof_record(intervals):
    return {'scope': 'timed-wave', 'eager_steps': 0, 'unclassified_steps': 0,
            **{key: sum(obs[key] for _, wave in intervals for obs in wave.values()) for key in DRAFT_FIELDS[:3]},
            'per_wave_by_gpu': dict(intervals)}


def verify_head_interval(before, after, required, uuids):
    def head_snapshot(data):
        participants = {uid for uid, item in data.items() if item.get('head_sizes')}
        if not uuids or set(uuids) != participants or len(uuids) != len(participants):
            raise ValueError('Missing target output-head graph participant')
        return {uid: {'pid': data[uid]['pid'], 'captures': data[uid]['head_sizes'], **data[uid]['head']} for uid in uuids}
    return verify_interval(head_snapshot(before), head_snapshot(after), required)
