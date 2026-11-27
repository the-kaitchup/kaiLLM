"""Read completed model dispatches under the standard, source-pinned plain serving policies."""
import json
from pathlib import Path
import struct

ENV = 'KAILLM_BENCHMARK_DEFAULT_DIR'
MODES = ('full', 'direct', 'other', 'direct_head')
PIECEWISE_MODES = (*MODES, 'piecewise')
PHASES = ('prefill', 'decode')
WAVES = ('prefill-1', 'prefill-2', 'prefill-3', 'full')


def pins(manifest, model, mode):
    if mode != 'plain':
        return ()
    catalog = manifest['kaillm']
    entries = (*catalog.get('default_dispatch_runtimes', ()), catalog.get('default_dispatch_runtime', {}))
    return tuple((entry['revision'], entry['source']) for entry in entries if model in entry.get('models', ()))


def runtime(manifest, model, mode, revision):
    return next((pin for pin in pins(manifest, model, mode) if pin[0] == revision), None)


CONFIG_MODES = {'inkling': 'FULL_DECODE_ONLY', 'lfm': 'FULL_AND_PIECEWISE'}


def launch(command, family='inkling'):
    if '--enforce-eager' in command or '--speculative-config' in command or '--compilation-config' not in command:
        raise ValueError('Default timing requires ordinary plain decoding with CUDA graphs enabled')
    config = json.loads(command[command.index('--compilation-config') + 1])
    if family not in CONFIG_MODES or config.get('cudagraph_mode') != CONFIG_MODES[family]:
        raise ValueError('Expected the standard ' + family + ' capture policy')
    if family == 'lfm':
        if command.count('--additional-config') != 1 or json.loads(command[command.index('--additional-config')+1]).get('cudagraph_specialize_prefill') is not True:
            raise ValueError('Expected ordinary LFM3 prefill capture configuration')
    return CONFIG_MODES[family]


def read(folder, uuids, family='inkling'):
    values = {}
    for file in Path(folder).glob('*.json'):
        meta = json.loads(file.read_bytes())
        uuid, pid = meta['uuid'], meta['pid']
        fields = MODES if meta.get('schema') == 1 else PIECEWISE_MODES if meta.get('schema') == 2 else ()
        if (not fields or family not in CONFIG_MODES or meta.get('config_mode') != CONFIG_MODES[family]
                or meta.get('enforce_eager') is not False or meta.get('fields') != list(fields)
                or uuid in values or type(pid) is not int or pid <= 0 or file.stem != f'{uuid}.{pid}'):
            raise ValueError('Unexpected default-policy observer identity or mode')
        values[uuid] = (pid, struct.unpack(f'<{2*len(fields)}Q', file.with_suffix('.dispatch-counts').read_bytes()))
    if set(values) != set(uuids) or len(values) != len(uuids):
        raise ValueError('Missing default-policy CUDA device')
    return values


def interval(first, last, required):
    if first.keys() != last.keys():
        raise ValueError('The serving devices changed mid-wave')
    result = {}
    for uuid, (pid, before) in first.items():
        after_pid, after = last[uuid]
        if after_pid != pid:
            raise ValueError('The serving worker restarted mid-wave')
        delta = [b - a for a, b in zip(before, after, strict=True)]
        if len(delta) not in (2 * len(MODES), 2 * len(PIECEWISE_MODES)):
            raise ValueError('Invalid number of default dispatch counters')
        fields = MODES if len(delta) == 2 * len(MODES) else PIECEWISE_MODES
        n = len(fields)
        phases = {name: dict(zip(fields, delta[i*n:(i+1)*n], strict=True)) for i, name in enumerate(PHASES)}
        if not valid_phases(phases, required):
            raise ValueError('Missing, contradicting or unknown dispatch in a timed wave')
        result[uuid] = phases
    return result


def valid_phases(phases, required):
    if not isinstance(phases, dict) or set(phases) != set(PHASES):
        return False
    for phase in PHASES:
        row = phases[phase]
        if not isinstance(row, dict) or set(row) not in (set(MODES), set(PIECEWISE_MODES)) or any(type(n) is not int or n < 0 for n in row.values()):
            return False
        count = row['full'] + row['direct'] + row.get('piecewise', 0)
        if row['other'] or row['direct_head'] > count:
            return False
        if phase in required:
            if not count or not row['direct_head'] or (phase == 'decode' and not row['full']):
                return False
        elif count:
            return False
    return True


def compact(raw, uuids):
    if not isinstance(raw, dict) or set(raw) != set(WAVES):
        raise ValueError('Incomplete default-policy timing')
    out = {}
    for wave in WAVES:
        observations = raw[wave]
        if set(observations) != set(uuids) or len(set(uuids)) != len(uuids):
            raise ValueError('Default dispatches do not cover the selected devices')
        out[wave] = [observations[u] for u in uuids]
    return out


def verified(row, manifest):
    from empirical_capacity import capacity_valid
    if not capacity_valid(row):
        return False
    family = manifest.get('models', {}).get(row.get('checkpoint'), {}).get('family','inkling')
    if family == 'lfm':
        if (row.get('kai_default_config_mode') != CONFIG_MODES['lfm'] or row.get('kai_humming_deterministic') is not False
                or row.get('kai_empirical_capacity') or row.get('mode') != 'plain'):
            return False
    elif family != 'inkling' or row.get('kai_humming_deterministic') is not True:
        return False
    evidence = row.get('kai_default_dispatch')
    if (row.get('engine') != 'kaiLLM' or not evidence or not isinstance(evidence, dict)
            or row.get('kai_enforce_eager')
            or not row.get('sha256') or row.get('measured_gguf_sha256') != row['sha256']
            or any(k in row for k in ('cuda_graphs','target_head_cuda_graphs','target_head_direct','drafter_cuda_graphs'))
            or runtime(manifest, row['checkpoint'], row['mode'], row['kai_source']) != (row['kai_source'], row['engine_revision'])
            or set(evidence) != set(WAVES) or len(row.get('gpu', ())) != 1):
        return False
    for wave in WAVES:
        ranks=evidence[wave]
        if (not isinstance(ranks, list) or len(ranks) != 1
                or not valid_phases(ranks[0], PHASES if wave == 'full' else ('prefill',))):
            return False
    return True
