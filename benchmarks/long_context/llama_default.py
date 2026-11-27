"""Verify llama.cpp's upstream CUDA execution policy without enabling the strict observer."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

from llama_graph import BINARY_NAMES, SOURCE_LOCK, certified, digest

WAVES = ('prefill-1', 'prefill-2', 'prefill-3', 'full')
BLOCKED = {'KAILLM_LLAMA_GRAPH_FILE', 'LD_PRELOAD', 'LD_AUDIT', 'CUDA_LAUNCH_BLOCKING'}


def overridden(key):
    return key in BLOCKED or key.startswith(('GGML_', 'LLAMA_'))


def prepare(server, revision, devices, env):
    directory, _ = certified(server, revision, devices)
    clean = {k: v for k, v in env.items() if not overridden(k)}
    clean['LD_LIBRARY_PATH'] = os.pathsep.join(filter(None, (str(directory), clean.get('LD_LIBRARY_PATH'))))
    return clean


def related_dso(name):
    return re.fullmatch(r'lib(?:ggml(?:-[\w-]+)?|llama(?:-[\w-]+)?|mtmd)\.so(?:\.[0-9]+)*', name) is not None


def process(pid, argv, revision, uuids, proc_root=Path('/proc')):
    """Check the executable, actual mapped inodes, and selected environment without retaining secrets."""
    root = proc_root / str(pid)
    directory, cert = certified(argv[0], revision, len(uuids))
    command = [os.fsdecode(x) for x in (root/'cmdline').read_bytes().split(b'\0') if x]
    if not command or command[1:] != argv[1:]:
        raise ValueError('Running llama.cpp arguments differ from the certified model and workload')
    actual = root/'exe'
    if not actual.samefile(directory/'llama-server'):
        raise ValueError('Running llama.cpp executable differs from its certificate')
    environment = dict(x.split(b'=', 1) for x in (root/'environ').read_bytes().split(b'\0') if b'=' in x)
    if any(overridden(os.fsdecode(k)) for k in environment):
        raise ValueError('The running llama.cpp process inherited an execution-policy override')
    visible = os.fsdecode(environment.get(b'CUDA_VISIBLE_DEVICES', b'')).split(',')
    libraries = os.fsdecode(environment.get(b'LD_LIBRARY_PATH', b'')).split(os.pathsep)
    if visible != uuids or libraries[0] != str(directory):
        raise ValueError('Running llama.cpp device visibility or library search path differs')
    mapped = {}
    for line in (root/'maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        _, perms, _, dev, inode, name = fields
        path = Path(name.removesuffix(' (deleted)'))
        if 'x' not in perms or not related_dso(path.name):
            continue
        if path.parent != directory or name.endswith(' (deleted)'):
            raise ValueError('llama.cpp loaded an unexpected server or GGML library')
        st = path.stat()
        if int(inode) != st.st_ino or tuple(int(n,16) for n in dev.split(':')) != (os.major(st.st_dev), os.minor(st.st_dev)):
            raise ValueError('Loaded llama.cpp library no longer matches its on-disk inode')
        mapped[path.name] = digest(path)
    for name in BINARY_NAMES[1:]:
        if mapped.get((directory/name).resolve().name) != cert['binary_sha256'][name]:
            raise ValueError('Missing certified llama.cpp CUDA or server implementation in process maps')
    # The executable and monotonic kernel start-time bind checks to one server through every wave.
    start = (root/'stat').read_text().rsplit(')', 1)[1].split()[19]
    return {'pid': pid, 'start_ticks': int(start), 'policy': 'default', 'upstream': revision,
            'source_sha256': cert['source_sha256'], 'binary_sha256': cert['binary_sha256'],
            'mapped_dso_sha256': mapped, 'observer_configured': False, 'environment_overrides': [],
            'cuda_visible_devices': uuids}


def counts(full, mode):
    if mode == 'plain':
        return None
    proposed = accepted = 0
    for req in full['requests']:
        info = req['server_timings']
        n, a = info.get('draft_n', 0), info.get('draft_n_accepted', 0)
        if ((n and 'draft_n_accepted' not in info) or type(n) is not int or type(a) is not int or not 0 <= a <= n):
            raise ValueError('Invalid llama.cpp full-generation speculative counters')
        proposed += n
        accepted += a
    if not proposed:
        raise ValueError('llama.cpp default requires positive full-generation proposals')
    return {'scope': 'full-wave', 'proposed': proposed, 'accepted': accepted, 'rejected': proposed-accepted}


def compact(raw, uuids, sha):
    p = raw.get('process') or {}
    if p.get('cuda_visible_devices') != uuids or not uuids or len(uuids) != len(set(uuids)):
        raise ValueError('Incorrect default-policy llama.cpp physical devices')
    if (type(p.get('pid')) is not int or p['pid'] <= 0 or type(p.get('start_ticks')) is not int or p['start_ticks'] <= 0
            or raw.get('measured_gguf_sha256') != sha):
        raise ValueError('Missing default-policy llama.cpp process or weight witness')
    return {**{k:p.get(k) for k in ('policy','upstream','source_sha256','binary_sha256','mapped_dso_sha256',
                                     'observer_configured','environment_overrides')}, 'wave_checks':raw.get('wave_checks'),
            'measured_gguf_sha256':sha, 'device_count':len(uuids), 'speculative':raw.get('speculative'),
            'microbatch':raw.get('microbatch'), 'startup_attempts':raw.get('startup_attempts')}


def _library_digests_match(certificate, loaded):
    if not isinstance(certificate, dict) or set(certificate) != set(BINARY_NAMES):
        return False
    if not isinstance(loaded, dict) or not loaded or not all(related_dso(name) for name in loaded):
        return False
    hashes = [*certificate.values(), *loaded.values()]
    if not all(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) for value in hashes):
        return False
    for required in BINARY_NAMES[1:]:
        observed = [value for name, value in loaded.items() if name == required or name.startswith(required + '.')]
        if observed != [certificate[required]]:
            return False
    return True


def verified(row):
    p=row.get('llama_default')
    if isinstance(p, dict) and 'published_client' in p:
        from published_llama import verified as published_verified
        return published_verified(row, p)
    if (not isinstance(p,dict) or row.get('engine')!='llama.cpp' or p.get('policy')!='default'
            or p.get('observer_configured') is not False or p.get('environment_overrides')!=[]
            or p.get('wave_checks')!=list(WAVES) or p.get('upstream')!=row.get('llama_source')
            or p.get('measured_gguf_sha256')!=row.get('sha256') or not row.get('sha256')
            or not row.get('gpu') or type(p.get('device_count')) is not int or p['device_count']!=len(row['gpu'])
            or any(key in row for key in ('cuda_graphs','target_head_cuda_graphs','drafter_cuda_graphs','llama_graph_launches_by_wave'))):
        return False
    if not valid_attempts(p.get('startup_attempts'), p.get('microbatch')):
        return False
    locks = json.loads(SOURCE_LOCK.read_bytes())
    revision = p['upstream']
    from benchmark import DSPARK_TOKENS, MANIFEST, dspark_digest, llama_revision, model_at_revision
    current = MANIFEST["models"].get(row.get("checkpoint"))
    if current and current.get("llama_revision_override") and revision != llama_revision(current):
        return False
    if revision not in locks or locks[revision] != p.get('source_sha256'):
        return False
    if not _library_digests_match(p.get('binary_sha256'), p.get('mapped_dso_sha256')):
        return False
    if row['mode'] not in ('plain', 'mtp5', *DSPARK_TOKENS, 'ngram1', 'ngram3'):
        return False
    if row['mode'] in DSPARK_TOKENS:
        try:
            expected = dspark_digest(model_at_revision(row['checkpoint'], row['model_revision']))
        except (KeyError, ValueError):
            return False
        if p.get('measured_drafter_sha256') != expected:
            return False
    spec = p.get('speculative')
    if row['mode'] == 'plain':
        return spec is None
    if not isinstance(spec, dict) or spec.get('scope') != 'full-wave':
        return False
    proposed, accepted, rejected = (spec.get(key) for key in ('proposed', 'accepted', 'rejected'))
    return (all(type(value) is int for value in (proposed, accepted, rejected))
            and proposed > 0 and min(accepted, rejected) >= 0 and accepted + rejected == proposed)


def validate_command(argv, mode, context, microbatch):
    from benchmark import DRAFT_TOKENS, DSPARK_TOKENS

    expected = {'-ngl':'999','--fit':'off','--flash-attn':'on','--cache-type-k':'bf16','--cache-type-v':'bf16',
                '--cache-ram':'0','-b':'2048','-ub':str(microbatch)}
    methods = {'plain':'none','ngram1':'ngram-map-k','ngram3':'ngram-simple','mtp5':'draft-mtp',
               **dict.fromkeys(DSPARK_TOKENS, 'draft-dspark')}
    if mode not in methods:
        raise ValueError('Unsupported default-policy llama.cpp mode')
    expected['--spec-type'] = methods[mode]
    draft = DRAFT_TOKENS[mode]
    if mode == 'mtp5' or mode in DSPARK_TOKENS:
        expected['--spec-draft-n-max'] = str(draft)
    if (context != 73728 + (draft+1 if draft else 0) or '--no-context-shift' not in argv or
            any(argv.count(k)!=1 or argv[argv.index(k)+1]!=v for k,v in expected.items())):
        raise ValueError('Default llama.cpp serving parameters differ from the matched benchmark policy')


MICROBATCHES = (512, 256, 128, 64)


def compute_oom(log):
    """Extract a failed CUDA allocation from a prefill compute-buffer reservation."""
    if not re.search(r'cudaMalloc failed: out of memory', log, re.I):
        return None
    if not re.search(r'failed to allocate compute(?: pp)? buffers', log, re.I):
        return None
    sizes = re.findall(r'failed to allocate CUDA\d+ buffer of size (\d+)', log)
    return max(map(int, sizes)) if sizes else None


def graph_launch_oom(log):
    # The CUDA backend reports the failing function on the lines following CUDA_CHECK.
    return bool(re.search(r'CUDA error: out of memory[^\n]*\n(?:[^\n]*\n){0,3}'
                          r'[^\n]*in function ggml_cuda_graph_launch_chunk\b', log, re.I))


def attention_launch_oom(log):
    """Confirm CUDA exhaustion inside Flash Attention, independently of a live probe."""
    return bool(re.search(r'CUDA error: out of memory[^\n]*\n(?:[^\n]*\n){0,3}'
                          r'[^\n]*in function launch_fattn\b', log, re.I))


def valid_attempts(attempts, chosen):
    if type(chosen) is not int or chosen not in MICROBATCHES or not isinstance(attempts, list):
        return False
    expected = MICROBATCHES[:MICROBATCHES.index(chosen) + 1]
    if len(attempts) != len(expected):
        return False
    for attempt, size in zip(attempts, expected, strict=True):
        if not isinstance(attempt, dict) or attempt.get('microbatch') != size:
            return False
        if size == chosen:
            ready = {'microbatch': size, 'status': 'ready'}
            return attempt == ready or attempt == {**ready, 'admission': 'separate-process'}
        if not isinstance(attempt.get('log_sha256'), str) or not re.fullmatch('[a-f0-9]{64}', attempt['log_sha256']):
            return False
        if attempt.get('status') == 'startup-attention-oom':
            if attempt.get('function') != 'launch_fattn':
                return False
        elif attempt.get('status') in ('prefill-graph-oom', 'prefill-attention-oom'):
            if attempt.get('prompt_tokens') != 8192 or attempt.get('probe_output_tokens') != 32:
                return False
        elif (attempt.get('status') != 'compute-buffer-oom' or type(attempt.get('requested_bytes')) is not int
              or attempt['requested_bytes'] <= 0):
            return False
    return False


def start_with_microbatches(start, stop, command, log, attempt_dir, save, *, probe=None):
    """Retry a certified stock server only for confirmed startup or first-request CUDA OOMs."""
    import shutil
    attempts = []
    slot = command.index('-ub') + 1
    if command[slot] != '512':
        raise ValueError('The normal initial llama.cpp microbatch must be 512 tokens')
    for size in MICROBATCHES:
        command[slot] = str(size)
        try:
            proc = start(command)
        except RuntimeError:
            if not log.exists():
                raise
            retained = attempt_dir / f'ub{size}.log'
            shutil.copyfile(log, retained)
            startup_log = retained.read_text(errors='replace')
            requested = compute_oom(startup_log)
            attention_oom = attention_launch_oom(startup_log)
            status = 'startup-failed'
            if requested is not None:
                status = 'compute-buffer-oom'
            elif attention_oom:
                status = 'startup-attention-oom'
            attempt = {'microbatch': size, 'status': status, 'log_sha256': digest(retained)}
            if requested is not None:
                attempt['requested_bytes'] = requested
            elif attention_oom:
                attempt['function'] = 'launch_fattn'
            attempts.append(attempt)
            save(attempt_dir/'attempts.json', attempts)
            if status == 'startup-failed' or size == MICROBATCHES[-1]:
                raise
            continue
        ready = {'microbatch': size, 'status': 'ready'}
        if probe is not None:
            try:
                probe()
            except Exception as exc:
                stop(proc)
                retained = attempt_dir / f'ub{size}.log'
                shutil.copyfile(log, retained)
                prefill_log = retained.read_text(errors='replace')
                status = 'prefill-failed'
                if graph_launch_oom(prefill_log):
                    status = 'prefill-graph-oom'
                elif attention_launch_oom(prefill_log):
                    status = 'prefill-attention-oom'
                attempts.append({'microbatch': size, 'status': status, 'prompt_tokens': 8192,
                                 'probe_output_tokens': 32, 'log_sha256': digest(retained)})
                save(attempt_dir/'attempts.json', attempts)
                if status == 'prefill-failed':
                    raise
                if size == MICROBATCHES[-1]:
                    raise RuntimeError(f"CUDA out of memory in an 8k prefill graph at the minimum physical microbatch ({size})") from exc
                continue
            except BaseException:
                stop(proc)
                raise
            stop(proc)
            # Warmup and timing must see the same initial state as every other configuration.
            # start() cleans up its child if the fresh process fails readiness.
            proc = start(command)
            ready['admission'] = 'separate-process'
        attempts.append(ready)
        try:
            save(attempt_dir/'attempts.json', attempts)
        except BaseException:
            stop(proc)
            raise
        return proc, attempts
    raise AssertionError('Exhausted default llama.cpp microbatch sequence')
