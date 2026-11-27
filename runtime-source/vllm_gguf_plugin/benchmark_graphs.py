"""Optional per-GPU proof of full CUDA graph launches for the long-context benchmark."""
import json
import ctypes
import mmap
import os
from pathlib import Path
import struct
from uuid import UUID


ENV = 'KAILLM_BENCHMARK_GRAPH_DIR'
FIELDS = ('scheduled', 'graph_replays', 'eager_steps', 'unclassified_steps')
PHASES = ('prefill', 'decode')
COUNTER = struct.Struct('<Q')
DRAFT_FIELDS = ('proposals', 'completed_proposals', 'graph_replays', 'eager_steps', 'unclassified_steps')


def physical_gpu_uuid(device):
    import torch
    ordinal = torch.cuda.current_device() if device.index is None else device.index
    driver = ctypes.CDLL('libcuda.so.1')
    handle = ctypes.c_int()
    value = (ctypes.c_ubyte * 16)()
    get_uuid = getattr(driver, 'cuDeviceGetUuid_v2', None) or driver.cuDeviceGetUuid
    if driver.cuDeviceGet(ctypes.byref(handle), ordinal) or get_uuid(ctypes.byref(value), handle):
        raise RuntimeError('Could not bind the graph observer to the physical CUDA device')
    return 'GPU-' + str(UUID(bytes=bytes(value)))


class TargetGraphCounters:
    def __init__(self, directory, gpu_uuid, mode, eager):
        if mode != 'FULL' or eager:
            raise RuntimeError('Graph-only benchmarking requires FULL CUDA graphs for prefill and decode')
        self.directory = Path(directory)
        self.gpu_uuid = gpu_uuid
        self.pid = os.getpid()
        self.path = self.directory / f'{gpu_uuid}.{self.pid}'
        with Path(str(self.path)+'.counts').open('xb+') as stream:
            size = len(PHASES) * len(FIELDS) * COUNTER.size
            stream.truncate(size)
            self.counts = mmap.mmap(stream.fileno(), size)
        self.mode = mode
        self.head_counts = None
        self.head = None
        self.direct_head = None
        self.draft_counts = None
        self.draft_active = False
        self.draft_launches = 0
        self.draft_captures = 0

    @classmethod
    def from_config(cls, config, device):
        directory = os.environ.get(ENV)
        if not directory:
            return None
        mode = config.compilation_config.cudagraph_mode.name
        return cls(directory, physical_gpu_uuid(device), mode, config.model_config.enforce_eager)

    def install_head(self, hidden, max_rows, project):
        route = os.environ.get('KAILLM_BENCHMARK_HEAD_EXECUTION', 'graph')
        if route == 'graph':
            from .benchmark_output_head import OutputHeadGraphs
            self.head = OutputHeadGraphs(hidden, max_rows, project)
            suffix = '.head-counts'
        elif route == 'direct':
            self.direct_head = project
            suffix = '.direct-head-counts'
        else:
            raise ValueError(f'Unknown benchmark head execution {route!r}')
        with Path(str(self.path) + suffix).open('xb+') as stream:
            size = len(PHASES) * 2 * COUNTER.size
            stream.truncate(size)
            self.head_counts = mmap.mmap(stream.fileno(), size)

    def _head_event(self, phases, field):
        if not phases:
            raise RuntimeError('Unsigned target output-head phase')
        for phase in phases:
            offset = (phase * 2 + field) * COUNTER.size
            value, = COUNTER.unpack_from(self.head_counts, offset)
            COUNTER.pack_into(self.head_counts, offset, value+1)

    def head_scheduled(self, phases):
        self._head_event(phases, 0)

    def head_replayed(self, phases):
        self._head_event(phases, 1)

    def project_head(self, hidden, phases):
        if self.direct_head is not None:
            # The same bound function used by the normal sampler, on its exact rows.
            # Empty shards produce no measured projection. An exception never completes one.
            rows = hidden.shape[0]
            if rows:
                self._head_event(phases, 0)
            logits = self.direct_head(hidden)
            if logits is None or logits.shape[0] != rows:
                raise RuntimeError('Direct target head did not return the scheduled vocabulary rows')
            if rows:
                self._head_event(phases, 1)
            return logits
        if self.head is None:
            raise RuntimeError('No captured target output head')
        return self.head.project(hidden, phases, self)

    def install_drafter(self, proposer):
        # Called after CUDA graph initialization, before admitting real requests.
        install = getattr(proposer, 'install_benchmark_graphs', None)
        captures = install(self) if install else self.install_mtp(proposer)
        if type(captures) is not int or captures < 1:
            raise RuntimeError('No captured GPU proposer graphs')
        self.draft_captures = captures
        with Path(str(self.path)+'.draft-counts').open('xb+') as stream:
            size = len(DRAFT_FIELDS) * COUNTER.size
            stream.truncate(size)
            self.draft_counts = mmap.mmap(stream.fileno(), size)

    def observe_manager(self, manager):
        if not manager or not manager.graphs or any(d.cg_mode.name != 'FULL' for d in manager.graphs):
            raise RuntimeError('Missing a full neural draft graph')
        manager.benchmark_draft_observer = self
        return len(manager.graphs)

    def install_mtp(self, proposer):
        # Only the audited upstream autoregressive MTP path. Its outer proposal
        # stages previously computed hidden states; all neural ops, LM heads,
        # and sampling are inside its prefill/decode FULL graph routines.
        cls = type(proposer)
        if (cls.__module__, cls.__name__) != ('vllm.v1.worker.gpu.spec_decode.mtp.speculator', 'MTPSpeculator'):
            raise RuntimeError('Neural proposer has no graph-only benchmark implementation')
        if proposer.method != 'mtp' or getattr(proposer, 'share_mtp_topk_indices', False):
            raise RuntimeError('Unsupported MTP method or shared index configuration')
        prefill = proposer.prefill_cudagraph_manager
        if not prefill or not any(d.uniform_token_count is None and d.max_query_len is None
                and d.num_tokens >= proposer.max_num_tokens for d in prefill.graphs):
            raise RuntimeError('No MTP draft prefill graph for the configured serving chunk')
        captures = self.observe_manager(prefill) + self.observe_manager(proposer.decode_cudagraph_manager)
        # These Python routines run at capture time. A subsequent direct call
        # is the exact upstream eager/PIECEWISE dispatch branch.
        def uncovered(*args, **kwargs):
            self.draft_fallback('MTP proposal selected an uncovered shape')
        proposer._prefill = uncovered
        proposer._generate_draft = uncovered
        proposer._generate_fused_drafts = uncovered
        return captures

    def _draft_event(self, field):
        offset = field * COUNTER.size
        value, = COUNTER.unpack_from(self.draft_counts, offset)
        COUNTER.pack_into(self.draft_counts, offset, value+1)

    def begin_proposal(self):
        if self.draft_active:
            self.draft_fallback('Nested GPU draft proposal', unclassified=True)
        self.draft_active = True
        self.draft_launches = 0
        self._draft_event(0)

    def draft_replayed(self):
        if not self.draft_active:
            self.draft_fallback('Unclassified neural graph replay', unclassified=True)
        self.draft_launches += 1
        self._draft_event(2)

    def draft_fallback(self, reason, *, unclassified=False):
        if self.draft_counts is not None:
            self._draft_event(4 if unclassified else 3)
        raise RuntimeError(reason)

    def end_proposal(self):
        if not self.draft_active or not self.draft_launches:
            self.draft_fallback('Neural draft proposal returned without a graph replay', unclassified=True)
        self._draft_event(1)
        self.draft_active = False

    def ready(self, descriptors, max_prefill):
        full = [d for d in descriptors if d.cg_mode.name == 'FULL' and not d.num_active_loras]
        captures = sorted({d.num_tokens for d in full})
        prefill = sorted({d.num_tokens for d in full if d.uniform_token_count is None and d.max_query_len is None})
        if not prefill or max(prefill) < max_prefill:
            raise RuntimeError('No full CUDA graph was captured for the configured prefill chunk')
        meta = {'schema': 1, 'uuid': self.gpu_uuid, 'pid': self.pid, 'mode': self.mode,
                'enforce_eager': False, 'captured_token_sizes': captures, 'prefill_capture_sizes': prefill,
                'max_prefill_tokens': max_prefill, 'draft_graph_captures': self.draft_captures,
                'output_head_capture_sizes': self.head.sizes if self.head is not None else []}
        if self.direct_head is not None:
            meta['output_head_execution'] = 'direct'
        pending = Path(str(self.path)+'.part')
        pending.write_text(json.dumps(meta))
        pending.replace(Path(str(self.path)+'.json'))

    def add(self, phase, field):
        offset = (phase * len(FIELDS) + field) * COUNTER.size
        value, = COUNTER.unpack_from(self.counts, offset)
        COUNTER.pack_into(self.counts, offset, value+1)

    def require(self, state, mode):
        if state is None or not len(state.is_prefilling_np):
            raise RuntimeError('Cannot classify this real target forward')
        phases = (0,) if state.has_prefill else (1,)
        if state.has_prefill and not state.is_prefilling_np.all():
            phases = (0, 1)
        for phase in phases:
            self.add(phase, 0)
        if mode != 'FULL':
            for phase in phases:
                self.add(phase, 2 if mode == 'NONE' else 3)
            raise RuntimeError(f'Blocked {mode} target forward before execution')
        return phases

    def replayed(self, phases):
        for phase in phases:
            self.add(phase, 1)


class DefaultDispatchCounters:
    """Observe the normal runner's completed target and output-head paths without changing dispatch."""

    def __init__(self, directory, gpu_uuid, config_mode, eager):
        self.path = Path(directory) / f'{gpu_uuid}.{os.getpid()}'
        with Path(str(self.path) + '.dispatch-counts').open('xb+') as stream:
            stream.truncate(2 * 5 * COUNTER.size)
            self.counts = mmap.mmap(stream.fileno(), 0)
        meta = {'schema': 2, 'uuid': gpu_uuid, 'pid': os.getpid(), 'config_mode': config_mode,
                'enforce_eager': eager, 'fields': ['full', 'direct', 'other', 'direct_head', 'piecewise']}
        pending = Path(str(self.path) + '.pending')
        pending.write_text(json.dumps(meta))
        pending.replace(Path(str(self.path) + '.json'))

    def add(self, phases, field):
        for phase in phases:
            offset = (phase * 5 + field) * COUNTER.size
            count, = COUNTER.unpack_from(self.counts, offset)
            COUNTER.pack_into(self.counts, offset, count + 1)

    def completed_forward(self, state, mode):
        if state is None or not len(state.is_prefilling_np):
            self.add((0, 1), 2)
            return None
        if state.has_prefill:
            phases = (0,) if state.is_prefilling_np.all() else (0, 1)
        else:
            phases = (1,)
        self.add(phases, {'FULL': 0, 'NONE': 1, 'PIECEWISE': 4}.get(mode, 2))
        return phases
