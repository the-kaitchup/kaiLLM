"""Exercise the V2 pre-capture startup boundary on CPU without a vLLM installation."""
import ast
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar, Context
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
import importlib.util

ROOT = Path(__file__).resolve().parents[1] / 'runtime-source'
CORE = ROOT / 'vllm-overrides/vllm/v1/worker/gpu'


def runner_fixture():
    tree = ast.parse((CORE / 'model_runner.py').read_text())
    cls = next(x for x in tree.body if getattr(x, 'name', '') == 'GPUModelRunner')
    wanted = ('benchmark_graphs', 'synthetic_benchmark_warmup')
    cls.body = [f for f in cls.body if getattr(f, 'name', '') in wanted]
    cls.bases = []
    space = {'contextmanager': contextmanager}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), 'model_runner.py', 'exec'), space)
    runner = space['GPUModelRunner']()
    runner._synthetic_benchmark_warmup = ContextVar('isolated_test', default=False)
    runner._benchmark_graphs_ready = False
    runner.speculative_config = object()
    runner.rejection_sampler = SimpleNamespace(enable_adaptive_verification=True)
    runner.adaptive_verification = object()
    return runner


def startup_function(synthetic):
    tree = ast.parse((CORE / 'warmup.py').read_text())
    fn = next(f for f in tree.body if getattr(f, 'name', '') == 'warmup_kernels')
    fn.decorator_list = []  # torch.inference_mode is immaterial for CPU dispatch doubles.
    space = {'nullcontext': nullcontext, '_warmup_kernels': synthetic}
    statement = ast.parse('from __future__ import annotations').body[0]
    exec(compile(ast.Module(body=[statement, fn], type_ignores=[]), 'warmup.py', 'exec'), space)
    return space['warmup_kernels']


class SyntheticWarmup(TestCase):
    def test_prefill_sampling_and_greedy_draft_bootstrap_are_isolated_until_capture(self):
        spec = importlib.util.spec_from_file_location('graph_observer', ROOT/'vllm_gguf_plugin/benchmark_graphs.py')
        backend = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(backend)
        with TemporaryDirectory() as d:
            runner = runner_fixture()
            observer = runner._benchmark_graphs = backend.TargetGraphCounters(d, 'GPU-test', 'FULL', False)
            visits = []
            def synthetic(model, execute, sample, greedy=False):
                self.assertIsNotNone(Context().run(lambda: model.benchmark_graphs))
                self.assertIsNone(model.benchmark_graphs)
                execute(); sample(); visits.append(greedy)
            def component():
                self.assertIsNone(runner.benchmark_graphs)
            try:
                startup_function(synthetic)(runner, component, component)
                self.assertEqual(visits, [False, True])
                self.assertIs(runner.benchmark_graphs, observer)
                self.assertFalse(any(observer.counts[:]))
                runner._benchmark_graphs_ready = True
                with self.assertRaisesRegex(RuntimeError, 'before graph capture'):
                    startup_function(synthetic)(runner, component, component)
                # Actual serving paths still reject an unsupported dynamic descriptor.
                with self.assertRaisesRegex(RuntimeError, 'Blocked NONE'):
                    observer.require(SimpleNamespace(has_prefill=False, is_prefilling_np=[False]), 'NONE')
            finally:
                observer.counts.close()

    def test_exception_restores_observer_sampling_and_adaptive_state(self):
        runner = runner_fixture()
        observer = runner._benchmark_graphs = SimpleNamespace(head_counts=None, draft_counts=None)
        adaptive = runner.adaptive_verification
        def fail(*args):
            self.assertIsNone(runner.benchmark_graphs)
            raise ValueError('synthetic kernel failed')
        with self.assertRaisesRegex(ValueError, 'synthetic kernel failed'):
            startup_function(fail)(runner, None, None)
        self.assertIs(runner.benchmark_graphs, observer)
        self.assertIs(runner.adaptive_verification, adaptive)
        self.assertTrue(runner.rejection_sampler.enable_adaptive_verification)
        for key in ('head_counts', 'draft_counts'):
            setattr(observer, key, object())
            with self.assertRaisesRegex(RuntimeError, 'before graph capture'):
                with runner.synthetic_benchmark_warmup():
                    self.fail('post-capture exemption')
            setattr(observer, key, None)
