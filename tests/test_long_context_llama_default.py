"""Validate the running ELF and DSO policy on CPU and the shared benchmark/report contracts."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks/long_context'))
import llama_default as stock
import llama_graph
import report
from benchmark import MANIFEST
from test_long_context_hybrid_report import hybrid_raw
from test_long_context_revisions import raw_plain

KEY='inkling-loop-iq4'
REV=MANIFEST['llama_cpp']['inkling_revision']
LOCK=json.loads(llama_graph.SOURCE_LOCK.read_bytes())[REV]


def fixture_certificate(bin_dir):
    cert={'protocol':2,'upstream':REV,'source_sha256':LOCK,
          'binary_sha256':{n:llama_graph.digest(bin_dir/n) for n in llama_graph.BINARY_NAMES}}
    (bin_dir/llama_graph.CERTIFICATE).write_text(json.dumps(cert))
    return cert


class DefaultPolicy(unittest.TestCase):
    @unittest.skipUnless(sys.platform.startswith('linux') and shutil.which('cc'),'needs Linux /proc and a C compiler')
    def test_live_elf_loaded_dsos_forced_environment_and_path_replacement(self):
        with TemporaryDirectory() as d:
            bindir=Path(d)
            subprocess.run(['cc','-shared','-fPIC','-x','c','-','-o',str(bindir/'libggml-cuda.so')],
                           input=b'int ggml_cuda_stub(void){return 0;}',check=True)
            subprocess.run(['cc','-shared','-fPIC','-x','c','-','-o',str(bindir/'libllama-server-impl.so')],
                           input=b'int server_stub(void){return 0;}',check=True)
            src=b'''#include <dlfcn.h>
#include <stdio.h>
int main(void){if(!dlopen("libggml-cuda.so",RTLD_NOW)||!dlopen("libllama-server-impl.so",RTLD_NOW))return 2;
puts("ready");fflush(stdout);return getchar()==EOF?0:1;}'''
            executable=bindir/'llama-server';subprocess.run(['cc','-x','c','-','-ldl','-o',str(executable)],input=src,check=True)
            fixture_certificate(bindir)
            dirty={'CUDA_VISIBLE_DEVICES':'GPU-test','KAILLM_LLAMA_GRAPH_FILE':'stale','GGML_CUDA_GRAPH_OPT':'1','LD_PRELOAD':'override'}
            env=stock.prepare(executable,REV,1,dirty);self.assertEqual(set(env),{'CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH'})
            def launch(env,check):
                with subprocess.Popen([str(executable)],env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE) as proc:
                    try:
                        self.assertEqual(proc.stdout.readline(),b'ready\n');check(proc.pid)
                    finally:proc.stdin.close()
            def verify(pid):
                row=stock.process(pid,[str(executable)],REV,['GPU-test']);self.assertEqual(stock.process(pid,[str(executable)],REV,['GPU-test']),row)
                self.assertEqual(row['mapped_dso_sha256'],{n:row['binary_sha256'][n] for n in llama_graph.BINARY_NAMES[1:]})
                with self.assertRaisesRegex(ValueError,'arguments'):stock.process(pid,[str(executable),'--fake'],REV,['GPU-test'])
                with self.assertRaisesRegex(ValueError,'visibility'):stock.process(pid,[str(executable)],REV,['GPU-other'])
            launch(env,verify)
            def rejects_env(pid):
                with self.assertRaisesRegex(ValueError,'execution-policy override'):stock.process(pid,[str(executable)],REV,['GPU-test'])
            launch({**env,'GGML_CUDA_DISABLE_GRAPHS':'1'},rejects_env)
            def replace_file(pid):
                original=bindir/'libggml-cuda.so';other=bindir/'same-bytes';shutil.copyfile(original,other);os.replace(other,original)
                with self.assertRaisesRegex(ValueError,'library'):stock.process(pid,[str(executable)],REV,['GPU-test'])
            launch(env,replace_file)

    def test_microbatch_retries_only_confirmed_compute_oom_and_preserves_receipts(self):
        memory_error = ('cudaMalloc failed: out of memory\n'
                        'ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 7673184512\n'
                        'graph_reserve: failed to allocate compute buffers\n')
        for text, retries in ((memory_error, [512,256]), ('CUDA error: out of memory', [512]),
                              ('failed to allocate compute buffers', [512]), ('bad weights', [512])):
            with self.subTest(text=text), TemporaryDirectory() as directory:
                root = Path(directory); log = root/'server.log'; sizes = []
                def start(argv):
                    sizes.append(int(argv[1])); log.write_text(text)
                    if len(sizes) == 1: raise RuntimeError('startup failed')
                    return 'server'
                def run():
                    return stock.start_with_microbatches(start, self.fail, ['-ub','512','-b','2048'], log, root,
                                                         lambda p,v:p.write_text(json.dumps(v)))
                if len(retries) == 1:
                    with self.assertRaisesRegex(RuntimeError,'startup failed'):run()
                else:
                    proc, attempts = run();self.assertEqual(proc,'server');self.assertTrue(stock.valid_attempts(attempts,256))
                    self.assertEqual(attempts[0]['requested_bytes'],7673184512)
                    self.assertEqual(attempts[0]['log_sha256'],hashlib.sha256((root/'ub512.log').read_bytes()).hexdigest())
                    self.assertFalse(stock.valid_attempts(attempts[1:],256))
                self.assertEqual(sizes,retries)

    def test_attention_warmup_oom_retries_lower_valid_microbatch(self):
        flash = ('CUDA error: out of memory\n'
                 'current device: 0, in function launch_fattn at fattn-common.cuh:1199\n')
        with TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / 'server.log'
            started = []

            def start(argv):
                size = int(argv[1])
                started.append(size)
                log.write_text(flash)
                if size == 512:
                    raise RuntimeError('attention warmup failed')
                return size

            selected, attempts = stock.start_with_microbatches(
                start, self.fail, ['-ub', '512'], log, root, lambda p, v: p.write_text(json.dumps(v))
            )
            self.assertEqual((selected, started), (256, [512, 256]))
            self.assertEqual(attempts[0], {'microbatch': 512, 'status': 'startup-attention-oom',
                                           'function': 'launch_fattn', 'log_sha256': stock.digest(root / 'ub512.log')})
            self.assertTrue(stock.valid_attempts(attempts, selected))
            self.assertFalse(stock.attention_launch_oom('CUDA error: out of memory'))
            self.assertFalse(stock.attention_launch_oom('in function launch_fattn'))

    def test_full_prompt_admission_retries_only_confirmed_cuda_oom(self):
        failed_log = ('CUDA error: out of memory\n'
                      '  current device: 0, in function ggml_cuda_graph_launch_chunk at ggml-cuda.cu:3495\n')
        for text, retry in ((failed_log, True), (failed_log.replace('ggml_cuda_graph_launch_chunk', 'launch_fattn'), True),
                            ('CUDA error: out of memory\n', False),
                            ('in function ggml_cuda_graph_launch_chunk', False), ('transport failure', False)):
            with self.subTest(text=text), TemporaryDirectory() as directory:
                root = Path(directory)
                log = root/'server.log'
                started, stopped = [], []
                def start(command):
                    size = int(command[1]); started.append(size); log.write_text('server ready\n')
                    return size
                def probe():
                    if started[-1] == 512:
                        with log.open('a') as f: f.write(text)
                        raise ConnectionError('prefill failed')
                def run():
                    return stock.start_with_microbatches(start, stopped.append, ['-ub', '512'], log, root,
                                                         lambda p,v:p.write_text(json.dumps(v)), probe=probe)
                if retry:
                    server, attempts = run()
                    self.assertEqual((server, started), (256, [512, 256, 256]))
                    self.assertTrue(stock.valid_attempts(attempts, server))
                    self.assertEqual(attempts[0]['log_sha256'], stock.digest(root/'ub512.log'))
                    self.assertEqual(attempts[-1].get('admission'), 'separate-process')
                    attempts[-1]['admission'] = 'unknown'
                    self.assertFalse(stock.valid_attempts(attempts, server))
                    attempts[-1]['admission'] = 'separate-process'
                    attempts[0]['prompt_tokens'] = 1024
                    self.assertFalse(stock.valid_attempts(attempts, server))
                else:
                    with self.assertRaisesRegex(ConnectionError, 'prefill failed'): run()
                    self.assertEqual(started, [512])
                self.assertEqual(stopped, [512, 256] if retry else [512])

    def test_successful_admission_probes_only_the_disposable_server(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            launched, probed, stopped = [], [], []
            def start(argv):
                proc = object()
                launched.append((tuple(argv), proc))
                return proc
            active, attempts = stock.start_with_microbatches(
                start, stopped.append, ['-ub', '512', '-b', '2048'], root/'server.log', root,
                lambda p,v:p.write_text(json.dumps(v)), probe=lambda: probed.append(launched[-1][1]))
            self.assertEqual([item[0] for item in launched], [('-ub', '512', '-b', '2048')] * 2)
            self.assertEqual(probed, stopped)
            self.assertEqual(probed, [launched[0][1]])
            self.assertIs(active, launched[1][1])
            self.assertIsNot(active, probed[0])
            self.assertTrue(stock.valid_attempts(attempts, 512))

    def test_exhausted_graph_oom_ladder_preserves_capacity_and_all_logs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); log = root/'server.log'; stopped=[]
            def fail():
                log.write_text('CUDA error: out of memory\ncurrent device: 0, in function ggml_cuda_graph_launch_chunk\n')
                raise ConnectionError('server disconnected')
            with self.assertRaisesRegex(RuntimeError, 'out of memory.*minimum physical microbatch'):
                stock.start_with_microbatches(lambda argv:int(argv[1]), stopped.append, ['-ub','512'], log, root,
                                              lambda p,v:p.write_text(json.dumps(v)), probe=fail)
            attempts=json.loads((root/'attempts.json').read_bytes())
            self.assertEqual(stopped,list(stock.MICROBATCHES))
            self.assertTrue(all(r['status']=='prefill-graph-oom' and
                                r['log_sha256']==stock.digest(root/f"ub{r['microbatch']}.log") for r in attempts))

    def test_default_requires_proposals_and_preserves_token_batch_limits(self):
        req=lambda v:{'requests':[{'server_timings':v}]}
        self.assertEqual(stock.counts(req({'draft_n':12,'draft_n_accepted':3}), 'mtp5'),
                         {'scope':'full-wave','proposed':12,'accepted':3,'rejected':9})
        for settings in ({},{'draft_n':7},{'draft_n':3,'draft_n_accepted':4}):
            with self.assertRaises(ValueError):stock.counts(req(settings),'mtp5')
        self.assertIsNone(stock.counts(req({}),'plain'))


def raw_stock():
    r=raw_plain(KEY,report.MANIFEST['models'][KEY]['revision']);r['engine']='llama.cpp';env=r['environment']
    env['gpu_names']=['NVIDIA GeForce RTX 5090'];uid=env['devices'][0]['uuid'];bins={n:hashlib.sha256(n.encode()).hexdigest() for n in llama_graph.BINARY_NAMES}
    witness={'policy':'default','pid':100,'start_ticks':1000,'upstream':REV,'source_sha256':LOCK,'binary_sha256':bins,
             'mapped_dso_sha256':{n:bins[n] for n in llama_graph.BINARY_NAMES[1:]},'observer_configured':False,'environment_overrides':[],'cuda_visible_devices':[uid]}
    env.update(llama_policy='default',llama_process=witness,measured_gguf_sha256=r['sha256'],cuda_graph_head_devices=[],cuda_graph_drafter_devices=[],
               llama_startup_attempts=[{'microbatch':512,'status':'ready'}],cuda_graph_capture_sizes_by_gpu={},engine_revision='llama.cpp '+REV[:7],llama_source=REV,
               server_argv=['llama-server','--ctx-size','73728','--parallel','1','-ngl','999','--fit','off','--flash-attn','on','--cache-type-k','bf16',
                            '--cache-type-v','bf16','--cache-ram','0','--no-context-shift','-b','2048','-ub','512','--spec-type','none'])
    r['llama_default']={'process':witness,'startup_attempts':env['llama_startup_attempts'],'microbatch':512,'wave_checks':list(stock.WAVES),'measured_gguf_sha256':r['sha256'],'speculative':None}
    return r


class DefaultReports(unittest.TestCase):
    def test_archived_plain_prose_reopens_pending_code_after_a_startup_allocation_failure(self):
        row = report.get_record(raw_stock())
        limit = {field: row[field] for field in ('checkpoint', 'gpu', 'engine', 'mode',
                                                  'batch', 'model_revision', 'llama_source')}
        limit['reason'] = 'Memory allocation failed at the requested context'
        pending = report.counts([], [limit])
        recovered = report.counts([row], [limit])
        self.assertEqual(pending['unavailable'] - recovered['unavailable'], 1)
        self.assertEqual(recovered['filled'] - pending['filled'], 1)
        self.assertEqual(recovered['remaining'] - pending['remaining'], 1)
        for view in report.tables([row], [limit]):
            self.assertNotIn('<summary>Configuration limits (1)</summary>', view)
        other_batch = {**limit, 'batch': 2}
        self.assertEqual(report.active_limits([row], [limit, other_batch]), [other_batch])

    def test_wave_identity_source_and_no_claim_of_graphs(self):
        raw=raw_stock();row=report.get_record(raw)
        self.assertTrue(stock.verified(row));self.assertTrue(report.timing_accepted(row));self.assertFalse(report.graphs_verified(row));self.assertNotIn('pid',row['llama_default'])
        for mutate in (lambda r:r['llama_default']['wave_checks'].pop(),lambda r:r['llama_default']['process'].update(observer_configured=True),
                       lambda r:r['llama_default']['process']['mapped_dso_sha256'].pop('libggml-cuda.so'),lambda r:r.update(cuda_graphs={}),
                       lambda r:r['environment']['server_argv'].__setitem__(-3,'256'),lambda r:r['environment'].update(measured_gguf_sha256=['bad'])):
            broken=copy.deepcopy(raw);mutate(broken)
            with self.assertRaises(ValueError):report.get_record(broken)

    def test_256_label_requires_witness_to_failed_standard_512_attempt(self):
        raw=raw_stock();proof=raw['llama_default'];env=raw['environment']
        failed={'microbatch':512,'status':'compute-buffer-oom','requested_bytes':7673184512,'log_sha256':'a'*64}
        proof.update(microbatch=256,startup_attempts=[failed,{'microbatch':256,'status':'ready'}]);env['llama_startup_attempts']=proof['startup_attempts']
        idx=env['server_argv'].index('-ub')+1;env['server_argv'][idx]='256'
        row=report.get_record(raw)
        for view in report.tables([row]):self.assertIn('| llama.cpp default (ub 256) |', view)
        del proof['startup_attempts'][0]
        with self.assertRaisesRegex(ValueError,'microbatch fallback'):report.get_record(raw)

    def test_show_both_policies_and_choose_default_once_for_the_same_kai_receipt(self):
        raw=raw_stock();stock_row=report.get_record(raw);kai=report.get_record(hybrid_raw());
        # The two engine test receipts deliberately use the same physical GPU, prompts, and driver.
        stock_row.update(hardware_id=kai['hardware_id'],llama_source=REV);kai.update(llama_source=REV)
        strict={**stock_row,'cuda_graphs':kai['cuda_graphs'],'full_seconds':stock_row['full_seconds']*2};del strict['llama_default'];
        by_gpu, by_model = report.tables([kai, stock_row, strict])
        self.assertIn('1 matched pair shares', by_model)
        for view in (by_gpu, by_model):
            self.assertIn('kaiLLM Graphed backbone, eager head · llama.cpp default',view)
            self.assertIn('kaiLLM Graphed backbone, eager head · llama.cpp Forced graphs',view)
        self.assertEqual(report.counts([kai,stock_row,strict],[])['filled'],2)
        for view in report.tables([stock_row]):self.assertIn('| llama.cpp default (ub 512) |',view)


    def test_summary_uses_latest_matched_measurement_even_if_slower(self):
        kai = report.get_record(hybrid_raw())
        stock_row = report.get_record(raw_stock())
        stock_row.update(hardware_id=kai["hardware_id"])
        first = dict(kai, timestamp_utc="2026-10-01T00:00:00Z",
                     prefill_seconds=[0.1]*3, full_seconds=0.2)
        second = dict(kai, timestamp_utc="2026-10-02T00:00:00Z",
                      prefill_seconds=[1e6]*3, full_seconds=2e6)
        pairs = [{"kaiLLM": run, "llama.cpp": stock_row} for run in (second, first)]
        summary = "\n".join(report.matched_performance(pairs))
        self.assertIn("1 matched pair shares", summary)
        self.assertIn("| Prefill | 0 | 0 | 1 | 1 |", summary)
        self.assertIn("| Decode | 0 | 0 | 1 | 1 |", summary)

    def test_default_attaches_to_matching_kai_with_obsolete_unused_reference_pin(self):
        reference = report.get_record(raw_stock())
        kai = report.get_record(hybrid_raw())
        reference['hardware_id'] = kai['hardware_id']
        kai['llama_source'] = 'obsolete-unused-by-kai'
        self.assertTrue(report.same_default_workload(kai, reference))
        for field, mismatch in (('sha256', ['other']), ('hardware_id', 'other'), ('driver', ['other']),
                                ('power_limit_w', ['1.0']), ('input_tokens_per_request', 2048),
                                ('output_tokens_per_request', 32768), ('batch', 2), ('domain', 'code'),
                                ('prompt_set_sha256', 'other')):
            wrong = {**kai, field: mismatch}
            self.assertFalse(report.same_default_workload(wrong, reference), field)
        by_gpu, by_model = report.tables([kai, reference])
        self.assertIn('1 matched pair shares', by_model)
        for view in (by_gpu, by_model):
            self.assertIn('kaiLLM Graphed backbone, eager head · llama.cpp default', view)
            self.assertIn(reference['llama_source'][:7]+'](https://github.com/ggml-org/llama.cpp/commit/'+reference['llama_source']+')', view)
            self.assertEqual(view.count('Graphed backbone, eager head'), 1)
