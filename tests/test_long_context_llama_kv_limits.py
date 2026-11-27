import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'benchmarks/long_context'))
import llama_kv_limits as cap
from benchmark import MANIFEST
from coverage_matrix import active_limits, counts
from report import tables

UUID='GPU-12345678-abcd-4444-aaaa-000000000001'
GPU='NVIDIA GeForce RTX 5070 Ti'

class StockKVTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.model=MANIFEST['models']['inkling-udq2'];self.upstream=MANIFEST['llama_cpp']['inkling_revision']
        (self.root/'public-build-inkling.json').write_text(json.dumps({'upstream':self.upstream,'public_executable_sha256':'f'*64}))
        for n,spec in [('inkling-udq2',self.model),('dspark',MANIFEST['dspark'])]:
            (self.root/f'download-{n}.json').write_text(json.dumps(dict(source='huggingface.co',**{k:spec[k] for k in ('repo','revision','files')})))
        (self.root/'physical-gpu.csv').write_text(f'{GPU}, {UUID}, 16303, 595.99.02, 300.00\n')

    def probe(self,batch,ub,kv_mib,resident=None,cause='kv cache',mode='plain',total=16303):
        draft=MANIFEST['dspark'] if mode=='dspark5' else None
        path=lambda m:'/weights/'+m['repo']+'/'+m['revision']+'/'+next(iter(m['files']))
        target=path(self.model);ctx=73734 if draft else 73728
        argv=[f'/build/{self.upstream}/build/bin/llama-server','-m',target,'--host','127.0.0.1','--port','18947','--ctx-size',str(batch*ctx),'--parallel',str(batch),'-ngl','999','--fit','off',
              '--flash-attn','on','--cache-type-k','bf16','--cache-type-v','bf16','--cache-ram','0','--no-context-shift','-b','2048','-ub',str(ub),'--spec-type','draft-dspark' if draft else 'none']
        if draft:argv += ['--model-draft',path(draft),'--spec-draft-n-max','5']
        if resident:argv += ['-lv','5']
        log=f"srv load_model: loading model '{target}'\n"
        if resident:log+=f'load_tensors:        CUDA0 model buffer size = {resident} MiB\n'
        log+=f'ggml_backend_cuda_buffer_type_alloc_buffer: allocating {kv_mib:.2f} MiB on device 0: cudaMalloc failed: out of memory\nggml_backend_buft_alloc_buffer_n_default: failed to allocate CUDA0 buffer of size {kv_mib*cap.MIB}\nllama_init_from_model: failed to initialize the context: failed to allocate buffer for {cause}\n'
        name=f'public-capacity-{mode}-b{batch}-ub{ub}'+('-memory' if resident else '')
        p=self.root/(name+'.json');p.with_suffix('.log').write_text(log)
        p.write_text(json.dumps(dict(outcome='cuda-oom',checkpoint='inkling-udq2',gpu=f'{UUID}, {total} MiB, 8000 MiB, 595.99.02, 300.00 W',
                                    model=self.model,draft=draft,upstream=self.upstream,mode=mode,batch=batch,microbatch=ub,argv=argv,log_sha256=hashlib.sha256(log.encode()).hexdigest())))
        return p

    def test_actual_batch_only_cpu_buffers_and_observations(self):
        for size in (512,64):
            self.probe(4,size,9216); self.probe(8,size,18432,mode='dspark5')
        # One endpoint alone never closes another batch; a generic compute OOM is independent.
        self.probe(16,512,36864)
        for size in (512,64):self.probe(32,size,73728,cause='compute buffers')
        rows=cap.measured_limits(self.root);self.assertEqual([r['batch'] for r in rows],[8,4]);self.assertEqual({r['batch']:r['observation_only'] for r in rows},{4:True,8:False})
        start = counts([], [])
        measured = counts([], rows)
        # Both B8 domains already exceed this board's independent stock allocation floor.
        self.assertEqual(start["remaining"], measured["remaining"])
        self.assertEqual(measured["excluded_by_reason"]["Measured configuration limit"], 2)
        for view in tables([],rows):
            self.assertIn('| llama.cpp default | plain | 4 | The 9,216 MiB GPU KV allocation failed',view)
            self.assertIn('| llama.cpp default | dspark5 | 8 | The 18,432 MiB GPU KV allocation exceeds',view)
        sample=self.probe(4,64,9216,'8000.00');r=next(r for r in cap.measured_limits(self.root) if r['batch']==4);self.assertFalse(r['observation_only']);self.assertLess(r['llama_kv_allocation']['resident_min_bytes'],8000*cap.MIB)
        log=sample.with_suffix('.log').read_text().replace('CUDA0 model buffer','CUDA_Host model buffer');sample.with_suffix('.log').write_text(log)
        with self.assertRaisesRegex(ValueError,'digest'):cap.measured_limits(self.root)
        receipt=json.loads(sample.read_text());receipt['log_sha256']=hashlib.sha256(log.encode()).hexdigest();sample.write_text(json.dumps(receipt));self.assertTrue(next(r for r in cap.measured_limits(self.root) if r['batch']==4)['observation_only'])

    def test_rounding_or_measured_free_never_proves_exhaustion(self):
        for size in (512,64):self.probe(4,size,16303,'0.50')
        self.assertTrue(cap.measured_limits(self.root)[0]['observation_only'])
        (self.root/'physical-gpu.csv').write_text(f'{GPU}, {UUID}, 16304, 595.99.02, 300.00\n')
        with self.assertRaisesRegex(ValueError,'inventory'):cap.measured_limits(self.root)

    def test_wrong_target_and_conflicting_allocation(self):
        self.probe(8,512,18432);p=self.probe(8,64,18431);self.assertFalse(cap.measured_limits(self.root))
        d=json.loads(p.read_bytes());d['argv'][d['argv'].index('-m')+1]='/other.gguf';p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'path'):cap.measured_limits(self.root)

    def test_draft_admission_is_scoped_to_physical_board(self):
        base = self.probe(2, 64, 578, mode='dspark5')
        argv = json.loads(base.read_bytes())['argv']
        base.unlink()
        base.with_suffix('.log').unlink()
        target = argv[argv.index('-m')+1]
        draft = argv[argv.index('--model-draft')+1]
        cert = dict(protocol=cap.VERSION, upstream=self.upstream,
                    source_sha256=json.loads(cap.SOURCE_LOCK.read_bytes())[self.upstream],
                    binary_sha256={name: 'b'*64 for name in cap.BINARY_NAMES})
        (self.root/'certified-upstream.json').write_text(json.dumps(cert))
        stamp = dict(checkpoint='inkling-udq2', uuid=UUID, batch=2, minimum_command=argv, log_sha256={})
        for ub in cap.MICROBATCHES:
            log = f"load_model: loading model '{target}'\n"
            if ub <= 128:
                log += f"loading draft model '{draft}'\n"
            log += 'on device 0: cudaMalloc failed: out of memory\nfailed to allocate CUDA0 buffer of size 606076928\n'
            if ub == 64:
                log += 'failed to allocate buffer for kv cache\nfailed to create MTP context\n'
            elif ub == 128:
                log += f"unable to allocate CUDA0 buffer\nfailed to load draft model, '{draft}'\n"
            else:
                log += 'failed to allocate compute buffers\n'
            (self.root/f'ub{ub}.log').write_text(log)
            stamp['log_sha256'][str(ub)] = cap.digest(log.encode())
        receipt = self.root/'stock-draft-admission.json'
        receipt.write_text(json.dumps(stamp))
        row, = cap.measured_limits(self.root)
        self.assertTrue(row['observation_only'])
        self.assertEqual([r['stage'] for r in row['llama_draft_admission']['witnesses']],
                         ['target compute','target compute','draft weights','draft KV'])
        self.assertEqual(counts([], [row])['remaining'], counts([], [])['remaining'])
        for table in tables([], [row]):
            self.assertIn('draft weights at ub128, draft KV at ub64', table)
        other = dict(row, hardware_id='another-board')
        with patch('coverage_matrix.provisional_reason', return_value=None), patch('coverage_matrix.default_llama_verified', return_value=True):
            self.assertEqual(active_limits([other], [row]), [row])
            self.assertEqual(active_limits([row], [row]), [])
        (self.root/'ub64.log').write_text('generic failure')
        with self.assertRaisesRegex(ValueError, 'digest'):
            cap.measured_limits(self.root)
        stamp['log_sha256']['64'] = cap.digest(b'generic failure')
        receipt.write_text(json.dumps(stamp))
        with self.assertRaises(ValueError):
            cap.measured_limits(self.root)
