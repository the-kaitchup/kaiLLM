from dataclasses import dataclass
import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime-source'))
from vllm_gguf_plugin.inkling import glue,moe,spark_verify_router as router


class SparkRouter(unittest.TestCase):
    def test_target_graph_twins_preserve_padded_shapes(self):
        @dataclass(frozen=True)
        class Descriptor:
            num_tokens:int
            num_reqs:int=1
            num_ubatches:int=1
            num_active_loras:int=0
            inkling_router_verify:bool=False
        original=[Descriptor(n) for n in (1,2,4,6,8)]+[Descriptor(6,2), Descriptor(2,1,2,0), Descriptor(4,1,1,1)]
        captured=original[:]
        twins=router.capture_twins(captured, 6)
        self.assertEqual(list(twins),[original[n] for n in (1,2,3)])
        self.assertEqual(len(captured),len(original)+3)
        self.assertEqual([v.num_tokens for v in twins.values()],[2,4,6])
        self.assertTrue(all(v.inkling_router_verify and not k.inkling_router_verify for k,v in twins.items()))
        with self.assertRaisesRegex(ValueError, 'widths 2 through 4'):
            router.capture_twins(original[:], 4)
        four = router.capture_twins([*original, Descriptor(3)], 4)
        self.assertEqual({key.num_tokens for key in four}, {2, 3, 4})

    def test_real_scheduler_phases(self):
        for n in range(1,9):
            for prefill in (False,True):
                for drafts in (None,[0],[n-1]):
                    state=S(has_prefill=prefill,req_ids=['a'],num_draft_tokens_np=drafts,num_tokens=n,num_scheduled_tokens=[n])
                    self.assertEqual(router.real_verify_width(state),not prefill and 2<=n<=6 and drafts==[n-1])
        self.assertFalse(router.real_verify_width(None))
        self.assertFalse(router.real_verify_width(S(has_prefill=False,req_ids=['a','b'],num_draft_tokens_np=[2,1])))

    def test_only_exact_placement_and_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'Inkling-2-Mini-dondon-Q2_K_XS.gguf';f.write_bytes(b'dummy')
            d=S(architectures=['DFlashInklingDSparkDraftModel'])
            m=S(model=str(f),dtype=torch.bfloat16,hf_config=S(model_type='inkling2_loop'),max_model_len=73728)
            spec=S(method='dflash',draft_model_config=d)
            cfg=S(speculative_config=spec,num_speculative_tokens=5,model_config=m,parallel_config=S(world_size_across_dp=1),
                  scheduler_config=S(max_num_seqs=1,max_num_batched_tokens=2048),cache_config=S(cache_dtype='auto'))
            with patch.dict('os.environ',{'KAILLM_SPARK_DSPARK5_ROW_ROUTER':'1'}),patch.object(torch.cuda,'device_count',return_value=1),patch.object(torch.cuda,'get_device_name',return_value='NVIDIA GB10'),patch.object(torch.cuda,'get_device_capability',return_value=(12,1)):
                self.assertFalse(router.supported(cfg,0))
                with patch.object(router,'_hash',return_value=router.SHA256):
                    self.assertTrue(router.supported(cfg,0))
                    cfg.scheduler_config.max_num_seqs=2;self.assertFalse(router.supported(cfg,0))
                    cfg.scheduler_config.max_num_seqs=1;cfg.cache_config.cache_dtype='fp8';self.assertFalse(router.supported(cfg,0))
                    cfg.cache_config.cache_dtype='auto';cfg.speculative_config=None;self.assertFalse(router.supported(cfg,0))

                cfg.speculative_config = spec
                for quant, size, digest in (('Q2_K_XS', 'XS', router.CHIMACHIMA_XS_SHA256),
                                            ('Q3_K_S', 'S', router.CHIMACHIMA_S_SHA256)):
                    chima = Path(tmp) / f'Inkling-2-Mini-chimachima-kaiPTQ-{quant}.gguf'
                    chima.write_bytes(b'dummy')
                    cfg.model_config.model = str(chima)
                    flag = 'KAILLM_SPARK_DSPARK5_CHIMACHIMA_' + size
                    with patch.dict('os.environ', {flag: '0'}):
                        self.assertFalse(router.supported(cfg, 0))
                    with patch.dict('os.environ', {flag: '1', 'KAILLM_SPARK_DSPARK5_MIXTURE_FMA4': '0'}):
                        with patch.object(router, '_hash', return_value=router.SHA256):
                            self.assertFalse(router.supported(cfg, 0))
                        with patch.object(router, '_hash', return_value=digest):
                            self.assertTrue(router.supported(cfg, 0))
                            cfg.num_speculative_tokens = 3
                            self.assertEqual(router.supported(cfg, 0), size == 'S')
                            cfg.num_speculative_tokens = 5
                            with patch.dict('os.environ', {'KAILLM_SPARK_DSPARK5_MIXTURE_FMA4': '1'}):
                                self.assertFalse(router.supported(cfg, 0))

    def test_each_verifier_row_uses_ordinary_mm_and_pair_fold(self):
        torch.manual_seed(20);x=torch.randn(6,2304);weight=torch.randn(128,2304);bias=torch.randn(128)
        def fp32(n):
            return struct.unpack('<f', struct.pack('<f', n))[0]

        def reference(row):
            logits = row.float() @ weight.float().t()
            probs = logits.sigmoid()
            ids = (probs + bias).topk(8, dim=-1).indices
            p0, p1, p2, p3, p4, p5, p6, p7 = probs.gather(1, ids)[0].tolist()
            even = fp32(fp32(p0 + p1) + fp32(p4 + p5))
            odd = fp32(fp32(p2 + p3) + fp32(p6 + p7))
            inverse = fp32(1 / fp32(fp32(even + odd) + 1e-20))
            return ids.int(), torch.tensor([[fp32(p * inverse) * 8 for p in (p0,p1,p2,p3,p4,p5,p6,p7)]])

        single = [reference(x[i:i+1]) for i in range(6)]
        for n in range(2,7):
            self.assertFalse(router.active())
            with router.capture(), patch.object(torch,'mm',wraps=torch.mm) as op:
                self.assertTrue(router.active());out=glue.route(x[:n],weight,bias,8,8.)
                self.assertEqual([c.args[0].shape[0] for c in op.call_args_list],[1]*n)
            self.assertFalse(router.active())
            self.assertTrue(torch.equal(out[0],torch.cat([r[0] for r in single[:n]]).int()))
            self.assertTrue(torch.equal(out[1],torch.cat([r[1] for r in single[:n]])))

    def test_capture_exception_and_ordinary_control(self):
        x, weight, bias = torch.randn(1,2304), torch.randn(128,2304), torch.randn(128)
        actual = glue.route(x, weight, bias, 8, 8.)
        baseline = moe.route(x, weight, bias, 8, 8.)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, baseline)))
        with self.assertRaises(RuntimeError),router.capture():raise RuntimeError('test')
        self.assertFalse(router.active())


if __name__=='__main__':unittest.main()
