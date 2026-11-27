"""Check input ownership, stream fences and ordinary fallback for independent projections."""
from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from vllm_gguf_plugin.inkling import projection_streams as ps

class IndependentProjections(unittest.TestCase):
    def test_only_registered_single_gpu_plain_donor(self):
        o=SimpleNamespace
        model=o(model_weights='/cache/Inkling-2-Mini-IQ4_NL.gguf',model='metadata',
                hf_config=o(model_type='inkling2_loop'))
        run=o(speculative_config=None,scheduler_config=o(max_num_seqs=1),model_config=model,parallel_config=o(world_size_across_dp=1))
        gpu='NVIDIA GeForce RTX 5090'
        with patch.object(ps,'SCHEDULE',(('q_proj',),('gate_proj',))):
            self.assertTrue(ps.supported(run,gpu,1))
            self.assertFalse(ps.supported(run,'NVIDIA GeForce RTX 5080',1))
            self.assertFalse(ps.supported(run,gpu,2))
            model.model_weights='Inkling-2-Mini-UD-Q2_K_XL.gguf'
            self.assertFalse(ps.supported(run,gpu,1))
            model.model_weights='Inkling-2-Mini-IQ4_NL.gguf'
            run.scheduler_config.max_num_seqs=2
            self.assertTrue(ps.supported(run,gpu,1))
            run.scheduler_config.max_num_seqs=4
            self.assertTrue(ps.supported(run,gpu,1))
            run.scheduler_config.max_num_seqs=8
            self.assertFalse(ps.supported(run,gpu,1))
            run.scheduler_config.max_num_seqs=1;run.speculative_config=o()
            self.assertFalse(ps.supported(run,gpu,1))

    def test_independent_projection_batch_limits_are_specific_to_the_gpu(self):
        config = SimpleNamespace(
            model_config=SimpleNamespace(model_weights="Inkling-2-Mini-IQ4_NL.gguf",
                                         hf_config=SimpleNamespace(model_type="inkling2_loop")),
            parallel_config=SimpleNamespace(world_size_across_dp=1),
            scheduler_config=SimpleNamespace(max_num_seqs=1), speculative_config=None)
        qualified = {"NVIDIA GeForce RTX 5090": {1, 2, 4}, ps.PRO_6000: {1}, "NVIDIA GeForce RTX 3090": set(),
                     "NVIDIA GeForce RTX 5080": set(), "NVIDIA GeForce RTX 5070 Ti": set()}
        for card, sizes in qualified.items():
            for size in (1, 2, 4, 8):
                config.scheduler_config.max_num_seqs = size
                with self.subTest(card=card, batch=size):
                    self.assertEqual(ps.supported(config, card, 1), size in sizes)

    def test_original_calls_and_stream_lifetimes(self):
        ledger=[]
        class Stream:
            def __init__(self,name):self.name=name
            def wait_stream(self,other):ledger.append(('wait',self.name,other.name))
        class Result:
            def __init__(self,key):self.key=key
            def record_stream(self,where):ledger.append(('hold',self.key,where.name))
        parent=Stream('main');other=Stream('side')
        @contextmanager
        def enter(stream):
            ledger.append(('enter', stream.name));yield
        tensors=[SimpleNamespace(id=n,device='dummy') for n in range(3)]
        def project(layer,x):
            ledger.append(('run',layer,x.id));return Result(layer)
        groups=tuple(tuple((n,n,ps.OWNERS[n]) for n in row) for row in
                     (('q_proj','k_proj','rel_proj','k_conv_proj'),('gate_proj','v_proj','sink_proj','v_conv_proj')))
        with patch.object(ps.torch.cuda,'current_stream',return_value=parent),patch.object(ps.torch.cuda,'stream',enter):
            vals=ps.project((groups,(other,)),project,tensors)
        self.assertEqual(set(vals),set(ps.OWNERS))
        self.assertEqual(ledger[0],('wait','side','main'))
        self.assertEqual(ledger[-1],('wait','main','side'))
        self.assertEqual({r[1]:r[2] for r in ledger if r[0]=='run'},ps.OWNERS)
        self.assertEqual({r[1] for r in ledger if r[0]=='hold'},{k[0] for k in groups[1]})

    def test_full_and_sliding_layers_retain_the_original_projection_inputs(self):
        config=SimpleNamespace(speculative_config=None,scheduler_config=SimpleNamespace(max_num_seqs=1),
                               parallel_config=SimpleNamespace(world_size_across_dp=1),
                               model_config=SimpleNamespace(model_weights='Inkling-2-Mini-IQ4_NL.gguf',
                                                            hf_config=SimpleNamespace(model_type='inkling2_loop')))
        import sys
        mod=SimpleNamespace(get_current_vllm_config_or_none=lambda:config)
        expected=(('q_proj','rel_proj'),('gate_proj','sink_proj'),('k_proj','k_conv_proj'),('v_proj','v_conv_proj'))
        with patch.dict(sys.modules,{'vllm.config':mod}),patch.object(ps.torch.cuda,'is_available',return_value=True), \
             patch.object(ps.torch.cuda,'get_device_name',return_value='NVIDIA GeForce RTX 5090'), \
             patch.object(ps.torch.cuda,'device_count',return_value=1),patch.object(ps.torch.cuda,'current_device',return_value=0), \
             patch.object(ps,'pool',return_value=('side1','side2','side3')):
            for full in (True,False):
                names={key for key in ps.OWNERS if full or '_conv_' not in key}
                layer=SimpleNamespace(**{n:object() for n in names})
                slots,(groups,side)=ps.prepare(layer)
                self.assertEqual(slots,config.scheduler_config.max_num_seqs)
                ordered=tuple(tuple(n for n in group if n in names) for group in expected)
                self.assertEqual(tuple(tuple(v[0] for v in group) for group in groups),ordered)
                self.assertEqual(side,('side1','side2','side3'))
                for group in groups:
                    for n,module,source in group:
                        self.assertIs(module,getattr(layer,n));self.assertEqual(source,ps.OWNERS[n])

    def test_gb10_uses_measured_stream_groups_only_for_ordinary_xs_one_row(self):
        from contextlib import ExitStack
        from pathlib import Path
        import sys
        from tempfile import TemporaryDirectory

        from vllm_gguf_plugin.inkling import spark_verify_router

        obj = SimpleNamespace
        with ExitStack() as stack:
            directory = Path(stack.enter_context(TemporaryDirectory()))
            target = directory / 'Inkling-2-Mini-dondon-Q2_K_XS.gguf'
            target.write_bytes(b'fixture')
            model = obj(model_weights=str(target), model='tok',
                        hf_config=obj(model_type='inkling2_loop'), dtype=ps.torch.bfloat16, max_model_len=73728)
            run = obj(model_config=model, scheduler_config=obj(max_num_seqs=1, max_num_batched_tokens=2048),
                      speculative_config=None, parallel_config=obj(world_size_across_dp=1),
                      cache_config=obj(cache_dtype='auto'))

            def mock(target, key, value):
                return stack.enter_context(patch.object(target, key, return_value=value))
            prop = obj(multi_processor_count=48)
            cuda = ps.torch.cuda
            arch = mock(cuda, 'get_device_capability', (12, 1))
            mock(cuda, 'get_device_properties', prop)
            def supported():
                return ps.supports_gb10_b1(run, 'NVIDIA GB10', 1)
            with patch.dict(ps.os.environ, {'KAILLM_INKLING_SPARK_PROJECTION_STREAMS': '0'}):
                self.assertFalse(supported())
            stack.enter_context(patch.dict(ps.os.environ, {'KAILLM_INKLING_SPARK_PROJECTION_STREAMS': '1'}))
            with patch.object(spark_verify_router, '_hash', return_value='other'):
                self.assertFalse(supported())
            mock(spark_verify_router, '_hash', spark_verify_router.SHA256)
            self.assertTrue(supported())
            self.assertFalse(ps.supports_gb10_b1(run, 'NVIDIA GeForce RTX 5090', 1))
            self.assertFalse(ps.supports_gb10_b1(run, 'NVIDIA GB10', 2))
            for owner, field, value in ((run, 'speculative_config', obj()),
                    (run.parallel_config, 'world_size_across_dp', 2), (run.scheduler_config, 'max_num_seqs', 2),
                    (run.scheduler_config, 'max_num_batched_tokens', 4096), (model, 'max_model_len', 8192),
                    (model, 'dtype', ps.torch.float16), (model.hf_config, 'model_type', 'other'),
                    (model, 'model_weights', str(target).replace('XS', 'M')), (prop, 'multi_processor_count', 120),
                    (run.cache_config, 'cache_dtype', 'fp8')):
                with self.subTest(field=field), patch.object(owner, field, value):
                    self.assertFalse(supported())
            with patch.object(model, 'model_weights', None), patch.object(model, 'model', str(target)):
                self.assertTrue(supported())
            arch.return_value = (12, 0)
            self.assertFalse(supported())
            arch.return_value = (12, 1)
            config = obj(get_current_vllm_config_or_none=lambda:run)
            stack.enter_context(patch.dict(sys.modules, {'vllm.config':config,'vllm':obj(envs=obj(VLLM_BATCH_INVARIANT=False))}))
            mock(cuda,'is_available',True)
            mock(cuda,'current_device',0)
            mock(cuda,'get_device_name','NVIDIA GB10')
            mock(cuda,'device_count',1)
            side = ('s1','s2','s3')
            pool = mock(ps, 'pool', side)
            for convolution in (True,False):
                fields = {n:object() for n in ps.OWNERS if convolution or '_conv_' not in n}
                layer = obj(**fields)
                with patch.dict(ps.os.environ, {'KAILLM_HUMMING_DETERMINISTIC':'0'}):
                    self.assertIsNone(ps.prepare(layer))
                with patch.dict(ps.os.environ, {'KAILLM_HUMMING_DETERMINISTIC':'1'}):
                    slots,(groups,streams) = ps.prepare(layer)
                self.assertEqual((slots,streams), (1, side))
                self.assertEqual([[n for n, mod, src in group] for group in groups],
                                 [[n for n in group if n in fields] for group in ps.SCHEDULE])
                for group in groups:
                    for n, mod, src in group:
                        self.assertIs(mod,fields[n])
                        self.assertEqual(src,ps.OWNERS[n])
            self.assertEqual(pool.call_count, 2)
            pool.assert_called_with(0, 3)
