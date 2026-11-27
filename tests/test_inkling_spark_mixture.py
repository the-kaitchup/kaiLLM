import ast
import hashlib
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime-source'))
from vllm_gguf_plugin.inkling import glue,spark_verify_router as scope,spark_mixture_fma4 as kernel


class GB10Mixture(unittest.TestCase):
    def test_original_validated_function_bodies(self):
        module=Path(kernel.__file__).read_text()
        ns={n.name:n for n in ast.parse(module).body if isinstance(n,ast.FunctionDef)}
        for name,expected in [('combine_fma4','f8f3557ee62aa0513256ffb7afcee275a0e5c2296a931b2daee4cc7dcee33167'),('_combine_fma4','7d1869155ddbf4b7b2dc0257955ddecb3d811532060f5ea2ea927970e365df22')]:
            self.assertEqual(hashlib.sha256(ast.get_source_segment(module,ns[name]).encode()).hexdigest(),expected)

    def test_only_authenticated_source_scale(self):
        data=torch.ones(128,dtype=torch.float32)
        raw=data.numpy().tobytes()
        with self.assertRaisesRegex(ValueError,'original GGUF'):scope.mark_mixture_scale(data,bytes(512))
        with self.assertRaisesRegex(ValueError,'layout'):scope.mark_mixture_scale(data.bfloat16(),raw)
        self.assertFalse(hasattr(data,'_inkling_spark_verified_mixture_scale'))
        scope.mark_mixture_scale(data,raw)
        self.assertTrue(data._inkling_spark_verified_mixture_scale)
        self.assertFalse(hasattr(data.clone(),'_inkling_spark_verified_mixture_scale'))

    def test_selected_geometry_and_ordinary_preload_fallback(self):
        class Tensor(SimpleNamespace):
            def is_contiguous(self):return self.contiguous
        def t(shape, dtype):return Tensor(shape=shape,dtype=dtype,is_cuda=True,device='cuda:0',contiguous=True)
        sc=t((128,),torch.float32);sc._inkling_spark_verified_mixture_scale=True
        for n in (1,2,3,4,5,6,7,128,2048):
            x,w,ids=t((8*n,2304),torch.float32),t((n,8),torch.float32),t((n,8),torch.int32)
            self.assertFalse(scope.mixture_supported(x,w,sc,ids,torch.float32))
            with scope.capture():
                self.assertEqual(scope.mixture_supported(x,w,sc,ids,torch.float32),2<=n<=6)
                self.assertFalse(scope.mixture_supported(x,w,None,ids,torch.float32))
                self.assertFalse(scope.mixture_supported(x,w,t((128,),torch.float32),ids,torch.float32))
                self.assertFalse(scope.mixture_supported(x,w,sc,ids,torch.bfloat16))
                ids.contiguous=False
                self.assertFalse(scope.mixture_supported(x,w,sc,ids,torch.float32))

    def test_fallback_preserves_object_identity_and_original_call(self):
        val = object()
        weights = SimpleNamespace(shape=(6, 8))
        args = (val, weights, val, val, torch.float32)
        for selected in (True,False):
            with patch.object(glue, 'SPARK_ROW_MIXTURE', False), \
                    patch.object(glue, 'SPARK_MIXTURE_FMA4', True), \
                    patch.object(scope, 'mixture_supported', return_value=selected), \
                    patch.object(kernel, 'combine_fma4', return_value=val) as run, \
                    patch.object(glue, '_fn') as old:
                if selected:
                    self.assertIs(glue.combine(*args), val)
                    run.assert_called_once_with(val, weights, val, val)
                    old.assert_not_called()
                else:
                    self.assertIs(glue.combine(*args), old.return_value.return_value)
                    old.return_value.assert_called_once_with(*args)
                    run.assert_not_called()

if __name__=='__main__':unittest.main()
