"""Keep the portable K64 option scoped to the checked projection contract."""

import json
import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from vllm_gguf_plugin.quantization.humming import _portable_k64_decode_tuning


class PortableK64(unittest.TestCase):
    def test_checked_geometry_retains_the_reduction_and_original_fallback(self):
        ns = SimpleNamespace
        serving = ns(speculative_config=None, parallel_config=ns(world_size_across_dp=1),
                     scheduler_config=ns(max_num_seqs=1))
        baseline = dict(block_shape=(16, 64, 32), warp_shape=(16, 32, 32), use_stream_k=False,
                        use_f16_accum=False, num_sms=16, num_stages=3, num_ctas_per_sm=1,
                        output_chunk_rows=0, use_pdl=True, mma_type="mma", raster_group_m=1)
        policy = {"gemm_type": "dense", "use_f16_accum": False, "use_batch_invariant": True}

        def choose(*, kind=12, shape=(1024, 2304), dtype=torch.bfloat16, device="cuda:0"):
            return _portable_k64_decode_tuning(ns(shape_n=1024), policy, shape, kind, dtype, torch.device(device))

        modules = {
            "vllm.config": ns(get_current_vllm_config_or_none=lambda: serving),
            "vllm.utils.humming": ns(get_heuristics_config=lambda *_a, **_kw: baseline),
        }
        with patch.dict(sys.modules, modules), patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)), \
             patch.dict(os.environ, KAILLM_HUMMING_PORTABLE_K64="1"):
            self.assertEqual(
                json.loads(choose()),
                {**baseline, "block_shape": [16, 64, 64], "warp_shape": [16, 32, 64]},
            )
            self.assertIsNone(choose(kind=18))
            self.assertIsNone(choose(shape=(32, 2304)))
            self.assertIsNone(choose(dtype=torch.float16))
            self.assertIsNone(choose(device="cpu"))
            with patch.dict(baseline, use_stream_k=True):
                self.assertIsNone(choose())
            with patch.dict(baseline, num_sms=40):
                self.assertIsNone(choose())
            with patch.object(serving, "speculative_config", ns()):
                self.assertIsNone(choose())
            with patch.object(serving.scheduler_config, "max_num_seqs", 2):
                self.assertIsNone(choose())
            with patch.object(torch.cuda, "get_device_capability", return_value=(8, 6)):
                self.assertIsNone(choose())
            with patch.dict(os.environ, KAILLM_HUMMING_PORTABLE_K64="0"):
                self.assertIsNone(choose())
