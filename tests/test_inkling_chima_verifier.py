"""Verify opt-in graph placement and dispatch without a CUDA device."""

from dataclasses import dataclass, replace
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime-source"))
from vllm_gguf_plugin.inkling import chima_verifier as verifier


@dataclass(frozen=True)
class Descriptor:
    num_tokens: int
    num_reqs: int = 1
    num_ubatches: int = 1
    num_active_loras: int = 0
    inkling_row_stable: bool = False


class ChimaVerifierTest(unittest.TestCase):
    def test_scope_is_explicit_and_restored_on_error(self):
        with patch.dict(os.environ, {verifier.ENV: "0"}):
            self.assertFalse(verifier.enabled())
        with patch.dict(os.environ, {verifier.ENV: "1"}):
            self.assertTrue(verifier.enabled())
        self.assertFalse(verifier.active())
        with self.assertRaises(ValueError), verifier.capture():
            self.assertTrue(verifier.active())
            raise ValueError("capture failed")
        self.assertFalse(verifier.active())

    def test_captures_cover_exact_single_slot_no_adapter_widths(self):
        selected = [Descriptor(width) for width in range(2, 7)]
        ordinary = [Descriptor(1), Descriptor(512), Descriptor(4, num_reqs=2),
                    Descriptor(4, num_ubatches=2), Descriptor(4, num_active_loras=1)]
        captures = [*selected, *ordinary]
        self.assertEqual(verifier.capture_twins(captures),
                         {d: replace(d, inkling_row_stable=True) for d in selected})
        self.assertEqual(len(captures), len(ordinary) + 2 * len(selected))
        with self.assertRaises(ValueError):
            verifier.capture_twins(selected[:-1])

    def test_dispatch_rejects_prefill_plain_padding_and_concurrency(self):
        state = dict(has_prefill=False, req_ids=["a"], num_draft_tokens_np=[5],
                     num_scheduled_tokens=[6], num_tokens=6)
        self.assertTrue(verifier.real_width(SimpleNamespace(**state)))
        invalid = (
            {"has_prefill": True},
            {"req_ids": ["a", "b"]},
            {"num_draft_tokens_np": None},
            {"num_draft_tokens_np": [4]},
            {"num_tokens": 512},
        )
        for change in invalid:
            with self.subTest(change=change):
                self.assertFalse(verifier.real_width(SimpleNamespace(**(state | change))))

    def test_explicit_placement_requires_gpu_shape_context_and_exact_weights(self):
        card = SimpleNamespace(major=12, minor=0, multi_processor_count=170,
                               name="NVIDIA GeForce RTX 5090")
        torch = SimpleNamespace(bfloat16="bfloat16", cuda=SimpleNamespace(
            get_device_properties=lambda _: card, device_count=lambda: 1))
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / verifier._TARGET
            data = b"test tensors"
            target.write_bytes(data)
            config = SimpleNamespace(
                model_config=SimpleNamespace(model_weights=target, max_model_len=73728, dtype="bfloat16",
                                             hf_config=SimpleNamespace(model_type="inkling2_loop")),
                speculative_config=SimpleNamespace(method="dflash"),
                num_speculative_tokens=5,
                scheduler_config=SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=512),
                parallel_config=SimpleNamespace(world_size_across_dp=1),
                cache_config=SimpleNamespace(cache_dtype="auto"),
                attention_config=SimpleNamespace(backend=SimpleNamespace(name="TRITON_ATTN")),
            )
            with (patch.dict(sys.modules, {"torch": torch}),
                  patch.dict(os.environ, {"KAILLM_INKLING_ATTN": "auto"}),
                  patch.object(verifier, "_TARGET_SHA256", hashlib.sha256(data).hexdigest())):
                verifier.validate(config, 0)
                for object_, attribute, invalid_value in (
                    (config, "num_speculative_tokens", 3),
                    (config.model_config, "max_model_len", 8192),
                    (config.scheduler_config, "max_num_batched_tokens", 2048),
                    (card, "multi_processor_count", 84),
                ):
                    with (patch.object(object_, attribute, invalid_value),
                          self.assertRaisesRegex(ValueError, "qualified RTX 5090 placement")):
                        verifier.validate(config, 0)
                target.write_bytes(data + b"x")
                with self.assertRaisesRegex(ValueError, "qualified target weights"):
                    verifier.validate(config, 0)
