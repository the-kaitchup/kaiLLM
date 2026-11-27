"""Keep Inkling pair attention confined to the exact ordinary two-request graph."""

import ast
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SOURCE = Path(__file__).resolve().parents[1] / "runtime-source"
GRAPH = SOURCE / "vllm-overrides/vllm/v1/worker/gpu/cudagraph_utils.py"


class OrdinaryPairAttention(unittest.TestCase):
    def test_only_exact_uniform_two_request_replay_can_select_pair_attention(self):
        tree = ast.parse(GRAPH.read_text())
        select = next(node for node in tree.body if getattr(node, "name", "") == "_is_compatible")
        env = {"BatchExecutionDescriptor": object}
        exec(compile(ast.Module(body=[select], type_ignores=[]), "graph selector", "exec"), env)
        generic = SimpleNamespace(inkling_ordinary_pair_attention=False, num_reqs=2, num_tokens=2,
                                  uniform_token_count=None, max_query_len=None, num_active_loras=0, num_ubatches=1)
        pair = SimpleNamespace(**vars(generic) | {"inkling_ordinary_pair_attention": True})
        compatible = env["_is_compatible"]
        self.assertTrue(compatible(pair, 2, 2, 1, 0, 1, 1))
        for args in ((1, 2, 2, 0, 2, 1), (1, 1, 1, 0, 1, 1), (2, 2, None, 0, 1, 1),
                     (2, 2, 1, 0, None, 1), (2, 2, 1, 1, 1, 1), (2, 2, 1, 0, 1, 2)):
            self.assertFalse(compatible(pair, *args))
        self.assertTrue(compatible(generic, 1, 2, None, 0, 2, 1))

    def test_capability_rechecks_runtime_dtype_and_backend_for_both_layouts(self):
        tree = ast.parse(GRAPH.read_text())
        assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign) and
                          any(isinstance(t, ast.Name) and t.id == "ordinary_pair_attention" for t in node.targets))
        code = compile(ast.Module(body=[assignment], type_ignores=[]), "pair capability", "exec")
        torch = SimpleNamespace(bfloat16=object())
        attn = SimpleNamespace(backend=SimpleNamespace(name="TRITON_ATTN"))
        model = SimpleNamespace(hf_config=SimpleNamespace(model_type="inkling2_loop"), dtype=torch.bfloat16)
        config = SimpleNamespace(additional_config={"inkling_ordinary_pair_attention": True},
                                 model_config=model, cache_config=SimpleNamespace(cache_dtype="auto"),
                                 attention_config=attn)
        manager = SimpleNamespace(vllm_config=config, cudagraph_mode="full", max_num_reqs=2, decode_query_len=1)

        def eligible(spec=None):
            env = dict(self=manager, speculative_config=spec, CUDAGraphMode=SimpleNamespace(FULL="full"),
                       os=os, torch=torch)
            exec(code, env)
            return env["ordinary_pair_attention"]

        with patch.dict(os.environ, {"KAILLM_INKLING_ATTN": "triton"}):
            self.assertTrue(eligible())
            self.assertFalse(eligible(spec=object()))
            config.cache_config.cache_dtype = "fp8"
            self.assertFalse(eligible())
            config.cache_config.cache_dtype = "bfloat16"
            self.assertTrue(eligible())
            attn.backend.name = "FLASH_ATTN"
            self.assertFalse(eligible())
            attn.backend.name = "TRITON_ATTN"
            model.dtype = object()
            self.assertFalse(eligible())
            model.dtype = torch.bfloat16
            model.hf_config.model_type = "qwen3"
            self.assertFalse(eligible())
            model.hf_config.model_type = "inkling2_loop"
        with patch.dict(os.environ, {"KAILLM_INKLING_ATTN": "torch"}):
            self.assertFalse(eligible())

    def test_exact_capture_owns_tiles_and_leaves_recurrent_metadata_intact(self):
        tree = ast.parse(GRAPH.read_text())
        capture = next(node for node in ast.walk(tree) if isinstance(node, ast.If) and
                       any(isinstance(item, ast.ImportFrom) and item.module == "vllm_gguf_plugin.inkling.attention"
                           for item in node.body))
        code = compile(ast.Module(body=[capture], type_ignores=[]), "capture", "exec")

        class Metadata:
            tiles = None
            one_pass_decode = False

        manager = SimpleNamespace(device="cpu")
        tile_buffer = object()
        torch = SimpleNamespace(int32="int32", tensor=Mock(return_value=tile_buffer))
        with patch.dict(sys.modules, {"vllm_gguf_plugin.inkling.attention": SimpleNamespace(InklingAttnMetadata=Metadata)}):
            for active in (False, True, True):
                attention, recurrent = Metadata(), SimpleNamespace()
                exec(code, dict(self=manager, desc=SimpleNamespace(inkling_ordinary_pair_attention=active),
                                attn_metadata={"attention": attention, "recurrent": recurrent}, torch=torch))
                self.assertIs(attention.one_pass_decode, active)
                self.assertIs(attention.tiles, tile_buffer if active else None)
                self.assertEqual(vars(recurrent), {})
        torch.tensor.assert_called_once_with(((0, 0), (1, 0)), dtype="int32", device="cpu")

    def test_only_marked_attention_skips_split_accumulation(self):
        path = SOURCE / "vllm_gguf_plugin/inkling/triton_attention.py"
        tree = ast.parse(path.read_text())
        func = next(node for node in tree.body if getattr(node, "name", "") == "rel_attention_triton")
        guard = next(node for node in func.body if isinstance(node, ast.If) and
                     any(isinstance(item, ast.Name) and item.id == "_rel_attn_split_kernel" for item in ast.walk(node)))
        code = compile(ast.Expression(guard.test), "attention selector", "eval")
        for serial, splits, selected, expected in [(True, 16, False, True), (True, 16, True, False),
                                                    (False, 16, False, False), (True, 1, False, False)]:
            self.assertIs(eval(code, dict(serial_attention=serial, SPLITS=splits,
                                         md=SimpleNamespace(one_pass_decode=selected))), expected)


if __name__ == "__main__":
    unittest.main()
