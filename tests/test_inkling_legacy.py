"""Compare the legacy fixed-loop layout and independent mainline layout through the weight loader."""
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "runtime-source/gguf-reader"), str(ROOT / "runtime-source")]
import gguf

from inkling_toy import TOY, build, read_kv
from vllm_gguf_plugin import serve
from vllm_gguf_plugin.inkling.dspark_weights import prepare as prepare_drafter
from vllm_gguf_plugin.inkling_metadata import inkling_config_dict, legacy_loop_metadata, prepare_inkling_metadata
from vllm_gguf_plugin.qwen35_metadata import read_gguf_metadata
from vllm_gguf_plugin.serving_defaults import inspect
from vllm_gguf_plugin.weights_adapter.inkling import InklingGGUFAdapter, legacy_message_queries


def finish(writer):
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def make_pair(directory):
    source, modern, legacy = [directory / f"{s}.gguf" for s in ("header", "modern", "legacy")]
    vt = gguf.GGUFValueType
    writer = gguf.GGUFWriter(str(source), "inkling2_loop")
    for name in ("vocab_size", "vocab_size_unpadded"):
        writer.add_key_value("inkling2_loop." + name, 32, vt.UINT32)
    writer.add_key_value("tokenizer.ggml.model", "gpt2", vt.STRING)
    writer.add_key_value("tokenizer.ggml.pre", "inkling2", vt.STRING)
    writer.add_key_value("tokenizer.ggml.tokens", ["a", "b", "ab"], vt.ARRAY, sub_type=vt.STRING)
    writer.add_key_value("tokenizer.ggml.token_type", [1, 1, 1], vt.ARRAY, sub_type=vt.INT32)
    writer.add_key_value("tokenizer.ggml.merges", ["a b"], vt.ARRAY, sub_type=vt.STRING)
    finish(writer)

    toy = dict(TOY, embedding_length=16, head=2, head_kv=1, head_dim=8, experts=3, used=2, ff=8, rank=3)
    weights = build(source, modern, scales=True, toy=toy, gentle=True)
    start, end = toy["start"], toy["end"]
    visits = list(range(start)) + list(range(start, end)) * toy["passes"] + list(range(end, toy["block_count"]))
    w = gguf.GGUFWriter(str(legacy), "inkling2")
    aliases = {"loop.start": "loop.core_start", "loop.end": "loop.core_end", "loop.count": "num_loops"}
    for name, kind, value, subtype in read_kv(modern):
        if name.startswith("tokenizer."):
            w.add_key_value(name, value, vt(kind), sub_type=vt(subtype) if subtype is not None else None)
            continue
        if not name.startswith("inkling2_loop.") or name.endswith(".loop.varying"):
            continue
        key = name.removeprefix("inkling2_loop.")
        if key == "block_count":
            value = len(visits)
        if kind == vt.ARRAY:
            value = [value[i] for i in visits]
        w.add_key_value("inkling2." + aliases.get(key, key), value, vt(kind), sub_type=vt(subtype) if subtype is not None else None)
    queries = []
    for name, value in weights.items():
        if name.endswith(".loop_msg_query.weight"):
            queries.append((int(name.split(".")[1]), value))
            continue
        name = re.sub(r"loop_(q|k|v|ffn)_ctrl_", lambda m: "loop_" +
                      {"q": "attn", "k": "k", "v": "v", "ffn": "mlp"}[m[1]] + "_", name)
        w.add_tensor(name, value)
    blocks = [value for i, value in sorted(queries)]
    w.add_tensor("loop_res_query.weight", np.concatenate([v[[0, 3]] for v in blocks]))
    w.add_tensor("loop_kv_res_query.weight", np.concatenate([v[[1, 2]] for v in blocks]))
    finish(w)
    return modern, legacy


def make_drafter(path, taps):
    writer = gguf.GGUFWriter(str(path), "dflash")
    for name, value in {
        "embedding_length": 16, "block_count": 2, "block_size": 6, "context_length": 1024,
        "feed_forward_length": 8, "attention.head_count": 2, "attention.head_count_kv": 1,
        "attention.key_length": 8, "attention.sliding_window": 8,
    }.items():
        writer.add_uint32("dflash." + name, value)
    writer.add_float32("dflash.attention.layer_norm_rms_epsilon", 1e-5)
    writer.add_float32("dflash.rope.freq_base", 10000.0)
    writer.add_float32("dflash.final_logit_softcapping", 30.0)
    writer.add_bool("dflash.sample_from_anchor", False)
    writer.add_array("dflash.attention.sliding_window_pattern", [True, False])
    writer.add_array("dflash.target_layers", taps)
    writer.add_tensor("joint_proj.weight", np.zeros((6, 36), np.float32))
    writer.add_tensor("markov_w1.weight", np.zeros((32, 2), np.float32))
    writer.add_tensor("markov_w2.weight", np.zeros((32, 2), np.float32))
    finish(writer)


class InklingLegacyTest(unittest.TestCase):
    def test_same_parameters_after_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            modern, legacy = make_pair(Path(temp))
            results = []
            configs = []
            for path in (modern, legacy):
                reader = gguf.GGUFReader(str(path))
                configs.append(inkling_config_dict(read_gguf_metadata(reader)))
                adapter = InklingGGUFAdapter(None)
                adapter.prepare_loading(path, None)
                results.append(dict(adapter.prepare_weights(None)))
                self.assertEqual(inspect(path)["cudagraph_mode"], "FULL_DECODE_ONLY")
                self.assertIsNone(inspect(path)["tokenizer"])
            metadata = [prepare_inkling_metadata(p, cache_dir=temp) for p in (modern, legacy)]
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
                self.assertEqual((metadata[0] / name).read_bytes(), (metadata[1] / name).read_bytes(), name)
            self.assertEqual(configs[0], configs[1])
            self.assertEqual(results[0].keys(), results[1].keys())
            for key in results[0]:
                self.assertTrue(torch.equal(results[0][key], results[1][key]), key)

    def test_native_drafter_uses_logical_visit_ids_for_both_layouts(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            modern, legacy = make_pair(directory)
            draft = directory / "drafter.gguf"
            # The toy target has eight physical layers and sixteen logical visits.
            taps = [2, 7, 13, 16]
            make_drafter(draft, taps)
            for target in (modern, legacy):
                _, config = prepare_drafter(draft, target, materialize=False)
                self.assertEqual(config["dflash_config"]["layer_ids"], taps)
                self.assertEqual(config["unpadded_vocab_size"], 32)
                args = serve.parser().parse_args([str(target), "--dspark", str(draft), "--kv-cache-memory-bytes", "1048576", "--dry-run"])
                result = serve.prepare_command(args, {"environment": {}}, family=inspect(target))
                command = result["command"]
                spec = json.loads(command[command.index("--speculative-config") + 1])
                self.assertEqual((spec["method"], spec["num_speculative_tokens"], spec["attention_backend"]),
                                 ("dflash", 5, "TRITON_ATTN"))
                self.assertNotIn("--enforce-eager", command)
                compilation = json.loads(command[command.index("--compilation-config") + 1])
                self.assertEqual(compilation["cudagraph_mode"], "FULL")
            wrong = directory / "wrong-taps.gguf"
            make_drafter(wrong, [2, 7, 13, 17])
            for target in (modern, legacy):
                with self.assertRaisesRegex(ValueError, "logical visit"):
                    prepare_drafter(wrong, target, materialize=False)
            unlooped = directory / "unlooped.gguf"
            finish(gguf.GGUFWriter(str(unlooped), "inkling2"))
            with self.assertRaisesRegex(ValueError, "fixed-loop"):
                prepare_drafter(draft, unlooped, materialize=False)

    def test_reject_missing_physical_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            _, legacy = make_pair(Path(temp))
            reader = gguf.GGUFReader(str(legacy))
            tensors = {t.name: t for t in reader.tensors}
            config = inkling_config_dict(read_gguf_metadata(reader))
            for key, error in (("blk.7.attn_q.weight", "physical attention"),
                               ("blk.3.loop_mlp_gain.weight", "loop controller"),
                               ("loop_kv_res_query.weight", "packed Inkling loop query")):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, error):
                    legacy_message_queries({k: v for k, v in tensors.items() if k != key}, config)

    def test_reject_inconsistent_logical_schedule(self):
        md = {"general.architecture": "inkling2", "inkling2.num_loops": 2,
              "inkling2.loop.core_start": 1, "inkling2.loop.core_end": 3, "inkling2.block_count": 6,
              "inkling2.attn_res.block_size": 1, "inkling2.loop.gate_rank": 1,
              "inkling2.attention.head_count": [1, 2, 3, 2, 7, 8]}
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            legacy_loop_metadata(md)
        for invalid in ({"inkling2.loop.core_start": 0}, {"inkling2.attn_res.block_size": 2},
                        {"inkling2.loop.gate_rank": 0}):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "loop bounds"):
                legacy_loop_metadata({**md, **invalid})


if __name__ == "__main__":
    unittest.main()
