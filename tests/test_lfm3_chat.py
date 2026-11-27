"""Exercise the LFM3 chat defaults and explicit vLLM options."""

import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runtime-source"))

from vllm_gguf_plugin import serve, serving_defaults


def gguf_string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


class Lfm3ChatTest(unittest.TestCase):
    def command(self, architecture="lfm3", *extra):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp) / "model.gguf"
            model.write_bytes(
                b"GGUF" + struct.pack("<IQQ", 3, 0, 1)
                + gguf_string("general.architecture") + struct.pack("<I", 8)
                + gguf_string(architecture)
            )
            args = serve.parser().parse_args([
                str(model), "--kv-cache-memory-bytes", "1048576", "--spec-tokens", "0", *extra,
            ])
            family = serving_defaults.inspect(model)
            target = {"conditional_adapter": None, "environment": {}}
            return serve.prepare_command(args, target, family=family)["command"]

    def test_standard_chat_selects_reasoning_and_function_parser(self):
        command = self.command()
        self.assertIn("--enable-auto-tool-choice", command)
        self.assertEqual(command[command.index("--tool-call-parser") + 1], "qwen3_xml")
        self.assertEqual(command[command.index("--reasoning-parser") + 1], "qwen3")

    def test_auto_tool_choice_can_be_disabled(self):
        command = self.command("lfm3", "--no-enable-auto-tool-choice")
        self.assertIn("--no-enable-auto-tool-choice", command)
        self.assertNotIn("--enable-auto-tool-choice", command)
        self.assertNotIn("--tool-call-parser", command)
        self.assertIn("--reasoning-parser", command)

    def test_custom_template_requires_requested_parsers(self):
        forwarded = json.dumps(["--chat-template", "custom.jinja"])
        command = self.command("lfm3", "--extra-vllm-args", forwarded)
        self.assertNotIn("--enable-auto-tool-choice", command)
        self.assertNotIn("--reasoning-parser", command)
        self.assertNotIn("--tool-call-parser", command)

    def test_explicit_options_apply_to_custom_tokenizer(self):
        command = self.command("lfm3", "--tokenizer", "example/custom",
                               "--enable-auto-tool-choice", "--tool-call-parser", "hermes")
        self.assertIn("--enable-auto-tool-choice", command)
        self.assertEqual(command[command.index("--tool-call-parser") + 1], "hermes")
        self.assertNotIn("--reasoning-parser", command)

    def test_forwarded_overrides_preserve_precedence(self):
        forwarded = json.dumps(["--enable-auto-tool-choice", "--tool-call-parser=hermes",
                                "--reasoning-parser", "deepseek_r1"])
        command = self.command("lfm3", "--extra-vllm-args", forwarded)
        self.assertEqual(command.count("--enable-auto-tool-choice"), 1)
        self.assertNotIn("--tool-call-parser", command)
        self.assertIn("--tool-call-parser=hermes", command)
        self.assertEqual(command.count("--reasoning-parser"), 1)
        self.assertEqual(command[command.index("--reasoning-parser") + 1], "deepseek_r1")

    def test_other_gguf_family_keeps_its_own_defaults(self):
        command = self.command("qwen35")
        for option in ("--enable-auto-tool-choice", "--tool-call-parser", "--reasoning-parser"):
            self.assertNotIn(option, command)
