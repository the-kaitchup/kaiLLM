# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from kaillm.install_native_runtime import write_launcher


class LauncherTests(unittest.TestCase):
    def test_managed_symlink_is_replaced_without_overwriting_entrypoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefix = root / "installation"
            target = prefix / "venv/bin/kaillm"
            target.parent.mkdir(parents=True)
            target.write_text("original Python entrypoint\n")
            (prefix / ".kaillm-installation").write_text("kaiLLM test\n")
            launcher = root / "bin/kaillm"
            launcher.parent.mkdir()
            launcher.symlink_to(target)
            write_launcher(launcher, target)
            self.assertEqual(target.read_text(), "original Python entrypoint\n")
            self.assertFalse(launcher.is_symlink())
            self.assertIn("exec " + str(target), launcher.read_text())
            self.assertTrue(launcher.stat().st_mode & 0o111)

    def test_unmanaged_destination_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefix = root / "installation"
            target = prefix / "venv/bin/kaillm"
            target.parent.mkdir(parents=True)
            target.write_text("managed entrypoint")
            (prefix / ".kaillm-installation").write_text("kaiLLM test\n")
            unrelated = root / "other"
            unrelated.write_text("unrelated")
            launcher = root / "kaillm"
            launcher.symlink_to(unrelated)
            with self.assertRaisesRegex(ValueError, "unmanaged"):
                write_launcher(launcher, target)
            self.assertEqual(unrelated.read_text(), "unrelated")
            self.assertTrue(launcher.is_symlink())

    def test_unsupported_gguf_fails_before_runtime_initialization(self):
        launcher = Path(__file__).resolve().parents[1] / "runtime-source/deployment/serve.py"

        def gguf_string(text):
            return struct.pack("<Q", len(text)) + text

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "target.gguf"
            for architecture, message in (
                (b"inkling2", "Choose a fixed-loop Inkling 2 Mini GGUF."),
                (b"other", "unsupported GGUF architecture 'other'"),
            ):
                with self.subTest(architecture=architecture):
                    model.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1)
                                      + gguf_string(b"general.architecture") + struct.pack("<I", 8)
                                      + gguf_string(architecture))
                    result = subprocess.run(
                        [sys.executable, "-B", str(launcher), str(model),
                         "--kv-cache-memory-bytes", "1", "--dry-run", "--compute-capability", "8.6",
                         "--catalog", str(root / "missing-catalog.json")],
                        capture_output=True, text=True, timeout=10,
                    )
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn(message, result.stderr)


if __name__ == "__main__":
    unittest.main()
