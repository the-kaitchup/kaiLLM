# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
