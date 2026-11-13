# SPDX-License-Identifier: Apache-2.0
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from install_release import extract_release


class ReleaseExtractionTests(unittest.TestCase):
    def archive(self, root, name, *, symlink=False):
        path = root / "test.tar.gz"
        with tarfile.open(path, "w:gz") as target:
            item = tarfile.TarInfo(name)
            if symlink:
                item.type = tarfile.SYMTYPE
                item.linkname = "../../outside"
                target.addfile(item)
            else:
                item.size = 7
                target.addfile(item, io.BytesIO(b"fixture"))
        return path

    def test_traversal_and_links_rejected(self):
        for name, link in (("release/../outside", False), ("/outside", False),
                           ("release/link", True), ("another/file", False)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                archive = self.archive(root, name, symlink=link)
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    extract_release(archive, root / "out", "release")

    def test_incomplete_archive_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self.archive(root, "release/unrelated.txt")
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                extract_release(archive, root / "out", "release")


if __name__ == "__main__":
    unittest.main()
