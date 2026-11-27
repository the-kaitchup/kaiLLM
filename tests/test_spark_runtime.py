import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT / "tools"), str(PROJECT / "runtime-source")]
import install_spark as install
from kaillm import cli
from vllm_gguf_plugin.native import cuda_architecture, spark
from vllm_gguf_plugin.serving_policy import load_target


class SparkRuntime(unittest.TestCase):
    @staticmethod
    def put_fixture(root, rel, data):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return p

    def native_tree(self, root):
        elf = bytearray(64)
        elf[:6] = b"\x7fELF\x02\x01"
        elf[18:20] = b"\xb7\x00"
        names = [*spark.HUMMING, *(spark.CORE + "/" + n + ".abi3.so"
                 for n in ("_C_stable_libtorch","_moe_C_stable_libtorch","cumem_allocator")),
                 "compatible-vllm/core/vllm.libs/libtest.so.1",spark.CORE + "/vllm-rs"]
        hashes = {n:spark.digest(self.put_fixture(root,n,elf)) for n in names}
        manifest = self.put_fixture(root,"third-party/humming/_native/aarch64/manifest.json",b"{}")
        value = {"vllm_commit":spark.CORE_COMMIT,"compute_capability":[12,1],
                 "native_sha256":hashes,"humming_manifest_sha256":spark.digest(manifest)}
        self.put_fixture(root,spark.RECEIPT,json.dumps(value).encode())
        return value

    def test_target_and_native_code_identities(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(spark.platform,"system",return_value="Linux"), \
                patch.object(spark.platform,"machine",return_value="aarch64"):
            root = Path(tmp)
            expected = self.native_tree(root)
            self.assertEqual(spark.verify_runtime(root),expected)
            (root / spark.HUMMING[0]).write_bytes(b"\x7fELF\x02\x01"+bytes(12)+b"\x3e\x00")
            with self.assertRaisesRegex(ValueError,"ARM64 ELF"):
                spark.verify_runtime(root)
            self.native_tree(root)
            (root / spark.HUMMING[-1]).unlink()
            with self.assertRaises(FileNotFoundError):
                spark.verify_runtime(root)
            with patch.object(spark.platform,"machine",return_value="x86_64"), self.assertRaisesRegex(ValueError,"AArch64"):
                spark.verify_runtime(root)
        self.assertEqual(cuda_architecture.architecture_identity("sm121")["gencode"],"-gencode=arch=compute_121,code=sm_121")
        for arch in ("sm86","sm120"):
            self.assertEqual(cuda_architecture.architecture_identity(arch)["architecture_contract"],"explicit-sm86-sm120-native-YAQA-v1")

    def test_frozen_wheel_supplies_only_portable_runtime_members(self):
        root = "kaillm/_runtime/"
        for f in ("compatible-vllm/core/vllm/utils/humming.py","compatible-vllm/core/vllm/config.json",
                  "third-party/humming/op.py","vllm_gguf_plugin-0.0.5.dist-info/entry_points.txt"):
            self.assertEqual(str(install.wheel_member(root+f)),f)
        for f in ("compatible-vllm/core/vllm/_C.abi3.so","compatible-vllm/core/vllm/linked.so.0",
                  "compatible-vllm/core/vllm/vllm-rs",
                  "portable-system/ld-linux-x86-64.so.2","third-party/humming/_native/x86_64/nvrtc_compile",
                  "vllm_gguf_plugin/native/libraries/sm120/metadata.json","third-party/../../escape"):
            self.assertIsNone(install.wheel_member(root+f))

    def test_extraction_validates_the_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)
            wheel=work/"kai.whl"
            with zipfile.ZipFile(wheel,"w") as z:
                z.writestr("kaillm/_runtime/compatible-vllm/core/vllm/utils/humming.py","# source")
                z.writestr("kaillm/_runtime/compatible-vllm/core/vllm/_C.abi3.so",b"x86")
            archive=work/"test.tar.gz"
            with tarfile.open(archive,"w:gz") as t:
                t.add(wheel,arcname="kai/dist/kai.whl")
            release={"directory":"kai","sha256":install.sha256(archive)}
            install.frozen_sources(archive,work,work/"root",release)
            self.assertFalse((work/"root"/spark.CORE/"_C.abi3.so").exists())
            with self.assertRaisesRegex(ValueError,"SHA-256"):
                install.frozen_sources(archive,work,work/"root",{**release,"sha256":"0"*64})

    def test_official_wheel_revision_and_inconsistent_provenance(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(install.metadata, "distribution") as dist:
            core = Path(tmp)
            dist.return_value.locate_file.return_value = core
            dist.return_value.version = "0.32.1rc1.dev399+gf3871c3cb"
            dist.return_value.read_text.return_value = None
            self.assertEqual(install.native_core(), (core, dist.return_value.version))
            (core / "_version.py").write_text("__commit_id__ = commit_id = 'g111111111'")
            with self.assertRaisesRegex(ValueError, "vLLM CUDA build"):
                install.native_core()

    def test_native_catalog_and_unified_memory_inventory(self):
        catalog=PROJECT/"runtime-source/deployment/serving-catalog.json"
        mirror=PROJECT/"runtime-source/vllm_gguf_plugin/native/serving-catalog.json"
        self.assertEqual(json.loads(catalog.read_bytes())["targets"]["12.1"],json.loads(mirror.read_bytes())["targets"]["12.1"])
        with patch.object(spark,"verify_runtime",return_value={}):
            target=load_target(catalog,"12.1",PROJECT/"runtime-source/vllm_gguf_plugin",
                               {"VLLM_GGUF_NATIVE_BODY_LIBRARY":"/wrong/sm120.so"})
        self.assertIsNone(target["conditional_adapter"])
        self.assertEqual(target["environment"]["VLLM_GGUF_YAQA_CUDA_ARCH"],"sm121")
        self.assertNotIn("VLLM_GGUF_NATIVE_BODY_LIBRARY",target["environment"])
        with patch("kaillm.cli.subprocess.run") as query:
            query.return_value.stdout="12.1, [N/A], NVIDIA GB10\n"
            self.assertIsNone(cli.query_gpu()["total_memory_MiB"])
            query.return_value.stdout="12.0, 32607, NVIDIA GeForce RTX 5090\n"
            self.assertEqual(cli.query_gpu()["total_memory_MiB"],32607)


if __name__ == "__main__":
    unittest.main()
