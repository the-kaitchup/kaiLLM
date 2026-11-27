"""Verify assignment to the cards named in the benchmark receipt."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/long_context"))
from devices import host_memory
from worker import admission_fraction, selected_devices


class BenchmarkDeviceTest(unittest.TestCase):
    def test_reordered_mask(self):
        inventory = [{"uuid": u} for u in ("GPU-A", "GPU-B", "GPU-C")]
        visible = ["GPU-C", "GPU-A"]
        self.assertEqual(selected_devices(inventory, None, visible), [inventory[2]])
        self.assertEqual(selected_devices(inventory, [0, 2], visible), [inventory[0], inventory[2]])
        for indices in ([1], [0, 0], [-1], [3]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                selected_devices(inventory, indices, visible)

    def test_no_visible_device(self):
        with self.assertRaisesRegex(ValueError, "No CUDA device"):
            selected_devices([], None, [])
        with self.assertRaisesRegex(ValueError, "missing"):
            selected_devices([{"uuid": "GPU-A"}], None, ["GPU-C"])

    def test_admission_uses_least_available_card(self):
        full = {"memory.free": "24300", "memory.total": "24576"}
        desktop = {"memory.free": "10782", "memory.total": "12288"}
        self.assertEqual(admission_fraction([full]), .94)
        self.assertEqual(admission_fraction([desktop, full]), .8566)
        self.assertEqual(admission_fraction([full, desktop]), .8566)
        with self.assertRaisesRegex(ValueError, "Insufficient free"):
            admission_fraction([{"memory.free": "250", "memory.total": "12288"}])

    def test_spark_uses_process_visible_shared_memory(self):
        gib = 1 << 30
        device = {"uuid": "GPU-test", "name": "NVIDIA GB10", "compute_cap": "12.1",
                  "memory.free": "[N/A]", "memory.total": "[N/A]"}
        memory = {"uuid": "GPU-test", "cuda_total_bytes": 119 * gib, "cuda_free_bytes": 74 * gib,
                  "host_total_bytes": 122 * gib, "host_available_bytes": 70 * gib,
                  "cgroup_limits": [{"maximum_bytes": 100 * gib, "current_bytes": 60 * gib}]}
        with patch("devices.subprocess.check_output", return_value=json.dumps(memory)):
            spark, = selected_devices([device], [0], [device["uuid"]])
            self.assertEqual(spark["memory.free"], str(40 * 1024))
            self.assertEqual(spark["memory.total"], str(119 * 1024))
            self.assertEqual(spark["unified_memory"]["nvml_free"], "[N/A]")
        with patch("devices.subprocess.check_output", return_value=json.dumps({**memory, "uuid": "GPU-other"})):
            with self.assertRaisesRegex(ValueError, "UUID"):
                selected_devices([device], [0], [device["uuid"]])

    def test_shared_memory_includes_parent_cgroup_limits(self):
        with TemporaryDirectory() as tmp:
            mount = Path(tmp)
            child = mount / "tenant"
            child.mkdir()
            for directory, maximum, current in ((mount, "1000", "860"), (child, "max", "20")):
                (directory / "memory.max").write_text(maximum)
                (directory / "memory.current").write_text(current)
            reading = host_memory("MemTotal: 16 kB\nMemAvailable: 7 kB\n", "0::/tenant\n",
                                  f"0 0 0:1 / {mount} rw - cgroup2 none rw\n")
            self.assertEqual(reading["host_available_bytes"], 7 * 1024)
            self.assertEqual(reading["cgroup_limits"], [{"maximum_bytes": 1000, "current_bytes": 860}])
