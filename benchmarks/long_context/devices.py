"""Enumerate the CUDA devices visible to the current process in CUDA ordinal order."""
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from uuid import UUID


def physical_id(uuids):
    """Compact identifier used by the two reports for the same ordered physical devices."""
    encoded = json.dumps(list(uuids), separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def is_gb10(device):
    name = device.get("name", "").upper()
    return device.get("compute_cap") == "12.1" and ("GB10" in name or "DGX SPARK" in name)


def check(status, operation):
    if status:
        raise RuntimeError(f"{operation} failed with CUDA driver status {status}")


def visible_cuda_uuids():
    """Read driver device identities without allocating a CUDA context."""
    driver = ctypes.CDLL("libcuda.so.1")

    status = driver.cuInit(0)
    if status == 100:  # CUDA_ERROR_NO_DEVICE
        return []
    check(status, "cuInit")
    count = ctypes.c_int()
    check(driver.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
    get_uuid = getattr(driver, "cuDeviceGetUuid_v2", None) or driver.cuDeviceGetUuid
    result = []
    for ordinal in range(count.value):
        device = ctypes.c_int()
        check(driver.cuDeviceGet(ctypes.byref(device), ordinal), "cuDeviceGet")
        value = (ctypes.c_ubyte * 16)()
        check(get_uuid(ctypes.byref(value), device), "cuDeviceGetUuid")
        result.append("GPU-" + str(UUID(bytes=bytes(value))))
    return result


def host_memory(meminfo, memberships, mounts):
    """Measure available RAM and the hard limits visible in the process's memory cgroup."""
    fields = dict(re.findall(r"^(MemTotal|MemAvailable):\s+(\d+) kB$", meminfo, re.M))
    total, available = (int(fields[key]) * 1024 for key in ("MemTotal", "MemAvailable"))
    if not 0 <= available <= total or total == 0:
        raise ValueError("Invalid host MemTotal/MemAvailable")
    groups = {}
    for row in memberships.splitlines():
        hierarchy, controllers, member = row.split(":", 2)
        if hierarchy == "0" and not controllers:
            groups["cgroup2"] = member
        elif "memory" in controllers.split(","):
            groups["cgroup"] = member
    limits = []
    seen = set()
    for row in mounts.splitlines():
        left, right = row.split(" - ", 1)
        kind, _, options = right.split()[:3]
        if kind not in groups or (kind == "cgroup" and "memory" not in options.split(",")):
            continue
        # Linux mountinfo escapes whitespace and backslashes as octal.
        root, mount = (Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), part))
                       for part in left.split()[3:5])
        member = Path(groups[kind])
        if member.is_relative_to(root) and ".." not in member.parts:
            current = mount / member.relative_to(root)
        elif member == Path("/"):
            current = mount
        else:
            continue
        if not current.is_dir():
            continue
        seen.add(kind)
        limit_name, usage_name = ("memory.max", "memory.current") if kind == "cgroup2" else (
            "memory.limit_in_bytes", "memory.usage_in_bytes")
        while True:
            limit_path = current / limit_name
            if limit_path.exists():
                maximum = limit_path.read_text().strip()
                if maximum != "max":
                    limit, usage = int(maximum), int((current / usage_name).read_text())
                    if limit < 0 or usage < 0:
                        raise ValueError("Invalid cgroup memory counters")
                    limits.append({"maximum_bytes": limit, "current_bytes": usage})
            if current == mount:
                break
            current = current.parent
    if set(groups) - seen:
        raise ValueError("Cannot locate this process's memory cgroup mount")
    return {"host_total_bytes": total, "host_available_bytes": available, "cgroup_limits": limits}


def unified_memory(uuid):
    """Read CUDA memory on the selected UUID. The inventory child releases its context before serving."""
    ordinal = visible_cuda_uuids().index(uuid)
    driver = ctypes.CDLL("libcuda.so.1")
    device = ctypes.c_int()
    check(driver.cuDeviceGet(ctypes.byref(device), ordinal), "cuDeviceGet")
    context = ctypes.c_void_p()
    check(driver.cuDevicePrimaryCtxRetain(ctypes.byref(context), device), "cuDevicePrimaryCtxRetain")
    try:
        check(driver.cuCtxSetCurrent(context), "cuCtxSetCurrent")
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        check(driver.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)), "cuMemGetInfo")
        host = host_memory(*(Path(p).read_text() for p in
                             ("/proc/meminfo", "/proc/self/cgroup", "/proc/self/mountinfo")))
        return {"uuid": uuid, "cuda_free_bytes": free.value, "cuda_total_bytes": total.value, **host}
    finally:
        check(driver.cuCtxSetCurrent(None), "cuCtxSetCurrent")
        check(driver.cuDevicePrimaryCtxRelease_v2(device), "cuDevicePrimaryCtxRelease")


def shared_memory_inventory(device):
    """Keep the raw NVML reading and bound shared-memory admission by CUDA, host and cgroup headroom."""
    if not is_gb10(device):
        return device
    reading = json.loads(subprocess.check_output(
        [sys.executable, str(Path(__file__).resolve()), "--unified-memory", device["uuid"]], text=True))
    free, total = reading["cuda_free_bytes"], reading["cuda_total_bytes"]
    if reading["uuid"] != device["uuid"] or not 0 <= free <= total or total <= 0:
        raise ValueError("Invalid physical UUID or CUDA memory reading for GB10")
    remaining = [max(0, x["maximum_bytes"] - x["current_bytes"]) for x in reading["cgroup_limits"]]
    ceiling = min(free, reading["host_available_bytes"], *remaining)
    return {**device, "unified_memory": {**reading, "admission_free_bytes": ceiling,
            "nvml_total": device["memory.total"], "nvml_free": device["memory.free"]},
            "memory.total": str(total // (1 << 20)), "memory.free": str(ceiling // (1 << 20))}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Measure the current process's GB10 shared memory")
    parser.add_argument("--unified-memory", required=True, metavar="GPU_UUID")
    print(json.dumps(unified_memory(parser.parse_args().unified_memory)))
