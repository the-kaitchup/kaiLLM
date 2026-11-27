"""Admit certified llama.cpp builds and read their CUDA graph counters at wave boundaries."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import struct


MAGIC = 0x4B41494752415048
VERSION = 2
PHASES = (
    "unattributed",
    "prefill",
    "decode",
    "mixed",
    "proposal",
    "draft_prefill",
    "draft_decode",
    "draft_mixed",
)
EVENTS = (
    "batches",
    "captures",
    "capture_launches",
    "replays",
    "rejected",
    "updates",
    "proposals",
    "completed_proposals",
)
DEVICE_COUNTS = (1, 2, 4)
LAYOUT = struct.Struct(f"={4 + len(PHASES) * 4 * len(EVENTS)}Q")
CERTIFICATE = "kaillm-graph-proof.json"
BINARY_NAMES = ("llama-server", "libggml-cuda.so", "libllama-server-impl.so")
SOURCE_NAMES = (
    "ggml/include/kaillm-graph.hpp",
    "ggml/src/ggml-cuda/ggml-cuda.cu",
    "tools/server/server-context.cpp",
)
SOURCE_LOCK = Path(__file__).with_name("llama_graph_sources.json")


@lru_cache(maxsize=24)
def _cached_digest(path, device, inode, size, modified, changed):
    # Invalidate the cache even when files are replaced with the same timestamps.
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(path):
    details = path.stat()
    return _cached_digest(
        path,
        details.st_dev,
        details.st_ino,
        details.st_size,
        details.st_mtime_ns,
        details.st_ctime_ns,
    )


def certified(server_path, upstream_rev, device_count):
    """Verify the pinned server binary and implementation before selecting execution policy."""
    server = Path(server_path).resolve(strict=True)
    binary_dir = server.parent
    certificate = json.loads((binary_dir / CERTIFICATE).read_bytes())
    sources = json.loads(SOURCE_LOCK.read_bytes())
    if (
        device_count not in DEVICE_COUNTS
        or certificate.get("protocol") != VERSION
        or upstream_rev not in sources
        or certificate.get("upstream") != upstream_rev
        or certificate.get("source_sha256") != sources[upstream_rev]
    ):
        raise ValueError("Uncertified llama.cpp graph protocol or source revision")
    if (
        server.name != "llama-server"
        or set(certificate.get("binary_sha256", {})) != set(BINARY_NAMES)
        or any(
            certificate["binary_sha256"][name]
            != digest((binary_dir / name).resolve(strict=True))
            for name in BINARY_NAMES
        )
    ):
        raise ValueError(
            "llama.cpp graph certificate does not match the executable and its libraries"
        )
    return binary_dir, certificate


def prepare(server_path, proof_path, upstream_rev, device_count, *, base_env=None):
    """Require strict graph execution and create fresh counters on this launch."""
    binary_dir, _ = certified(server_path, upstream_rev, device_count)
    path = Path(proof_path).absolute()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(
            LAYOUT.pack(
                MAGIC,
                VERSION,
                device_count,
                0,
                *([0] * (len(PHASES) * 4 * len(EVENTS))),
            )
        )
    env = os.environ if base_env is None else base_env
    library_path = os.pathsep.join(
        filter(None, (str(binary_dir), env.get("LD_LIBRARY_PATH")))
    )
    return {"KAILLM_LLAMA_GRAPH_FILE": str(path), "LD_LIBRARY_PATH": library_path}


def snapshot(proof_path):
    """Read only between completed waves, with all requested responses drained."""
    values = LAYOUT.unpack(Path(proof_path).read_bytes())
    signature, version, devices, active = values[:4]
    if (
        signature != MAGIC
        or version != VERSION
        or devices not in DEVICE_COUNTS
        or active != 0
    ):
        raise ValueError(
            "Missing llama.cpp graph evidence or model batch still active at a wave boundary"
        )
    phases = {}
    for index, phase in enumerate(PHASES):
        rows = []
        for device in range(devices):
            first = 4 + (index * 4 + device) * len(EVENTS)
            rows.append(
                dict(zip(EVENTS, values[first : first + len(EVENTS)], strict=True))
            )
        phases[phase] = rows
    return {"version": VERSION, "device_count": devices, "phases": phases}


def delta(before, after):
    if (
        before["device_count"] != after["device_count"]
        or before["version"] != VERSION
        or after["version"] != VERSION
    ):
        raise ValueError("llama.cpp graph counter layout changed within a wave")
    phases = {}
    for phase in PHASES:
        rows = []
        for first, last in zip(
            before["phases"][phase], after["phases"][phase], strict=True
        ):
            counts = {event: last[event] - first[event] for event in EVENTS}
            if min(counts.values()) < 0:
                raise ValueError(
                    "llama.cpp graph counters restarted during a benchmark wave"
                )
            rows.append(counts)
        phases[phase] = rows
    return {
        "version": VERSION,
        "device_count": before["device_count"],
        "phases": phases,
    }


def verify_interval(before, after, required, uuids):
    """Require successful CUDA executable launches in each phase on every selected GPU."""
    counts = delta(before, after)
    if len(uuids) != counts["device_count"] or len(set(uuids)) != len(uuids):
        raise ValueError(
            "Selected GPU identities do not match the llama.cpp runtime order"
        )
    for phase, rows in counts["phases"].items():
        for uuid, row in zip(uuids, rows, strict=True):
            if row["rejected"] or (phase == "unattributed" and any(row.values())):
                raise ValueError(
                    f"Rejected or unattributable llama.cpp {phase} model work on {uuid}"
                )
            launches = row["capture_launches"] + row["replays"]
            if (
                row["captures"] != row["capture_launches"]
                or launches < row["batches"]
                or row["updates"] > row["capture_launches"]
            ):
                raise ValueError(
                    f"Missing successful model graph launch for llama.cpp {phase} on {uuid}"
                )
    # One classified mixed batch can service both a prompt chunk and a decoding
    # slot. Its original counter is retained, and both phases receive credit.
    phases = counts["phases"]
    result = {}
    for phase in ("prefill", "decode"):
        per_gpu = {}
        for uuid, pure, mixed in zip(
            uuids, phases[phase], phases["mixed"], strict=True
        ):
            combined = {event: pure[event] + mixed[event] for event in EVENTS}
            combined["graph_launches"] = (
                combined["capture_launches"] + combined["replays"]
            )
            if phase in required and (
                not combined["batches"] or not combined["graph_launches"]
            ):
                raise ValueError(
                    f"No successful llama.cpp CUDA graph launch for {phase} on {uuid}"
                )
            per_gpu[uuid] = {**combined, "mixed_replays": mixed["replays"]}
        result[phase] = per_gpu
    return result


def verify_draft_interval(before, after, uuids, require_proposal):
    """Count each neural proposal API call only after a real CUDA launch in that same invocation."""
    counts = delta(before, after)
    if len(uuids) != counts["device_count"] or len(set(uuids)) != len(uuids):
        raise ValueError("Missing or duplicate neural draft device")
    result = {}
    for uuid, row in zip(uuids, counts["phases"]["proposal"], strict=True):
        launches = row["capture_launches"] + row["replays"]
        if (
            row["rejected"]
            or row["proposals"] != row["completed_proposals"]
            or launches < row["proposals"]
            or bool(launches) != bool(row["proposals"])
            or (require_proposal and not row["proposals"])
        ):
            raise ValueError(f"Incomplete neural proposal graph execution on {uuid}")
        result[uuid] = {
            "proposals": row["proposals"],
            "completed_proposals": row["completed_proposals"],
            "graph_replays": launches,
        }
    return result
