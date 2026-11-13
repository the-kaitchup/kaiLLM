# SPDX-License-Identifier: Apache-2.0
"""Default-off complete-checkpoint canonical BF16 reuse in an owned tmpfs slot."""

import fcntl
import hashlib
import json
import mmap
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

MODE_ENV = "VLLM_GGUF_YAQA_PREFILL_CPU_CACHE"
BUDGET_ENV = MODE_ENV + "_BYTES"
DIR_ENV = MODE_ENV + "_DIR"
SCHEMA = "gguf-yaqa-canonical-cpu-cache-v1"
MAX_BUDGET_BYTES = 3 * 1024**3
DEFAULT_DIR = "/dev/shm/vllm-yaqa-canonical-cpu-cache"


def mode():
    value = os.getenv(MODE_ENV, "none")
    if value not in ("none", "ram-bf16-v1"):
        raise ValueError(MODE_ENV + " must be none or ram-bf16-v1")
    return value


def settings():
    selected = mode()
    if selected == "none":
        return {"mode": "none", "budget_bytes": 0, "directory": None}
    budget = int(os.getenv(BUDGET_ENV, "0"))
    if not 0 < budget <= MAX_BUDGET_BYTES:
        raise ValueError("RAM canonical reuse needs an explicit CPU byte budget <=3GiB")
    return {
        "mode": selected,
        "budget_bytes": budget,
        "directory": os.getenv(DIR_ENV, DEFAULT_DIR),
    }


def sha_file(path):
    value = hashlib.sha256()
    before = os.stat(path)
    stamp = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            value.update(chunk)
    after = os.stat(path)
    if stamp != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError("Source changed while hashing: " + str(path))
    return value.hexdigest()


def json_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def require_tmpfs(directory):
    resolved = directory.resolve()
    matches = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        before, separator, after = line.partition(" - ")
        if not separator:
            continue
        mount = before.split()[4]
        for escaped, literal in (("\\040", " "), ("\\011", "\t"), ("\\134", "\\")):
            mount = mount.replace(escaped, literal)
        if resolved.is_relative_to(Path(mount)):
            matches.append((len(mount), after.split()[0]))
    if not matches or max(matches)[1] != "tmpfs":
        raise ValueError(
            "Canonical CPU reuse requires an explicitly owned tmpfs directory"
        )


class CacheSession:
    """An exclusive whole-checkpoint lease; partial files are never cache hits."""

    def __init__(self, entries, reconstruction_sources, directory, budget, backend):
        self.entries = entries
        self.directory = Path(directory).absolute()
        self.budget = budget
        self.required_bytes = sum(row["bytes"] for row in entries.values())
        self.metadata_reserve = 65536 + len(entries) * 8192
        self.closed = False
        self.complete = False
        self.records = {}
        self.used = set()
        self.reconstructed = 0
        self.hits = 0
        self.views_open = 0
        self.lease = None
        self.incoming = None
        self.hit = False
        self.reason = "empty"
        if (
            not entries
            or not 0 < budget <= MAX_BUDGET_BYTES
            or budget < self.required_bytes + self.metadata_reserve
        ):
            raise ValueError(
                "Complete checkpoint plus metadata exceeds explicit CPU budget"
            )
        if self.directory.is_symlink() or not self.directory.name.startswith(
            "vllm-yaqa-canonical-"
        ):
            raise ValueError("CPU cache must use its own canonical-cache namespace")
        require_tmpfs(self.directory)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.directory.stat().st_uid != os.getuid():
            raise ValueError("Canonical CPU cache namespace belongs to another owner")
        if self.directory.stat().st_mode & 0o022:
            raise ValueError("CPU cache namespace must not be writable by other users")
        lock_fd = os.open(
            self.directory / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
        )
        self.lease = os.fdopen(lock_fd, "r+b")
        try:
            fcntl.flock(self.lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.lease.close()
            self.lease = None
            raise RuntimeError(
                "Canonical CPU cache is in use by another constructor"
            ) from error
        try:
            owner = self.directory / "owner.json"
            names = {path.name for path in self.directory.iterdir()}
            if not owner.exists():
                if names != {".lock"}:
                    raise ValueError("Refusing a nonempty unowned RAM cache directory")
                write_json(owner, {"schema": SCHEMA})
            if owner.is_symlink() or json.loads(owner.read_text()) != {
                "schema": SCHEMA
            }:
                raise ValueError("RAM cache owner/schema mismatch")
            if any(
                name not in {".lock", "owner.json", "slot"}
                and not name.startswith("incoming-")
                for name in names
            ):
                raise ValueError("Unrecognized files in owned RAM cache namespace")
            ggufs = {}
            for row in entries.values():
                source = str(Path(row["source"]).absolute())
                if source not in ggufs:
                    ggufs[source] = {
                        "sha256": sha_file(source),
                        "bytes": os.stat(source).st_size,
                    }
            actual_sources = {}
            for path, expected in reconstruction_sources.items():
                actual = sha_file(path)
                if actual != expected:
                    raise ValueError("Canonical reconstruction source hash changed")
                actual_sources[str(path)] = actual
            material = {
                "schema": SCHEMA,
                "canonical_dtype": "bfloat16",
                "cache_implementation_sha256": sha_file(Path(__file__).resolve()),
                "backend": backend,
                "reconstruction_sha256": sorted(actual_sources.values()),
                "GGUF": sorted(ggufs.values(), key=lambda row: row["sha256"]),
                "entries": {
                    key: {
                        "GGUF_sha256": ggufs[str(Path(row["source"]).absolute())][
                            "sha256"
                        ],
                        "logical_name": row["logical_name"],
                        "shape": row["shape"],
                        "bytes": row["bytes"],
                        "descriptor_sha256": row["descriptor_sha256"],
                    }
                    for key, row in entries.items()
                },
            }
            self.key = json_sha(material)
            self.material = material
            self.source_paths = {"GGUF": ggufs, "reconstruction": actual_sources}
            self.slot = self.directory / "slot"
            for path in self.directory.glob("incoming-*"):
                if path.is_symlink() or not path.is_dir():
                    raise ValueError("Unexpected RAM cache staging object")
                shutil.rmtree(path)
            if self.slot.exists() or self.slot.is_symlink():
                try:
                    self._verify_complete()
                except (
                    ValueError,
                    OSError,
                    KeyError,
                    TypeError,
                    json.JSONDecodeError,
                ) as error:
                    self.reason = "rebuild: " + str(error)
                    if self.slot.is_symlink() or not self.slot.is_dir():
                        raise ValueError(
                            "Refusing to remove an unexpected cache slot object"
                        ) from error
                    shutil.rmtree(self.slot)
                else:
                    self.hit = True
                    self.reason = "verified-complete-checkpoint"
            if not self.hit:
                if (
                    shutil.disk_usage(self.directory).free
                    < self.required_bytes + self.metadata_reserve
                ):
                    raise ValueError(
                        "Insufficient tmpfs capacity for complete canonical checkpoint"
                    )
                self.incoming = self.directory / ("incoming-" + uuid.uuid4().hex)
                self.incoming.mkdir()
        except BaseException:
            self.close()
            raise

    def _verify_complete(self):
        if self.slot.is_symlink():
            raise ValueError("RAM cache slot must not be a symlink")
        manifest_path = self.slot / "manifest.json"
        if manifest_path.is_symlink():
            raise ValueError("CPU cache manifest must not be a symlink")
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("schema") != SCHEMA
            or manifest.get("key") != self.key
            or manifest.get("material") != self.material
            or manifest.get("complete") is not True
            or manifest.get("raw_bytes") != self.required_bytes
            or set(manifest["records"]) != set(self.entries)
        ):
            raise ValueError("Complete RAM cache belongs to another checkpoint/source")
        records = manifest["records"]
        expected_names = {"manifest.json"}
        allocated = (self.slot / "manifest.json").stat().st_size
        for key, row in records.items():
            filename = hashlib.sha256(key.encode()).hexdigest() + ".bf16"
            if row["file"] != filename:
                raise ValueError("RAM cache filename differs from logical key")
            path = self.slot / filename
            identity = row["identity"]
            if (
                path.is_symlink()
                or not path.is_file()
                or identity["bytes"] != self.entries[key]["bytes"]
                or identity["shape"] != self.entries[key]["shape"]
                or identity["dtype"] != "torch.bfloat16"
                or identity["descriptor_sha256"]
                != self.entries[key]["descriptor_sha256"]
                or path.stat().st_size != identity["bytes"]
                or sha_file(path) != identity["sha256"]
            ):
                raise ValueError(
                    "Canonical BF16 cache integrity check failed for " + key
                )
            expected_names.add(filename)
            allocated += path.stat().st_size
        if {
            path.name for path in self.slot.iterdir()
        } != expected_names or allocated > self.budget:
            raise ValueError("Unexpected data or budget overflow in complete RAM cache")
        self.records = records

    def lookup(self, key, original_components):
        if self.closed or key in self.used or key not in self.entries:
            raise ValueError("Invalid or repeated canonical CPU cache request")
        self.used.add(key)
        if not self.hit:
            return None
        row = self.records[key]
        if row["identity"]["original_components"] != original_components:
            raise ValueError("Original components differ from complete RAM checkpoint")
        self.hits += 1
        return self.slot / row["file"]

    def record(self, key, raw, identity):
        if self.closed or self.hit or key not in self.used or key in self.records:
            raise ValueError("Invalid canonical CPU cache publication")
        view = memoryview(raw).cast("B")
        if (
            len(view) != self.entries[key]["bytes"]
            or hashlib.sha256(view).hexdigest() != identity["sha256"]
        ):
            raise ValueError("Canonical CPU record differs from its actual BF16 bytes")
        if (
            identity["shape"] != self.entries[key]["shape"]
            or identity["dtype"] != "torch.bfloat16"
            or identity["descriptor_sha256"] != self.entries[key]["descriptor_sha256"]
        ):
            raise ValueError("Canonical CPU record descriptor/shape differs")
        filename = hashlib.sha256(key.encode()).hexdigest() + ".bf16"
        path = self.incoming / filename
        with path.open("xb") as stream:
            stream.write(view)
            stream.flush()
            os.fsync(stream.fileno())
        if sha_file(path) != identity["sha256"]:
            raise ValueError("RAM publication byte verification failed")
        path.chmod(0o444)
        self.records[key] = {"file": filename, "identity": identity}
        self.reconstructed += 1

    def finish(self):
        if (
            self.closed
            or self.used != self.entries.keys()
            or self.records.keys() != self.entries.keys()
        ):
            raise ValueError("Incomplete canonical CPU checkpoint cannot be published")
        if not self.hit:
            manifest = {
                "schema": SCHEMA,
                "key": self.key,
                "material": self.material,
                "source_paths": self.source_paths,
                "records": self.records,
                "raw_bytes": self.required_bytes,
                "complete": True,
                "created_unix": time.time(),
            }
            write_json(self.incoming / "manifest.json", manifest)
            if (self.incoming / "manifest.json").stat().st_size > self.metadata_reserve:
                raise ValueError("Canonical metadata exceeds its reserved CPU budget")
            os.rename(self.incoming, self.slot)
            self.incoming = None
        self.complete = True
        self.close()

    def close(self):
        if self.lease is not None:
            fcntl.flock(self.lease.fileno(), fcntl.LOCK_UN)
            self.lease.close()
            self.lease = None
        self.closed = True

    def abort(self):
        if self.incoming is not None and self.incoming.is_dir():
            shutil.rmtree(self.incoming)
            self.incoming = None
        self.close()

    def summary(self):
        return {
            "mode": "ram-bf16-v1",
            "directory": str(self.directory),
            "budget_bytes": self.budget,
            "canonical_bytes": self.required_bytes,
            "checkpoint_key": self.key,
            "complete_hit": self.hit,
            "reason": self.reason,
            "hits": self.hits,
            "reconstructed_matrices": self.reconstructed,
            "open_CPU_views": self.views_open,
            "complete": self.complete,
            "persistent_extra_GPU_bytes": 0,
            "single_checkpoint_only": True,
            "no_partial_hits": True,
        }

    def __del__(self):
        if getattr(self, "lease", None) is not None:
            self.close()


@contextmanager
def canonical_weight(plan, key, descriptor, host, reconstruct, tensor_identity, torch):
    """Yield exact old reconstruction or a verified complete-store CPU view."""
    policy = settings()
    if policy["mode"] == "none":
        yield reconstruct(descriptor, host, torch.bfloat16).contiguous()
        return
    session = getattr(plan, "_canonical_cpu_session", None)
    if session is None:
        source = Path(reconstruct.__code__.co_filename).resolve()
        session = CacheSession(
            plan.entries,
            {str(source): sha_file(source)},
            policy["directory"],
            policy["budget_bytes"],
            {
                "torch": str(torch.__version__),
                "git": str(torch.version.git_version),
                "CPU_threads": int(torch.get_num_threads()),
                "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
            },
        )
        plan._canonical_cpu_session = session
    mapped = None
    weight = None
    failed = False
    try:
        original_components = (
            {name: tensor_identity(value) for name, value in host.items()}
            if session.hit
            else None
        )
        path = session.lookup(key, original_components)
        if path is not None:
            with path.open("rb") as stream:
                mapped = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY)
            session.views_open += 1
            weight = torch.frombuffer(mapped, dtype=torch.bfloat16).reshape(
                descriptor["shape"]
            )
        else:
            weight = reconstruct(descriptor, host, torch.bfloat16).contiguous()
        yield weight
        if not session.hit:
            raw = memoryview(weight.reshape(-1).view(torch.uint8).numpy())
            session.record(key, raw, plan.loaded[key])
            del raw
    except BaseException:
        failed = True
        raise
    finally:
        weight = None
        try:
            if mapped is not None:
                mapped.close()
                session.views_open -= 1
        finally:
            if failed:
                session.abort()
    if plan.loaded.keys() == plan.entries.keys():
        try:
            session.finish()
        except BaseException:
            session.abort()
            raise


def runtime_marker():
    value = settings()
    if value["mode"] != "none":
        path = Path(__file__).resolve()
        value.update(
            source={"path": str(path), "sha256": sha_file(path)},
            canonical_dtype="torch.bfloat16",
            single_checkpoint_only=True,
            extra_GPU_bytes=0,
            complete_checkpoint_only=True,
        )
    return value
