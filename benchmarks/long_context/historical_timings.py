"""Identify the exact full-length measurements accepted before per-wave instrumentation."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

# Bind both speeds and their workload/provenance without duplicating full receipts.
FIELDS = ("hardware_id", "gpu", "driver", "power_limit_w", "engine", "engine_revision",
          "kai_source", "llama_source", "model_revision", "checkpoint", "mode", "batch", "domain",
          "protocol", "prompt_set_sha256", "input_tokens_per_request", "output_tokens_per_request",
          "requested_context_per_slot", "prefill_seconds", "full_seconds", "generated_text_hashes", "timestamp_utc")


def receipt_hash(row):
    payload = json.dumps([row.get(key) for key in FIELDS], separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


@lru_cache(maxsize=1)
def approved_receipts():
    index = Path(__file__).with_name("historical_timings.json")
    return frozenset(json.loads(index.read_bytes())["receipt_sha256"])


def previously_accepted(row):
    return "timestamp_utc" in row and receipt_hash(row) in approved_receipts()
