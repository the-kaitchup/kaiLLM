# SPDX-License-Identifier: Apache-2.0
"""Model-family serving defaults read from the GGUF header (no tensor data is read).

The launcher applies these only to settings the user did not pass explicitly:

* LFM3 GGUFs, and references that are not local files: the qualified LFM3
  tokenizer and the catalog profile, unchanged.
* Qwen3.5-architecture GGUFs (``qwen35``: Qwen3.5, Qwen3.8): the tokenizer, HF
  config and generation settings come from the GGUF itself (the plugin's
  path-only metadata, token-identical to the published tokenizer); the
  LFM3-sized memory plans (Q6 f16 representation, shared BF16 prefill cache)
  are disabled, as in the qualified Qwen3.8 27B runs; when the GGUF carries
  MTP (nextn) layers, MTP speculative decoding is the default, otherwise
  speculation is off unless requested (--spec-tokens selects n-gram drafts).
"""

import re
import struct
from pathlib import Path

LFM3_TOKENIZER = "LiquidAI/LFM3-1.7B"
QWEN35_ENVIRONMENT = {
    "VLLM_GGUF_Q6_PRECISION": "none",
    "VLLM_GGUF_BODY_PREFILL_CACHE": "none",
}

_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
            10: "Q", 11: "q", 12: "d"}
_STRING, _ARRAY = 8, 9
_SPLIT = re.compile(r"^(?P<stem>.*)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$", re.IGNORECASE)


class _Reader:
    def __init__(self, stream):
        self.stream = stream

    def unpack(self, fmt):
        size = struct.calcsize("<" + fmt)
        data = self.stream.read(size)
        if len(data) != size:
            raise ValueError("Truncated GGUF header")
        return struct.unpack("<" + fmt, data)[0]

    def string(self):
        return self.stream.read(self.unpack("Q")).decode("utf-8", "replace")

    def skip_string(self):
        self.stream.seek(self.unpack("Q"), 1)

    def value(self, kind):
        if kind == _STRING:
            return self.string()
        if kind == _ARRAY:
            item, count = self.unpack("I"), self.unpack("Q")
            if item == _STRING:
                for _ in range(count):
                    self.skip_string()
            elif item in _SCALARS:
                self.stream.seek(count * struct.calcsize("<" + _SCALARS[item]), 1)
            else:
                raise ValueError("Unsupported GGUF array type")
            return None
        if kind not in _SCALARS:
            raise ValueError("Unsupported GGUF value type")
        return self.unpack(_SCALARS[kind])


def read_header(path, tensor_names=True):
    """Return ({key: scalar or string}, [tensor names]) from a GGUF v2/v3 header."""
    with open(path, "rb", buffering=1 << 20) as stream:
        reader = _Reader(stream)
        if stream.read(4) != b"GGUF" or reader.unpack("I") not in (2, 3):
            raise ValueError("Not a GGUF v2/v3 file")
        tensors, count = reader.unpack("Q"), reader.unpack("Q")
        metadata = {}
        for _ in range(count):
            key = reader.string()
            value = reader.value(reader.unpack("I"))
            if value is not None:
                metadata[key] = value
        names = []
        if tensor_names:
            for _ in range(tensors):
                names.append(reader.string())
                dims = reader.unpack("I")
                stream.seek(8 * dims + 4 + 8, 1)
        return metadata, names


def _shards(path):
    match = _SPLIT.match(path.name)
    if match is None:
        return [path]
    count = int(match["count"])
    return [path.with_name(f"{match['stem']}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1)]


def inspect(model):
    """Serving defaults for a GGUF reference; empty defaults when it is not a local file."""
    result = {"architecture": None, "mtp_layers": 0, "speculation": "ngram",
              "tokenizer": LFM3_TOKENIZER, "environment": {}}
    path = Path(str(model)).expanduser()
    if not path.is_file():
        return result
    try:
        metadata, names = read_header(path)
        for shard in _shards(path)[1:]:
            if shard.is_file():
                names += read_header(shard)[1]
    except (OSError, ValueError, struct.error):
        return result
    architecture = metadata.get("general.architecture")
    result["architecture"] = architecture
    if architecture == "qwen35":
        layers = int(metadata.get("qwen35.nextn_predict_layers", 0) or 0)
        result["mtp_layers"] = layers if any(".nextn." in name for name in names) else 0
        result["speculation"] = "mtp" if result["mtp_layers"] else "none"
        result["tokenizer"] = None
        result["environment"] = dict(QWEN35_ENVIRONMENT)
    return result
