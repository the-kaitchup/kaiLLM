"""Hy4 GGUF adapter name map on toy GGUFs: total, one-to-one, Engram table left to the host."""
import sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime-source"))
import importlib.util
spec = importlib.util.spec_from_file_location("hy4cfg", ROOT / "runtime-source/vllm-overrides/vllm/transformers_utils/configs/hy_v4_flash.py")
cfgmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfgmod)
import gguf, json
from vllm_gguf_plugin.hy4_metadata import prepare_hy4_metadata
from vllm_gguf_plugin.weights_adapter.hy4 import hy4_name_map, TABLE, LINEAR_SUFFIXES
from vllm_gguf_plugin.weight_utils import SCALAR_GGUF_TYPES

for path in sys.argv[1:]:
    with tempfile.TemporaryDirectory() as td:
        d = prepare_hy4_metadata(Path(path), cache_dir=td)
        cfg = cfgmod.HYV4FlashConfig(**json.loads((d / "config.json").read_text()))
    r = gguf.GGUFReader(path)
    types = {t.name: t.tensor_type.name for t in r.tensors}
    m = hy4_name_map(cfg, set(types))
    unmapped = sorted(set(types) - set(m) - {TABLE})
    dup = len(m) - len(set(m.values()))
    unq = sorted({v.rsplit(".", 1)[0] for g, v in m.items() if v.endswith(".weight")
                  and v.rsplit(".", 2)[-2] in LINEAR_SUFFIXES and types[g] in SCALAR_GGUF_TYPES})
    q = sorted({types[g] for g, v in m.items() if types[g] not in SCALAR_GGUF_TYPES})
    print(f"{Path(path).name}: {len(types)} tensors, mapped {len(m)}, unmapped {unmapped}, duplicate targets {dup}, "
          f"quantized types {q}, unquantized linears {len(unq)}")
