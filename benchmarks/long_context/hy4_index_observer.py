"""Hash selected Hy4 attention indices inside the single-GPU qualification process."""
import hashlib
import json
import os
from pathlib import Path


def observe(layer, positions, indices):
    marker = Path(os.environ["KAILLM_HY4_OBSERVATION"])
    try:
        request = json.loads(marker.read_bytes())
    except FileNotFoundError:
        return
    wanted = set(request["positions"])
    samples = []
    for offset, position in enumerate(positions.detach().cpu().tolist()):
        if position in wanted:
            row = indices[offset].detach().cpu().contiguous().numpy().astype("<i4", copy=False)
            samples.append({"tag": request["tag"], "layer": layer, "position": position, "width": len(row),
                            "sha256": hashlib.sha256(row.tobytes()).hexdigest()})
    if samples:
        with Path(os.environ["KAILLM_HY4_INDICES"]).open("a") as stream:
            for sample in samples:
                stream.write(json.dumps(sample) + "\n")
