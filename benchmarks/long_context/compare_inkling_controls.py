"""Compare an optimized capture against two independent starts of the same original runtime."""
import argparse
import hashlib
import json
from pathlib import Path

from check_inkling_optimized_pair import compare_domains


def compare_controls(first, repeat, candidate):
    if first["role"] != "baseline" or repeat["role"] != "baseline" or candidate["role"] != "candidate":
        raise ValueError("Expected two original control runs and one candidate")
    if any(first[key] != repeat[key] for key in ("source_revision", "source_lock", "observed_environment")):
        raise ValueError("The original controls did not use the same build and environment")
    controls = compare_domains(first, repeat)
    comparisons = {"first":compare_domains(first, candidate), "repeat":compare_domains(repeat, candidate)}
    return {"reference_repeat":controls, "candidate_comparisons":comparisons,
            "exact_matching_controls":[key for key, value in comparisons.items() if value["passed"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("first", "repeat", "candidate", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.first.resolve() == args.repeat.resolve():
        parser.error("The two original runs need independent capture files")
    raw = {name:getattr(args,name).read_bytes() for name in ("first", "repeat", "candidate")}
    result = compare_controls(*(json.loads(raw[name]) for name in ("first", "repeat", "candidate")))
    result["capture_sha256"] = {name:hashlib.sha256(b).hexdigest() for name,b in raw.items()}
    args.out.write_text(json.dumps(result,indent=2) + "\n")


if __name__ == "__main__":
    main()
