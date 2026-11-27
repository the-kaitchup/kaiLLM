"""Apply strict, opt-in CUDA graph accounting to a clean, pinned llama.cpp worktree."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

PINS = {
    "b5ed9f2845358124d1cbce91f475011047515e74",
    "f4f6f2e7db76e98c95cc9b7fc608ea669c806044",
    "e4b1cb5318f1ac55d148be228c6347c6a94cc4c6",
    "0a054d0b02454c998e068381d15f3f6615bdd987",
}
HEADER = Path(__file__).with_name("llama_graph.hpp")
CUDA = "ggml/src/ggml-cuda/ggml-cuda.cu"
SERVER = "tools/server/server-context.cpp"
INCLUDE = "ggml/include/kaillm-graph.hpp"
CUDA_MARKER = "ggml_kaillm_graph_proof_cuda_v2"
SERVER_MARKER = "llama_kaillm_graph_proof_server_v2"


def replace_once(code, target, replacement):
    if code.count(target) != 1:
        raise ValueError("Pinned source does not match anchor: " + target[:100])
    return code.replace(target, replacement)


def marker(name):
    signature = f'extern "C" __attribute__((visibility("default"), used)) int {name}()'
    return f"\n{signature};\n{signature} {{ return 1; }}\n"


def instrument_cuda(code):
    anchor = '#include "ggml-cuda/common.cuh"'
    code = replace_once(
        code,
        anchor,
        anchor
        + '\n#include "kaillm-graph.hpp"\n#ifdef USE_CUDA_GRAPH'
        + marker(CUDA_MARKER)
        + "#endif\n",
    )
    target = "            if (!graph->warmup_complete) {"
    code = replace_once(
        code,
        target,
        """            if (kaillm_graph::data() != nullptr) {
                // Never execute the first or a changed model shape directly.
                use_cuda_graph = true;
                cuda_graph_update_required = properties_changed || graph->instances.empty();
                graph->warmup_complete = true;
            } else if (!graph->warmup_complete) {""",
    )
    target = "    if (use_cuda_graph && cuda_graph_update_required) {\n        // Start CUDA graph capture"
    code = replace_once(
        code,
        target,
        """    if (kaillm_graph::data() != nullptr) {
        if (!use_cuda_graph) {
            kaillm_graph::record(cuda_ctx->device, kaillm_graph::rejected);
            GGML_LOG_ERROR("%s: required CUDA graph unavailable; model batch rejected before evaluation\\n", __func__);
            return GGML_STATUS_FAILED;
        }
        kaillm_graph::record(cuda_ctx->device, kaillm_graph::batches);
    }

"""
        + target,
    )
    # Both upstream launch paths are checked only after the CUDA API succeeds.
    end = "    CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &cuda_graph));"
    code = replace_once(
        code,
        end,
        end + "\n    kaillm_graph::record(cuda_ctx->device, kaillm_graph::captures);",
    )
    update = "static void ggml_cuda_graph_update_executable("
    code = replace_once(code, update, update.replace("void", "bool"))
    code = replace_once(
        code,
        "        instance.clusters = std::move(clusters);\n        return;",
        "        instance.clusters = std::move(clusters);\n        return false;",
    )
    anchor = "        GGML_ASSERT(stat == cudaSuccess);\n    }\n}\n\n// number of ggml nodes per CUDA graph chunk"
    code = replace_once(
        code,
        anchor,
        anchor.replace("    }\n}", "    }\n    return stat == cudaSuccess;\n}"),
    )
    anchor = "    if (idx < graph->instances.size()) {\n        ggml_cuda_graph_update_executable("
    code = replace_once(
        code,
        anchor,
        "    bool updated = false;\n"
        + anchor.replace("        ggml_", "        updated = ggml_"),
    )
    fresh = (
        "    CUDA_CHECK(cudaGraphLaunch(graph->instances[idx], cuda_ctx->stream()));"
    )
    code = replace_once(
        code,
        fresh,
        fresh
        + """
    kaillm_graph::record(cuda_ctx->device, kaillm_graph::capture_launches);
    if (updated) kaillm_graph::record(cuda_ctx->device, kaillm_graph::updates);""",
    )
    reuse = "            CUDA_CHECK(cudaGraphLaunch(instance, cuda_ctx->stream()));"
    return replace_once(
        code,
        reuse,
        reuse
        + "\n            kaillm_graph::record(cuda_ctx->device, kaillm_graph::replays);",
    )


def instrument_server(code):
    code = replace_once(
        code,
        '#include "server-context.h"',
        '#include "server-context.h"\n#include "kaillm-graph.hpp"'
        + marker(SERVER_MARKER),
    )
    code = replace_once(
        code,
        "                common_speculative_draft(spec.get());",
        "                kaillm_graph::scope draft_phase(kaillm_graph::proposal);\n                common_speculative_draft(spec.get());",
    )
    anchor = """        bool has_output = false;
        for (int i = off; i < off + batch.view.size(); ++i) {
            has_output |= batch.tokens[i].output;
        }"""
    code = replace_once(
        code,
        anchor,
        """        bool has_output = false;
        bool has_prompt = false;
        bool has_decode = false;
        for (int i = off; i < off + batch.view.size(); ++i) {
            has_output |= batch.tokens[i].output;
            has_prompt |= batch.tokens[i].is_prompt;
            has_decode |= !batch.tokens[i].is_prompt;
        }
        const auto token_phase = has_prompt && has_decode ? kaillm_graph::mixed :
                                 has_prompt ? kaillm_graph::prefill : kaillm_graph::decode;
        const auto state_phase = has_prompt && has_decode ? kaillm_graph::draft_mixed :
                                 has_prompt ? kaillm_graph::draft_prefill : kaillm_graph::draft_decode;""",
    )
    target = "            ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());"
    code = replace_once(
        code, target, "            kaillm_graph::scope target(token_phase);\n" + target
    )
    synchronize = (
        "                ok = common_speculative_process(spec.get(), batch.view);"
    )
    return replace_once(
        code,
        synchronize,
        "                kaillm_graph::scope state(state_phase);\n" + synchronize,
    )


def source_files(tree):
    revision = subprocess.check_output(
        ["git", "-C", str(tree), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision not in PINS:
        raise ValueError("Unsupported llama.cpp revision")
    previous = json.loads(HEADER.with_name("llama_graph_sources_v1.json").read_bytes()).get(revision, {})
    result = {INCLUDE: HEADER.read_bytes()}
    for relative, patch in ((CUDA, instrument_cuda), (SERVER, instrument_server)):
        original = subprocess.check_output(
            ["git", "-C", str(tree), "show", revision + ":" + relative]
        )
        result[relative] = patch(original.decode()).encode()
        current = (tree / relative).read_bytes()
        if (
            current not in (original, result[relative])
            and hashlib.sha256(current).hexdigest() != previous.get(relative)
        ):
            raise ValueError("Unexpected local edits to " + relative)
    return revision, result


def apply(tree):
    rev, data = source_files(tree)
    for relative, content in data.items():
        out = tree / relative
        if relative == INCLUDE and out.exists() and out.read_bytes() != content:
            old = json.loads(
                HEADER.with_name("llama_graph_sources_v1.json").read_bytes()
            )
            if hashlib.sha256(out.read_bytes()).hexdigest() != old.get(rev, {}).get(INCLUDE):
                raise ValueError("Conflicting graph header")
    for relative, content in data.items():
        out = tree / relative
        if not out.exists() or out.read_bytes() != content:
            out.write_bytes(content)
    return rev


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def symbols(path):
    result = subprocess.check_output(
        ["nm", "-D", "--defined-only", str(path)], text=True
    )
    return {line.split()[-1] for line in result.splitlines() if line.strip()}


def certify(tree, binary_dir):
    rev, expected = source_files(tree)
    changes = (
        subprocess.check_output(
            ["git", "-C", str(tree), "diff", "--name-only", "-z", "HEAD"]
        )
        .decode()
        .split("\0")
    )
    if set(filter(None, changes)) - set(expected):
        raise ValueError(
            "Unexpected tracked source changes in llama.cpp reference build"
        )
    if any((tree / k).read_bytes() != v for k, v in expected.items()):
        raise ValueError("Instrumentation not applied")
    cuda = (binary_dir / "libggml-cuda.so").resolve(strict=True)
    impl = (binary_dir / "libllama-server-impl.so").resolve(strict=True)
    if CUDA_MARKER not in symbols(cuda) or SERVER_MARKER not in symbols(impl):
        raise ValueError("Missing strict graph binary markers")
    instrumented = {k: hashlib.sha256(v).hexdigest() for k, v in expected.items()}
    certificate = {
        "protocol": 2,
        "upstream": rev,
        "source_sha256": instrumented,
        "binary_sha256": {
            name: digest(binary_dir / name)
            for name in ("llama-server", "libggml-cuda.so", "libllama-server-impl.so")
        },
    }
    result = binary_dir / "kaillm-graph-proof.json"
    tmp = result.with_suffix(".pending")
    tmp.write_text(json.dumps(certificate, sort_keys=True, indent=2) + "\n")
    tmp.replace(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--certify", type=Path, metavar="BIN_DIRECTORY")
    args = parser.parse_args()
    print(certify(args.source, args.certify) if args.certify else apply(args.source))
