"""Small, GPU-free contracts shared by the model acceptance entry points."""

from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from email.parser import BytesParser
from pathlib import Path

REPOSITORIES = {
    "ktransformers": "kvcache-ai/ktransformers",
    "sglang": "kvcache-ai/sglang",
    "transformers": "kvcache-ai/transformers",
    "accelerate": "kvcache-ai/accelerate",
}
PACKAGES = {
    "ktransformers",
    "kt-kernel",
    "sglang-kt",
    "transformers-kt",
    "accelerate-kt",
}
CASES = ("qwen3_lora", "deepseek_v31_lora", "glm53_inference")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(value):
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value),
        "Expected a full commit SHA",
    )
    return value


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def validate_request(request):
    require(request["schema"] == 1, "Unsupported request schema")
    require(request["mode"] in ("pypi", "pr"), "Unknown mode")
    sha(request["harness_sha"])
    if request["mode"] == "pypi":
        version = request["version"]
        require(
            isinstance(request.get("expected_version"), str)
            and re.fullmatch(r"[0-9][0-9A-Za-z.!+_-]*", request["expected_version"]),
            "Freeze the expected PyPI version before installation",
        )
        require(
            version == "latest" or re.fullmatch(r"[0-9][0-9A-Za-z.!+_-]*", version),
            "Invalid PyPI version",
        )
        require("pr" not in request, "PyPI mode cannot report a PR result")
    else:
        pr = request["pr"]
        require(
            pr["repository"] in REPOSITORIES.values(), "PR repository is not allowed"
        )
        require(type(pr["number"]) is int and pr["number"] > 0, "Invalid PR number")
        sha(pr["sha"])
        require(
            set(request["sources"]) == set(REPOSITORIES),
            "Require all four source snapshots",
        )
        for key, source in request["sources"].items():
            require(
                source["repository"] == REPOSITORIES[key],
                "Unexpected source repository",
            )
            sha(source["sha"])
        key = next(
            key for key, repo in REPOSITORIES.items() if repo == pr["repository"]
        )
        require(
            request["sources"][key]["sha"] == pr["sha"],
            "PR head does not match source lock",
        )
    return request


def verify_candidate(root, request):
    """Validate a FINAL five-wheel stack; raw/native intermediates are not installable candidates."""
    root = Path(root).resolve()
    manifest = json.loads((root / "candidate.json").read_text())
    require(manifest["schema"] == 1, "Unsupported candidate schema")
    require(
        manifest["sources"] == request["sources"],
        "Candidate sources differ from the approved snapshot",
    )
    require(manifest["build_run_id"] == request["build_run_id"], "Wrong build run")
    require(
        manifest["build_workflow_sha"] == request["build_workflow_sha"],
        "Wrong build workflow SHA",
    )
    require(
        set(manifest["wheels"]) == PACKAGES, "Require exactly five final package wheels"
    )
    paths = set()
    for package, entry in manifest["wheels"].items():
        name = entry["file"]
        require(
            isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_.+-]+\.whl", name),
            "Unsafe wheel filename",
        )
        path = root / name
        require(
            not path.is_symlink() and path.is_file(), "Wheel must be a regular file"
        )
        require(digest(path) == entry["sha256"], f"Wheel digest mismatch: {package}")
        with zipfile.ZipFile(path) as wheel:
            metadata = [
                name
                for name in wheel.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            require(len(metadata) == 1, "Expected one wheel METADATA")
            meta = BytesParser().parsebytes(wheel.read(metadata[0]))
            require(
                meta["Name"].lower().replace("_", "-") == package,
                "Wrong wheel distribution",
            )
            require(meta["Version"] == entry["version"], "Wrong wheel version")
        paths.add(name)
    require(len(paths) == len(PACKAGES), "Duplicate wheel files")
    require(
        {path.name for path in root.iterdir()} == paths | {"candidate.json"},
        "Unexpected artifact files",
    )
    return manifest


def validate_lora(records, world_size):
    require(type(world_size) is int and world_size > 0, "Invalid world size")
    require(len(records) == world_size, "Missing rank evidence")
    require(
        {record["rank"] for record in records} == set(range(world_size)),
        "Duplicate/missing ranks",
    )
    for record in records:
        require(record["global_step"] == 1, "Exactly one optimizer step is required")
        require(
            record["optimizer_steps"] == 1, "Optimizer step callback was not observed"
        )
        require(record["train_end"] is True, "Training did not finish")
        losses = record["raw_losses"]
        require(bool(losses), "No raw loss was observed")
        require(
            all(type(loss) in (float, int) and math.isfinite(loss) for loss in losses),
            "Non-finite raw loss",
        )
    return {"status": "passed", "optimizer_steps": 1, "ranks": records}


def validate_answer(response):
    require(
        response.get("usage", {}).get("completion_tokens", 0) > 0, "No decode tokens"
    )
    choice = response["choices"][0]
    text = choice["message"].get("content")
    require(
        isinstance(text, str) and text.strip(),
        "Empty final answer (reasoning alone is not a pass)",
    )
    require(
        "\ufffd" not in text
        and not any(ord(c) < 32 and c not in "\n\r\t" for c in text),
        "Garbled answer",
    )
    require(choice.get("finish_reason") == "stop", "Generation did not finish normally")
    # The fixed prompt asks for the capital of France; check final content, not
    # the prompt or reasoning. This is a smoke test, not a quality benchmark.
    require(
        "paris" in text.lower() or "巴黎" in text,
        "Answer does not match the smoke-test question",
    )
    return {"status": "passed", "answer": text, "usage": response["usage"]}


def suite_passed(results):
    return set(results) == set(CASES) and all(
        result.get("status") == "passed" for result in results.values()
    )
