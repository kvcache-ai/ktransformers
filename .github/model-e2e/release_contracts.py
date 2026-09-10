"""Read-only contract for a same-run, five-package release and its dependency closure."""

import json
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

from contracts import PACKAGES, REPOSITORIES, digest, require, sha


def verify_release(root, expected_digest):
    root = Path(root)
    manifest_path = root / "release.json"
    require(not manifest_path.is_symlink(), "Manifest must not be a symlink")
    require(digest(manifest_path) == expected_digest, "Release manifest hash changed")
    manifest = json.loads(manifest_path.read_text())
    require(manifest["schema"] == 1, "Unknown release schema")
    require(type(manifest["run_id"]) is int and manifest["run_id"] > 0, "Invalid run")
    require(
        type(manifest["run_attempt"]) is int and manifest["run_attempt"] > 0,
        "Invalid attempt",
    )
    sha(manifest["workflow_sha"])
    if "assembly_workflow_sha" in manifest:
        sha(manifest["assembly_workflow_sha"])
        recovery = manifest["source_lock"].get("recovery")
        if recovery:
            require(recovery["assembly_workflow_sha"] == manifest["assembly_workflow_sha"], "Recovery assembler mismatch")
            require(recovery["workflow_sha"] == manifest["workflow_sha"], "Recovery source mismatch")
        else:
            require(manifest["assembly_workflow_sha"] == manifest["workflow_sha"], "Different assembler without recovery provenance")
    lock = manifest["source_lock"]
    require(lock["workflow_sha"] == manifest["workflow_sha"], "Workflow lock mismatch")
    require(set(lock["sources"]) == set(REPOSITORIES), "Need four main sources")
    for key, source in lock["sources"].items():
        require(source["repository"] == REPOSITORIES[key], "Wrong source repository")
        require(source["ref"] == "refs/heads/main", "Release sources must be main")
        sha(source["sha"])
    require(
        lock["sources"]["ktransformers"]["sha"] == manifest["workflow_sha"],
        "KT source must match the trusted release workflow",
    )
    require(set(manifest["wheels"]) == PACKAGES, "Need five final wheels")
    directory = root / "wheelhouse"
    require(directory.is_dir() and not directory.is_symlink(), "Unsafe wheelhouse")
    require(
        {p.name for p in root.iterdir()} == {"release.json", "wheelhouse"},
        "Unexpected release files",
    )
    require(
        {p.name for p in directory.iterdir()} == set(manifest["wheelhouse"]),
        "Wheelhouse file set changed",
    )
    names = set()
    for filename, entry in manifest["wheelhouse"].items():
        require(
            re.fullmatch(r"[A-Za-z0-9_.+-]+\.whl", filename), "Unsafe wheel filename"
        )
        path = directory / filename
        require(path.is_file() and not path.is_symlink(), "Unsafe wheel")
        require(digest(path) == entry["sha256"], "Wheel SHA256 changed: " + filename)
        require(entry["name"] not in names, "Duplicate distribution in wheelhouse")
        names.add(entry["name"])
        require(
            entry["name"]
            not in {
                "sglang",
                "sgl-kernel",
                "sgl-kernel-kt",
                "transformers",
                "accelerate",
            },
            "Namespace collision",
        )
    for package, entry in manifest["wheels"].items():
        selected = manifest["wheelhouse"][entry["file"]]
        require(selected["name"] == package, "Wrong carrier name")
        require(
            all(selected[key] == entry[key] for key in ("version", "sha256")),
            "Carrier differs from wheelhouse",
        )
    require(
        set(manifest["plans"]) == {"sglang", "sglang,sft"},
        "Require both user installation plans",
    )
    for plan in manifest["plans"].values():
        for name, filename in plan.items():
            require(
                manifest["wheelhouse"][filename]["name"] == name,
                "Invalid resolution plan",
            )
    require(
        PACKAGES <= set(manifest["plans"]["sglang,sft"]),
        "Combined extra must install all five packages",
    )
    return manifest


def verify_install_report(report, manifest, extra, *, public):
    entries = report["install"]
    names = [item["metadata"]["name"].lower().replace("_", "-") for item in entries]
    require(len(set(names)) == len(names), "Duplicate installed distribution")
    plan = manifest["plans"][extra]
    require(
        set(names) == set(plan), "Dependency resolution differs from tested wheelhouse"
    )
    for name, item in zip(names, entries):
        expected = manifest["wheelhouse"][plan[name]]
        require(
            item["metadata"]["version"] == expected["version"],
            "Resolved version changed: " + name,
        )
        download = item["download_info"]
        require(
            download["archive_info"]["hashes"]["sha256"] == expected["sha256"],
            "Installed wheel hash changed: " + name,
        )
        url = urlparse(download["url"])
        require(
            Path(unquote(url.path)).name == plan[name],
            "Installed wheel filename changed",
        )
        if public:
            require(
                url.scheme == "https" and url.hostname == "files.pythonhosted.org",
                "Not downloaded from public PyPI",
            )
        else:
            require(url.scheme == "file", "Candidate installation must be offline")
