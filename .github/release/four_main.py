#!/usr/bin/env python3
"""Freeze four public main heads and audit source-built release artifacts.

This module never uploads packages or changes package versions/dependencies.
Raw wheels remain non-publishable; release_stack.py gates final promotion.
"""

from __future__ import annotations

import argparse
import email
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename

REPOSITORIES = {
    "ktransformers": "kvcache-ai/ktransformers",
    "sglang": "kvcache-ai/sglang",
    "transformers": "kvcache-ai/transformers",
    "accelerate": "kvcache-ai/accelerate",
}
PACKAGE_SOURCES = {
    "ktransformers": "ktransformers",
    "kt-kernel": "ktransformers",
    "sglang-kt": "sglang",
    "sgl-kernel-kt": "sglang",
    "transformers-kt": "transformers",
    "accelerate-kt": "accelerate",
}
EXACT_DEPENDENCIES = {
    "ktransformers": ("kt-kernel", "sglang-kt", "transformers-kt", "accelerate-kt"),
    "sglang-kt": ("kt-kernel", "transformers-kt"),
}
FULL_SHA = re.compile(r"[0-9a-f]{40}\Z")


def run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def resolve_main(repository: str) -> str:
    output = run(
        "git",
        "ls-remote",
        "--exit-code",
        f"https://github.com/{repository}.git",
        "refs/heads/main",
    )
    rows = [row.split() for row in output.splitlines() if row.strip()]
    if len(rows) != 1 or len(rows[0]) != 2:
        raise ValueError(f"Ambiguous main ref for {repository}: {output!r}")
    sha, ref = rows[0]
    if ref != "refs/heads/main" or not FULL_SHA.fullmatch(sha):
        raise ValueError(f"Invalid main ref for {repository}: {output!r}")
    return sha


def snapshot(workflow_sha: str, resolver=resolve_main, attempts: int = 3) -> dict:
    if not FULL_SHA.fullmatch(workflow_sha):
        raise ValueError("workflow_sha must be the full workflow commit SHA")
    # GitHub has no atomic snapshot across repositories. Require two consecutive
    # identical observations; if a merge races us, restart the entire snapshot.
    for _ in range(attempts):
        first = {key: resolver(repo) for key, repo in REPOSITORIES.items()}
        second = {key: resolver(repo) for key, repo in REPOSITORIES.items()}
        if first == second:
            lock = {
                "schema_version": 1,
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "workflow_sha": workflow_sha,
                "sources": {
                    key: {
                        "repository": repo,
                        "ref": "refs/heads/main",
                        "sha": first[key],
                    }
                    for key, repo in REPOSITORIES.items()
                },
            }
            validate_lock(lock)
            return lock
    raise RuntimeError(
        "main heads kept moving; retry the workflow after the merges settle"
    )


def validate_lock(lock: dict) -> None:
    if lock.get("schema_version") != 1:
        raise ValueError("Unsupported source lock schema")
    if not FULL_SHA.fullmatch(lock.get("workflow_sha", "")):
        raise ValueError("Invalid workflow SHA")
    if set(lock.get("sources", {})) != set(REPOSITORIES):
        raise ValueError(
            "The source lock must contain exactly the four KT repositories"
        )
    for key, repo in REPOSITORIES.items():
        source = lock["sources"][key]
        if source.get("repository") != repo or source.get("ref") != "refs/heads/main":
            raise ValueError(f"Only {repo} main is an allowed source")
        if not FULL_SHA.fullmatch(source.get("sha", "")):
            raise ValueError(f"Invalid source SHA for {key}")


def read_lock(path: Path) -> dict:
    lock = json.loads(path.read_text())
    validate_lock(lock)
    return lock


def checkout(lock: dict, destination: Path) -> None:
    validate_lock(lock)
    destination.mkdir(parents=True, exist_ok=False)
    for key, source in lock["sources"].items():
        target = destination / key
        run("git", "init", str(target))
        run(
            "git",
            "remote",
            "add",
            "origin",
            f"https://github.com/{source['repository']}.git",
            cwd=target,
        )
        run("git", "fetch", "--depth=1", "origin", source["sha"], cwd=target)
        run("git", "checkout", "--detach", "FETCH_HEAD", cwd=target)
        if run("git", "rev-parse", "HEAD", cwd=target) != source["sha"]:
            raise RuntimeError(f"Checkout SHA mismatch for {key}")
        modules = target / ".gitmodules"
        if modules.exists():
            entries = run(
                "git",
                "config",
                "--file",
                ".gitmodules",
                "--get-regexp",
                r"^submodule\..*\.path$",
                cwd=target,
            )
            paths = [line.split(maxsplit=1)[1] for line in entries.splitlines()]
            # SGLang is checked out independently at the locked main SHA, never
            # built from the potentially stale KT gitlink.
            paths = [
                p
                for p in paths
                if not (key == "ktransformers" and p == "third_party/sglang")
            ]
            for path in paths:
                if (
                    PurePosixPath(path).is_absolute()
                    or ".." in PurePosixPath(path).parts
                ):
                    raise ValueError(f"Unsafe submodule path: {path}")
            if paths:
                run(
                    "git",
                    "submodule",
                    "update",
                    "--init",
                    "--recursive",
                    "--",
                    *paths,
                    cwd=target,
                )


def source_evidence(lock: dict, sources: Path) -> dict:
    evidence = {}
    for key, source in lock["sources"].items():
        target = sources / key
        sha = run("git", "rev-parse", "HEAD", cwd=target)
        # Native builds can apply checked-in patches inside dependency
        # submodules. Record their status separately, without accepting edits
        # to the four repositories' own tracked files.
        diff = run("git", "diff", "HEAD", "--ignore-submodules=dirty", cwd=target)
        if sha != source["sha"] or diff:
            raise RuntimeError(f"Source changed during the build: {key}")
        evidence[key] = {
            **source,
            "submodules": run(
                "git", "submodule", "status", "--recursive", cwd=target
            ).splitlines(),
        }
    return evidence


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_wheel(path: Path) -> dict:
    name, version, _, tags = parse_wheel_filename(path.name)
    with zipfile.ZipFile(path) as wheel:
        metadata_files = [
            p
            for p in wheel.namelist()
            if p.endswith(".dist-info/METADATA") and len(PurePosixPath(p).parts) == 2
        ]
        if len(metadata_files) != 1:
            raise ValueError(f"Expected one METADATA in {path.name}")
        metadata = email.message_from_bytes(wheel.read(metadata_files[0]))
        if canonicalize_name(metadata["Name"]) != name or metadata["Version"] != str(
            version
        ):
            raise ValueError(f"Wheel filename and METADATA disagree: {path.name}")
    return {
        "name": name,
        "version": str(version),
        "filename": path.name,
        "sha256": sha256(path),
        "size": path.stat().st_size,
        "tags": sorted(str(tag) for tag in tags),
        "requires_dist": metadata.get_all("Requires-Dist", []),
    }


def active_requirements(wheel: dict) -> list[Requirement]:
    # Test/dev extras are intentionally outside the published user workflows.
    result = []
    for text in wheel["requires_dist"]:
        req = Requirement(text)
        if req.marker is None or any(
            req.marker.evaluate({"extra": extra}) for extra in ("", "sglang", "sft")
        ):
            result.append(req)
    return result


def audit_wheels(wheels: list[dict], *, python_only: bool = False) -> list[str]:
    errors = []
    names = [wheel["name"] for wheel in wheels]
    expected = set(PACKAGE_SOURCES)
    if python_only:
        expected -= {"kt-kernel", "sgl-kernel-kt"}
    if len(names) != len(set(names)) or set(names) != expected:
        errors.append(
            f"Expected one raw wheel per project: {sorted(expected)}; got {names}"
        )
        return errors
    by_name = {wheel["name"]: wheel for wheel in wheels}
    versions = {name: wheel["version"] for name, wheel in by_name.items()}
    if python_only:
        # Both KT projects use the same checked-in version.py. Check pins
        # before spending hours on native builds; verify the real native
        # wheel's version again in the complete audit.
        versions["kt-kernel"] = versions["ktransformers"]
    for name, wheel in by_name.items():
        requirements = active_requirements(wheel)
        for req in requirements:
            dep = canonicalize_name(req.name)
            if dep in {"transformers", "accelerate", "sgl-kernel"}:
                errors.append(
                    f"{name} requires conflicting upstream distribution {req}"
                )
            if dep in versions and (req.url or versions[dep] not in req.specifier):
                errors.append(
                    f"{name} requires {req}, but this source stack selects {dep}=={versions[dep]}"
                )
        for dep in EXACT_DEPENDENCIES.get(name, ()):
            matches = [r for r in requirements if canonicalize_name(r.name) == dep]
            expected_pin = f"=={versions[dep]}"
            if not matches or any(
                r.url or str(r.specifier) != expected_pin for r in matches
            ):
                errors.append(
                    f"{name} must pin {dep}{expected_pin} in its main source metadata"
                )
    return errors


def audit(
    lock: dict,
    sources: Path,
    wheels_dir: Path,
    output: Path,
    *,
    python_only: bool = False,
) -> bool:
    evidence = source_evidence(lock, sources)
    wheels = [inspect_wheel(path) for path in sorted(wheels_dir.rglob("*.whl"))]
    for wheel in wheels:
        key = PACKAGE_SOURCES.get(wheel["name"])
        if key:
            wheel["source"] = evidence[key]
    errors = audit_wheels(wheels, python_only=python_only)
    report = {
        "schema_version": 1,
        "stage": "python-metadata-preflight" if python_only else "raw-native-wheels",
        "source_lock": lock,
        "wheels": wheels,
        "errors": errors,
        "metadata_consistent": not errors,
        "publishable": False,
        "pending_gates": [
            "Assemble fresh SGL CUDA payload into version-parameterized carrier wheels",
            "Resolve complete dependencies into an isolated, hashed wheelhouse",
            "Install the final wheels in the isolated qj5090 runner and run the three-model E2E suite",
            "Promote the exact validated wheel hashes and verify installation from PyPI",
        ],
    }
    if python_only:
        report["pending_gates"].insert(
            0, "Build and audit fresh KT and SGL native wheels"
        )
    save_json(output, report)
    for error in errors:
        print(error, file=sys.stderr)
    return not errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("snapshot")
    freeze.add_argument("--workflow-sha", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    fetch = commands.add_parser("checkout")
    fetch.add_argument("--lock", type=Path, required=True)
    fetch.add_argument("--destination", type=Path, required=True)
    verify = commands.add_parser("audit")
    verify.add_argument("--lock", type=Path, required=True)
    verify.add_argument("--sources", type=Path, required=True)
    verify.add_argument("--wheels", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--python-only", action="store_true")
    args = parser.parse_args()
    if args.command == "snapshot":
        save_json(args.output, snapshot(args.workflow_sha))
    elif args.command == "checkout":
        checkout(read_lock(args.lock), args.destination)
    else:
        if not audit(
            read_lock(args.lock),
            args.sources,
            args.wheels,
            args.output,
            python_only=args.python_only,
        ):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
