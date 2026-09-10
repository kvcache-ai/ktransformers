"""Lock dependencies, check E2E evidence, and promote exactly the tested wheels.

The publish command is called only on a GitHub-hosted job with protected PyPI
credentials, never on the native build or community/model-test runner.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from cleanup import cleanup
from four_main import inspect_wheel, read_lock, save_json, sha256
from packaging.version import Version

# #2195 owns the acceptance contract. Merge it before enabling this workflow.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model-e2e"))
from contracts import CASES, PACKAGES, require, suite_passed  # noqa: E402
from release_contracts import verify_install_report, verify_release  # noqa: E402

ORDER = ("accelerate-kt", "transformers-kt", "kt-kernel", "sglang-kt", "ktransformers")


def pypi_files(name, version):
    request = urllib.request.Request(
        f"https://pypi.org/pypi/{name}/{version}/json",
        headers={"Cache-Control": "no-cache"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)["urls"]
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return []
        raise


def check_unused(wheels, query=pypi_files):
    for wheel in wheels:
        version = Version(wheel["version"])
        require(
            not version.is_prerelease and not version.local,
            "This release entry point requires stable source versions",
        )
        require(
            not query(wheel["name"], wheel["version"]),
            f"{wheel['name']}=={wheel['version']} already exists on PyPI; commit new source versions/pins before rebuilding",
        )


def pip(*args):
    subprocess.run(
        [sys.executable, "-m", "pip", "--isolated", *map(str, args)],
        check=True,
        timeout=7200,
    )


def make_release(final, root, lock_path, evidence):
    root.mkdir(exist_ok=False)
    house = root / "wheelhouse"
    house.mkdir()
    wheels = [inspect_wheel(path) for path in sorted(final.glob("*.whl"))]
    require(
        {item["name"] for item in wheels} == PACKAGES and len(wheels) == len(PACKAGES),
        "Need five final wheels",
    )
    check_unused(wheels)
    # Resolve user-facing extras without direct wheel URLs or --no-deps. Fresh
    # versions prevent PyPI from silently substituting an older same-version file.
    pip(
        "download",
        "--index-url",
        "https://pypi.org/simple",
        "--no-cache-dir",
        "--only-binary=:all:",
        "--find-links",
        final,
        "--dest",
        house,
        "ktransformers[sglang,sft]",
    )
    house_entries = [inspect_wheel(path) for path in sorted(house.glob("*.whl"))]
    require(
        len({entry["name"] for entry in house_entries}) == len(house_entries),
        "Duplicate dependency version",
    )
    require(
        all(
            entry["name"]
            not in {
                "transformers",
                "accelerate",
                "sglang",
                "sgl-kernel",
                "sgl-kernel-kt",
            }
            for entry in house_entries
        ),
        "Conflicting upstream namespace in dependency closure",
    )
    for wheel in wheels:
        require(
            (house / wheel["filename"]).is_file()
            and sha256(house / wheel["filename"]) == wheel["sha256"],
            "Resolver did not select this run's final wheel: " + wheel["name"],
        )
    lock = read_lock(lock_path)
    manifest = {
        "schema": 1,
        "run_id": int(os.environ["GITHUB_RUN_ID"]),
        "run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
        "workflow_sha": lock["workflow_sha"],
        # A packaging-only recovery can use a newer trusted CI implementation
        # while preserving the original compiled four-main runtime snapshot.
        "assembly_workflow_sha": os.environ["GITHUB_SHA"],
        "source_lock": lock,
        "wheels": {
            entry["name"]: {
                "file": entry["filename"],
                "version": entry["version"],
                "sha256": entry["sha256"],
            }
            for entry in wheels
        },
        "wheelhouse": {
            entry["filename"]: {
                "name": entry["name"],
                "version": entry["version"],
                "sha256": entry["sha256"],
            }
            for entry in house_entries
        },
        "plans": {},
    }
    for extra in ("sglang", "sglang,sft"):
        report = evidence / ("resolution-" + extra.replace(",", "-") + ".json")
        pip(
            "install",
            "--dry-run",
            "--ignore-installed",
            "--no-cache-dir",
            "--only-binary=:all:",
            "--no-index",
            "--find-links",
            house,
            "--report",
            report,
            f"ktransformers[{extra}]",
        )
        selected = {}
        for item in json.loads(report.read_text())["install"]:
            name = item["metadata"]["name"].lower().replace("_", "-")
            matches = [
                filename
                for filename, entry in manifest["wheelhouse"].items()
                if entry["name"] == name
            ]
            require(len(matches) == 1, "Unexpected dependency resolution")
            selected[name] = matches[0]
        manifest["plans"][extra] = selected
        verify_install_report(
            json.loads(report.read_text()), manifest, extra, public=False
        )
    save_json(root / "release.json", manifest)
    digest = sha256(root / "release.json")
    verify_release(root, digest)
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        stream.write(
            f"manifest_sha256={digest}\nbuild_attempt={manifest['run_attempt']}\n"
        )


def check_evidence(directory, manifest, manifest_digest, stage):
    directory = Path(directory)
    request = json.loads((directory / "request.json").read_text())
    require(request["mode"] == "release-" + stage, "Wrong acceptance stage")
    require(
        request["manifest_sha256"] == manifest_digest,
        "Acceptance tested a different candidate",
    )
    require(
        request["build_run_id"] == manifest["run_id"],
        "Acceptance belongs to another run",
    )
    require(
        request["build_run_attempt"] == manifest["run_attempt"],
        "Acceptance belongs to another build attempt",
    )
    require(
        request["build_workflow_sha"]
        == manifest["workflow_sha"]
        == request["harness_sha"],
        "Wrong acceptance harness",
    )
    for path in (directory / "host-result.json", directory / "tests/suite.json"):
        result = json.loads(path.read_text())
        require(
            result["status"] == "passed" and suite_passed(result["cases"]),
            "Incomplete/failed model acceptance",
        )
    for extra, label in (("sglang", "serving"), ("sglang,sft", "combined")):
        report = json.loads(
            (directory / f"tests/release-{label}-install.json").read_text()
        )
        verify_install_report(report, manifest, extra, public=stage == "pypi")


def existing_matches(entry, files):
    if not files:
        return False
    require(
        len(files) == 1,
        "Published version contains unexpected files; refuse to mix artifacts",
    )
    published = files[0]
    require(not published.get("yanked", False), "Published candidate was yanked")
    require(
        published["filename"] == entry["file"]
        and published["digests"]["sha256"] == entry["sha256"],
        "PyPI version already contains different bytes; never skip-existing",
    )
    return True


def promote(root, digest, evidence, output, query=pypi_files, upload=None):
    manifest = verify_release(root, digest)
    check_evidence(evidence, manifest, digest, "candidate")
    require(os.environ.get("GITHUB_REF") == "refs/heads/main", "Publish requires main")
    require(
        os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch",
        "Publish requires manual dispatch",
    )
    require(
        os.environ.get("GITHUB_REPOSITORY") == "kvcache-ai/ktransformers",
        "Publish requires official repository",
    )
    require(
        int(os.environ["GITHUB_RUN_ID"]) == manifest["run_id"],
        "Publish must use the same workflow run",
    )
    require(
        os.environ["GITHUB_SHA"] == manifest["workflow_sha"], "Publish workflow changed"
    )
    if upload is None:

        def upload(path):
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "twine",
                    "upload",
                    "--non-interactive",
                    "--disable-progress-bar",
                    "--repository-url",
                    "https://upload.pypi.org/legacy/",
                    str(path),
                ],
                check=True,
                timeout=900,
            )

    # Preflight the ENTIRE batch before any write. On a retry, allow only exact
    # filename+SHA256 matches from this candidate, never --skip-existing.
    for name in ORDER:
        entry = manifest["wheels"][name]
        existing_matches(entry, query(name, entry["version"]))
    result = {"manifest_sha256": digest, "status": "uploading", "packages": {}}
    try:
        for name in ORDER:
            entry = manifest["wheels"][name]
            path = Path(root) / "wheelhouse" / entry["file"]
            require(sha256(path) == entry["sha256"], "Wheel changed before upload")
            present = existing_matches(entry, query(name, entry["version"]))
            if not present:
                # A failed job may be rerun against the retained candidate. Do
                # not rebuild or silently continue after an upload error.
                upload(path)
            deadline = time.monotonic() + 600
            while not existing_matches(entry, query(name, entry["version"])):
                require(
                    time.monotonic() < deadline,
                    "PyPI visibility timeout; rerun failed jobs with the same artifact",
                )
                time.sleep(15)
            result["packages"][name] = entry | {"already_present": present}
            save_json(output, result)
        result["status"] = "uploaded-awaiting-public-e2e"
    except BaseException:
        result["status"] = "partial-or-failed-upload"
        raise
    finally:
        save_json(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("unused")
    preflight.add_argument("--wheels", type=Path, required=True)
    make = sub.add_parser("prepare")
    make.add_argument("--final", type=Path, required=True)
    make.add_argument("--root", type=Path, required=True)
    make.add_argument("--lock", type=Path, required=True)
    make.add_argument("--evidence", type=Path, required=True)
    for name in ("publish", "verify"):
        action = sub.add_parser(name)
        action.add_argument("--root", type=Path, required=True)
        action.add_argument("--digest", required=True)
        action.add_argument("--evidence", type=Path, required=True)
        action.add_argument("--output", type=Path, required=True)
    clean = sub.add_parser("cleanup")
    clean.add_argument("--directory", type=Path, required=True)
    clean.add_argument("--parent", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "unused":
        wheels = [inspect_wheel(path) for path in args.wheels.glob("*.whl")]
        if not any(wheel["name"] == "kt-kernel" for wheel in wheels):
            kt = next(wheel for wheel in wheels if wheel["name"] == "ktransformers")
            wheels.append({"name": "kt-kernel", "version": kt["version"]})
        check_unused(wheels)
    elif args.command == "prepare":
        make_release(args.final, args.root, args.lock, args.evidence)
    elif args.command == "publish":
        promote(args.root, args.digest, args.evidence, args.output)
    elif args.command == "verify":
        manifest = verify_release(args.root, args.digest)
        check_evidence(args.evidence, manifest, args.digest, "pypi")
        save_json(
            args.output,
            {
                "status": "released-and-verified",
                "manifest_sha256": args.digest,
                "run_id": manifest["run_id"],
                "cases": list(CASES),
            },
        )
    else:
        cleanup(args.directory, args.parent)


if __name__ == "__main__":
    main()
