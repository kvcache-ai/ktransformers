"""Same-run release entry point; no main re-resolution, publication, or PR status."""

import os

from contracts import require, sha, validate_request, write_json
from release_contracts import verify_release


def main():
    require(
        os.environ.get("ENABLED") == "true",
        "Configure and enable the isolated model runner first",
    )
    require(
        os.environ["GITHUB_REPOSITORY"] == "kvcache-ai/ktransformers",
        "Wrong repository",
    )
    require(
        os.environ["GITHUB_REF"] == "refs/heads/main",
        "Release entry point requires main",
    )
    require(
        os.environ["GITHUB_EVENT_NAME"] == "workflow_dispatch",
        "Release must be manually dispatched",
    )
    require(
        os.environ["GITHUB_WORKFLOW_REF"]
        .split("@")[0]
        .endswith("/.github/workflows/release-four-main.yml"),
        "Unexpected caller workflow",
    )
    stage = os.environ["STAGE"]
    require(stage in ("candidate", "pypi"), "Invalid release stage")
    request = validate_request(
        {
            "schema": 1,
            "mode": "release-" + stage,
            "harness_sha": sha(os.environ["GITHUB_SHA"]),
            "manifest_sha256": os.environ["MANIFEST_SHA256"],
            "build_run_id": int(os.environ["GITHUB_RUN_ID"]),
            "build_run_attempt": int(os.environ["BUILD_ATTEMPT"]),
            "build_workflow_sha": sha(os.environ["GITHUB_SHA"]),
        }
    )
    manifest = verify_release("input/release", request["manifest_sha256"])
    require(manifest["run_id"] == request["build_run_id"], "Wrong release run")
    require(
        manifest["run_attempt"] == request["build_run_attempt"], "Wrong build attempt"
    )
    require(
        manifest["workflow_sha"] == request["build_workflow_sha"], "Wrong workflow SHA"
    )
    write_json("input/request.json", request)


if __name__ == "__main__":
    main()
