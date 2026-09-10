"""Resolve public metadata on a GitHub-hosted runner. Never execute PR code."""

from __future__ import annotations

import json
import os
import urllib.request

from contracts import REPOSITORIES, require, sha, validate_request, write_json


def api(path):
    request = urllib.request.Request(
        "https://api.github.com/" + path,
        headers={
            "Authorization": "Bearer " + os.environ["GH_TOKEN"],
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main():
    request = {
        "schema": 1,
        "mode": os.environ["MODE"],
        "harness_sha": sha(os.environ["GITHUB_SHA"]),
    }
    require(
        os.environ.get("ENABLED") == "true",
        "KT_MODEL_E2E_ENABLED must be explicitly enabled after sandbox setup",
    )
    if request["mode"] == "pypi":
        request["version"] = os.environ["VERSION"]
        # Keep the user's unpinned installation, but reject silent backtracking.
        version = request["version"]
        require(
            version and all(char.isalnum() or char in ".!+_-" for char in version),
            "Invalid version",
        )
        url = (
            "https://pypi.org/pypi/ktransformers/"
            + ("" if version == "latest" else version + "/")
            + "json"
        )
        with urllib.request.urlopen(url, timeout=60) as response:
            request["expected_version"] = json.load(response)["info"]["version"]
    else:
        repo = os.environ["PR_REPOSITORY"]
        require(repo in REPOSITORIES.values(), "Unexpected PR repository")
        number = int(os.environ["PR_NUMBER"])
        require(number > 0, "PR number is required")
        expected = sha(os.environ["PR_SHA"])
        pr = api(f"repos/{repo}/pulls/{number}")
        require(
            pr["state"] == "open" and pr["base"]["ref"] == "main",
            "Require an open PR targeting main",
        )
        require(
            pr["head"]["sha"] == expected,
            "PR changed: review and approve its new SHA before dispatching",
        )
        request["pr"] = {"repository": repo, "number": number, "sha": expected}
        # Read twice so a moving main is not silently mixed into this snapshot.
        sources = {}
        for key, repository in REPOSITORIES.items():
            commit = (
                expected
                if repository == repo
                else api(f"repos/{repository}/commits/main")["sha"]
            )
            sources[key] = {"repository": repository, "sha": sha(commit)}
        for key, repository in REPOSITORIES.items():
            if repository != repo:
                require(
                    api(f"repos/{repository}/commits/main")["sha"]
                    == sources[key]["sha"],
                    "Main moved; dispatch again",
                )
        request["sources"] = sources
        run_id = int(os.environ["BUILD_RUN_ID"])
        require(run_id > 0, "Candidate build run ID is required")
        run = api(f"repos/kvcache-ai/ktransformers/actions/runs/{run_id}")
        require(run["conclusion"] == "success", "Candidate build has not succeeded")
        require(
            run["event"] == "workflow_dispatch" and run["head_branch"] == "main",
            "Require a trusted main build workflow",
        )
        workflow = os.environ["BUILD_WORKFLOW"]
        require(
            workflow.startswith(".github/workflows/") and workflow.endswith(".yml"),
            "Configure the approved builder workflow",
        )
        require(run["path"] == workflow, "Unexpected candidate builder workflow")
        request["build_run_id"] = run_id
        request["build_workflow_sha"] = sha(run["head_sha"])
    write_json("request.json", validate_request(request))


if __name__ == "__main__":
    main()
