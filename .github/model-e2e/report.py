"""Report on a GitHub-hosted runner; no write token ever reaches qj5090."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

from contracts import CASES, suite_passed, validate_request


def main():
    request_path = Path("input/request.json")
    result_path = Path("evidence/host-result.json")
    request = (
        validate_request(json.loads(request_path.read_text()))
        if request_path.exists()
        else None
    )
    result = (
        json.loads(result_path.read_text())
        if result_path.exists()
        else {"status": "not_run", "cases": {}}
    )
    passed = (
        os.environ["GPU_JOB_RESULT"] == "success"
        and result["status"] == "passed"
        and suite_passed(result["cases"])
    )
    # Do not render arbitrary model/PR output as Markdown or workflow commands.
    lines = [
        "## 模型 E2E 验收",
        "",
        f"结论：{'通过' if passed else '未通过或未执行'}",
        "",
        "| 项目 | 结果 |",
        "| --- | --- |",
    ]
    for case in CASES:
        status = result.get("cases", {}).get(case, {}).get("status", "not_run")
        lines.append(
            f"| {case} | {'通过' if status == 'passed' else '未通过 / 未执行'} |"
        )
    if result["status"] == "resource_unavailable":
        lines += ["", "qj5090 排队等待超时；未进行模型测试，不归因为模型失败。"]
    Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text("\n".join(lines) + "\n")
    if (
        request
        and request["mode"] == "pr"
        and request["pr"]["repository"] == "kvcache-ai/ktransformers"
    ):
        # Associate ONLY with the approved head SHA, never a later PR revision.
        pr = request["pr"]
        payload = {
            "state": "success" if passed else "failure",
            "context": "model-e2e/qj5090",
            "description": "3 model smoke tests passed"
            if passed
            else "Acceptance incomplete; inspect the run report",
            "target_url": f"https://github.com/kvcache-ai/ktransformers/actions/runs/{os.environ['GITHUB_RUN_ID']}",
        }
        api_request = urllib.request.Request(
            f"https://api.github.com/repos/kvcache-ai/ktransformers/statuses/{pr['sha']}",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": "Bearer " + os.environ["GH_TOKEN"],
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(api_request, timeout=60):
            pass
    # Cross-repository runs still have an artifact/report. A dedicated GitHub App
    # is required before enabling status writes to the other three repositories.
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
