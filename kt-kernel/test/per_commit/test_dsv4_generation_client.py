"""Validate generation responses and stop queued requests after a failed sample."""

import importlib.util
import io
import json
from pathlib import Path
import sys

import pytest


spec = importlib.util.spec_from_file_location(
    "dsv4_generation", Path(__file__).parents[1] / "generate_dsv4_acceptance.py"
)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


@pytest.mark.parametrize("invalid", [None, "abort", "nonfinite", "empty"])
def test_generation_response_validation(monkeypatch, invalid):
    record = {"encoded_prompt": "exact chat prefix", "source_index": 7}
    response = {
        "text": "answer" if invalid != "empty" else "",
        "meta_info": {
            "completion_tokens": 1,
            "finish_reason": {"type": "abort" if invalid == "abort" else "stop"},
            "output_token_logprobs": [
                [float("nan") if invalid == "nonfinite" else -0.5, 42, None]
            ],
        },
    }

    def open_request(request, timeout):
        body = json.loads(request.data)
        assert body["text"] == record["encoded_prompt"]
        assert body["sampling_params"] == {"temperature": 0, "max_new_tokens": 16}
        assert timeout == 600
        return io.BytesIO(json.dumps(response).encode())

    monkeypatch.setattr(client.urllib.request, "urlopen", open_request)
    if invalid:
        with pytest.raises(AssertionError):
            client.generate("http://localhost", record, 16)
    else:
        result = client.generate("http://localhost", record, 16)
        assert result["source_index"] == 7 and result["response"] == response


def test_failed_sample_cancels_pending_and_preserves_completed_rows(
    tmp_path, monkeypatch
):
    class FailingPool:
        cancelled = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, function, rows):
            yield {**rows[0], "seconds": 0.1, "response": {"text": "answer"}}
            raise RuntimeError("sample failed")

        def shutdown(self, *, wait, cancel_futures):
            assert wait
            self.cancelled = cancel_futures

    selected, output = tmp_path / "selected.json", tmp_path / "output.jsonl"
    selected.write_text(
        json.dumps({"validation": [{"source_index": 1}, {"source_index": 2}]})
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate",
            "--selected",
            str(selected),
            "--output",
            str(output),
            "--count",
            "2",
        ],
    )
    pool = FailingPool()
    monkeypatch.setattr(client, "ThreadPoolExecutor", lambda max_workers: pool)
    with pytest.raises(RuntimeError, match="sample failed"):
        client.main()
    assert pool.cancelled
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["source_index"] == 1
