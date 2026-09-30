# SPDX-License-Identifier: Apache-2.0
"""Async native tasks must outlive shared-buffer replacement."""

from contextlib import nullcontext
import gc
from types import SimpleNamespace
import weakref

import pytest
import torch

from kt_kernel.sft.base import BaseSFTMoEWrapper, KExpertsCPUBuffer


@pytest.fixture
def runtime(monkeypatch):
    class Stream:
        def __init__(self, *args, **kwargs):
            pass

    class Event:
        complete = False

        def record(self, stream):
            assert isinstance(stream, Stream)

        def query(self):
            return self.complete

    arena = SimpleNamespace(latest=None)
    recorded = []

    def get_buffer(hidden, topk):
        rows, width = hidden.shape
        arena.latest = (
            [torch.zeros(rows, width, dtype=torch.bfloat16)],
            [torch.zeros(rows, topk, dtype=torch.int64)],
            [torch.zeros(rows, topk, dtype=torch.int64)],
            [torch.zeros(rows, topk)],
            [torch.zeros(rows, width, dtype=torch.bfloat16)],
            [torch.tensor([rows], dtype=torch.int32)],
            [torch.zeros(rows, width, dtype=torch.bfloat16)],
        )
        return arena.latest

    monkeypatch.setattr(KExpertsCPUBuffer, "get_buffer", get_buffer)
    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch.cuda, "ExternalStream", Stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, stream: recorded.append(stream))
    wrapper = SimpleNamespace(
        layer_idx=0,
        num_experts_per_tok=2,
        _inference_inflight=[],
        _validate_forward_inputs=lambda *args: None,
        _wait_for_pending_backward_repack=lambda: None,
        _make_forward_task=lambda buffer, **kwargs: weakref.ref(buffer.bsz_tensor),
        cpu_infer=SimpleNamespace(
            submit_with_cuda_stream=lambda *args: None,
            sync_with_cuda_stream=lambda stream: None,
        ),
    )

    def submit(rows):
        BaseSFTMoEWrapper.submit_forward_inference(
            wrapper,
            torch.ones(rows, 4, dtype=torch.bfloat16),
            torch.zeros(rows, 2, dtype=torch.int64),
            torch.ones(rows, 2),
            1,
        )

    return wrapper, arena, submit, recorded


def test_inflight_cpu_scalar_survives_shape_change_until_completion(runtime):
    wrapper, arena, submit, recorded = runtime
    submit(2)
    old_scalar = weakref.ref(arena.latest[5][0])
    BaseSFTMoEWrapper.sync_forward_inference(wrapper, 1)

    submit(5)
    gc.collect()
    assert old_scalar() is not None and old_scalar().item() == 2
    first_event = wrapper._inference_inflight[0][0]
    BaseSFTMoEWrapper.sync_forward_inference(wrapper, 1)
    assert len(wrapper._inference_inflight) == 2
    assert len(recorded) == 2

    first_event.complete = True
    submit(1)
    gc.collect()
    assert old_scalar() is None
    assert len(wrapper._inference_inflight) == 1


def test_pending_submission_cannot_be_overwritten(runtime):
    wrapper, _, submit, _ = runtime
    submit(2)
    with pytest.raises(RuntimeError, match="already pending"):
        submit(3)
    BaseSFTMoEWrapper.sync_forward_inference(wrapper, 1)
    with pytest.raises(RuntimeError, match="No pending inference"):
        BaseSFTMoEWrapper.sync_forward_inference(wrapper, 1)


def test_capture_does_not_query_captured_events(runtime, monkeypatch):
    wrapper, _, submit, _ = runtime
    submit(2)
    BaseSFTMoEWrapper.sync_forward_inference(wrapper, 1)
    event = wrapper._inference_inflight[0][0]
    monkeypatch.setattr(event, "query", lambda: pytest.fail("queried a captured event"))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    submit(2)
    assert len(wrapper._inference_inflight) == 1
