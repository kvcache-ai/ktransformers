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


def _xpu_hidden_states(rows, width):
    """An ``xpu`` input for ``submit_forward_inference``'s device-type branch.

    That branch only reads ``hidden_states.device`` before handing the batch to ``submit_forward``,
    which this file stubs out anyway, so a device stand-in is enough on a host without an Intel GPU.
    """
    if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        return torch.ones(rows, width, dtype=torch.bfloat16, device="xpu")
    return SimpleNamespace(device=torch.device("xpu"))


def test_xpu_inference_takes_the_synchronous_fallback(runtime, monkeypatch):
    """torch.xpu has no ExternalStream/Event/record_stream, so xpu must not enter the stream path."""
    wrapper, _, _, recorded = runtime

    def forbidden(*args, **kwargs):
        pytest.fail("xpu must not touch the torch.cuda stream API")

    monkeypatch.setattr(torch.cuda, "ExternalStream", forbidden)
    monkeypatch.setattr(torch.cuda, "stream", forbidden)
    monkeypatch.setattr(torch.cuda, "Event", forbidden)

    expected = torch.full((2, 4), 3.0, dtype=torch.bfloat16)
    submitted, synced = [], []
    wrapper.submit_forward = lambda *args, **kwargs: submitted.append((args, kwargs))

    def sync_forward(output_device=None):
        synced.append(output_device)
        return expected

    wrapper.sync_forward = sync_forward
    hidden = _xpu_hidden_states(2, 4)

    BaseSFTMoEWrapper.submit_forward_inference(
        wrapper, hidden, torch.zeros(2, 2, dtype=torch.int64), torch.ones(2, 2), 1
    )
    assert len(submitted) == 1
    assert submitted[0][0][0] is hidden
    assert submitted[0][1] == {"save_for_backward": False}
    assert wrapper._pending_inference_fallback is True
    assert not hasattr(wrapper, "_pending_inference_buffer")

    assert BaseSFTMoEWrapper.sync_forward_inference(wrapper, 1) is expected
    assert synced == [hidden.device] and synced[0].type == "xpu"
    assert wrapper._pending_inference_fallback is False
    assert wrapper._inference_inflight == [] and recorded == []
