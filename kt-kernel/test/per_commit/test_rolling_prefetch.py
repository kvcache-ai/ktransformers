"""
Unit tests for RollingPrefetchScheduler — the pure state machine used by the
opt-in Rolling Layer Prefetch (RLP) MESH prefill strategy. No I/O is performed
here; the scheduler only tracks per-layer pipeline state and tells the caller
which layer to submit next.
"""

import os
import sys

import pytest

try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../build"))
    import kt_kernel_ext as ext

    HAS_RLP = hasattr(ext, "RollingPrefetchScheduler") and hasattr(ext, "LayerPipelineState")
except (ImportError, AttributeError):
    HAS_RLP = False

pytestmark = pytest.mark.skipif(
    not HAS_RLP, reason="RollingPrefetchScheduler binding not available (build kt_kernel_ext first)"
)


def _state():
    return ext.LayerPipelineState


def test_constructor_rejects_zero_or_negative_depth():
    with pytest.raises(ValueError):
        ext.RollingPrefetchScheduler(0, 5)
    with pytest.raises(ValueError):
        ext.RollingPrefetchScheduler(-1, 5)


def test_constructor_rejects_zero_or_negative_total():
    with pytest.raises(ValueError):
        ext.RollingPrefetchScheduler(3, 0)
    with pytest.raises(ValueError):
        ext.RollingPrefetchScheduler(3, -2)


def test_initial_state_all_empty():
    s = ext.RollingPrefetchScheduler(3, 5)
    assert s.depth == 3
    assert s.total_layers == 5
    assert s.next_submit_layer == 0
    assert s.next_compute_layer == 0
    assert s.bootstrapped is False
    for layer in range(5):
        assert s.state_of(layer) == _state().Empty


def test_bootstrap_returns_first_depth_layers():
    s = ext.RollingPrefetchScheduler(3, 10)
    submitted = s.bootstrap()
    assert submitted == [0, 1, 2]
    assert s.bootstrapped is True
    assert s.next_submit_layer == 3
    for layer in range(3):
        assert s.state_of(layer) == _state().Reading
    for layer in range(3, 10):
        assert s.state_of(layer) == _state().Empty


def test_bootstrap_clamps_when_depth_exceeds_total():
    s = ext.RollingPrefetchScheduler(10, 5)
    submitted = s.bootstrap()
    assert submitted == [0, 1, 2, 3, 4]
    assert s.next_submit_layer == 5


def test_bootstrap_twice_raises():
    s = ext.RollingPrefetchScheduler(3, 5)
    s.bootstrap()
    with pytest.raises(RuntimeError):
        s.bootstrap()


def test_mark_ready_transitions_reading_to_ready():
    s = ext.RollingPrefetchScheduler(3, 5)
    s.bootstrap()
    s.mark_ready(0)
    assert s.state_of(0) == _state().Ready
    assert s.state_of(1) == _state().Reading
    assert s.state_of(2) == _state().Reading


def test_mark_ready_on_wrong_state_raises():
    s = ext.RollingPrefetchScheduler(3, 5)
    s.bootstrap()
    # Layer 3 is Empty (not yet submitted)
    with pytest.raises(RuntimeError):
        s.mark_ready(3)
    # Layer 0 already Ready cannot be re-marked
    s.mark_ready(0)
    with pytest.raises(RuntimeError):
        s.mark_ready(0)


def test_begin_compute_requires_in_order():
    s = ext.RollingPrefetchScheduler(3, 5)
    s.bootstrap()
    s.mark_ready(0)
    s.mark_ready(1)
    # next_compute_layer is 0; computing 1 first must fail
    with pytest.raises(RuntimeError):
        s.begin_compute(1)


def test_begin_compute_requires_ready_state():
    s = ext.RollingPrefetchScheduler(3, 5)
    s.bootstrap()
    # Layer 0 is still Reading
    with pytest.raises(RuntimeError):
        s.begin_compute(0)


def test_on_layer_compute_done_releases_and_advances():
    s = ext.RollingPrefetchScheduler(3, 5)
    s.bootstrap()
    s.mark_ready(0)
    s.begin_compute(0)
    nxt = s.on_layer_compute_done(0)
    assert nxt == 3
    assert s.state_of(0) == _state().Released
    assert s.state_of(3) == _state().Reading
    assert s.next_compute_layer == 1
    assert s.next_submit_layer == 4


def test_on_layer_compute_done_requires_computing_state():
    s = ext.RollingPrefetchScheduler(3, 5)
    s.bootstrap()
    s.mark_ready(0)
    # Skipped begin_compute, layer 0 still Ready
    with pytest.raises(RuntimeError):
        s.on_layer_compute_done(0)


def test_full_pipeline_depth3_total5():
    s = ext.RollingPrefetchScheduler(3, 5)
    submitted = s.bootstrap()
    assert submitted == [0, 1, 2]

    next_after_done = []
    for layer in range(5):
        s.mark_ready(layer)
        s.begin_compute(layer)
        next_after_done.append(s.on_layer_compute_done(layer))

    # Compute(0) submits layer 3, Compute(1) submits 4, Compute(2..4) tail (-1)
    assert next_after_done == [3, 4, -1, -1, -1]
    for layer in range(5):
        assert s.state_of(layer) == _state().Released
    assert s.next_submit_layer == 5
    assert s.next_compute_layer == 5


def test_depth_exceeds_total_no_tail_resubmits():
    s = ext.RollingPrefetchScheduler(10, 3)
    s.bootstrap()
    nxts = []
    for layer in range(3):
        s.mark_ready(layer)
        s.begin_compute(layer)
        nxts.append(s.on_layer_compute_done(layer))
    assert nxts == [-1, -1, -1]


def test_depth_one_acts_as_serial_pipeline():
    s = ext.RollingPrefetchScheduler(1, 4)
    submitted = s.bootstrap()
    assert submitted == [0]
    nxts = []
    for layer in range(4):
        s.mark_ready(layer)
        s.begin_compute(layer)
        nxts.append(s.on_layer_compute_done(layer))
    assert nxts == [1, 2, 3, -1]


def test_total_one_single_layer():
    s = ext.RollingPrefetchScheduler(5, 1)
    assert s.bootstrap() == [0]
    s.mark_ready(0)
    s.begin_compute(0)
    assert s.on_layer_compute_done(0) == -1
    assert s.state_of(0) == _state().Released


def test_drain_returns_reading_layers_with_request_ids():
    s = ext.RollingPrefetchScheduler(3, 5)
    s.bootstrap()
    s.record_layer_requests(0, [10, 11])
    s.record_layer_requests(1, [20, 21, 22])
    s.record_layer_requests(2, [30])
    in_flight = s.drain_and_cancel()
    by_layer = {layer: list(ids) for layer, ids in in_flight}
    assert set(by_layer.keys()) == {0, 1, 2}
    assert by_layer[0] == [10, 11]
    assert by_layer[1] == [20, 21, 22]
    assert by_layer[2] == [30]
    for layer in range(5):
        assert s.state_of(layer) == _state().Released


def test_drain_after_complete_pipeline_returns_empty():
    s = ext.RollingPrefetchScheduler(2, 3)
    s.bootstrap()
    for layer in range(3):
        s.mark_ready(layer)
        s.begin_compute(layer)
        s.on_layer_compute_done(layer)
    assert s.drain_and_cancel() == []


def test_drain_mid_pipeline_only_reading_layers_returned():
    s = ext.RollingPrefetchScheduler(3, 6)
    s.bootstrap()
    s.record_layer_requests(0, [100])
    s.record_layer_requests(1, [200])
    s.record_layer_requests(2, [300])
    s.mark_ready(0)
    s.begin_compute(0)
    nxt = s.on_layer_compute_done(0)
    assert nxt == 3
    s.record_layer_requests(3, [400])
    # Now: 0=Released, 1=Reading, 2=Reading, 3=Reading, 4/5=Empty
    in_flight = s.drain_and_cancel()
    by_layer = {layer: list(ids) for layer, ids in in_flight}
    assert set(by_layer.keys()) == {1, 2, 3}
    assert by_layer[1] == [200]
    assert by_layer[2] == [300]
    assert by_layer[3] == [400]


def test_record_layer_requests_requires_reading_state():
    s = ext.RollingPrefetchScheduler(3, 5)
    s.bootstrap()
    s.mark_ready(0)
    # layer 0 is now Ready, not Reading
    with pytest.raises(RuntimeError):
        s.record_layer_requests(0, [1, 2])
    # Empty layer also rejected
    with pytest.raises(RuntimeError):
        s.record_layer_requests(3, [9])


def test_state_of_out_of_range_raises():
    s = ext.RollingPrefetchScheduler(3, 5)
    with pytest.raises(IndexError):
        s.state_of(-1)
    with pytest.raises(IndexError):
        s.state_of(5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
