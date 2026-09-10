# SPDX-License-Identifier: Apache-2.0
import importlib.util
from contextlib import closing
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default")

_directory = Path(__file__).resolve().parents[2] / "bench" / "sft_perf"
sys.path.insert(0, str(_directory))
try:
    _spec = importlib.util.spec_from_file_location("sft_trace_report", _directory / "trace_report.py")
    report = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(report)
finally:
    sys.path.pop(0)


def test_interval_union_does_not_add_parallel_time():
    assert report.union_ns([(0, 10), (5, 20), (3, 8), (30, 40)]) == 30
    assert report.union_ns([]) == 0
    assert report.union_ns([(7, 7), (10, 0)]) == 0


def test_overlap_clips_boundaries_and_separates_nccl():
    kernels = [
        {"start": 0, "end": 12, "deviceId": 0, "name": "gemm"},
        {"start": 0, "end": 12, "deviceId": 1, "name": "gemm"},
        {"start": 18, "end": 25, "deviceId": 0, "name": "attention"},
        {"start": 10, "end": 20, "deviceId": 0, "name": "ncclDevKernel"},
        {"start": 25, "end": 30, "deviceId": 0, "name": "outside"},
    ]
    result = report.summarize_range({"start": 10, "end": 20}, kernels)
    assert result["any_gpu_other_kernel_union_ns"] == 4
    assert result["any_gpu_other_kernel_overlap_fraction"] == 0.4
    assert result["per_device_union_ns"][0] == {"other": 4, "nccl_named": 10}
    assert result["per_device_union_ns"][1] == {"other": 2}
    assert len(result["clipped_kernel_events"]) == 4


def test_incomplete_cpu_interval_is_not_a_timing_sample():
    with pytest.raises(ValueError, match="positive-duration"):
        report.summarize_range({"start": 10, "end": 10}, [])


def test_missing_database_is_not_created(tmp_path):
    path = tmp_path / "absent.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        report.read_trace(path)
    assert not path.exists()


@pytest.fixture
def capture(tmp_path):
    path = tmp_path / "capture.sqlite"
    with closing(sqlite3.connect(path)) as database:
        database.executescript("""
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE NVTX_EVENTS (
                start INTEGER, end INTEGER, globalTid INTEGER, domainId INTEGER,
                int64Value INTEGER, category INTEGER, text TEXT, textId INTEGER,
                eventType INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER, end INTEGER, deviceId INTEGER, streamId INTEGER,
                globalPid INTEGER, demangledName INTEGER
            );
            INSERT INTO StringIds VALUES (1, 'kt.sft'), (2, 'repack.async'), (3, 'gemm');
            INSERT INTO NVTX_EVENTS VALUES
                (0, NULL, 16777217, 7, NULL, NULL, NULL, 1, 75),
                (10, 20, 16777218, 7, 59, 1, NULL, 2, 60),
                (0, NULL, 33554433, 7, NULL, NULL, 'unrelated', NULL, 75),
                (30, 40, 33554434, 7, 58, 1, NULL, 2, 60);
        """)
        database.commit()
    return path


def add_kernel(capture, start, end):
    with closing(sqlite3.connect(capture)) as database:
        database.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 0, 1, 16777216, 3)", (start, end))
        database.commit()


def test_empty_cuda_table_is_not_zero_overlap(capture):
    with pytest.raises(ValueError, match="no CUDA kernel events"):
        report.read_trace(capture)


def test_missing_cuda_table_is_not_zero_overlap(capture):
    with closing(sqlite3.connect(capture)) as database:
        database.execute("DROP TABLE CUPTI_ACTIVITY_KIND_KERNEL")
        database.commit()
    with pytest.raises(ValueError, match="lacks required.*CUPTI_ACTIVITY_KIND_KERNEL"):
        report.read_trace(capture)


def test_reader_matches_nvtx_domain_with_process_identity(capture):
    add_kernel(capture, 12, 18)
    result = report.read_trace(capture)
    assert result["devices_with_kernel_events"] == [0]
    assert result["kernel_events_in_capture"] == 1
    assert len(result["ranges"]) == 1
    assert result["ranges"][0]["cpu"]["layer"] == 59
    assert result["ranges"][0]["any_gpu_other_kernel_overlap_fraction"] == 0.6


def test_observed_cuda_activity_outside_cpu_range_can_have_zero_overlap(capture):
    add_kernel(capture, 21, 25)
    result = report.read_trace(capture)
    assert result["kernel_events_in_capture"] == 1
    assert result["ranges"][0]["any_gpu_other_kernel_overlap_fraction"] == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
