# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import patch


CONFIG_PATH = Path(__file__).resolve().parents[2] / "python" / "sft" / "config.py"
SPEC = importlib.util.spec_from_file_location("kt_sft_config_under_test", CONFIG_PATH)
assert SPEC is not None and SPEC.loader is not None
config = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = config
SPEC.loader.exec_module(config)


def test_detect_physical_cpu_count_deduplicates_smt_siblings():
    topology = {
        0: (0, 0),
        1: (0, 1),
        2: (0, 0),
        3: (0, 1),
        4: (1, 0),
        5: (1, 0),
    }
    with (
        patch.object(config, "_available_cpu_ids", return_value=set(topology)),
        patch.object(config, "_read_cpu_topology", side_effect=topology.get),
    ):
        assert config.detect_physical_cpu_count() == 3


def test_configure_omp_threads_replaces_accelerate_single_thread_default():
    with (
        patch.dict(os.environ, {"OMP_NUM_THREADS": "1"}, clear=False),
        patch.object(config, "detect_physical_cpu_count", return_value=96),
        patch.object(config, "_set_torch_num_threads") as set_torch_threads,
    ):
        os.environ.pop("ACCELERATE_KT_OMP_NUM_THREADS", None)
        assert config.configure_omp_threads() == 96
        assert os.environ["OMP_NUM_THREADS"] == "96"
        set_torch_threads.assert_called_once_with(96)


def test_configure_omp_threads_preserves_explicit_generic_value():
    with (
        patch.dict(os.environ, {"OMP_NUM_THREADS": "48"}, clear=False),
        patch.object(config, "_set_torch_num_threads") as set_torch_threads,
    ):
        os.environ.pop("ACCELERATE_KT_OMP_NUM_THREADS", None)
        assert config.configure_omp_threads() == 48
        set_torch_threads.assert_called_once_with(48)


def test_configure_omp_threads_supports_explicit_single_thread_override():
    with (
        patch.dict(
            os.environ,
            {"OMP_NUM_THREADS": "96", "ACCELERATE_KT_OMP_NUM_THREADS": "1"},
            clear=False,
        ),
        patch.object(config, "_set_torch_num_threads") as set_torch_threads,
    ):
        assert config.configure_omp_threads() == 1
        assert os.environ["OMP_NUM_THREADS"] == "1"
        set_torch_threads.assert_called_once_with(1)
