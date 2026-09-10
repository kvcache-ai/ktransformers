# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

PACKAGE_PATH = Path(__file__).resolve().parents[2] / "python" / "sft"
PACKAGE_NAME = "kt_sft_config_under_test"
package = types.ModuleType(PACKAGE_NAME)
package.__path__ = [str(PACKAGE_PATH)]
sys.modules[PACKAGE_NAME] = package
SPEC = importlib.util.spec_from_file_location(f"{PACKAGE_NAME}.config", PACKAGE_PATH / "config.py")
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
        patch.dict(os.environ, {"OMP_NUM_THREADS": "1", "RANK": "0", "WORLD_SIZE": "1"}, clear=False),
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


def test_configure_omp_threads_gives_machine_budget_only_to_cpu_owner():
    for rank, expected in (("0", 96), ("1", 1), ("7", 1)):
        with (
            patch.dict(os.environ, {"OMP_NUM_THREADS": "1", "RANK": rank, "WORLD_SIZE": "8"}, clear=False),
            patch.object(config, "detect_physical_cpu_count", return_value=96),
            patch.object(config, "_set_torch_num_threads") as set_torch_threads,
        ):
            os.environ.pop("ACCELERATE_KT_OMP_NUM_THREADS", None)
            assert config.configure_omp_threads() == expected
            assert config.configure_omp_threads() == expected
            assert set_torch_threads.call_count == 2
            set_torch_threads.assert_called_with(expected)


def test_configure_omp_threads_preserves_non_owner_explicit_overrides():
    for explicit, generic, expected in ((None, "24", 24), ("12", "1", 12), ("1", "96", 1)):
        with (
            patch.dict(os.environ, {"OMP_NUM_THREADS": generic, "RANK": "3", "WORLD_SIZE": "8"}, clear=False),
            patch.object(config, "_set_torch_num_threads") as set_torch_threads,
        ):
            if explicit is None:
                os.environ.pop("ACCELERATE_KT_OMP_NUM_THREADS", None)
            else:
                os.environ["ACCELERATE_KT_OMP_NUM_THREADS"] = explicit
            assert config.configure_omp_threads() == expected
            set_torch_threads.assert_called_once_with(expected)
