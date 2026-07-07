import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path

import click

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=0.1, suite="default")


KT_KERNEL_PATH = Path(__file__).resolve().parents[2] / "python"
kt_kernel = types.ModuleType("kt_kernel")
kt_kernel.__path__ = [str(KT_KERNEL_PATH)]
sys.modules.setdefault("kt_kernel", kt_kernel)

RUN_PATH = KT_KERNEL_PATH / "cli" / "commands" / "run.py"
SPEC = importlib.util.spec_from_file_location("run_command", RUN_PATH)
assert SPEC is not None and SPEC.loader is not None
run_command = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_command)


class Settings:
    def __init__(self, sglang_args):
        self.sglang_args = sglang_args

    def get(self, key, default=None):
        if key == "advanced.sglang_args":
            return self.sglang_args
        return default


def build_command(sglang_args):
    return run_command._build_sglang_command(
        model_path=Path("/tmp/model"),
        weights_path=None,
        host="0.0.0.0",
        port=30000,
        gpu_experts=1,
        cpu_threads=1,
        numa_nodes=1,
        tensor_parallel_size=1,
        kt_method="AMXINT4",
        kt_gpu_prefill_threshold=4096,
        attention_backend="flashinfer",
        max_total_tokens=40000,
        max_running_requests=32,
        chunked_prefill_size=4096,
        mem_fraction_static=0.98,
        watchdog_timeout=3000,
        served_model_name="",
        disable_shared_experts_fusion=False,
        kt_numa_nodes=None,
        tool_call_parser=None,
        reasoning_parser=None,
        settings=Settings(sglang_args),
    )


class TestRunSglangArgs(unittest.TestCase):
    def test_string_sglang_args_are_split_into_argv_tokens(self):
        cmd = build_command("--log-level warning")

        self.assertEqual(cmd[-2:], ["--log-level", "warning"])
        self.assertNotEqual(cmd[-19:], list("--log-level warning"))

    def test_list_sglang_args_are_appended_unchanged(self):
        cmd = build_command(["--log-level", "warning"])

        self.assertEqual(cmd[-2:], ["--log-level", "warning"])

    def test_invalid_sglang_args_type_is_rejected(self):
        with self.assertRaises(click.BadParameter):
            build_command({"log-level": "warning"})


if __name__ == "__main__":
    unittest.main()
