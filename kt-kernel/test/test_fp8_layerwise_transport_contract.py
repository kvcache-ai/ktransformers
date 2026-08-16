"""CPU-only contract guards for the native FP8 layerwise transport."""

from __future__ import annotations

import ast
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace


AMX_PATH = Path(__file__).resolve().parents[1] / "python/utils/amx.py"


def _compile_thin_method():
    tree = ast.parse(AMX_PATH.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NativeMoEWrapper")
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "run_layerwise_fp8_batch"
    )
    function = copy.deepcopy(method)
    function.name = "invoke"
    function.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(AMX_PATH), "exec"), namespace)
    return namespace["invoke"]


class _MoeRecorder:
    def __init__(self):
        self.calls = []

    def run_layerwise_fp8_batch(self, *args):
        self.calls.append(args)
        return "native-result"


class _CPUInferRecorder:
    def __init__(self):
        self.sync_count = 0

    def sync(self):
        self.sync_count += 1


class TestFP8LayerwiseTransportContract(unittest.TestCase):
    def test_thin_wrapper_forwards_exact_layer_contract(self):
        invoke = _compile_thin_method()
        moe = _MoeRecorder()
        cpu_infer = _CPUInferRecorder()
        wrapper = SimpleNamespace(method="FP8", moe=moe, cpu_infer=cpu_infer)
        transport = object()

        result = invoke(wrapper, transport, 7, 12, 288)

        self.assertEqual(result, "native-result")
        self.assertEqual(cpu_infer.sync_count, 1)
        self.assertEqual(moe.calls, [(transport, 7, 12, 288)])

    def test_thin_wrapper_rejects_non_block_fp8(self):
        invoke = _compile_thin_method()
        wrapper = SimpleNamespace(method="BF16", moe=_MoeRecorder())

        with self.assertRaisesRegex(RuntimeError, "only valid for the block-FP8"):
            invoke(wrapper, object(), 1, 0, 1)

    def test_native_source_keeps_hot_protocol_out_of_python(self):
        source = AMX_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NativeMoEWrapper")
        method = next(
            node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "run_layerwise_fp8_batch"
        )
        self.assertFalse(any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(method)))


if __name__ == "__main__":
    unittest.main()
