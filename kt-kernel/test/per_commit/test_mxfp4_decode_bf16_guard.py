"""CPU-only control-flow regression for MXFP4 decode input preparation (#2192).

Compile the production method with buffer stubs, with/without the BF16 macro
and NDEBUG. No SIMD instructions, model weights, or KT extension are required;
this checks the buffer handoff, not GEMM numerics or hardware compatibility.
"""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="default")

HEADER = Path(__file__).resolve().parents[2] / "operators/amx/fp4-moe.hpp"


def _production_method():
    source = HEADER.read_text(encoding="utf-8")
    signature = "void prepare_decode_down_input(int expert_idx, int qlen) {"
    if source.count(signature) != 1:
        raise AssertionError("Expected exactly one MXFP4 decode input preparation method")
    start = source.index(signature)
    depth = 0
    for offset in range(source.index("{", start), len(source)):
        if source[offset] == "{":
            depth += 1
        elif source[offset] == "}":
            depth -= 1
            if depth == 0:
                return source[start : offset + 1]
    raise AssertionError("Unterminated MXFP4 decode input preparation method")


HARNESS = r"""
#include <cassert>
#include <cstdlib>
#include <iostream>

struct Buffer {
  bool natural_order = false;
  int value = -777;
  int copies = 0;
  void from_mat(int, int* source, int, int) {
    value = *source;
    ++copies;
    natural_order = false;
  }
};

struct MockBase {
  struct Config {
    struct QuantConfig { int group_size = 32; } quant_config;
    int intermediate_size = 2048;
  } config_;
  Buffer buffers[2];
  Buffer* down_ba_[2] = {&buffers[0], &buffers[1]};
  int activation[2] = {};
  int* m_local_gate_output_ptr_[2] = {&activation[0], &activation[1]};

  void prepare_decode_down_input(int expert_idx, int qlen) {
    down_ba_[expert_idx]->from_mat(qlen, m_local_gate_output_ptr_[expert_idx], 0, 1);
  }
};

struct Harness : MockBase {
  using Base = MockBase;
  /* PRODUCTION_METHOD */
};

int run_round(Harness& moe, int qlen, int round) {
  bool direct_write = false;
#if defined(__AVX512BF16__)
  direct_write = qlen == 1 && moe.config_.quant_config.group_size == 32 &&
                 moe.config_.intermediate_size % 32 == 0;
#endif
  for (int expert_idx = 0; expert_idx < 2; ++expert_idx) {
    auto& buffer = moe.buffers[expert_idx];
    // Model BufferA::set_data(): reset the layout flag, but retain old data.
    buffer.natural_order = false;
    const int copies_before = buffer.copies;
    const int expected = 100 * round + expert_idx;
    if (direct_write) {
      buffer.value = expected;
      buffer.natural_order = true;
      moe.activation[expert_idx] = -999;  // Copying here would corrupt the fast path.
    } else {
      moe.activation[expert_idx] = expected;
    }
    moe.prepare_decode_down_input(expert_idx, qlen);
    // Do not use assert: Release/NDEBUG must check the data, too.
    if (buffer.value != expected ||
        buffer.copies != copies_before + (direct_write ? 0 : 1) ||
        buffer.natural_order != direct_write) {
      std::cerr << "round=" << round << " expert=" << expert_idx
                << " qlen=" << qlen << " expected=" << expected
                << " actual=" << buffer.value
                << " copies=" << buffer.copies - copies_before << '\n';
      return 1;
    }
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc != 4) return 2;
  Harness moe;
  const int qlen = std::atoi(argv[1]);
  moe.config_.quant_config.group_size = std::atoi(argv[2]);
  moe.config_.intermediate_size = std::atoi(argv[3]);
  // Exercise multi-token -> requested shape -> repeated call with new values.
  return run_round(moe, 2, 1) || run_round(moe, qlen, 2) || run_round(moe, qlen, 3);
}
"""


class TestMXFP4DecodeBF16Guard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        if not compiler or shutil.which(compiler[0]) is None:
            raise RuntimeError("This regression test requires a C++17 compiler (set CXX)")
        temporary = tempfile.TemporaryDirectory(prefix="kt-mxfp4-decode-guard-")
        cls.addClassCleanup(temporary.cleanup)
        cls.build_dir = Path(temporary.name)
        source = cls.build_dir / "decode_guard.cpp"
        source.write_text(
            HARNESS.replace("/* PRODUCTION_METHOD */", _production_method()),
            encoding="utf-8",
        )
        cls.binaries = {}
        for bf16 in (False, True):
            for release in (False, True):
                name = f"bf16_{int(bf16)}_release_{int(release)}"
                binary = cls.build_dir / name
                flags = ["-std=c++17", "-Wall", "-Wextra", "-Werror", "-U__AVX512BF16__", "-UNDEBUG"]
                if bf16:
                    # Simulate only the compile-time branch, not a CPU ISA.
                    flags.append("-D__AVX512BF16__")
                flags += ["-O2", "-DNDEBUG"] if release else ["-O0"]
                result = subprocess.run(
                    compiler + flags + [str(source), "-o", str(binary)],
                    capture_output=True, text=True, cwd=cls.build_dir, timeout=60,
                )
                if result.returncode:
                    raise AssertionError(f"{name} failed to compile:\n{result.stdout}\n{result.stderr}")
                cls.binaries[name] = binary

    def _check_shape(self, qlen, group_size, intermediate_size):
        for name, binary in self.binaries.items():
            with self.subTest(build=name):
                result = subprocess.run(
                    [str(binary), str(qlen), str(group_size), str(intermediate_size)],
                    capture_output=True, text=True, cwd=self.build_dir, timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_single_token_decode_and_reused_buffers(self):
        self._check_shape(1, 32, 2048)

    def test_multi_token_fallback(self):
        self._check_shape(2, 32, 2048)

    def test_other_group_size_fallback(self):
        self._check_shape(1, 64, 2048)

    def test_unaligned_intermediate_size_fallback(self):
        self._check_shape(1, 32, 48)


if __name__ == "__main__":
    unittest.main()
