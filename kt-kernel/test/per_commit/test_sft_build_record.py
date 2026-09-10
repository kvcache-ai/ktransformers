# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default")


def test_build_record_captures_actual_files_and_never_overwrites(tmp_path):
    source, build = tmp_path / "source", tmp_path / "build"
    (source / "operators").mkdir(parents=True)
    (source / "bench").mkdir()
    (source / "operators" / "kernel.hpp").write_text("// kernel input\n")
    (source / "bench" / "unrelated.cpp").write_text("// not a production input\n")
    target = build / "CMakeFiles" / "kt_kernel_ext.dir"
    target.mkdir(parents=True)
    (build / "CMakeCache.txt").write_text("CMAKE_BUILD_TYPE:STRING=Release\nUNRELATED:STRING=ignored\n")
    (target / "flags.make").write_text("CXX_FLAGS = -O3\n")
    (target / "link.txt").write_text("c++ -shared\n")
    extension, archive = tmp_path / "extension.so", tmp_path / "source.tar.gz"
    extension.write_bytes(b"binary fixture")
    archive.write_bytes(b"archive fixture")
    output = tmp_path / "record.json"
    script = Path(__file__).resolve().parents[2] / "bench" / "sft_perf" / "record_build.py"
    command = [
        sys.executable,
        str(script),
        "--source",
        str(source),
        "--build",
        str(build),
        "--extension",
        str(extension),
        "--archive",
        str(archive),
        "--revision",
        "test-revision",
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True)
    before = output.read_bytes()
    result = json.loads(before)
    assert result["extension_sha256"] == hashlib.sha256(b"binary fixture").hexdigest()
    assert result["selected_cmake_cache"] == {"CMAKE_BUILD_TYPE": "Release"}
    assert list(result["production_source_sha256_at_record_time"]) == ["operators/kernel.hpp"]
    repeated = subprocess.run(command, capture_output=True)
    assert repeated.returncode != 0
    assert output.read_bytes() == before


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
