"""Backend selection tests without importing GPU extensions or torch."""

import ast
import os
from pathlib import Path
from unittest.mock import patch

import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default")

SOURCE = Path(__file__).parents[2] / "python/utils/amx.py"


def select(legacy=object(), xmx=object()):
    tree = ast.parse(SOURCE.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_select_sycl_int4_backend")
    scope = {"os": os, "SYCLGPTQInt4_MOE": legacy, "SYCLXMXGPTQInt4_MOE": xmx}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope["_select_sycl_int4_backend"]


def test_default_remains_legacy():
    legacy = object()
    with patch.dict(os.environ, {}, clear=True):
        assert select(legacy=legacy)() is legacy


def test_xmx_is_explicit_and_normalized():
    xmx = object()
    with patch.dict(os.environ, {"KT_SYCL_INT4_BACKEND": " XMX "}):
        assert select(xmx=xmx)() is xmx


@pytest.mark.parametrize("value", ["auto", "typo", ""])
def test_invalid_selection_is_not_a_fallback(value):
    with patch.dict(os.environ, {"KT_SYCL_INT4_BACKEND": value}):
        with pytest.raises(ValueError):
            select()()


def test_uncompiled_xmx_is_not_a_fallback():
    with patch.dict(os.environ, {"KT_SYCL_INT4_BACKEND": "xmx"}):
        with pytest.raises(RuntimeError, match="not compiled"):
            select(xmx=None)()


@pytest.mark.parametrize("profile", ["", "auto", "typo", "Panther-Lake"])
def test_invalid_profile_does_not_silently_select_legacy(profile):
    with patch.dict(os.environ, {"KT_SYCL_XMX_PROFILE": profile}, clear=True):
        with pytest.raises(ValueError, match="KT_SYCL_XMX_PROFILE"):
            select()()


def test_profile_selects_xmx_without_setting_other_environment():
    class XMX:
        @staticmethod
        def implementation_revision():
            return 7

    with patch.dict(os.environ, {"KT_SYCL_XMX_PROFILE": "panther-lake"}, clear=True):
        before = dict(os.environ)
        assert select(xmx=XMX)() is XMX
        assert dict(os.environ) == before


def test_profile_rejects_conflicting_backend():
    with patch.dict(os.environ, {"KT_SYCL_XMX_PROFILE": "panther-lake", "KT_SYCL_INT4_BACKEND": "legacy"}, clear=True):
        with pytest.raises(ValueError, match="requires the xmx backend"):
            select()()


@pytest.mark.parametrize("revision", [0, 6])
def test_profile_rejects_old_extension(revision):
    class XMX:
        @staticmethod
        def implementation_revision():
            return revision

    with patch.dict(os.environ, {"KT_SYCL_XMX_PROFILE": "panther-lake"}, clear=True):
        with pytest.raises(RuntimeError, match="revision >= 7"):
            select(xmx=XMX)()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
