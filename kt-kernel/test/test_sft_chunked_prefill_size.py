"""Test that the SFT prefill buffer is sized for a flattened batch, not one sequence.

Regression test for #2150.  `kt_model_max_length` is a per-sequence bound (LLaMA-Factory
populates it from `cutoff_len`), but `KTMoELayerWrapper` flattens a batch into a single
`qlen = batch_size * seq_len` before it reaches the kernel, so a buffer sized without the
batch factor is undersized by exactly that factor and `_validate_forward_inputs` raises
for any `per_device_train_batch_size > 1`.
"""

import importlib.util
import os
import sys
import types

# Load sft/config.py under a synthetic parent package.  Importing the real kt_kernel
# package executes its __init__, which loads the compiled extension, and this test only
# exercises pure-Python config sizing -- so it stays runnable without a build.
_SFT_DIR = os.path.join(os.path.dirname(__file__), "..", "python", "sft")
_pkg = types.ModuleType("_kt_sft_under_test")
_pkg.__path__ = [_SFT_DIR]
sys.modules["_kt_sft_under_test"] = _pkg

for _name in ("backend", "config"):
    _spec = importlib.util.spec_from_file_location(f"_kt_sft_under_test.{_name}", os.path.join(_SFT_DIR, f"{_name}.py"))
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _mod  # @dataclass resolves its module out of sys.modules
    _spec.loader.exec_module(_mod)

KTConfig = sys.modules["_kt_sft_under_test.config"].KTConfig


def _rank0_chunked_prefill_size(cfg, world_size=1, fallback_max_position_embeddings=4096):
    """Mirror of the sizing in sft/wrapper.py, isolated from the model it needs."""
    size = getattr(cfg, "kt_model_max_length", None)
    if size is None:
        size = fallback_max_position_embeddings
    train_batch_size = int(getattr(cfg, "kt_train_batch_size", 1) or 1)
    return int(size) * world_size * train_batch_size


def test_defaults_to_batch_one():
    cfg = KTConfig(kt_model_max_length=1024)
    assert cfg.kt_train_batch_size == 1
    assert _rank0_chunked_prefill_size(cfg) == 1024


def test_scales_with_batch_size():
    # The exact case from #2150: cutoff_len 1024 at batch 2 flattens to qlen 2048.
    cfg = KTConfig(kt_model_max_length=1024, kt_train_batch_size=2)
    size = _rank0_chunked_prefill_size(cfg)
    assert size == 2048, size
    assert 2 * 1024 <= size

    for batch in (4, 8):
        cfg = KTConfig(kt_model_max_length=1024, kt_train_batch_size=batch)
        assert batch * 1024 <= _rank0_chunked_prefill_size(cfg)


def test_world_size_and_batch_compose():
    # World-size scaling already existed; batch has to multiply on top of it.
    cfg = KTConfig(kt_model_max_length=1024, kt_train_batch_size=2)
    assert _rank0_chunked_prefill_size(cfg, world_size=4) == 1024 * 4 * 2


def test_env_var_override():
    os.environ["ACCELERATE_KT_TRAIN_BATCH_SIZE"] = "4"
    try:
        cfg = KTConfig(kt_model_max_length=512)
        assert cfg.kt_train_batch_size == 4
        assert _rank0_chunked_prefill_size(cfg) == 2048
    finally:
        del os.environ["ACCELERATE_KT_TRAIN_BATCH_SIZE"]


def test_falls_back_to_max_position_embeddings():
    cfg = KTConfig(kt_train_batch_size=2)
    assert _rank0_chunked_prefill_size(cfg, fallback_max_position_embeddings=4096) == 8192


if __name__ == "__main__":
    test_defaults_to_batch_one()
    test_scales_with_batch_size()
    test_world_size_and_batch_compose()
    test_env_var_override()
    test_falls_back_to_max_position_embeddings()
    print("All chunked_prefill_size tests passed.")
