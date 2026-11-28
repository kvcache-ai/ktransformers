# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import safetensors
import torch

from kt_kernel.sft import checkpoint

_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_SHAPES = {"gate_proj": (2, 3, 4), "up_proj": (2, 3, 4), "down_proj": (2, 4, 3)}


def _wrapper(layer, fill):
    params = {
        f"{name}_buf": torch.nn.Parameter(torch.full(_SHAPES[name], fill + i, dtype=torch.bfloat16))
        for i, name in enumerate(_PROJECTIONS)
    }
    return SimpleNamespace(
        layer_idx=layer,
        hidden_size=4,
        moe_config=SimpleNamespace(expert_num=2, intermediate_size=3),
        _full_weight_grad=True,
        wrapper=SimpleNamespace(**params, _base_weights_dirty=False, _kt_full_checkpoint_load_failed=False),
    )


def _parameters(wrapper):
    return [getattr(wrapper.wrapper, f"{name}_buf") for name in _PROJECTIONS]


def _assert_layer(tensors, fill):
    for value, tensor in enumerate(tensors, fill):
        torch.testing.assert_close(tensor, torch.full_like(tensor, value))


@pytest.mark.skipif(os.name != "posix", reason="requires unlinking already open checkpoint shards")
@pytest.mark.parametrize("full_resume", [True, False], ids=["full-resume", "single-layer-import"])
def test_validated_shards_survive_checkpoint_rollover(tmp_path, monkeypatch, full_resume):
    output = str(tmp_path)
    assert checkpoint.save_full_weight_checkpoint([_wrapper(0, 10), _wrapper(1, 20)], output)
    index_path = tmp_path / checkpoint.FULL_WEIGHT_INDEX_NAME
    before = json.loads(index_path.read_text())
    old_shards = [tmp_path / row["file"] for row in before["layers"].values()]
    last_old = tmp_path / before["layers"]["1"]["file"]
    real_open = safetensors.safe_open
    switched = False

    @contextmanager
    def open_with_rollover(path, *args, **kwargs):
        with real_open(path, *args, **kwargs) as inner:

            class Reader:
                def __getattr__(self, name):
                    return getattr(inner, name)

                def get_slice(self, name):
                    nonlocal switched
                    tensor_slice = inner.get_slice(name)
                    if not switched and Path(path) == last_old and name == "down_proj":
                        switched = True
                        assert checkpoint.save_full_weight_checkpoint([_wrapper(0, 110), _wrapper(1, 120)], output)
                        assert json.loads(index_path.read_text())["generation"] != before["generation"]
                        assert not any(file.exists() for file in old_shards)
                    return tensor_slice

            yield Reader()

    monkeypatch.setattr(safetensors, "safe_open", open_with_rollover)
    if full_resume:
        readers = [_wrapper(0, -1), _wrapper(1, -1)]
        params = [_parameters(reader) for reader in readers]
        pointers = [[param.data_ptr() for param in layer] for layer in params]
        assert checkpoint.load_full_weight_checkpoint(readers, output)
        for reader, fill in zip(readers, (10, 20)):
            _assert_layer(_parameters(reader), fill)
            assert reader.wrapper._base_weights_dirty
            assert not reader.wrapper._kt_full_checkpoint_load_failed
        assert [[param.data_ptr() for param in _parameters(reader)] for reader in readers] == pointers
        assert checkpoint.load_full_weight_checkpoint(readers, output)
        for reader, fill in zip(readers, (110, 120)):
            _assert_layer(_parameters(reader), fill)
    else:

        def read_layer():
            return checkpoint.load_full_weight_layer(output, layer_idx=1, expected_shapes=_SHAPES)

        _assert_layer(read_layer(), 20)
        _assert_layer(read_layer(), 120)
    assert switched


def test_quiescent_full_checkpoint_restores_without_changing_index(tmp_path):
    restored = [_wrapper(0, 0), _wrapper(2, 0)]
    assert checkpoint.save_full_weight_checkpoint([_wrapper(0, 4), _wrapper(2, 7)], str(tmp_path))
    index = tmp_path / checkpoint.FULL_WEIGHT_INDEX_NAME
    before = index.read_bytes()
    assert checkpoint.load_full_weight_checkpoint(restored, str(tmp_path))
    for reader, fill in zip(restored, (4, 7)):
        _assert_layer(_parameters(reader), fill)
    assert index.read_bytes() == before


def test_later_invalid_shard_closes_readers_without_partial_restore(tmp_path, monkeypatch):
    assert checkpoint.save_full_weight_checkpoint([_wrapper(0, 10), _wrapper(1, 20)], str(tmp_path))
    index_path = tmp_path / checkpoint.FULL_WEIGHT_INDEX_NAME
    index = json.loads(index_path.read_text())
    index["layers"]["1"]["shapes"]["gate_proj"][0] += 1
    index_path.write_text(json.dumps(index))
    real_open = safetensors.safe_open
    still_open = set()

    @contextmanager
    def track_reader(path, *args, **kwargs):
        with real_open(path, *args, **kwargs) as handle:
            still_open.add(path)
            try:
                yield handle
            finally:
                still_open.remove(path)

    monkeypatch.setattr(safetensors, "safe_open", track_reader)
    readers = [_wrapper(0, -5), _wrapper(1, -10)]
    with pytest.raises(RuntimeError, match="Layer 1: KT Full checkpoint index shapes mismatch"):
        checkpoint.load_full_weight_checkpoint(readers, str(tmp_path))
    assert not still_open
    for reader, fill in zip(readers, (-5, -10)):
        _assert_layer(_parameters(reader), fill)
        assert not reader.wrapper._base_weights_dirty
        assert not reader.wrapper._kt_full_checkpoint_load_failed
