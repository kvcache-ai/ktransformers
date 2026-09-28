"""Verify that a second Python ABI keeps the same five-package CUDA payload."""
import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import carriers
import supplement
from four_main import sha256
from test_carriers import raw_wheels


def native_inputs(root, python):
    root.mkdir()
    raw = raw_wheels(root, python=python)
    for package, module in [('kt-kernel', 'kt_kernel'), ('sgl-kernel-kt', 'mscclpp')]:
        tree = root / package
        native = tree / module / ('extension.cpython-' + python[2:] + '-x86_64-linux-gnu.so')
        native.parent.mkdir(exist_ok=True)
        native.write_bytes(b'native-' + python.encode())
        carriers.pack(tree, next(raw.glob(package.replace('-', '_') + '-*.whl')))
    return raw


def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(carriers, 'binary_evidence', lambda _: {})
    monkeypatch.setattr(supplement, 'needed_libraries', lambda _: set())
    old = native_inputs(tmp_path / 'cp312', 'cp312')
    new = native_inputs(tmp_path / 'cp311', 'cp311')
    evidence = tmp_path / 'evidence'
    evidence.mkdir()
    reference = tmp_path / 'reference'
    carriers.assemble(old, reference, evidence)
    kt = next(new.glob('kt_kernel-*.whl'))
    sgl = next(new.glob('sgl_kernel_kt-*.whl'))
    lock = {
        'reference_wheels': {p.name: sha256(p) for p in reference.glob('*.whl')},
        'native_wheels': {p.name: sha256(p) for p in [kt, sgl]},
        'source_lock': {'scope': 'synthetic unit fixture'},
    }
    return reference, kt, sgl, lock


def test_second_abi_preserves_shared_runtime_and_payload(tmp_path, monkeypatch):
    reference, kt, sgl, lock = inputs(tmp_path, monkeypatch)
    wheel = supplement.supplement(reference, kt, sgl, lock, tmp_path / 'output', tmp_path / 'supplement-evidence')
    assert '-cp311-cp311-' in wheel.name
    assert {p.name: sha256(p) for p in reference.glob('*.whl')} == lock['reference_wheels']
    with zipfile.ZipFile(next(reference.glob('kt_kernel-*.whl'))) as old, zipfile.ZipFile(wheel) as new:
        replaced = [n for n in old.namelist() if '.cpython-312-' in n]
        assert len(replaced) == 2
        assert all(n not in new.namelist() for n in replaced)
        for name in old.namelist():
            if '.cpython-312-' not in name and not name.endswith(('/WHEEL', '/RECORD')):
                assert new.read(name) == old.read(name), name
        assert len([n for n in new.namelist() if '.cpython-311-' in n]) == 2
    report = json.loads((tmp_path / 'supplement-evidence/supplement.json').read_text())
    assert report['archive_sha256']
    assert len(report['retained_payload_parts']) == 5


def test_second_abi_rejects_runtime_drift(tmp_path, monkeypatch):
    reference, kt, sgl, lock = inputs(tmp_path, monkeypatch)
    tree = tmp_path / 'cp311/kt-kernel'
    (tree / 'kt_kernel/__init__.py').write_text('CHANGED = True\n')
    carriers.pack(tree, kt)
    lock['native_wheels'][kt.name] = sha256(kt)
    with pytest.raises(ValueError, match='Runtime file changed'):
        supplement.supplement(reference, kt, sgl, lock, tmp_path / 'output', tmp_path / 'supplement-evidence')


def test_second_abi_rejects_mixed_reference_carriers(tmp_path, monkeypatch):
    reference, kt, sgl, lock = inputs(tmp_path, monkeypatch)
    with next(reference.glob('transformers*.whl')).open('ab') as f:
        f.write(b'unexpected bytes')
    with pytest.raises(ValueError, match='Reference checksum mismatch'):
        supplement.supplement(reference, kt, sgl, lock, tmp_path / 'output', tmp_path / 'supplement-evidence')
