from types import SimpleNamespace

from kt_kernel.sft.profiler import collect_kt_sft_profile, format_kt_sft_profile, reset_kt_sft_profile


class _FakeMoe:
    def __init__(self):
        self.reset_calls = 0
        self.get_reset_values = []

    def get_profile_stats(self, reset=False):
        self.get_reset_values.append(reset)
        return {
            "layer_idx": 3,
            "tp_count": 1,
            "wrapper.enabled": 1,
            "wrapper.tokens": 8,
            "wrapper.tp.forward.total.total_ns": 2_000_000,
            "wrapper.tp.forward.total.calls": 2,
            "wrapper.tp.forward.numa_compute.total_ns": 1_500_000,
            "wrapper.tp.forward.numa_compute.calls": 2,
            "wrapper.tp.forward.numa_compute.bytes": 2 * 1024 * 1024,
            "tp.0.enabled": 1,
            "tp.0.tokens": 8,
            "tp.0.forward.total.total_ns": 1_400_000,
            "tp.0.forward.total.calls": 2,
            "tp.0.forward.route.total_ns": 140_000,
            "tp.0.forward.route.calls": 2,
            "tp.0.backward.total.total_ns": 5_000_000,
            "tp.0.backward.total.calls": 1,
            "tp.0.backward.down.total.total_ns": 2_000_000,
            "tp.0.backward.down.total.calls": 1,
            "tp.0.backward.base_weight_grad.total_ns": 1_000_000,
            "tp.0.backward.base_weight_grad.calls": 1,
            "tp.0.backward.base_weight_grad.worker_cpu.store.total_ns": 2_000_000,
            "tp.0.backward.base_weight_grad.worker_cpu.store.calls": 4,
        }

    def reset_profile_stats(self):
        self.reset_calls += 1


def _fake_model(moe):
    backend = SimpleNamespace(moe=moe)
    layer = SimpleNamespace(layer_idx=3, wrapper=backend)
    return SimpleNamespace(_kt_wrappers=[layer])


def test_collect_and_format_profile():
    moe = _FakeMoe()
    profile = collect_kt_sft_profile(_fake_model(moe), reset=True)

    assert profile["enabled"] is True
    assert list(profile["layers"]) == [3]
    assert moe.get_reset_values == [True]

    output = format_kt_sft_profile(profile)
    assert "tp.forward.numa_compute" in output
    assert "forward.route" in output
    assert "75.0%" in output
    assert "2.00" in output
    assert "10.0%" in output
    assert "40.0%" in output
    worker_store = next(line for line in output.splitlines() if "worker_cpu.store" in line)
    assert worker_store.endswith("0.0%")


def test_reset_and_disabled_profile():
    moe = _FakeMoe()
    model = _fake_model(moe)
    reset_kt_sft_profile(model)
    assert moe.reset_calls == 1

    empty = collect_kt_sft_profile(SimpleNamespace(_kt_wrappers=[]))
    assert empty == {"enabled": False, "layers": {}}
    assert "disabled" in format_kt_sft_profile(empty)
