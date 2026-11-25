#!/usr/bin/env python
# coding=utf-8
"""Intel XPU stream-bridge tests for KT-Kernel.

Exercises ``CPUInfer.submit_with_cuda_stream`` / ``sync_with_cuda_stream`` on a torch XPU stream, which
``cpu_backend/xpu_stream_bridge.cpp`` implements on top of SYCL / Level Zero:

* ``KT_XPU_BRIDGE`` / ``KT_XPU_BRIDGE_EVENTS`` parsing, caching, and re-reading after shutdown;
* the two ordering edges of {D2H copy -> CPU MoE -> H2D copy} against a torch reference, each
  isolated so that neither passes by timing coincidence (see ``_MoeUnderTest.stage_and_run``);
* Level Zero event-ring wrap-around, with the smallest legal ring and no host synchronisation;
* the ``allow_n_pending`` drain the deferred-expert path in ``experts_base.submit_forward`` uses;
* asynchrony -- a bridge that ran the CPU task inline would satisfy every correctness check here;
* exactly one worker thread per ``sycl::queue`` in ``l0`` mode, none in ``host_task`` mode and none
  under XPUGraph capture, and none left behind by ``shutdown_xpu_stream_bridge()``;
* XPU graph capture / replay;
* ``kt_kernel.current_stream_handle("xpu")`` against real ``torch.xpu`` streams.

Requires an XPU build of the extension (``CPUINFER_USE_XPU=1``) and an Intel GPU; skipped otherwise.
"""

import os
import sys
import time

import pytest

# Add parent directory to path for CI registration
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_xpu_ci

if os.environ.get("KT_KERNEL_EXT_DIR"):
    sys.path.insert(0, os.environ["KT_KERNEL_EXT_DIR"])

# Register this test for XPU CI with estimated runtime of 120 seconds
register_xpu_ci(est_time=120, suite="default")

# Check if dependencies are available. ``except Exception`` and not ``except ImportError``: importing
# the package runs the CPU-variant probe, which raises RuntimeError when the installed extension needs
# a newer ISA than this host -- that has to degrade to a skip, not to a collection error.
try:
    import torch

    kt_kernel = None
    try:
        import kt_kernel  # import the package first, it registers kt_kernel_ext

        kt_kernel_ext = kt_kernel.kt_kernel_ext  # access the extension module
    except Exception:
        kt_kernel = None
        import kt_kernel_ext  # bare extension module, e.g. from KT_KERNEL_EXT_DIR

    HAS_DEPS = True
except Exception as e:
    HAS_DEPS = False
    import_error = f"{type(e).__name__}: {e}"

# Reason this module cannot run here, or None. Evaluated once; pytest skips the module, script mode
# prints it (CI runs each file as ``python3 <file>`` and treats a skip as success, like the AMX tests).
_SKIP_REASON = None
if not HAS_DEPS:
    _SKIP_REASON = f"Dependencies not available: {import_error}"
elif getattr(torch, "xpu", None) is None or not torch.xpu.is_available():
    _SKIP_REASON = "torch.xpu is not available on this host"
elif getattr(kt_kernel_ext, "gpu_backend", None) != "xpu":
    _SKIP_REASON = (
        f"kt_kernel_ext.gpu_backend is {getattr(kt_kernel_ext, 'gpu_backend', None)!r}, not 'xpu' "
        "(rebuild with CPUINFER_USE_XPU=1)"
    )
if _SKIP_REASON is not None and __name__ != "__main__":
    pytest.skip(_SKIP_REASON, allow_module_level=True)

# Test parameters: a small AMX INT8 MoE (see test_moe_amx_accuracy_int8.py) so the whole test fits in
# the CI budget while still moving real data through the stream bridge.
expert_num = 8
hidden_size = 7168
intermediate_size = 2048
num_experts_per_tok = 4
max_len = 256
qlen = 8
CPUINFER_PARAM = 32
BUSY_MATMUL_DIM = 8192  # keeps the GPU busy so early-firing callbacks would read stale buffers

# Name ``Worker::run`` gives its thread with pthread_setname_np(); one such thread per sycl::queue
# exists in l0 mode and none at all in host_task mode.
BRIDGE_THREAD_NAME = "kt-xpu-bridge"
# Default of KT_XPU_BRIDGE_EVENTS and the clamp floor, both from xpu_stream_bridge.cpp.
DEFAULT_BRIDGE_EVENTS = 1024
MIN_BRIDGE_EVENTS = 2
# Ring wrap-around check: the smallest legal ring, recycled RING_ROUNDS / MIN_BRIDGE_EVENTS times.
RING_ROUNDS = 12
# Poison values for the pinned staging buffers; the real input is randn/100, so a CPU MoE that ran on
# either of these produces an output nowhere near the reference.
POISON_INPUT = 1.0
POISON_WEIGHT = 0.0

# Where ``stage_and_run`` puts the long GPU matmul relative to the bridge calls.
FILLER_NONE = "none"
FILLER_BEFORE_COPIES = "before_copies"
FILLER_BETWEEN_SUBMIT_AND_SYNC = "between_submit_and_sync"
_FILLER_POSITIONS = (FILLER_NONE, FILLER_BEFORE_COPIES, FILLER_BETWEEN_SUBMIT_AND_SYNC)


def act_fn(x):
    """Activation function for MoE."""
    return x / (1.0 + torch.exp(-x))


def mlp_torch(input, gate_proj, up_proj, down_proj):
    """PyTorch reference implementation of MLP."""
    gate_buf = torch.mm(input, gate_proj.t())
    up_buf = torch.mm(input, up_proj.t())
    intermediate = act_fn(gate_buf) * up_buf
    ret = torch.mm(intermediate, down_proj.t())
    return ret


def moe_torch(input, expert_ids, weights, gate_proj, up_proj, down_proj):
    """PyTorch reference implementation of MoE."""
    cnts = expert_ids.new_zeros((expert_ids.shape[0], expert_num))
    cnts.scatter_(1, expert_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = expert_ids.view(-1).argsort()
    sorted_tokens = input[idxs // expert_ids.shape[1]]

    outputs = []
    start_idx = 0

    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
        expert_out = mlp_torch(tokens_for_this_expert, gate_proj[i], up_proj[i], down_proj[i])
        outputs.append(expert_out)
        start_idx = end_idx

    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

    new_x = torch.empty_like(outs)
    new_x[idxs] = outs
    t_output = (
        new_x.view(*expert_ids.shape, -1)
        .type(weights.dtype)
        .mul_(weights.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )

    return t_output


def _bridge_mode():
    return kt_kernel_ext.xpu_stream_bridge_mode()


def _bridge_events():
    return kt_kernel_ext.xpu_stream_bridge_events()


def _shutdown_bridge():
    kt_kernel_ext.shutdown_xpu_stream_bridge()


def _select_bridge(monkeypatch, mode, events=None):
    """Point KT_XPU_BRIDGE(_EVENTS) at `mode` and make the bridge re-read them."""
    monkeypatch.setenv("KT_XPU_BRIDGE", mode)
    if events is not None:
        monkeypatch.setenv("KT_XPU_BRIDGE_EVENTS", str(events))
    _shutdown_bridge()
    assert _bridge_mode() == mode
    if events is not None:
        assert _bridge_events() == events, f"KT_XPU_BRIDGE_EVENTS={events} was not honoured"


def _bridge_thread_count():
    """Exact number of live bridge worker threads, by thread name (no slack, no other thread counted)."""
    count = 0
    for tid in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{tid}/comm") as f:
                name = f.read().strip()
        except OSError:
            continue  # the thread exited between listdir() and open()
        if name == BRIDGE_THREAD_NAME:
            count += 1
    return count


def _settled_bridge_thread_count(expected, timeout=5.0):
    """``_bridge_thread_count()`` once it reaches `expected`; a joined thread lingers in /proc a moment."""
    deadline = time.monotonic() + timeout
    count = _bridge_thread_count()
    while count != expected and time.monotonic() < deadline:
        time.sleep(0.02)
        count = _bridge_thread_count()
    return count


def _current_handle():
    """The ``sycl::queue*`` of the current stream, as CPUInfer takes it.

    Computed straight from torch on purpose: ``test_current_stream_handle_matches_torch`` is what
    asserts that ``kt_kernel.current_stream_handle("xpu")`` agrees with this expression.
    """
    return int(torch.xpu.current_stream().sycl_queue)


def _time_ms(fn):
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1e3


def _fresh_inputs():
    """Random MoE inputs living on the XPU; the CPU only ever sees them through the D2H copies."""
    input_dev = torch.randn((qlen, hidden_size), dtype=torch.bfloat16, device="xpu") / 100
    expert_ids_dev = torch.rand((qlen, expert_num), device="xpu").argsort(dim=-1)[:, :num_experts_per_tok].contiguous()
    weights_dev = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32, device="xpu")
    return input_dev, expert_ids_dev, weights_dev


def _check_output(actual, reference, label):
    output = actual.cpu()
    diff = torch.mean(torch.abs(output - reference)) / torch.mean(torch.abs(reference))
    print(f"{label}: diff = {diff:.6f}")
    # INT8 should have diff < 0.05; NaN (poisoned output never overwritten) also fails this.
    assert diff < 0.05, f"{label}: diff={diff} >= 0.05 (stale or missing output -> ordering broken)"


class _Staging:
    """Pinned host buffers for one bridged round, plus the device tensor its H2D copy writes.

    One set per round that must not wait for its predecessor: the native MoE task captures the raw
    buffer pointers, so rounds sharing a ``_Staging`` would overwrite each other's data.
    """

    def __init__(self):
        self.bsz_tensor = torch.tensor([qlen], device="cpu")
        self.input_pinned = torch.zeros((qlen, hidden_size), dtype=torch.bfloat16, pin_memory=True)
        self.expert_ids_pinned = torch.zeros((qlen, num_experts_per_tok), dtype=torch.int64, pin_memory=True)
        self.weights_pinned = torch.zeros((qlen, num_experts_per_tok), dtype=torch.float32, pin_memory=True)
        self.output_pinned = torch.zeros((qlen, hidden_size), dtype=torch.bfloat16, pin_memory=True)
        self.out_dev = torch.empty((qlen, hidden_size), dtype=torch.bfloat16, device="xpu")

    def poison(self):
        """Make every shortcut through {D2H -> CPU MoE -> H2D} observable in the final output.

        The inputs catch the submit edge (a CPU MoE that starts before the D2H copies have landed
        reads these instead of the real input), the outputs catch the sync edge and a missing H2D.
        The expert ids are left valid on purpose -- an out-of-range id would fault the kernel rather
        than produce a wrong result.
        """
        self.input_pinned.fill_(POISON_INPUT)
        self.weights_pinned.fill_(POISON_WEIGHT)
        self.output_pinned.fill_(float("nan"))
        self.out_dev.fill_(float("nan"))

    def copy_inputs_from(self, input_dev, expert_ids_dev, weights_dev):
        """D2H half of the protocol: async copies, ordered by the current stream only."""
        self.input_pinned.copy_(input_dev, non_blocking=True)
        self.expert_ids_pinned.copy_(expert_ids_dev, non_blocking=True)
        self.weights_pinned.copy_(weights_dev, non_blocking=True)


class _MoeUnderTest:
    """A small AMX INT8 MoE driven through the stream-callback protocol."""

    def __init__(self):
        self.cpu_infer = kt_kernel_ext.CPUInfer(CPUINFER_PARAM)
        self.physical_to_logical_map = torch.tensor(
            data=range(expert_num), device="cpu", dtype=torch.int64
        ).contiguous()

        with torch.inference_mode(mode=True):
            self.gate_proj = torch.randn(
                (expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16, device="cpu"
            ).contiguous()
            self.up_proj = torch.randn(
                (expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16, device="cpu"
            ).contiguous()
            self.down_proj = torch.randn(
                (expert_num, hidden_size, intermediate_size), dtype=torch.bfloat16, device="cpu"
            ).contiguous()

            config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
            config.max_len = max_len
            config.gate_proj = self.gate_proj.data_ptr()
            config.up_proj = self.up_proj.data_ptr()
            config.down_proj = self.down_proj.data_ptr()
            config.gate_scale = 0
            config.pool = self.cpu_infer.backend_

            self.moe = kt_kernel_ext.moe.AMXInt8_MOE(config)
            self.cpu_infer.submit(self.moe.load_weights_task(self.physical_to_logical_map.data_ptr()))
            self.cpu_infer.sync()

    def forward_task(self, staging):
        return self.moe.forward_task(
            staging.bsz_tensor.data_ptr(),
            num_experts_per_tok,
            staging.expert_ids_pinned.data_ptr(),
            staging.weights_pinned.data_ptr(),
            staging.input_pinned.data_ptr(),
            staging.output_pinned.data_ptr(),
            False,
        )

    def reference(self, input_dev, expert_ids_dev, weights_dev):
        return moe_torch(
            input_dev.cpu(),
            expert_ids_dev.cpu(),
            weights_dev.cpu(),
            self.gate_proj,
            self.up_proj,
            self.down_proj,
        )

    def run_cpu_moe(self, staging):
        """Run the CPU MoE synchronously on this thread (no stream involved)."""
        self.cpu_infer.submit(self.forward_task(staging))
        self.cpu_infer.sync()

    def stage_and_run(self, staging, input_dev, expert_ids_dev, weights_dev, use_bridge=True, gpu_filler=FILLER_NONE):
        """Enqueue {D2H copies, CPU MoE, H2D copy} for `staging` on the current XPU stream.

        ``gpu_filler`` decides where a ~11 ms GPU matmul goes, which is what separates the two
        ordering edges:

        * ``none`` -- nothing stands between the submit and the H2D copy, so the H2D is enqueued
          while the CPU MoE still has milliseconds of work left (it takes ~3 ms versus ~0.03 ms for
          the copy). A sync edge that fails to hold the queue back therefore copies the
          NaN-poisoned output buffer, and the caller's comparison fails.
        * ``before_copies`` -- the D2H copies are still queued behind the matmul when the submit
          edge runs, so a submit that does not wait for the queue makes the CPU MoE read the
          poisoned input buffer. ``_run_ordering_check`` arranges the same thing from outside.
        * ``between_submit_and_sync`` -- the realistic shape: GPU experts overlapping the CPU MoE.
          Both edges are still required, but the CPU MoE now finishes while the matmul runs, so
          this case alone would not prove the sync edge does anything.

        With ``use_bridge`` the CPU work is scheduled through submit/sync_with_cuda_stream; without
        it the stream is synchronised on the host and the plain submit/sync path is used (warm-up
        only). Returns the host-side cost in milliseconds of the submit alone (``submit_ms``) and of
        submit + sync + H2D together (``enqueue_ms``), which
        ``test_stream_bridge_submit_is_asynchronous`` compares against the cost of the CPU MoE.
        """
        assert gpu_filler in _FILLER_POSITIONS, f"unknown gpu_filler {gpu_filler!r}"
        handle = _current_handle()
        if gpu_filler == FILLER_BEFORE_COPIES:
            _busy_gpu()
        staging.copy_inputs_from(input_dev, expert_ids_dev, weights_dev)
        submit_start = time.perf_counter()
        if use_bridge:
            self.cpu_infer.submit_with_cuda_stream(handle, self.forward_task(staging))
        else:
            torch.xpu.synchronize()
            self.cpu_infer.submit(self.forward_task(staging))
        submit_end = time.perf_counter()
        if gpu_filler == FILLER_BETWEEN_SUBMIT_AND_SYNC:
            _busy_gpu()
        if use_bridge:
            self.cpu_infer.sync_with_cuda_stream(handle)
        else:
            self.cpu_infer.sync()
        staging.out_dev.copy_(staging.output_pinned, non_blocking=True)
        return {
            "submit_ms": (submit_end - submit_start) * 1e3,
            "enqueue_ms": (time.perf_counter() - submit_start) * 1e3,
        }


_busy_operands = {}


def _busy_gpu():
    """A large bf16 matmul on the current stream; whatever is enqueued next must wait for it.

    ~11 ms on an Arc Pro B60. The operands (~268 MB) are shared by every stream on purpose: two
    streams writing the same destination concurrently is harmless here, nothing ever reads it.
    """
    if not _busy_operands:
        a = torch.randn((BUSY_MATMUL_DIM, BUSY_MATMUL_DIM), dtype=torch.bfloat16, device="xpu")
        _busy_operands["a"] = a
        _busy_operands["out"] = torch.empty_like(a)
    torch.matmul(_busy_operands["a"], _busy_operands["a"], out=_busy_operands["out"])


def _run_ordering_check(moe, iterations, label, gpu_filler=FILLER_NONE):
    """`iterations` rounds of fresh device inputs -> bridge -> compare against the torch reference."""
    staging = _Staging()
    with torch.inference_mode(mode=True):
        for i in range(iterations):
            input_dev, expert_ids_dev, weights_dev = _fresh_inputs()
            staging.poison()
            # Submit edge: the D2H copies inside stage_and_run queue up behind this matmul, so they
            # are still pending when submit_with_cuda_stream runs.
            _busy_gpu()
            moe.stage_and_run(staging, input_dev, expert_ids_dev, weights_dev, gpu_filler=gpu_filler)
            torch.xpu.synchronize()
            reference = moe.reference(input_dev, expert_ids_dev, weights_dev)
            _check_output(staging.out_dev, reference, f"[{label}] iteration {i}")


@pytest.fixture(scope="module", autouse=True)
def gpu_scratch():
    """Hand the ~268 MB of ``_busy_gpu()`` operands back when this module is done."""
    yield
    torch.xpu.synchronize()
    _busy_operands.clear()
    torch.xpu.empty_cache()


@pytest.fixture(scope="module")
def moe_under_test():
    if not hasattr(kt_kernel_ext.moe, "AMXInt8_MOE"):
        pytest.skip("kt_kernel_ext.moe.AMXInt8_MOE is not available in this build")
    torch.manual_seed(0)
    moe = _MoeUnderTest()
    # Exercise every GPU op once outside the bridge so lazily created runtime state (kernels, host
    # threads of the SYCL runtime / allocator) does not perturb the bridge checks below.
    staging = _Staging()
    with torch.inference_mode(mode=True):
        input_dev, expert_ids_dev, weights_dev = _fresh_inputs()
        staging.poison()
        moe.stage_and_run(
            staging, input_dev, expert_ids_dev, weights_dev, use_bridge=False, gpu_filler=FILLER_BEFORE_COPIES
        )
        torch.xpu.synchronize()
        _check_output(staging.out_dev, moe.reference(input_dev, expert_ids_dev, weights_dev), "warm-up (no bridge)")
    return moe


@pytest.fixture
def bridge_env(monkeypatch):
    """Clean KT_XPU_BRIDGE* env for a test and shut the bridge down before and after it."""
    monkeypatch.delenv("KT_XPU_BRIDGE", raising=False)
    monkeypatch.delenv("KT_XPU_BRIDGE_EVENTS", raising=False)
    _shutdown_bridge()  # forget any config cached by an earlier test
    yield monkeypatch
    _shutdown_bridge()  # forget this test's config; monkeypatch restores the env afterwards


@pytest.mark.xpu
def test_bridge_mode_selection(bridge_env):
    monkeypatch = bridge_env
    assert _bridge_mode() == "l0"

    # The config is read once and cached until shutdown() forgets it.
    monkeypatch.setenv("KT_XPU_BRIDGE", "host_task")
    assert _bridge_mode() == "l0"
    _shutdown_bridge()
    assert _bridge_mode() == "host_task"

    monkeypatch.setenv("KT_XPU_BRIDGE", "bogus")
    _shutdown_bridge()
    with pytest.raises(RuntimeError, match="KT_XPU_BRIDGE"):
        _bridge_mode()

    monkeypatch.setenv("KT_XPU_BRIDGE", "l0")
    _shutdown_bridge()
    assert _bridge_mode() == "l0"


@pytest.mark.xpu
def test_bridge_events_selection(bridge_env):
    monkeypatch = bridge_env
    assert _bridge_events() == DEFAULT_BRIDGE_EVENTS

    # Like the mode, the ring size is read once and cached until shutdown() forgets it.
    monkeypatch.setenv("KT_XPU_BRIDGE_EVENTS", "16")
    assert _bridge_events() == DEFAULT_BRIDGE_EVENTS
    _shutdown_bridge()
    assert _bridge_events() == 16

    # An unparseable ring size is a hard error on first use, not a silent fallback to the default.
    for bad in ("abc", "4.5", "-1", "0x10", "16 "):
        monkeypatch.setenv("KT_XPU_BRIDGE_EVENTS", bad)
        _shutdown_bridge()
        with pytest.raises(RuntimeError, match="KT_XPU_BRIDGE_EVENTS"):
            _bridge_events()

    # Below the minimum the value is clamped to kMinEvents (with a warning on stderr), not
    # rejected: a one-slot ring would serialise every sync behind the previous one's barrier.
    for low in ("0", "1"):
        monkeypatch.setenv("KT_XPU_BRIDGE_EVENTS", low)
        _shutdown_bridge()
        assert _bridge_events() == MIN_BRIDGE_EVENTS

    # Empty means "not set", as for every other env var here.
    monkeypatch.setenv("KT_XPU_BRIDGE_EVENTS", "")
    _shutdown_bridge()
    assert _bridge_events() == DEFAULT_BRIDGE_EVENTS


@pytest.mark.xpu
@pytest.mark.parametrize(
    "mode,gpu_filler",
    [
        # Default: no GPU work hides the CPU MoE from the H2D copy, so the sync edge is load-bearing.
        ("l0", FILLER_NONE),
        ("host_task", FILLER_NONE),
        # One case in the shape a real model produces, GPU experts overlapping the CPU MoE.
        ("l0", FILLER_BETWEEN_SUBMIT_AND_SYNC),
    ],
)
def test_stream_bridge_end_to_end_ordering(bridge_env, moe_under_test, mode, gpu_filler):
    _select_bridge(bridge_env, mode)
    _run_ordering_check(moe_under_test, iterations=4, label=f"{mode}, {gpu_filler}", gpu_filler=gpu_filler)


@pytest.mark.xpu
def test_stream_bridge_l0_event_ring_wraps_without_host_sync(bridge_env, moe_under_test):
    """Recycle the smallest legal Level Zero event ring while its barriers are still in flight.

    This is how SGLang drives the bridge: many MoE layers per forward, back to back, with no host
    synchronisation in between. With MIN_BRIDGE_EVENTS slots and RING_ROUNDS rounds, ``launch_sync``
    has to wait for the barrier still consuming a slot before resetting and re-arming its event --
    measured on an Arc Pro B60, the enqueue loop below takes ~320 ms with a 2-slot ring against
    ~3 ms with the default 1024-slot one, i.e. the ring really does throttle the host here.
    """
    moe = moe_under_test
    _select_bridge(bridge_env, "l0", events=MIN_BRIDGE_EVENTS)
    assert _bridge_events() == MIN_BRIDGE_EVENTS

    # Every round needs its own buffers: the native task holds the raw pointers and no round waits
    # for its predecessor, so a shared staging set would be overwritten mid-flight.
    rounds = [(_Staging(), _fresh_inputs()) for _ in range(RING_ROUNDS)]
    with torch.inference_mode(mode=True):
        for staging, _ in rounds:
            staging.poison()
        torch.xpu.synchronize()  # settle the setup above; the measured section must not synchronise

        for staging, inputs in rounds:
            moe.stage_and_run(staging, *inputs, gpu_filler=FILLER_BEFORE_COPIES)

        torch.xpu.synchronize()
        for i, (staging, inputs) in enumerate(rounds):
            _check_output(staging.out_dev, moe.reference(*inputs), f"[ring of {MIN_BRIDGE_EVENTS}] round {i}")


@pytest.mark.xpu
def test_stream_bridge_worker_thread_lifecycle(bridge_env, moe_under_test):
    moe = moe_under_test

    # host_task mode runs the callbacks on the SYCL runtime's own threads: no kt-kernel thread at all.
    _select_bridge(bridge_env, "host_task")
    assert _settled_bridge_thread_count(0) == 0, "a bridge worker exists before the bridge was ever used"
    _run_ordering_check(moe, iterations=1, label="threads: host_task")
    assert _bridge_thread_count() == 0, "host_task mode must not start a bridge worker thread"

    # l0 mode: exactly one worker for the one queue in use, and none left after shutdown.
    _select_bridge(bridge_env, "l0")
    assert _settled_bridge_thread_count(0) == 0
    _run_ordering_check(moe, iterations=2, label="threads: l0")
    assert _bridge_thread_count() == 1, "l0 mode must run exactly one worker thread per sycl::queue"
    _shutdown_bridge()
    assert _settled_bridge_thread_count(0) == 0, "shutdown_xpu_stream_bridge() did not join the worker thread"


@pytest.mark.xpu
def test_stream_bridge_l0_two_streams(bridge_env, moe_under_test):
    """Two torch.xpu.Streams in flight at once: one worker each, both results correct."""
    moe = moe_under_test
    _select_bridge(bridge_env, "l0")
    assert _settled_bridge_thread_count(0) == 0

    streams = [torch.xpu.Stream(), torch.xpu.Stream()]
    rounds = [(_Staging(), _fresh_inputs()) for _ in streams]
    with torch.inference_mode(mode=True):
        for staging, _ in rounds:
            staging.poison()
        # The inputs and the poison were produced on the default stream. Publish them with one host
        # synchronisation rather than relying on cross-stream ordering, which is not what is tested.
        torch.xpu.synchronize()

        for stream, (staging, inputs) in zip(streams, rounds):
            with torch.xpu.stream(stream):
                assert _current_handle() == int(stream.sycl_queue)
                _busy_gpu()
                moe.stage_and_run(staging, *inputs)
        assert _bridge_thread_count() == 2, "each sycl::queue in use must get its own l0 worker thread"

        torch.xpu.synchronize()
        for i, (staging, inputs) in enumerate(rounds):
            _check_output(staging.out_dev, moe.reference(*inputs), f"[two streams] stream {i}")

    _shutdown_bridge()
    assert _settled_bridge_thread_count(0) == 0, "shutdown_xpu_stream_bridge() left a worker thread behind"


@pytest.mark.xpu
def test_stream_bridge_sync_allows_one_pending_task(bridge_env, moe_under_test):
    """The deferred-expert shape of ``experts_base.submit_forward``: two submits, then sync(.., 1).

    ``sync_with_cuda_stream(handle, 1)`` drains with ``task_queue_->sync(1)``, which returns while
    one enqueued task is still pending. The immediate task must nevertheless be complete before the
    queue is released, because the H2D copy behind the sync reads its output buffer.
    """
    moe = moe_under_test
    _select_bridge(bridge_env, "l0")
    handle = _current_handle()
    immediate, deferred = _Staging(), _Staging()
    immediate_inputs, deferred_inputs = _fresh_inputs(), _fresh_inputs()

    with torch.inference_mode(mode=True):
        immediate.poison()
        deferred.poison()
        _busy_gpu()  # both D2H copies are still queued behind this when the submits run
        immediate.copy_inputs_from(*immediate_inputs)
        deferred.copy_inputs_from(*deferred_inputs)
        moe.cpu_infer.submit_with_cuda_stream(handle, moe.forward_task(immediate))
        moe.cpu_infer.submit_with_cuda_stream(handle, moe.forward_task(deferred))
        moe.cpu_infer.sync_with_cuda_stream(handle, 1)
        immediate.out_dev.copy_(immediate.output_pinned, non_blocking=True)
        torch.xpu.synchronize()
        _check_output(immediate.out_dev, moe.reference(*immediate_inputs), "[allow_n_pending=1] immediate")

        # The deferred task was allowed to still be pending; a full drain finishes it. Its output is
        # checked in the pinned buffer, which is where the next layer's H2D copy would read it.
        moe.cpu_infer.sync()
        _check_output(deferred.output_pinned, moe.reference(*deferred_inputs), "[allow_n_pending=1] deferred")


@pytest.mark.xpu
def test_stream_bridge_submit_is_asynchronous(bridge_env, moe_under_test):
    """The bridge must enqueue the CPU MoE, not run it on the calling thread.

    An implementation that waited for the queue and called the task inline would satisfy every
    correctness check in this file while removing the whole point of the feature, so compare the
    cost of the bridge calls against the cost of the CPU MoE measured on this very host.
    """
    moe = moe_under_test
    _select_bridge(bridge_env, "l0")
    staging = _Staging()
    input_dev, expert_ids_dev, weights_dev = _fresh_inputs()

    with torch.inference_mode(mode=True):
        # Warm-up round: the first launch on a queue creates the worker thread and its Level Zero
        # event pool (~1.7 ms), and it leaves real data in the pinned input buffer for the
        # measurement below.
        staging.poison()
        _busy_gpu()
        moe.stage_and_run(staging, input_dev, expert_ids_dev, weights_dev)
        torch.xpu.synchronize()
        _check_output(staging.out_dev, moe.reference(input_dev, expert_ids_dev, weights_dev), "[async] warm-up")

        # What the CPU MoE costs right now, on this host, with no stream involved.
        cpu_ms = min(_time_ms(lambda: moe.run_cpu_moe(staging)) for _ in range(3))

        # Same task through the bridge, with ~11 ms of GPU work queued in front of it so it cannot
        # even have started by the time the bridge calls return.
        timings = []
        for i in range(5):
            staging.poison()
            _busy_gpu()
            timings.append(moe.stage_and_run(staging, input_dev, expert_ids_dev, weights_dev))
            torch.xpu.synchronize()
            _check_output(staging.out_dev, moe.reference(input_dev, expert_ids_dev, weights_dev), f"[async] round {i}")

    submit_ms = min(t["submit_ms"] for t in timings)
    enqueue_ms = min(t["enqueue_ms"] for t in timings)
    print(f"[async] cpu_moe={cpu_ms:.3f} ms submit={submit_ms:.4f} ms submit+sync+h2d={enqueue_ms:.4f} ms")

    # Measured on an Arc Pro B60: ~3 ms for the CPU MoE against ~0.006 ms for the submit and
    # ~0.012 ms for submit+sync+H2D, i.e. a factor of a few hundred. The box is shared and noisy, so
    # only one order of magnitude is required, and the minimum of several rounds is used so a single
    # descheduled sample cannot fail the test.
    assert submit_ms < cpu_ms / 10, f"submit_with_cuda_stream took {submit_ms:.3f} ms, CPU MoE {cpu_ms:.3f} ms"
    assert enqueue_ms < cpu_ms / 10, f"submit+sync+H2D took {enqueue_ms:.3f} ms, CPU MoE {cpu_ms:.3f} ms"


@pytest.mark.xpu
def test_current_stream_handle_matches_torch(bridge_env, moe_under_test):
    """``kt_kernel.current_stream_handle("xpu")`` against real torch streams, not a fake module."""
    if kt_kernel is None:
        pytest.skip("the kt_kernel package is not importable here (bare extension module)")

    assert kt_kernel.current_stream_handle("xpu") == torch.xpu.current_stream().sycl_queue
    assert kt_kernel.current_stream_handle(torch.device("xpu")) == torch.xpu.current_stream().sycl_queue

    other = torch.xpu.Stream()
    assert other.sycl_queue != torch.xpu.current_stream().sycl_queue
    with torch.xpu.stream(other):
        assert kt_kernel.current_stream_handle("xpu") == other.sycl_queue
        assert kt_kernel.current_stream_handle(torch.device("xpu")) == other.sycl_queue

        # And it is the handle the bridge accepts, on whichever stream is current.
        _select_bridge(bridge_env, "l0")
        staging = _Staging()
        inputs = _fresh_inputs()
        with torch.inference_mode(mode=True):
            staging.poison()
            torch.xpu.synchronize()
            _busy_gpu()
            assert _current_handle() == kt_kernel.current_stream_handle("xpu")
            moe_under_test.stage_and_run(staging, *inputs)
            torch.xpu.synchronize()
            _check_output(staging.out_dev, moe_under_test.reference(*inputs), "[handle] other stream")

    assert kt_kernel.current_stream_handle("xpu") == torch.xpu.current_stream().sycl_queue


@pytest.mark.xpu
def test_stream_bridge_xpu_graph_replay(bridge_env, moe_under_test):
    moe = moe_under_test
    _select_bridge(bridge_env, "l0")  # recording queues must switch to host_task on their own

    input_dev, expert_ids_dev, weights_dev = _fresh_inputs()
    staging = _Staging()
    stream = torch.xpu.Stream()

    with torch.inference_mode(mode=True):
        # Warm-up on the capture stream (as with CUDA graphs) doubles as an eager correctness check.
        staging.poison()
        torch.xpu.synchronize()
        with torch.xpu.stream(stream):
            _busy_gpu()
            moe.stage_and_run(staging, input_dev, expert_ids_dev, weights_dev)
        torch.xpu.synchronize()
        _check_output(staging.out_dev, moe.reference(input_dev, expert_ids_dev, weights_dev), "[graph] eager")

        # Drop the l0 worker the eager round created for this queue. Everything below must then run
        # without one: a recording queue uses host_task whatever KT_XPU_BRIDGE says, and a
        # misdetection would otherwise only show up as a hang.
        _shutdown_bridge()
        assert _bridge_mode() == "l0"
        assert _settled_bridge_thread_count(0) == 0

        graph = torch.xpu.XPUGraph()
        try:
            with torch.xpu.graph(graph, stream=stream):
                assert torch.xpu.current_stream().sycl_queue == stream.sycl_queue
                moe.stage_and_run(staging, input_dev, expert_ids_dev, weights_dev)
        except (RuntimeError, NotImplementedError) as e:  # torch's XPU graph may reject an op
            pytest.skip(f"torch.xpu.graph could not capture the stream-bridge sequence: {e!r}")
        assert _bridge_thread_count() == 0, "a recording queue must use host_task, not an l0 worker thread"

        for replay in range(3):
            new_input, new_expert_ids, new_weights = _fresh_inputs()
            input_dev.copy_(new_input)
            expert_ids_dev.copy_(new_expert_ids)
            weights_dev.copy_(new_weights)
            # Poison both ends: a replay that skips the captured D2H copies runs the CPU MoE on
            # the poisoned input, and one that skips the host callbacks copies NaNs back instead.
            staging.poison()
            torch.xpu.synchronize()
            graph.replay()
            torch.xpu.synchronize()
            _check_output(
                staging.out_dev,
                moe.reference(new_input, new_expert_ids, new_weights),
                f"[graph] replay {replay}",
            )
        assert _bridge_thread_count() == 0, "graph replay must not start an l0 worker thread"


if __name__ == "__main__":
    if _SKIP_REASON is not None:
        print(f"Skipping XPU stream bridge tests: {_SKIP_REASON}")
        raise SystemExit(0)
    raise SystemExit(pytest.main([__file__, "-q"]))
