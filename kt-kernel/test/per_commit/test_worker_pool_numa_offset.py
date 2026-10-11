"""Regression test for per-NUMA worker thread offsets (#2221).

``WorkerPool`` maps each subpool onto a physical NUMA node and hands the pool a
per-node starting thread index so workers pin to distinct cores. The counter
used to derive that offset is indexed by the *physical* NUMA ID, but it used to
be sized by the number of subpools. A single subpool mapped to a NUMA node whose
ID is >= the subpool count (e.g. ``--kt-numa-nodes 1``) therefore read/wrote out
of bounds, shifting the core offset and logging ``Core ... not found`` (or
corrupting memory).

This test builds two subpools on a high-numbered NUMA node and checks that the
worker thread names carry the expected cumulative offsets. It only needs a CPU
and at least two NUMA nodes; it is skipped otherwise.
"""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=30, suite="default")

try:
    import kt_kernel  # noqa: F401

    kt_kernel_ext = kt_kernel.kt_kernel_ext
    HAS_KT_KERNEL = True
except ImportError:
    HAS_KT_KERNEL = False
    kt_kernel_ext = None


def _online_numa_nodes():
    """Return [(node_id, physical_core_count), ...] for online NUMA nodes."""
    base = "/sys/devices/system/node"
    if not os.path.isdir(base):
        return []
    nodes = []
    for name in sorted(os.listdir(base)):
        match = re.fullmatch(r"node(\d+)", name)
        if match is None:
            continue
        node_dir = os.path.join(base, name)
        core_ids = set()
        for entry in os.listdir(node_dir):
            if not entry.startswith("cpu"):
                continue
            try:
                with open(os.path.join(node_dir, entry, "topology", "core_id")) as f:
                    core_ids.add(f.read().strip())
            except OSError:
                pass
        if core_ids:
            nodes.append((int(match.group(1)), len(core_ids)))
    return nodes


def _thread_names():
    names = set()
    task_dir = os.path.join("/proc", str(os.getpid()), "task")
    if not os.path.isdir(task_dir):
        return names
    for tid in os.listdir(task_dir):
        try:
            with open(os.path.join(task_dir, tid, "comm")) as f:
                names.add(f.read().strip())
        except OSError:
            pass
    return names


def _worker_suffixes(names, node_id):
    pattern = re.compile(rf"^numa_{node_id}_t_(\d+)$")
    suffixes = set()
    for name in names:
        match = pattern.match(name)
        if match is not None:
            suffixes.add(int(match.group(1)))
    return suffixes


@pytest.mark.cpu
def test_subpool_thread_offsets_on_high_numa_node(capfd):
    if not HAS_KT_KERNEL:
        pytest.skip("kt_kernel_ext not built or available")
    if not hasattr(kt_kernel_ext, "WorkerPoolConfig"):
        pytest.skip("WorkerPoolConfig binding not available in this build")

    nodes = _online_numa_nodes()
    if len(nodes) < 2:
        pytest.skip("test requires at least two online NUMA nodes")

    # The bug only manifests when a subpool targets a NUMA node whose physical ID
    # is >= the subpool count. Use the highest node ID present.
    node_id, cores = nodes[-1]
    if cores < 4:
        pytest.skip(f"NUMA node {node_id} has too few physical cores ({cores})")

    first_count = min(3, cores - 1)
    second_count = min(3, cores - first_count)

    capfd.readouterr()  # drain output produced during import

    cfg = kt_kernel_ext.WorkerPoolConfig()
    cfg.subpool_count = 2
    cfg.subpool_numa_map = [node_id, node_id]
    cfg.subpool_thread_count = [first_count, second_count]
    cpuinfer = kt_kernel_ext.CPUInfer(cfg)  # noqa: F841  (kept alive on purpose)

    out, err = capfd.readouterr()
    not_found = [
        line
        for line in (out + err).splitlines()
        if "not found" in line and f"NUMA node {node_id}" in line
    ]
    assert not not_found, f"worker threads mis-bound on NUMA node {node_id}: {not_found}"

    suffixes = _worker_suffixes(_thread_names(), node_id)

    # Subpool 0 (``first_count`` threads) owns offsets 1..first_count-1; subpool 1
    # starts at offset ``first_count`` and owns first_count+1..first_count+second_count-1.
    # Offset ``first_count`` belongs to subpool 1's calling thread and is unnamed.
    expected = set(range(1, first_count)) | set(range(first_count + 1, first_count + second_count))
    assert suffixes == expected, (
        f"unexpected worker thread offsets on NUMA node {node_id}: "
        f"got {sorted(suffixes)}, expected {sorted(expected)}. "
        "WorkerPool computed per-NUMA thread offsets out of bounds."
    )

    del cpuinfer


@pytest.mark.cpu
def test_single_subpool_on_nonzero_numa_node(capfd):
    """#2221 also reproduces with a single subpool on a nonzero NUMA node.

    The two-subpool test above only indexes out of bounds when the selected
    NUMA ID is >= the subpool count (hosts with three or more NUMA nodes). A
    single subpool targeting node 1 reads index 1 of a size-1 counter, which
    reproduces the original bug on any dual-socket host.
    """
    if not HAS_KT_KERNEL:
        pytest.skip("kt_kernel_ext not built or available")
    if not hasattr(kt_kernel_ext, "WorkerPoolConfig"):
        pytest.skip("WorkerPoolConfig binding not available in this build")

    nonzero_nodes = [(nid, cores) for nid, cores in _online_numa_nodes() if nid != 0 and cores >= 2]
    if not nonzero_nodes:
        pytest.skip("test requires a NUMA node with a nonzero ID and at least 2 cores")
    node_id, cores = nonzero_nodes[0]
    threads = min(cores, 8)

    capfd.readouterr()  # drain output produced during import

    cfg = kt_kernel_ext.WorkerPoolConfig()
    cfg.subpool_count = 1
    cfg.subpool_numa_map = [node_id]
    cfg.subpool_thread_count = [threads]
    cpuinfer = kt_kernel_ext.CPUInfer(cfg)  # noqa: F841  (kept alive on purpose)

    out, err = capfd.readouterr()
    not_found = [
        line
        for line in (out + err).splitlines()
        if "not found" in line and f"NUMA node {node_id}" in line
    ]
    assert not not_found, f"worker threads mis-bound on NUMA node {node_id}: {not_found}"

    # A single subpool starts at offset 0, so its workers are named
    # numa_<node>_t_1 .. numa_<node>_t_(threads-1). A nonzero starting offset
    # (the out-of-bounds read) shifts this set and loses t_1.
    suffixes = _worker_suffixes(_thread_names(), node_id)
    expected = set(range(1, threads))
    assert suffixes == expected, (
        f"unexpected worker thread offsets on NUMA node {node_id}: "
        f"got {sorted(suffixes)}, expected {sorted(expected)}. "
        "WorkerPool computed the single-subpool start offset out of bounds."
    )

    del cpuinfer
