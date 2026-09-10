# SPDX-License-Identifier: Apache-2.0
"""Stop this harness run's own launcher and marked detached ranks.

PyTorch workers can call setsid(), so the launcher's process group is not the
training job. Match the exact entrypoint and unique --run-dir argument, then
bind signaling to Linux pidfds. Never signal by a broad Python/CUDA process name.
"""

from contextlib import ExitStack
import ctypes
import os
from pathlib import Path
import platform
import select
import signal
import sys
import time


def _syscall(number, *arguments):
    if sys.platform != "linux" or platform.machine() not in ("x86_64", "amd64"):
        raise RuntimeError("safe process cleanup requires Linux pidfd support")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *arguments)
    if result == -1:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result


def _pidfd_open(pid):
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid, 0)
    # Some Python builds lack the wrappers. Probe the kernel capability
    # before launch; these syscall numbers are for Linux x86-64 only.
    return _syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))


def _send(fd, number):
    try:
        if hasattr(signal, "pidfd_send_signal"):
            signal.pidfd_send_signal(fd, number)
        else:
            _syscall(424, ctypes.c_int(fd), ctypes.c_int(number), ctypes.c_void_p(), ctypes.c_uint(0))
    except ProcessLookupError:
        pass  # The same referenced process has already exited.


def check_support():
    """Fail closed before starting a GPU job; signal 0 changes no process state."""
    fd = _pidfd_open(os.getpid())
    try:
        _send(fd, 0)
    finally:
        os.close(fd)


def matches_run(arguments, entrypoint, run_dir):
    if str(entrypoint) not in arguments or arguments.count("--run-dir") != 1:
        return False
    offset = arguments.index("--run-dir") + 1
    return offset < len(arguments) and arguments[offset] == str(run_dir)


def _identity(directory):
    # comm may contain spaces and ')'; fields after its last ')' are stable.
    fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
    return int(directory.name), int(fields[19])  # PID, starttime (field 22).


def _exited(fd):
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    return bool(poll.poll(0))


def stop_training(process, entrypoint, run_dir, grace_seconds=20):
    """Terminate this exact run, then escalate only still-live pidfd targets.

    Detached workers remain discoverable by the unique run argument even if
    the launcher has already exited. Unmarked, detached tool-service daemons
    are outside this guard's ownership; it never kills a global Nsight service.
    """
    actions = []
    with ExitStack() as lifetime:
        handles = {}

        # Our direct child is owned by Popen, even if a launcher changes argv.
        # A live, unreaped child cannot have its PID recycled. The supervisor is
        # the sole caller of this Popen's wait/poll methods.
        if process.poll() is None:
            identity = _identity(Path("/proc") / str(process.pid))
            fd = _pidfd_open(process.pid)
            lifetime.callback(os.close, fd)
            handles[identity] = fd

        def discover():
            fresh = []
            for directory in Path("/proc").iterdir():
                if not directory.name.isdecimal():
                    continue
                fd = None
                try:
                    arguments = (directory / "cmdline").read_bytes().decode(errors="surrogateescape").split("\0")
                    if not matches_run(arguments, entrypoint, run_dir):
                        continue
                    identity = _identity(directory)
                    if identity in handles:
                        continue
                    fd = _pidfd_open(identity[0])
                    # A PID can disappear or be reused between discovery and
                    # opening. Recheck identity and argv after binding the fd.
                    arguments = (directory / "cmdline").read_bytes().decode(errors="surrogateescape").split("\0")
                    if _identity(directory) != identity or not matches_run(arguments, entrypoint, run_dir):
                        os.close(fd)
                        fd = None
                        continue
                    lifetime.callback(os.close, fd)
                    handles[identity] = fd
                    fresh.append((identity, fd))
                    fd = None  # ExitStack now owns it.
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    pass
                finally:
                    if fd is not None:
                        os.close(fd)
            return fresh

        def send(targets, number):
            for identity, fd in targets:
                if not _exited(fd):
                    _send(fd, number)
                    actions.append({"pid": identity[0], "start_ticks": identity[1], "signal": number})

        discover()
        send(list(handles.items()), signal.SIGTERM)
        deadline = time.monotonic() + grace_seconds
        while any(not _exited(fd) for fd in handles.values()) and time.monotonic() < deadline:
            time.sleep(0.05)
        # Drain late children as well: a rank can be spawned while its launcher
        # is stopping. Keep all old fds bound until the complete drain finishes.
        deadline = time.monotonic() + 5
        while True:
            discover()
            send(list(handles.items()), signal.SIGKILL)
            if all(_exited(fd) for fd in handles.values()) and not discover():
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("owned training processes remain alive after SIGKILL")
            time.sleep(0.05)
    process.wait(timeout=5)
    return actions
