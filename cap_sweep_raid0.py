#!/usr/bin/env python3
"""MESH cap sweep benchmark — RAID0 O_DIRECT version.

Tests multiple cap values, measures decode speed, prefill speed, and memory peak.
NO pkill -f: uses screen quit + ss+kill for safe process management.
Does NOT touch gpu4 processes (port 30001/30002).

Usage: python3 cap_sweep_raid0.py "32 64 96 128 160 196 224"
"""
import subprocess, time, json, threading, requests, sys, os, re

CAP_LIST = [int(x) for x in sys.argv[1].split()] if len(sys.argv) > 1 else [32, 64, 96, 128, 160, 196, 224]
PORT = 50052
CUDA = "6,7"
SCREEN_NAME = "mesh_sweep"
RESULT_FILE = "/mnt/data2/tmp/qujing_mesh/cap_sweep_results.txt"
SCRIPT = "run_mesh_35b_cap_graph_raid0.sh"
WORKDIR = "/mnt/data2/tmp/qujing_mesh"
GPU_IDS = [6, 7]  # physical GPU indices for nvidia-smi query

URL = f"http://localhost:{PORT}/v1/completions"


def run(cmd, timeout=30):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)


def stop_server():
    """Stop server safely without pkill -f."""
    # 1. Quit screen session
    run(f"screen -X -S {SCREEN_NAME} quit")
    time.sleep(2)
    # 2. Find PID listening on PORT and kill it (only our port, not gpu4's 30001/30002)
    out = run(f"ss -tlnp 2>/dev/null | grep ':{PORT} '")
    pids = re.findall(r'pid=(\d+)', out.stdout)
    for pid in pids:
        run(f"kill {pid} 2>/dev/null")
    # 3. Wait for port release
    for _ in range(10):
        out = run(f"ss -tln 2>/dev/null | grep ':{PORT} '")
        if not out.stdout.strip():
            break
        time.sleep(2)
    time.sleep(3)


def start_server(cap):
    """Start server with given cap via screen + taskset."""
    log_file = f"/tmp/mesh_sweep_cap{cap}.log"
    cmd = (
        f"cd {WORKDIR} && screen -dmS {SCREEN_NAME} bash -c "
        f"'taskset -c 24-47,72-191 bash {SCRIPT} {cap} {PORT} {CUDA} > {log_file} 2>&1'"
    )
    run(cmd)
    return log_file


def wait_ready(log_file, max_wait=900):
    """Wait for server ready, check fatal errors."""
    for i in range(max_wait // 5):
        time.sleep(5)
        try:
            with open(log_file) as f:
                content = f.read()
            if "fired up and ready to roll" in content:
                return True
            for fatal in ["Segmentation fault", "CUDA error", "RuntimeError:",
                          "Traceback (most recent call last):", "core dumped",
                          "Out of memory", "timeout", "Address already in use"]:
                if fatal in content:
                    print(f"FATAL detected: {fatal}")
                    return False
        except:
            pass
    return False


def get_all_pids():
    """Get all PIDs related to our server (parent + children + grandchildren)."""
    # Find parent PID by port
    out = run(f"ss -tlnp 2>/dev/null | grep ':{PORT} '")
    pids = re.findall(r'pid=(\d+)', out.stdout)
    if not pids:
        return []
    all_pids = list(pids)
    # Recursively find children
    to_check = list(pids)
    while to_check:
        pid = to_check.pop(0)
        out = run(f"pgrep -P {pid}")
        for child in out.stdout.strip().split('\n'):
            if child and child not in all_pids:
                all_pids.append(child)
                to_check.append(child)
    return all_pids


def get_vmhwm(pids):
    """Get VmHWM (peak resident set) for processes."""
    results = []
    for pid in pids:
        try:
            name = ""
            hwm = 0
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("Name:"):
                        name = line.split()[1] if len(line.split()) > 1 else "?"
                    elif line.startswith("VmHWM:"):
                        hwm = int(line.split()[1])  # KB
            if hwm > 0:
                results.append((pid, name, hwm))
        except:
            pass
    return results


def get_numastat(pids):
    """Get NUMA memory distribution for processes."""
    pid_str = ",".join(pids)
    out = run(f"numastat -p {pid_str} 2>/dev/null")
    return out.stdout


def sample_memory(stop_event, samples):
    """Background thread: sample system + GPU memory every 2 seconds."""
    gpu_query = ",".join(str(i) for i in GPU_IDS)
    while not stop_event.is_set():
        # System memory (free -m)
        try:
            out = subprocess.check_output(["free", "-m"], text=True)
            for line in out.splitlines():
                if line.startswith("Mem:"):
                    used = int(line.split()[2])
                    samples.append(("mem", used))
                    break
        except:
            pass
        # GPU memory (only our GPUs)
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                 "-i", gpu_query],
                text=True
            )
            gpu_sum = sum(int(x) for x in out.strip().split("\n") if x.strip())
            samples.append(("gpu", gpu_sum))
        except:
            pass
        time.sleep(2)


def bench_decode():
    """Benchmark decode with 512 tokens (stream mode for accurate timing)."""
    data = {
        "model": "default",
        "prompt": "Write a detailed essay about artificial intelligence and its impact on society, including historical context, current developments, and future implications:",
        "max_tokens": 512,
        "temperature": 0,
        "stream": True,
    }
    start = time.time()
    first = None
    cnt = 0
    r = requests.post(URL, json=data, stream=True, timeout=300)
    for line in r.iter_lines():
        if line:
            line = line.decode()
            if line.startswith("data: "):
                line = line[6:]
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                    if chunk["choices"][0]["text"]:
                        cnt += 1
                        if first is None:
                            first = time.time()
                except:
                    pass
    end = time.time()
    dec = end - first if first else end - start
    dtps = cnt / dec if dec > 0 else 0
    return dtps, cnt


def bench_prefill():
    """Benchmark prefill with ~2000-token prompt."""
    prompt = "The quick brown fox jumps over the lazy dog. " * 260
    data = {"model": "default", "prompt": prompt, "max_tokens": 1, "temperature": 0, "stream": False}
    start = time.time()
    r = requests.post(URL, json=data, timeout=300)
    end = time.time()
    resp = r.json()
    pt = resp.get("usage", {}).get("prompt_tokens", 0)
    elapsed = end - start
    tps = pt / elapsed if elapsed > 0 else 0
    return pt, elapsed, tps


def warmup():
    """Warmup with short generation."""
    try:
        warmup = {"model": "default", "prompt": "Hello, how are you?", "max_tokens": 16, "temperature": 0}
        r = requests.post(URL, json=warmup, timeout=60)
        return r.status_code == 200
    except:
        return False


# ===== Main =====
with open(RESULT_FILE, "w") as f:
    f.write("cap | decode_tok_s | decode_tokens | prefill_tok_s | prefill_tokens | "
            "peak_sys_used_gib | peak_gpu_gib | scheduler_vmhwm\n")

print(f"Cap sweep: {CAP_LIST}")
print(f"Results -> {RESULT_FILE}")
print(f"Config: mesh=ON, TP=2, GE=32, dual-numa(0,1), defer=3, RAID0 O_DIRECT")

for cap in CAP_LIST:
    cap_label = "full(224)" if cap == 224 else str(cap)
    print(f"\n========== Testing cap={cap_label} ==========")

    # 1. Stop old server
    print("Stopping old server...")
    stop_server()

    # 2. Start new server
    log_file = start_server(cap)
    print(f"Started cap={cap}, log={log_file}, waiting for ready...")

    if not wait_ready(log_file):
        print(f"FAILED: Server not ready for cap={cap}")
        with open(RESULT_FILE, "a") as f:
            f.write(f"{cap} | FAILED | - | FAILED | - | FAILED | FAILED | FAILED\n")
        # Print last 30 lines of log for debugging
        run(f"tail -30 {log_file}")
        continue

    print("Server ready")
    time.sleep(5)

    # 3. Get PIDs
    all_pids = get_all_pids()
    print(f"Server PIDs: {all_pids}")

    # 4. Start memory sampling
    samples = []
    stop_event = threading.Event()
    mem_thread = threading.Thread(target=sample_memory, args=(stop_event, samples))
    mem_thread.start()
    print("Memory sampler started")

    # 5. Warmup
    print("Warmup...")
    if not warmup():
        print("Warmup failed!")
        stop_event.set()
        mem_thread.join()
        with open(RESULT_FILE, "a") as f:
            f.write(f"{cap} | WARMUP_FAIL | - | - | - | - | - | -\n")
        continue
    time.sleep(2)

    # 6. Prefill benchmark
    try:
        pt, ptime, ptps = bench_prefill()
        print(f"Prefill: {pt} tokens, {ptime:.2f}s, {ptps:.2f} tok/s")
    except Exception as e:
        print(f"Prefill failed: {e}")
        pt, ptime, ptps = 0, 0, 0

    time.sleep(3)

    # 7. Decode benchmark
    try:
        dtps, dcnt = bench_decode()
        print(f"Decode: {dtps:.2f} tok/s ({dcnt} tokens)")
    except Exception as e:
        print(f"Decode failed: {e}")
        dtps, dcnt = 0, 0

    # 8. Stop memory sampling
    stop_event.set()
    mem_thread.join()

    # 9. Get VmHWM (cumulative peak)
    vmhwm_info = get_vmhwm(all_pids)
    scheduler_hwm = "; ".join(f"{name}:{hwm//1024}MB" for pid, name, hwm in vmhwm_info if "scheduler" in name)

    # Calculate peaks
    mem_vals = [v for k, v in samples if k == "mem"]
    gpu_vals = [v for k, v in samples if k == "gpu"]
    peak_mem = max(mem_vals) / 1024 if mem_vals else 0  # GiB
    peak_gpu = max(gpu_vals) / 1024 if gpu_vals else 0  # GiB

    print(f"Peak sys used: {peak_mem:.2f} GiB, Peak GPU(6,7): {peak_gpu:.2f} GiB")
    print(f"Scheduler VmHWM: {scheduler_hwm}")

    # Write result
    with open(RESULT_FILE, "a") as f:
        f.write(f"{cap} | {dtps:.2f} | {dcnt} | {ptps:.2f} | {pt} | "
                f"{peak_mem:.2f} | {peak_gpu:.2f} | {scheduler_hwm}\n")

    # Print numastat for this cap
    numastat_out = get_numastat(all_pids)
    print(f"NUMA distribution:\n{numastat_out}")

# Stop final server
print("\nStopping final server...")
stop_server()

print("\n=== Results ===")
with open(RESULT_FILE) as f:
    print(f.read())
