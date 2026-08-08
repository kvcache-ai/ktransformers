#!/usr/bin/env python3
"""Standard kt (non-MESH) benchmark with memory sampling — Python version.
Usage: python3 kt_bench_py.py <MODEL> <PORT> <CUDA>
Example: python3 kt_bench_py.py 35b 10004 0,1,2,3
         python3 kt_bench_py.py 397b 10004 0,1,2,3
"""
import subprocess, time, json, threading, requests, sys, os

MODEL = sys.argv[1]  # 35b / 397b
PORT = int(sys.argv[2])
CUDA = sys.argv[3]

if MODEL == "35b":
    SCRIPT = "run_kt_35b.sh"
elif MODEL == "397b":
    SCRIPT = "run_kt_397b.sh"
else:
    print(f"Unknown model: {MODEL}")
    sys.exit(1)

RESULT_FILE = f"/tmp/kt_bench_{MODEL}_results.txt"
with open(RESULT_FILE, "w") as f:
    f.write("mode | decode_tok_s | prefill_tok_s | prefill_s | peak_sys_used_gib | peak_gpu_gib\n")

URL = f"http://localhost:{PORT}/v1/completions"


def run(cmd, timeout=30):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)


def wait_ready(log_file, server_pid, max_wait=900):
    """Wait for server to be ready."""
    for i in range(max_wait // 5):
        time.sleep(5)
        try:
            with open(log_file) as f:
                content = f.read()
            if "fired up and ready to roll" in content:
                return True
            for fatal in ["Segmentation fault", "CUDA error", "RuntimeError:",
                          "Traceback (most recent call last):", "core dumped", "Out of memory"]:
                if fatal in content:
                    print(f"FATAL: {fatal}")
                    return False
        except:
            pass
        try:
            os.kill(server_pid, 0)
        except ProcessLookupError:
            print("Server process exited")
            return False
    return False


def sample_memory(stop_event, samples):
    """Background thread to sample memory every 2 seconds."""
    while not stop_event.is_set():
        try:
            out = subprocess.check_output(["free", "-m"], text=True)
            for line in out.splitlines():
                if line.startswith("Mem:"):
                    used = int(line.split()[2])
                    samples.append(("mem", used))
                    break
        except:
            pass
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True
            )
            gpu_sum = sum(int(x) for x in out.strip().split("\n")[:4])
            samples.append(("gpu", gpu_sum))
        except:
            pass
        time.sleep(2)


def bench_prefill():
    """Benchmark prefill with 2048-token prompt (CPU path, < 4096 threshold)."""
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


def bench_decode():
    """Benchmark decode with 256 tokens."""
    data = {"model": "default", "prompt": "Write a short essay about artificial intelligence and its impact on society:",
            "max_tokens": 256, "temperature": 0, "stream": True}
    start = time.time()
    first = None
    cnt = 0
    r = requests.post(URL, json=data, stream=True, timeout=120)
    for line in r.iter_lines():
        if line:
            line = line.decode()
            if line.startswith("data: "):
                line = line[6:]
                if line == "[DONE]":
                    break
                chunk = json.loads(line)
                if chunk["choices"][0]["text"]:
                    cnt += 1
                    if first is None:
                        first = time.time()
    end = time.time()
    dec = end - first if first else end - start
    dtps = cnt / dec if dec > 0 else 0
    return dtps


print(f"\n========== Testing standard kt {MODEL} (no MESH) ==========")

# Stop old server
run("pkill -f 'sglang.launch_server.*{}'".format(PORT))
time.sleep(5)

# Start server (no cap argument for standard kt)
log_file = f"/tmp/sglang_kt_{MODEL}.log"
server_cmd = f"cd /mnt/data2/tmp/qujing_mesh && bash {SCRIPT} {PORT} {CUDA}"
proc = subprocess.Popen(server_cmd, shell=True, stdout=open(log_file, "w"), stderr=subprocess.STDOUT)
print(f"Started server PID={proc.pid}, waiting for ready...")

if not wait_ready(log_file, proc.pid):
    print(f"FAILED: Server not ready")
    with open(RESULT_FILE, "a") as f:
        f.write(f"standard_kt | FAILED | FAILED | FAILED | FAILED | FAILED\n")
    run("pkill -f 'sglang.launch_server.*{}'".format(PORT))
    sys.exit(1)

print(f"Server ready")
time.sleep(5)

# Start memory sampling
samples = []
stop_event = threading.Event()
mem_thread = threading.Thread(target=sample_memory, args=(stop_event, samples))
mem_thread.start()
print("Memory sampler started")

# Prefill benchmark
try:
    pt, ptime, ptps = bench_prefill()
    print(f"Prefill: {pt} tokens, {ptime:.2f}s, {ptps:.2f} tok/s")
except Exception as e:
    print(f"Prefill failed: {e}")
    pt, ptime, ptps = 0, 0, 0

time.sleep(3)

# Decode benchmark
try:
    dtps = bench_decode()
    print(f"Decode: {dtps:.2f} tok/s")
except Exception as e:
    print(f"Decode failed: {e}")
    dtps = 0

# Stop memory sampling
stop_event.set()
mem_thread.join()

# Calculate peaks
mem_vals = [v for k, v in samples if k == "mem"]
gpu_vals = [v for k, v in samples if k == "gpu"]
peak_mem = max(mem_vals) / 1024 if mem_vals else 0  # GiB
peak_gpu = max(gpu_vals) / 1024 if gpu_vals else 0  # GiB
print(f"Peak sys used: {peak_mem:.2f} GiB, Peak GPU: {peak_gpu:.2f} GiB, Samples: {len(mem_vals)}")

# Write result
with open(RESULT_FILE, "a") as f:
    f.write(f"standard_kt | {dtps:.2f} | {ptps:.2f} | {ptime:.2f} | {peak_mem:.2f} | {peak_gpu:.2f}\n")

# Stop server
run("pkill -f 'sglang.launch_server.*{}'".format(PORT))
time.sleep(5)

print("\n=== Results ===")
with open(RESULT_FILE) as f:
    print(f.read())
