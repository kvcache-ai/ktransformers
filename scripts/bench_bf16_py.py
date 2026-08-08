#!/usr/bin/env python3
"""通用 benchmark 脚本 — 支持 bf16 和 AMXINT4，标准和 MESH 模式。
Usage:
  标准 kt:   python3 bench_bf16_py.py kt <MODEL> <PORT> <CUDA>
  MESH:      python3 bench_bf16_py.py mesh <MODEL> <CAP_LIST> <PORT> <CUDA>

Example:
  python3 bench_bf16_py.py kt 35b 10004 0,1,2,3
  python3 bench_bf16_py.py kt 397b 10004 0,1,2,3
  python3 bench_bf16_py.py mesh 35b "64 128 192 224" 10004 0,1,2,3
  python3 bench_bf16_py.py mesh 397b "128 192 256 480" 10004 0,1,2,3
"""
import subprocess, time, json, threading, requests, sys, os

MODE = sys.argv[1]  # kt / mesh
MODEL = sys.argv[2]  # 35b / 397b

if MODE == "kt":
    PORT = int(sys.argv[3])
    CUDA = sys.argv[4]
    SCRIPT = f"run_kt_{MODEL}_bf16.sh"
    CAP_LIST = [None]
    RESULT_FILE = f"/tmp/kt_bench_{MODEL}_bf16_results.txt"
elif MODE == "mesh":
    CAP_LIST = [int(x) for x in sys.argv[3].split()]
    PORT = int(sys.argv[4])
    CUDA = sys.argv[5]
    SCRIPT = f"run_mesh_{MODEL}_bf16.sh"
    RESULT_FILE = f"/tmp/mesh_bench_{MODEL}_bf16_results.txt"
else:
    print(f"Unknown mode: {MODE}")
    sys.exit(1)

with open(RESULT_FILE, "w") as f:
    f.write("mode | decode_tok_s | prefill_tok_s | prefill_s | peak_sys_used_gib | peak_gpu_gib\n")

URL = f"http://localhost:{PORT}/v1/completions"


def run(cmd, timeout=30):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)


def wait_ready(log_file, server_pid, max_wait=900):
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


for cap in CAP_LIST:
    label = f"cap={cap}" if cap else "standard_kt"
    print(f"\n========== Testing {MODEL} bf16 {label} ==========")

    run("pkill -f 'sglang.launch_server.*{}'".format(PORT))
    time.sleep(5)

    log_file = f"/tmp/sglang_bf16_{MODEL}_{label}.log"
    if cap:
        server_cmd = f"cd /mnt/data2/tmp/qujing_mesh && bash {SCRIPT} {cap} {PORT} {CUDA}"
    else:
        server_cmd = f"cd /mnt/data2/tmp/qujing_mesh && bash {SCRIPT} {PORT} {CUDA}"
    proc = subprocess.Popen(server_cmd, shell=True, stdout=open(log_file, "w"), stderr=subprocess.STDOUT)
    print(f"Started server PID={proc.pid}, waiting for ready...")

    if not wait_ready(log_file, proc.pid):
        print(f"FAILED: Server not ready for {label}")
        with open(RESULT_FILE, "a") as f:
            f.write(f"{label} | FAILED | FAILED | FAILED | FAILED | FAILED\n")
        run("pkill -f 'sglang.launch_server.*{}'".format(PORT))
        time.sleep(5)
        continue

    print(f"Server ready")
    time.sleep(5)

    samples = []
    stop_event = threading.Event()
    mem_thread = threading.Thread(target=sample_memory, args=(stop_event, samples))
    mem_thread.start()
    print("Memory sampler started")

    try:
        pt, ptime, ptps = bench_prefill()
        print(f"Prefill: {pt} tokens, {ptime:.2f}s, {ptps:.2f} tok/s")
    except Exception as e:
        print(f"Prefill failed: {e}")
        pt, ptime, ptps = 0, 0, 0

    time.sleep(3)

    try:
        dtps = bench_decode()
        print(f"Decode: {dtps:.2f} tok/s")
    except Exception as e:
        print(f"Decode failed: {e}")
        dtps = 0

    stop_event.set()
    mem_thread.join()

    mem_vals = [v for k, v in samples if k == "mem"]
    gpu_vals = [v for k, v in samples if k == "gpu"]
    peak_mem = max(mem_vals) / 1024 if mem_vals else 0
    peak_gpu = max(gpu_vals) / 1024 if gpu_vals else 0
    print(f"Peak sys used: {peak_mem:.2f} GiB, Peak GPU: {peak_gpu:.2f} GiB, Samples: {len(mem_vals)}")

    with open(RESULT_FILE, "a") as f:
        f.write(f"{label} | {dtps:.2f} | {ptps:.2f} | {ptime:.2f} | {peak_mem:.2f} | {peak_gpu:.2f}\n")

    run("pkill -f 'sglang.launch_server.*{}'".format(PORT))
    time.sleep(5)

print("\n=== Results ===")
with open(RESULT_FILE) as f:
    print(f.read())
