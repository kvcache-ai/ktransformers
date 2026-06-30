#!/usr/bin/env python3
"""完整 benchmark 脚本 — 支持所有格式，采集所有规范指标。

采集指标：
  1. decode tok/s（256 tokens）
  2. 短 prompt prefill tok/s（~2601 tokens）
  3. 长 prompt prefill tok/s（~8192 tokens）
  4. peak_sys_used_gib（从 server 启动前开始采样，覆盖 bootstrap）
  5. peak_gpu_gib
  6. MESH stats（hit_rate、iouring_read_gib 等，从日志 grep）
  7. iostat 磁盘读取带宽（r_MB_s）

Usage:
  kt 模式:   python3 bench_full_py.py kt <MODEL> <FORMAT> <PORT> <CUDA>
  mesh 模式: python3 bench_full_py.py mesh <MODEL> <FORMAT> <CAP_LIST> <PORT> <CUDA>

  MODEL:  35b / 397b
  FORMAT: amxint4 / bf16

Example:
  python3 bench_full_py.py kt 35b amxint4 10004 0,1,2,3
  python3 bench_full_py.py kt 397b bf16 10004 0,1,2,3
  python3 bench_full_py.py mesh 35b amxint4 "64 128 192 224" 10004 0,1,2,3
  python3 bench_full_py.py mesh 397b bf16 "128 192 256 480" 10004 0,1,2,3
"""
import subprocess, time, json, threading, requests, sys, os, re

MODE = sys.argv[1]    # kt / mesh
MODEL = sys.argv[2]   # 35b / 397b
FORMAT = sys.argv[3]  # amxint4 / bf16

if MODE == "kt":
    PORT = int(sys.argv[4])
    CUDA = sys.argv[5]
    CAP_LIST = [None]
    SCRIPT = f"run_kt_{MODEL}_{FORMAT}.sh" if FORMAT != "amxint4" else f"run_kt_{MODEL}.sh"
    RESULT_FILE = f"/tmp/full_bench_{MODEL}_{FORMAT}_kt_results.txt"
elif MODE == "mesh":
    CAP_LIST = [int(x) for x in sys.argv[4].split()]
    PORT = int(sys.argv[5])
    CUDA = sys.argv[6]
    SCRIPT = f"run_mesh_{MODEL}_cap.sh" if FORMAT != "bf16" else f"run_mesh_{MODEL}_bf16.sh"
    RESULT_FILE = f"/tmp/full_bench_{MODEL}_{FORMAT}_mesh_results.txt"
else:
    print(f"Unknown mode: {MODE}")
    sys.exit(1)

with open(RESULT_FILE, "w") as f:
    f.write("mode | decode_tok_s | prefill_short_tok_s | prefill_long_tok_s | "
            "peak_sys_gib | peak_gpu_gib | hit_rate | iouring_read_gib | "
            "iostat_r_mbs | eviction_count | decode_tokens\n")

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
    """从 server 启动前开始采样，覆盖 bootstrap + 测试全过程。"""
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


def start_iostat(iostat_file):
    """启动 iostat 后台采样，每 5 秒一次。"""
    # 采样所有设备，解析时筛选数据盘
    cmd = f"iostat -x -d 5 > {iostat_file} 2>&1"
    proc = subprocess.Popen(cmd, shell=True)
    return proc


def parse_iostat(iostat_file):
    """解析 iostat 输出，返回数据盘的峰值读取带宽 (MB/s)。

    iostat -x -d 输出格式：
    Device  r/s  rkB/s  rrqm/s  %rrqm  r_await  rareq-sz  w/s  wkB/s  ...
    dm-1    203  35097  0       0      3.77     172       220  1308   ...

    返回 (peak_MB/s, avg_MB/s)
    """
    try:
        with open(iostat_file) as f:
            content = f.read()
        # 采样每轮所有非 loop/ram 设备的 rkB/s 总和
        values = []
        current_round = {}
        for line in content.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0] not in ("Device", "Linux", ""):
                dev = parts[0]
                if dev.startswith(("loop", "ram", "sr")):
                    continue
                try:
                    rkbs = float(parts[2])  # rkB/s
                    current_round[dev] = current_round.get(dev, 0) + rkbs
                except (ValueError, IndexError):
                    pass
            elif parts and parts[0] == "Device":
                if current_round:
                    values.append(sum(current_round.values()))
                current_round = {}
        if current_round:
            values.append(sum(current_round.values()))

        if values:
            peak_mbs = max(values) / 1024
            avg_mbs = sum(values) / len(values) / 1024
            return peak_mbs, avg_mbs
        return 0.0, 0.0
    except:
        return 0.0, 0.0


def bench_prefill_short():
    """短 prompt prefill (~2601 tokens)。"""
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


def bench_prefill_long():
    """长 prompt prefill (~8192 tokens)。"""
    prompt = "The quick brown fox jumps over the lazy dog. " * 820
    data = {"model": "default", "prompt": prompt, "max_tokens": 1, "temperature": 0, "stream": False}
    start = time.time()
    r = requests.post(URL, json=data, timeout=600)
    end = time.time()
    resp = r.json()
    pt = resp.get("usage", {}).get("prompt_tokens", 0)
    elapsed = end - start
    tps = pt / elapsed if elapsed > 0 else 0
    return pt, elapsed, tps


def bench_decode():
    """Decode 256 tokens。"""
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


def grep_mesh_stats(log_file):
    """从 server 日志 grep 最后一次 MESH_STATS_KV 输出。"""
    try:
        with open(log_file) as f:
            content = f.read()
        # 找最后一次 [MESH_STATS_KV] 行
        matches = re.findall(r'\[MESH_STATS_KV\]\s*(.*)', content)
        if matches:
            kv_line = matches[-1].strip()
            # 解析 key=value 对
            stats = {}
            for pair in kv_line.split():
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    stats[k] = v
            return stats
    except:
        pass
    return {}


for cap in CAP_LIST:
    label = f"cap={cap}" if cap else "standard_kt"
    print(f"\n========== Testing {MODEL} {FORMAT} {label} ==========")

    run("pkill -f 'sglang.launch_server.*{}'".format(PORT))
    time.sleep(5)

    log_file = f"/tmp/sglang_full_{MODEL}_{FORMAT}_{label}.log"
    iostat_file = f"/tmp/iostat_{MODEL}_{FORMAT}_{label}.log"

    # 1. 在 server 启动前开始内存采样（修复时序问题）
    samples = []
    stop_event = threading.Event()
    mem_thread = threading.Thread(target=sample_memory, args=(stop_event, samples))
    mem_thread.start()
    print("Memory sampler started (pre-server)")

    # 2. 启动 iostat 采样
    iostat_proc = start_iostat(iostat_file)
    print(f"iostat started (PID={iostat_proc.pid})")

    # 3. 启动 server
    if cap:
        server_cmd = f"cd /mnt/data2/tmp/qujing_mesh && bash {SCRIPT} {cap} {PORT} {CUDA}"
    else:
        server_cmd = f"cd /mnt/data2/tmp/qujing_mesh && bash {SCRIPT} {PORT} {CUDA}"
    proc = subprocess.Popen(server_cmd, shell=True, stdout=open(log_file, "w"), stderr=subprocess.STDOUT)
    print(f"Started server PID={proc.pid}, waiting for ready...")

    if not wait_ready(log_file, proc.pid):
        print(f"FAILED: Server not ready for {label}")
        with open(RESULT_FILE, "a") as f:
            f.write(f"{label} | FAILED | FAILED | FAILED | FAILED | FAILED | - | - | - | - | -\n")
        stop_event.set()
        mem_thread.join()
        iostat_proc.kill()
        run("pkill -f 'sglang.launch_server.*{}'".format(PORT))
        time.sleep(5)
        continue

    print(f"Server ready")
    time.sleep(5)

    # 4. 短 prompt prefill（< 4096 threshold，走 GPU prefill）
    try:
        pt_s, ptime_s, ptps_s = bench_prefill_short()
        print(f"Prefill short: {pt_s} tokens, {ptime_s:.2f}s, {ptps_s:.2f} tok/s")
    except Exception as e:
        print(f"Prefill short failed: {e}")
        pt_s, ptime_s, ptps_s = 0, 0, 0

    time.sleep(3)

    # 5. Decode（先于长 prompt prefill，确保 MESH stats 能输出）
    try:
        dtps = bench_decode()
        print(f"Decode: {dtps:.2f} tok/s")
    except Exception as e:
        print(f"Decode failed: {e}")
        dtps = 0

    time.sleep(3)

    # 6. 长 prompt prefill（最后执行，可能触发 GPU prefill bug 导致 server 崩溃）
    try:
        pt_l, ptime_l, ptps_l = bench_prefill_long()
        print(f"Prefill long: {pt_l} tokens, {ptime_l:.2f}s, {ptps_l:.2f} tok/s")
    except Exception as e:
        print(f"Prefill long failed: {e}")
        pt_l, ptime_l, ptps_l = 0, 0, 0

    # 7. 停止采样
    time.sleep(3)  # 等待最后的 MESH stats 输出
    stop_event.set()
    mem_thread.join()
    iostat_proc.kill()

    # 8. 计算峰值
    mem_vals = [v for k, v in samples if k == "mem"]
    gpu_vals = [v for k, v in samples if k == "gpu"]
    peak_mem = max(mem_vals) / 1024 if mem_vals else 0
    peak_gpu = max(gpu_vals) / 1024 if gpu_vals else 0
    print(f"Peak sys used: {peak_mem:.2f} GiB, Peak GPU: {peak_gpu:.2f} GiB, Samples: {len(mem_vals)}")

    # 9. 解析 iostat
    iostat_peak, iostat_avg = parse_iostat(iostat_file)
    print(f"iostat: peak={iostat_peak:.1f} MB/s, avg={iostat_avg:.1f} MB/s")

    # 10. 从日志 grep MESH stats
    mesh_stats = grep_mesh_stats(log_file)
    hit_rate = float(mesh_stats.get("hit_rate", 0))
    iouring_gib = float(mesh_stats.get("iouring_read_gib", 0))
    eviction_count = int(mesh_stats.get("eviction_count", 0))
    decode_tokens = int(mesh_stats.get("decode_tokens", 0))
    if mesh_stats:
        print(f"MESH stats: hit_rate={hit_rate:.4f}, iouring_read={iouring_gib:.2f} GiB, "
              f"evictions={eviction_count}, decode_tokens={decode_tokens}")
    else:
        print("No MESH stats found in log")

    # 11. 写入结果
    with open(RESULT_FILE, "a") as f:
        f.write(f"{label} | {dtps:.2f} | {ptps_s:.2f} | {ptps_l:.2f} | "
                f"{peak_mem:.2f} | {peak_gpu:.2f} | "
                f"{hit_rate:.4f} | {iouring_gib:.2f} | "
                f"{iostat_peak:.1f} | {eviction_count} | {decode_tokens}\n")

    run("pkill -f 'sglang.launch_server.*{}'".format(PORT))
    time.sleep(5)

print("\n=== Results ===")
with open(RESULT_FILE) as f:
    print(f.read())
