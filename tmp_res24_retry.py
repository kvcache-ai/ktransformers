import os, subprocess, time, requests, json, re
from pathlib import Path

MODEL_PATH = '/mnt/data2/models/Qwen3.5-35B-A3B'
VENV = '/mnt/data2/work/venv-ktransformers-raqiu/bin'
LOG = Path('/mnt/data2/work/16g_wave_hold_baseline_probe_res24_retry.log')
SAMPLES = Path('/mnt/data2/work/16g_wave_hold_baseline_probe_res24_retry_samples.jsonl')
PORT = 31672
BASE_ENV = os.environ.copy()
BASE_ENV.update({
    'PATH': f'{VENV}:' + BASE_ENV.get('PATH', ''),
    'CUDA_VISIBLE_DEVICES': '4,5',
    'SGLANG_DISABLE_CUDNN_CHECK': '1',
    'SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK': '0',
    'KT_MAX_RESIDENT_EXPERTS': '24',
    'KT_ENABLE_BF16_WAVE_RESIDENT': '1',
    'KT_LANG': 'en',
    'PYTHONFAULTHANDLER': '1',
    'KT_TRACE_CPU_MEM_BREAKDOWN': '1',
    'TORCH_SHOW_CPP_STACKTRACES': '1',
})
for cmd in [
    "pkill -KILL -f 'sglang.launch_server.*31670|kt run /mnt/data2/models/Qwen3.5-35B-A3B --port 31670'",
]:
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for p in [LOG, SAMPLES]:
    if p.exists():
        p.unlink()
cmd = [
    'systemd-run', '--user', '--scope', '-p', 'MemoryMax=16G', '-p', 'MemorySwapMax=0',
    f'{VENV}/kt', 'run', MODEL_PATH,
    '--port', str(PORT),
    '--gpu-experts', '16',
    '--cpu-threads', '96',
    '--numa-nodes', '2',
    '--tensor-parallel-size', '2',
    '--kt-method', 'BF16',
    '--weight-strategy', 'tiered',
    '--residency-policy', 'baseline',
    '--served-model-name', 'Qwen3.5-35B-A3B-16g-wave-probe-res24_retry',
    '--max-running-requests', '1',
    '--max-total-tokens', '4096',
    '--chunked-prefill-size', '2048',
    '--disable-shared-experts-fusion',
]
with LOG.open('w') as f:
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=BASE_ENV, preexec_fn=os.setsid)

scope = None
cg_base = None
start = time.time()
status = 'timeout'
health = f'http://127.0.0.1:{PORT}/health'
model = f'http://127.0.0.1:{PORT}/model_info'
pat = re.compile(r'Running scope as unit: (.*\.scope)')
last_oom_kill = 0
while time.time() - start < 240:
    if scope is None and LOG.exists():
        text = LOG.read_text(errors='ignore')
        m = pat.search(text)
        if m:
            scope = m.group(1)
            cg = subprocess.check_output(['systemctl', '--user', 'show', '-p', 'ControlGroup', '--value', scope], text=True).strip()
            cg_base = Path('/sys/fs/cgroup' + cg)
    if cg_base is not None and cg_base.exists():
        sample = {'t': round(time.time() - start, 3)}
        for name in ['memory.current', 'memory.peak']:
            p = cg_base / name
            if p.exists():
                sample[name.replace('.', '_')] = p.read_text().strip()
        stat = {}
        sp = cg_base / 'memory.stat'
        if sp.exists():
            for line in sp.read_text().splitlines():
                k, v = line.split()
                if k in {'anon','file','shmem','file_mapped','pagetables','kernel_stack','slab','inactive_anon','active_anon','inactive_file','active_file'}:
                    stat[k] = int(v)
        sample.update(stat)
        ep = cg_base / 'memory.events'
        if ep.exists():
            events = {}
            for line in ep.read_text().splitlines():
                k, v = line.split()
                events[k] = int(v)
            sample.update({f'evt_{k}': v for k, v in events.items()})
            last_oom_kill = events.get('oom_kill', 0)
        with SAMPLES.open('a') as f:
            f.write(json.dumps(sample) + '\n')
        if last_oom_kill > 0:
            status = 'oom_kill'
            break
    try:
        if requests.get(health, timeout=2).status_code == 200:
            status = 'ready'
            break
    except Exception:
        pass
    try:
        requests.get(model, timeout=2)
    except Exception:
        pass
    time.sleep(0.2)
print(json.dumps({'pid': proc.pid, 'scope': scope, 'status': status, 'samples': str(SAMPLES), 'log': str(LOG)}))
for _ in range(120):
    time.sleep(5)
