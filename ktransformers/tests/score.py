import subprocess
import time
import requests
import sys
import os

def wait_for_server(base_url: str, timeout: int = None) -> None:
    start_time = time.time()
    while True:
        try:
            response = requests.get(
                f"{base_url}/v1/models",
                headers={"Authorization": "Bearer None"},
            )
            if response.status_code == 200:
                print("Server is ready.")
                break
        except requests.exceptions.RequestException:
            time.sleep(1)
            if timeout and time.time() - start_time > timeout:
                raise TimeoutError("Server did not become ready within timeout period")

server_cmd = [
    "numactl", "-N", "1", "-m", "1",
    "/home/qujing3/anaconda3/envs/ktransformers-dev/bin/ktransformers",
    "--model_path", "/home/qujing3/models/DeepSeek-R1-Q4_K_M/config",
    "--gguf_path", "/home/qujing3/models/DeepSeek-V3-GGUF/DeepSeek-V3-Q4_K_M",
    "--port", "10002",
    "--cpu_infer", "48",
    "--optimize_config_path", "ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat.yaml",
    "--max_new_tokens", "3000",
    "--cache_lens", "6000"
]

print("Starting ktransformers server...")
print(" ".join(server_cmd))
with open("/tmp/server_log.txt", "w") as f:
    server_process = subprocess.Popen(server_cmd, stdout=f, stderr=f, text=True)

try:
    wait_for_server("http://localhost:10002", timeout=600)

    eval_cmd = ["python", "ktransformers/tests/humaneval/eval_api.py"]
    print("Running eval_api.py...")
    print(f"Command: {' '.join(eval_cmd)}")
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    eval_process = subprocess.Popen(
        eval_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
        universal_newlines=True
    )
    
    import threading
    import queue
    
    def enqueue_output(out, queue):
        for line in iter(out.readline, ''):
            queue.put(line)
        out.close()
    
    stdout_queue = queue.Queue()
    stderr_queue = queue.Queue()
    
    stdout_thread = threading.Thread(target=enqueue_output, args=(eval_process.stdout, stdout_queue))
    stderr_thread = threading.Thread(target=enqueue_output, args=(eval_process.stderr, stderr_queue))
    
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()
    
    while eval_process.poll() is None:
        try:
            line = stdout_queue.get_nowait()
            print(line, end='', flush=True)
        except queue.Empty:
            pass
            
        try:
            line = stderr_queue.get_nowait()
            print(line, end='', file=sys.stderr, flush=True)
        except queue.Empty:
            pass
        
        time.sleep(1)

    while not stdout_queue.empty():
        print(stdout_queue.get(), end='', flush=True)
    while not stderr_queue.empty():
        print(stderr_queue.get(), end='', file=sys.stderr, flush=True)
        
    eval_process.wait()
    print(f"eval_api.py completed with exit code: {eval_process.returncode}")

    evaluate_cmd = [
        "evaluate_functional_correctness",
        "ktransformers/tests/humaneval/results/api/eval_b.jsonl"
    ]
    print("Running evaluate_functional_correctness...")
    print(f"Command: {' '.join(evaluate_cmd)}")
    
    evaluate_process = subprocess.Popen(
        evaluate_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    for line in evaluate_process.stdout:
        print(line, end='', flush=True)
    for line in evaluate_process.stderr:
        print(line, end='', file=sys.stderr, flush=True)
        
    evaluate_process.wait()
    
    print(f"evaluate_functional_correctness completed with exit code: {evaluate_process.returncode}")
    if evaluate_process.returncode != 0:
        print(f"evaluate_functional_correctness exited with code {evaluate_process.returncode}")
        sys.exit(evaluate_process.returncode)

finally:
    print("Stopping ktransformers server...")
    server_process.terminate()
    try:
        server_process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        print("Server did not terminate gracefully, forcing...")
        server_process.kill()