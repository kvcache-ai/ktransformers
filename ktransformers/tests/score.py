import subprocess
import time

server_cmd = [
    "/home/qujing3/anaconda3/envs/ktransformers-dev/bin/ktransformers",
    "--model_path", "/home/qujing3/models/DeepSeek-R1-Q4_K_M/config",
    "--gguf_path", "/home/qujing3/models/DeepSeek-R1-Q4_K_M/",
    "--port", "10002",
    "--cpu-infer", "48"
]

print("Starting ktransformers server...")
server_process = subprocess.Popen(server_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

while True:
    output = server_process.stdout.readline()
    if not output:
        break
    print(output.strip())
    if "Uvicorn running on http://0.0.0.0:10002" in output:
        print("Server started successfully!")
        break

eval_cmd = ["python", "ktransformers/tests/humaneval/eval_api.py"]
print("Running eval_api.py...")
eval_process = subprocess.run(eval_cmd, capture_output=True, text=True)

print("Stopping ktransformers server...")
server_process.terminate()
server_process.wait()

evaluate_cmd = [
    "evaluate_functional_correctness",
    "ktransformers/tests/humaneval/results/api/eval_b.jsonl"
]
print("Running evaluate_functional_correctness...")
evaluate_process = subprocess.run(evaluate_cmd, capture_output=True, text=True)

print("Evaluation Output:")
print(evaluate_process.stdout)
print(evaluate_process.stderr)
