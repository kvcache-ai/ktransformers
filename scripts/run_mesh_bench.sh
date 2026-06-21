#!/bin/bash
# 自动化 MESH benchmark：依次测试多个 cap 值
# 用法: run_mesh_bench.sh <MODEL> <CAP_LIST> <PORT> <CUDA>
# MODEL: 35b / 397b
# CAP_LIST: 空格分隔的 cap 值，如 "64 128 192 224"
# PORT: 端口
# CUDA: CUDA 设备列表，如 0,1,2,3

MODEL=$1
CAP_LIST=$2
PORT=$3
CUDA=$4

if [ "$MODEL" = "35b" ]; then
    SCRIPT="run_mesh_35b_cap.sh"
elif [ "$MODEL" = "397b" ]; then
    SCRIPT="run_mesh_397b_cap.sh"
else
    echo "Unknown model: $MODEL (use 35b or 397b)"
    exit 1
fi

RESULT_FILE="/tmp/mesh_bench_${MODEL}_results.txt"
echo "=== MESH ${MODEL} benchmark results ===" > $RESULT_FILE
echo "cap | tokens | total_s | prefill_s | decode_s | decode_tok_s" >> $RESULT_FILE

for CAP in $CAP_LIST; do
    echo ""
    echo "========== Testing ${MODEL} cap=${CAP} =========="

    # 停旧服务
    pkill -f "sglang.launch_server.*${PORT}" 2>/dev/null
    sleep 3

    # 启动新服务
    cd /mnt/data2/tmp/qujing_mesh
    nohup bash $SCRIPT $CAP $PORT $CUDA > /tmp/sglang_mesh_${MODEL}_cap${CAP}.log 2>&1 &
    SERVER_PID=$!
    echo "Started server PID=$SERVER_PID, waiting for ready..."

    # 等待服务就绪（最多 10 分钟，MESH 启动慢）
    READY=0
    for i in $(seq 1 120); do
        sleep 5
        if grep -q "fired up and ready to roll" /tmp/sglang_mesh_${MODEL}_cap${CAP}.log 2>/dev/null; then
            READY=1
            echo "Server ready after $((i*5))s"
            break
        fi
        # 只检测真正致命的错误，忽略 "import error"/"side-effect import failed" 等正常警告
        if grep -qi "Segmentation fault\|CUDA error\|RuntimeError:\|AssertionError:\|Traceback (most recent call last):\|core dumped\|Out of memory" /tmp/sglang_mesh_${MODEL}_cap${CAP}.log 2>/dev/null; then
            echo "FATAL error detected in log:"
            grep -i "Segmentation fault\|CUDA error\|RuntimeError:\|AssertionError:\|Traceback (most recent call last):\|core dumped\|Out of memory" /tmp/sglang_mesh_${MODEL}_cap${CAP}.log | tail -5
            READY=0
            break
        fi
        # 检查服务进程是否已退出
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "Server process exited unexpectedly"
            tail -20 /tmp/sglang_mesh_${MODEL}_cap${CAP}.log
            READY=0
            break
        fi
    done

    if [ $READY -eq 0 ]; then
        echo "FAILED: Server not ready for cap=${CAP}"
        echo "${CAP} | FAILED" >> $RESULT_FILE
        pkill -f "sglang.launch_server.*${PORT}" 2>/dev/null
        sleep 3
        continue
    fi

    sleep 3  # 额外等待稳定

    # 功能测试
    FUNC_OUT=$(curl -s http://localhost:${PORT}/v1/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"default","prompt":"Hello, my name is","max_tokens":16,"temperature":0}' | \
        python3 -c "import json,sys; r=json.load(sys.stdin); print(r['choices'][0]['text'][:50])" 2>/dev/null)
    echo "Function test output: ${FUNC_OUT}"

    # 测速
    source /mnt/data2/tmp/qujing_mesh/.venv/bin/activate
    SPEED_OUT=$(python3 -c "
import requests, time, json
url = 'http://localhost:${PORT}/v1/completions'
data = {'model':'default','prompt':'Write a short essay about artificial intelligence and its impact on society:','max_tokens':256,'temperature':0,'stream':True}
start = time.time()
first_token_time = None
token_count = 0
r = requests.post(url, json=data, stream=True)
for line in r.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            line = line[6:]
            if line == '[DONE]':
                break
            chunk = json.loads(line)
            if chunk['choices'][0]['text']:
                token_count += 1
                if first_token_time is None:
                    first_token_time = time.time()
end = time.time()
total = end - start
prefill = first_token_time - start if first_token_time else 0
decode = end - first_token_time if first_token_time else total
dtps = token_count/decode if decode > 0 else 0
print(f'{token_count}|{total:.2f}|{prefill:.2f}|{decode:.2f}|{dtps:.2f}')
" 2>/dev/null)

    echo "Speed: ${SPEED_OUT}"
    echo "${CAP} | ${SPEED_OUT}" >> $RESULT_FILE

    # 停服务
    pkill -f "sglang.launch_server.*${PORT}" 2>/dev/null
    sleep 3
done

echo ""
echo "=== Results ==="
cat $RESULT_FILE
