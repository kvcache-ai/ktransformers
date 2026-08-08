#!/bin/bash
# 自动化 MESH benchmark：依次测试多个 cap 值
# 用法: run_mesh_bench.sh <MODEL> <CAP_LIST> <PORT> <CUDA>
# MODEL: 35b / 397b
# CAP_LIST: 空格分隔的 cap 值，如 "64 128 192 224"
# PORT: 端口
# CUDA: CUDA 设备列表，如 0,1,2,3
#
# 输出指标：decode tok/s, prefill tok/s, peak RSS, peak GPU mem, peak anon, peak file

MODEL=$1
CAP_LIST=$2
PORT=$3
CUDA=$4

if [ "$MODEL" = "35b" ]; then
    SCRIPT="run_mesh_35b_cap.sh"
    PREFILL_TOKENS=8192
elif [ "$MODEL" = "397b" ]; then
    SCRIPT="run_mesh_397b_cap.sh"
    PREFILL_TOKENS=8192
else
    echo "Unknown model: $MODEL (use 35b or 397b)"
    exit 1
fi

RESULT_FILE="/tmp/mesh_bench_${MODEL}_results.txt"
echo "=== MESH ${MODEL} benchmark results ===" > $RESULT_FILE
echo "cap | decode_tok_s | prefill_tok_s | prefill_s | peak_rss_gib | peak_sys_used_gib | peak_gpu_gib | peak_anon_gib | peak_file_gib" >> $RESULT_FILE

for CAP in $CAP_LIST; do
    echo ""
    echo "========== Testing ${MODEL} cap=${CAP} =========="

    # 停旧服务
    pkill -f "sglang.launch_server.*${PORT}" 2>/dev/null
    sleep 5

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
        # 只检测真正致命的错误
        if grep -qi "Segmentation fault\|CUDA error\|RuntimeError:\|AssertionError:\|Traceback (most recent call last):\|core dumped\|Out of memory" /tmp/sglang_mesh_${MODEL}_cap${CAP}.log 2>/dev/null; then
            echo "FATAL error detected in log:"
            grep -i "Segmentation fault\|CUDA error\|RuntimeError:\|AssertionError:\|Traceback (most recent call last):\|core dumped\|Out of memory" /tmp/sglang_mesh_${MODEL}_cap${CAP}.log | tail -5
            READY=0
            break
        fi
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "Server process exited unexpectedly"
            tail -20 /tmp/sglang_mesh_${MODEL}_cap${CAP}.log
            READY=0
            break
        fi
    done

    if [ $READY -eq 0 ]; then
        echo "FAILED: Server not ready for cap=${CAP}"
        echo "${CAP} | FAILED | FAILED | FAILED | FAILED | FAILED | FAILED | FAILED | FAILED" >> $RESULT_FILE
        pkill -f "sglang.launch_server.*${PORT}" 2>/dev/null
        sleep 5
        continue
    fi

    sleep 5  # 额外等待稳定

    # 启动内存采样（后台运行，每 2 秒采样一次）
    MEM_SAMPLE_FILE="/tmp/mesh_mem_samples_${MODEL}_cap${CAP}.csv"
    echo "timestamp,rss_kb,sys_used_mb,sys_avail_mb,anon_kb,file_kb,gpu0_mb,gpu1_mb,gpu2_mb,gpu3_mb" > $MEM_SAMPLE_FILE
    (
        while true; do
            # 采样所有 sglang 相关进程的 RSS 总和（用 ps 匹配，更可靠）
            RSS_TOTAL=0
            ANON_TOTAL=0
            FILE_TOTAL=0
            for pid in $(ps -eo pid,rss,args | grep -i "sglang\|launch_server\|sglang.srt" | grep -v grep | awk '{print $1}'); do
                if [ -f "/proc/$pid/status" ]; then
                    rss=$(grep "VmRSS" /proc/$pid/status 2>/dev/null | awk '{print $2}')
                    RSS_TOTAL=$((RSS_TOTAL + ${rss:-0}))
                fi
                if [ -r "/proc/$pid/smaps_rollup" ]; then
                    anon=$(grep "^Anonymous:" /proc/$pid/smaps_rollup 2>/dev/null | awk '{print $2}')
                    ANON_TOTAL=$((ANON_TOTAL + ${anon:-0}))
                    # file-backed = RSS - Anonymous（smaps_rollup 无 File: 字段）
                    FILE_TOTAL=$((RSS_TOTAL - ANON_TOTAL))
                fi
            done
            # 系统总内存 + page cache（Cached 列）
            SYS_MEM=$(free -m | awk '/^Mem:/ {printf "%d,%d", $3, $7}')
            CACHE_MB=$(grep "^Cached:" /proc/meminfo | awk '{print $2}')
            # GPU memory
            GPU_LINE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -4 | tr '\n' ',')
            echo "$(date +%s),${RSS_TOTAL},${SYS_MEM},${ANON_TOTAL},${FILE_TOTAL},${GPU_LINE},cache_mb=${CACHE_MB}" >> $MEM_SAMPLE_FILE
            sleep 2
        done
    ) &
    MEM_SAMPLER_PID=$!
    echo "Memory sampler started PID=$MEM_SAMPLER_PID"

    source /mnt/data2/tmp/qujing_mesh/.venv/bin/activate

    # ===== 1. Prefill 测速（2048 tokens，< 4096 threshold 走 CPU prefill）=====
    echo "Running prefill benchmark (2048 tokens, CPU prefill path)..."
    PREFILL_OUT=$(python3 -c "
import requests, time, json
# 生成约 2048 token 的 prompt（重复文本，< 4096 threshold 避免触发 GPU prefill bug）
prompt = 'The quick brown fox jumps over the lazy dog. ' * 260  # 约 2080 tokens
url = 'http://localhost:${PORT}/v1/completions'
data = {'model':'default','prompt':prompt,'max_tokens':1,'temperature':0,'stream':False}
start = time.time()
r = requests.post(url, json=data, timeout=300)
end = time.time()
resp = r.json()
# 从 usage 获取 prompt tokens
prompt_tokens = resp.get('usage',{}).get('prompt_tokens',0)
elapsed = end - start
prefill_tps = prompt_tokens / elapsed if elapsed > 0 else 0
print(f'{prompt_tokens}|{elapsed:.2f}|{prefill_tps:.2f}')
" 2>&1)
    echo "Prefill result: ${PREFILL_OUT}"
    PREFILL_S=$(echo "$PREFILL_OUT" | cut -d'|' -f2)
    PREFILL_TPS=$(echo "$PREFILL_OUT" | cut -d'|' -f3)

    sleep 3

    # ===== 2. Decode 测速（streaming 256 tokens）=====
    echo "Running decode benchmark (256 tokens)..."
    DECODE_OUT=$(python3 -c "
import requests, time, json
url = 'http://localhost:${PORT}/v1/completions'
data = {'model':'default','prompt':'Write a short essay about artificial intelligence and its impact on society:','max_tokens':256,'temperature':0,'stream':True}
start = time.time()
first_token_time = None
token_count = 0
r = requests.post(url, json=data, stream=True, timeout=120)
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
print(f'{dtps:.2f}')
" 2>&1)
    echo "Decode tok/s: ${DECODE_OUT}"

    # ===== 3. 停止内存采样，计算 peak =====
    kill $MEM_SAMPLER_PID 2>/dev/null
    sleep 1

    # 从 CSV 计算峰值
    PEAK_RSS_KB=$(python3 -c "
import csv
with open('${MEM_SAMPLE_FILE}') as f:
    reader = csv.DictReader(f)
    vals = [int(row['rss_kb']) for row in reader if row.get('rss_kb')]
    print(max(vals) if vals else 0)
" 2>/dev/null)
    PEAK_SYS_USED_MB=$(python3 -c "
import csv
with open('${MEM_SAMPLE_FILE}') as f:
    reader = csv.DictReader(f)
    vals = [int(row['sys_used_mb']) for row in reader if row.get('sys_used_mb')]
    print(max(vals) if vals else 0)
" 2>/dev/null)
    PEAK_ANON_KB=$(python3 -c "
import csv
with open('${MEM_SAMPLE_FILE}') as f:
    reader = csv.DictReader(f)
    vals = [int(row['anon_kb']) for row in reader if row.get('anon_kb')]
    print(max(vals) if vals else 0)
" 2>/dev/null)
    PEAK_FILE_KB=$(python3 -c "
import csv
with open('${MEM_SAMPLE_FILE}') as f:
    reader = csv.DictReader(f)
    vals = [int(row['file_kb']) for row in reader if row.get('file_kb')]
    print(max(vals) if vals else 0)
" 2>/dev/null)
    PEAK_GPU_MB=$(python3 -c "
import csv
with open('${MEM_SAMPLE_FILE}') as f:
    reader = csv.DictReader(f)
    vals = []
    for row in reader:
        gpu_sum = 0
        for k in ['gpu0_mb','gpu1_mb','gpu2_mb','gpu3_mb']:
            v = row.get(k,'0').strip().rstrip(',')
            if v:
                gpu_sum += int(v)
        vals.append(gpu_sum)
    print(max(vals) if vals else 0)
" 2>/dev/null)

    # 转换为 GiB
    PEAK_RSS_GIB=$(python3 -c "print(f'{${PEAK_RSS_KB:-0}/1024/1024:.2f}')")
    PEAK_SYS_USED_GIB=$(python3 -c "print(f'{${PEAK_SYS_USED_MB:-0}/1024:.2f}')")
    PEAK_ANON_GIB=$(python3 -c "print(f'{${PEAK_ANON_KB:-0}/1024/1024:.2f}')")
    PEAK_FILE_GIB=$(python3 -c "print(f'{${PEAK_FILE_KB:-0}/1024/1024:.2f}')")
    PEAK_GPU_GIB=$(python3 -c "print(f'{${PEAK_GPU_MB:-0}/1024:.2f}')")

    echo "Peak RSS: ${PEAK_RSS_GIB} GiB, Peak SysUsed: ${PEAK_SYS_USED_GIB} GiB, Peak GPU: ${PEAK_GPU_GIB} GiB, Peak Anon: ${PEAK_ANON_GIB} GiB, Peak File: ${PEAK_FILE_GIB} GiB"

    # 写入结果
    echo "${CAP} | ${DECODE_OUT:-0} | ${PREFILL_TPS:-0} | ${PREFILL_S:-0} | ${PEAK_RSS_GIB} | ${PEAK_SYS_USED_GIB} | ${PEAK_GPU_GIB} | ${PEAK_ANON_GIB} | ${PEAK_FILE_GIB}" >> $RESULT_FILE

    # 停服务
    pkill -f "sglang.launch_server.*${PORT}" 2>/dev/null
    sleep 5
done

echo ""
echo "=== Results ==="
cat $RESULT_FILE
