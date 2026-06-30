#!/bin/bash
# MESH cap sweep benchmark — fine-grained per-rank memory + GPU + utilization.
# Tests multiple cap values with decode speed + per-rank VRAM/GPU util + per-NUMA memory.
# NO pkill -f: uses screen quit + ss+kill + pgrep -P for safe cleanup.
# Does NOT touch gpu4 processes (port 30001/30002).
#
# Naming convention (per user requirement):
#   - "tensor_parallel0" / "tensor_parallel1" = the two scheduler processes (was TP0/TP1)
#   - GPU6 = tensor_parallel0's device, GPU7 = tensor_parallel1's device
#   - threadpool0/1 = CPU expert worker pools (not tracked separately here)
#
# Usage: bash cap_sweep.sh "32 64 96 128 160 192"

CAP_LIST=${1:-"32 64 96 128 160 192"}
PORT=50052
CUDA=6,7
RANK0_GPU=6
RANK1_GPU=7
SCREEN_NAME="mesh_sweep"
WORKDIR=/mnt/data2/tmp/qujing_mesh
SCRIPT=run_mesh_35b_cap_graph_raid0.sh
RESULT=$WORKDIR/cap_sweep_results.txt
VENV=$WORKDIR/.venv/bin/python

# Header columns (no "TP" abbreviation — use full "tensor_parallel0/1")
echo "cap | decode_tok_s | decode_tokens | total_elapsed | peak_sys_gib | tensor_parallel0_vram_gib | tensor_parallel1_vram_gib | tensor_parallel0_gpu_util_pct | tensor_parallel1_gpu_util_pct | tensor_parallel0_vmhwm_mb | tensor_parallel0_numa0_mb | tensor_parallel0_numa1_mb | tensor_parallel1_vmhwm_mb | tensor_parallel1_numa0_mb | tensor_parallel1_numa1_mb | cpu_load" > $RESULT

stop_server() {
    echo "  Stopping server..."
    screen -X -S $SCREEN_NAME quit 2>/dev/null
    sleep 3
    local PID=$(ss -tlnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | head -1)
    if [ -n "$PID" ]; then
        local CHILDREN=$(pgrep -P $PID 2>/dev/null)
        for CHILD in $CHILDREN; do
            local GRANDCHILDREN=$(pgrep -P $CHILD 2>/dev/null)
            for GC in $GRANDCHILDREN; do
                kill $GC 2>/dev/null
            done
            kill $CHILD 2>/dev/null
        done
        kill $PID 2>/dev/null
    fi
    for i in $(seq 1 15); do
        if ! ss -tln 2>/dev/null | grep -q ":$PORT "; then
            break
        fi
        sleep 2
    done
    sleep 3
    if ss -tln 2>/dev/null | grep -q ":$PORT "; then
        echo "  WARNING: port $PORT still in use after cleanup"
    else
        echo "  Port $PORT free"
    fi
}

# Get per-rank process info: VmHWM (MB) + NUMA0 mem (MB) + NUMA1 mem (MB)
# Returns 6 space-separated integers:
#   tp0_vmhwm tp0_numa0 tp0_numa1 tp1_vmhwm tp1_numa0 tp1_numa1
get_rank_info() {
    local SERVER_PID=$1
    local TP0_VMHWM=0 TP0_NUMA0=0 TP0_NUMA1=0
    local TP1_VMHWM=0 TP1_NUMA0=0 TP1_NUMA1=0
    local CHILDREN=$(pgrep -P $SERVER_PID 2>/dev/null)
    for CHILD in $CHILDREN; do
        # Use ps -o args= to get full command line (not truncated like /proc/PID/status Name)
        local ARGS=$(ps -p $CHILD -o args= 2>/dev/null)
        local HWM=$(grep "^VmHWM:" /proc/$CHILD/status 2>/dev/null | awk '{print $2}')
        [ -z "$HWM" ] && HWM=0
        # numastat -p PID: take "Total" row, Node0 and Node1 columns (MB, converted to int)
        local NUMA_OUT=$(numastat -p $CHILD 2>/dev/null | awk '/^Total/ {printf "%d %d", $2, $3}')
        local N0=$(echo "$NUMA_OUT" | awk '{print $1}')
        local N1=$(echo "$NUMA_OUT" | awk '{print $2}')
        [ -z "$N0" ] && N0=0
        [ -z "$N1" ] && N1=0
        if echo "$ARGS" | grep -q "scheduler_TP0"; then
            TP0_VMHWM=$((HWM / 1024))
            TP0_NUMA0=$N0
            TP0_NUMA1=$N1
        elif echo "$ARGS" | grep -q "scheduler_TP1"; then
            TP1_VMHWM=$((HWM / 1024))
            TP1_NUMA0=$N0
            TP1_NUMA1=$N1
        fi
    done
    echo "$TP0_VMHWM $TP0_NUMA0 $TP0_NUMA1 $TP1_VMHWM $TP1_NUMA0 $TP1_NUMA1"
}

# Check CPU load from other processes (djh training etc.)
check_cpu_load() {
    local DJH_CPU=$(ps -p 1620567 -o %cpu= 2>/dev/null | awk '{printf "%.0f", $1}')
    local TOTAL_USED=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d% -f1)
    echo "djh_cpu=${DJH_CPU}% total_user=${TOTAL_USED}%"
}

# Per-second sampler: system RAM + per-rank GPU VRAM + per-rank GPU util
MEM_SAMPLES=/tmp/mesh_mem_samples.txt
GPU0_VRAM_SAMPLES=/tmp/mesh_gpu0_vram.txt   # GPU6 = tensor_parallel0
GPU1_VRAM_SAMPLES=/tmp/mesh_gpu1_vram.txt   # GPU7 = tensor_parallel1
GPU0_UTIL_SAMPLES=/tmp/mesh_gpu0_util.txt
GPU1_UTIL_SAMPLES=/tmp/mesh_gpu1_util.txt

sample_loop() {
    > $MEM_SAMPLES
    > $GPU0_VRAM_SAMPLES
    > $GPU1_VRAM_SAMPLES
    > $GPU0_UTIL_SAMPLES
    > $GPU1_UTIL_SAMPLES
    while true; do
        # System memory (MB used)
        free -m | grep "^Mem:" | awk '{print $3}' >> $MEM_SAMPLES
        # Query both GPUs at once: columns = memory.used, utilization.gpu
        local OUT=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i $RANK0_GPU,$RANK1_GPU 2>/dev/null)
        local L0=$(echo "$OUT" | sed -n '1p')
        local L1=$(echo "$OUT" | sed -n '2p')
        echo "$L0" | awk -F', ' '{print $1}' >> $GPU0_VRAM_SAMPLES
        echo "$L0" | awk -F', ' '{print $2}' >> $GPU0_UTIL_SAMPLES
        echo "$L1" | awk -F', ' '{print $1}' >> $GPU1_VRAM_SAMPLES
        echo "$L1" | awk -F', ' '{print $2}' >> $GPU1_UTIL_SAMPLES
        sleep 1
    done
}

echo "Config: mesh=ON, tensor_parallel_size=2, GE=32, dual-numa(0,1), defer=3, RAID0 O_DIRECT"
echo "Cap list: $CAP_LIST"
echo "GPU6 = tensor_parallel0 device, GPU7 = tensor_parallel1 device"
echo "Results -> $RESULT"
echo ""

for CAP in $CAP_LIST; do
    echo "========== Testing cap=$CAP =========="

    # 1. Stop old server
    stop_server

    # 2. Start new server
    LOG=/tmp/mesh_sweep_cap${CAP}.log
    cd $WORKDIR
    screen -dmS $SCREEN_NAME bash -c "taskset -c 24-47,72-191 bash $SCRIPT $CAP $PORT $CUDA > $LOG 2>&1"
    echo "  Started cap=$CAP, waiting for ready..."

    # 3. Wait for ready (up to 10 min)
    READY=0
    for i in $(seq 1 120); do
        sleep 5
        if grep -q "fired up and ready to roll" $LOG 2>/dev/null; then
            READY=1
            echo "  Server ready after $((i*5))s"
            break
        fi
        if grep -q "scheduler is dead" $LOG 2>/dev/null; then
            echo "  FATAL: scheduler dead"
            break
        fi
        if grep -q "Segmentation fault" $LOG 2>/dev/null; then
            echo "  FATAL: segfault"
            break
        fi
    done

    if [ $READY -eq 0 ]; then
        echo "  FAILED: cap=$CAP not ready"
        echo "$CAP | FAILED | - | - | - | - | - | - | - | - | - | - | - | - | - | $(check_cpu_load)" >> $RESULT
        tail -20 $LOG 2>/dev/null
        continue
    fi

    sleep 5

    # 4. Get server PID (the listening process)
    SERVER_PID=$(ss -tlnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | head -1)
    echo "  Server PID: $SERVER_PID"

    # 5. Start sampler (per-second)
    sample_loop &
    SAMPLER_PID=$!
    echo "  Sampler PID: $SAMPLER_PID"

    # 6. Run decode benchmark
    echo "  Running decode benchmark..."
    BENCH_OUT=$($VENV $WORKDIR/bench_decode_stream.py $PORT 512 2>&1)
    echo "  $BENCH_OUT"

    DECODE_TPS=$(echo "$BENCH_OUT" | grep "decode_tok_s:" | awk '{print $2}')
    DECODE_TOKENS=$(echo "$BENCH_OUT" | grep "decode_tokens:" | awk '{print $2}')
    TOTAL_ELAPSED=$(echo "$BENCH_OUT" | grep "total_elapsed:" | awk '{print $2}')

    # 7. Stop sampler
    kill $SAMPLER_PID 2>/dev/null
    wait $SAMPLER_PID 2>/dev/null

    # 8. Calculate peak memory + GPU metrics
    PEAK_SYS_MB=$(sort -n $MEM_SAMPLES | tail -1)
    PEAK_SYS_GIB=$(echo "scale=2; ${PEAK_SYS_MB:-0} / 1024" | bc)
    PEAK_GPU0_VRAM_MB=$(sort -n $GPU0_VRAM_SAMPLES | tail -1)
    PEAK_GPU1_VRAM_MB=$(sort -n $GPU1_VRAM_SAMPLES | tail -1)
    PEAK_GPU0_VRAM_GIB=$(echo "scale=2; ${PEAK_GPU0_VRAM_MB:-0} / 1024" | bc)
    PEAK_GPU1_VRAM_GIB=$(echo "scale=2; ${PEAK_GPU1_VRAM_MB:-0} / 1024" | bc)
    # GPU utilization: take max during decode (peak utilization)
    PEAK_GPU0_UTIL=$(sort -n $GPU0_UTIL_SAMPLES | tail -1)
    PEAK_GPU1_UTIL=$(sort -n $GPU1_UTIL_SAMPLES | tail -1)
    [ -z "$PEAK_GPU0_UTIL" ] && PEAK_GPU0_UTIL=0
    [ -z "$PEAK_GPU1_UTIL" ] && PEAK_GPU1_UTIL=0

    # 9. Get per-rank VmHWM + NUMA breakdown
    RANK_INFO=$(get_rank_info $SERVER_PID)
    TP0_VMHWM=$(echo "$RANK_INFO" | awk '{print $1}')
    TP0_NUMA0=$(echo "$RANK_INFO" | awk '{print $2}')
    TP0_NUMA1=$(echo "$RANK_INFO" | awk '{print $3}')
    TP1_VMHWM=$(echo "$RANK_INFO" | awk '{print $4}')
    TP1_NUMA0=$(echo "$RANK_INFO" | awk '{print $5}')
    TP1_NUMA1=$(echo "$RANK_INFO" | awk '{print $6}')

    # 10. CPU load
    CPU_LOAD=$(check_cpu_load)

    echo "  Peak sys: ${PEAK_SYS_GIB}GiB"
    echo "  tensor_parallel0 (GPU${RANK0_GPU}): VRAM=${PEAK_GPU0_VRAM_GIB}GiB, gpu_util_peak=${PEAK_GPU0_UTIL}%"
    echo "  tensor_parallel1 (GPU${RANK1_GPU}): VRAM=${PEAK_GPU1_VRAM_GIB}GiB, gpu_util_peak=${PEAK_GPU1_UTIL}%"
    echo "  tensor_parallel0 proc: VmHWM=${TP0_VMHWM}MB, NUMA0=${TP0_NUMA0}MB, NUMA1=${TP0_NUMA1}MB"
    echo "  tensor_parallel1 proc: VmHWM=${TP1_VMHWM}MB, NUMA0=${TP1_NUMA0}MB, NUMA1=${TP1_NUMA1}MB"
    echo "  CPU load: $CPU_LOAD"

    echo "$CAP | $DECODE_TPS | $DECODE_TOKENS | $TOTAL_ELAPSED | $PEAK_SYS_GIB | $PEAK_GPU0_VRAM_GIB | $PEAK_GPU1_VRAM_GIB | $PEAK_GPU0_UTIL | $PEAK_GPU1_UTIL | $TP0_VMHWM | $TP0_NUMA0 | $TP0_NUMA1 | $TP1_VMHWM | $TP1_NUMA0 | $TP1_NUMA1 | $CPU_LOAD" >> $RESULT

    echo ""
done

# Stop final server
stop_server

echo ""
echo "=== Results ==="
cat $RESULT
