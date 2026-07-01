#!/bin/bash
# MESH cap sweep benchmark — 35B BF16 原版
# Tests cap values with decode speed + per-rank VRAM/GPU util + per-NUMA memory.
# Uses md398 (3-NVMe raid0) for BF16 aligned weights.
# Naming: tensor_parallel0 (GPU6) / tensor_parallel1 (GPU7)
#
# Usage: bash cap_sweep_bf16.sh "32 64 96 128 160 192 full"

CAP_LIST=${1:-"32 64 96 128 160 192 full"}
PORT=50052
CUDA=6,7
RANK0_GPU=6
RANK1_GPU=7
SCREEN_NAME="mesh_sweep_bf16"
WORKDIR=/mnt/data2/tmp/qujing_mesh
SCRIPT=run_mesh_35b_bf16_cap_tp2.sh
RESULT=$WORKDIR/cap_sweep_bf16_results.txt
VENV=$WORKDIR/.venv/bin/python
CGROUP=/sys/fs/cgroup/mesh_sweep_bf16
RAID_DEV=md398

# Header columns
echo "cap | decode_tok_s | decode_tokens | total_elapsed | cgroup_peak_gib | peak_sys_gib | tensor_parallel0_vram_gib | tensor_parallel1_vram_gib | tensor_parallel0_gpu_util_pct | tensor_parallel1_gpu_util_pct | tensor_parallel0_vmhwm_mb | tensor_parallel0_numa0_mb | tensor_parallel0_numa1_mb | tensor_parallel1_vmhwm_mb | tensor_parallel1_numa0_mb | tensor_parallel1_numa1_mb | raid_util_pct | raid_read_mbs | cpu_load" > $RESULT

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

setup_cgroup() {
    sudo rmdir $CGROUP 2>/dev/null
    sudo bash -c 'echo +memory > /sys/fs/cgroup/cgroup.subtree_control' 2>/dev/null
    sudo mkdir -p $CGROUP 2>/dev/null
    sudo chown -R $(id -u):$(id -g) $CGROUP 2>/dev/null
    if [ -f $CGROUP/memory.current ]; then
        echo "  cgroup $CGROUP ready"
    else
        echo "  WARNING: cgroup $CGROUP not available"
    fi
}

add_to_cgroup() {
    local SERVER_PID=$1
    if [ ! -f $CGROUP/memory.current ]; then
        return
    fi
    local ALL_PIDS=$(pgrep -f "sglang|launch_server|scheduler_TP|kt_kernel|python.*serve" 2>/dev/null)
    ALL_PIDS="$SERVER_PID $ALL_PIDS"
    local COUNT=0
    for p in $ALL_PIDS; do
        if [ -n "$p" ]; then
            echo $p | sudo tee $CGROUP/cgroup.procs > /dev/null 2>&1
            COUNT=$((COUNT + 1))
        fi
    done
    local RECURSIVE_PIDS=$(pgrep -P $SERVER_PID 2>/dev/null)
    for p in $RECURSIVE_PIDS; do
        echo $p | sudo tee $CGROUP/cgroup.procs > /dev/null 2>&1
        for GC in $(pgrep -P $p 2>/dev/null); do
            echo $GC | sudo tee $CGROUP/cgroup.procs > /dev/null 2>&1
            for GGC in $(pgrep -P $GC 2>/dev/null); do
                echo $GGC | sudo tee $CGROUP/cgroup.procs > /dev/null 2>&1
            done
        done
    done
    echo "  Added $COUNT processes to cgroup"
}

get_rank_info() {
    local SERVER_PID=$1
    local TP0_VMHWM=0 TP0_NUMA0=0 TP0_NUMA1=0
    local TP1_VMHWM=0 TP1_NUMA0=0 TP1_NUMA1=0
    for CHILD in $(pgrep -f "scheduler_TP[01]" 2>/dev/null); do
        local ARGS=$(ps -p $CHILD -o args= 2>/dev/null)
        local HWM=$(grep "^VmHWM:" /proc/$CHILD/status 2>/dev/null | awk '{print $2}')
        [ -z "$HWM" ] && HWM=0
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

check_cpu_load() {
    local TOTAL_USED=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d% -f1)
    echo "total_user=${TOTAL_USED}%"
}

MEM_SAMPLES=/tmp/mesh_bf16_mem.txt
CGROUP_SAMPLES=/tmp/mesh_bf16_cgroup.txt
GPU0_VRAM_SAMPLES=/tmp/mesh_bf16_gpu0_vram.txt
GPU1_VRAM_SAMPLES=/tmp/mesh_bf16_gpu1_vram.txt
GPU0_UTIL_SAMPLES=/tmp/mesh_bf16_gpu0_util.txt
GPU1_UTIL_SAMPLES=/tmp/mesh_bf16_gpu1_util.txt
RAID_UTIL_SAMPLES=/tmp/mesh_bf16_raid_util.txt
RAID_READ_SAMPLES=/tmp/mesh_bf16_raid_read.txt

sample_loop() {
    > $MEM_SAMPLES
    > $CGROUP_SAMPLES
    > $GPU0_VRAM_SAMPLES
    > $GPU1_VRAM_SAMPLES
    > $GPU0_UTIL_SAMPLES
    > $GPU1_UTIL_SAMPLES
    > $RAID_UTIL_SAMPLES
    > $RAID_READ_SAMPLES
    while true; do
        free -m | grep "^Mem:" | awk '{print $3}' >> $MEM_SAMPLES
        if [ -f $CGROUP/memory.current ]; then
            cat $CGROUP/memory.current >> $CGROUP_SAMPLES
        else
            echo 0 >> $CGROUP_SAMPLES
        fi
        local OUT=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i $RANK0_GPU,$RANK1_GPU 2>/dev/null)
        local L0=$(echo "$OUT" | sed -n '1p')
        local L1=$(echo "$OUT" | sed -n '2p')
        echo "$L0" | awk -F', ' '{print $1}' >> $GPU0_VRAM_SAMPLES
        echo "$L0" | awk -F', ' '{print $2}' >> $GPU0_UTIL_SAMPLES
        echo "$L1" | awk -F', ' '{print $1}' >> $GPU1_VRAM_SAMPLES
        echo "$L1" | awk -F', ' '{print $2}' >> $GPU1_UTIL_SAMPLES
        local IOSTAT_OUT=$(iostat -dx $RAID_DEV 1 2 2>/dev/null | grep "^$RAID_DEV" | tail -1)
        echo "$IOSTAT_OUT" | awk '{print $23}' >> $RAID_UTIL_SAMPLES
        echo "$IOSTAT_OUT" | awk '{print $3}' >> $RAID_READ_SAMPLES
        sleep 1
    done
}

echo "Config: 35B BF16 原版, 方案 A, mesh=ON, tensor_parallel=2, GE=32, RAID0 (md398) O_DIRECT"
echo "Cap list: $CAP_LIST  (full=224, i.e. 256 experts - 32 GPU experts)"
echo "GPU6 = tensor_parallel0 device, GPU7 = tensor_parallel1 device"
echo "Results -> $RESULT"
echo ""

setup_cgroup

for CAP_INPUT in $CAP_LIST; do
    if [ "$CAP_INPUT" = "full" ]; then
        CAP=224
        CAP_LABEL="full(224)"
    else
        CAP=$CAP_INPUT
        CAP_LABEL=$CAP
    fi

    echo "========== Testing cap=$CAP_LABEL =========="

    stop_server

    LOG=/tmp/mesh_sweep_bf16_cap${CAP}.log
    cd $WORKDIR
    screen -dmS $SCREEN_NAME bash -c "taskset -c 0-191 bash $SCRIPT $CAP $PORT $CUDA > $LOG 2>&1"
    echo "  Started cap=$CAP_LABEL, waiting for ready..."

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
        echo "  FAILED: cap=$CAP_LABEL not ready"
        echo "$CAP_LABEL | FAILED | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | $(check_cpu_load)" >> $RESULT
        tail -20 $LOG 2>/dev/null
        continue
    fi

    sleep 5

    SERVER_PID=$(ss -tlnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | head -1)
    echo "  Server PID: $SERVER_PID"
    add_to_cgroup $SERVER_PID

    sample_loop &
    SAMPLER_PID=$!
    echo "  Sampler PID: $SAMPLER_PID"

    echo "  Running decode benchmark..."
    BENCH_OUT=$($VENV $WORKDIR/bench_decode_stream.py $PORT 512 2>&1)
    echo "  $BENCH_OUT"

    DECODE_TPS=$(echo "$BENCH_OUT" | grep "decode_tok_s:" | awk '{print $2}')
    DECODE_TOKENS=$(echo "$BENCH_OUT" | grep "decode_tokens:" | awk '{print $2}')
    TOTAL_ELAPSED=$(echo "$BENCH_OUT" | grep "total_elapsed:" | awk '{print $2}')

    kill $SAMPLER_PID 2>/dev/null
    wait $SAMPLER_PID 2>/dev/null

    PEAK_SYS_MB=$(sort -n $MEM_SAMPLES | tail -1)
    PEAK_SYS_GIB=$(echo "scale=2; ${PEAK_SYS_MB:-0} / 1024" | bc)
    PEAK_CGROUP_B=$(sort -n $CGROUP_SAMPLES | tail -1)
    PEAK_CGROUP_GIB=$(echo "scale=2; ${PEAK_CGROUP_B:-0} / 1073741824" | bc)
    PEAK_GPU0_VRAM_MB=$(sort -n $GPU0_VRAM_SAMPLES | tail -1)
    PEAK_GPU1_VRAM_MB=$(sort -n $GPU1_VRAM_SAMPLES | tail -1)
    PEAK_GPU0_VRAM_GIB=$(echo "scale=2; ${PEAK_GPU0_VRAM_MB:-0} / 1024" | bc)
    PEAK_GPU1_VRAM_GIB=$(echo "scale=2; ${PEAK_GPU1_VRAM_MB:-0} / 1024" | bc)
    PEAK_GPU0_UTIL=$(sort -n $GPU0_UTIL_SAMPLES | tail -1)
    PEAK_GPU1_UTIL=$(sort -n $GPU1_UTIL_SAMPLES | tail -1)
    [ -z "$PEAK_GPU0_UTIL" ] && PEAK_GPU0_UTIL=0
    [ -z "$PEAK_GPU1_UTIL" ] && PEAK_GPU1_UTIL=0
    PEAK_RAID_UTIL=$(sort -n $RAID_UTIL_SAMPLES | tail -1)
    [ -z "$PEAK_RAID_UTIL" ] && PEAK_RAID_UTIL=0
    AVG_RAID_READ_KBS=$(awk '{sum+=$1; n++} END {if(n>0) printf "%.0f", sum/n; else print 0}' $RAID_READ_SAMPLES)
    AVG_RAID_READ_MBS=$(echo "scale=1; ${AVG_RAID_READ_KBS:-0} / 1024" | bc)

    RANK_INFO=$(get_rank_info $SERVER_PID)
    TP0_VMHWM=$(echo "$RANK_INFO" | awk '{print $1}')
    TP0_NUMA0=$(echo "$RANK_INFO" | awk '{print $2}')
    TP0_NUMA1=$(echo "$RANK_INFO" | awk '{print $3}')
    TP1_VMHWM=$(echo "$RANK_INFO" | awk '{print $4}')
    TP1_NUMA0=$(echo "$RANK_INFO" | awk '{print $5}')
    TP1_NUMA1=$(echo "$RANK_INFO" | awk '{print $6}')

    CPU_LOAD=$(check_cpu_load)

    echo "  Decode: ${DECODE_TPS} tok/s"
    echo "  cgroup peak: ${PEAK_CGROUP_GIB}GiB | sys peak: ${PEAK_SYS_GIB}GiB"
    echo "  tensor_parallel0 (GPU${RANK0_GPU}): VRAM=${PEAK_GPU0_VRAM_GIB}GiB, gpu_util_peak=${PEAK_GPU0_UTIL}%"
    echo "  tensor_parallel1 (GPU${RANK1_GPU}): VRAM=${PEAK_GPU1_VRAM_GIB}GiB, gpu_util_peak=${PEAK_GPU1_UTIL}%"
    echo "  tensor_parallel0 proc: VmHWM=${TP0_VMHWM}MB, NUMA0=${TP0_NUMA0}MB, NUMA1=${TP0_NUMA1}MB"
    echo "  tensor_parallel1 proc: VmHWM=${TP1_VMHWM}MB, NUMA0=${TP1_NUMA0}MB, NUMA1=${TP1_NUMA1}MB"
    echo "  RAID $RAID_DEV: peak_util=${PEAK_RAID_UTIL}%, avg_read=${AVG_RAID_READ_MBS}MB/s"
    echo "  CPU load: $CPU_LOAD"

    echo "$CAP_LABEL | $DECODE_TPS | $DECODE_TOKENS | $TOTAL_ELAPSED | $PEAK_CGROUP_GIB | $PEAK_SYS_GIB | $PEAK_GPU0_VRAM_GIB | $PEAK_GPU1_VRAM_GIB | $PEAK_GPU0_UTIL | $PEAK_GPU1_UTIL | $TP0_VMHWM | $TP0_NUMA0 | $TP0_NUMA1 | $TP1_VMHWM | $TP1_NUMA0 | $TP1_NUMA1 | $PEAK_RAID_UTIL | $AVG_RAID_READ_MBS | $CPU_LOAD" >> $RESULT

    echo ""
done

stop_server

echo ""
echo "=== Results ==="
cat $RESULT
