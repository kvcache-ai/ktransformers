#!/bin/bash
# MESH cap sweep benchmark — 方案 A 验证：cgroup 内存 + RAID util + GPU + NUMA.
# Tests multiple cap values with decode speed + per-rank VRAM/GPU util + per-NUMA memory.
# NO pkill -f: uses screen quit + ss+kill + pgrep -P for safe cleanup.
# Does NOT touch gpu4 processes (port 30001/30002).
#
# Naming convention (per user requirement):
#   - "tensor_parallel0" / "tensor_parallel1" = the two scheduler processes
#   - GPU6 = tensor_parallel0's device, GPU7 = tensor_parallel1's device
#   - threadpool0/1 = CPU expert worker pools
#
# Usage: bash cap_sweep.sh "32 64 96 128 160 192 full"

CAP_LIST=${1:-"32 64 96 128 160 192 full"}
PORT=50052
CUDA=6,7
RANK0_GPU=6
RANK1_GPU=7
SCREEN_NAME="mesh_sweep"
WORKDIR=/mnt/data2/tmp/qujing_mesh
SCRIPT=run_mesh_35b_cap_graph_raid0.sh
RESULT=$WORKDIR/cap_sweep_results.txt
VENV=$WORKDIR/.venv/bin/python
CGROUP=/sys/fs/cgroup/mesh_sweep
RAID_DEV=md397

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

# Setup cgroup for memory tracking (needs sudo for cgroup v2 subtree_control)
setup_cgroup() {
    # 清理旧 cgroup
    sudo rmdir $CGROUP 2>/dev/null
    # 启用 memory controller (需要 root)
    sudo bash -c 'echo +memory > /sys/fs/cgroup/cgroup.subtree_control' 2>/dev/null
    sudo mkdir -p $CGROUP 2>/dev/null
    # 让当前用户可读写 cgroup.procs 和 memory.current
    sudo chown -R $(id -u):$(id -g) $CGROUP 2>/dev/null
    if [ -f $CGROUP/memory.current ]; then
        echo "  cgroup $CGROUP ready"
    else
        echo "  WARNING: cgroup $CGROUP not available, memory tracking disabled"
    fi
}

# Add process tree to cgroup
add_to_cgroup() {
    local SERVER_PID=$1
    if [ ! -f $CGROUP/memory.current ]; then
        return
    fi
    # 直接用 pgrep -f 找所有 sglang 相关进程并加入 cgroup
    # 不依赖递归 pgrep -P（sglang 进程树可能较深且复杂）
    local ALL_PIDS=$(pgrep -f "sglang|launch_server|scheduler_TP|kt_kernel|python.*serve" 2>/dev/null)
    # 也加入 server PID 本身
    ALL_PIDS="$SERVER_PID $ALL_PIDS"
    local COUNT=0
    for p in $ALL_PIDS; do
        if [ -n "$p" ]; then
            echo $p | sudo tee $CGROUP/cgroup.procs > /dev/null 2>&1
            COUNT=$((COUNT + 1))
        fi
    done
    # 递归加入 server PID 的所有子孙进程
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

# Get per-rank process info: VmHWM (MB) + NUMA0 mem (MB) + NUMA1 mem (MB)
get_rank_info() {
    local SERVER_PID=$1
    local TP0_VMHWM=0 TP0_NUMA0=0 TP0_NUMA1=0
    local TP1_VMHWM=0 TP1_NUMA0=0 TP1_NUMA1=0
    # 用 pgrep -f 直接找 scheduler_TP 进程，不依赖进程树
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

# Per-second sampler: cgroup memory + system RAM + per-rank GPU VRAM/util + RAID util
MEM_SAMPLES=/tmp/mesh_mem_samples.txt
CGROUP_SAMPLES=/tmp/mesh_cgroup_samples.txt
GPU0_VRAM_SAMPLES=/tmp/mesh_gpu0_vram.txt
GPU1_VRAM_SAMPLES=/tmp/mesh_gpu1_vram.txt
GPU0_UTIL_SAMPLES=/tmp/mesh_gpu0_util.txt
GPU1_UTIL_SAMPLES=/tmp/mesh_gpu1_util.txt
RAID_UTIL_SAMPLES=/tmp/mesh_raid_util.txt
RAID_READ_SAMPLES=/tmp/mesh_raid_read.txt

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
        # System memory (MB used)
        free -m | grep "^Mem:" | awk '{print $3}' >> $MEM_SAMPLES
        # cgroup memory.current (bytes)
        if [ -f $CGROUP/memory.current ]; then
            cat $CGROUP/memory.current >> $CGROUP_SAMPLES
        else
            echo 0 >> $CGROUP_SAMPLES
        fi
        # GPU VRAM + util
        local OUT=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i $RANK0_GPU,$RANK1_GPU 2>/dev/null)
        local L0=$(echo "$OUT" | sed -n '1p')
        local L1=$(echo "$OUT" | sed -n '2p')
        echo "$L0" | awk -F', ' '{print $1}' >> $GPU0_VRAM_SAMPLES
        echo "$L0" | awk -F', ' '{print $2}' >> $GPU0_UTIL_SAMPLES
        echo "$L1" | awk -F', ' '{print $1}' >> $GPU1_VRAM_SAMPLES
        echo "$L1" | awk -F', ' '{print $2}' >> $GPU1_UTIL_SAMPLES
        # RAID md397 %util + rkB/s (iostat columns: $23=%util, $3=rkB/s)
        # 注意 iostat 末尾有空行，必须用 grep 过滤
        local IOSTAT_OUT=$(iostat -dx $RAID_DEV 1 2 2>/dev/null | grep "^$RAID_DEV" | tail -1)
        echo "$IOSTAT_OUT" | awk '{print $23}' >> $RAID_UTIL_SAMPLES
        echo "$IOSTAT_OUT" | awk '{print $3}' >> $RAID_READ_SAMPLES
        sleep 1
    done
}

echo "Config: 方案 A (all tensor_parallel ranks compute CPU experts), mesh=ON, tensor_parallel_size=2, GE=32, RAID0 O_DIRECT"
echo "Cap list: $CAP_LIST  (full=224, i.e. 256 experts - 32 GPU experts)"
echo "GPU6 = tensor_parallel0 device, GPU7 = tensor_parallel1 device"
echo "Results -> $RESULT"
echo ""

# Setup cgroup once
setup_cgroup

for CAP_INPUT in $CAP_LIST; do
    # "full" -> 224
    if [ "$CAP_INPUT" = "full" ]; then
        CAP=224
        CAP_LABEL="full(224)"
    else
        CAP=$CAP_INPUT
        CAP_LABEL=$CAP
    fi

    echo "========== Testing cap=$CAP_LABEL =========="

    # 1. Stop old server
    stop_server

    # 2. Start new server
    LOG=/tmp/mesh_sweep_cap${CAP}.log
    cd $WORKDIR
    screen -dmS $SCREEN_NAME bash -c "taskset -c 0-191 bash $SCRIPT $CAP $PORT $CUDA > $LOG 2>&1"
    echo "  Started cap=$CAP_LABEL, waiting for ready..."

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
        echo "  FAILED: cap=$CAP_LABEL not ready"
        echo "$CAP_LABEL | FAILED | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | $(check_cpu_load)" >> $RESULT
        tail -20 $LOG 2>/dev/null
        continue
    fi

    sleep 5

    # 4. Get server PID and add to cgroup
    SERVER_PID=$(ss -tlnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | head -1)
    echo "  Server PID: $SERVER_PID"
    add_to_cgroup $SERVER_PID

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

    # 8. Calculate peak memory + GPU metrics + cgroup + RAID
    PEAK_SYS_MB=$(sort -n $MEM_SAMPLES | tail -1)
    PEAK_SYS_GIB=$(echo "scale=2; ${PEAK_SYS_MB:-0} / 1024" | bc)
    # cgroup peak (bytes -> GiB)
    PEAK_CGROUP_B=$(sort -n $CGROUP_SAMPLES | tail -1)
    PEAK_CGROUP_GIB=$(echo "scale=2; ${PEAK_CGROUP_B:-0} / 1073741824" | bc)
    # GPU VRAM
    PEAK_GPU0_VRAM_MB=$(sort -n $GPU0_VRAM_SAMPLES | tail -1)
    PEAK_GPU1_VRAM_MB=$(sort -n $GPU1_VRAM_SAMPLES | tail -1)
    PEAK_GPU0_VRAM_GIB=$(echo "scale=2; ${PEAK_GPU0_VRAM_MB:-0} / 1024" | bc)
    PEAK_GPU1_VRAM_GIB=$(echo "scale=2; ${PEAK_GPU1_VRAM_MB:-0} / 1024" | bc)
    # GPU utilization: max during decode
    PEAK_GPU0_UTIL=$(sort -n $GPU0_UTIL_SAMPLES | tail -1)
    PEAK_GPU1_UTIL=$(sort -n $GPU1_UTIL_SAMPLES | tail -1)
    [ -z "$PEAK_GPU0_UTIL" ] && PEAK_GPU0_UTIL=0
    [ -z "$PEAK_GPU1_UTIL" ] && PEAK_GPU1_UTIL=0
    # RAID: max %util + avg read kB/s during decode
    PEAK_RAID_UTIL=$(sort -n $RAID_UTIL_SAMPLES | tail -1)
    [ -z "$PEAK_RAID_UTIL" ] && PEAK_RAID_UTIL=0
    AVG_RAID_READ_KBS=$(awk '{sum+=$1; n++} END {if(n>0) printf "%.0f", sum/n; else print 0}' $RAID_READ_SAMPLES)
    AVG_RAID_READ_MBS=$(echo "scale=1; ${AVG_RAID_READ_KBS:-0} / 1024" | bc)

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

# Stop final server
stop_server

echo ""
echo "=== Results ==="
cat $RESULT
