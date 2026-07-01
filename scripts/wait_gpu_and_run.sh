#!/bin/bash
# 等待 GPU 空闲后自动启动完整 benchmark
source /mnt/data2/tmp/qujing_mesh/.venv/bin/activate
cd /mnt/data2/tmp/qujing_mesh

while true; do
    GPU0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | tr -d ' ')
    GPU1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')
    GPU2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2 | tr -d ' ')
    GPU3=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 | tr -d ' ')

    echo "[$(date '+%H:%M:%S')] GPU0=${GPU0}MiB GPU1=${GPU1}MiB GPU2=${GPU2}MiB GPU3=${GPU3}MiB"

    if [ "$GPU0" -lt 5000 ] && [ "$GPU1" -lt 5000 ] && [ "$GPU2" -lt 5000 ] && [ "$GPU3" -lt 5000 ]; then
        echo "GPU 0,1,2,3 all free! Starting benchmark..."
        bash run_full_bench.sh 0,1,2,3 > /tmp/full_bench_output.log 2>&1
        echo "DONE" >> /tmp/full_bench_output.log
        exit 0
    fi

    GPU4=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4 | tr -d ' ')
    GPU5=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 5 | tr -d ' ')
    GPU6=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6 | tr -d ' ')
    GPU7=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 7 | tr -d ' ')

    if [ "$GPU4" -lt 5000 ] && [ "$GPU5" -lt 5000 ] && [ "$GPU6" -lt 5000 ] && [ "$GPU7" -lt 5000 ]; then
        echo "GPU 4,5,6,7 all free! Starting benchmark..."
        bash run_full_bench.sh 4,5,6,7 > /tmp/full_bench_output.log 2>&1
        echo "DONE" >> /tmp/full_bench_output.log
        exit 0
    fi

    sleep 300
done
