#!/bin/bash

# 进入脚本所在的目录
cd "$(dirname "$0")" || { echo "Failed to enter the script's directory"; exit 1; }

# 进入 ../build 目录
cd ../build || { echo "Failed to enter ../build directory"; exit 1; }

# 设置线程数列表
threads=(1 2 4 8 16 24 36 48 72)

# 遍历每个线程数并运行命令
for t in "${threads[@]}"; do
    echo "Running with OMP_NUM_THREADS=$t"
    OMP_NUM_THREADS=$t numactl -N 0 ./la/amx-test
    sleep 1s
done
