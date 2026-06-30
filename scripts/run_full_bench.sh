#!/bin/bash
# 完整 benchmark 批量测试脚本
# 等 GPU 空闲后执行所有测试组合
# 用法: bash run_full_bench.sh <CUDA_LIST>
# 例: bash run_full_bench.sh 0,1,2,3

source /mnt/data2/tmp/qujing_mesh/.venv/bin/activate
cd /mnt/data2/tmp/qujing_mesh

CUDA=${1:-0,1,2,3}
PORT=10004

echo "============================================"
echo "Full Benchmark Suite"
echo "CUDA: $CUDA"
echo "Start: $(date)"
echo "============================================"

# 35B 测试
echo ""
echo "=== 35B AMXINT4 标准 kt ==="
python3 -u bench_full_py.py kt 35b amxint4 $PORT $CUDA

echo ""
echo "=== 35B AMXINT4 MESH ==="
python3 -u bench_full_py.py mesh 35b amxint4 "64 128 192 224" $PORT $CUDA

echo ""
echo "=== 35B BF16 标准 kt ==="
python3 -u bench_full_py.py kt 35b bf16 $PORT $CUDA

echo ""
echo "=== 35B BF16 MESH ==="
python3 -u bench_full_py.py mesh 35b bf16 "64 128 192 224" $PORT $CUDA

# 397B 测试
echo ""
echo "=== 397B AMXINT4 标准 kt ==="
python3 -u bench_full_py.py kt 397b amxint4 $PORT $CUDA

echo ""
echo "=== 397B AMXINT4 MESH ==="
python3 -u bench_full_py.py mesh 397b amxint4 "128 192 256 480" $PORT $CUDA

echo ""
echo "=== 397B BF16 标准 kt ==="
python3 -u bench_full_py.py kt 397b bf16 $PORT $CUDA

echo ""
echo "=== 397B BF16 MESH ==="
python3 -u bench_full_py.py mesh 397b bf16 "128 192 256 480" $PORT $CUDA

echo ""
echo "============================================"
echo "All tests completed: $(date)"
echo "============================================"
echo ""
echo "=== Results Summary ==="
for f in /tmp/full_bench_*_results.txt; do
    echo "--- $f ---"
    cat "$f"
    echo ""
done
