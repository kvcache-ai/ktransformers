#!/bin/bash

# 检查是否提供了 disk_cache_path 参数
if [ -z "$1" ]; then
    echo "Usage: $0 <disk_cache_path>"
    exit 1
fi

# 将 disk_cache_path 参数赋值给变量
disk_cache_path=$1

# 定义测试命令数组，并使用变量替换 disk_cache_path
tests=(
    "./build/test/kvc2_export_header_test --disk_cache_path=$disk_cache_path"
    "./build/test/kvcache_disk_insert_read_test --disk_cache_path=$disk_cache_path"
    "./build/test/kvcache_mem_eviction_test --disk_cache_path=$disk_cache_path"
    "./build/test/kvcache_mem_insert_read_test --disk_cache_path=$disk_cache_path"
    "./build/test/kvcache_save_load_test --disk_cache_path=$disk_cache_path"
)


# 遍历每个测试命令
for test in "${tests[@]}"; do
    echo "Running: $test"
    # 运行测试并捕获输出
    output=$($test)
    
    # 检查测试输出中是否包含 "Test Passed"
    if echo "$output" | grep -q "Test Passed"; then
        echo "  Test Passed"
    else
        echo "  Test Failed"
    fi

    sleep 1
done