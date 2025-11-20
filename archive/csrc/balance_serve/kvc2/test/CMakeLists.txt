
set(CMAKE_CXX_FLAGS "-Og -march=native -Wall -Wextra -g -fopenmp")
# set(CMAKE_CXX_FLAGS "-O3 -march=native -Wall -Wextra -pthread")

add_subdirectory(kvc2test)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)

add_executable(hashmap_test hashmap_test.cpp)
target_link_libraries(hashmap_test PRIVATE TBB::tbb)


add_executable(xxHash_test xxHash_test.cpp)
target_link_libraries(xxHash_test PRIVATE xxhash)

function(add_async_store_executable source_file)
    get_filename_component(target_name ${source_file} NAME_WE) # 获取不带扩展名的文件名作为目标名
    add_executable(${target_name} ${source_file})
    target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
    target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/nlohmann/single_include)
    target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/spdlog/include)
    target_link_libraries(${target_name} PRIVATE async_store gflags)
endfunction()

add_async_store_executable(async_store_test.cpp)


function(add_kvc2_executable source_file)
    get_filename_component(target_name ${source_file} NAME_WE) # 获取不带扩展名的文件名作为目标名
    add_executable(${target_name} ${source_file})
    # target_compile_options(${target_name} PRIVATE -fopenmp  -fno-strict-aliasing)
    target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
    target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/nlohmann/single_include)
    target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/spdlog/include)
    target_link_libraries(${target_name} PRIVATE kvc2 async_store gflags)
endfunction()




add_kvc2_executable(test_lock_free_queue.cpp)
add_kvc2_executable(test_queue_perf.cpp)

# Disable deprecated test
# add_kvc2_executable(prefix_test.cpp)
# add_kvc2_executable(kvcache_disk_insert_read_test.cpp)
# add_kvc2_executable(kvcache_mem_eviction_test.cpp)
# add_kvc2_executable(kvcache_mem_insert_read_test.cpp)
# add_kvc2_executable(kvcache_save_load_test.cpp)
# add_kvc2_executable(kvc2_export_header_test.cpp)
# add_kvc2_executable(kvc2_export_load_test.cpp)





target_include_directories(async_store_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..//third_party/nlohmann/single_include)
target_include_directories(async_store_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..//third_party/spdlog/include)
target_link_libraries(async_store_test PRIVATE xxhash)

add_executable(test_std_list test_std_list.cpp)


add_executable(test_cuda_stream test_cuda_stream.cpp)
target_include_directories(test_cuda_stream PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(test_cuda_stream PRIVATE CUDA::cudart)

add_executable(test_cuda_stream_manager test_cuda_stream_manager.cpp)
target_include_directories(test_cuda_stream_manager PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_link_libraries(test_cuda_stream_manager PRIVATE cuda_stream_manager)

add_executable(test_periodic_task test_periodic_task.cpp)
target_include_directories(test_periodic_task PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)

add_executable(test_page_pool page_pool_test.cpp)
target_include_directories(test_page_pool PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../src)
target_include_directories(test_page_pool PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/spdlog/include)