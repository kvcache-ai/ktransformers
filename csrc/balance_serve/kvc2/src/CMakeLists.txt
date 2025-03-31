include_directories(${THIRD_PARTY_DIR}/asyncio/include)

add_library(kvc2_metrics STATIC metrics.cpp)
target_link_libraries(kvc2_metrics PUBLIC prometheus-cpp::pull)

add_library(page_aligned_memory_pool page_aligned_memory_pool.cpp)
target_include_directories(page_aligned_memory_pool PRIVATE ${THIRD_PARTY_DIR}/spdlog/include)

function(add_third_party_includes TARGET_NAME)
    target_include_directories(${TARGET_NAME} PRIVATE
        ${THIRD_PARTY_BUILD_DIR}/prometheus-cpp/core/include
        ${THIRD_PARTY_BUILD_DIR}/prometheus-cpp/pull/include
        ${THIRD_PARTY_DIR}/prometheus-cpp/core/include
        ${THIRD_PARTY_DIR}/prometheus-cpp/pull/include
        ${THIRD_PARTY_DIR}/spdlog/include
    )
endfunction()


add_library(cache_entry cache_entry.cpp)
add_third_party_includes(cache_entry)
target_link_libraries(cache_entry PUBLIC gpu_cache)

add_library(gpu_cache gpu_cache.cpp)
add_third_party_includes(gpu_cache)
target_link_libraries(gpu_cache PUBLIC xxHash::xxhash ${TORCH_LIBRARIES} cuda_stream_manager)

add_library(kvc2 prefix.cpp)
target_include_directories(kvc2 PRIVATE ${THIRD_PARTY_DIR}/nlohmann/single_include)
add_third_party_includes(kvc2)
target_link_libraries(kvc2 PUBLIC TBB::tbb xxHash::xxhash cache_entry cuda_stream_manager page_aligned_memory_pool ${TORCH_LIBRARIES} prometheus-cpp::pull kvc2_metrics)

message(STATUS "CMAKE_SOURCE_DIR: " ${CMAKE_SOURCE_DIR})
add_library(async_store async_store.cpp)
target_include_directories(async_store PRIVATE ${THIRD_PARTY_DIR}/nlohmann/single_include)
target_include_directories(async_store PRIVATE ${THIRD_PARTY_DIR}/spdlog/include)
target_link_libraries(async_store PUBLIC pthread)



add_library(cuda_stream_manager cuda_stream_manager.cpp)
target_include_directories(cuda_stream_manager PUBLIC ${THIRD_PARTY_DIR}/nlohmann/single_include)
target_include_directories(cuda_stream_manager PUBLIC ${THIRD_PARTY_DIR}/spdlog/include)
target_include_directories(cuda_stream_manager  PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(cuda_stream_manager PUBLIC CUDA::cudart)
