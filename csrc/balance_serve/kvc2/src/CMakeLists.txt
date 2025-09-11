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
option(KTRANSFORMERS_USE_NPU                 "ktransformers: use NPU"                           OFF)

if(KTRANSFORMERS_USE_NPU)
    add_definitions(-DKTRANSFORMERS_USE_NPU=1)
endif()

# gpu_cache
if(KTRANSFORMERS_USE_NPU)
    target_include_directories(gpu_cache PUBLIC ${TORCH_NPU_PATH}/include)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    message("python include location: " ${Python_INCLUDE_DIRS} " lib location: " ${Python_LIBRARIES})
    target_link_libraries(gpu_cache PUBLIC xxHash::xxhash "${TORCH_LIBRARIES}"  "${TORCH_PYTHON_LIBRARY}" "${PTA_LIBRARY}" ${Python_LIBRARIES} cuda_stream_manager)
else()
    target_link_libraries(gpu_cache PUBLIC xxHash::xxhash ${TORCH_LIBRARIES} cuda_stream_manager)
endif()

# kvc2
add_library(kvc2 prefix.cpp)
target_include_directories(kvc2 PRIVATE ${THIRD_PARTY_DIR}/nlohmann/single_include)
add_third_party_includes(kvc2)
if(KTRANSFORMERS_USE_NPU)
    target_link_libraries(kvc2 PUBLIC TBB::tbb xxHash::xxhash cache_entry cuda_stream_manager page_aligned_memory_pool prometheus-cpp::pull kvc2_metrics)
else()
    target_link_libraries(kvc2 PUBLIC TBB::tbb xxHash::xxhash cache_entry cuda_stream_manager page_aligned_memory_pool ${TORCH_LIBRARIES} prometheus-cpp::pull kvc2_metrics)
endif()

message(STATUS "CMAKE_SOURCE_DIR: " ${CMAKE_SOURCE_DIR})

# async_store
add_library(async_store async_store.cpp)
target_include_directories(async_store PRIVATE ${THIRD_PARTY_DIR}/nlohmann/single_include)
target_include_directories(async_store PRIVATE ${THIRD_PARTY_DIR}/spdlog/include)
if(KTRANSFORMERS_USE_NPU)
    target_link_libraries(async_store PUBLIC pthread)
else()
    target_include_directories(async_store PRIVATE ${THIRD_PARTY_DIR}/PhotonLibOS/include)
    target_link_libraries(async_store PUBLIC photon_static pthread)
endif()

# cuda_stream_manager
add_library(cuda_stream_manager cuda_stream_manager.cpp)
target_include_directories(cuda_stream_manager PUBLIC ${THIRD_PARTY_DIR}/nlohmann/single_include)
target_include_directories(cuda_stream_manager PUBLIC ${THIRD_PARTY_DIR}/spdlog/include)
if(KTRANSFORMERS_USE_NPU)
    set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
    target_include_directories(cuda_stream_manager PUBLIC ${ASCEND_HOME_PATH}/include)
    target_link_directories(cuda_stream_manager PUBLIC ${ASCEND_HOME_PATH}/lib64)
    target_link_libraries(cuda_stream_manager PUBLIC ascendcl)
else()
    target_include_directories(cuda_stream_manager PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(cuda_stream_manager PUBLIC CUDA::cudart)
endif()
