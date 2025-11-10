set(CMAKE_CXX_FLAGS "-Og -march=native -Wall -Wextra -g -fPIC")
# set(CMAKE_CXX_FLAGS "-O3 -march=native -Wall -Wextra -fPIC")
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI})

set(UTILS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/utils)

option(KTRANSFORMERS_USE_NPU "ktransformers: use NPU" OFF)

if(KTRANSFORMERS_USE_NPU)
    add_definitions(-DKTRANSFORMERS_USE_NPU=1)
endif()

add_library(sched_metrics metrics.cpp)
target_include_directories(sched_metrics PRIVATE ${UTILS_DIR})
target_link_libraries(sched_metrics PUBLIC prometheus-cpp::pull)


add_library(sched scheduler.cpp)
if(KTRANSFORMERS_USE_NPU)
    #target_link_directories(sched PUBLIC ${TORCH_NPU_PATH}/lib)
    target_include_directories(sched PUBLIC ${TORCH_NPU_PATH}/include)
endif()
target_include_directories(sched PRIVATE ${SPDLOG_DIR}/include ${FMT_DIR}/include ${UTILS_DIR} ${KVC2_INCLUDE_DIR})
if(KTRANSFORMERS_USE_NPU)
    target_link_libraries(sched PUBLIC pthread "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}" "${PTA_LIBRARY}" kvc2 async_store sched_metrics)
else()
    target_link_libraries(sched PUBLIC pthread ${TORCH_LIBRARIES} kvc2 async_store sched_metrics)
endif()


pybind11_add_module(sched_ext bind.cpp)
if(KTRANSFORMERS_USE_NPU)
    #target_link_directories(sched_ext PUBLIC ${TORCH_NPU_PATH}/lib)
    target_include_directories(sched_ext PUBLIC ${TORCH_NPU_PATH}/include)
    target_link_libraries(sched_ext PUBLIC "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}" "${PTA_LIBRARY}" sched)
else()
    target_link_libraries(sched_ext PUBLIC sched ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
endif()



