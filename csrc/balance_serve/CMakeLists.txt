
cmake_minimum_required(VERSION 3.21)
find_program(GCC_COMPILER NAMES g++-13 g++-12 g++-11 g++ REQUIRED)
set(CMAKE_CXX_COMPILER ${GCC_COMPILER})

# 显示选定的编译器
message(STATUS "Using compiler: ${CMAKE_CXX_COMPILER}")


project(balance_serve VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_FLAGS "-Og -march=native -Wall -Wextra -g -fPIC")
# set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-O3 -march=native -Wall -Wextra -fPIC")
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI})
set(CMAKE_BUILD_TYPE "Release")

file(GLOB_RECURSE FMT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_custom_target(
    format
    COMMAND clang-format
    -i
    -style=file
    ${FMT_SOURCES}
    COMMENT "Running clang-format on all source files"
)

set(BUILD_SHARED_LIBS ON)
set(ENABLE_PUSH OFF)
set(ENABLE_COMPRESSION OFF)

# set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party)
set(THIRD_PARTY_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/third_party)
add_subdirectory(${THIRD_PARTY_DIR}/prometheus-cpp ${THIRD_PARTY_BUILD_DIR}/prometheus-cpp EXCLUDE_FROM_ALL)
add_subdirectory(${THIRD_PARTY_DIR}/xxHash/cmake_unofficial ${THIRD_PARTY_BUILD_DIR}/xxHash EXCLUDE_FROM_ALL)
set_target_properties(xxhash PROPERTIES POSITION_INDEPENDENT_CODE ON)

# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/prometheus-cpp ${CMAKE_CURRENT_BINARY_DIR}/third_party/prometheus-cpp)
set(SPDLOG_DIR ${THIRD_PARTY_DIR}/spdlog)
set(FMT_DIR ${THIRD_PARTY_DIR}/fmt)

set(KVC2_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/kvc2/src)

include_directories(${THIRD_PARTY_DIR})

add_subdirectory(${THIRD_PARTY_DIR}/pybind11 ${THIRD_PARTY_BUILD_DIR}/pybind11)

execute_process(
    COMMAND python3 -c "import torch; print(torch.__path__[0])"
    OUTPUT_VARIABLE TORCH_INSTALL_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Found PyTorch at: ${TORCH_INSTALL_PREFIX}")

# set(TORCH_INSTALL_PREFIX "/home/xwy/.conda/envs/kvc/lib/python3.12/site-packages/torch")
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
find_package(Torch REQUIRED PATHS "${TORCH_INSTALL_PREFIX}/share/cmake/Torch" NO_DEFAULT_PATH)

add_subdirectory(kvc2)
add_subdirectory(sched)

# add_subdirectory(test)
