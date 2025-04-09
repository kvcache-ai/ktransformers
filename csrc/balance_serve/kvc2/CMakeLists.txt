cmake_minimum_required(VERSION 3.21)

find_program(GCC_COMPILER NAMES g++-13 g++-12 g++-11 g++ REQUIRED)
set(CMAKE_CXX_COMPILER ${GCC_COMPILER})

project(kvcache-manager VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20)

# set(CMAKE_CXX_FLAGS "-fPIC -O3 -ffast-math -march=native -Wall -Wextra -Wpedantic  -fvisibility=hidden -s")
# set(CMAKE_CXX_FLAGS "-Og -march=native -Wall -Wextra -Wpedantic -g -fsanitize=address")
# set(CMAKE_CXX_FLAGS "-march=native -Wall -Wextra -Wpedantic -g")
# set(CMAKE_CXX_FLAGS "-fPIC -O3 -ffast-math -march=native -Wall -Wextra -g")
set(CMAKE_BUILD_TYPE "Release")
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI})
# set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(BUILD_TEST OFF)
set(BUILD_PYTHON_EXT OFF)

# set(USE_IO_URING ON)
if(USE_IO_URING)
    message(STATUS "Using io_uring")
    add_compile_definitions(USE_IO_URING)
else()
    message(STATUS "Using aio")
endif()

file(GLOB_RECURSE ALL_SOURCE_FILES src/*.cpp src/*.h test/*.cpp test/*.h test/*.hpp)

# 添加一个自定义目标来格式化所有代码
if(NOT TARGET format)
    add_custom_target(
        format
        COMMAND clang-format
        -i
        -style=file
        ${ALL_SOURCE_FILES}
        COMMENT "Running clang-format on all source files"
    )
endif()

execute_process(
    COMMAND python3 -c "import torch; print(torch.__path__[0])"
    OUTPUT_VARIABLE TORCH_INSTALL_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Found PyTorch at: ${TORCH_INSTALL_PREFIX}")

# set(TORCH_INSTALL_PREFIX "/home/xwy/.conda/envs/kvc/lib/python3.12/site-packages/torch")
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
find_package(Torch REQUIRED PATHS "${TORCH_INSTALL_PREFIX}/share/cmake/Torch" NO_DEFAULT_PATH)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../third_party)

find_package(TBB REQUIRED)
find_package(CUDA REQUIRED)

# find_package(prometheus-cpp CONFIG REQUIRED)
if(NOT TARGET prometheus-cpp::pull)
    message(FATAL_ERROR "prometheus-cpp::pull not found")
else()
    message(STATUS "prometheus Found!")
endif()

if(CUDA_FOUND)
    message(STATUS "CUDA Found!")
    message(STATUS "CUDA Version: ${CUDA_VERSION_STRING}")
    message(STATUS "CUDA Toolkit Root: ${CUDA_TOOLKIT_ROOT_DIR}")
else()
    message(FATAL_ERROR "CUDA not found!")
endif()

add_subdirectory(src)

if(BUILD_TEST)
    add_subdirectory(test)
endif()

message(STATUS "BUILD_PYTHON_EXT: ${BUILD_PYTHON_EXT}")

if(BUILD_PYTHON_EXT)
    if(NOT TARGET pybind11::pybind11)
        add_subdirectory(${THIRD_PARTY_DIR}/pybind11 ${THIRD_PARTY_BUILD_DIR}/pybind11)
    endif()

    pybind11_add_module(kvc2_ext src/bind.cpp)

    # EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
    # define (VERSION_INFO) here.
    target_compile_definitions(kvc2_ext PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
    message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
    target_include_directories(kvc2_ext PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/spdlog/include)

    target_link_libraries(kvc2_ext PUBLIC kvc2 async_store)

    install(TARGETS kvc2_ext LIBRARY
        DESTINATION ${CMAKE_BINARY_DIR}/output)
    install(FILES src/kvc2_utils.py
        DESTINATION ${CMAKE_BINARY_DIR}/output)
endif()

