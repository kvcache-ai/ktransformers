cmake_minimum_required(VERSION 3.16)
project(cpuinfer_ext VERSION 0.1.0)


set(CMAKE_CXX_STANDARD 17)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -fopenmp")
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI})
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -ffast-math -fopenmp")
# set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


include(CheckCXXCompilerFlag)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


option(LLAMA_NATIVE                     "llama: enable -march=native flag"                      ON)

# instruction set specific
if (LLAMA_NATIVE)
    set(INS_ENB OFF)
else()
    set(INS_ENB ON)
endif()

option(LLAMA_AVX                             "llama: enable AVX"                                OFF)
option(LLAMA_AVX2                            "llama: enable AVX2"                               OFF)
option(LLAMA_AVX512                          "llama: enable AVX512"                             OFF)
option(LLAMA_AVX512_VBMI                     "llama: enable AVX512-VBMI"                        OFF)
option(LLAMA_AVX512_VNNI                     "llama: enable AVX512-VNNI"                        OFF)
option(LLAMA_AVX512_BF16                     "llama: enable AVX512-BF16"                        OFF)
option(LLAMA_FMA                             "llama: enable FMA"                                OFF)
# in MSVC F16C is implied with AVX2/AVX512
if (NOT MSVC)
    option(LLAMA_F16C                        "llama: enable F16C"                               OFF)
endif()
option(LLAMA_AVX512_FANCY_SIMD               "llama: enable AVX512-VL, AVX512-BW, AVX512-DQ, AVX512-VNNI"                        OFF)
option(KTRANSFORMERS_USE_CUDA                "ktransformers: use CUDA"                          ON)
option(KTRANSFORMERS_USE_MUSA                "ktransformers: use MUSA"                          OFF)
option(KTRANSFORMERS_USE_ROCM                "ktransformers: use ROCM"                          OFF)
option(KTRANSFORMERS_USE_XPU                 "ktransformers: use XPU"                           OFF)

# Architecture specific
# TODO: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if (MSVC)
    string(TOLOWER "${CMAKE_GENERATOR_PLATFORM}" CMAKE_GENERATOR_PLATFORM_LWR)
    message(STATUS "CMAKE_GENERATOR_PLATFORM: ${CMAKE_GENERATOR_PLATFORM}")
else ()
    set(CMAKE_GENERATOR_PLATFORM_LWR "")
endif ()

if (NOT MSVC)
    if (LLAMA_STATIC)
        add_link_options(-static)
        if (MINGW)
            add_link_options(-static-libgcc -static-libstdc++)
        endif()
    endif()
    if (LLAMA_GPROF)
        add_compile_options(-pg)
    endif()
endif()

set(ARCH_FLAGS "")

if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64" OR CMAKE_GENERATOR_PLATFORM_LWR STREQUAL "arm64" OR
    (NOT CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_GENERATOR_PLATFORM_LWR AND
     CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm.*|ARM64)$"))
    message(STATUS "ARM detected")
    if (MSVC)
        add_compile_definitions(__aarch64__) # MSVC defines _M_ARM64 instead
        add_compile_definitions(__ARM_NEON)
        add_compile_definitions(__ARM_FEATURE_FMA)

        set(CMAKE_REQUIRED_FLAGS_PREV ${CMAKE_REQUIRED_FLAGS})
        string(JOIN " " CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS} "/arch:armv8.2")
        check_cxx_source_compiles("#include <arm_neon.h>\nint main() { int8x16_t _a, _b; int32x4_t _s = vdotq_s32(_s, _a, _b); return 0; }" GGML_COMPILER_SUPPORT_DOTPROD)
        if (GGML_COMPILER_SUPPORT_DOTPROD)
            add_compile_definitions(__ARM_FEATURE_DOTPROD)
        endif ()
        check_cxx_source_compiles("#include <arm_neon.h>\nint main() { float16_t _a; float16x8_t _s = vdupq_n_f16(_a); return 0; }" GGML_COMPILER_SUPPORT_FP16_VECTOR_ARITHMETIC)
        if (GGML_COMPILER_SUPPORT_FP16_VECTOR_ARITHMETIC)
            add_compile_definitions(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        endif ()
        set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_PREV})
    else()
        check_cxx_compiler_flag(-mfp16-format=ieee COMPILER_SUPPORTS_FP16_FORMAT_I3E)
        if (NOT "${COMPILER_SUPPORTS_FP16_FORMAT_I3E}" STREQUAL "")
            list(APPEND ARCH_FLAGS -mfp16-format=ieee)
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv6")
            # Raspberry Pi 1, Zero
            list(APPEND ARCH_FLAGS -mfpu=neon-fp-armv8 -mno-unaligned-access)
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv7")
            if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Android")
                # Android armeabi-v7a
                list(APPEND ARCH_FLAGS -mfpu=neon-vfpv4 -mno-unaligned-access -funsafe-math-optimizations)
            else()
                # Raspberry Pi 2
                list(APPEND ARCH_FLAGS -mfpu=neon-fp-armv8 -mno-unaligned-access -funsafe-math-optimizations)
            endif()
        endif()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv8")
            # Android arm64-v8a
            # Raspberry Pi 3, 4, Zero 2 (32-bit)
            list(APPEND ARCH_FLAGS -mno-unaligned-access)
        endif()
    endif()
elseif (CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64" OR CMAKE_GENERATOR_PLATFORM_LWR MATCHES "^(x86_64|i686|amd64|x64|win32)$" OR
        (NOT CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_GENERATOR_PLATFORM_LWR AND
         CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|i686|AMD64)$"))
    message(STATUS "x86 detected")
    set(HOST_IS_X86 TRUE)
    set(HAS_AVX512 TRUE)
    set(__HAS_AMX__ TRUE)
    add_compile_definitions(__x86_64__)
    # check AVX512
    execute_process(
        COMMAND lscpu
        OUTPUT_VARIABLE LSCPU_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    # message(STATUS "LSCPU_OUTPUT: ${LSCPU_OUTPUT}")

    string(FIND "${LSCPU_OUTPUT}" "avx512" COMPILER_SUPPORTS_AVX512F)
    
    if (COMPILER_SUPPORTS_AVX512F GREATER -1)
        message(STATUS "Compiler and CPU support AVX512F (tested by compiling a program)")
        add_compile_definitions(__HAS_AVX512F__)
    else()
        message(STATUS "Compiler and/or CPU do NOT support AVX512F")
        set(HAS_AVX512 False)
    endif()

    # check AMX
    string(FIND "${LSCPU_OUTPUT}" "amx" COMPILER_SUPPORTS_AMX)
    
    if(COMPILER_SUPPORTS_AMX GREATER -1)
        message(STATUS "Compiler supports AMX")
        add_compile_definitions(__HAS_AMX__)
    else()
        message(STATUS "Compiler does NOT support AMX")
    endif()
    if (MSVC)
        # instruction set detection for MSVC only
        if (LLAMA_NATIVE)
            include(cmake/FindSIMD.cmake)
        endif ()
        if (LLAMA_AVX512)
            list(APPEND ARCH_FLAGS /arch:AVX512)
            # MSVC has no compile-time flags enabling specific
            # AVX512 extensions, neither it defines the
            # macros corresponding to the extensions.
            # Do it manually.
            if (LLAMA_AVX512_VBMI)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512VBMI__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512VBMI__>)
            endif()
            if (LLAMA_AVX512_VNNI)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512VNNI__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512VNNI__>)
            endif()
            if (LLAMA_AVX512_FANCY_SIMD)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512VL__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512VL__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512BW__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512BW__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512DQ__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512DQ__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512VNNI__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512VNNI__>)
            endif()
            if (LLAMA_AVX512_BF16)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512BF16__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512BF16__>)
            endif()
        elseif (LLAMA_AVX2)
            list(APPEND ARCH_FLAGS /arch:AVX2)
        elseif (LLAMA_AVX)
            list(APPEND ARCH_FLAGS /arch:AVX)
        endif()
    else()
        if (LLAMA_NATIVE)
            list(APPEND ARCH_FLAGS -mfma -mavx -mavx2)
            list(APPEND ARCH_FLAGS -march=native)
        endif()
        if (LLAMA_F16C)
            list(APPEND ARCH_FLAGS -mf16c)
        endif()
        if (LLAMA_FMA)
            list(APPEND ARCH_FLAGS -mfma)
        endif()
        if (LLAMA_AVX)
            list(APPEND ARCH_FLAGS -mavx)
        endif()
        if (LLAMA_AVX2)
            list(APPEND ARCH_FLAGS -mavx2)
        endif()
        if (LLAMA_AVX512)
            list(APPEND ARCH_FLAGS -mavx512f)
            list(APPEND ARCH_FLAGS -mavx512bw)
        endif()
        if (LLAMA_AVX512_VBMI)
            list(APPEND ARCH_FLAGS -mavx512vbmi)
        endif()
        if (LLAMA_AVX512_VNNI)
            list(APPEND ARCH_FLAGS -mavx512vnni)
        endif()
        if (LLAMA_AVX512_FANCY_SIMD)
            message(STATUS "AVX512-VL, AVX512-BW, AVX512-DQ, AVX512-VNNI enabled")
            list(APPEND ARCH_FLAGS -mavx512vl)
            list(APPEND ARCH_FLAGS -mavx512bw)
            list(APPEND ARCH_FLAGS -mavx512dq)
            list(APPEND ARCH_FLAGS -mavx512vnni)
            list(APPEND ARCH_FLAGS -mavx512vpopcntdq)
        endif()
        if (LLAMA_AVX512_BF16)
            list(APPEND ARCH_FLAGS -mavx512bf16)
        endif()
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64")
    message(STATUS "PowerPC detected")
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64le")
        list(APPEND ARCH_FLAGS -mcpu=powerpc64le)
    else()
        list(APPEND ARCH_FLAGS -mcpu=native -mtune=native)
        #TODO: Add  targets for Power8/Power9 (Altivec/VSX) and Power10(MMA) and query for big endian systems (ppc64/le/be)
    endif()
else()
    message(STATUS "Unknown architecture")
endif()

# message(STATUS "CUDAToolkit_ROOT:${CUDAToolkit_ROOT}")
# find_package(FindCUDAToolkit REQUIRED)
# if(CUDAToolkit_FOUND)
#     message(STATUS "Found CUDA cudart lib at:${CUDAToolkit_LIBRARY_DIR}")
# else()
#     message(STATUS "Can't found CUDA lib")
# endif()

if (NOT EXISTS $ENV{ROCM_PATH})
    if (NOT EXISTS /opt/rocm)
        set(ROCM_PATH /usr)
    else()
        set(ROCM_PATH /opt/rocm)
    endif()
else()
    set(ROCM_PATH $ENV{ROCM_PATH})
endif()

list(APPEND CMAKE_PREFIX_PATH  ${ROCM_PATH})
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib64/cmake")

if (NOT EXISTS $ENV{MUSA_PATH})
    if (NOT EXISTS /opt/musa)
        set(MUSA_PATH /usr/local/musa)
    else()
        set(MUSA_PATH /opt/musa)
    endif()
else()
    set(MUSA_PATH $ENV{MUSA_PATH})
endif()

list(APPEND CMAKE_MODULE_PATH "${MUSA_PATH}/cmake")

add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${ARCH_FLAGS}>")
add_compile_options("$<$<COMPILE_LANGUAGE:C>:${ARCH_FLAGS}>")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/pybind11 ${CMAKE_CURRENT_BINARY_DIR}/third_party/pybind11)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/llama.cpp ${CMAKE_CURRENT_BINARY_DIR}/third_party/llama.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../third_party)
if (WIN32)
    include_directories("$ENV{CUDA_PATH}/include")
    add_compile_definitions(KTRANSFORMERS_USE_CUDA=1)
elseif (UNIX)
    if (KTRANSFORMERS_USE_ROCM)
        find_package(HIP REQUIRED)
        if(HIP_FOUND)
            include_directories("${HIP_INCLUDE_DIRS}")
            add_compile_definitions(KTRANSFORMERS_USE_ROCM=1)
        endif()
    elseif (KTRANSFORMERS_USE_MUSA)
        if (NOT EXISTS $ENV{MUSA_PATH})
            if (NOT EXISTS /opt/musa)
                set(MUSA_PATH /usr/local/musa)
            else()
                set(MUSA_PATH /opt/musa)
            endif()
        else()
            set(MUSA_PATH $ENV{MUSA_PATH})
        endif()

        list(APPEND CMAKE_MODULE_PATH "${MUSA_PATH}/cmake")

        find_package(MUSAToolkit)
        if (MUSAToolkit_FOUND)
            message(STATUS "MUSA Toolkit found")
            add_compile_definitions(KTRANSFORMERS_USE_MUSA=1)
        endif()
    elseif (KTRANSFORMERS_USE_XPU)
        add_compile_definitions(KTRANSFORMERS_USE_XPU=1)
    else()
        find_package(CUDA REQUIRED)
        include_directories("${CUDA_INCLUDE_DIRS}")
        include(CheckLanguage)
        check_language(CUDA)
        if(CMAKE_CUDA_COMPILER)
            message(STATUS "CUDA detected")
            find_package(CUDAToolkit REQUIRED)
            include_directories(${CUDAToolkit_INCLUDE_DIRS})
        endif()
        message(STATUS "enabling CUDA")
        enable_language(CUDA)
        add_compile_definitions(KTRANSFORMERS_USE_CUDA=1)
    endif()
endif()

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SOURCE_DIR1)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/cpu_backend SOURCE_DIR2)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/operators/llamafile SOURCE_DIR3)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/llamafile SOURCE_DIR4)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/operators/kvcache SOURCE_DIR5)

if (HOST_IS_X86 AND HAS_AVX512 AND __HAS_AMX__)
    aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/operators/amx SOURCE_DIR6)
endif()


set(ALL_SOURCES ${SOURCE_DIR1} ${SOURCE_DIR2} ${SOURCE_DIR3} ${SOURCE_DIR4} ${SOURCE_DIR5} ${SOURCE_DIR6})

file(GLOB_RECURSE FMT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")

add_custom_target(
    format
    COMMAND clang-format
    -i
    -style=file
    ${FMT_SOURCES}
    COMMENT "Running clang-format on all source files"
)


add_library(llamafile STATIC ${SOURCE_DIR4})

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "ARCH_FLAGS: ${ARCH_FLAGS}")
pybind11_add_module(${PROJECT_NAME} MODULE ${ALL_SOURCES})
target_link_libraries(${PROJECT_NAME} PRIVATE llama)


if(WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE "$ENV{CUDA_PATH}/lib/x64/cudart.lib")#CUDA::cudart
elseif(UNIX)
    if (KTRANSFORMERS_USE_ROCM)
        add_compile_definitions(USE_HIP=1)
        target_link_libraries(${PROJECT_NAME} PRIVATE "${ROCM_PATH}/lib/libamdhip64.so")
        message(STATUS "Building for HIP")
    elseif(KTRANSFORMERS_USE_MUSA)
        target_link_libraries(${PROJECT_NAME} PRIVATE MUSA::musart)
    elseif(KTRANSFORMERS_USE_XPU)
    else()
        target_link_libraries(${PROJECT_NAME} PRIVATE "${CUDAToolkit_LIBRARY_DIR}/libcudart.so")
    endif()
endif()

# Define the USE_NUMA option
option(USE_NUMA "Disable NUMA support" OFF)

# Check if the USE_NUMA environment variable is set
if(DEFINED ENV{USE_NUMA})
    set(USE_NUMA ON)
endif()

if(USE_NUMA)
    message(STATUS "NUMA support is enabled")
else()
    message(STATUS "NUMA support is disabled")
endif()

find_library(NUMA_LIBRARY NAMES numa)

if(NUMA_LIBRARY AND USE_NUMA)
    message(STATUS "NUMA library found: ${NUMA_LIBRARY} - enabling NUMA support")
    target_link_libraries(${PROJECT_NAME} PRIVATE ${NUMA_LIBRARY})
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_NUMA)
else()
    if(USE_NUMA)
        message(FATAL_ERROR "NUMA library not found - maybe sudo apt install libnuma-dev")
    else()
        message(STATUS "NUMA library not found or user not set USE_NUMA - disabling NUMA support")
    endif()
endif()