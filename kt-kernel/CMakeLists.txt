cmake_minimum_required(VERSION 3.16)

# Toggle: default to system compilers; optionally use conda toolchain
option(USE_CONDA_TOOLCHAIN "Use C/C++ compilers and libraries from active conda env" OFF)
option(LLAMA_NATIVE "llama: enable -march=native flag" OFF)
option(LLAMA_AVX "llama: enable AVX" OFF)
option(LLAMA_AVX2 "llama: enable AVX2" OFF)
option(LLAMA_AVX512 "llama: enable AVX512" OFF)
option(LLAMA_AVX512_VBMI "llama: enable AVX512-VBMI" OFF)
option(LLAMA_AVX512_VNNI "llama: enable AVX512-VNNI" OFF)
option(LLAMA_AVX512_BF16 "llama: enable AVX512-BF16" OFF)
option(LLAMA_FMA "llama: enable FMA" OFF)
# in MSVC F16C is implied with AVX2/AVX512
if(NOT MSVC)
    option(LLAMA_F16C "llama: enable F16C" OFF)
endif()
option(LLAMA_AVX512_FANCY_SIMD "llama: enable AVX512-VL, AVX512-BW, AVX512-DQ, AVX512-VNNI" OFF)
option(KTRANSFORMERS_USE_CUDA "ktransformers: use CUDA" OFF)
option(KTRANSFORMERS_USE_MUSA "ktransformers: use MUSA" OFF)
option(KTRANSFORMERS_USE_ROCM "ktransformers: use ROCM" OFF)
option(KTRANSFORMERS_CPU_USE_KML "ktransformers: CPU use KML" OFF)
option(KTRANSFORMERS_CPU_USE_AMX_AVX512 "ktransformers: CPU use AMX or AVX512" OFF)
option(KTRANSFORMERS_CPU_USE_AMX "ktransformers: CPU use AMX" OFF)
option(KTRANSFORMERS_CPU_DEBUG "ktransformers: DEBUG CPU use AMX" OFF)
option(KTRANSFORMERS_CPU_MLA "ktransformers: CPU use MLA" OFF)
option(KTRANSFORMERS_CPU_MOE_KERNEL "ktransformers: CPU use moe kernel" OFF)
option(KTRANSFORMERS_CPU_MOE_AMD "ktransformers: CPU use moe kernel for amd" OFF)
# LTO control
option(CPUINFER_ENABLE_LTO "Enable link time optimization (IPO)" OFF)

project(kt_kernel_ext VERSION 0.1.0)
# Choose compilers BEFORE project() so CMake honors them
if(USE_CONDA_TOOLCHAIN)
    if(NOT DEFINED ENV{CONDA_PREFIX} OR NOT EXISTS "$ENV{CONDA_PREFIX}")
        message(FATAL_ERROR "USE_CONDA_TOOLCHAIN=ON but CONDA_PREFIX is not set. Activate your conda env or pass -DCONDA_PREFIX=/path")
    endif()
    # Locate conda GCC wrappers
    find_program(CONDA_CC NAMES x86_64-conda-linux-gnu-cc HINTS "$ENV{CONDA_PREFIX}/bin")
    find_program(CONDA_CXX NAMES x86_64-conda-linux-gnu-c++ HINTS "$ENV{CONDA_PREFIX}/bin")
    if(NOT CONDA_CC OR NOT CONDA_CXX)
        message(FATAL_ERROR "Conda compilers not found in $ENV{CONDA_PREFIX}/bin (expected x86_64-conda-linux-gnu-cc/c++).")
    endif()
    set(CMAKE_C_COMPILER   ${CONDA_CC}  CACHE FILEPATH "C compiler" FORCE)
    set(CMAKE_CXX_COMPILER ${CONDA_CXX} CACHE FILEPATH "C++ compiler" FORCE)
else()
    # Prefer system compilers explicitly to avoid accidentally picking conda wrappers from PATH
    if(EXISTS "/usr/bin/gcc" AND EXISTS "/usr/bin/g++")
        set(CMAKE_C_COMPILER   "/usr/bin/gcc" CACHE FILEPATH "C compiler" FORCE)
        set(CMAKE_CXX_COMPILER "/usr/bin/g++" CACHE FILEPATH "C++ compiler" FORCE)
    endif()
endif()


# If explicitly using conda toolchain, prefer its libraries/headers and RPATH
if(USE_CONDA_TOOLCHAIN)
    message(STATUS "Conda prefix detected: $ENV{CONDA_PREFIX}; prioritizing it for search paths and RPATH")
    # Make conda come first for CMake package discovery
    list(PREPEND CMAKE_PREFIX_PATH
        "$ENV{CONDA_PREFIX}"
        "$ENV{CONDA_PREFIX}/lib/cmake"
        "$ENV{CONDA_PREFIX}/share/cmake"
    )
    # Also hint direct include/lib searches
    list(PREPEND CMAKE_LIBRARY_PATH "$ENV{CONDA_PREFIX}/lib")
    list(PREPEND CMAKE_INCLUDE_PATH "$ENV{CONDA_PREFIX}/include")

    # Ensure pkg-config prefers conda .pc files
    set(ENV{PKG_CONFIG_PATH} "$ENV{CONDA_PREFIX}/lib/pkgconfig:$ENV{CONDA_PREFIX}/share/pkgconfig:$ENV{PKG_CONFIG_PATH}")
    # Make FindPkgConfig also search CMAKE_PREFIX_PATH
    set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH ON)

    # Configure RPATH so the built extension prefers conda's shared libs at runtime
    # Use install RPATH during build to avoid mixing with implicit system paths
    set(CMAKE_SKIP_BUILD_RPATH FALSE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    set(CMAKE_BUILD_RPATH "$ENV{CONDA_PREFIX}/lib")
    set(CMAKE_INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib")
    # Do not auto-append link directories to RPATH; we want only conda path here
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH OFF)
endif()

## Ensure git hooks are installed when configuring the project (monorepo-aware)
# If we are inside a git worktree (repo root is outside kt-kernel now), invoke the installer
# which will link kt-kernel/.githooks into the top-level .git/hooks. Otherwise, skip.
find_program(GIT_BIN git)
if(GIT_BIN)
    execute_process(
        COMMAND "${GIT_BIN}" rev-parse --show-toplevel
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        OUTPUT_VARIABLE _GIT_TOP
        RESULT_VARIABLE _GIT_RV
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(_GIT_RV EQUAL 0 AND EXISTS "${_GIT_TOP}/.git" AND IS_DIRECTORY "${_GIT_TOP}/.git")
        if(EXISTS "${CMAKE_SOURCE_DIR}/scripts/install-git-hooks.sh")
            message(STATUS "Detected git worktree at ${_GIT_TOP}; installing hooks from kt-kernel/.githooks")
            execute_process(
                COMMAND sh "${CMAKE_SOURCE_DIR}/scripts/install-git-hooks.sh"
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                RESULT_VARIABLE _INSTALL_GIT_HOOKS_RESULT
                OUTPUT_VARIABLE _INSTALL_GIT_HOOKS_OUT
                ERROR_VARIABLE _INSTALL_GIT_HOOKS_ERR
            )
            if(NOT _INSTALL_GIT_HOOKS_RESULT EQUAL 0)
                message(FATAL_ERROR "Installing git hooks failed (exit ${_INSTALL_GIT_HOOKS_RESULT}).\nOutput:\n${_INSTALL_GIT_HOOKS_OUT}\nError:\n${_INSTALL_GIT_HOOKS_ERR}")
            endif()
        else()
            message(FATAL_ERROR "Required script 'scripts/install-git-hooks.sh' not found in kt-kernel; cannot install hooks.")
        endif()
    else()
        message(STATUS "No git worktree detected; skipping git hooks installation")
    endif()
else()
    message(STATUS "git not found; skipping git hooks installation")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use header-only fmt to avoid needing to link libfmt (fix undefined symbol vprint)
add_compile_definitions(FMT_HEADER_ONLY)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math")
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address -fno-omit-frame-pointer")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
# set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(OpenMP REQUIRED)
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")


include(CheckCXXCompilerFlag)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)



# instruction set specific
if(LLAMA_NATIVE)
    set(INS_ENB OFF)
else()
    set(INS_ENB ON)
endif()
# Architecture specific
# TODO: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")

set(ARCH_FLAGS "")

if(CMAKE_OSX_ARCHITECTURES STREQUAL "arm64" OR CMAKE_GENERATOR_PLATFORM_LWR STREQUAL "arm64" OR
    (NOT CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_GENERATOR_PLATFORM_LWR AND
         CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm.*|ARM64)$"))
    message(STATUS "ARM detected")
    if(MSVC)
        add_compile_definitions(__aarch64__) # MSVC defines _M_ARM64 instead
        add_compile_definitions(__ARM_NEON)
        add_compile_definitions(__ARM_FEATURE_FMA)

        set(CMAKE_REQUIRED_FLAGS_PREV ${CMAKE_REQUIRED_FLAGS})
        string(JOIN " " CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS} "/arch:armv8.2")
        check_cxx_source_compiles("#include <arm_neon.h>\nint main() { int8x16_t _a, _b; int32x4_t _s = vdotq_s32(_s, _a, _b); return 0; }" GGML_COMPILER_SUPPORT_DOTPROD)
        if(GGML_COMPILER_SUPPORT_DOTPROD)
            add_compile_definitions(__ARM_FEATURE_DOTPROD)
        endif()
        check_cxx_source_compiles("#include <arm_neon.h>\nint main() { float16_t _a; float16x8_t _s = vdupq_n_f16(_a); return 0; }" GGML_COMPILER_SUPPORT_FP16_VECTOR_ARITHMETIC)
        if(GGML_COMPILER_SUPPORT_FP16_VECTOR_ARITHMETIC)
            add_compile_definitions(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        endif()
        set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_PREV})
    else()
        check_cxx_compiler_flag(-mfp16-format=ieee COMPILER_SUPPORTS_FP16_FORMAT_I3E)
        if(NOT "${COMPILER_SUPPORTS_FP16_FORMAT_I3E}" STREQUAL "")
            list(APPEND ARCH_FLAGS -mfp16-format=ieee)
        endif()
        if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv6")
            # Raspberry Pi 1, Zero
            list(APPEND ARCH_FLAGS -mfpu=neon-fp-armv8 -mno-unaligned-access)
        endif()
        if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv7")
            if("${CMAKE_SYSTEM_NAME}" STREQUAL "Android")
                # Android armeabi-v7a
                list(APPEND ARCH_FLAGS -mfpu=neon-vfpv4 -mno-unaligned-access -funsafe-math-optimizations)
            else()
                # Raspberry Pi 2
                list(APPEND ARCH_FLAGS -mfpu=neon-fp-armv8 -mno-unaligned-access -funsafe-math-optimizations)
            endif()
        endif()
        if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv8")
            # Android arm64-v8a
            # Raspberry Pi 3, 4, Zero 2 (32-bit)
            list(APPEND ARCH_FLAGS -mno-unaligned-access)
        endif()
        # add_compile_definitions(__ARM_NEON)
        # list(APPEND ARCH_FLAGS -march=armv8.2-a+fp16+dotprod)
        # add_compile_definitions(__ARM_FEATURE_DOTPROD)
        # add_compile_definitions(__aarch64__)

        # add_compile_definitions(__ARM_NEON)
        list(APPEND ARCH_FLAGS -march=armv8.2-a+fp16+dotprod+sve+bf16)
        # list(APPEND ARCH_FLAGS -march=armv8-a+dotprod+sha3+sm4+fp16fml+sve+rng+sb+ssbs+i8mm+bf16+flagm+pauth)
        # add_compile_definitions(__ARM_FEATURE_DOTPROD)
        # add_compile_definitions(__ARM_FEATURE_SVE)
        # add_compile_definitions(__ARM_FEATURE_MATMUL_INT8)
        # add_compile_definitions(__aarch64__)
    endif()
elseif(CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64" OR CMAKE_GENERATOR_PLATFORM_LWR MATCHES "^(x86_64|i686|amd64|x64|win32)$" OR
    (NOT CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_GENERATOR_PLATFORM_LWR AND
         CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|i686|AMD64)$"))
    message(STATUS "x86 detected")
    set(HOST_IS_X86 TRUE)
    add_compile_definitions(__x86_64__)
    if(MSVC)
        # instruction set detection for MSVC only
        if(LLAMA_NATIVE)
            include(cmake/FindSIMD.cmake)
        endif()
        if(LLAMA_AVX512)
            list(APPEND ARCH_FLAGS /arch:AVX512)
            # MSVC has no compile-time flags enabling specific
            # AVX512 extensions, neither it defines the
            # macros corresponding to the extensions.
            # Do it manually.
            if(LLAMA_AVX512_VBMI)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512VBMI__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512VBMI__>)
            endif()
            if(LLAMA_AVX512_VNNI)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512VNNI__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512VNNI__>)
            endif()
            if(LLAMA_AVX512_FANCY_SIMD)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512VL__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512VL__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512BW__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512BW__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512DQ__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512DQ__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512VNNI__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512VNNI__>)
            endif()
            if(LLAMA_AVX512_BF16)
                add_compile_definitions($<$<COMPILE_LANGUAGE:C>:__AVX512BF16__>)
                add_compile_definitions($<$<COMPILE_LANGUAGE:CXX>:__AVX512BF16__>)
            endif()
        elseif(LLAMA_AVX2)
            list(APPEND ARCH_FLAGS /arch:AVX2)
        elseif(LLAMA_AVX)
            list(APPEND ARCH_FLAGS /arch:AVX)
        endif()
    else()
        if(LLAMA_NATIVE)
            list(APPEND ARCH_FLAGS -mfma -mavx -mavx2)
            list(APPEND ARCH_FLAGS -march=native)
        endif()
        if(LLAMA_F16C)
            list(APPEND ARCH_FLAGS -mf16c)
        endif()
        if(LLAMA_FMA)
            list(APPEND ARCH_FLAGS -mfma)
        endif()
        if(LLAMA_AVX)
            list(APPEND ARCH_FLAGS -mavx -mfma -msse3 -mf16c)
            message(WARNING "pure AVX is not supported at least avx2")
        endif()
        if(LLAMA_AVX2)
            list(APPEND ARCH_FLAGS -mavx2 -mfma -msse3 -mf16c)
        endif()
        if(LLAMA_AVX512)
            list(APPEND ARCH_FLAGS -mavx512f -mavx512bw -mavx512dq -mfma -mf16c -msse3)
        endif()
        if(LLAMA_AVX512_VBMI)
            list(APPEND ARCH_FLAGS -mavx512vbmi)
        endif()
        if(LLAMA_AVX512_VNNI)
            list(APPEND ARCH_FLAGS -mavx512vnni)
        endif()
        if(LLAMA_AVX512_FANCY_SIMD)
            message(STATUS "AVX512-VL, AVX512-BW, AVX512-DQ, AVX512-VNNI enabled")
            list(APPEND ARCH_FLAGS -mavx512vl)
            list(APPEND ARCH_FLAGS -mavx512bw)
            list(APPEND ARCH_FLAGS -mavx512dq)
            list(APPEND ARCH_FLAGS -mavx512vnni)
            list(APPEND ARCH_FLAGS -mavx512vpopcntdq)
        endif()
        if(LLAMA_AVX512_BF16)
            list(APPEND ARCH_FLAGS -mavx512bf16)
        endif()
    endif()
elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64")
    message(STATUS "PowerPC detected")
    if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64le")
        list(APPEND ARCH_FLAGS -mcpu=powerpc64le)
    else()
        list(APPEND ARCH_FLAGS -mcpu=native -mtune=native)
        #TODO: Add  targets for Power8/Power9 (Altivec/VSX) and Power10(MMA) and query for big endian systems (ppc64/le/be)
    endif()
else()
    message(STATUS "Unknown architecture")
endif()

if(NOT EXISTS $ENV{ROCM_PATH})
    if(NOT EXISTS /opt/rocm)
        set(ROCM_PATH /usr)
    else()
        set(ROCM_PATH /opt/rocm)
    endif()
else()
    set(ROCM_PATH $ENV{ROCM_PATH})
endif()

list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}/lib64/cmake")

if(NOT EXISTS $ENV{MUSA_PATH})
    if(NOT EXISTS /opt/musa)
        set(MUSA_PATH /usr/local/musa)
    else()
        set(MUSA_PATH /opt/musa)
    endif()
else()
    set(MUSA_PATH $ENV{MUSA_PATH})
endif()

list(APPEND CMAKE_MODULE_PATH "${MUSA_PATH}/cmake")

if(KTRANSFORMERS_CPU_MOE_AMD)
    set(BLIS_ROOT "" CACHE PATH "Root directory of BLIS installation")
    set(_BLIS_SEARCH_DIRS)
    if(BLIS_ROOT)
        list(APPEND _BLIS_SEARCH_DIRS "${BLIS_ROOT}")
    endif()
    list(APPEND _BLIS_SEARCH_DIRS "/usr/local" "/usr")

    find_path(BLIS_INCLUDE_DIR
        NAMES blis.h
        HINTS ${_BLIS_SEARCH_DIRS}
        PATH_SUFFIXES include include/blis
    )
    find_library(BLIS_LIBRARY
        NAMES blis
        HINTS ${_BLIS_SEARCH_DIRS}
        PATH_SUFFIXES lib lib64
    )

    if(NOT BLIS_INCLUDE_DIR OR NOT BLIS_LIBRARY)
        message(WARNING "BLIS not found; set BLIS_ROOT or specify BLIS_INCLUDE_DIR/BLIS_LIBRARY")
    else()
        message(STATUS "Found BLIS include at ${BLIS_INCLUDE_DIR}")
        message(STATUS "Found BLIS library ${BLIS_LIBRARY}")
        set(_KT_BLIS_INCLUDE_DIR ${BLIS_INCLUDE_DIR})
        set(_KT_BLIS_LIBRARY ${BLIS_LIBRARY})
    endif()
    # The Python extension target (${PROJECT_NAME}) is created later by
    # pybind11_add_module(). Calling target_include_directories/target_link_libraries
    # here would fail because the target doesn't exist yet. Save the discovered
    # BLIS paths and apply them after the module target is created.
endif()


if(HOST_IS_X86)
    if(KTRANSFORMERS_CPU_USE_AMX_AVX512)
        add_compile_definitions(USE_AMX_AVX_KERNEL=1)
        if(KTRANSFORMERS_CPU_USE_AMX)
            add_compile_definitions(HAVE_AMX=1)
            list(APPEND ARCH_FLAGS -mamx-tile -mamx-bf16 -mamx-int8)
            message(STATUS "AMX enabled")
        endif()
        # add_executable(amx-test ${CMAKE_CURRENT_SOURCE_DIR}/operators/amx/amx-test.cpp)
        # target_link_libraries(amx-test llama)
        if(KTRANSFORMERS_CPU_DEBUG)
            file(GLOB AMX_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/operators/amx/test/*.cpp")
            foreach(test_src ${AMX_TEST_SOURCES})
                # 获取不带扩展名的文件名作为 target 名
                get_filename_component(test_name ${test_src} NAME_WE)
                add_executable(${test_name} ${test_src} ${CMAKE_CURRENT_SOURCE_DIR}/cpu_backend/shared_mem_buffer.cpp)
                target_link_libraries(${test_name} llama OpenMP::OpenMP_CXX numa)
            endforeach()
        endif()
        list(APPEND ARCH_FLAGS -mfma -mf16c -mavx512bf16 -mavx512vnni)
    endif()
endif()

message(STATUS "ARCH_FLAGS: ${ARCH_FLAGS}")

add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${ARCH_FLAGS}>")
add_compile_options("$<$<COMPILE_LANGUAGE:C>:${ARCH_FLAGS}>")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/pybind11 ${CMAKE_CURRENT_BINARY_DIR}/third_party/pybind11)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llama.cpp ${CMAKE_CURRENT_BINARY_DIR}/third_party/llama.cpp)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../third_party)
if(KTRANSFORMERS_USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        message(STATUS "CUDA detected")
        find_package(CUDAToolkit REQUIRED)
        include_directories(${CUDAToolkit_INCLUDE_DIRS})
    else()
        message(FATAL_ERROR "KTRANSFORMERS_USE_CUDA=ON but CUDA compiler not found")
    endif()
    message(STATUS "enabling CUDA")
    enable_language(CUDA)
    add_compile_definitions(KTRANSFORMERS_USE_CUDA=1)
elseif(KTRANSFORMERS_USE_ROCM)
    find_package(HIP REQUIRED)
    if(HIP_FOUND)
        include_directories("${HIP_INCLUDE_DIRS}")
        add_compile_definitions(KTRANSFORMERS_USE_ROCM=1)
    endif()
elseif(KTRANSFORMERS_USE_MUSA)
    if(NOT EXISTS $ENV{MUSA_PATH})
        if(NOT EXISTS /opt/musa)
            set(MUSA_PATH /usr/local/musa)
        else()
            set(MUSA_PATH /opt/musa)
        endif()
    else()
        set(MUSA_PATH $ENV{MUSA_PATH})
    endif()

    list(APPEND CMAKE_MODULE_PATH "${MUSA_PATH}/cmake")

    find_package(MUSAToolkit)
    if(MUSAToolkit_FOUND)
        message(STATUS "MUSA Toolkit found")
        add_compile_definitions(KTRANSFORMERS_USE_MUSA=1)
    endif()
elseif(KTRANSFORMERS_CPU_USE_KML)
    message(STATUS "KML CPU detected")
else()
    message(STATUS "No GPU support enabled, building for CPU only")
    add_compile_definitions(KTRANSFORMERS_CPU_ONLY=1)
endif()

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SOURCE_DIR1)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/cpu_backend SOURCE_DIR2)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/operators/llamafile SOURCE_DIR3)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/../third_party/llamafile SOURCE_DIR4)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/operators/kvcache SOURCE_DIR5)
# message(STATUS "SOURCE_DIR3: ${SOURCE_DIR3}")

# arm64
if(NOT HOST_IS_X86 AND KTRANSFORMERS_CPU_USE_KML)
    aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/operators/kml SOURCE_DIR6)
    if(NOT KTRANSFORMERS_CPU_MLA)
        list(REMOVE_ITEM SOURCE_DIR6 ${CMAKE_CURRENT_SOURCE_DIR}/operators/kml/mla/)
    endif()
endif()
# message(STATUS "SOURCE_DIR6: ${SOURCE_DIR6}")

if(KTRANSFORMERS_CPU_MOE_KERNEL)
    aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/operators/moe_kernel/la SOURCE_DIR7)
    if(KTRANSFORMERS_CPU_MOE_AMD)
        aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/operators/moe_kernel/mat_kernel/aocl_kernel SOURCE_DIR7_KERNEL)
        add_compile_definitions(USE_MOE_KERNEL_AMD=1)
    elseif(NOT HOST_IS_X86 AND KTRANSFORMERS_CPU_USE_KML)
        aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/operators/moe_kernel/mat_kernel/kml_kernel SOURCE_DIR7_KERNEL)
    endif()
    list(APPEND SOURCE_DIR7 ${SOURCE_DIR7_KERNEL})
    if(NOT KTRANSFORMERS_CPU_MLA)
        list(REMOVE_ITEM SOURCE_DIR7 ${CMAKE_CURRENT_SOURCE_DIR}/operators/moe_kernel/mla/)
    endif()
    add_compile_definitions(USE_MOE_KERNEL=1)
endif()
message(STATUS "SOURCE_DIR7: ${SOURCE_DIR7}")



set(ALL_SOURCES ${SOURCE_DIR1} ${SOURCE_DIR2} ${SOURCE_DIR3} ${SOURCE_DIR4} ${SOURCE_DIR5} ${SOURCE_DIR6} ${SOURCE_DIR7})

file(GLOB_RECURSE FMT_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
)
# Exclude third_party directory
list(FILTER FMT_SOURCES EXCLUDE REGEX "/third_party/")

## Locate a specific clang-format executable to avoid version drift
## Prefer newer versions first to support modern .clang-format keys
## You can override by passing -DCLANG_FORMAT_BIN=/full/path/to/clang-format
if(NOT DEFINED CLANG_FORMAT_BIN)
    set(_CF_HINTS
        $ENV{CONDA_PREFIX}/bin
        $ENV{MAMBA_ROOT_PREFIX}/envs/$ENV{CONDA_DEFAULT_ENV}/bin
        $ENV{VIRTUAL_ENV}/bin
        $ENV{HOME}/.local/bin
    )
    find_program(CLANG_FORMAT_BIN
        NAMES clang-format-20 clang-format-19 clang-format-18 clang-format-17 clang-format-16 clang-format-15 clang-format
        HINTS ${_CF_HINTS}
    )
endif()
if(NOT CLANG_FORMAT_BIN)
    message(WARNING "ONLY for developer: clang-format not found. Please install clang-format (>=18) or pass -DCLANG_FORMAT_BIN=/full/path and reconfigure.")
else()
    execute_process(
        COMMAND ${CLANG_FORMAT_BIN} --version
        OUTPUT_VARIABLE _CLANG_FORMAT_VER
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    # message(STATUS "CMake PATH: $ENV{PATH}")
    # Parse version string, e.g. "Ubuntu clang-format version 19.1.0" or "clang-format version 18.1.8"
    string(REGEX MATCH "version[ ]+([0-9]+(\\.[0-9]+)*)" _CF_VER_MATCH "${_CLANG_FORMAT_VER}")
    if(NOT _CF_VER_MATCH)
        message(WARNING "Failed to parse clang-format version from: ${_CLANG_FORMAT_VER}")
    endif()
    set(CLANG_FORMAT_VERSION "${CMAKE_MATCH_1}")
    message(STATUS "Using clang-format ${CLANG_FORMAT_VERSION} at ${CLANG_FORMAT_BIN}")
    if(CLANG_FORMAT_VERSION VERSION_LESS "18.0.0")
        message(WARNING "clang-format >=18.0.0 required (found ${CLANG_FORMAT_VERSION} at ${CLANG_FORMAT_BIN}).\n"
                            "Tip: Ensure your desired clang-format (e.g., conda's ${CONDA_PREFIX}/bin/clang-format) is earlier in PATH when running CMake,\n"
                            "or pass -DCLANG_FORMAT_BIN=/full/path/to/clang-format.")
    endif()
    add_custom_target(
        format
        COMMAND ${CLANG_FORMAT_BIN}
                -i
                -style=file
                -fallback-style=none
                ${FMT_SOURCES}
        COMMENT "Running clang-format on all source files"
    )

    # Optional: target to check formatting without modifying files (CI-friendly)
    add_custom_target(
        format-check
        COMMAND ${CLANG_FORMAT_BIN}
                -n --Werror
                -style=file
                -fallback-style=none
                ${FMT_SOURCES}
        COMMENT "Checking clang-format on all source files"
    )
endif()

include(FindPkgConfig)
if(PKG_CONFIG_FOUND)
    pkg_search_module(HWLOC REQUIRED IMPORTED_TARGET hwloc)
else(PKG_CONFIG_FOUND)
    message(FATAL_ERROR "FindHWLOC needs pkg-config program and PKG_CONFIG_PATH must contain the path to hwloc.pc file.")
endif(PKG_CONFIG_FOUND)


add_library(llamafile STATIC ${SOURCE_DIR4})


if(CPUINFER_ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    # Use THIN_LTO keyword only if supported compiler (Clang). GCC ignores it.
    pybind11_add_module(${PROJECT_NAME} MODULE THIN_LTO ${ALL_SOURCES})
    message(STATUS "LTO: enabled")
else()
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
    pybind11_add_module(${PROJECT_NAME} MODULE ${ALL_SOURCES})
    message(STATUS "LTO: disabled")
endif()

# If BLIS was detected earlier, apply its include directory and library to the
# created Python extension target. We only do this after the module target
# (${PROJECT_NAME}) has been created by pybind11_add_module().
if(DEFINED _KT_BLIS_INCLUDE_DIR AND DEFINED _KT_BLIS_LIBRARY)
    if(TARGET ${PROJECT_NAME})
        target_include_directories(${PROJECT_NAME} PRIVATE ${_KT_BLIS_INCLUDE_DIR})
        target_link_libraries(${PROJECT_NAME} PRIVATE ${_KT_BLIS_LIBRARY})
    else()
        message(WARNING "BLIS was detected earlier but ${PROJECT_NAME} target was not found when attempting to apply BLIS link/include settings.")
    endif()
endif()

# Ensure the module target also has correct RPATH when conda is active
if(TARGET ${PROJECT_NAME} AND DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        BUILD_RPATH "$ENV{CONDA_PREFIX}/lib"
        INSTALL_RPATH "$ENV{CONDA_PREFIX}/lib"
        SKIP_BUILD_RPATH OFF
    )
endif()
if(NOT HOST_IS_X86 AND KTRANSFORMERS_CPU_USE_KML)
    message(STATUS "KML CPU detected")

    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/operators/moe_kernel/mat_kernel/kml_kernel/prefillgemm)
    target_link_libraries(${PROJECT_NAME} PRIVATE prefillint8gemm)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/operators/moe_kernel/mat_kernel/kml_kernel/prefillgemm_int4)
    target_link_libraries(${PROJECT_NAME} PRIVATE prefillint4gemm)

    set(DECODE_GEMM_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/operators/moe_kernel/mat_kernel/kml_kernel/batch_gemm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/operators/moe_kernel/mat_kernel/kml_kernel/batch_gemm_kernels.cpp
    )
    add_library(decode_gemm SHARED ${DECODE_GEMM_SOURCES})
    target_link_libraries(${PROJECT_NAME} PRIVATE decode_gemm)  
    if(KTRANSFORMERS_CPU_MLA)
        target_link_libraries(${PROJECT_NAME} PRIVATE kml_rt)
    endif()
    target_compile_definitions(${PROJECT_NAME} PRIVATE CPU_USE_KML)
endif()
target_link_libraries(${PROJECT_NAME} PRIVATE llama PkgConfig::HWLOC OpenMP::OpenMP_CXX)
if(NOT HOST_IS_X86 AND KTRANSFORMERS_CPU_USE_KML)
    if(KTRANSFORMERS_CPU_DEBUG)
        # add_executable(convert-test ${CMAKE_CURRENT_SOURCE_DIR}/operators/kml/convert-test.cpp)
        # target_link_libraries(convert-test llama)
        file(GLOB KML_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/operators/kml/test/*.cpp")
        foreach(test_src ${KML_TEST_SOURCES})
            # 获取不带扩展名的文件名作为 target 名
            get_filename_component(test_name ${test_src} NAME_WE)
            add_executable(${test_name} ${test_src} ${CMAKE_CURRENT_SOURCE_DIR}/cpu_backend/shared_mem_buffer.cpp)
            if(KTRANSFORMERS_CPU_MLA)
                target_link_libraries(${test_name} llama OpenMP::OpenMP_CXX numa kml_rt)
            endif()
        endforeach()
    endif()
endif()



if(KTRANSFORMERS_USE_CUDA)
    target_link_libraries(${PROJECT_NAME} PRIVATE "${CUDAToolkit_LIBRARY_DIR}/libcudart.so")
endif()
if(KTRANSFORMERS_USE_ROCM)
    add_compile_definitions(USE_HIP=1)
    target_link_libraries(${PROJECT_NAME} PRIVATE "${ROCM_PATH}/lib/libamdhip64.so")
    message(STATUS "Building for HIP")
endif()
if(KTRANSFORMERS_USE_MUSA)
    target_link_libraries(${PROJECT_NAME} PRIVATE MUSA::musart)
endif()



find_library(NUMA_LIBRARY NAMES numa)
if(NUMA_LIBRARY)
    message(STATUS "NUMA library found: ${NUMA_LIBRARY} - enabling NUMA support")
    target_link_libraries(${PROJECT_NAME} PRIVATE ${NUMA_LIBRARY})
else()
    message(FATAL_ERROR "NUMA library not found, please install NUMA, sudo apt install libnuma-dev")
endif()

