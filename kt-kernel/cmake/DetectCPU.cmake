# CPU Feature Detection for kt-kernel
# Detects CPU capabilities and sets appropriate compiler flags

function(detect_cpu_features)
    set(HAS_AVX2 OFF PARENT_SCOPE)
    set(HAS_AVX512F OFF PARENT_SCOPE)
    set(HAS_AVX512_VNNI OFF PARENT_SCOPE)
    set(HAS_AVX512_BF16 OFF PARENT_SCOPE)
    set(HAS_AVX512_VBMI OFF PARENT_SCOPE)
    set(HAS_AMX OFF PARENT_SCOPE)

    if(NOT EXISTS "/proc/cpuinfo")
        message(STATUS "CPU detection: /proc/cpuinfo not found, skipping auto-detection")
        return()
    endif()

    # Read CPU flags from /proc/cpuinfo
    file(READ "/proc/cpuinfo" CPUINFO_CONTENT)
    string(REGEX MATCH "flags[ \t]*:[ \t]*([^\n]*)" FLAGS_LINE "${CPUINFO_CONTENT}")
    if(NOT CMAKE_MATCH_1)
        message(STATUS "CPU detection: Could not parse CPU flags")
        return()
    endif()

    set(CPU_FLAGS "${CMAKE_MATCH_1}")
    string(REPLACE " " ";" CPU_FLAGS_LIST "${CPU_FLAGS}")

    # Check for each feature
    if("avx2" IN_LIST CPU_FLAGS_LIST)
        set(HAS_AVX2 ON PARENT_SCOPE)
    endif()

    if("avx512f" IN_LIST CPU_FLAGS_LIST)
        set(HAS_AVX512F ON PARENT_SCOPE)
    endif()

    if("avx512_vnni" IN_LIST CPU_FLAGS_LIST OR "avx512vnni" IN_LIST CPU_FLAGS_LIST)
        set(HAS_AVX512_VNNI ON PARENT_SCOPE)
    endif()

    if("avx512_bf16" IN_LIST CPU_FLAGS_LIST OR "avx512bf16" IN_LIST CPU_FLAGS_LIST)
        set(HAS_AVX512_BF16 ON PARENT_SCOPE)
    endif()

    if("avx512_vbmi" IN_LIST CPU_FLAGS_LIST OR "avx512vbmi" IN_LIST CPU_FLAGS_LIST)
        set(HAS_AVX512_VBMI ON PARENT_SCOPE)
    endif()

    # Check for AMX (need all three)
    set(AMX_COUNT 0)
    foreach(flag "amx_tile" "amx_int8" "amx_bf16")
        if("${flag}" IN_LIST CPU_FLAGS_LIST)
            math(EXPR AMX_COUNT "${AMX_COUNT} + 1")
        endif()
    endforeach()
    if(AMX_COUNT EQUAL 3)
        set(HAS_AMX ON PARENT_SCOPE)
    endif()

    # Get CPU model name for display
    string(REGEX MATCH "model name[ \t]*:[ \t]*([^\n]*)" MODEL_LINE "${CPUINFO_CONTENT}")
    if(CMAKE_MATCH_1)
        set(CPU_MODEL "${CMAKE_MATCH_1}" PARENT_SCOPE)
    endif()
endfunction()

# Main detection and configuration
message(STATUS "")
message(STATUS "========================================")
message(STATUS "CPU Feature Detection (CMake)")
message(STATUS "========================================")

# Check if variables were already set by install.sh/setup.py
set(FROM_INSTALL_SH OFF)
if(DEFINED LLAMA_AVX512_VNNI OR DEFINED LLAMA_AVX512_BF16 OR DEFINED LLAMA_AVX512_VBMI)
    set(FROM_INSTALL_SH ON)
    message(STATUS "Detected configuration from install.sh/setup.py")
    message(STATUS "  LLAMA_AVX512:      ${LLAMA_AVX512}")
    message(STATUS "  LLAMA_AVX512_VNNI: ${LLAMA_AVX512_VNNI}")
    message(STATUS "  LLAMA_AVX512_BF16: ${LLAMA_AVX512_BF16}")
    message(STATUS "  LLAMA_AVX512_VBMI: ${LLAMA_AVX512_VBMI}")
    message(STATUS "")
    message(STATUS "Skipping auto-detection (using install.sh settings)")
    message(STATUS "========================================")
    message(STATUS "")
    return()
endif()

# Detect CPU features (only if not set by install.sh)
detect_cpu_features()

if(CPU_MODEL)
    message(STATUS "CPU Model: ${CPU_MODEL}")
endif()

message(STATUS "")
message(STATUS "Detected features:")
message(STATUS "  AVX2:         ${HAS_AVX2}")
message(STATUS "  AVX512F:      ${HAS_AVX512F}")
message(STATUS "  AVX512_VNNI:  ${HAS_AVX512_VNNI}")
message(STATUS "  AVX512_BF16:  ${HAS_AVX512_BF16}")
message(STATUS "  AVX512_VBMI:  ${HAS_AVX512_VBMI}")
message(STATUS "  AMX:          ${HAS_AMX}")
message(STATUS "")

# Auto-enable features based on detection
# Only set if not already defined by user via -D flags
if(NOT DEFINED LLAMA_AVX2 AND HAS_AVX2)
    set(LLAMA_AVX2 ON CACHE BOOL "Enable AVX2" FORCE)
    message(STATUS "Auto-enabled: AVX2")
endif()

if(NOT DEFINED LLAMA_AVX512 AND HAS_AVX512F)
    set(LLAMA_AVX512 ON CACHE BOOL "Enable AVX512F" FORCE)
    message(STATUS "Auto-enabled: AVX512F")
endif()

if(NOT DEFINED LLAMA_AVX512_VNNI AND HAS_AVX512_VNNI)
    set(LLAMA_AVX512_VNNI ON CACHE BOOL "Enable AVX512_VNNI" FORCE)
    message(STATUS "Auto-enabled: AVX512_VNNI")
endif()

if(NOT DEFINED LLAMA_AVX512_BF16 AND HAS_AVX512_BF16)
    set(LLAMA_AVX512_BF16 ON CACHE BOOL "Enable AVX512_BF16" FORCE)
    message(STATUS "Auto-enabled: AVX512_BF16")
endif()

if(NOT DEFINED LLAMA_AVX512_VBMI AND HAS_AVX512_VBMI)
    set(LLAMA_AVX512_VBMI ON CACHE BOOL "Enable AVX512_VBMI" FORCE)
    message(STATUS "Auto-enabled: AVX512_VBMI")
endif()

if(NOT DEFINED KTRANSFORMERS_CPU_USE_AMX AND HAS_AMX)
    set(KTRANSFORMERS_CPU_USE_AMX ON CACHE BOOL "Enable AMX" FORCE)
    message(STATUS "Auto-enabled: AMX")
endif()

message(STATUS "")
message(STATUS "Note: You can override by passing -DLLAMA_AVX512_BF16=OFF etc.")
message(STATUS "Note: Or use install.sh with environment variables")
message(STATUS "========================================")
message(STATUS "")
