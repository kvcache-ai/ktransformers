cmake_minimum_required(VERSION 3.14)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# can be compiled for SVE256 (32) or SVE512 (64)
set(SV_LENGTH 32) # in bytes

include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/standalone")
add_compile_options(-fPIC -fvisibility=hidden -fstack-protector-strong -march=armv8.3-a+sve+i8mm -O3)
# add_compile_options(-Wall -Wextra -Werror)

# sources are split into several groups: matmul kernels (sources are compiled multiple times),
#                                           packing kernels (sources are compiled multiple times),
#                                           sequential/parallel pipelines,
#                                           interface (sources are compiled multiple times)
set(INT_GEMM_KERNELS "")
set(INT_PACK_KERNELS "")
set(INT_GEMM_INTERFACE "")
set(INT_GEMM_PARALLEL "")
set(INT_GEMM_SEQ "")
set(BETA_KERNELS "")
set(POST_OPS_KERNELS "")
set(INT_SMALL_KERNELS "")
set(GEMM_DRIVERS "")

# Supported precisions are i/u for A and B matrices (4 combinations)
set(LHS_TYPES LHS_INT LHS_UINT)
set(RHS_TYPES RHS_INT RHS_UINT)

# compile matrix-multiplication kernels multiple times
set(INTEGER_MM_KERNELS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/integer_gemm_kernels.c")
set(M_SIZES 1 2 3 4)
set(N_SIZES 1 2 3 4)
foreach(M_SIZE ${M_SIZES})
  foreach(N_SIZE ${N_SIZES})
    foreach(LHS_TYPE ${LHS_TYPES})
      foreach(RHS_TYPE ${RHS_TYPES})
        add_library(integer_gemm_kernels_${LHS_TYPE}_${RHS_TYPE}_${M_SIZE}_${N_SIZE}_${SV_LENGTH} OBJECT ${INTEGER_MM_KERNELS_SRC})
        target_compile_options(integer_gemm_kernels_${LHS_TYPE}_${RHS_TYPE}_${M_SIZE}_${N_SIZE}_${SV_LENGTH} PUBLIC -DM_SIZE=${M_SIZE} 
          -DN_SIZE=${N_SIZE} -D${LHS_TYPE} -D${RHS_TYPE} -DCOMP_SV_LEN=${SV_LENGTH})
        target_include_directories(integer_gemm_kernels_${LHS_TYPE}_${RHS_TYPE}_${M_SIZE}_${N_SIZE}_${SV_LENGTH} PUBLIC
          "${CMAKE_SOURCE_DIR};${BLAS_INCLUDE_DIR};${DRIVER_INCLUDE_DIR};${CMAKE_BINARY_DIR}")
        list(APPEND INT_GEMM_KERNELS $<TARGET_OBJECTS:integer_gemm_kernels_${LHS_TYPE}_${RHS_TYPE}_${M_SIZE}_${N_SIZE}_${SV_LENGTH}>)
      endforeach()
    endforeach()
  endforeach()
endforeach()

# compile interface multiple times
set(INTEGER_GEMM_IFACE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/integer_gemm_interface.c")
foreach(LHS_TYPE ${LHS_TYPES})
  foreach(RHS_TYPE ${RHS_TYPES})
    add_library(integer_gemm_interface_${LHS_TYPE}_${RHS_TYPE}_${SV_LENGTH} OBJECT ${INTEGER_GEMM_IFACE_SRC})
    target_compile_options(integer_gemm_interface_${LHS_TYPE}_${RHS_TYPE}_${SV_LENGTH} PUBLIC -D${LHS_TYPE} -D${RHS_TYPE} -DCOMP_SV_LEN=${SV_LENGTH})
    target_include_directories(integer_gemm_interface_${LHS_TYPE}_${RHS_TYPE}_${SV_LENGTH} PUBLIC
      "${CMAKE_SOURCE_DIR};${BLAS_INCLUDE_DIR};${DRIVER_INCLUDE_DIR};${CMAKE_BINARY_DIR}")
    list(APPEND INT_GEMM_INTERFACE $<TARGET_OBJECTS:integer_gemm_interface_${LHS_TYPE}_${RHS_TYPE}_${SV_LENGTH}>)
  endforeach()
endforeach()

# compile threading layer
# set(INTEGER_GEMM_PAR_PIPE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/parallel_int_gemm_pipeline.c")
# add_library(integer_gemm_par_pipe_${SV_LENGTH} OBJECT ${INTEGER_GEMM_PAR_PIPE_SRC})
# target_compile_options(integer_gemm_par_pipe_${SV_LENGTH} PUBLIC -DCOMP_SV_LEN=${SV_LENGTH})
# target_include_directories(integer_gemm_par_pipe_${SV_LENGTH} PUBLIC "${CMAKE_SOURCE_DIR};${BLAS_INCLUDE_DIR};${DRIVER_INCLUDE_DIR};${CMAKE_BINARY_DIR}")
# list(APPEND INT_GEMM_PARALLEL $<TARGET_OBJECTS:integer_gemm_par_pipe_${SV_LENGTH}>)


# compile sequential layer
set(INTEGER_GEMMSEQ_PIPE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/sequential_int_gemm_pipeline.c")
add_library(integer_gemm_seq_pipe_${SV_LENGTH} OBJECT ${INTEGER_GEMMSEQ_PIPE_SRC})
target_compile_options(integer_gemm_seq_pipe_${SV_LENGTH} PUBLIC -DCOMP_SV_LEN=${SV_LENGTH})
target_include_directories(integer_gemm_seq_pipe_${SV_LENGTH} PUBLIC 
  "${CMAKE_SOURCE_DIR};${BLAS_INCLUDE_DIR};${DRIVER_INCLUDE_DIR};${CMAKE_BINARY_DIR}")
list(APPEND INT_GEMM_SEQ $<TARGET_OBJECTS:integer_gemm_seq_pipe_${SV_LENGTH}>)

# compile packingA kernels
set(INT_PACK_A_KERNELS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/integer_pack_a_kernels.c")
set(TRANSA_VALS TRANSA NOTRANSA)
foreach(LHS_TYPE ${LHS_TYPES})
  foreach(TRANSA_VAL ${TRANSA_VALS})
    add_library(integer_gemm_pack_a_${LHS_TYPE}_${TRANSA_VAL}_${SV_LENGTH} OBJECT ${INT_PACK_A_KERNELS_SRC})
    target_compile_options(integer_gemm_pack_a_${LHS_TYPE}_${TRANSA_VAL}_${SV_LENGTH} PUBLIC -DCOMP_SV_LEN=${SV_LENGTH} -D${LHS_TYPE} -D${TRANSA_VAL})
    target_include_directories(integer_gemm_pack_a_${LHS_TYPE}_${TRANSA_VAL}_${SV_LENGTH} PUBLIC "${CMAKE_SOURCE_DIR};${BLAS_INCLUDE_DIR};${DRIVER_INCLUDE_DIR};${CMAKE_BINARY_DIR}")
    list(APPEND INT_PACK_KERNELS $<TARGET_OBJECTS:integer_gemm_pack_a_${LHS_TYPE}_${TRANSA_VAL}_${SV_LENGTH}>)
  endforeach()
endforeach()

# compile packingB kernels
set(INT_PACK_B_KERNELS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/integer_pack_b_kernels.c")
set(TRANSB_VALS TRANSB NOTRANSB)
foreach(RHS_TYPE ${RHS_TYPES})
  foreach(TRANSB_VAL ${TRANSB_VALS})
    add_library(integer_gemm_pack_b_${RHS_TYPE}_${TRANSB_VAL}_${SV_LENGTH} OBJECT ${INT_PACK_B_KERNELS_SRC})
    target_compile_options(integer_gemm_pack_b_${RHS_TYPE}_${TRANSB_VAL}_${SV_LENGTH} PUBLIC -DCOMP_SV_LEN=${SV_LENGTH} -D${RHS_TYPE} -D${TRANSB_VAL})
    target_include_directories(integer_gemm_pack_b_${RHS_TYPE}_${TRANSB_VAL}_${SV_LENGTH} PUBLIC "${CMAKE_SOURCE_DIR};${BLAS_INCLUDE_DIR};${DRIVER_INCLUDE_DIR};${CMAKE_BINARY_DIR}")
    list(APPEND INT_PACK_KERNELS $<TARGET_OBJECTS:integer_gemm_pack_b_${RHS_TYPE}_${TRANSB_VAL}_${SV_LENGTH}>)
  endforeach()
endforeach()

# compile beta kernels
set(BETA_OPTS BETA_OPT BETA_NO_OPT)

foreach(B_TYPE ${BETA_OPTS})
  add_library(beta_kernels_${B_TYPE} OBJECT "${CMAKE_CURRENT_SOURCE_DIR}/integer_beta_kernels.c")
  target_compile_options(beta_kernels_${B_TYPE} PUBLIC -D${B_TYPE} -DCOMP_SV_LEN=${SV_LENGTH})
  target_include_directories(beta_kernels_${B_TYPE} PUBLIC "${CMAKE_SOURCE_DIR};${BLAS_INCLUDE_DIR};${DRIVER_INCLUDE_DIR};${CMAKE_BINARY_DIR}")
  list(APPEND BETA_KERNELS $<TARGET_OBJECTS:beta_kernels_${B_TYPE}>)
endforeach()

# compile int gemm drivers
foreach(B_TYPE ${BETA_OPTS})
  add_library(gemm_driver_${B_TYPE} OBJECT "${CMAKE_CURRENT_SOURCE_DIR}/integer_gemm_driver.c")
  target_compile_options(gemm_driver_${B_TYPE} PUBLIC -D${B_TYPE} -DCOMP_SV_LEN=${SV_LENGTH})
  target_include_directories(gemm_driver_${B_TYPE} PUBLIC "${CMAKE_SOURCE_DIR};${BLAS_INCLUDE_DIR};${DRIVER_INCLUDE_DIR};${CMAKE_BINARY_DIR}")
  list(APPEND GEMM_DRIVERS $<TARGET_OBJECTS:gemm_driver_${B_TYPE}>)
endforeach()

# compile post-ops kernels
foreach(B_TYPE ${BETA_OPTS})
  add_library(post_ops_kernels_${B_TYPE} OBJECT "${CMAKE_CURRENT_SOURCE_DIR}/integer_post_ops_kernels.c")
  target_compile_options(post_ops_kernels_${B_TYPE} PUBLIC -D${B_TYPE} -DCOMP_SV_LEN=${SV_LENGTH})
  target_include_directories(post_ops_kernels_${B_TYPE} PUBLIC "${CMAKE_SOURCE_DIR};${BLAS_INCLUDE_DIR};${DRIVER_INCLUDE_DIR};${CMAKE_BINARY_DIR}")
  list(APPEND POST_OPS_KERNELS $<TARGET_OBJECTS:post_ops_kernels_${B_TYPE}>)
endforeach()

# compile matrix-multiplication small kernels multiple times
set(SMALL_KERNELS_KERNELS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/integer_gemm_small_kernels.c")
set(TRANSA_VALS TRANSA NOTRANSA)
set(TRANSB_VALS TRANSB NOTRANSB)
set(OC_TYPES OC_FIX OC_COL OC_ROW)

foreach(LHS_TYPE ${LHS_TYPES})
  foreach(RHS_TYPE ${RHS_TYPES})
    foreach(TRANSA_VAL ${TRANSA_VALS})
      foreach(TRANSB_VAL ${TRANSB_VALS})
        foreach(OC_TYPE ${OC_TYPES})
          add_library(small_kernels_${LHS_TYPE}_${RHS_TYPE}_${TRANSA_VAL}_${TRANSB_VAL}_${OC_TYPE}${SV_LENGTH} OBJECT ${SMALL_KERNELS_KERNELS_SRC})
          target_compile_options(small_kernels_${LHS_TYPE}_${RHS_TYPE}_${TRANSA_VAL}_${TRANSB_VAL}_${OC_TYPE}${SV_LENGTH} PUBLIC -D${LHS_TYPE} -D${RHS_TYPE} -D${TRANSA_VAL} -D${TRANSB_VAL} -D${OC_TYPE} -DCOMP_SV_LEN=${SV_LENGTH})
          target_include_directories(small_kernels_${LHS_TYPE}_${RHS_TYPE}_${TRANSA_VAL}_${TRANSB_VAL}_${OC_TYPE}${SV_LENGTH} PUBLIC "${CMAKE_SOURCE_DIR};${BLAS_INCLUDE_DIR};${DRIVER_INCLUDE_DIR};${CMAKE_BINARY_DIR}")
          list(APPEND INT_SMALL_KERNELS $<TARGET_OBJECTS:small_kernels_${LHS_TYPE}_${RHS_TYPE}_${TRANSA_VAL}_${TRANSB_VAL}_${OC_TYPE}${SV_LENGTH}>)
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endforeach()

list(APPEND OBJ_FILES_STANDALONE_DIR ${INT_GEMM_KERNELS} ${INT_GEMM_INTERFACE} ${INT_GEMM_SEQ} ${INT_PACK_KERNELS} ${BETA_KERNELS} ${POST_OPS_KERNELS} ${INT_SMALL_KERNELS} ${GEMM_DRIVERS})
# set(OBJ_FILES_STANDALONE_DIR ${OBJ_FILES_STANDALONE_DIR} PARENT_SCOPE)
# all compiled object files are united into one object library
add_library(prefillint8gemm SHARED ${OBJ_FILES_STANDALONE_DIR})

set_target_properties(prefillint8gemm PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/prefillint8gemm
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/prefillint8gemm
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/prefillint8gemm
)
