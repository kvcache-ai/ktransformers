cmake_minimum_required(VERSION 3.14.1)

project(THREAD_TEST)


set(CMAKE_CXX_FLAGS ${CFLAGS_OPT})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")


message(STATUS "Current source dir: ${CMAKE_CURRENT_SOURCE_DIR}/..")
set(BLAS_PATH "")
message(STATUS "BLAS_PATH=${BLAS_PATH}")

if ("${BLAS_PATH}" STREQUAL "")

    set(BLAS_LIB "${CMAKE_CURRENT_SOURCE_DIR}/../libint8gemm.so")
    message(STATUS "BLAS_LIB=${BLAS_LIB}")

    # Add threading library to linker
    #find_package(Threads)

    add_executable(integer_tester integer_gemm.cpp)
    #target_include_directories(integer_tester PUBLIC "${BLAS_PATH}")
    #target_include_directories(integer_tester PUBLIC "${BLAS_BUILD_PATH}")
    target_link_libraries(integer_tester PRIVATE ${BLAS_LIB})
    set_property(TARGET integer_tester PROPERTY CXX_STANDARD 17)
    add_test(integer_tester ${CMAKE_CURRENT_BINARY_DIR}/integer_tester)
else ()
    message(FATAL_ERROR "Can not find int8_gemm path, pls set right path!")
endif ()