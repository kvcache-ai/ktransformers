# The oneAPI driver links its own runtime libraries implicitly. CMake omits
# implicit compiler directories from generated RPATHs, and the conda setup can
# additionally replace the target's install RPATH. Source-built SYCL extensions
# therefore need the selected toolchain's runtime directory recorded explicitly.
function(kt_add_sycl_runtime_rpath target)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        return()
    endif()

    set(_kt_runtime_dirs)
    foreach(_kt_library IN ITEMS libsycl.so libiomp5.so libsvml.so libimf.so libirng.so libintlc.so.5)
        execute_process(
            COMMAND "${CMAKE_CXX_COMPILER}" "-print-file-name=${_kt_library}"
            RESULT_VARIABLE _kt_query_result
            OUTPUT_VARIABLE _kt_library_path
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(_kt_query_result EQUAL 0 AND IS_ABSOLUTE "${_kt_library_path}" AND EXISTS "${_kt_library_path}")
            get_filename_component(_kt_library_path "${_kt_library_path}" REALPATH)
            get_filename_component(_kt_runtime_dir "${_kt_library_path}" DIRECTORY)
            list(APPEND _kt_runtime_dirs "${_kt_runtime_dir}")
        elseif(_kt_library STREQUAL "libsycl.so" OR _kt_library STREQUAL "libiomp5.so")
            message(FATAL_ERROR "SYCL compiler ${CMAKE_CXX_COMPILER} could not locate ${_kt_library}; initialize the matching oneAPI installation before building")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _kt_runtime_dirs)
    # Append after any conda target setup, preserving its existing search path.
    # Set both properties because setuptools packages the build output directly,
    # while BUILD_WITH_INSTALL_RPATH may already be enabled by the conda setup.
    set_property(TARGET ${target} APPEND PROPERTY BUILD_RPATH "${_kt_runtime_dirs}")
    set_property(TARGET ${target} APPEND PROPERTY INSTALL_RPATH "${_kt_runtime_dirs}")
    set_property(TARGET ${target} PROPERTY SKIP_BUILD_RPATH OFF)
    message(STATUS "SYCL runtime RPATH: ${_kt_runtime_dirs}")
endfunction()
