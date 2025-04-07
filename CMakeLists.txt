cmake_minimum_required(VERSION 3.10)

find_package(PythonInterp REQUIRED)

execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "
import torch
print(torch.compiled_with_cxx11_abi())
"
    OUTPUT_VARIABLE TORCH_CXX11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(TORCH_CXX11_ABI MATCHES "True|False")
    if(TORCH_CXX11_ABI)
        message(STATUS "PyTorch was compiled with CXX11 ABI: ON")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
    else()
        message(STATUS "PyTorch was compiled with CXX11 ABI: OFF")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    endif()
else()
    message(WARNING "Could not determine if PyTorch was compiled with CXX11 ABI. Defaulting to OFF.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()
