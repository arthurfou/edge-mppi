cmake_minimum_required(VERSION 3.24)

project(edge_mppi LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 86 = RTX 3060 (dev machine), 87 = Jetson Orin (deployment target)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 87)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(hello_cuda cuda/tests/hello_cuda.cu)
target_include_directories(hello_cuda PRIVATE ${CMAKE_SOURCE_DIR}/cuda/include)
