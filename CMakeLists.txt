cmake_minimum_required(VERSION 3.24)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 89 = RTX 4000 Ada Generation (cluster GPU node)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

project(edge_mppi LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(hello_cuda cuda/tests/hello_cuda.cu)
target_include_directories(hello_cuda PRIVATE ${CMAKE_SOURCE_DIR}/cuda/include)

# STEP 3
find_package(CUDAToolkit REQUIRED)   # CUDA::cudart, for the .cpp files that call the runtime
find_package(yaml-cpp CONFIG REQUIRED)

# The model, the cost and (part F) the controller, shared by every executable
add_library(mppi_core STATIC
  cuda/src/config.cpp
  cuda/src/track.cpp
  cuda/src/mppi_kernel.cu
)
target_include_directories(mppi_core PUBLIC ${CMAKE_SOURCE_DIR}/cuda/include)
target_link_libraries(mppi_core PUBLIC CUDA::cudart yaml-cpp::yaml-cpp)
# Line numbers in device code, for compute-sanitizer and Nsight. No effect on speed.
target_compile_options(mppi_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# Checkpoint B: prints the config and the track
add_executable(check_config cuda/apps/check_config.cpp)
target_link_libraries(check_config PRIVATE mppi_core)
