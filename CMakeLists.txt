cmake_minimum_required(VERSION 3.17)
project(cuda_test CUDA CXX)

set(PROJECT_NAME cuda_test)

find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CXX_STANDARD 23)

set(CMAKE_CUDA_ARCHITECTURES 52)

file (GLOB_RECURSE SOURCES "src/*.cu" "src/*.cpp" "src/*.hpp" "src/*.h" "src/*.cuh")

add_executable(${PROJECT_NAME} ${SOURCES})

set_target_properties(
        cuda_test
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        LINKER_LANGUAGE CUDA)

target_include_directories(${PROJECT_NAME} PRIVATE vendor)
target_precompile_headers(${PROJECT_NAME} PRIVATE vendor/ywl_header/all.hpp)