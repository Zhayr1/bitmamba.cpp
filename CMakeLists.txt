cmake_minimum_required(VERSION 3.10)
project(BitMamba2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fopenmp")

# Include directories
include_directories(include)
include_directories(src) # For tokenizer.h

# Core library
add_library(bitmamba
    src/basic_utils.cpp
    src/quantization.cpp
    src/kernels.cpp
    src/block.cpp
    src/model.cpp
)

# CLI Executable
add_executable(bitmamba_cli
    examples/main.cpp
)

target_link_libraries(bitmamba_cli bitmamba)
