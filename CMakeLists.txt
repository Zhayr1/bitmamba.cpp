cmake_minimum_required(VERSION 3.10)

# Project name
project(BitMamba2 VERSION 1.0 LANGUAGES CXX)

# Use C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP library (Vital for multi-core performance)
find_package(OpenMP REQUIRED)

# Aggressive optimizations depending on compiler
if(MSVC)
    # Flags for Windows (Visual Studio)
    add_compile_options(/O2 /arch:AVX2)
else()
    # Flags for GCC/Clang (Linux/Mac)
    # -O3: Max optimization
    # -march=native: Use CPU specific instructions (AVX2, FMA)
    # -ffast-math: Small loss of precision for speed (common in AI)
    add_compile_options(-O3 -march=native -ffast-math -Wall)
endif()

# Define executable
add_executable(bitmamba bitmamba.cpp)

# Link OpenMP to executable
if(OpenMP_CXX_FOUND)
    target_link_libraries(bitmamba PUBLIC OpenMP::OpenMP_CXX)
endif()

message(STATUS "Build config: Release mode with AVX2 & OpenMP enabled.")
