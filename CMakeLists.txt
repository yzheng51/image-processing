# Set the minimum version of CMake that can be used
# To find the cmake version run
# $ cmake --version
cmake_minimum_required(VERSION 3.13)

# Set CXX standard
set(CMAKE_CXX_STANDARD 14)

# Set the project name
project(mosaic)

# OpenMP
find_package(OpenMP REQUIRED)

# CUDA
find_package(CUDA REQUIRED)

# Create a sources variable with a link to all cpp files to compile
aux_source_directory(src SOURCES)

# Set output executable path
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# Add an executable
cuda_add_executable(mosaic ${SOURCES})

if(OpenMP_CXX_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
