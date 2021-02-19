# Set the minimum version of CMake that can be used
# To find the cmake version run
# $ cmake --version
cmake_minimum_required(VERSION 3.13)

# Set CXX standard
set(CMAKE_CXX_STANDARD 14)

# Set the project name
project(mosaic)

# Create a sources variable with a link to all cpp files to compile
aux_source_directory(src SOURCES)

# Set output executable path
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# Add an executable
add_executable(mosaic ${SOURCES})
