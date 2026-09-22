cmake_minimum_required(VERSION 3.16)
project(polar_subcodes LANGUAGES CXX)
add_library(polar_subcodes SHARED cpp/candidate_block_neighborhood_kernel.cpp)
target_compile_features(polar_subcodes PRIVATE cxx_std_17)
set_target_properties(polar_subcodes PROPERTIES PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/build")
