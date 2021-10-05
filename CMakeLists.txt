cmake_minimum_required(VERSION 3.13)
project(ext)
set(CMAKE_VERBOSE_MAKEFILE ON)

add_subdirectory(pybind11)

pybind11_add_module(ext src/bindings.cpp)
add_subdirectory(src/ext)
target_link_libraries(ext PRIVATE hello)