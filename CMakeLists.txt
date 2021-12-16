cmake_minimum_required(VERSION 3.13)
project(ext)
set(CMAKE_VERBOSE_MAKEFILE ON)

include(FetchContent)
message(STATUS "Fetching/configuring OSQP")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
FetchContent_Declare(
  osqp
  GIT_REPOSITORY https://github.com/osqp/osqp.git
  GIT_TAG develop-1.0)

list(POP_BACK CMAKE_MESSAGE_INDENT)
FetchContent_MakeAvailable(osqp)

add_subdirectory(pybind11)

pybind11_add_module(ext src/bindings.cpp)
target_link_libraries(ext PUBLIC pybind11::module osqpstatic)
