cmake_minimum_required(VERSION 3.13)
project(foo_ext)

set(CMAKE_POSITION_INDEPENDENT_CODE ON) # -fPIC

# --------------
# osqpstatic
# --------------
file(
  GLOB
  OSQP_SOURCES
  src/*.c)

add_library(osqpstatic
  STATIC
    ${OSQP_SOURCES}
  )

target_include_directories(osqpstatic PUBLIC inc/public inc/private .)
target_include_directories(osqpstatic PRIVATE inc/private .)

# --------------
# foo
# --------------
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git)
FetchContent_MakeAvailable(pybind11)

file(
  GLOB
  FOO_SOURCES
  workspace.c
  workspace.h
  bindings.cpp)

pybind11_add_module(foo ${FOO_SOURCES})

target_link_libraries(foo PUBLIC pybind11::module osqpstatic)