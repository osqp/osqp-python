cmake_minimum_required(VERSION 3.13)
project(ext)
set(CMAKE_VERBOSE_MAKEFILE ON)

include(FetchContent)
message(STATUS "Fetching/configuring OSQP")
list(APPEND CMAKE_MESSAGE_INDENT "  ")

FetchContent_Declare(
  osqp
  GIT_REPOSITORY https://github.com/osqp/osqp.git
  GIT_TAG 5b77f133e81d248c989b787c426072c34ac13ccc
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/osqp3_sources)
FetchContent_GetProperties(osqp)

if(NOT osqp_POPULATED)
  FetchContent_Populate(osqp)
  add_subdirectory(${osqp_SOURCE_DIR} ${osqp_BINARY_DIR})
endif()

list(POP_BACK CMAKE_MESSAGE_INDENT)

#find_package(osqp REQUIRED)

add_subdirectory(pybind11)

pybind11_add_module(ext src/bindings.cpp)
add_subdirectory(src/ext)

#add_subdirectory(osqp2_sources)

target_include_directories(ext PRIVATE osqp2_sources/include)
target_link_libraries(ext PRIVATE hello osqp::osqpstatic)