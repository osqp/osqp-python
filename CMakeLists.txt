cmake_minimum_required(VERSION 3.13)
project(ext)

include(FetchContent)
message(STATUS "Fetching/configuring OSQP")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
FetchContent_Declare(
  osqp
  GIT_REPOSITORY https://github.com/osqp/osqp.git
  GIT_TAG develop-1.0)

list(POP_BACK CMAKE_MESSAGE_INDENT)
FetchContent_MakeAvailable(osqp)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git)
FetchContent_MakeAvailable(pybind11)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp.in
               ${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp)
pybind11_add_module(${OSQP_EXT_MODULE_NAME} src/bindings.cpp)

# TODO: We shouldn't have to do this once the interfaces are set up correctly
if(${ALGEBRA} STREQUAL "default")
    target_link_libraries(osqp_default PUBLIC pybind11::module osqpstatic)
elseif(${ALGEBRA} STREQUAL "mkl")
    target_link_libraries(osqp_mkl PUBLIC pybind11::module osqpstatic $<LINK_ONLY:MKL::MKL>)
elseif(${ALGEBRA} STREQUAL "cuda")
    enable_language(CUDA)
    find_package(CUDA)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_directories(osqp_cuda PUBLIC ${CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES})
    target_link_libraries(osqp_cuda PUBLIC pybind11::module osqpstatic cublas cusparse) 
endif()
