cmake_minimum_required(VERSION 3.15...3.26)
project(ext)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYTHON "ON")
set(OSQP_BUILD_UNITTESTS "OFF")
set(OSQP_USE_LONG "OFF")
set(OSQP_CUSTOM_PRINTING "${CMAKE_CURRENT_SOURCE_DIR}/cmake/printing.h")
set(OSQP_CUSTOM_MEMORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake/memory.h")
set(OSQP_CODEGEN_INSTALL_DIR "codegen/codegen_src" CACHE PATH "" FORCE)

if(APPLE)
    message(STATUS "Building for Apple arches: ${CMAKE_OSX_ARCHITECTURES}")
endif()

include(FetchContent)

# 03/05/24 - Use modern python discovery
set(PYBIND11_FINDPYTHON "ON")

find_package(pybind11 CONFIG REQUIRED)

# 03/05/24 - Workaround because OSQP CMakeLists.txt is using old variable names
set(PYTHON_FOUND "ON")
set(PYTHON_INCLUDE_DIRS ${Python_INCLUDE_DIRS})

message(STATUS "Fetching/configuring OSQP")
list(APPEND CMAKE_MESSAGE_INDENT "  ")
FetchContent_Declare(
  osqp
  GIT_REPOSITORY https://github.com/osqp/osqp.git
  GIT_TAG v1.0.0
)
list(POP_BACK CMAKE_MESSAGE_INDENT)
FetchContent_MakeAvailable(osqp)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/bindings.cpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/src/bindings.cpp)
pybind11_add_module(${OSQP_EXT_MODULE_NAME} ${CMAKE_CURRENT_BINARY_DIR}/src/bindings.cpp)
install(TARGETS ${OSQP_EXT_MODULE_NAME} DESTINATION . COMPONENT python)

# Free-threaded (no-GIL) Python on Windows ships only pythonXYt.lib. The MSVC
# auto-link pragma in pyconfig.h selects pythonXY.lib (no "t") unless
# Py_GIL_DISABLED is defined at compile time, producing a spurious LNK1104 for
# the non-existent pythonXY.lib when building cp3XXt wheels. Mirror what
# setuptools does: ask the interpreter and define the macro ourselves so the
# correct import library is linked. No-op on non-free-threaded interpreters.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c
            "import sysconfig;print(1 if sysconfig.get_config_var('Py_GIL_DISABLED') else 0)"
    OUTPUT_VARIABLE OSQP_PY_GIL_DISABLED
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(OSQP_PY_GIL_DISABLED STREQUAL "1")
    message(STATUS "Free-threaded Python detected; defining Py_GIL_DISABLED")
    target_compile_definitions(${OSQP_EXT_MODULE_NAME} PRIVATE Py_GIL_DISABLED=1)
endif()

# TODO: We shouldn't have to do this once the interfaces are set up correctly
if(${OSQP_ALGEBRA_BACKEND} STREQUAL "builtin")
    target_link_libraries(ext_builtin PUBLIC pybind11::module osqpstatic)
elseif(${OSQP_ALGEBRA_BACKEND} STREQUAL "mkl")
    if(APPLE)
        target_link_libraries(osqp_mkl PUBLIC pybind11::module osqpstatic)
    else()
        target_link_libraries(osqp_mkl PUBLIC pybind11::module osqpstatic $<LINK_ONLY:MKL::MKL>)
    endif()
elseif(${OSQP_ALGEBRA_BACKEND} STREQUAL "cuda")
    enable_language(CUDA)
    find_package(CUDA)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    target_link_directories(osqp_cuda PUBLIC ${CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES})
    target_link_libraries(osqp_cuda PUBLIC pybind11::module osqpstatic cublas cusparse)
endif()
