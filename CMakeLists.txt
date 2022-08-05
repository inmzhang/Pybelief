cmake_minimum_required(VERSION 3.4...3.18)

project(bp)

if(NOT DEFINED CMAKE_MODULE_LINKER_FLAGS_MAINTAINER)
    set(
        CMAKE_MODULE_LINKER_FLAGS_MAINTAINER ""
        CACHE STRING ""
    )
endif()

file(GLOB SOURCES "src/pybelief/*.cpp")

add_subdirectory(extern/pybind11)
pybind11_add_module(_cpp_bp ${SOURCES})

target_include_directories(_cpp_bp PRIVATE "${PROJECT_BINARY_DIR}")

add_compile_options(-O3)

# set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection")
