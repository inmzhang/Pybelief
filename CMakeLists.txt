cmake_minimum_required(VERSION 3.0)

project(bptest)

aux_source_directory(. DIR_SRCS)

add_subdirectory(src)

add_executable(main main.cpp)

target_link_libraries(main BeliefPropagation)
