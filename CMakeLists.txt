cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 11)

project(node2ket)

add_compile_options(-lm -lgsl -lgslcblas -pthread -Ofast -march=native -Wall -funroll-loops -ffast-math -Wno-unused-result)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(Threads)

file(GLOB SRC_LIST src/libn2k/*.cpp)
add_library(tnelib STATIC ${SRC_LIST})

include_directories(libn2k)

add_executable(node2ket src/node2ket.cpp)
add_executable(EvalNR src/EvalNR.cpp)
target_link_libraries(node2ket libn2k ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(EvalNR libn2k ${CMAKE_THREAD_LIBS_INIT})
