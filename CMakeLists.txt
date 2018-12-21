cmake_minimum_required(VERSION 2.8.9)
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_PREFIX_PATH deps/lib/cmake)
find_package(xtensor 0.19 REQUIRED)
find_package(xtl 0.5 REQUIRED)

project (miniflow)
add_executable(miniflow main.cpp)
include_directories(deps/include)