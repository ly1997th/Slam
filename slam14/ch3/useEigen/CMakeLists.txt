cmake_minimum_required(VERSION 2.8)
project(useEigen)
set(CMAKE_BUILD_TYPE  "Release")
set( CMAKE_CXX_FLAG "_03")
include_directories("/usr/include/eigen3")
add_executable(useEigen eigenMatrix.cpp)


