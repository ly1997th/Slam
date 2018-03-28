cmake_minimum_required(VERSION 2.8)
project(pose_estimation_3d3d)

set(CMAKE_CXX_FLAGS "-std=c++11 -O3")
set(CMAKE_BUILD_TYPES "Release")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories("/usr/include/eigen3")

add_executable(pose_estimation_3d3d pose_estimation_3d3d.cpp)
target_link_libraries(pose_estimation_3d3d ${OpenCV_LIBS})
