cmake_minimum_required(VERSION 2.8)
project(useLK)

set(CMAKE_CXX_FLAGS "-std=c++11 -O3")
set(CMAKE_BUILD_TYPES "Release")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories("/usr/include/eigen3")

add_executable(useLK useLK.cpp)
target_link_libraries(useLK ${OpenCV_LIBS})
