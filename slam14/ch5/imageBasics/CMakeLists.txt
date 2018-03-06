#添加C++ 11标准
set(CMAKE_CXX_FLAGS "-std=c++11")
#寻找opencv库
find_package(OpenCV 3.1 REQUIRED)
#添加头文件
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(imageBasics imageBasics.cpp)
#链接opencv库
target_link_libraries(imageBasics ${OpenCV_LIBS}) 