#添加C++ 11标准
set(CMAKE_CXX_FLAGS "-std=c++11")
#寻找opencv库
find_package(OpenCV REQUIRED)
#添加头文件
include_directories(${OpenCV_INCLUDE_DIRS})
add_EXECUTABLE(feature_extraction feature_extraction.cpp)
#链接opencv库
target_link_libraries(feature_extraction  ${OpenCV_LIBS})
