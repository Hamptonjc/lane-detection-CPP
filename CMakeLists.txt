cmake_minimum_required(VERSION "3.17.2")

set(CMAKE_CXX_STANDARD 14)
project(LaneDetection)

add_executable(LaneDetection main.cpp)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(LaneDetection ${OpenCV_LIBS})