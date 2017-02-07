cmake_minimum_required(VERSION 2.8)
project(SSDDetection)
add_compile_options(-std=c++11)

find_package(OpenCV REQUIRED)
find_package( Boost REQUIRED COMPONENTS)
include_directories( ${Boost_INCLUDE_DIRS} )

SET(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
SET(BUILD_SHARED_LIBRARIES OFF)

add_executable(SSDDetection ssd_detection.cpp)
target_link_libraries(SSDDetection
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_SYSTEM_LIBRARY}
        )
