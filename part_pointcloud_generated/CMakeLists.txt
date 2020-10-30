cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(part_pointcloud_generated)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (part_pointcloud_generated part_pointcloud_generated.cpp)
target_link_libraries (part_pointcloud_generated ${PCL_LIBRARIES})


