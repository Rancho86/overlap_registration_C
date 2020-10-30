cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(part_pointcloud_registration)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (part_registration part_registration.cpp)
target_link_libraries (part_registration ${PCL_LIBRARIES})


