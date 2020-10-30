cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(accuracy_assessment)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (accuracy_assessment accuracy_assessment.cpp)
target_link_libraries (accuracy_assessment ${PCL_LIBRARIES})


