cmake_minimum_required(VERSION 3.0.2)
project(4fn_basic_viewer)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  message_generation
  std_msgs
  sensor_msgs
)

find_package(Threads REQUIRED)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)
catkin_package(
  INCLUDE_DIRS include
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${pthread_INCLUDE_DIRS}
)

## simul_node build option
add_executable(
  radar_node
  src/main.cpp
  src/socket.cpp
  src/data_handler.cpp
)
add_dependencies(radar_node ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS})
target_link_libraries(radar_node ${catkin_LIBRARIES} stdc++fs)

add_definitions(-DROSCONSOLE_MIN_SEVERITY=ROSCONSOLE_SEVERITY_WARN)
