cmake_minimum_required(VERSION 3.0.2)
project(perception)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
  tf2_ros
  tf2_geometry_msgs
  message_filters
  cv_bridge
  image_transport
  message_generation
)

## 메시지 파일 등록
add_message_files(
  FILES
    BoundingBox.msg
    Detection.msg
    DetectionArray.msg
    Association.msg
    AssociationArray.msg
)

catkin_python_setup()

## 메시지 생성
generate_messages(
  DEPENDENCIES
    std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    visualization_msgs
    tf2_ros
    tf2_geometry_msgs
    message_filters
    cv_bridge
    image_transport
    message_runtime
)

include_directories(${catkin_INCLUDE_DIRS})

## 파이썬 노드 설치
catkin_install_python(PROGRAMS
  src/nodes/association_node.py
  src/nodes/object_detector_node.py
  src/nodes/lane_detector_node.py
  src/nodes/speed_estimator_node.py
  src/nodes/tracker_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## 공유 리소스 설치(launch/config/rviz/models)
install(DIRECTORY launch config rviz models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  OPTIONAL
)
