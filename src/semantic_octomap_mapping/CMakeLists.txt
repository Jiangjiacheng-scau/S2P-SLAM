cmake_minimum_required(VERSION 3.0.2)
project(semantic_octomap_mapping)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  image_transport
  message_filters
  nav_msgs
  octomap_msgs
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
  std_msgs
  tf
  visualization_msgs
)

find_package(Eigen3 REQUIRED)
find_package(GTSAM REQUIRED)
find_package(octomap REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED COMPONENTS
  common
  filters
  kdtree
  sample_consensus
  search
  segmentation
)

catkin_package(
  CATKIN_DEPENDS
    cv_bridge
    geometry_msgs
    image_transport
    message_filters
    nav_msgs
    octomap_msgs
    pcl_conversions
    pcl_ros
    roscpp
    sensor_msgs
    std_msgs
    tf
    visualization_msgs
  DEPENDS Eigen3 GTSAM octomap OpenCV PCL
)

add_executable(semantic_octomap_node
  src/semantic_octomap_node.cpp
)

target_include_directories(semantic_octomap_node PRIVATE
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${GTSAM_INCLUDE_DIRS}
  ${OCTOMAP_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

target_compile_options(semantic_octomap_node PRIVATE
  -O3
  -Wall
  -Wextra
  -Wpedantic
)

target_link_libraries(semantic_octomap_node
  ${catkin_LIBRARIES}
  ${OCTOMAP_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  gtsam
)

add_dependencies(semantic_octomap_node ${catkin_EXPORTED_TARGETS})

install(TARGETS semantic_octomap_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
