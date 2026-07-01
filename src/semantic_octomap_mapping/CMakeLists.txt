cmake_minimum_required(VERSION 3.0.2)
project(semantic_octomap_mapping)

# ==========================================================
# 1. 编译设置与优化 (关键修改)
# ==========================================================
# 开启 C++17 (支持 std::shared_mutex)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# [新增] 开启 Release 模式和 O3 优化
# 如果没有这行，因子图优化可能会导致 CPU 占用率 100% 并卡顿
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O2)

# ==========================================================
# 2. OpenMP 并行计算支持
# ==========================================================
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP Found! Parallelization enabled.")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# ==========================================================
# 3. 查找依赖库
# ==========================================================
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs       
  visualization_msgs  
  pcl_ros
  pcl_conversions     
  octomap_ros
  octomap_msgs
  cv_bridge       
  image_transport 
  tf
  message_filters     
)

# 核心算法库
find_package(PCL REQUIRED COMPONENTS common io visualization segmentation filters sac sample_consensus kdtree search features)
find_package(Octomap REQUIRED)
find_package(OpenCV REQUIRED) 
find_package(GTSAM REQUIRED)
find_package(Eigen3 REQUIRED) # [建议] 显式查找 Eigen

# ==========================================================
# 4. 导出依赖
# ==========================================================
catkin_package(
  CATKIN_DEPENDS 
  roscpp std_msgs sensor_msgs nav_msgs geometry_msgs visualization_msgs 
  pcl_ros pcl_conversions octomap_ros octomap_msgs cv_bridge image_transport tf message_filters
)

# ==========================================================
# 5. 头文件路径
# ==========================================================
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OCTOMAP_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIRS} # [建议]
)

# ==========================================================
# 6. 编译可执行文件
# ==========================================================
add_executable(semantic_octomap_node src/semantic_octomap_node.cpp)

# ==========================================================
# 7. 链接库文件
# ==========================================================
target_link_libraries(semantic_octomap_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OCTOMAP_LIBRARIES}
  ${OpenCV_LIBRARIES}
  
  # GTSAM 依赖
  gtsam
  tbb                   
  boost_system          
  boost_serialization   

  # OpenMP 链接 (CMake 3.0+ 推荐方式)
  OpenMP::OpenMP_CXX
)

# ==========================================================
# 8. 依赖顺序
# ==========================================================
add_dependencies(semantic_octomap_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})