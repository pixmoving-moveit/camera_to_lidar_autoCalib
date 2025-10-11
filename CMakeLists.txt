cmake_minimum_required(VERSION 3.8)
project(calib_board)

# Set CMake policy for find_package to use <PackageName>_ROOT variables
cmake_policy(SET CMP0074 NEW)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/lib/onnxruntime-linux-x64")
include_directories(SYSTEM ${ONNXRUNTIME_ROOT}/include)

# Include directories
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Create executable
add_executable(calib_board_node 
  src/main.cpp
  src/rclcomm.cpp
  src/extraboard.cpp
  src/calibrate.cpp
  src/detaruco.cpp
  src/cornersmatch.cpp
)

# Add include directories
target_include_directories(calib_board_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Specify libraries to link
ament_target_dependencies(calib_board_node
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  visualization_msgs
  pcl_ros
  pcl_conversions
  yaml-cpp
  tf2
  tf2_ros
)

target_link_libraries(calib_board_node 
  ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
  ${PCL_LIBRARIES} ${OpenCV_LIBS} yaml-cpp)

# Install targets
install(TARGETS calib_board_node
  DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch)

# Install config files with force overwrite
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING PATTERN "*.yaml"
  PATTERN "*~" EXCLUDE)

install(DIRECTORY ${ONNXRUNTIME_ROOT}/lib/
  DESTINATION lib
  FILES_MATCHING PATTERN "*.so*"
)

# Add custom target to ensure config files are always updated
add_custom_target(update_config_files ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/config
    ${CMAKE_CURRENT_BINARY_DIR}/config
  COMMENT "Updating config files"
)

# Make sure the executable depends on config update
add_dependencies(calib_board_node update_config_files)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
