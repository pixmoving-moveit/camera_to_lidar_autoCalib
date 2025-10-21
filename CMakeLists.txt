cmake_minimum_required(VERSION 3.8)
project(calib_board)

# ---------- 1. 策略 ----------
cmake_policy(SET CMP0074 NEW)   # 支持 <PackageName>_ROOT

# ---------- 2. 编译选项 ----------
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---------- 3. 找 ROS 2 包 ----------
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

# ---------- 4. 第三方库路径 ----------
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/lib/onnxruntime-linux-x64")
set(FMT_ROOT         "${CMAKE_CURRENT_SOURCE_DIR}/lib/fmt")   # 你自己编译的 fmt
set(SPDLOG_ROOT      "${CMAKE_CURRENT_SOURCE_DIR}/lib/spdlog") # 头文件即可

include_directories(SYSTEM ${ONNXRUNTIME_ROOT}/include)

# ---------- 5. 强制使用外部 fmt ----------
add_compile_definitions(SPDLOG_FMT_EXTERNAL)

# ---------- 6. 找自己编译的 fmt ----------
list(PREPEND CMAKE_PREFIX_PATH "${FMT_ROOT}")
find_package(fmt REQUIRED CONFIG)   # 会找到 <fmt>/lib/cmake/fmt/fmtConfig.cmake

# ---------- 7. 创建 spdlog 头文件接口 ----------
add_library(spdlog_iface INTERFACE)
target_include_directories(spdlog_iface SYSTEM INTERFACE ${SPDLOG_ROOT})
target_compile_definitions(spdlog_iface INTERFACE SPDLOG_FMT_EXTERNAL)

# ---------- 8. 生成可执行文件 ----------
add_executable(calib_board_node
  src/main.cpp
  src/rclcomm.cpp
  src/extraboard.cpp
  src/calibrate.cpp
  src/detaruco.cpp
  src/cornersmatch.cpp)

# ---------- 9. 头文件目录 ----------
target_include_directories(calib_board_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# ---------- 10. ROS 2 ament 依赖 ----------
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

# ---------- 11. 链接库 ----------
target_link_libraries(calib_board_node
  spdlog_iface
  fmt::fmt                         # 自己的 libfmt.so.12
  ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  yaml-cpp
)

# ---------- 12. 安装可执行文件 ----------
install(TARGETS calib_board_node
  DESTINATION lib/${PROJECT_NAME})

# ---------- 13. 安装 launch ----------
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch)

# ---------- 14. 安装 config ----------
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING PATTERN "*.yaml"
  PATTERN "*.rviz")

# ---------- 15. 把 so 装到 lib 下（可选，方便运行时找） ----------
install(DIRECTORY ${ONNXRUNTIME_ROOT}/lib/ ${FMT_ROOT}/lib/
  DESTINATION lib
  FILES_MATCHING PATTERN "*.so*"
)

# ---------- 16. 保证运行时找到 so ----------
set_target_properties(calib_board_node PROPERTIES
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

# ---------- 17. 测试 ----------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# ---------- 18. 导出包 ----------
ament_package()