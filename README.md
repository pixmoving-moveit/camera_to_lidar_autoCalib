# CalibBoard ROS2 节点

这是一个ROS2 Humble点云处理节点，用于校准板相关的点云处理任务。

## 功能特性

- **ROS2通信**: 使用RclComm类处理ROS2通信
- **点云处理**: 使用ExtraBoard类进行PCL点云处理，提取标定板点云及其信息
- **PCD文件支持**: 支持加载和处理PCD格式的点云文件
- **点云滤波**: 支持体素滤波、统计异常值移除、直通滤波等

## 项目结构

```

```

## 编译和运行

### 编译
```bash
cd /home/pix/code/calibration_ws
colcon build --packages-select calib_board
source install/setup.bash
```

### 运行
```bash

# 方式2: 使用launch文件
ros2 launch calib_board calib_board.launch.py
```

## 主要类说明

### RclComm类
- 继承自rclcpp::Node
- 负责ROS2通信，包括点云发布
- 管理定时器和回调函数

### ExtraBoard类
- 负责PCD文件的加载和处理
- 提供多种点云滤波功能
- 支持PCL到ROS2消息的转换

## 话题

- `/pointcloud` (sensor_msgs/PointCloud2): 发布处理后的点云数据

## 依赖

- ROS2 Humble
- PCL (Point Cloud Library)
- sensor_msgs
- geometry_msgs
- std_msgs
- pcl_ros
- pcl_conversions
