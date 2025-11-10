# 相机-激光雷达自动标定系统

这是一个基于ROS2 Humble的相机到激光雷达自动标定系统，能够实现多传感器数据融合的精确外部参数标定。

## 功能特性

- **相机-激光雷达自动标定**: 自动计算相机和激光雷达之间的外部参数
- **ArUco标记检测**: 支持基于ArUco标记的标定板识别和检测
- **角点匹配**: 使用匈牙利算法实现2D图像角点与3D点云角点的精确匹配
- **点云处理**: 支持多种点云滤波和处理功能
- **ROS2通信**: 完整的ROS2节点架构，支持多传感器数据获取和发布
- **配置系统**: YAML配置文件支持，方便参数调整
- **日志系统**: 集成spdlog，提供详细的日志记录

## 项目结构

```
camera_to_lidar_autoCalib/ 
├── CMakeLists.txt 
├── package.xml 
├── README.md 
├── include/ 
│ ├── base_struct.h # 基础数据结构定义 
│ ├── calibrate.h # 标定核心功能 
│ ├── cornersmatch.h # 角点匹配算法 
│ ├── cvui.h # UI相关功能 
│ ├── detaruco.h # ArUco标记检测 
│ ├── extraboard.h # 点云处理功能 
│ ├── rclcomm.h # ROS2通信接口 
│ └── spd_log.h # 日志系统 
├── src/ 
│ ├── calibrate.cpp # 标定实现 
│ ├── cornersmatch.cpp # 角点匹配实现 
│ ├── detaruco.cpp # ArUco检测实现 
│ ├── extraboard.cpp # 点云处理实现 
│ ├── main.cpp # 主程序入口 
│ └── rclcomm.cpp # ROS2通信实现 
├── launch/ 
│ └── calib_board.launch.py # 启动文件 
├── config/ 
│ └── calib_board_config.yaml # 配置文件 
├── data/ # 数据存储目录 
├── cache/ # 缓存目录 
└── lib/ # 第三方库目录
```

## 核心模块

### 1. 标定功能 (CalibFunc)

- 负责相机内参和外参的计算
- 支持多相机标定
- 实现标定结果的保存和加载

### 2. ArUco标记检测 (detaruco)

- 基于OpenCV的ArUco标记检测
- 支持YOLO模型加速检测
- 实现标定板的快速识别

### 3. 角点匹配 (CornersMatch)

- 使用匈牙利算法进行最优匹配
- 支持3D点云与2D图像的角点对应
- 提供精确的匹配结果

### 4. 点云处理 (ExtraBoard)

- 支持多种点云滤波方法：体素滤波、统计异常值移除、直通滤波
- 点云分割和聚类
- PCL与ROS2消息格式转换

### 5. ROS2通信 (RclComm)

- 继承自rclcpp::Node
- 管理点云发布和订阅
- 处理传感器数据同步

## 编译和运行

### 环境要求

- ROS2 Humble
- PCL (Point Cloud Library)
- OpenCV 4.x
- Eigen3
- yaml-cpp
- spdlog
- onnxruntime (用于YOLO模型)

### 编译

```bash
# 编译项目
cd ~/calibration_ws
colcon build --packages-select calib_board

# 加载环境变量
source install/setup.bash
```

### 运行

```bash
# 使用launch文件
ros2 launch calib_board calib_board.launch.py
```

## 配置说明

系统配置文件位于`config/calib_board_config.yaml`，主要配置项包括：

- 传感器参数配置
- 标定板参数设置
- 算法参数调整
- 数据路径配置

## 话题接口

### 订阅话题

- 相机图像话题
- 激光雷达点云话题

### 发布话题

- `/pointcloud` (sensor_msgs/PointCloud2): 处理后的点云数据
- 标定结果相关话题

## 使用流程

1. 准备标定板（带有ArUco标记的标定板）
2. 确保相机和激光雷达正确安装并能够获取数据
3. 启动标定程序
4. 按照提示移动标定板到不同位置进行数据采集
5. 系统自动计算标定参数
6. 保存标定结果用于后续使用

## 注意事项

- 标定过程中确保标定板可见且光照条件良好
- 建议采集多个不同位置和角度的数据以提高标定精度
- 标定结果将保存在配置的路径下，可用于后续数据融合应用