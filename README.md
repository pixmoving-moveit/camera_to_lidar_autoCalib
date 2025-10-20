# CalibBoard ROS2 节点

这是一个ROS2 Humble点云处理节点，用于同时进行多个相机到激光雷达的标定工作。

## 编译和运行

### 编译
```bash
cd /home/pix/code/calibration_ws
colcon build --packages-select calib_board
source install/setup.bash
```

### 运行
```bash
# 方式1: 直接运行节点
ros2 run calib_board calib_board_node

# 方式2: 使用launch文件
ros2 launch calib_board calib_board.launch.py
```
