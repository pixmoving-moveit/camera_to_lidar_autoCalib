#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import yaml
import os
import math
from pathlib import Path

class CalibrationTransformPublisher(Node):
    def __init__(self):
        super().__init__('calibration_transform_publisher')
        
        # 创建 TransformBroadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # 获取当前文件所在目录
        current_dir = Path(__file__).parent
        
        # 读取yaml文件
        self.sensor_kit_data = self.load_yaml(current_dir / '../config/extrinsic_parameters/sensor_kit_calibration.yaml')
        self.sensors_data = self.load_yaml(current_dir / '../config/extrinsic_parameters/sensors_calibration.yaml')

        # 创建定时器，10Hz发布频率
        self.timer = self.create_timer(0.1, self.broadcast_transforms)
        
        self.get_logger().info('Calibration transform publisher started')

    def load_yaml(self, file_path):
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.get_logger().error(f'Error loading {file_path}: {str(e)}')
            return {}

    def create_transform_stamped(self, parent_frame, child_frame, transform_data):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame
        
        # 设置平移
        t.transform.translation.x = float(transform_data.get('x', 0.0))
        t.transform.translation.y = float(transform_data.get('y', 0.0))
        t.transform.translation.z = float(transform_data.get('z', 0.0))
        
        # 获取欧拉角
        roll = float(transform_data.get('roll', 0.0))
        pitch = float(transform_data.get('pitch', 0.0))
        yaw = float(transform_data.get('yaw', 0.0))
        
        # 将欧拉角转换为四元数
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        t.transform.rotation.w = cr * cp * cy + sr * sp * sy
        t.transform.rotation.x = sr * cp * cy - cr * sp * sy
        t.transform.rotation.y = cr * sp * cy + sr * cp * sy
        t.transform.rotation.z = cr * cp * sy - sr * sp * cy
        
        return t

    def broadcast_transforms(self):
        transforms = []
        
        # 发布传感器组到base_link的变换
        if 'base_link' in self.sensors_data:
            for child_frame, transform_data in self.sensors_data['base_link'].items():
                transforms.append(self.create_transform_stamped('base_link', child_frame, transform_data))

        # 发布传感器组内部各个传感器的变换
        if self.sensor_kit_data:
            for child_frame, transform_data in self.sensor_kit_data.get('sensor_kit_base_link', {}).items():
                transforms.append(self.create_transform_stamped('sensor_kit_base_link', child_frame, transform_data))
        
        # 一次性发布所有变换
        self.tf_broadcaster.sendTransform(transforms)

def main():
    rclpy.init()
    node = CalibrationTransformPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
