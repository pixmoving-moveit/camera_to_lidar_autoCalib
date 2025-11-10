#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster           # ← 改动 1
from geometry_msgs.msg import TransformStamped
import yaml
import math
from pathlib import Path

class CalibrationStaticTransformPublisher(Node):
    def __init__(self):
        super().__init__('calibration_static_transform_publisher')

        # 静态广播器
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)  # ← 改动 2

        # 读取外参
        current_dir = Path(__file__).parent
        self.sensor_kit_data = self.load_yaml(
            current_dir / '../config/extrinsic_parameters/sensor_kit_calibration.yaml')
        self.sensors_data = self.load_yaml(
            current_dir / '../config/extrinsic_parameters/sensors_calibration.yaml')

        # 一次性发布所有静态变换
        self.publish_static_transforms()                                # ← 改动 3
        self.get_logger().info('Static calibration transforms published.')

    # ---------- 以下工具函数不变 ----------
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

        t.transform.translation.x = float(transform_data.get('x', 0.0))
        t.transform.translation.y = float(transform_data.get('y', 0.0))
        t.transform.translation.z = float(transform_data.get('z', 0.0))

        roll  = float(transform_data.get('roll',  0.0))
        pitch = float(transform_data.get('pitch', 0.0))
        yaw   = float(transform_data.get('yaw',   0.0))

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

    # ---------- 一次性发布 ----------
    def publish_static_transforms(self):
        transforms = []

        if 'base_link' in self.sensors_data:
            for child_frame, transform_data in self.sensors_data['base_link'].items():
                transforms.append(
                    self.create_transform_stamped('base_link', child_frame, transform_data))

        if self.sensor_kit_data:
            for child_frame, transform_data in \
                    self.sensor_kit_data.get('sensor_kit_base_link', {}).items():
                transforms.append(
                    self.create_transform_stamped('sensor_kit_base_link',
                                                  child_frame, transform_data))
        # 静态广播器一次性写入 /tf_static
        self.static_tf_broadcaster.sendTransform(transforms)


def main():
    rclpy.init()
    node = CalibrationStaticTransformPublisher()
    try:
        rclpy.spin(node)          # 保持节点存活即可
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()