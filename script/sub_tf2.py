#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformListener, Buffer
import math




class TF2Listener(Node):
    def __init__(self, parent_frame, child_frame):
        super().__init__('tf2_listener')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.parent_frame = parent_frame
        self.child_frame = child_frame
        self.timer = self.create_timer(0.5, self.timer_callback)

    def timer_callback(self):
        try:
            trans: TransformStamped = self.tf_buffer.lookup_transform(
                self.parent_frame,
                self.child_frame,
                rclpy.time.Time())
                # 计算欧拉角
            q = trans.transform.rotation
            # 四元数转欧拉角 (roll, pitch, yaw)
            sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
            cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
            roll = math.atan2(sinr_cosp, cosr_cosp)

            sinp = 2 * (q.w * q.y - q.z * q.x)
            if abs(sinp) >= 1:
                pitch = math.copysign(math.pi / 2, sinp)
            else:
                pitch = math.asin(sinp)

            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny_cosp, cosy_cosp)

            # 计算旋转矩阵
            qx, qy, qz, qw = q.x, q.y, q.z, q.w
            r00 = 1 - 2 * (qy * qy + qz * qz)
            r01 = 2 * (qx * qy - qz * qw)
            r02 = 2 * (qx * qz + qy * qw)
            r10 = 2 * (qx * qy + qz * qw)
            r11 = 1 - 2 * (qx * qx + qz * qz)
            r12 = 2 * (qy * qz - qx * qw)
            r20 = 2 * (qx * qz - qy * qw)
            r21 = 2 * (qy * qz + qx * qw)
            r22 = 1 - 2 * (qx * qx + qy * qy)
            rot_matrix = [r00, r01, r02, r10, r11, r12, r20, r21, r22]
            self.get_logger().info(f"Rotation matrix (3x3):"
                                   f" [{rot_matrix[0]:.3f}, {rot_matrix[1]:.3f}, {rot_matrix[2]:.3f},"
                                   f"{rot_matrix[3]:.3f}, {rot_matrix[4]:.3f}, {rot_matrix[5]:.3f},"
                                   f"{rot_matrix[6]:.3f}, {rot_matrix[7]:.3f}, {rot_matrix[8]:.3f}]")
            
            self.get_logger().info(
                f"Transform from {self.parent_frame} to {self.child_frame}:\n"
                f"Translation: [{trans.transform.translation.x:.3f}, {trans.transform.translation.y:.3f}, {trans.transform.translation.z:.3f}]\n"
                f"Rotation: x={trans.transform.rotation.x:.3f}, "
                f"y={trans.transform.rotation.y:.3f}, "
                f"z={trans.transform.rotation.z:.3f}, "
                f"w={trans.transform.rotation.w:.3f}\n"
                f"Euler angles (radian): roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}"
            )

        except Exception as e:
            self.get_logger().warn(f'Could not transform {self.parent_frame} to {self.child_frame}: {e}')

def main():
    rclpy.init()
    parent_frame = 'gnss_link'  # 修改为你的父坐标系
    child_frame = 'lidar_fr_base_link'  # 修改为你的子坐标系
    node = TF2Listener(parent_frame, child_frame)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
