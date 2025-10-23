#!/usr/bin/env python3
"""
merge_clouds.py
ROS 2 节点：接收 2 个点云话题，取各自第一帧（非空），
通过 TF2 转换到 base_link 后拼成一个大点云，保存为 ASCII PCD。
不使用 open3d，纯字符串写入。
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from rclpy.qos import qos_profile_sensor_data   # BEST_EFFORT
import tf2_ros
import numpy as np
import struct
import yaml
import os
import sys
from pathlib import Path

test = True

# ---------- 参数区 ----------
# 1. 话题名 -> frame_id 映射
TOPIC_MAP = {
    "/sensing/lidar/concatenated/pointcloud": "base_link",
}
TARGET_FRAME = "sensor_kit_base_link"      # 要变换到的目标坐标系
OUTPUT_PCD   = "/home/pix/code/calibration_ws/src/calibBoard/data/first_frame_raw.pcd"     # 保存文件名
# ----------------------------


class MergeCloudsNode(Node):
    def __init__(self):
        super().__init__("merge_clouds")

        # 获取当前文件所在目录
        current_dir = Path(__file__).parent
        # 2. 成员变量
        self.task_complete = False
        
        self.first_frames = {}
        self.subs         = {}

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.sensor_kit_data = self.load_yaml(current_dir / '../config/extrinsic_parameters/sensors_calibration.yaml')

        for topic in TOPIC_MAP:
            self.subs[topic] = self.create_subscription(
                PointCloud2, topic,
                lambda msg, t=topic: self.cloud_cb(msg, t),
                qos_profile_sensor_data
            )

        self.get_logger().info("等待两帧非空点云...")
        self.timer = self.create_timer(0.5, self.check_done)
    
    def load_yaml(self, file_path):
        try:
            with open(file_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.get_logger().error(f'Error loading {file_path}: {str(e)}')
            return {}

    # ---------------------------
    def cloud_cb(self, msg: PointCloud2, topic: str):
        if topic in self.first_frames:
            return
        if msg.width == 0 or msg.height == 0:
            self.get_logger().warn(f"{topic} 空点云，跳过")
            return
        self.get_logger().info(f"收到 {topic} 第一帧，{msg.width * msg.height} 点")
        self.first_frames[topic] = msg
        self.destroy_subscription(self.subs[topic])

    def check_done(self):
        if len(self.first_frames) != len(TOPIC_MAP):
            return
        self.timer.cancel()
        self.get_logger().info("开始 yaml 位姿拼接...")

        merged = []   # list of Nx4 (x,y,z,intensity)
        for topic, msg in self.first_frames.items():
            frame_id = TOPIC_MAP[topic]          # 子坐标系名称，如 lidar_fl_base_link
            try:
                pose = self.sensor_kit_data[frame_id][TARGET_FRAME]
                x, y, z = pose["x"], pose["y"], pose["z"]
                roll, pitch, yaw = pose["roll"], pose["pitch"], pose["yaw"]
                self.get_logger().info(f"使用 yaml 位姿：{TARGET_FRAME} -> {frame_id}  平移 [{x}, {y}, {z}]  欧拉角 (radian) [{roll}, {pitch}, {yaw}]")
            except KeyError:
                self.get_logger().error(f"yaml 中找不到 {TARGET_FRAME} -> {frame_id}")
                continue

            # 1. 欧拉角 -> 旋转矩阵（xyz 顺序）
            R = self.euler2mat_xyz(roll, pitch, yaw)

            # 2. 构造 4×4 齐次矩阵
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [x, y, z]
            T_inv = np.linalg.inv(T)

            # 3. 点云变换
            xyzi = self.pc2_to_xyzi(msg)  # Nx4
            if xyzi.size == 0:
                continue
            xyzi[:, :3] = self.transform_points_matrix(xyzi[:, :3], T_inv)
            merged.append(xyzi)

        if not merged:
            self.get_logger().error("无点云可拼接")
            rclpy.shutdown()
            return

        all_points = np.vstack(merged)
        self.save_pcd_ascii_xyzi(all_points, OUTPUT_PCD)
        self.get_logger().info(f"已保存 -> {OUTPUT_PCD}  共 {all_points.shape[0]} 点")
        self.task_complete = True
        sys.exit(0)
        rclpy.shutdown()

    # ---------- 纯 NumPy 实现 ----------
    def euler2mat_xyz(self, roll, pitch, yaw):
        """xyz 顺序欧拉角 -> 3×3 旋转矩阵"""
        cx, sx = np.cos(roll),  np.sin(roll)
        cy, sy = np.cos(pitch), np.sin(pitch)
        cz, sz = np.cos(yaw),   np.sin(yaw)
        Rx = np.array([[1,  0,   0],
                    [0, cx, -sx],
                    [0, sx,  cx]])
        Ry = np.array([[ cy, 0, sy],
                    [  0, 1,  0],
                    [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0],
                    [sz,  cz, 0],
                    [ 0,   0, 1]])
        return Rz @ Ry @ Rx          # 注意顺序：Rz*Ry*Rx

    def transform_points_matrix(self, pts, T):
        """pts: (N,3)  T: 4×4  返回 (N,3)"""
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])  # (N,4)
        return (T @ pts_h.T).T[:, :3]

    # ---------- 解码 ----------
    def pc2_to_xyzi(self, cloud: PointCloud2) -> np.ndarray:
        """
        只处理前 4 个字段为 x,y,z,intensity 且均为 float32 的情况
        返回 N×4 np.float32
        """
        # 计算偏移
        field_map = {f.name: f.offset for f in cloud.fields}
        x_off  = field_map['x']
        y_off  = field_map['y']
        z_off  = field_map['z']
        i_off  = field_map['intensity']
        step   = cloud.point_step

        N = cloud.width * cloud.height
        out = np.empty((N, 4), dtype=np.float32)
        idx = 0
        for v in range(cloud.height):
            row_start = v * cloud.row_step
            for u in range(cloud.width):
                off = row_start + u * step
                out[idx, 0] = struct.unpack('<f', cloud.data[off+x_off:off+x_off+4])[0]
                out[idx, 1] = struct.unpack('<f', cloud.data[off+y_off:off+y_off+4])[0]
                out[idx, 2] = struct.unpack('<f', cloud.data[off+z_off:off+z_off+4])[0]
                out[idx, 3] = struct.unpack('<f', cloud.data[off+i_off:off+i_off+4])[0]
                idx += 1
        return out

    # ---------- TF ----------
    def transform_points(self, pts: np.ndarray, tf) -> np.ndarray:
        """pts: (N,3)"""
        t = np.array([tf.transform.translation.x,
                      tf.transform.translation.y,
                      tf.transform.translation.z])
        q = tf.transform.rotation
        quat = [q.w, q.x, q.y, q.z]
        R = self.quat_to_rot(quat)
        return (R @ pts.T).T + t

    def quat_to_rot(self, q):
        w, x, y, z = q
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
        ])

    # ---------- 保存 ----------
    def save_pcd_ascii_xyzi(self, pts: np.ndarray, filename: str):
        """pts: (N,4) x y z intensity"""
        N = pts.shape[0]
        header = f"""# .PCD v0.7 - Point Cloud Data file format
        VERSION 0.7
        FIELDS x y z intensity
        SIZE 4 4 4 4
        TYPE F F F F
        COUNT 1 1 1 1
        WIDTH {N}
        HEIGHT 1
        VIEWPOINT 0 0 0 1 0 0 0
        POINTS {N}
        DATA ascii
        """
        with open(filename, 'w') as f:
            f.write(header)
            for p in pts:
                f.write(f"{p[0]} {p[1]} {p[2]} {p[3]}\n")


# ---------------------------
def main():
    rclpy.init()
    node = MergeCloudsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 确保程序退出
        if not node.task_complete:
            node.get_logger().info("程序正在退出...")
        node.destroy_node()
        rclpy.shutdown()
        # 强制退出程序
        sys.exit(0)

if __name__ == '__main__':
    main()