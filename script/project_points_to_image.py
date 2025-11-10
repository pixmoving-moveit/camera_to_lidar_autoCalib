#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增量改动：同时订阅点云和 Image，1 Hz 拿最新帧做投影，其余不变
"""
import cv2
import numpy as np
import yaml
import sys
import os
import struct
import tkinter as tk
from tkinter import ttk
import threading
import rclpy
from pathlib import Path
from rclpy.node         import Node
from rclpy.qos          import qos_profile_sensor_data
from sensor_msgs.msg    import PointCloud2, Image
from cv_bridge          import CvBridge
from scipy.spatial.transform import Rotation as R

# ---------- 以下全部沿用你原来的函数 ----------
# read_pcd_xyz_intensity / load_camera_intrinsics / load_camera_extrinsics
# project_pcd_to_image  全部保持不动，这里只贴改动部分

def load_camera_intrinsics(intrinsic_file):
    """加载相机内参"""
    with open(intrinsic_file, 'r') as f:
        data = yaml.safe_load(f)
    
    # 使用修正后的rectification_matrix作为内参矩阵（适用于去畸变图像）
    if 'rectification_matrix' in data and 'data' in data['rectification_matrix']:
        rectification_matrix_data = data['rectification_matrix']['data']
        K = np.array(rectification_matrix_data).reshape(3, 3).astype(np.float32)
    else:
        # 如果没有rectification_matrix，回退到使用camera_matrix
        camera_matrix_data = data['camera_matrix']['data']
        K = np.array(camera_matrix_data).reshape(3, 3).astype(np.float32)
        print("警告: 未找到rectification_matrix，使用camera_matrix")
    
    # 畸变系数设为0，因为图像已经去畸变
    dist = np.zeros(5, dtype=np.float32)
    
    return K, dist

def homogeneous_to_xyzrpy(T):
    """从 4x4 齐次矩阵中提取 [x, y, z, roll, pitch, yaw]"""
    x, y, z = T[:3, 3]
    rot = R.from_matrix(T[:3, :3])
    roll, pitch, yaw = rot.as_euler('xyz', degrees=False)
    return [x, y, z, roll, pitch, yaw]

def extParams_Inverse(x, y, z, roll, pitch, yaw):
    def euler_to_rotation_matrix(roll, pitch, yaw):
        cr, sr = np.cos(roll),  np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw),   np.sin(yaw)

        R_x = np.array([[1,  0,   0],
                        [0, cr, -sr],
                        [0, sr,  cr]])

        R_y = np.array([[cp,  0, sp],
                        [0,   1,  0],
                        [-sp, 0, cp]])

        R_z = np.array([[cy, -sy, 0],
                        [sy,  cy, 0],
                        [0,    0, 1]])
        return R_z @ R_y @ R_x

    R = euler_to_rotation_matrix(roll, pitch, yaw)
    t = np.array([x, y, z], dtype=np.float64).reshape(3, 1)

    # 组装 4×4 齐次矩阵
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3]  = t.squeeze()

    # 求逆
    T_inv = np.linalg.inv(T)

    xyzrpy_inv = homogeneous_to_xyzrpy(T_inv)
    # print(f"相机 {camera_id} 外参逆: x={xyzrpy_inv[0]:.4f}, y={xyzrpy_inv[1]:.4f}, z={xyzrpy_inv[2]:.4f}, roll={xyzrpy_inv[3]:.4f}, pitch={xyzrpy_inv[4]:.4f}, yaw={xyzrpy_inv[5]:.4f}")

    # 提取逆后的 R 和 t
    R_inv = T_inv[:3, :3]
    t_inv = T_inv[:3, 3]

    return R_inv, t_inv


def load_camera_extrinsics(extrinsic_file, camera_id):
    """加载相机外参"""
    with open(extrinsic_file, 'r') as f:
        data = yaml.safe_load(f)
    
    # 获取相机外参
    camera_key = f'camera{camera_id}/camera_link'
    if camera_key not in data['sensor_kit_base_link']:
        raise ValueError(f"Camera {camera_id} not found in extrinsic parameters")
    
    extrinsics = data['sensor_kit_base_link'][camera_key]
    
    # 提取平移和旋转
    x, y, z = extrinsics['x'], extrinsics['y'], extrinsics['z']
    roll, pitch, yaw = extrinsics['roll'], extrinsics['pitch'], extrinsics['yaw']
    
    print(f"相机 {camera_id} 外参: x={x:.4f}, y={y:.4f}, z={z:.4f}, roll={roll:.4f}, pitch={pitch:.4f}, yaw={yaw:.4f}")

    return x, y, z, roll, pitch, yaw


def write_camera_extrinsics(camera_id, x, y, z, roll, pitch, yaw):
    proj_root = Path(__file__).resolve().parent.parent
    extrin_file = proj_root / 'config' / 'extrinsic_parameters' / 'sensor_kit_calibration.yaml'

    # 1. 读（顺序保留）
    with extrin_file.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # 2. 就地改值，不重建 dict
    camera_key = f'camera{camera_id}/camera_link'
    node = data['sensor_kit_base_link'][camera_key]   # 先拿到子 dict
    node['x'] = x
    node['y'] = y
    node['z'] = z
    node['roll'] = roll
    node['pitch'] = pitch
    node['yaw'] = yaw

    # 3. 写回（sort_keys=False 保序）
    with extrin_file.open('w', encoding='utf-8') as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True, width=2**20)

def project_pcd_to_image(cam_id_, pcd_points, pcd_intensity, R, t, K, dist, img):
    """将点云投影到图像上"""
    if len(pcd_points) == 0:
        print("点云数据为空")
        return img
    
    # 应用外参变换
    pcd_transformed = pcd_points @ R.T + t
    
    # 过滤掉相机后方的点（z > 0表示在相机前方）
    valid_mask = pcd_transformed[:, 2] > 0
    pcd_filtered = pcd_transformed[valid_mask]
    intensity_filtered = pcd_intensity[valid_mask]
    
    if len(pcd_filtered) == 0:
        print("没有有效的点云数据在相机前方")
        return img
    
    # 投影到图像平面
    pcd_reshaped = pcd_filtered.reshape(-1, 1, 3)
    rvec = np.zeros((3, 1), dtype=np.float32)  # 已经通过外参R变换，这里用单位旋转
    tvec = np.zeros((3, 1), dtype=np.float32)  # 已经通过外参t变换，这里用零平移
    
    img_points, _ = cv2.projectPoints(pcd_reshaped, rvec, tvec, K, dist)
    img_points = img_points.reshape(-1, 2)
    
    # 过滤在图像范围内的点
    h, w = img.shape[:2]
    valid_points = (
        (img_points[:, 0] >= 0) & (img_points[:, 0] < w) &
        (img_points[:, 1] >= 0) & (img_points[:, 1] < h)
    )
    
    img_points_valid = img_points[valid_points]
    intensity_valid = intensity_filtered[valid_points]
    
    # intensity归一化到0-255
    if len(intensity_valid) > 0:
        intensity_range = np.ptp(intensity_valid)
        if intensity_range > 0:
            norm_int = np.clip(255 * (intensity_valid - np.min(intensity_valid)) / intensity_range, 0, 255).astype(np.uint8)
        else:
            norm_int = np.full(len(intensity_valid), 128, dtype=np.uint8)
    else:
        norm_int = np.array([], dtype=np.uint8)
    
    # 创建结果图像的副本
    result_img = img.copy()
    
    pen_width = 1
    if cam_id_ == 0:
        pen_width = 2
    
    # 伪彩色映射并绘制点
    if len(norm_int) > 0:
        color_map = cv2.applyColorMap(norm_int.reshape(-1, 1), cv2.COLORMAP_JET)
        for idx, pt in enumerate(img_points_valid):
            color = tuple(int(c) for c in color_map[idx, 0])  # BGR
            cv2.circle(result_img, (int(pt[0]), int(pt[1])), pen_width, color, -1)
    
    print(f"成功投影了 {len(img_points_valid)} 个点到图像上")
    return result_img


# ====================== ROS 2 节点 ======================
class FusionNode(Node):
    def __init__(self, camera_id: int):
        super().__init__(f'fusion_camera{camera_id}')
        self.cam_id  = camera_id
        self.bridge  = CvBridge()
        self.result = None

        self.x_ = 0.0
        self.y_ = 0.0
        self.z_ = 0.0
        self.roll_ = 0.0
        self.pitch_ = 0.0
        self.yaw_ = 0.0

        # 外参 yaml
        cfg_dir = Path(__file__).parent / '../config/extrinsic_parameters'
        self.sensor_kit_data = yaml.safe_load(
            (cfg_dir / 'sensor_kit_calibration.yaml').read_text())
        self.base_kit_data = yaml.safe_load(
            (cfg_dir / 'sensors_calibration.yaml').read_text())

        # 最新数据缓存
        self.latest_cloud = None      # np.ndarray  N×4
        self.latest_image = None      # np.ndarray  H×W×3

        # 订阅
        self.cloud_sub = self.create_subscription(
            PointCloud2,
            '/sensing/lidar/concatenated/pointcloud',
            self.cloud_cb,
            qos_profile_sensor_data)

        self.img_sub   = self.create_subscription(
            Image,
            f'/sensing/camera/camera{camera_id}/camera_image',
            self.img_cb,
            qos_profile_sensor_data)

        # 1 Hz 定时器
        self.timer     = self.create_timer(1.0, self.fuse_once)
        # 30 Hz 键盘监听定时器
        self.key_timer  = self.create_timer(0.033, self.keyboard_cb)

        # 相机参数
        self.K, self.dist = self.load_cam_param()
        self.R, self.t    = self.load_cam_extrin()

        self.get_logger().info(f'camera{camera_id} 节点启动，等待图像和点云...')
    
    def update_params(self,index, val):
        if index == 0:
            self.roll_ = self.roll_ + val
        elif index == 1:
            self.pitch_ = self.pitch_ + val
        elif index == 2:
            self.yaw_ = self.yaw_ + val
        print("roll:",self.roll_," pitch:",self.pitch_," yaw:",self.yaw_)
        self.R, self.t = extParams_Inverse(self.x_,self.y_,self.z_,self.roll_,self.pitch_,self.yaw_)

    # ----------- 参数读取 -----------
    def load_cam_param(self):
        proj_root = Path(__file__).parent.parent
        intrinsic_file = proj_root / 'config' / 'intrinsic_parameters' / f'camera{self.cam_id}_params.yaml'
        return load_camera_intrinsics(str(intrinsic_file))

    def load_cam_extrin(self):
        proj_root = Path(__file__).parent.parent
        extrin_file = proj_root / 'config' / 'extrinsic_parameters' / 'sensor_kit_calibration.yaml'
        self.x_,self.y_,self.z_,self.roll_,self.pitch_,self.yaw_ = load_camera_extrinsics(str(extrin_file), self.cam_id)
        return extParams_Inverse(self.x_,self.y_,self.z_,self.roll_,self.pitch_,self.yaw_)

    # ----------- 回调 -----------
    def cloud_cb(self, msg: PointCloud2):
        self.latest_cloud = self.transform_cloud_to_sensor_kit(msg)

    def img_cb(self, msg: Image):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    # ----------- 1 Hz 融合 -----------
    def fuse_once(self):
        if self.latest_cloud is None:
            self.get_logger().info('等待数据中（缺少点云），跳过本次...')
            return
        if self.latest_image is None:
            self.get_logger().info('等待数据中（缺少图像），跳过本次...')
            return

        pts  = self.latest_cloud[:, :3]
        ints = self.latest_cloud[:, 3]
        img  = self.latest_image.copy()

        self.result = project_pcd_to_image(self.cam_id, pts, ints, self.R, self.t, self.K, self.dist, img)
        
        self.result = cv2.resize(self.result, (1920, 1080))

        # 保存 / 显示

        cv2.imshow(f'Camera{self.cam_id} 点云投影', self.result)
        cv2.waitKey(1)   # 1 ms 让 imshow 刷新
    
    # ----------- 键盘回调 -----------
    def keyboard_cb(self):
        key = cv2.waitKey(1) & 0xFF
        if key == 27:                      # ESC
            self.get_logger().info('ESC  pressed，退出节点')
            cv2.destroyAllWindows()
            rclpy.shutdown()
        elif key == ord('s') or key == ord('S'):
            if self.current_result is not None:
                out_dir = Path(__file__).parent.parent / 'cache'
                out_dir.mkdir(exist_ok=True)
                out_file = str(out_dir / f'camera{self.cam_id}_projection_result.png')
                cv2.imwrite(out_file, self.result)
                self.get_logger().info(f'已保存投影结果 -> {out_file}')
            else:
                self.get_logger().warn('暂无投影结果，无法保存')

    # ====================== 以下全部抄自 merge_clouds.py ======================
    def transform_cloud_to_sensor_kit(self, cloud: PointCloud2):
        frame_id = 'base_link'
        target   = 'sensor_kit_base_link'
        pose     = self.base_kit_data[frame_id][target]
        x,y,z    = pose['x'], pose['y'], pose['z']
        roll,pitch,yaw = pose['roll'], pose['pitch'], pose['yaw']
        R = self.euler2mat_xyz(roll, pitch, yaw)
        T = np.eye(4)
        T[:3,:3] = R
        T[:3, 3] = [x,y,z]
        T_inv = np.linalg.inv(T)
        xyzi = self.pc2_to_xyzi(cloud)
        xyzi[:,:3] = (T_inv @ np.hstack([xyzi[:,:3], np.ones((xyzi.shape[0],1))]).T).T[:,:3]
        return xyzi

    def euler2mat_xyz(self, roll, pitch, yaw):
        cx,sx = np.cos(roll),  np.sin(roll)
        cy,sy = np.cos(pitch), np.sin(pitch)
        cz,sz = np.cos(yaw),   np.sin(yaw)
        Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
        Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
        Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
        return Rz @ Ry @ Rx

    def pc2_to_xyzi(self, cloud: PointCloud2):
        fld = {f.name:f.offset for f in cloud.fields}
        N = cloud.width * cloud.height
        out = np.empty((N,4), np.float32)
        for i in range(N):
            off = i * cloud.point_step
            out[i,0] = struct.unpack('<f', cloud.data[fld['x']+off: fld['x']+off+4])[0]
            out[i,1] = struct.unpack('<f', cloud.data[fld['y']+off: fld['y']+off+4])[0]
            out[i,2] = struct.unpack('<f', cloud.data[fld['z']+off: fld['z']+off+4])[0]
            out[i,3] = struct.unpack('<f', cloud.data[fld['intensity']+off: fld['intensity']+off+4])[0]
        return out


# 定义一个简单的 GUI 类，用于输入参数
class GuiApp(threading.Thread):
    def __init__(self, node: FusionNode):
        super().__init__(daemon=True)
        self.node = node
        self.entries = []          # 保存三个输入框引用
        self.root = None

    def run(self):
        """线程入口：启动 Tkinter 主循环"""
        self.root = tk.Tk()
        self.root.title('ROS2 参数面板')
        # 标题
        ttk.Label(self.root, text='参数设置', font=('Arial', 16)).grid(
            row=0, column=0, columnspan=2, pady=6)

        # 三行标签+输入框
        for idx, label in enumerate(('Roll', 'Pitch', 'Yaw'), 1):
            ttk.Label(self.root, text=f'{label}:').grid(
                row=idx, column=0, padx=6, pady=6, sticky='e')
            e = ttk.Entry(self.root, width=10)
            e.insert(0, '0')
            e.grid(row=idx, column=1, padx=6, pady=6)
            self.entries.append(e)

        # 三个按钮
        btn_frame = ttk.Frame(self.root)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=10)
        ttk.Button(btn_frame, text='设置 Roll',
                   command=lambda: self._on_btn(0)).pack(side='left', padx=4)
        ttk.Button(btn_frame, text='设置 Pitch',
                   command=lambda: self._on_btn(1)).pack(side='left', padx=4)
        ttk.Button(btn_frame, text='设置 Yaw',
                   command=lambda: self._on_btn(2)).pack(side='left', padx=4)
        ttk.Button(btn_frame, text='配置参数',
                   command=lambda: self._on_btn(3)).pack(side='left', padx=4)

        self.root.mainloop()

    def _on_btn(self, idx: int):
        """按钮统一回调：取出对应输入框数字，更新节点参数"""
        if idx == 3:
            self.node.get_logger().info(f'配置参数：x={self.node.x_:.5f}, y={self.node.y_:.5f}, z={self.node.z_:.5f}, roll={self.node.roll_:.5f}, pitch={self.node.pitch_:.5f}, yaw={self.node.yaw_:.5f}')
            write_camera_extrinsics(self.node.cam_id, self.node.x_, self.node.y_, self.node.z_, self.node.roll_, self.node.pitch_, self.node.yaw_)
            return

        val = 0.0
        try:
            val = float(self.entries[idx].get())
        except ValueError:
            self.node.get_logger().warn(f'输入框 {idx} 不是合法数字！')
            return
        # 组装三个当前值
        # vals = [float(e.get()) for e in self.entries]
        # 直接调节点函数即可（线程安全，纯赋值）
        print("idx:",idx," val:",val)
        self.node.update_params(idx, val)

# ====================== main ======================
def main():
    if len(sys.argv) != 2:
        print('用法: python project_pcd_to_image.py <camera_id>   # 0-6')
        sys.exit(1)
    cam_id = int(sys.argv[1])
    if cam_id < 0 or cam_id > 6:
        print('camera_id 必须在 0-6 之间')
        sys.exit(1)

    rclpy.init()
    node = FusionNode(cam_id)

    gui = GuiApp(node)
    gui.start()          # 启动 GUI 线程

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
