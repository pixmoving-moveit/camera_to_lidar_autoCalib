import cv2
import numpy as np
import yaml
import sys
import os
import struct
from scipy.spatial.transform import Rotation as R


def read_pcd_xyz_intensity(filename):
    """读取PCD文件，返回xyz和intensity（如有）"""
    points = []
    intensities = []
    header_lines = []
    
    with open(filename, 'rb') as f:
        while True:
            line = f.readline()
            if not line:
                break
            header_lines.append(line.decode('latin1'))
            if line.strip().startswith(b'DATA'):
                break
        
        data_line = [l for l in header_lines if l.strip().startswith('DATA')]
        if not data_line:
            print('PCD header缺少DATA行')
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        
        if 'ascii' in data_line[0]:
            for line in f:
                vals = line.decode('latin1').strip().split()
                if len(vals) >= 3:
                    try:
                        x, y, z = float(vals[0]), float(vals[1]), float(vals[2])
                        intensity = float(vals[3]) if len(vals) > 3 else 0.0
                        points.append([x, y, z])
                        intensities.append(intensity)
                    except ValueError:
                        continue
            return np.array(points, dtype=np.float32), np.array(intensities, dtype=np.float32)
        
        elif 'binary' in data_line[0]:
            points_count = 0
            field_names = []
            field_sizes = []
            field_types = []
            field_counts = []
            
            for line in header_lines:
                line = line.strip()
                if line.startswith('POINTS'):
                    points_count = int(line.split()[1])
                elif line.startswith('FIELDS'):
                    field_names = line.split()[1:]
                elif line.startswith('SIZE'):
                    field_sizes = [int(x) for x in line.split()[1:]]
                elif line.startswith('TYPE'):
                    field_types = line.split()[1:]
                elif line.startswith('COUNT'):
                    field_counts = [int(x) for x in line.split()[1:]]
            
            if not field_names or points_count == 0:
                print('无法解析PCD头部信息')
                return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
            
            point_size = sum(size * count for size, count in zip(field_sizes, field_counts))
            
            # 重新读取文件获取二进制数据
            f.seek(0)
            content = f.read()
            data_marker = b"DATA binary\n"
            data_start = content.find(data_marker)
            if data_start == -1:
                data_marker = b"DATA binary\r\n"
                data_start = content.find(data_marker)
            if data_start == -1:
                print('无法找到二进制数据开始位置')
                return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
            
            data_start += len(data_marker)
            binary_data = content[data_start:]
            
            # 找到xyz和intensity字段的索引
            xyz_indices = [i for i, name in enumerate(field_names) if name in ['x', 'y', 'z']]
            intensity_index = None
            if 'intensity' in field_names:
                intensity_index = field_names.index('intensity')
            
            # 构建struct格式
            fmt_map = {'F': 'f', 'U': 'B', 'I': 'i'}
            struct_fmt = '<' + ''.join(fmt_map.get(t, 'x') * c for t, c in zip(field_types, field_counts))
            struct_size = struct.calcsize(struct_fmt)
            
            if struct_size != point_size:
                # 简化为只读取xyz
                struct_fmt = '<' + 'f' * 3
                struct_size = 12
            
            for i in range(points_count):
                offset = i * point_size
                if offset + point_size > len(binary_data):
                    break
                raw = binary_data[offset:offset+point_size]
                vals = struct.unpack(struct_fmt, raw[:struct_size])
                
                if len(vals) >= max(xyz_indices)+1:
                    xyz = [vals[xyz_indices[0]], vals[xyz_indices[1]], vals[xyz_indices[2]]]
                    points.append(xyz)
                    
                    if intensity_index is not None and len(vals) > intensity_index:
                        intensities.append(vals[intensity_index])
                    else:
                        intensities.append(0.0)
            
            return np.array(points, dtype=np.float32), np.array(intensities, dtype=np.float32)
        
        else:
            print('未知PCD数据格式')
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)


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
    
    # 将欧拉角转换为旋转矩阵
    # def euler_to_rotation_matrix(roll, pitch, yaw):
    #     """欧拉角转旋转矩阵（ZYX顺序）"""
    #     cos_r, sin_r = np.cos(roll), np.sin(roll)
    #     cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    #     cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        
    #     R_x = np.array([[1, 0, 0],
    #                    [0, cos_r, -sin_r],
    #                    [0, sin_r, cos_r]])
        
    #     R_y = np.array([[cos_p, 0, sin_p],
    #                    [0, 1, 0],
    #                    [-sin_p, 0, cos_p]])
        
    #     R_z = np.array([[cos_y, -sin_y, 0],
    #                    [sin_y, cos_y, 0],
    #                    [0, 0, 1]])
        
    #     return R_z @ R_y @ R_x
    
    # R = euler_to_rotation_matrix(roll, pitch, yaw)
    # t = np.array([x, y, z], dtype=np.float32)
    
    # return R, t
    # 欧拉角(ZYX) -> 旋转矩阵
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
    print(f"相机 {camera_id} 外参逆: x={xyzrpy_inv[0]:.4f}, y={xyzrpy_inv[1]:.4f}, z={xyzrpy_inv[2]:.4f}, roll={xyzrpy_inv[3]:.4f}, pitch={xyzrpy_inv[4]:.4f}, yaw={xyzrpy_inv[5]:.4f}")

    # 提取逆后的 R 和 t
    R_inv = T_inv[:3, :3]
    t_inv = T_inv[:3, 3]

    return R_inv, t_inv


def project_pcd_to_image(pcd_points, pcd_intensity, R, t, K, dist, img):
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
    
    # 伪彩色映射并绘制点
    if len(norm_int) > 0:
        color_map = cv2.applyColorMap(norm_int.reshape(-1, 1), cv2.COLORMAP_JET)
        for idx, pt in enumerate(img_points_valid):
            color = tuple(int(c) for c in color_map[idx, 0])  # BGR
            cv2.circle(result_img, (int(pt[0]), int(pt[1])), 1, color, -1)
    
    print(f"成功投影了 {len(img_points_valid)} 个点到图像上")
    return result_img


def main():
    if len(sys.argv) != 2:
        print("使用方法: python project_pcd_to_image.py <camera_id>")
        print("camera_id: 0-6的整数")
        print("参数说明:")
        print("    0: camera_front_4k")
        print("    1: camera_front")
        print("    2: camera_rear")
        print("    3: camera_front_left")
        print("    4: camera_front_right")
        print("    5: camera_rear_left")
        print("    6: camera_rear_right")
        sys.exit(1)
    
    try:
        camera_id = int(sys.argv[1])
        if camera_id < 0 or camera_id > 6:
            raise ValueError("camera_id必须是0-5之间的整数")
    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)
    
    # 定义图像路径映射
    image_paths = {
        0: 'camera_front_4k',
        1: 'camera_front',
        2: 'camera_rear',
        3: 'camera_front_left',
        4: 'camera_front_right',
        5: 'camera_rear_left',
        6: 'camera_rear_right'
    }
    
    # 获取脚本所在目录的上级目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 根据camera_id获取对应的文件夹名称
    camera_folder = image_paths[camera_id]
    
    # 构建文件路径
    extrinsic_file = os.path.join(project_root, 'config', 'extrinsic_parameters', 'sensor_kit_calibration.yaml')
    intrinsic_file = os.path.join(project_root, 'config', 'intrinsic_parameters', f'camera{camera_id}_params.yaml')
    # pcd_file = os.path.join(project_root, 'data', 'first_frame_raw.pcd')
    # image_file = os.path.join(project_root, 'data', camera_folder, 'test.png')
    pcd_file = os.path.join(project_root, 'cache', 'processed_cloud.pcd')
    image_file = os.path.join(project_root, 'data', camera_folder, 'image_undistort.png')
    
    # 检查文件是否存在
    for file_path, file_desc in [(extrinsic_file, '外参配置文件'),
                                 (intrinsic_file, '内参配置文件'),
                                 (pcd_file, 'PCD点云文件'),
                                 (image_file, '图像文件')]:
        if not os.path.exists(file_path):
            print(f"错误: {file_desc} 不存在: {file_path}")
            sys.exit(1)
    
    try:
        # 加载相机内外参
        print(f"加载camera{camera_id}的参数...")
        K, dist = load_camera_intrinsics(intrinsic_file)
        R, t = load_camera_extrinsics(extrinsic_file, camera_id)
        
        print(f"内参矩阵 K:\n{K}")
        print(f"畸变系数 dist: {dist}")
        print(f"旋转矩阵 R:\n{R}")
        print(f"平移向量 t: {t}")
        
        # 读取点云数据
        print("读取点云数据...")
        pcd_points, pcd_intensity = read_pcd_xyz_intensity(pcd_file)
        # pcd_points, pcd_intensity = read_pcd_xyz_intensity("/home/pix/code/slam/first_frame.pcd")
        print(f"读取到 {len(pcd_points)} 个点")
        
        # 读取图像
        print("读取图像...")
        img = cv2.imread(image_file)
        if img is None:
            raise FileNotFoundError(f'无法读取图像文件: {image_file}')
        
        # 投影点云到图像
        print("将点云投影到图像...")
        result_img = project_pcd_to_image(pcd_points, pcd_intensity, R, t, K, dist, img)
        
        # 显示结果
        cv2.imshow(f'Camera{camera_id} 点云投影结果', result_img)
        print("按任意键退出...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 保存结果图像
        output_file = os.path.join(project_root, 'cache', f'camera{camera_id}_projection_result.png')
        cv2.imwrite(output_file, result_img)
        print(f"结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
