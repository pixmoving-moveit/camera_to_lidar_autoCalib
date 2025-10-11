import cv2
import numpy as np
import yaml
import os
import sys

def main():
    if len(sys.argv) != 2:
        print("使用方法: python3 undistort_2k.py <camera_id>")
        print("参数说明: camera_id 必须是1-5之间的整数")
        print("    1: camera_rear")
        print("    2: camera_front_left")
        print("    3: camera_front_right")
        print("    4: camera_rear_left")
        print("    5: camera_rear_right")
        print("示例: python3 undistort_2k.py 1")
        return

    try:
        camera_id = int(sys.argv[1])
        if not 1 <= camera_id <= 5:
            raise ValueError
    except ValueError:
        print("错误：camera_id 必须是1到5之间的整数")
        return
        
    # 预定义的图像路径列表
    image_paths = {
        1: 'camera_rear',
        2: 'camera_front_left',
        3: 'camera_front_right',
        4: 'camera_rear_left',
        5: 'camera_rear_right'
    }
        
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(script_dir, f'../config/intrinsic_parameters/camera{camera_id}_params.yaml')
    img_path = os.path.join(script_dir, f'../data/{image_paths[camera_id]}/')
    crop_w, crop_h = 1024, 576

    # 读取相机参数
    with open(yaml_path, 'r') as f:
        camera_intrinsic_data = yaml.safe_load(f)
    K = np.array(camera_intrinsic_data['camera_matrix']['data']).reshape(
        camera_intrinsic_data['camera_matrix']['rows'], 
        camera_intrinsic_data['camera_matrix']['cols'])
    D = np.array(camera_intrinsic_data['distortion_coefficients']['data']).astype(np.float32)
    original_width = camera_intrinsic_data['image_width']
    original_height = camera_intrinsic_data['image_height']

    print("K:", K)
    print("D:", D)
    print("=" * 60)
    print("相机内参调整分析")
    print("=" * 60)
    print(f"原始图像尺寸: {original_width} x {original_height}")
    print("原始内参矩阵:")
    print(K)
    print()

    # 单张图片处理

    # 支持多种图片后缀
    img = None
    img_file = None
    for ext in ['png', 'jpg', 'jpeg']:
        img_file = os.path.join(img_path, f'image.{ext}')
        if os.path.exists(img_file):
            img = cv2.imread(img_file)
            if img is not None:
                break
    if img is None:
        print(f"错误：文件不存在：{img_file}")
        return
    h, w = img.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    img_undistorted = cv2.undistort(img, K, D, None, new_K)
    uh, uw = img_undistorted.shape[:2]
    # 计算中心裁剪区域
    x0 = (uw - crop_w) // 2
    y0 = (uh - crop_h) // 2
    img_cropped = img_undistorted[y0:y0+crop_h, x0:x0+crop_w]
    # 内参主点同步平移
    K_cropped = new_K.copy()
    K_cropped[0, 2] -= x0
    K_cropped[1, 2] -= y0
    cv2.imwrite(img_path + 'image_undistort.png', img_cropped)
    print(f"已保存去畸变+中心裁剪图像：{img_path + 'image_undistort.png'}")
    print(f"中心裁剪后分辨率: {img_cropped.shape[1]} x {img_cropped.shape[0]}")
    print("中心裁剪后内参矩阵:")
    print(K_cropped)
    
    # 更新yaml文件中的rectification_matrix，保持原有结构
    # 读取原始yaml文件内容
    with open(yaml_path, 'r') as f:
        yaml_content = f.read()
    
    # 准备新的rectification_matrix数据
    matrix_data = K_cropped.flatten().tolist()
    new_matrix_str = "  rows: 3\n  cols: 3\n  data:\n  - {}\n  - {}\n  - {}\n  - {}\n  - {}\n  - {}\n  - {}\n  - {}\n  - {}".format(*matrix_data)
    
    # 在文件内容中定位并替换rectification_matrix部分
    start = yaml_content.find("rectification_matrix:")
    if start != -1:
        # 找到下一个主要字段的开始位置
        next_field = yaml_content.find("\n", start)
        next_field = yaml_content.find("\n", next_field + 1)
        while next_field != -1 and yaml_content[next_field + 1] in [' ', '\n']:
            next_field = yaml_content.find("\n", next_field + 1)
        if next_field == -1:
            next_field = len(yaml_content)
        
        # 替换内容
        yaml_content = yaml_content[:start] + "rectification_matrix:\n" + new_matrix_str + yaml_content[next_field:]
        
        # 写回文件
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        print("已更新yaml文件中的rectification_matrix（保持原有结构）")
    else:
        print("警告：未找到rectification_matrix字段")
    print("=" * 60)

if __name__ == '__main__':
    main()
