import numpy as np
import yaml
import sys
import os
import shutil

def get_home_path():
    """获取当前系统的home目录路径"""
    # 方法1: 使用os.path.expanduser
    home_path = os.path.expanduser("~")
    
    # 方法2: 使用os.environ获取环境变量
    # home_path = os.environ.get('HOME')  # Linux/Mac
    # home_path = os.environ.get('USERPROFILE')  # Windows
    
    # 方法3: 使用pathlib.Path (Python 3.5+)
    # from pathlib import Path
    # home_path = Path.home()
    
    return home_path

def copy_pix_params(src_dir, dst_dir):
    """
    将指定目录A下的所有文件夹拷贝到另一指定文件夹B
    若B中已经存在，则提示并覆盖
    
    Args:
        src_dir: 源目录路径
        dst_dir: 目标目录路径
    """
    # 检查源目录是否存在
    if not os.path.exists(src_dir):
        print(f"错误: 源目录 '{src_dir}' 不存在")
        return False
    
    if not os.path.isdir(src_dir):
        print(f"错误: '{src_dir}' 不是目录")
        return False
    
    # 确保目标目录存在
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        print(f"创建目标目录: {dst_dir}")
    
    # 获取源目录下的所有文件夹
    try:
        items = os.listdir(src_dir)
    except PermissionError:
        print(f"错误: 没有权限访问源目录 '{src_dir}'")
        return False
    
    folder_copied = 0
    folder_overwritten = 0
    
    for item in items:
        src_path = os.path.join(src_dir, item)
        
        # 只处理文件夹
        if os.path.isdir(src_path):
            dst_path = os.path.join(dst_dir, item)
            
            # 检查目标位置是否已存在
            if os.path.exists(dst_path):
                if os.path.isdir(dst_path):
                    print(f"提示: 目标文件夹 '{item}' 已存在，将覆盖...")
                    # 删除已存在的文件夹
                    try:
                        shutil.rmtree(dst_path)
                        folder_overwritten += 1
                    except Exception as e:
                        print(f"错误: 无法删除已存在的文件夹 '{dst_path}': {e}")
                        continue
                else:
                    print(f"错误: 目标路径 '{dst_path}' 存在但不是文件夹")
                    continue
            
            # 拷贝文件夹
            try:
                shutil.copytree(src_path, dst_path)
                folder_copied += 1
                if os.path.exists(dst_path):
                    print(f"成功覆盖文件夹: {item}")
                else:
                    print(f"成功拷贝文件夹: {item}")
            except Exception as e:
                print(f"错误: 拷贝文件夹 '{item}' 失败: {e}")
    
    print(f"\n拷贝完成:")
    print(f"  新拷贝文件夹数量: {folder_copied - folder_overwritten}")
    print(f"  覆盖文件夹数量: {folder_overwritten}")
    print(f"  总计处理文件夹数量: {folder_copied}")
    
    return True

def check_camera_intrinsics(intrinsic_file):
    """
    检查相机内参，如果rectification_matrix是单位阵，则用camera_matrix替换
    """
    with open(intrinsic_file, 'r') as f:
        data = yaml.safe_load(f)
    
    # 定义单位阵
    identity_matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    
    # 检查是否存在camera_matrix
    camera_matrix_data = None
    if 'camera_matrix' in data and 'data' in data['camera_matrix']:
        camera_matrix_data = data['camera_matrix']['data']
        print(f"找到camera_matrix: {camera_matrix_data}")
    
    # 检查是否存在rectification_matrix
    if 'rectification_matrix' in data and 'data' in data['rectification_matrix']:
        rectification_matrix_data = data['rectification_matrix']['data']
        print(f"找到rectification_matrix: {rectification_matrix_data}")
        
        # 检查rectification_matrix是否为单位阵
        is_identity = True
        for i in range(9):
            if abs(rectification_matrix_data[i] - identity_matrix[i]) > 1e-6:
                is_identity = False
                break
        
        if is_identity:
            print("检测到rectification_matrix是单位阵")
            
            # 如果camera_matrix存在，则用camera_matrix替换rectification_matrix
            if camera_matrix_data is not None:
                print("将camera_matrix赋值给rectification_matrix")
                data['rectification_matrix']['data'] = camera_matrix_data
                
                # 保存修改后的数据回文件
                with open(intrinsic_file, 'w') as f:
                    yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True, width=2**20)
                print(f"已更新文件: {intrinsic_file}")
            else:
                print("警告: camera_matrix不存在，无法替换rectification_matrix")
        else:
            print("rectification_matrix不是单位阵，无需修改")
    else:
        print("未找到rectification_matrix")
        


# 测试获取home目录
if __name__ == "__main__":
    home = get_home_path()

    raw_data_dir = os.path.join(home, "pix/parameter/sensor_kit/robobus_sensor_kit_description")
    target_dir = os.path.join(home, "pix/calibration_new/calib_data/config")
    
    print(f"原始数据目录: {raw_data_dir}")
    print(f"目标配置目录: {target_dir}")
    copy_pix_params(raw_data_dir, target_dir)
    print(f"已更新pix参数配置文件")
    print(" ")

    for i in range(0, 6):
        print(f"========检查相机{i}内参========")
        intrinsic_file = os.path.join(target_dir, f"intrinsic_parameters/camera{i}_params.yaml")
        check_camera_intrinsics(intrinsic_file)