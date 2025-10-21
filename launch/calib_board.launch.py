from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    package_name = 'calib_board'

    # 仅使用 install 目录下的配置文件
    config_file = os.path.join(
        get_package_share_directory(package_name),
        'config',
        'calib_board_config.yaml'
    )
    # 调试打印，可删
    print(f'使用安装目录配置文件: {config_file}')

    return LaunchDescription([
        Node(
            package=package_name,
            executable='calib_board_node',
            name='calib_board_node',
            output='screen',
            parameters=[config_file]
        ),
    ])