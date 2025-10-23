#ifndef RCLCOMM_H
#define RCLCOMM_H

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <memory>
#include "extraboard.h"
#include "calibrate.h"
#include "detaruco.h"
#include "cornersmatch.h"
#include "spd_log.h"
#include <thread>
#include <atomic>
#include <yaml-cpp/yaml.h>

struct SensorExtrinsics
{
    std::string father_frame_id; // 父坐标系
    double x; // 位置 x 坐标
    double y; // 位置 y 坐标
    double z; // 位置 z 坐标
    double roll; // 绕 x 轴旋转角度
    double pitch; // 绕 y 轴旋转角度
    double yaw; // 绕 z 轴旋转角度
    SensorExtrinsics() : x(0.0), y(0.0), z(0.0), roll(0.0), pitch(0.0), yaw(0.0) {}
};

class RclComm : public rclcpp::Node
{
public:
    RclComm();
    ~RclComm();

    // 启动节点
    void start();
    SPD_LOG *logger_;

    std::string data_path_;
    std::string template_image_path_;
    std::string parameters_path_;
    std::string cache_directory_;

private:
    // 点云发布器
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_target_publisher_;
    
    // BBox发布器
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr bbox_publisher_;
    
    // 定时器
    rclcpp::TimerBase::SharedPtr timer_;
    
    // TF2广播器
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;
    
    // ExtraBoard实例
    std::unique_ptr<ExtraBoard> extraboard_;

    // DetAruco实例

    // CornersMatch实例
    CornersMatch* cornersmatch_;

    
    // 定时器回调函数
    void timer_callback();
    
    // 发布点云数据
    void publish_pointcloud();
    
    // 发布目标点云数据（BoardInfo转换的点云）
    void publish_target_pointcloud(const std::vector<BoardInfo>& boards);
    
    // 发布BBox信息
    void publish_bbox_array(const std::vector<BBox>& bboxes);
    
    // 发布board_id文字标签
    void publish_board_id_labels(const std::vector<BoardInfo>& boards);
    
    // 初始化发布器
    void init_publishers();
    
    // 初始化定时器
    void init_timers();
    
    // 初始化参数
    void init_parameters();

    // 存储最新的BoardInfo
    BoardInfo board_info;
    
    // 参数变量
    double publish_frequency_;
    std::string pointcloud_topic_;
    std::string pointcloud_target_topic_;
    std::string bbox_topic_;
    std::string frame_id_;
    // 算法参数
    AlgorithmParams algorithm_params_;
    // ArUco参数
    ArucoInitParams aruco_params_;

    std::vector<int> calib_camera_list_; // 待标定相机ID列表
    std::vector<int> camera_id_list_; // 所有相机ID列表
    std::vector<std::string> camera_name_list_; // 所有相机名称列表

public:
    YAML::Node read_yaml_file(const std::string& file_path);
    void publish_sensor_tf2();
    std::map<std::string, SensorExtrinsics> sensor_extrinsics_map; // 存储传感器外参
    void init_tf2_map();
    void compute_base_to_sensor_kit_transform(); // 计算base_link到sensor_kit_base_link的变换矩阵

    DetAruco* detaruco_;
};

#endif // RCLCOMM_H
