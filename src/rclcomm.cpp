#include "rclcomm.h"
#include <chrono>
#include <visualization_msgs/msg/marker_array.hpp>
#include <thread>
#include <atomic>

using namespace std::chrono_literals;

RclComm::RclComm() : Node("calib_board_node")
{
    spdlog::info("CalibBoard节点开始初始化");
    // 初始化参数
    init_parameters();
    init_tf2_map();
    
    
    // 初始化发布器和定时器
    init_publishers();
    init_timers();

    // 立即发布一次TF2变换，用于程序中，点云读取后立刻变换到sensor_kit_base_link坐标系
    publish_sensor_tf2();
    
    // 初始化ExtraBoard实例
    extraboard_ = std::make_unique<ExtraBoard>(cache_directory_);
    
    // 初始化DetAruco实例
    detaruco_ = DetAruco::getInstance();
    detaruco_->init_DetAruco(aruco_params_);

    cornersmatch_ = CornersMatch::getInstance();
    cornersmatch_->init_cornersMatch(aruco_params_);
    
    spdlog::info("CalibBoard初始化完成");
}

RclComm::~RclComm()
{
    spdlog::info("CalibBoard节点已关闭");
    
    SPD_LOG::shutdown();    //在节点关闭前手动关闭log
}

void RclComm::start()
{
    spdlog::info("CalibBoard节点开始运行...");
    
    // 设置算法参数到ExtraBoard
    extraboard_->set_algorithm_params(algorithm_params_);

    // 使用参数加载模板图像
    spdlog::info("加载模板图像:{}", template_image_path_);
    if (!extraboard_->load_template_image(template_image_path_))
    {
        spdlog::error("无法载入模板图像:{}", template_image_path_);
        return;
    }
    spdlog::info("成功载入模板图像");

    // 使用参数加载PCD文件
    spdlog::info("加载PCD文件:{}", data_path_);
    extraboard_->load_pcd_file(data_path_ + "/input.pcd");

    int check_result = extraboard_->check_orientation(calib_camera_list_, camera_id_list_, camera_name_list_);
    if(check_result <= 0)
    {
        spdlog::error("标定相机列表有问题，不符合前后分组要求");
        return;
    }

    std::vector<PointCloudT> input;
    extraboard_->extract_points(input);
    extraboard_->board_register(input, extraboard_->boards, check_result);

    cornersmatch_->setBoardPointCloudCenters_Q(extraboard_->boards);    //设置所有检测到的点云标定板

    // return;
    auto calibfunc = CalibFunc::getInstance();
    calibfunc->init_CalibFunc(parameters_path_);                    //读取所有相机的内参

    for(const int& cam_id : calib_camera_list_)
    {
        spdlog::info("待标定相机ID:{}", cam_id);
    }

    for(const int& cam_id : calib_camera_list_)
    {
        spdlog::info("====>准备标定相机 ID:{}", cam_id);

        IntrParams intr_params = calibfunc->getIntrParams(cam_id);   //获取相机cam_id的内参
        auto detaruco_info = detaruco_->detectArucoMarkers(cam_id, intr_params); //检测相机cam_id拍摄到的Aruco标记
        cornersmatch_->setBoardArucoCenters_P(detaruco_info);

        std::vector<BoardInfo> result;
        cornersmatch_->getMatchResult(result);

        calibfunc->setDataForCalib(cam_id, result);
        calibfunc->calibrate(cam_id);

        spdlog::info("====>相机标定完成 ID:{}", cam_id);
    }


    calibfunc->write_Extrinsics();  //写入所有相机的外参

}

void RclComm::init_publishers()
{
    // 创建点云发布器
    pointcloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        pointcloud_topic_, 10);
    
    // 创建目标点云发布器
    pointcloud_target_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        pointcloud_target_topic_, 10);
    
    // 创建BBox发布器
    bbox_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        bbox_topic_, 10);

    // 初始化TF2广播器
    static_tf_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(*this);
    
    spdlog::info("发布器已初始化 - 话题:{}, {}, {}", pointcloud_topic_, pointcloud_target_topic_, bbox_topic_);
}

void RclComm::init_timers()
{
    // 使用参数中的频率创建定时器
    auto timer_period = std::chrono::milliseconds(static_cast<int>(1000.0 / publish_frequency_));
    timer_ = this->create_wall_timer(
        timer_period, std::bind(&RclComm::timer_callback, this));
    
    spdlog::info("定时器已初始化，发布频率: {} Hz", publish_frequency_);

}

void RclComm::timer_callback()
{
    // 定时发布点云数据
    publish_sensor_tf2();
    
    publish_pointcloud();

    publish_target_pointcloud(extraboard_->boards);

    if (!extraboard_->bboxes.empty())
    {
        publish_bbox_array(extraboard_->bboxes);
    }
}

void RclComm::publish_pointcloud()
{
    if (!extraboard_->is_empty())
    {
        // 获取ROS2点云消息
        auto pointcloud_msg = extraboard_->get_ros_pointcloud(frame_id_);
        
        // 设置时间戳
        pointcloud_msg.header.stamp = this->get_clock()->now();
        
        // 发布点云
        pointcloud_publisher_->publish(pointcloud_msg);
        
    }
    else
    {
        spdlog::warn("点云数据为空, 请检查PCD文件加载情况!");
    }
}


void RclComm::publish_target_pointcloud(const std::vector<BoardInfo>& boards)
{
    // 创建点云
    PointCloudT::Ptr target_cloud(new PointCloudT);
    
    
    for(size_t idx=0; idx<boards.size(); ++idx)
    {
        if (!boards.at(idx).corners_cloud.is_valid) continue; //跳过无效的标定板信息
        const auto& board_info = boards[idx].corners_cloud;
        std::vector<PointT> board_points = {
            board_info.center_3d,
            board_info.corn_left_top_3d,
            board_info.corn_right_top_3d,
            board_info.corn_right_bottom_3d,
            board_info.corn_left_bottom_3d
        };
        
        for (size_t i = 0; i < board_points.size(); ++i)
        {
            auto point = board_points[i];
            point.intensity = 100.0f + i * 50.0f; // 为不同点设置不同强度值以便区分
            target_cloud->points.push_back(point);
        }
    }
    
    // 设置点云属性
    target_cloud->width = target_cloud->points.size();
    target_cloud->height = 1;
    target_cloud->is_dense = true;
    
    if (!target_cloud->points.empty())
    {
        // 转换为ROS2消息
        sensor_msgs::msg::PointCloud2 target_msg;
        pcl::toROSMsg(*target_cloud, target_msg);
        target_msg.header.frame_id = frame_id_;
        target_msg.header.stamp = this->get_clock()->now();
        
        // 发布目标点云
        pointcloud_target_publisher_->publish(target_msg);
        
        // 发布board_id文字标签
        publish_board_id_labels(boards);
        
    }
    else
    {
        spdlog::warn("目标点云为空，无法发布!");
    }
}

void RclComm::publish_board_id_labels(const std::vector<BoardInfo>& boards)
{
    // 创建MarkerArray消息，用于显示board_id文字标签
    visualization_msgs::msg::MarkerArray marker_array_msg;
    
    // 遍历每个Board并创建文字标签
    for (size_t i = 0; i < boards.size(); ++i)
    {
        const auto& board = boards[i];
        
        // 跳过无效的标定板信息
        if (!board.corners_cloud.is_valid) continue;
        
        // 创建TEXT_VIEW_FACING类型的Marker
        visualization_msgs::msg::Marker text_marker;
        
        // 设置消息头部
        text_marker.header.frame_id = frame_id_;
        text_marker.header.stamp = this->get_clock()->now();
        
        // 设置命名空间和ID
        text_marker.ns = "board_id_labels";
        text_marker.id = i;
        
        // 设置Marker类型为TEXT_VIEW_FACING
        text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::msg::Marker::ADD;
        
        // 设置位置：center_3d的Z轴上方50cm处
        text_marker.pose.position.x = board.corners_cloud.center_3d.x;
        text_marker.pose.position.y = board.corners_cloud.center_3d.y;
        text_marker.pose.position.z = board.corners_cloud.center_3d.z + 0.6; // 上方60cm
        
        // 设置方向（文字标签不需要特殊方向）
        text_marker.pose.orientation.x = 0.0;
        text_marker.pose.orientation.y = 0.0;
        text_marker.pose.orientation.z = 0.0;
        text_marker.pose.orientation.w = 1.0;
        
        // 设置尺寸（文字大小）
        text_marker.scale.x = 0.0; // TEXT_VIEW_FACING类型不使用x和y
        text_marker.scale.y = 0.0;
        text_marker.scale.z = 0.3; // 文字高度30cm
        
        // 设置颜色（白色文字）
        text_marker.color.r = 1.0;
        text_marker.color.g = 1.0;
        text_marker.color.b = 1.0;
        text_marker.color.a = 1.0;
        
        // 设置文字内容为board_id
        text_marker.text = "Aruco:" + std::to_string(board.board_id);
        
        // 设置生命周期
        text_marker.lifetime = rclcpp::Duration::from_seconds(0); // 永久显示
        
        // 添加到MarkerArray
        marker_array_msg.markers.push_back(text_marker);
    }
    
    // 发布MarkerArray
    if (!marker_array_msg.markers.empty())
    {
        bbox_publisher_->publish(marker_array_msg);
        RCLCPP_DEBUG(this->get_logger(), "已发布 %zu 个board_id标签", marker_array_msg.markers.size());
    }
}

void RclComm::publish_bbox_array(const std::vector<BBox>& bboxes)
{
    // 创建MarkerArray消息
    visualization_msgs::msg::MarkerArray marker_array_msg;
    
    // 遍历每个BBox并转换为CUBE类型的Marker
    for (size_t i = 0; i < bboxes.size(); ++i)
    {
        const auto& bbox = bboxes[i];
        
        // 1. 创建CUBE类型的Marker
        visualization_msgs::msg::Marker cube_marker;
        
        // 设置消息头部
        cube_marker.header.frame_id = frame_id_;
        cube_marker.header.stamp = this->get_clock()->now();
        
        // 设置marker的基本属性
        cube_marker.ns = "bounding_boxes";
        cube_marker.id = static_cast<int>(i);
        cube_marker.type = visualization_msgs::msg::Marker::CUBE;
        cube_marker.action = visualization_msgs::msg::Marker::ADD;
        
        // 设置中心位置
        cube_marker.pose.position.x = bbox.position.x();
        cube_marker.pose.position.y = bbox.position.y();
        cube_marker.pose.position.z = bbox.position.z();
        
        // 设置方向（四元数）
        cube_marker.pose.orientation.x = bbox.orientation.x();
        cube_marker.pose.orientation.y = bbox.orientation.y();
        cube_marker.pose.orientation.z = bbox.orientation.z();
        cube_marker.pose.orientation.w = bbox.orientation.w();
        
        // 设置尺寸
        cube_marker.scale.x = bbox.size.x();
        cube_marker.scale.y = bbox.size.y();
        cube_marker.scale.z = bbox.size.z();
        
        // 设置颜色（半透明蓝色）
        cube_marker.color.r = 0.0f;
        cube_marker.color.g = 0.0f;
        cube_marker.color.b = 1.0f;
        cube_marker.color.a = 0.3f;  // 半透明
        
        // 设置生存时间（0表示永久）
        cube_marker.lifetime = rclcpp::Duration::from_seconds(0);
        
        // 添加CUBE marker到数组
        marker_array_msg.markers.push_back(cube_marker);
        
        // 2. 创建TEXT类型的Marker来显示序号
        visualization_msgs::msg::Marker text_marker;
        
        // 设置消息头部
        text_marker.header.frame_id = frame_id_;
        text_marker.header.stamp = this->get_clock()->now();
        
        // 设置marker的基本属性
        text_marker.ns = "bbox_labels";
        text_marker.id = static_cast<int>(i);
        text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::msg::Marker::ADD;
        
        // 设置文本位置（在BBox上方偏移一点）
        text_marker.pose.position.x = bbox.position.x();
        text_marker.pose.position.y = bbox.position.y();
        text_marker.pose.position.z = bbox.position.z() + bbox.size.z() / 2.0f + 0.1f; // 在BBox顶部上方10cm
        
        // 设置方向（文本面向相机）
        text_marker.pose.orientation.x = 0.0;
        text_marker.pose.orientation.y = 0.0;
        text_marker.pose.orientation.z = 0.0;
        text_marker.pose.orientation.w = 1.0;
        
        // 设置文本大小
        text_marker.scale.z = 0.2; // 文本高度为20cm
        
        // 设置颜色（白色）
        text_marker.color.r = 1.0f;
        text_marker.color.g = 1.0f;
        text_marker.color.b = 1.0f;
        text_marker.color.a = 1.0f;  // 不透明
        
        // 设置文本内容
        text_marker.text = "BBox_" + std::to_string(i);
        
        // 设置生存时间（0表示永久）
        text_marker.lifetime = rclcpp::Duration::from_seconds(0);
        
        // 添加TEXT marker到数组
        marker_array_msg.markers.push_back(text_marker);
        
        RCLCPP_DEBUG(this->get_logger(), 
                    "BBox %zu - 位置: (%.3f, %.3f, %.3f), 尺寸: (%.3f, %.3f, %.3f)",
                    i, bbox.position.x(), bbox.position.y(), bbox.position.z(),
                    bbox.size.x(), bbox.size.y(), bbox.size.z());
    }
    
    // 发布MarkerArray
    if (!marker_array_msg.markers.empty())
    {
        bbox_publisher_->publish(marker_array_msg);
    }
}

void RclComm::init_parameters()
{
    // 声明并获取基本参数
    this->declare_parameter<std::string>("data_path", "/home/pix/data/camera_lidar/calibBoard/lio_board2.pcd");
    this->declare_parameter<std::string>("template_image_path", "/home/pix/data/camera_lidar/calibBoard/template.png");
    this->declare_parameter<std::string>("parameters_path", "/home/pix/code/calibration_ws/src/calibBoard/config");
    this->declare_parameter<double>("publish_frequency", 1.0);
    this->declare_parameter<std::string>("pointcloud_topic", "pointcloud_board");
    this->declare_parameter<std::string>("pointcloud_target_topic", "pointcloud_target");
    this->declare_parameter<std::string>("bbox_topic", "bounding_boxes");
    this->declare_parameter<std::string>("frame_id", "base_link");
    this->declare_parameter<std::string>("cache_directory", "/tmp/calib_cache");
    this->declare_parameter<std::vector<int>>("camera_list", std::vector<int>{0});
    this->declare_parameter<std::vector<std::string>>("camera_name_list", std::vector<std::string>{"camera_default"});
    this->declare_parameter<std::vector<int>>("calib_camera_list", std::vector<int>{0});
    this->declare_parameter<int>("export_yolo_data", 0);

    
    // 声明算法参数
    this->declare_parameter<std::vector<double>>("algorithm_params.min_point", std::vector<double>{8.3, -4.8, -0.5, 1.0});
    this->declare_parameter<std::vector<double>>("algorithm_params.max_point", std::vector<double>{16.5, 4.6, 3.4, 1.0});
    this->declare_parameter<double>("algorithm_params.min_size", 0.5);
    this->declare_parameter<double>("algorithm_params.max_size", 1.1);
    this->declare_parameter<double>("algorithm_params.aspect_ratio_threshold", 1.2);
    this->declare_parameter<double>("algorithm_params.bracket_width", 0.15); // 支架宽度（米）



    // 声明Aruco参数
    this->declare_parameter<double>("aruco_params.marker_length", 0.637); // 单个Aruco标记的边长（米）
    this->declare_parameter<double>("aruco_params.marker_scale", 1.7);    // 外围框放大比例
    this->declare_parameter<double>("aruco_params.marker_shift", 0.13725); // 外围框平移比例
    this->declare_parameter<int>("aruco_params.marker_num", 6);
    this->declare_parameter<std::vector<int>>("aruco_params.marker_order", std::vector<int>{1, 2, 3, 4, 5, 6});

    
    // 获取基本参数值
    data_path_ = this->get_parameter("data_path").as_string();
    template_image_path_ = this->get_parameter("template_image_path").as_string();
    parameters_path_ = this->get_parameter("parameters_path").as_string();
    publish_frequency_ = this->get_parameter("publish_frequency").as_double();
    pointcloud_topic_ = this->get_parameter("pointcloud_topic").as_string();
    pointcloud_target_topic_ = this->get_parameter("pointcloud_target_topic").as_string();
    bbox_topic_ = this->get_parameter("bbox_topic").as_string();
    frame_id_ = this->get_parameter("frame_id").as_string();
    cache_directory_ = this->get_parameter("cache_directory").as_string();
    auto list = this->get_parameter("calib_camera_list").as_integer_array();
    calib_camera_list_.assign(list.begin(), list.end());   // 自动转型

    auto camera_id_list = this->get_parameter("camera_list").as_integer_array();
    camera_id_list_.assign(camera_id_list.begin(), camera_id_list.end());   // 自动转型
    camera_name_list_ = this->get_parameter("camera_name_list").as_string_array();


    // 获取算法参数值
    auto min_point_vec = this->get_parameter("algorithm_params.min_point").as_double_array();
    auto max_point_vec = this->get_parameter("algorithm_params.max_point").as_double_array();
    
    algorithm_params_.min_point = Eigen::Vector4f(min_point_vec[0], min_point_vec[1], min_point_vec[2], min_point_vec[3]);
    algorithm_params_.max_point = Eigen::Vector4f(max_point_vec[0], max_point_vec[1], max_point_vec[2], max_point_vec[3]);
    algorithm_params_.min_size = static_cast<float>(this->get_parameter("algorithm_params.min_size").as_double());
    algorithm_params_.max_size = static_cast<float>(this->get_parameter("algorithm_params.max_size").as_double());
    algorithm_params_.aspect_ratio_threshold = static_cast<float>(this->get_parameter("algorithm_params.aspect_ratio_threshold").as_double());
    algorithm_params_.bracket_width = static_cast<float>(this->get_parameter("algorithm_params.bracket_width").as_double()); // 支架宽度（米）



    // 获取Aruco参数值
    aruco_params_.path = data_path_;
    aruco_params_.marker_length = static_cast<float>(this->get_parameter("aruco_params.marker_length").as_double());
    aruco_params_.marker_scale = static_cast<float>(this->get_parameter("aruco_params.marker_scale").as_double());
    aruco_params_.marker_shift = static_cast<float>(this->get_parameter("aruco_params.marker_shift").as_double());
    aruco_params_.marker_num = this->get_parameter("aruco_params.marker_num").as_int();
    auto order = this->get_parameter("aruco_params.marker_order").as_integer_array();
    aruco_params_.marker_order.assign(order.begin(), order.end());   // 自动转型
    aruco_params_.export_yolo_data = static_cast<bool>(this->get_parameter("export_yolo_data").as_int());
    aruco_params_.camera_id_list = camera_id_list_;    // long int转int
    aruco_params_.camera_name_list = camera_name_list_;

    // 创建log文件及初始化
    logger_ = new SPD_LOG(cache_directory_ + "/log");  //必须在参数读取后，创建log文件夹

    // 打印参数信息
    spdlog::info("参数已初始化:");
    spdlog::info("  PCD文件路径: {}", data_path_);
    spdlog::info("  模板图像路径: {}", template_image_path_);
    spdlog::info("  外参文件路径: {}", parameters_path_);
    spdlog::info("  发布频率: {:.1f} Hz", publish_frequency_);
    spdlog::info("  点云话题: {}", pointcloud_topic_);
    spdlog::info("  目标点云话题: {}", pointcloud_target_topic_);
    spdlog::info("  BBox话题: {}", bbox_topic_);
    spdlog::info("  坐标系: {}", frame_id_);
    spdlog::info("  算法参数:");
    spdlog::info("    min_point: [{:.1f}, {:.1f}, {:.1f}, {:.1f}]", 
                algorithm_params_.min_point[0], algorithm_params_.min_point[1], 
                algorithm_params_.min_point[2], algorithm_params_.min_point[3]);
    spdlog::info("    max_point: [{:.1f}, {:.1f}, {:.1f}, {:.1f}]", 
                algorithm_params_.max_point[0], algorithm_params_.max_point[1], 
                algorithm_params_.max_point[2], algorithm_params_.max_point[3]);
    spdlog::info("    min_size: {:.1f} m", algorithm_params_.min_size);
    spdlog::info("    max_size: {:.1f} m", algorithm_params_.max_size);
    spdlog::info("    aspect_ratio_threshold: {:.1f}", algorithm_params_.aspect_ratio_threshold);
}

void RclComm::init_tf2_map()
{
    sensor_extrinsics_map["lidar_fl_base_link"] = SensorExtrinsics();
    sensor_extrinsics_map["lidar_fr_base_link"] = SensorExtrinsics();
    sensor_extrinsics_map["lidar_ft_base_link"] = SensorExtrinsics();
    sensor_extrinsics_map["lidar_rear_base_link"] = SensorExtrinsics();
    sensor_extrinsics_map["lidar_rt_base_link"] = SensorExtrinsics();
    sensor_extrinsics_map["gnss_link"] = SensorExtrinsics();
    sensor_extrinsics_map["sensor_kit_base_link"] = SensorExtrinsics();

    // 读取YAML文件并只更新已定义的key
    std::string yaml_path = parameters_path_ + "/extrinsic_parameters/sensor_kit_calibration.yaml";
    YAML::Node config = read_yaml_file(yaml_path);
    if (!config["sensor_kit_base_link"]) {
        spdlog::error("YAML文件缺少sensor_kit_base_link节点: {}", yaml_path);
        return;
    }
    const YAML::Node& sensors = config["sensor_kit_base_link"];
    for (auto& kv : sensor_extrinsics_map) {
        const std::string& key = kv.first;
        if (sensors[key]) {
            const YAML::Node& val = sensors[key];
            kv.second.father_frame_id  = "sensor_kit_base_link";
            kv.second.x = val["x"].as<double>();
            kv.second.y = val["y"].as<double>();
            kv.second.z = val["z"].as<double>();
            kv.second.roll = val["roll"].as<double>();
            kv.second.pitch = val["pitch"].as<double>();
            kv.second.yaw = val["yaw"].as<double>();
        }
    }

    yaml_path = parameters_path_ + "/extrinsic_parameters/sensors_calibration.yaml";
    YAML::Node config_ = read_yaml_file(yaml_path);
    if (!config_["base_link"]) {
        spdlog::error("YAML文件缺少base_link节点: {}", yaml_path);
        return;
    }
    const YAML::Node& sensors_ = config_["base_link"];
    if(sensors_["sensor_kit_base_link"]) {
        const YAML::Node& val = sensors_["sensor_kit_base_link"];
        sensor_extrinsics_map["sensor_kit_base_link"].father_frame_id  = "base_link";
        sensor_extrinsics_map["sensor_kit_base_link"].x = val["x"].as<double>();
        sensor_extrinsics_map["sensor_kit_base_link"].y = val["y"].as<double>();
        sensor_extrinsics_map["sensor_kit_base_link"].z = val["z"].as<double>();
        sensor_extrinsics_map["sensor_kit_base_link"].roll = val["roll"].as<double>();
        sensor_extrinsics_map["sensor_kit_base_link"].pitch = val["pitch"].as<double>();
        sensor_extrinsics_map["sensor_kit_base_link"].yaw = val["yaw"].as<double>();
        
    }
    // 计算front lidar到sensor_kit_base_link的变换矩阵
    compute_base_to_sensor_kit_transform();
}

YAML::Node RclComm::read_yaml_file(const std::string& file_path)
{
    try
    {
        YAML::Node config = YAML::LoadFile(file_path);
        return config;
    }
    catch (const YAML::BadFile& e)
    {
        spdlog::error("无法打开YAML文件: {}，错误: {}", file_path, e.what());
        return YAML::Node();
    }
    catch (const std::exception& e)
    {
        spdlog::error("读取YAML文件时发生异常: {}", e.what());
        return YAML::Node();
    }
}


void RclComm::publish_sensor_tf2()
{
    if (!static_tf_broadcaster_)
    {
        RCLCPP_ERROR(this->get_logger(), "Static TF broadcaster not initialized");
        return;
    }

    std::vector<geometry_msgs::msg::TransformStamped> static_transforms;

    for (const auto& kv : sensor_extrinsics_map)
    {
        const std::string& child_frame = kv.first;
        const std::string& parent_frame = kv.second.father_frame_id;
        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header.stamp = this->get_clock()->now();
        tf_msg.header.frame_id = parent_frame;
        tf_msg.child_frame_id = child_frame;
        tf_msg.transform.translation.x = kv.second.x;
        tf_msg.transform.translation.y = kv.second.y;
        tf_msg.transform.translation.z = kv.second.z;
        tf2::Quaternion q;
        q.setRPY(kv.second.roll, kv.second.pitch, kv.second.yaw);
        tf_msg.transform.rotation.x = q.x();
        tf_msg.transform.rotation.y = q.y();
        tf_msg.transform.rotation.z = q.z();
        tf_msg.transform.rotation.w = q.w();
        static_transforms.push_back(tf_msg);
    }
    
    if (!static_transforms.empty())
    {
        static_tf_broadcaster_->sendTransform(static_transforms);
    }
}

void RclComm::compute_base_to_sensor_kit_transform()
{
    // 检查是否有lidar_ft_base_link的外参信息
    if (sensor_extrinsics_map.find("lidar_ft_base_link") == sensor_extrinsics_map.end())
    {
        RCLCPP_WARN(this->get_logger(), "未找到lidar_ft_base_link的外参信息");
        algorithm_params_.has_transform = false;
        return;
    }
    
    const auto& extrinsics = sensor_extrinsics_map["lidar_ft_base_link"];
    
    // 创建旋转矩阵 (RPY -> Rotation Matrix)
    tf2::Quaternion q;
    q.setRPY(extrinsics.roll, extrinsics.pitch, extrinsics.yaw);
    
    // 构建4x4变换矩阵
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    
    // 设置旋转部分
    Eigen::Quaternionf eigen_q(q.w(), q.x(), q.y(), q.z());
    Eigen::Matrix3f rotation_matrix = eigen_q.toRotationMatrix();
    transform.block<3,3>(0,0) = rotation_matrix;
    
    // 设置平移部分
    transform(0,3) = static_cast<float>(extrinsics.x);
    transform(1,3) = static_cast<float>(extrinsics.y);
    transform(2,3) = static_cast<float>(extrinsics.z);
    
    // 保存到算法参数中
    algorithm_params_.transform_base_to_sensor_kit = transform; // 得到lidar_ft_base_link到sensor_kit_base_link的变换
    algorithm_params_.has_transform = true;

    spdlog::info("计算lidar_ft_base_link到sensor_kit_base_link变换矩阵成功");
    spdlog::info("变换矩阵为:");
    spdlog::info("    | {:.3f} {:.3f} {:.3f} {:.3f} |", transform(0,0), transform(0,1), transform(0,2), transform(0,3));
    spdlog::info("    | {:.3f} {:.3f} {:.3f} {:.3f} |", transform(1,0), transform(1,1), transform(1,2), transform(1,3));
    spdlog::info("    | {:.3f} {:.3f} {:.3f} {:.3f} |", transform(2,0), transform(2,1), transform(2,2), transform(2,3));
    spdlog::info("    | {:.3f} {:.3f} {:.3f} {:.3f} |", transform(3,0), transform(3,1), transform(3,2), transform(3,3));
}

