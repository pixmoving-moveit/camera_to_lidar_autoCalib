#ifndef EXTRABOARD_H
#define EXTRABOARD_H

// 图像配置宏定义
#define IMAGE_SIZE 1000    // 图像尺寸 (像素)
#define PIXEL_TO_MM 2.0f   // 像素到毫米的比例 (1像素 = PIXEL_TO_MM毫米)

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/centroid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/common/common.h>
#include <string>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <cmath>
#include "base_struct.h"

struct BBox
{
    Eigen::Vector3f size; // 尺寸 (长宽高)
    Eigen::Vector3f position; // 位置 (中心点)
    Eigen::Quaternionf orientation; // 方向 (四元数)
    BBox() : size(0,0,0), position(0,0,0), orientation(1,0,0,0) {}
};

struct AlgorithmParams
{
    Eigen::Vector4f min_point; // 点云过滤最小范围
    Eigen::Vector4f max_point; // 点云过滤最大范围
    float min_size; // 最小尺寸（米）
    float max_size; // 最大尺寸（米）
    float aspect_ratio_threshold; // 长宽比阈值
    float bracket_width; // 支架过滤宽度（米）
    
    // base_link到sensor_kit_base_link的变换矩阵
    Eigen::Matrix4f transform_base_to_sensor_kit; // 4x4变换矩阵
    bool has_transform; // 是否有有效的变换矩阵
    
    AlgorithmParams() : 
        min_point(8.3, -4.8, -0.5, 1.0),
        max_point(16.5, 4.6, 3.4, 1.0),
        min_size(0.5f),
        max_size(1.1f),
        aspect_ratio_threshold(1.2f),
        transform_base_to_sensor_kit(Eigen::Matrix4f::Identity()),
        has_transform(false) {}
};

struct IPTransInfo
{
    Eigen::Vector4f centroid_; // 质心
    Eigen::Matrix3f rotation_matrix_; // 旋转矩阵
    Eigen::Matrix3f inverse_rotation_; // 逆旋转矩阵
    bool transform_valid_; // 变换矩阵是否有效
    IPTransInfo() : centroid_(Eigen::Vector4f::Zero()), rotation_matrix_(Eigen::Matrix3f::Identity()), inverse_rotation_(Eigen::Matrix3f::Identity()), transform_valid_(false) {}
};

class ExtraBoard
{
public:
    ExtraBoard(const std::string& cache_directory);
    ~ExtraBoard();

    // 读取PCD文件
    bool load_pcd_file(const std::string& file_path);
    
    // 获取点云数据
    PointCloudT::Ptr get_pointcloud() const;
    
    // 获取ROS2点云消息
    sensor_msgs::msg::PointCloud2 get_ros_pointcloud(const std::string& frame_id = "base_link") const;


    std::vector<BBox> bboxes;   // 外部访问过滤后的包围盒信息
    std::vector<BoardInfo> boards; // 外部访问识别到的棋盘信息
    std::vector<IPTransInfo> img_pc_trans_info; // 图像与点云的变换信息

    
    // 点云处理功能
    void board_register(std::vector<PointCloudT>& board_pointclouds, std::vector<BoardInfo>& boards, const int& check_result_);
    void extract_points(std::vector<PointCloudT>& extracted_clouds);
    
    // 设置算法参数
    void set_algorithm_params(const AlgorithmParams& params);
    
    // 重置为原始点云
    void reset_to_original();
    
    // 获取点云统计信息
    size_t get_point_count() const;
    bool is_empty() const;
    
    // 显示控制方法
    void start_display_thread();
    void stop_display_thread();
    
    // 坐标变换方法
    bool image_to_world(int img_x, int img_y, float& world_x, float& world_y, float& world_z, const IPTransInfo& trans_info) const;
    bool world_to_image(float world_x, float world_y, float world_z, int& img_x, int& img_y, const IPTransInfo& trans_info) const;

    // 图像配准功能
    bool load_template_image(const std::string& png_file_path);
    Board2DCorners register_template_with_projection(cv::Mat& protected_image);
    void fastTemplateMatchY(const cv::Mat& template_gray,
                        const cv::Mat& projection_gray,
                        double&         best_angle_y,   // 输出：与 Y 轴夹角
                        cv::Point&     best_match_location,
                        double&        best_match_value,
                        double&        scale_value);

    cv::Mat get_template_image() const;
    cv::Mat get_binary_template() const;
    
    // 检查标定相机列表是否符合前后分组要求
    int check_orientation(const std::vector<int>& calib_camera_list,
                                  const std::vector<int>& camera_id,
                                  const std::vector<std::string>& camera_name);

private:
    std::string cache_directory_;
    PointCloudT::Ptr original_cloud_;    // 原始点云
    PointCloudT::Ptr processed_cloud_;   // 处理后的点云
    
    // 算法参数
    AlgorithmParams algorithm_params_;   // 算法参数
    
    // 图像显示相关
    cv::Mat projection_image_;           // 投影图像
    std::mutex image_mutex_;             // 图像访问互斥锁
    std::atomic<bool> display_running_;  // 显示线程运行标志
    std::thread display_thread_;         // 显示线程
    std::atomic<bool> image_updated_;    // 图像更新标志
    
    // 图像配准相关
    cv::Mat template_image_;             // 模板图像（原始）
    cv::Mat binary_template_;            // 二值化模板图像
    cv::Point2f registration_offset_;    // 配准偏移量
    float registration_angle_;           // 配准旋转角度
    bool template_loaded_;               // 模板是否已加载
    
    // 内部处理函数
    PointCloudT::Ptr merge_pointclouds(const std::vector<PointCloudT::Ptr>& clouds);
    void copy_cloud_to_processed(const PointCloudT::Ptr cloud);
    void display_loop();                 // 显示线程主循环
    
    // 点云处理辅助函数
    bool apply_voxel_filter(const PointCloudT::Ptr& input_cloud, float voxel_size, PointCloudT::Ptr& output_cloud);

    bool apply_box_filter(const PointCloudT::Ptr& input_cloud, PointCloudT::Ptr& output_cloud,
                         const Eigen::Vector4f& min_point, const Eigen::Vector4f& max_point);
    bool apply_outlier_removal(const PointCloudT::Ptr& input_cloud, PointCloudT::Ptr& output_cloud,
                              int mean_k = 10, double std_dev_mul_thresh = 1.5);
    
    bool apply_scaffold_filter(const std::vector<PointCloudT::Ptr>& input_cloud, std::vector<PointCloudT::Ptr>& output_cloud);

    bool apply_euclidean_clustering(const PointCloudT::Ptr& input_cloud, 
                                   std::vector<PointCloudT::Ptr>& cluster_clouds,
                                   double cluster_tolerance = 0.1, int min_cluster_size = 100, 
                                   int max_cluster_size = 1e5);
    bool apply_euclidean_filter(PointCloudT::Ptr& cloud,
                                        double cluster_tolerance, int min_cluster_size,
                                        int max_cluster_size);
    bool apply_ransac_plane_segmentation(const PointCloudT::Ptr& input_cloud, 
                                        PointCloudT::Ptr& plane_cloud,
                                        pcl::ModelCoefficients::Ptr& coefficients,
                                        int max_iterations = 50, double distance_threshold = 0.015);
    bool compute_bounding_box(const PointCloudT::Ptr& input_cloud, float voxel_size, BBox& box);

    bool apply_intensity_filter(const PointCloudT::Ptr& input_cloud, PointCloudT::Ptr& output_cloud,
                                 const float min_intensity, const float max_intensity);

    inline float dist_x_y(const PointT &p1, const PointT &p2)
    {
        return std::hypot(p1.x - p2.x, p1.y - p2.y);
    }
};

#endif // EXTRABOARD_H
