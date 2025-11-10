#define CVUI_IMPLEMENTATION
#define WINDOW_NAME "CVUI Calibration Tool!"
/**
 * @brief 提取点云中的主平面并将其投影到YOZ平面的二维图像
 *
 * 此函数主要完成以下步骤：
 * 1. 使用RANSAC算法从输入点云(processed_cloud_)中分割出主平面，并获取平面方程系数与内点索引。
 * 2. 提取平面上的点云，并计算其质心。
 * 3. 计算平面法向量，并构建旋转矩阵，使平面法向量对齐到X轴（YOZ平面）。
 * 4. 将平面点云以质心为中心进行坐标变换，并投影到YOZ平面。
 * 5. 将投影后的点映射到二维图像（projection_image_），图像中心对应点云质心。
 *    - 投影点通过cv::circle绘制到图像上，圆的半径为3像素，最后一个参数"-1"表示填充圆。
 *    - 所有存在点云的位置绘制为白色圆点，其余位置保持黑色背景。
 * 6. 线程安全地更新投影图像，并将处理后的点云更新为平面点云。
 *
 * 注意事项：
 * - 圆的半径由cv::circle的第三个参数（3）指定，表示每个点在图像上的圆形半径为3像素。
 * - 圆的粗细由cv::circle的第四个参数（-1）指定，-1表示填充圆，即圆的粗细为填充整个圆。
 * - 图像坐标系Z轴方向与点云坐标系相反，因此Z坐标需要翻转。
 * - 投影到YOZ平面：Y轴映射到图像X轴，Z轴映射到图像Y轴（翻转）。
 * - 仅当点云不为空且成功分割出平面时才会进行投影操作。
 */

#include "extraboard.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cvui.h>
#include <cmath>
#include <limits>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/common/transforms.h>
#include <filesystem>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

ExtraBoard::ExtraBoard(const std::string& cache_directory) : display_running_(false), image_updated_(false), template_loaded_(false), cache_directory_(cache_directory)
{
    // 自动创建缓存目录
    std::filesystem::create_directories(cache_directory_);
    // 初始化点云指针
    original_cloud_.reset(new PointCloudT());
    processed_cloud_.reset(new PointCloudT());
    
    // 初始化图像
    projection_image_ = cv::Mat::zeros(IMAGE_SIZE, IMAGE_SIZE, CV_8UC3);
    
    
    // 初始化配准参数
    registration_offset_ = cv::Point2f(0, 0);
    registration_angle_ = 0.0f;

    // start_display_thread();  //暂时不用显示了
    
    spdlog::info("ExtraBoard初始化完成");
}

ExtraBoard::~ExtraBoard()
{
    stop_display_thread();
    spdlog::info("ExtraBoard已销毁");
}

void ExtraBoard::set_algorithm_params(const AlgorithmParams& params)
{
    algorithm_params_ = params;
}

bool ExtraBoard::load_pcd_file(const std::string& file_path)
{
    // 清空当前点云
    original_cloud_->clear();

    PointCloudT::Ptr temp_cloud(new PointCloudT);
    
    // 读取PCD文件
    if (pcl::io::loadPCDFile<PointT>(file_path, *temp_cloud) == -1)
    {
        spdlog::error("无法读取PCD文件: {}", file_path);
        return false;
    }

    // 应用base_link到sensor_kit_base_link的变换
    if (algorithm_params_.has_transform && false)
    {
        // 使用PCL变换点云从base_link到sensor_kit_base_link坐标系
        pcl::transformPointCloud(*temp_cloud, *temp_cloud, algorithm_params_.transform_base_to_sensor_kit);
        spdlog::info("已将点云从base_link变换到sensor_kit_base_link坐标系");
    }
    else
    {
        spdlog::warn("未设置base_link到sensor_kit_base_link的变换，点云保持原始坐标系");
    }
    
    // 将变换后的点云复制到原始点云和处理点云
    *original_cloud_ = *temp_cloud;

    spdlog::info("成功加载PCD文件: {}", file_path);
    spdlog::info("点云包含 {} 个点", temp_cloud->size());

    return true;
}

PointCloudT::Ptr ExtraBoard::get_pointcloud() const
{
    return processed_cloud_;
}

sensor_msgs::msg::PointCloud2 ExtraBoard::get_ros_pointcloud(const std::string& frame_id) const
{
    sensor_msgs::msg::PointCloud2 ros_cloud;
    
    if (!processed_cloud_->empty())
    {
        pcl::toROSMsg(*processed_cloud_, ros_cloud);
        ros_cloud.header.frame_id = frame_id;
    }
    
    return ros_cloud;
}

void ExtraBoard::extract_points(std::vector<PointCloudT>& extracted_clouds)
{
    extracted_clouds.clear();
    
    if (original_cloud_->empty())
    {
        spdlog::error("原始点云为空");
        return;
    }

    spdlog::info("开始点云提取处理，原始点云包含 {} 个点", original_cloud_->size());

    // 1. 使用BOX滤波器提取点云
    spdlog::info("Step1: 使用BOX滤波器提取点云, 去除无效点");
    PointCloudT::Ptr box_filtered_cloud(new PointCloudT);
    if (!apply_box_filter(original_cloud_, box_filtered_cloud, algorithm_params_.min_point, algorithm_params_.max_point))
    {
        spdlog::error("BOX滤波失败");
        return;
    }

    // 1.1 去拖影导致的无效点
    PointCloudT::Ptr no_nan_cloud(new PointCloudT);
    if(!apply_intensity_filter(box_filtered_cloud, no_nan_cloud, 0.0f, 200.0f))
    {
        spdlog::error("去除无效点失败");
        return;
    }


    // 2. 滤除离群点
    spdlog::info("Step2: 滤除离群点");
    PointCloudT::Ptr filtered_cloud(new PointCloudT);
    if (!apply_outlier_removal(no_nan_cloud, filtered_cloud, 5, 1.5))
    {
        spdlog::error("离群点滤除失败");
        return;
    }

    // 3. 使用欧式聚类分割点云
    spdlog::info("Step3: 使用欧式聚类分割点云");
    std::vector<PointCloudT::Ptr> clustered_clouds;
    if (!apply_euclidean_clustering(filtered_cloud, clustered_clouds, 0.1, 5e3, 1e6))
    {
        spdlog::error("欧式聚类失败");
    }

    // 4. 对处理后的点云进行RANSAC平面分割,提取平面内点
    spdlog::info("Step4: 对处理后的点云进行RANSAC平面分割,提取平面内点");
    std::vector<PointCloudT::Ptr> ransac_filtered_clouds;
    for(size_t i = 0; i < clustered_clouds.size(); ++i)
    {
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        PointCloudT::Ptr plane_cloud(new PointCloudT);
        
        if (apply_ransac_plane_segmentation(clustered_clouds[i], plane_cloud, coefficients, 50, 0.025))
        {
            ransac_filtered_clouds.push_back(plane_cloud);
            // std::cout << "聚类 " << i << " 平面分割完成，提取到平面点云包含 " << plane_cloud->size() << " 个点" << std::endl;
        }
    }
    spdlog::info("RANSAC平面分割完成，提取到 {} 个平面", ransac_filtered_clouds.size());


    //这里需要对点云进行微分滤波，去除标定板支架等细小结构
    std::vector<PointCloudT::Ptr> scaffold_filtered_clouds;
    apply_scaffold_filter(ransac_filtered_clouds, scaffold_filtered_clouds);


    processed_cloud_ = merge_pointclouds(scaffold_filtered_clouds);
    // 保存处理后的点云为PCD文件
    std::string save_path = cache_directory_ + "/processed_cloud.pcd";
    if (pcl::io::savePCDFileBinary(save_path, *processed_cloud_) == 0) {
        spdlog::info("处理后的点云已保存为: {}", save_path);
    } else {
        spdlog::error("保存处理后的点云失败: {}", save_path);
    }

    //5. 进行欧式聚类过滤，滤除混在平面中的杂点
    spdlog::info("Step5: 进行欧式聚类过滤，滤除混在平面中的杂点");
    for(size_t i = 0; i < scaffold_filtered_clouds.size(); ++i)
    {
        if (!apply_euclidean_filter(scaffold_filtered_clouds[i], 0.1, 5e3, 1e6))
        {
            spdlog::error("欧式聚类过滤失败");
        }
    }


    // 6. 计算点云尺寸，进行尺寸过滤，只保留尺寸符合要求的点云
    spdlog::info("Step6: 计算点云尺寸，进行尺寸过滤，只保留尺寸符合要求的点云");
    std::vector<PointCloudT::Ptr> size_filtered_clouds;
    bboxes.clear();
    for (size_t i = 0; i < scaffold_filtered_clouds.size(); ++i)
    {
        BBox bbox;
        if (compute_bounding_box(scaffold_filtered_clouds[i], 0.05, bbox))
        {
            // 使用配置的尺寸范围
            std::vector<float> sizes = {bbox.size.x(), bbox.size.y(), bbox.size.z()};
            std::sort(sizes.begin(), sizes.end(), std::greater<float>());

            if ((sizes[0] > algorithm_params_.max_size || sizes[1] < algorithm_params_.min_size) || 
                sizes[0]/sizes[1] < algorithm_params_.aspect_ratio_threshold)
            {
                spdlog::info("平面 {} 尺寸过滤未通过: {:.2f} x {:.2f} x {:.2f} 米", i, sizes[0], sizes[1], sizes[2]);
                continue;
            }
            else
            {
                spdlog::info("平面 {} 尺寸正常: {:.2f} x {:.2f} x {:.2f} 米", i, sizes[0], sizes[1], sizes[2]);
            }
            size_filtered_clouds.push_back(scaffold_filtered_clouds[i]);
            bboxes.push_back(bbox);
        }
    }

    // 7. 将所有处理后的点云添加到结果中
    for(const auto& cloud_ptr : size_filtered_clouds)
    {
        if (cloud_ptr && !cloud_ptr->empty())
        {
            extracted_clouds.push_back(*cloud_ptr);  // 解引用智能指针
        }
    }

    spdlog::info("点云提取处理完成，共生成 {} 簇点云", extracted_clouds.size());
}

/**
 * @brief 提取点云中的主平面并将其投影到二维图像
 *
 * 此函数主要完成以下步骤：
 * 1. 使用RANSAC算法从输入点云(processed_cloud_)中分割出主平面，并获取平面方程系数与内点索引。
 * 2. 提取平面上的点云，并计算其质心。
 * 3. 计算平面法向量，并构建旋转矩阵，使平面法向量对齐到Z轴（XOY平面）。
 * 4. 将平面点云以质心为中心进行坐标变换，并投影到XOY平面。
 * 5. 将投影后的点映射到二维图像（projection_image_），图像中心对应点云质心。
 *    - 投影点通过cv::circle绘制到图像上，圆的半径为1像素，最后一个参数“-1”表示填充圆（粗细为填充）。
 *    - intensity用于设置圆的颜色（灰度值）。
 * 6. 线程安全地更新投影图像，并将处理后的点云更新为平面点云。
 *
 * 注意事项：
 * - 圆的半径由cv::circle的第三个参数（1）指定，表示每个点在图像上的圆形半径为1像素。
 * - 圆的粗细由cv::circle的第四个参数（-1）指定，-1表示填充圆，即圆的粗细为填充整个圆。
 * - 图像坐标系Y轴方向与点云坐标系相反，因此Y坐标需要翻转。
 * - 仅当点云不为空且成功分割出平面时才会进行投影操作。
 */
void ExtraBoard::board_register(std::vector<PointCloudT>& board_pointclouds, std::vector<BoardInfo>& boards, const int& check_result_)
{
    if (board_pointclouds.size() < 1)
    {
        spdlog::warn("点云为空");
        return;
    }

    spdlog::info("Step7: 对每簇点云进行平面提取和投影到图像");

    //0.1 初始化配准图像
    std::vector<cv::Mat> registered_images;
    registered_images.resize(board_pointclouds.size());
    for(auto& img : registered_images)  img = cv::Mat::zeros(IMAGE_SIZE, IMAGE_SIZE, CV_8UC3);

    //0.2 初始化标定板数据
    boards.resize(board_pointclouds.size());
    for(auto& board : boards)  board = BoardInfo();

    //0.3 初始化变换矩阵
    img_pc_trans_info.clear();
    img_pc_trans_info.resize(board_pointclouds.size());
    for(auto& info : img_pc_trans_info) info = IPTransInfo();

    spdlog::info("开始进行各个点云的平面提取和投影...");
    for(size_t idx=0; idx < board_pointclouds.size(); ++idx)
    {
        PointCloudT::Ptr cloud_ptr(new PointCloudT);
        cloud_ptr = std::make_shared<PointCloudT>(board_pointclouds[idx]);

        if (cloud_ptr->empty())
        {
            spdlog::warn("点云为空");
            return;
        }

        // 1. RANSAC平面分割
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        PointCloudT::Ptr plane_cloud(new PointCloudT);
        if (!apply_ransac_plane_segmentation(cloud_ptr, plane_cloud, coefficients, 50, 0.025))
        {
            spdlog::error("未能找到平面");
            return;
        }
        
        // 3. 计算点云质心
        pcl::compute3DCentroid(*plane_cloud, img_pc_trans_info.at(idx).centroid_);
        spdlog::info("点云质心: ({:.2f}, {:.2f}, {:.2f})", img_pc_trans_info.at(idx).centroid_[0], img_pc_trans_info.at(idx).centroid_[1], img_pc_trans_info.at(idx).centroid_[2]);

        // 4. 获取平面法向量并归一化
        Eigen::Vector3f plane_normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);

        plane_normal.normalize();
        spdlog::info("平面法向量: ({:.2f}, {:.2f}, {:.2f})", plane_normal.x(), plane_normal.y(), plane_normal.z());

        //在XOY平面上，如果平面法向量由原点向外，则d<0；如果平面法向量由外向原点，则d>0
        float d = coefficients->values[3];
        if(d > 0) plane_normal = -plane_normal; //确保法向量由原点向外，以保证投影方向正确
        spdlog::info("方向判据：{} 法向量修正：{},{},{}", d, plane_normal.x(), plane_normal.y(), plane_normal.z());

        // 5. 构建坐标变换矩阵，将平面变换到YOZ平面
        Eigen::Matrix3f rotation_matrix = Eigen::Matrix3f::Identity();
        
        Eigen::Vector3f x_axis(1, 0, 0);

        if(1 == check_result_)
        {
            spdlog::info("当前待标定相机id隶属 前 雷达点云， 执行方案1");
            // 目标：平面法向量对齐到X轴 (1, 0, 0)    
            // 计算旋转轴（叉积）
            Eigen::Vector3f rotation_axis = plane_normal.cross(x_axis);
            rotation_axis.normalize();
            
            // 计算旋转角度（点积）
            float cos_angle = plane_normal.dot(x_axis);
            float rotation_angle = std::acos(std::clamp(cos_angle, -1.0f, 1.0f));
            
            // 构建旋转矩阵（Rodrigues公式）
            if (rotation_angle > 1e-6) // 避免数值误差
            {
                Eigen::Matrix3f K; // 反对称矩阵
                K << 0, -rotation_axis.z(), rotation_axis.y(),
                    rotation_axis.z(), 0, -rotation_axis.x(),
                    -rotation_axis.y(), rotation_axis.x(), 0;
                
                rotation_matrix = Eigen::Matrix3f::Identity() + 
                                std::sin(rotation_angle) * K + 
                                (1 - std::cos(rotation_angle)) * K * K;
            }
        }
        else if(2 == check_result_)
        {
            spdlog::info("当前待标定相机id隶属 后 雷达点云， 执行方案2");
            Eigen::Vector3f rot_axis = plane_normal.cross(x_axis).normalized();
            float cos_ang = plane_normal.dot(x_axis);
            float ang = std::acos(std::clamp(cos_ang,-1.f,1.f));
            Eigen::Matrix3f R1 = Eigen::AngleAxisf(ang, rot_axis).toRotationMatrix();

            // 2. 再锁“绕 X 轴的滚转”：把旧 Y 轴转到目标 Y 方向
            Eigen::Vector3f old_y(0,-1,0);
            Eigen::Vector3f new_y = R1 * old_y;          // 当前已经转到的方向
            // 我们希望 new_y 落在 YOZ 平面，即与 (0,-1,0) 夹角最小
            float roll = std::atan2(new_y.z(), new_y.y());
            Eigen::Matrix3f R2 = Eigen::AngleAxisf(-roll, x_axis).toRotationMatrix();

            rotation_matrix = R2 * R1;   // 最终矩阵
        }
        


        // 保存变换矩阵用于后续的坐标映射
        img_pc_trans_info.at(idx).rotation_matrix_ = rotation_matrix;
        img_pc_trans_info.at(idx).inverse_rotation_ = rotation_matrix.transpose(); // 旋转矩阵的逆等于转置
        img_pc_trans_info.at(idx).transform_valid_ = true;

        // 6. 创建投影图像 (使用宏定义的尺寸)
        const int image_size = IMAGE_SIZE;
        const float pixel_to_mm = PIXEL_TO_MM; // 1像素 = PIXEL_TO_MM 毫米
        
        // 图像中心坐标
        const int center_x = image_size / 2;
        const int center_y = image_size / 2;
        
        // 7. 将点云投影到YOZ平面，然后映射到图像
        int projected_points_ = 0;
        for (const auto& point : plane_cloud->points)
        {
            // 7.1 将点相对于质心平移
            Eigen::Vector3f point_centered(point.x - img_pc_trans_info.at(idx).centroid_[0], 
                                        point.y - img_pc_trans_info.at(idx).centroid_[1], 
                                        point.z - img_pc_trans_info.at(idx).centroid_[2]);
            
            // 7.2 应用旋转变换，将点投影到YOZ平面
            Eigen::Vector3f point_transformed = img_pc_trans_info.at(idx).rotation_matrix_ * point_centered;
            
            // 7.3 取Y和Z坐标作为2D投影坐标（忽略X坐标）
            float proj_y = point_transformed.y(); // 单位：米
            float proj_z = point_transformed.z(); // 单位：米
            
            // 7.4 将投影坐标转换为图像坐标
            // 坐标系变换：3D坐标系 -> 图像坐标系
            // Y轴（左/右） -> 图像X轴（右）
            // Z轴（上/下） -> 图像Y轴（下，需要翻转）
            // 映射关系：1像素 = PIXEL_TO_MM 毫米
            // proj_y(米) * 1000 = 毫米，然后除以PIXEL_TO_MM得到像素数
            int img_x = center_x + static_cast<int>(proj_y * 1000.0f / pixel_to_mm);
            int img_y = center_y - static_cast<int>(proj_z * 1000.0f / pixel_to_mm); // 注意Z轴翻转
            
            // 7.5 检查是否在图像范围内
            if (img_x >= 0 && img_x < image_size && img_y >= 0 && img_y < image_size)
            {
                // 存在点云的位置绘制为白色圆点
                cv::circle(registered_images.at(idx), cv::Point(img_x, img_y), 3, cv::Scalar(255, 255, 255), -1);
                projected_points_++;
            }

        }
        Board2DCorners board_2d_corners = register_template_with_projection(registered_images.at(idx));

        boards.at(idx).corners_2d = board_2d_corners;

        auto& display_image = registered_images.at(idx);
        // 添加UI元素
        cvui::beginRow(display_image, 10, 10, -1, -1, 6);
        cvui::text("Planar projection result");
        cvui::endRow();
        
        cvui::beginRow(display_image, 10, 40, -1, -1, 6);
        cvui::printf("Projection points: %d", projected_points_);
        cvui::endRow();
        
        cvui::beginRow(display_image, 10, 70, -1, -1, 6);
        cvui::printf("Centroid: (%.3f, %.3f, %.0.010,3f)", img_pc_trans_info.at(idx).centroid_[0], img_pc_trans_info.at(idx).centroid_[1], img_pc_trans_info.at(idx).centroid_[2]);
        cvui::endRow();
        
        cvui::beginRow(display_image, 10, 100, -1, -1, 6);
        cvui::printf("Pixel ratio: 1 pixel = %.1fmm", PIXEL_TO_MM);
        cvui::endRow();
        
        // 在图像中心绘制十字线标记质心位置
        cv::line(display_image, cv::Point(center_x - 20, center_y), 
                cv::Point(center_x + 20, center_y), cv::Scalar(0, 0, 255), 2);
        cv::line(display_image, cv::Point(center_x, center_y - 20), 
                cv::Point(center_x, center_y + 20), cv::Scalar(0, 0, 255), 2);
        
        // 保存当前投影图像到本地
        std::string filename = cache_directory_ + "/" + cv::format("projection_result_%zu.png", idx);
        cv::imwrite(filename, registered_images.at(idx));
        spdlog::info("成功投影 {} 个点到图像", projected_points_);

    }
    
    for(size_t idx=0; idx < board_pointclouds.size(); ++idx)
    {
        if (!boards.at(idx).corners_2d.is_valid) continue; //跳过无效的标定板信息

        PointT point;

        image_to_world(boards.at(idx).corners_2d.corn_left_top.x, boards.at(idx).corners_2d.corn_left_top.y, point.x, point.y, point.z, img_pc_trans_info.at(idx));
        boards.at(idx).corners_cloud.corn_left_top_3d = point;
        // std::cout<<"corn_left_top_3d: ("<< boards.at(idx).corners_cloud.corn_left_top_3d.x << ", " << boards.at(idx).corners_cloud.corn_left_top_3d.y << ", " << boards.at(idx).corners_cloud.corn_left_top_3d.z << ")" << std::endl;

        image_to_world(boards.at(idx).corners_2d.corn_right_top.x, boards.at(idx).corners_2d.corn_right_top.y, point.x, point.y, point.z, img_pc_trans_info.at(idx));
        boards.at(idx).corners_cloud.corn_right_top_3d = point;
        // std::cout<<"corn_right_top_3d: ("<< boards.at(idx).corners_cloud.corn_right_top_3d.x << ", " << boards.at(idx).corners_cloud.corn_right_top_3d.y << ", " << boards.at(idx).corners_cloud.corn_right_top_3d.z << ")" << std::endl;

        image_to_world(boards.at(idx).corners_2d.corn_right_bottom.x, boards.at(idx).corners_2d.corn_right_bottom.y, point.x, point.y, point.z, img_pc_trans_info.at(idx));
        boards.at(idx).corners_cloud.corn_right_bottom_3d = point;
        // std::cout<<"corn_right_bottom_3d: ("<< boards.at(idx).corners_cloud.corn_right_bottom_3d.x << ", " << boards.at(idx).corners_cloud.corn_right_bottom_3d.y << ", " << boards.at(idx).corners_cloud.corn_right_bottom_3d.z << ")" << std::endl;

        image_to_world(boards.at(idx).corners_2d.corn_left_bottom.x, boards.at(idx).corners_2d.corn_left_bottom.y, point.x, point.y, point.z, img_pc_trans_info.at(idx));
        boards.at(idx).corners_cloud.corn_left_bottom_3d = point;
        // std::cout<<"corn_left_bottom_3d: ("<< boards.at(idx).corners_cloud.corn_left_bottom_3d.x << ", " << boards.at(idx).corners_cloud.corn_left_bottom_3d.y << ", " << boards.at(idx).corners_cloud.corn_left_bottom_3d.z << ")" << std::endl;

        image_to_world(boards.at(idx).corners_2d.center.x, boards.at(idx).corners_2d.center.y, point.x, point.y, point.z, img_pc_trans_info.at(idx));
        boards.at(idx).corners_cloud.center_3d = point;
        // std::cout<<"center_3d: ("<< boards.at(idx).corners_cloud.center_3d.x << ", " << boards.at(idx).corners_cloud.center_3d.y << ", " << boards.at(idx).corners_cloud.center_3d.z << ")" << std::endl;
        boards.at(idx).corners_cloud.is_valid = true;
    }

    spdlog::info("-----------点云平面提取和投影完成-----------");
    
}

int ExtraBoard::check_orientation(const std::vector<int>& calib_camera_list,
                                  const std::vector<int>& camera_id,
                                  const std::vector<std::string>& camera_name)
{
    /* -------------- 1. 基本合法性检查 -------------- */
    if (camera_id.size() != camera_name.size()) {
        return -1;          // 长度不一致
    }

    /* -------------- 2. 建立 id->name 映射，并收集 calib_names -------------- */
    std::unordered_map<int, std::string> id2name;
    for (size_t i = 0; i < camera_id.size(); ++i) {
        id2name[camera_id[i]] = camera_name[i];
    }

    std::vector<std::string> calib_names;
    calib_names.reserve(calib_camera_list.size());

    for (int cid : calib_camera_list) {
        auto it = id2name.find(cid);
        if (it == id2name.end()) {
            return -2;      // 出现非法 id
        }
        calib_names.push_back(it->second);
    }

    /* -------------- 3. 新增：前后分组判断 -------------- */
    const std::unordered_set<std::string> front_set = {
        "camera_front_4k", "camera_front", "camera_rear_left", "camera_rear_right"
    };
    const std::unordered_set<std::string> rear_set = {
        "camera_rear", "camera_front_left", "camera_front_right"
    };

    bool all_in_front = true;
    bool all_in_rear  = true;

    for (const auto& name : calib_names) {
        if (front_set.find(name) == front_set.end()) all_in_front = false;
        if (rear_set .find(name) == rear_set.end())  all_in_rear  = false;
        if (!all_in_front && !all_in_rear) break;   // 提前短路
    }

    if (all_in_front && all_in_rear) return 0;      // 前后都能匹配
    if (all_in_front)                return 1;      // 仅 front 组
    if (all_in_rear)                 return 2;      // 仅 rear 组

    return -3;  // 都不满足，业务可自行定义错误码
}

void ExtraBoard::reset_to_original()
{
    copy_cloud_to_processed(original_cloud_);
    spdlog::info("已重置为原始点云");
}

size_t ExtraBoard::get_point_count() const
{
    return processed_cloud_->size();
}

bool ExtraBoard::is_empty() const
{
    return processed_cloud_->empty();
}

void ExtraBoard::copy_cloud_to_processed(const PointCloudT::Ptr cloud)

{
    if (!cloud->empty())
    {
        *processed_cloud_ = *cloud;
    }
}

void ExtraBoard::start_display_thread()
{
    if (!display_running_)
    {
        display_running_ = true;
        display_thread_ = std::thread(&ExtraBoard::display_loop, this);
        spdlog::info("显示线程已启动");
    }
}

void ExtraBoard::stop_display_thread()
{
    if (display_running_)
    {
        display_running_ = false;
        if (display_thread_.joinable())
        {
            display_thread_.join();
        }
        cv::destroyAllWindows();
        spdlog::info("显示线程已停止");
    }
}

void ExtraBoard::display_loop()
{
    // 初始化cvui
    cvui::init(WINDOW_NAME);
    
    while (display_running_)
    {
        cv::Mat display_image;
        
        // 线程安全地获取图像
        {
            std::lock_guard<std::mutex> lock(image_mutex_);
            if (image_updated_)
            {
                display_image = projection_image_.clone();
                image_updated_ = false;
            }
        }
        
        
        // 检查键盘输入和窗口事件
        int key = cv::waitKey(30); // 30ms延时
        if (key == 27) // ESC键退出
        {
            display_running_ = false;
        }
    }
}

bool ExtraBoard::image_to_world(int img_x, int img_y, float& world_x, float& world_y, float& world_z, const IPTransInfo& trans_info) const
{
    if (!trans_info.transform_valid_)
    {
        spdlog::error("变换矩阵无效，请先执行board_register()");
        return false;
    }
    
    const int image_size = IMAGE_SIZE;
    const float pixel_to_mm = PIXEL_TO_MM;
    const int center_x = image_size / 2;
    const int center_y = image_size / 2;
    
    // 1. 图像坐标 -> 投影平面坐标（YOZ平面）
    // 映射关系：1像素 = PIXEL_TO_MM 毫米
    // (img_x - center_x) * pixel_to_mm 得到相对中心的毫米距离
    // 除以1000转换为米
    float proj_y = (img_x - center_x) * pixel_to_mm / 1000.0f; // 转换为米
    float proj_z = -(img_y - center_y) * pixel_to_mm / 1000.0f; // Z轴翻转，转换为米
    float proj_x = 0.0f; // 在YOZ平面上，X=0
    
    // 2. 投影平面坐标 -> 原始3D坐标（应用逆旋转）
    Eigen::Vector3f proj_point(proj_x, proj_y, proj_z);
    Eigen::Vector3f world_point = trans_info.inverse_rotation_ * proj_point;

    // 3. 相对于质心的坐标 -> 世界坐标
    world_x = world_point.x() + trans_info.centroid_[0];
    world_y = world_point.y() + trans_info.centroid_[1];
    world_z = world_point.z() + trans_info.centroid_[2];

    return true;
}

bool ExtraBoard::world_to_image(float world_x, float world_y, float world_z, int& img_x, int& img_y, const IPTransInfo& trans_info) const
{
    if (!trans_info.transform_valid_)
    {
        spdlog::error("变换矩阵无效，请先执行board_register()");
        return false;
    }
    
    const int image_size = IMAGE_SIZE;
    const float pixel_to_mm = PIXEL_TO_MM;
    const int center_x = image_size / 2;
    const int center_y = image_size / 2;
    
    // 1. 世界坐标 -> 相对于质心的坐标
    Eigen::Vector3f point_centered(world_x - trans_info.centroid_[0],
                                  world_y - trans_info.centroid_[1],
                                  world_z - trans_info.centroid_[2]);

    // 2. 应用旋转变换到YOZ平面
    Eigen::Vector3f point_transformed = trans_info.rotation_matrix_ * point_centered;
    
    // 3. 投影平面坐标 -> 图像坐标
    // 映射关系：1像素 = PIXEL_TO_MM 毫米
    // point_transformed.y() * 1000 得到毫米距离
    // 除以pixel_to_mm得到像素偏移
    img_x = center_x + static_cast<int>(point_transformed.y() * 1000.0f / pixel_to_mm);
    img_y = center_y - static_cast<int>(point_transformed.z() * 1000.0f / pixel_to_mm); // Z轴翻转
    
    // 4. 检查是否在图像范围内
    return (img_x >= 0 && img_x < image_size && img_y >= 0 && img_y < image_size);
}


/**
 * @brief 载入PNG模板图像并进行二值化处理
 * 
 * @param png_file_path PNG图像文件路径
 * @return bool 成功返回true，失败返回false
 */
bool ExtraBoard::load_template_image(const std::string& png_file_path)
{
    // 读取PNG图像
    template_image_ = cv::imread(png_file_path, cv::IMREAD_COLOR);
    
    if (template_image_.empty())
    {
        spdlog::error("无法读取PNG文件: {}", png_file_path);
        template_loaded_ = false;
        return false;
    }

    spdlog::info("成功加载模板图像: {}", png_file_path);
    spdlog::info("图像尺寸: {}x{}", template_image_.cols, template_image_.rows);

    // 转换为灰度图像
    cv::Mat gray_image;
    cv::cvtColor(template_image_, gray_image, cv::COLOR_BGR2GRAY);
    
    // 使用OTSU方法自动选择阈值进行二值化
    cv::threshold(gray_image, binary_template_, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    
    // 转换为3通道图像以便与projection_image_匹配
    cv::cvtColor(binary_template_, binary_template_, cv::COLOR_GRAY2BGR);
    
    template_loaded_ = true;
    spdlog::info("模板图像二值化完成");
    
    return true;
}

/**
 * @brief 获取原始模板图像
 * 
 * @return cv::Mat 原始模板图像的副本
 */
cv::Mat ExtraBoard::get_template_image() const
{
    if (!template_loaded_)
    {
        spdlog::warn("警告: 模板图像未加载");
        return cv::Mat();
    }
    return template_image_.clone();
}

/**
 * @brief 获取二值化模板图像
 * 
 * @return cv::Mat 二值化模板图像的副本
 */
cv::Mat ExtraBoard::get_binary_template() const
{
    if (!template_loaded_)
    {
        spdlog::warn("警告: 模板图像未加载");
        return cv::Mat();
    }
    return binary_template_.clone();
}

/**
 * @brief 将模板图像与投影图像进行配准
 * 
 * 此函数使用模板匹配技术来找到模板图像在投影图像中的最佳匹配位置。
 * 支持旋转不变的匹配，通过多角度模板匹配来找到最佳的旋转角度和位置。
 * 
 * @return bool 配准成功返回true，失败返回false
 */
Board2DCorners ExtraBoard::register_template_with_projection(cv::Mat& protected_image)
{
    Board2DCorners board_info = {}; // 初始化结构体
    
    if (!template_loaded_)
    {
        spdlog::error("错误: 模板图像未加载，请先调用load_template_image()");
        return board_info; // 返回空的BoardInfo
    }
    
    // 获取当前投影图像
    cv::Mat projection_gray;
    {
        std::lock_guard<std::mutex> lock(image_mutex_);
        if (protected_image.empty())
        {
            spdlog::error("错误: 投影图像为空，请先执行board_register()");
            return board_info; // 返回空的BoardInfo
        }
        cv::cvtColor(protected_image, projection_gray, cv::COLOR_BGR2GRAY);
        
        // 对投影图像进行形态学处理以连接稠密白点并去除小孤立点
        cv::Mat processed_image;
        
        // 1. 先进行闭运算（膨胀+腐蚀）连接临近的白点
        int close_size = 2; // 连接半径
        cv::Mat close_element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                                         cv::Size(2 * close_size + 1, 2 * close_size + 1),
                                                         cv::Point(close_size, close_size));
        cv::morphologyEx(projection_gray, processed_image, cv::MORPH_CLOSE, close_element);
        
        // 2. 使用连通域分析去除小的孤立区域
        cv::Mat labels, stats, centroids;
        int num_components = cv::connectedComponentsWithStats(processed_image, labels, stats, centroids);
        
        // 计算连通域大小阈值（去除小于总像素数0.2%的区域）
        int total_white_pixels = cv::countNonZero(processed_image);
        int min_area_threshold = std::max(10, total_white_pixels / 1000); // 最小10像素或总白像素的0.1%

        // 创建清理后的图像
        cv::Mat cleaned_image = cv::Mat::zeros(processed_image.size(), CV_8UC1);
        for (int i = 1; i < num_components; ++i) // 跳过背景（标签0）
        {
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area >= min_area_threshold)
            {
                // 保留足够大的连通域
                cv::Mat mask = (labels == i);
                cleaned_image.setTo(255, mask);
            }
        }
        
        projection_gray = cleaned_image;

        static int save_index = 0;
        std::string filename = cache_directory_ + "/" + cv::format("processed_projection_%d.png", save_index++);
        spdlog::info("保存处理后的投影图像: {}", filename);
        cv::imwrite(filename, projection_gray);

        // 将处理后的projection_gray更新到protected_image（转换为3通道）
        cv::cvtColor(projection_gray, protected_image, cv::COLOR_GRAY2BGR);
        image_updated_ = true;
    }
    
    spdlog::info("开始进行模板配准...");
    
    // 将二值化模板转换为灰度图像用于匹配
    cv::Mat template_gray;
    cv::cvtColor(binary_template_, template_gray, cv::COLOR_BGR2GRAY);
    
    // 最佳匹配参数
    double best_match_value = -1.0;
    cv::Point best_match_location;
    double best_angle = 0.0f;
    double best_scale = 1.0f;    //处理点云拖影、畸变导致的平面点云膨胀问题

    fastTemplateMatchY(template_gray,projection_gray,best_angle,best_match_location,best_match_value, best_scale);
    
    // 检查匹配质量
    const double MATCH_THRESHOLD = 0.4; // 匹配阈值
    if (best_match_value < MATCH_THRESHOLD)
    {
        spdlog::error("错误: 模板匹配质量过低 ({} < {})", best_match_value, MATCH_THRESHOLD);
        return board_info; // 返回空的BoardInfo
    }
    
    // 保存配准结果
    registration_offset_ = cv::Point2f(best_match_location.x, best_match_location.y);
    registration_angle_ = best_angle;

    spdlog::info("模板配准成功：");
    spdlog::info("  最佳匹配度: {}", best_match_value);
    spdlog::info("  偏移量: ({}, {})", registration_offset_.x, registration_offset_.y);
    spdlog::info("  旋转角度: {} 度", registration_angle_);

    // 在投影图像上绘制匹配结果用于可视化
    {
        std::lock_guard<std::mutex> lock(image_mutex_);
        
        // 获取原始模板尺寸
        cv::Mat template_gray;
        cv::cvtColor(binary_template_, template_gray, cv::COLOR_BGR2GRAY);
        
        // 计算模板的四个角点（相对于模板中心）
        float half_width = template_gray.cols / 2.0f;
        float half_height = template_gray.rows / 2.0f;
        
        std::vector<cv::Point2f> template_corners = {
            cv::Point2f(-half_width, -half_height),  // 左上
            cv::Point2f(half_width, -half_height),   // 右上
            cv::Point2f(half_width, half_height),    // 右下
            cv::Point2f(-half_width, half_height)    // 左下
        };
        
        // 创建旋转矩阵（注意图像坐标系中Y轴向下，旋转方向相反）
        float angle_rad = -best_angle * M_PI / 180.0f; // 取负号修正旋转方向
        cv::Mat rotation_mat = (cv::Mat_<float>(2, 2) << 
            cos(angle_rad), -sin(angle_rad),
            sin(angle_rad), cos(angle_rad));
        
        // 计算匹配中心点（在投影图像中的位置）
        cv::Point2f match_center(
            best_match_location.x + template_gray.cols / 2.0f * best_scale,
            best_match_location.y + template_gray.rows / 2.0f
        );
        
        // 旋转角点并转换到投影图像坐标系
        std::vector<cv::Point> rotated_corners;
        for (const auto& corner : template_corners)
        {
            // 应用旋转
            cv::Mat corner_mat = (cv::Mat_<float>(2, 1) << corner.x, corner.y);
            cv::Mat rotated_corner_mat = rotation_mat * corner_mat;
            
            // 转换到投影图像坐标系
            cv::Point rotated_corner(
                static_cast<int>(match_center.x + rotated_corner_mat.at<float>(0, 0)),
                static_cast<int>(match_center.y + rotated_corner_mat.at<float>(1, 0))
            );
            rotated_corners.push_back(rotated_corner);
        }
        
        // 绘制旋转后的匹配框（四边形）
        const cv::Point* pts = rotated_corners.data();
        int npts = rotated_corners.size();
        cv::polylines(protected_image, &pts, &npts, 1, true, cv::Scalar(0, 255, 0), 2);
        
        // 在匹配位置绘制中心点
        cv::circle(protected_image, cv::Point(static_cast<int>(match_center.x), static_cast<int>(match_center.y)), 
                   5, cv::Scalar(0, 255, 0), -1);
        
        // 绘制指向角度的箭头（显示旋转方向）
        float arrow_length = 30.0f;
        float arrow_angle_rad = -best_angle * M_PI / 180.0f; // 与旋转矩阵保持一致
        cv::Point arrow_end(
            static_cast<int>(match_center.x + arrow_length * cos(arrow_angle_rad)),
            static_cast<int>(match_center.y + arrow_length * sin(arrow_angle_rad))
        );
        cv::arrowedLine(protected_image, cv::Point(static_cast<int>(match_center.x), static_cast<int>(match_center.y)), 
                       arrow_end, cv::Scalar(255, 0, 0), 2, 8, 0, 0.3);
        
        // 创建变换后的模板图像并以50%透明度叠加
        cv::Mat transformed_template;
        cv::Point2f template_center(template_image_.cols / 2.0f, template_image_.rows / 2.0f);
        cv::Mat transform_matrix = cv::getRotationMatrix2D(template_center, best_angle, 1.0);
        
        // 添加平移变换
        transform_matrix.at<double>(0, 2) += (match_center.x - template_center.x);
        transform_matrix.at<double>(1, 2) += (match_center.y - template_center.y);
        
        // 对原始模板图像进行变换
        cv::warpAffine(template_image_, transformed_template, transform_matrix, 
                      cv::Size(protected_image.cols, protected_image.rows),
                      cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        
        // 创建掩码，只在非黑色区域叠加
        cv::Mat mask;
        cv::cvtColor(transformed_template, mask, cv::COLOR_BGR2GRAY);
        cv::threshold(mask, mask, 1, 255, cv::THRESH_BINARY);
        
        // 将变换后的模板图像以50%透明度叠加到投影图像上
        cv::Mat overlay_region;
        transformed_template.copyTo(overlay_region, mask);
        
        // 对有内容的区域进行透明度混合，将模板图像转换为绿色
        for (int y = 0; y < protected_image.rows; ++y)
        {
            for (int x = 0; x < protected_image.cols; ++x)
            {
                if (mask.at<uchar>(y, x) > 0)
                {
                    cv::Vec3b& proj_pixel = protected_image.at<cv::Vec3b>(y, x);
                    cv::Vec3b& template_pixel = transformed_template.at<cv::Vec3b>(y, x);
                    
                    // 将模板像素转换为绿色（保持原始亮度）
                    uchar green_intensity = static_cast<uchar>(0.299 * template_pixel[2] + 0.587 * template_pixel[1] + 0.114 * template_pixel[0]);
                    cv::Vec3b green_pixel(0, green_intensity, 0); // BGR格式：绿色
                    
                    // 30%透明度混合 (0.3 * green_template + 0.7 * projection)
                    proj_pixel[0] = static_cast<uchar>(0.3 * green_pixel[0] + 0.7 * proj_pixel[0]); // B
                    proj_pixel[1] = static_cast<uchar>(0.3 * green_pixel[1] + 0.7 * proj_pixel[1]); // G
                    proj_pixel[2] = static_cast<uchar>(0.3 * green_pixel[2] + 0.7 * proj_pixel[2]); // R
                }
            }
        }
        
        // 添加配准信息文本
        cv::putText(protected_image, 
                   cv::format("Match: %.3f, Angle: %.1f", best_match_value, best_angle),
                   cv::Point(10, 130), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        
        cv::putText(protected_image, 
                   "Template Overlay: 30% Alpha (Green)",
                   cv::Point(10, 160), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
        
        image_updated_ = true;
    }
    
    // 计算并填充BoardInfo结构体
    {
        // 获取原始模板尺寸
        cv::Mat template_gray;
        cv::cvtColor(binary_template_, template_gray, cv::COLOR_BGR2GRAY);
        
        // 计算模板的四个角点（相对于模板中心）
        float half_width = template_gray.cols / 2.0f * 0.86f;
        float half_height = template_gray.rows / 2.0f * 0.86f;
        
        std::vector<cv::Point2f> template_corners = {
            cv::Point2f(-half_width, -half_height),  // 左上
            cv::Point2f(half_width, -half_height),   // 右上
            cv::Point2f(half_width, half_height),    // 右下
            cv::Point2f(-half_width, half_height)    // 左下
        };
        
        // 创建旋转矩阵（注意图像坐标系中Y轴向下，旋转方向相反）
        float angle_rad = -best_angle * M_PI / 180.0f; // 取负号修正旋转方向
        cv::Mat rotation_mat = (cv::Mat_<float>(2, 2) << 
            cos(angle_rad), -sin(angle_rad),
            sin(angle_rad), cos(angle_rad));
        
        // 计算匹配中心点（在投影图像中的位置）
        cv::Point2f match_center(
            best_match_location.x + template_gray.cols / 2.0f,
            best_match_location.y + template_gray.rows / 2.0f
        );
        
        // 设置中心点
        board_info.center = match_center;
        
        // 旋转角点并转换到投影图像坐标系
        std::vector<cv::Point2f> rotated_corners;
        for (const auto& corner : template_corners)
        {
            // 应用旋转
            cv::Mat corner_mat = (cv::Mat_<float>(2, 1) << corner.x, corner.y);
            cv::Mat rotated_corner_mat = rotation_mat * corner_mat;
            
            // 转换到投影图像坐标系
            cv::Point2f rotated_corner(
                match_center.x + rotated_corner_mat.at<float>(0, 0),
                match_center.y + rotated_corner_mat.at<float>(1, 0)
            );
            rotated_corners.push_back(rotated_corner);
        }
        
        // 设置四个角点
        board_info.corn_left_top = rotated_corners[0];     // 左上
        board_info.corn_right_top = rotated_corners[1];    // 右上
        board_info.corn_right_bottom = rotated_corners[2]; // 右下
        board_info.corn_left_bottom = rotated_corners[3];  // 左下
        board_info.is_valid = true;
    }
    
    return board_info;
}

// 返回：矩形长边与图像 Y 轴的夹角，单位：度，范围 [-10,10]
void ExtraBoard::fastTemplateMatchY(const cv::Mat& template_gray,
                        const cv::Mat& projection_gray,
                        double&         best_angle_y,   // 输出：与 Y 轴夹角
                        cv::Point&     best_match_location,
                        double&        best_match_value,
                        double&        scale_value)
{
    if(0)
    {
            // 多角度模板匹配 (从-10度到10度，每0.1度一个步长)
        for (float angle = -10.0f; angle <= 10.0f; angle += 0.1f)
        {
            // 旋转模板图像
            cv::Mat rotated_template;
            cv::Point2f center(template_gray.cols / 2.0f, template_gray.rows / 2.0f);
            cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, angle, 1.0);
            cv::warpAffine(template_gray, rotated_template, rotation_matrix, template_gray.size());
            
            // 确保旋转后的模板不大于投影图像
            if (rotated_template.rows > projection_gray.rows || rotated_template.cols > projection_gray.cols)
            {
                continue;
            }
            
            // 模板匹配
            cv::Mat match_result;
            cv::matchTemplate(projection_gray, rotated_template, match_result, cv::TM_CCOEFF_NORMED);
            
            // 找到最佳匹配位置
            double min_val, max_val;
            cv::Point min_loc, max_loc;
            cv::minMaxLoc(match_result, &min_val, &max_val, &min_loc, &max_loc);
            
            // 更新最佳匹配
            if (max_val > best_match_value)
            {
                best_match_value = max_val;
                best_match_location = max_loc;
                best_angle_y = angle;
            }
        }
    }
    else
    {
        /* ---------- 1. 最小外接矩形 ---------- */
        std::vector<cv::Point> pts;
        cv::findNonZero(projection_gray, pts);
        if (pts.empty()) { best_match_value = -1; return; }
        cv::RotatedRect rbox = cv::minAreaRect(pts);   // angle [-90,0)

        /* ---------- 2. 两条边的方向向量 ---------- */
        if(fabs(rbox.angle) > 45.f)  best_angle_y = -rbox.angle + 90.0f;    //处理最小包围框的长短边相差90度的情况
        else                        best_angle_y = -rbox.angle;


        /* ---------- 4. 仅一次旋转+匹配 ---------- */
        cv::Mat R = cv::getRotationMatrix2D(
                    cv::Point2f(template_gray.cols*0.5f, template_gray.rows*0.5f),
                    best_angle_y, 1.0);
        cv::Mat rotated_tpl;
        cv::warpAffine(template_gray, rotated_tpl, R, template_gray.size(),
                    cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);

        if (rotated_tpl.rows > projection_gray.rows ||
            rotated_tpl.cols > projection_gray.cols)
        {
            best_match_value = -1.0;
            return;
        }

        if(0)
        {
            cv::Mat match_result;
            cv::matchTemplate(projection_gray, rotated_tpl, match_result, cv::TM_CCOEFF_NORMED);
            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            cv::minMaxLoc(match_result, &minVal, &maxVal, &minLoc, &maxLoc);

            best_match_value = maxVal;
            best_match_location   = maxLoc;
        }
        else
        {
            // 1. 先把原始模板尺寸记下来
            int tpl_w = rotated_tpl.cols;
            int tpl_h = rotated_tpl.rows;

            // 2. 准备遍历的尺度序列：0.7 ~ 1.3，步长 0.05
            std::vector<double> scales;
            for (double s = 1.0; s <= 1.05 + 1e-5; s += 0.01) scales.push_back(s);

            // 3. 清空最佳记录
            best_match_value = -1.0;
            scale_value = 1.0;
            // cv::Point best_match_location;

            // 4. 多尺度暴力循环
            for (double s : scales)
            {
                // 4-1 按当前尺度缩放模板
                int new_w = cvRound(tpl_w * s);
                int new_h = cvRound(tpl_h * s);
                cv::Mat resized_tpl;
                cv::resize(rotated_tpl, resized_tpl, cv::Size(new_w, new_h));

                // 4-2 如果模板比图还大就跳过
                if (resized_tpl.cols > projection_gray.cols ||
                    resized_tpl.rows > projection_gray.rows)
                    continue;

                // 4-3 单尺度模板匹配
                cv::Mat match_result;
                cv::matchTemplate(projection_gray, resized_tpl, match_result, cv::TM_CCOEFF_NORMED);

                // 4-4 找最值
                double minVal, maxVal;
                cv::Point minLoc, maxLoc;
                cv::minMaxLoc(match_result, &minVal, &maxVal, &minLoc, &maxLoc);

                spdlog::info("  s: {}  maxVal: {}", s, maxVal);

                // 4-5 记录所有尺度里的最佳
                if (maxVal > best_match_value)
                {
                    best_match_value = maxVal;
                    best_match_location = maxLoc;
                    scale_value = s;
                    // 如果想保存最佳尺度，可再留一个 double best_scale = s;
                }
            }
            // 循环结束后，best_match_location 就是模板左上角在原图中的位置
        }
        

    }
}

bool ExtraBoard::apply_intensity_filter(const PointCloudT::Ptr& input_cloud, PointCloudT::Ptr& output_cloud,
                                 const float min_intensity, const float max_intensity)
{
    if (input_cloud->empty())
    {
        spdlog::warn("警告: 输入点云为空，无法进行强度滤波");
        return false;
    }

    // 创建一个新的点云用于存储滤波后的结果
    output_cloud.reset(new PointCloudT);

    // 遍历输入点云，筛选强度在指定范围内的点
    for (const auto& point : input_cloud->points)
    {
        if (point.intensity >= min_intensity && point.intensity <= max_intensity)
        {
            output_cloud->points.push_back(point);
        }
    }

    // 更新点云的宽高信息
    output_cloud->width = output_cloud->points.size();
    output_cloud->height = 1; // 设为1表示无序点云

    spdlog::info("强度滤波完成: {} -> {} 个点", input_cloud->size(), output_cloud->size());
    return !output_cloud->empty();
}


bool ExtraBoard::apply_box_filter(const PointCloudT::Ptr& input_cloud, PointCloudT::Ptr& output_cloud,
                                 const Eigen::Vector4f& min_point, const Eigen::Vector4f& max_point)
{
    if (input_cloud->empty())
    {
        return false;
    }
    
    pcl::CropBox<PointT> crop_box;
    crop_box.setMin(min_point);
    crop_box.setMax(max_point);
    crop_box.setInputCloud(input_cloud);
    crop_box.filter(*output_cloud);

    return !output_cloud->empty();
}

bool ExtraBoard::apply_outlier_removal(const PointCloudT::Ptr& input_cloud, PointCloudT::Ptr& output_cloud,
                                      int mean_k, double std_dev_mul_thresh)
{
    if (input_cloud->empty())
    {
        spdlog::warn("警告: 输入点云为空，无法进行离群点滤除");
        return false;
    }
    
    pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud(input_cloud);
    sor.setMeanK(mean_k);
    sor.setStddevMulThresh(std_dev_mul_thresh);
    sor.filter(*output_cloud);

    spdlog::info("离群点滤除完成: {} -> {} 个点", input_cloud->size(), output_cloud->size());
    return !output_cloud->empty();
}

/**
 * @brief 对输入点云进行欧式聚类，只保留最大点数的那一簇点云（直接修改输入点云）
 * 
 * @param cloud 输入输出点云（处理后只保留最大簇）
 * @param cluster_tolerance 聚类距离阈值
 * @param min_cluster_size 最小簇大小
 * @param max_cluster_size 最大簇大小
 * @return bool 成功返回true，失败返回false
 */
bool ExtraBoard::apply_euclidean_filter(PointCloudT::Ptr& cloud,
                                        double cluster_tolerance, int min_cluster_size,
                                        int max_cluster_size)
{
    std::vector<PointCloudT::Ptr> cluster_clouds;
    if (!apply_euclidean_clustering(cloud, cluster_clouds, cluster_tolerance, min_cluster_size, max_cluster_size)) {
        return false;
    }
    if (cluster_clouds.empty()) {
        spdlog::warn("欧式滤波后无有效点云");
        return false;
    }
    // 找到最大点数的簇
    size_t max_idx = 0;
    size_t max_size = cluster_clouds[0]->size();
    for (size_t i = 1; i < cluster_clouds.size(); ++i) {
        if (cluster_clouds[i]->size() > max_size) {
            max_size = cluster_clouds[i]->size();
            max_idx = i;
        }
    }
    *cloud = *cluster_clouds[max_idx];
    return true;
}

bool ExtraBoard::apply_euclidean_clustering(const PointCloudT::Ptr& input_cloud, 
                                           std::vector<PointCloudT::Ptr>& cluster_clouds,
                                           double cluster_tolerance, int min_cluster_size, 
                                           int max_cluster_size)
{
    if (input_cloud->empty())
    {
        spdlog::warn("警告: 输入点云为空，无法进行欧式聚类");
        return false;
    }
    
    cluster_clouds.clear();

    // 不使用KdTree，直接聚类
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setInputCloud(input_cloud);
    // ec.setSearchMethod(tree); // 不设置搜索树
    ec.setClusterTolerance(cluster_tolerance);
    ec.setMinClusterSize(min_cluster_size);
    ec.setMaxClusterSize(max_cluster_size);
    ec.extract(cluster_indices);

    for (const auto& indices : cluster_indices)
    {
        PointCloudT::Ptr cluster(new PointCloudT);
        for (const auto& index : indices.indices)
        {
            cluster->points.push_back(input_cloud->points[index]);
        }
        cluster->width = cluster->points.size();
        cluster->height = 1;
        cluster->is_dense = true;

        cluster_clouds.push_back(cluster);
    }

    spdlog::info("欧式聚类完成: 发现 {} 个聚类", cluster_clouds.size());
    return !cluster_clouds.empty();
}

bool ExtraBoard::apply_scaffold_filter(
    const std::vector<PointCloudT::Ptr>& input_cloud,
    std::vector<PointCloudT::Ptr>& output_cloud)
{
    if (input_cloud.empty())
    {
        spdlog::warn("警告: 输入点云列表为空，无法进行支架滤除");
        return false;
    }

    for (const auto& cloud : input_cloud)
    {
        if (!cloud || cloud->empty())
        {
            spdlog::warn("警告: 输入点云为空，跳过");
            continue;
        }

        PointCloudT::Ptr filtered_cloud(new PointCloudT);

        // 1. 全局 Z 范围
        PointT min_pt, max_pt;
        pcl::getMinMax3D(*cloud, min_pt, max_pt);

        // 2. 自顶向下滑动 4 cm 切片
        for (float z = max_pt.z; z > min_pt.z; z -= 0.04f)
        {
            PointCloudT::Ptr slice(new PointCloudT);
            Eigen::Vector4f min_box(min_pt.x, min_pt.y, z - 0.025f, 1.0f);
            Eigen::Vector4f max_box(max_pt.x, max_pt.y, z + 0.025f, 1.0f);
            apply_box_filter(cloud,slice, min_box, max_box);

            if (slice->empty()) continue;

            PointT slice_min, slice_max;
            pcl::getMinMax3D(*slice, slice_min, slice_max);

            // 宽度 > bracket_width + 10 cm 保留，其余视为支架丢弃
            if (dist_x_y(slice_min, slice_max) > (algorithm_params_.bracket_width + 0.1f))
                *filtered_cloud += *slice;
        }

        output_cloud.push_back(filtered_cloud);
    }
    return true;
}

bool ExtraBoard::apply_ransac_plane_segmentation(const PointCloudT::Ptr& input_cloud, 
                                                PointCloudT::Ptr& plane_cloud,
                                                pcl::ModelCoefficients::Ptr& coefficients,
                                                int max_iterations, double distance_threshold)
{
    if (input_cloud->empty())
    {
        spdlog::warn("警告: 输入点云为空，无法进行RANSAC平面分割");
        return false;
    }
    
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    
    // 创建分割对象
    pcl::SACSegmentation<PointT> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(max_iterations);
    seg.setDistanceThreshold(distance_threshold);
    
    seg.setInputCloud(input_cloud);
    seg.segment(*inliers, *coefficients);
    
    if (inliers->indices.size() == 0 || coefficients->values.size() < 4)
    {
        spdlog::error("错误: 未能找到平面");
        return false;
    }
    
    // 提取平面点云
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(input_cloud);
    extract.setIndices(inliers);
    extract.setNegative(false);
    extract.filter(*plane_cloud);
    
    return !plane_cloud->empty();
}


/**
 * @brief 对输入点云进行体素滤波
 * 
 * @param input_cloud 输入点云
 * @param voxel_size 体素滤波的分辨率（单位：米）
 * @param output_cloud 输出体素滤波后的点云
 * @return bool 成功返回true，失败返回false
 */
bool ExtraBoard::apply_voxel_filter(const PointCloudT::Ptr& input_cloud, float voxel_size, PointCloudT::Ptr& output_cloud)
{
    if (!input_cloud || input_cloud->empty()) {
        spdlog::warn("警告: 输入点云为空，无法进行体素滤波");
        return false;
    }
    pcl::VoxelGrid<PointT> voxel_filter;
    voxel_filter.setInputCloud(input_cloud);
    voxel_filter.setLeafSize(voxel_size, voxel_size, voxel_size);
    voxel_filter.filter(*output_cloud);
    if (output_cloud->empty()) {
        spdlog::warn("体素滤波后点云为空");
        return false;
    }
    return true;
}

/**
 * @brief 对输入点云进行体素滤波并计算其最小外接矩形（包围盒）
 * 
 * @param input_cloud 输入点云
 * @param voxel_size 体素滤波的分辨率（单位：米）
 * @param box 输出的包围盒信息
 * @return bool 成功返回true，失败返回false
 */
bool ExtraBoard::compute_bounding_box(const PointCloudT::Ptr& input_cloud, float voxel_size, BBox& box)
{
    if (!input_cloud || input_cloud->empty()) {
        spdlog::warn("警告: 输入点云为空，无法计算包围盒");
        return false;
    }

    // 1. 体素滤波
    PointCloudT::Ptr filtered_cloud(new PointCloudT);
    if (!apply_voxel_filter(input_cloud, voxel_size, filtered_cloud)) {
        return false;
    }

    // 2. 计算最小外接矩形
    pcl::MomentOfInertiaEstimation<PointT> feature_extractor;
    feature_extractor.setInputCloud(filtered_cloud);
    feature_extractor.compute();

    PointT min_point_AABB, max_point_AABB;
    PointT min_point_OBB, max_point_OBB, position_OBB;
    Eigen::Matrix3f rotational_matrix_OBB;

    feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);

    // 包围盒中心
    Eigen::Vector3f min_pt(min_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
    Eigen::Vector3f max_pt(max_point_OBB.x, max_point_OBB.y, max_point_OBB.z);
    box.position = Eigen::Vector3f(position_OBB.x, position_OBB.y, position_OBB.z);

    // 包围盒尺寸
    box.size = max_pt - min_pt;

    // 包围盒姿态（四元数）
    box.orientation = Eigen::Quaternionf(rotational_matrix_OBB);

    return true;
}

/**
 * @brief 合并多个点云为一个点云
 * 
 * @param clouds 输入的点云vector
 * @return PointCloudT::Ptr 合并后的点云
 */
PointCloudT::Ptr ExtraBoard::merge_pointclouds(const std::vector<PointCloudT::Ptr>& clouds)
{
    PointCloudT::Ptr merged_cloud(new PointCloudT);
    for (const auto& cloud : clouds)
    {
        if (cloud && !cloud->empty())
        {
            *merged_cloud += *cloud;
        }
    }
    return merged_cloud;
}
