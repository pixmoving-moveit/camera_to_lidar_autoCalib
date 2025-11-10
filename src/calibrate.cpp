#include "calibrate.h"
#include <fstream>
#include <iostream>
#include <yaml-cpp/yaml.h>

CalibFunc::CalibFunc()
{
    camera_params_.resize(7);   // 最多有7个摄像头
    board_data_.resize(7);      // 最多有7个摄像头
}

void CalibFunc::init_CalibFunc(const std::string& param_path)
{
    config_path = param_path;
    
    readAllParams(config_path);

    flag_init = true;
}

void CalibFunc::readAllParams(const std::string param_path)
{
    std::string ext_path = param_path + "/extrinsic_parameters/sensor_kit_calibration.yaml";
    
    for(int i=0;i<7;++i)
    {
        std::string intr_path = param_path + "/intrinsic_parameters/camera" + std::to_string(i) + "_params.yaml";
        auto cam_params = read_intrinsicParams(intr_path);

        auto ext_params = read_extrinsicParams(ext_path, i);

        camera_params_.at(i).intrinsics_ = cam_params;
        camera_params_.at(i).extrinsics_ = ext_params;
    }
}

IntrParams CalibFunc::read_intrinsicParams(const std::string intrinsics_path)
{
    IntrParams intr_params;
    YAML::Node intrinsics;
    try {
        intrinsics = YAML::LoadFile(intrinsics_path);
    } catch (const std::exception& e) {
        spdlog::error("[CalibFunc] 配置文件打开失败: {} \n{}", intrinsics_path, e.what());
        throw std::runtime_error("无法打开配置文件: " + intrinsics_path);
    }

    // 读取相机内参中，修正后的内参矩阵
    if (intrinsics["rectification_matrix"]) 
    {
        auto cam_intr = intrinsics["rectification_matrix"];
        if (cam_intr["rows"] && cam_intr["cols"] && cam_intr["data"]) 
        {
            int rows = cam_intr["rows"].as<int>();
            int cols = cam_intr["cols"].as<int>();
            if (rows == 3 && cols == 3) 
            {
                auto data = cam_intr["data"];
                intr_params.K_ = cv::Mat(3, 3, CV_64F);
                for (int i = 0; i < 9; ++i) 
                {
                    intr_params.K_.at<double>(i/3, i%3) = data[i].as<double>();
                }
            } else {
                throw std::runtime_error("相机内参矩阵必须是3x3的");
            }
        } else {
            throw std::runtime_error("相机内参格式错误：缺少rows、cols或data字段");
        }
    }
    intr_params.dist_ = cv::Mat::zeros(5, 1, CV_64F);    //已经去畸变

    return intr_params;
}

ExtParams CalibFunc::read_extrinsicParams(const std::string extrinsics_path, const int cam_id)
{
    ExtParams ext_params;
    YAML::Node extrinsics;
    try {
        extrinsics = YAML::LoadFile(extrinsics_path);
    } catch (const std::exception& e) {
        spdlog::error("[CalibFunc] 配置文件打开失败: {} \n{}", extrinsics_path, e.what());
        throw std::runtime_error("无法打开配置文件: " + extrinsics_path);
    }
    // camera0/camera_link
    std::string cam_key = "camera" + std::to_string(cam_id) + "/camera_link";

    // 读取相机外参
    if (extrinsics["sensor_kit_base_link"][cam_key]) 
    {
        auto node = extrinsics["sensor_kit_base_link"][cam_key];
        ext_params.x = node["x"].as<double>();
        ext_params.y = node["y"].as<double>();
        ext_params.z = node["z"].as<double>();
        ext_params.roll = node["roll"].as<double>();
        ext_params.pitch = node["pitch"].as<double>();
        ext_params.yaw = node["yaw"].as<double>();
    }
    return ext_params;
}

void CalibFunc::write_Extrinsics()
{
    std::string ext_path = config_path + "/extrinsic_parameters/sensor_kit_calibration.yaml";
    
    YAML::Node extrinsics;
    try {
        extrinsics = YAML::LoadFile(ext_path);
    } catch (const std::exception& e) {
        spdlog::error("[CalibFunc] 配置文件打开失败: {} \n{}", ext_path, e.what());
        throw std::runtime_error("无法打开配置文件: " + ext_path);
    }

    for(int i=0;i<7;++i)
    {
        std::string cam_key = "camera" + std::to_string(i) + "/camera_link";

        if(extrinsics["sensor_kit_base_link"][cam_key])
        {
            auto node = extrinsics["sensor_kit_base_link"][cam_key];
            node["x"] = camera_params_.at(i).extrinsics_.x;
            node["y"] = camera_params_.at(i).extrinsics_.y;
            node["z"] = camera_params_.at(i).extrinsics_.z;
            node["roll"] = camera_params_.at(i).extrinsics_.roll;
            node["pitch"] = camera_params_.at(i).extrinsics_.pitch;
            node["yaw"] = camera_params_.at(i).extrinsics_.yaw;

            extrinsics["sensor_kit_base_link"][cam_key] = node;
        }
        else
        {
            spdlog::error("[CalibFunc] 配置文件缺少节点: {}", cam_key);
        }
    }
    spdlog::info("写入外参到: {}", ext_path);

    std::ofstream fout(ext_path);
    fout << extrinsics;
}


void CalibFunc::setDataForCalib(const int cam_id, const std::vector<BoardInfo>& data_list)
{
    if(cam_id < 0 || cam_id >= static_cast<int>(camera_params_.size()))
    {
        throw std::out_of_range("无效的摄像头ID");
        spdlog::error("[CalibFunc] 无效的摄像头ID: {}", cam_id);
    }
    DataPair data_pair;
    data_pair.cam_id = cam_id;
    for(const auto& board : data_list)
    {
        data_pair.points_2d_.push_back(board.corners_2d.corn_left_top);
        data_pair.points_3d_.push_back(cv::Point3f(board.corners_cloud.corn_left_top_3d.x, board.corners_cloud.corn_left_top_3d.y, board.corners_cloud.corn_left_top_3d.z));

        data_pair.points_2d_.push_back(board.corners_2d.corn_right_top);
        data_pair.points_3d_.push_back(cv::Point3f(board.corners_cloud.corn_right_top_3d.x, board.corners_cloud.corn_right_top_3d.y, board.corners_cloud.corn_right_top_3d.z));

        data_pair.points_2d_.push_back(board.corners_2d.corn_right_bottom);
        data_pair.points_3d_.push_back(cv::Point3f(board.corners_cloud.corn_right_bottom_3d.x, board.corners_cloud.corn_right_bottom_3d.y, board.corners_cloud.corn_right_bottom_3d.z));
        
        data_pair.points_2d_.push_back(board.corners_2d.corn_left_bottom);
        data_pair.points_3d_.push_back(cv::Point3f(board.corners_cloud.corn_left_bottom_3d.x, board.corners_cloud.corn_left_bottom_3d.y, board.corners_cloud.corn_left_bottom_3d.z));
    }

    board_data_.at(cam_id) = data_pair;
}


void CalibFunc::calibrate(const int cam_id) 
{
    if(!flag_init)
    {
        spdlog::error("[CalibFunc] 请先调用 init_CalibFunc() 初始化");
        return;
    }

    cv::Mat rvec, tvec;
    auto& cam_params = camera_params_.at(cam_id);
    auto& data_pair = board_data_.at(cam_id);

    // 调用solvePnP进行标定, 计算得到的是sensor_kit_base_link到camera_link的外参
    bool success = cv::solvePnP(data_pair.points_3d_, data_pair.points_2d_, cam_params.intrinsics_.K_, cam_params.intrinsics_.dist_, rvec, tvec, false, cv::SOLVEPNP_EPNP);
    if(success)
    {
        cv::Mat R_cv;
        cv::Rodrigues(rvec, R_cv); // R_cv为3x3 CV_64F
        Eigen::Matrix3d R_eigen;
        for(int i=0;i<3;++i)
            for(int j=0;j<3;++j)
                R_eigen(i,j) = R_cv.at<double>(i,j);

        
        double sy = sqrt(R_eigen(0,0)*R_eigen(0,0) + R_eigen(1,0)*R_eigen(1,0));
        double roll, pitch, yaw;
        if (sy > 1e-6) {
            roll = atan2(R_eigen(2,1), R_eigen(2,2));
            pitch = atan2(-R_eigen(2,0), sy);
            yaw = atan2(R_eigen(1,0), R_eigen(0,0));
        } else {
            roll = atan2(-R_eigen(1,2), R_eigen(1,1));
            pitch = atan2(-R_eigen(2,0), sy);
            yaw = 0;
        }
        ExtParams ext_test;
        ext_test.x = tvec.at<double>(0,0);
        ext_test.y = tvec.at<double>(1,0);
        ext_test.z = tvec.at<double>(2,0);
        ext_test.roll = roll;
        ext_test.pitch = pitch;
        ext_test.yaw = yaw;
        std::cout<<"x: "<<ext_test.x<<", y: "<<ext_test.y<<", z: "<<ext_test.z<<std::endl;
        std::cout<<"roll: "<<ext_test.roll<<", pitch: "<<ext_test.pitch<<", yaw: "<<ext_test.yaw<<std::endl;

        /*  1. 把 cv 的 rvec/tvec 转到 Eigen *************************/
        Eigen::Vector3d rvec_eig(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2));
        Eigen::Vector3d tvec_eig(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));

        Eigen::AngleAxisd aa(rvec_eig.norm(), rvec_eig.normalized());
        Eigen::Isometry3d T_o2c = Eigen::Isometry3d::Identity();   // Object -> Camera
        T_o2c.linear()  = aa.toRotationMatrix();
        T_o2c.translation() = tvec_eig;

        /*  2. 取逆 -> Camera -> Object (即 camera_link -> sensor_kit_base_link) */
        Eigen::Isometry3d T_c2o = T_o2c.inverse();   // 这就是我们要的外参

        /*  3. 转成 xyzrpy ****************************************************/
        Eigen::Vector3d    xyz  = T_c2o.translation();
        Eigen::Vector3d    rpy  = T_c2o.rotation().eulerAngles(2, 1, 0);   // roll, pitch, yaw

        ExtParams ext_;
        ext_.x = xyz(0);
        ext_.y = xyz(1);
        ext_.z = xyz(2);
        ext_.roll = rpy(2);
        ext_.pitch = rpy(1);
        ext_.yaw = rpy(0);

        cam_params.extrinsics_ = ext_;

        spdlog::info("[CalibFunc] 相机: {} 标定成功! "
             "camera_link -> sensor_kit_base_link: "
             "xyz=[{:.4f}, {:.4f}, {:.4f}]  rpy=[{:.4f}, {:.4f}, {:.4f}] (rad)",
             cam_id, ext_.x, ext_.y, ext_.z,
             ext_.roll, ext_.pitch, ext_.yaw);
    }
    
    // if (success) 
    // {
    //     spdlog::info("[CalibFunc] 相机: {} 标定成功!", cam_id);
    //     spdlog::info("rvec: {},{},{}, tvec: {},{},{}", rvec.at<double>(0,0), rvec.at<double>(1,0), rvec.at<double>(2,0),
    //         tvec.at<double>(0,0), tvec.at<double>(1,0), tvec.at<double>(2,0));
    //     // 提取欧拉角和平移
    //     cv::Mat R_cv;
    //     cv::Rodrigues(rvec, R_cv); // R_cv为3x3 CV_64F
    //     Eigen::Matrix3d R_eigen;
    //     for(int i=0;i<3;++i)
    //         for(int j=0;j<3;++j)
    //             R_eigen(i,j) = R_cv.at<double>(i,j);

        
    //     double sy = sqrt(R_eigen(0,0)*R_eigen(0,0) + R_eigen(1,0)*R_eigen(1,0));
    //     double roll, pitch, yaw;
    //     if (sy > 1e-6) {
    //         roll = atan2(R_eigen(2,1), R_eigen(2,2));
    //         pitch = atan2(-R_eigen(2,0), sy);
    //         yaw = atan2(R_eigen(1,0), R_eigen(0,0));
    //     } else {
    //         roll = atan2(-R_eigen(1,2), R_eigen(1,1));
    //         pitch = atan2(-R_eigen(2,0), sy);
    //         yaw = 0;
    //     }
    //     ExtParams ext_;
    //     ext_.x = tvec.at<double>(0,0);
    //     ext_.y = tvec.at<double>(1,0);
    //     ext_.z = tvec.at<double>(2,0);
    //     ext_.roll = roll;
    //     ext_.pitch = pitch;
    //     ext_.yaw = yaw;

    //     cam_params.extrinsics_ = ext_;

    // }
     else {
        spdlog::error("[CalibFunc] 相机: {} 标定失败!", cam_id);
    }
}

IntrParams CalibFunc::getIntrParams(const int cam_id)
{
    if(cam_id < 0 || cam_id >= static_cast<int>(camera_params_.size()))
    {
        throw std::out_of_range("无效的摄像头ID");
    }
    return camera_params_.at(cam_id).intrinsics_;
}

ExtParams CalibFunc::getExtParams(const int cam_id)
{
    if(cam_id < 0 || cam_id >= static_cast<int>(camera_params_.size()))
    {
        throw std::out_of_range("无效的摄像头ID");
    }
    return camera_params_.at(cam_id).extrinsics_;
}

// void CalibFunc::projectPointsToImage(const cv::Mat& rvec, const cv::Mat& tvec) {
//     std::vector<cv::Point3f> obj_pts;
//     for (const auto& pt : points_3d_) obj_pts.emplace_back(pt.x(), pt.y(), pt.z());
//     std::vector<cv::Point2f> img_pts;
//     cv::projectPoints(obj_pts, rvec, tvec, K_, dist_, img_pts);
//     cv::Mat img_show = img_.clone();
//     for (const auto& pt : img_pts) {
//         cv::circle(img_show, pt, 6, cv::Scalar(0,255,0), 2);
//     }
//     cv::imshow("CSV 3D点投影", img_show);
//     cv::waitKey(0);
// }
CalibFunc* CalibFunc::getInstance() 
{
    static CalibFunc instance;
    return &instance;
}
