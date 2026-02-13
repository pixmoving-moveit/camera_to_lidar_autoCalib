#include "calibrate.h"
#include <fstream>
#include <iostream>

// 代价函数逻辑保持：P_cam = q * P_lidar + t
struct LidarCamPnPError {
    LidarCamPnPError(double u, double v, const cv::Point3f& p3d, const cv::Mat& K)
        : u_(u), v_(v), p3d_(p3d) {
        fx = K.at<double>(0, 0); fy = K.at<double>(1, 1);
        cx = K.at<double>(0, 2); cy = K.at<double>(1, 2);
    }

    template <typename T>
    bool operator()(const T* const q_ptr, const T* const t_ptr, T* residuals) const {
        T p_l[3] = {T(p3d_.x), T(p3d_.y), T(p3d_.z)};
        T p_c[3];
        // ceres::QuaternionRotatePoint 期望顺序为 [w, x, y, z]
        ceres::QuaternionRotatePoint(q_ptr, p_l, p_c);
        p_c[0] += t_ptr[0];
        p_c[1] += t_ptr[1];
        p_c[2] += t_ptr[2];

        T xp = p_c[0] / p_c[2];
        T yp = p_c[1] / p_c[2];
        residuals[0] = T(fx) * xp + T(cx) - T(u_);
        residuals[1] = T(fy) * yp + T(cy) - T(v_);
        return true;
    }

    double u_, v_;
    cv::Point3f p3d_;
    double fx, fy, cx, cy;
};
struct MountingDecoupledError {
    MountingDecoupledError(double u, double v, const cv::Point3f& p3d, const cv::Mat& K)
        : u_(u), v_(v), p3d_(p3d) {
        fx = K.at<double>(0, 0); fy = K.at<double>(1, 1);
        cx = K.at<double>(0, 2); cy = K.at<double>(1, 2);
    }

    template <typename T>
    bool operator()(const T* const q_ptr, const T* const p_mount_ptr, T* residuals) const {
        // q_ptr: [w, x, y, z] (LiDAR to Camera)
        // p_mount_ptr: [x, y, z] (Camera center in LiDAR frame)

        // 1. 投影公式: P_cam = R * (P_lidar - P_mount)
        // 逻辑等价于: P_cam = R * P_lidar - R * P_mount (即 t = -R * P_mount)
        T p_l[3] = {T(p3d_.x), T(p3d_.y), T(p3d_.z)};
        T p_relative[3] = {p_l[0] - p_mount_ptr[0], 
                           p_l[1] - p_mount_ptr[1], 
                           p_l[2] - p_mount_ptr[2]};
        
        T p_c[3];
        ceres::QuaternionRotatePoint(q_ptr, p_relative, p_c);

        // 2. 投影到像素
        T xp = p_c[0] / p_c[2];
        T yp = p_c[1] / p_c[2];
        residuals[0] = T(fx) * xp + T(cx) - T(u_);
        residuals[1] = T(fy) * yp + T(cy) - T(v_);

        return true;
    }

    double u_, v_;
    cv::Point3f p3d_;
    double fx, fy, cx, cy;
};

CalibFunc::CalibFunc()
{
    camera_params_.resize(7);   // 最多有7个摄像头
    board_data_.resize(7);      // 最多有7个摄像头
}

void CalibFunc::init_CalibFunc(const std::string& param_path)
{
    config_path = param_path;
    
    readAllParams(config_path);

    gt_utils.init_params(config_path);
    gt_utils.getExtrinsicParamsGT();
    gt_utils.saveExtrinsicParamsGT();

    readExtrinsics_gt("sensor_kit_base_link");

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

bool CalibFunc::readExtrinsics_gt(const std::string node_key)
{
    return gt_utils.getNodePart(node_key, extrinsics_gt);
}

bool CalibFunc::getExtrinsics_gt(const int cam_id, Eigen::Vector3d& xyz, Eigen::Vector3d& rpy)
{
    std::string cam_key = "camera" + std::to_string(cam_id) + "/camera_link";
    if(extrinsics_gt[cam_key])
    {
        xyz << extrinsics_gt[cam_key]["x"].as<double>(),
               extrinsics_gt[cam_key]["y"].as<double>(),
               extrinsics_gt[cam_key]["z"].as<double>();
        rpy <<  extrinsics_gt[cam_key]["roll"].as<double>(),
                extrinsics_gt[cam_key]["pitch"].as<double>(),
                extrinsics_gt[cam_key]["yaw"].as<double>();
        return true;
    }
    else
    {
        spdlog::error("[CalibFunc] sensor_kit_calibration_gt 配置文件缺少节点: {}", cam_key);
        return false;
    }
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


void CalibFunc::calibrate(const int cam_id, const std::string& algorithm_type) 
{
    if(!flag_init)
    {
        spdlog::error("[CalibFunc] 请先调用 init_CalibFunc() 初始化");
        return;
    }

    cv::Mat rvec, tvec;
    auto& cam_params = camera_params_.at(cam_id);
    auto& data_pair = board_data_.at(cam_id);

    rvec = cv::Mat::zeros(3, 1, CV_64F);
    tvec = cv::Mat::zeros(3, 1, CV_64F);

    // Eigen::Vector3d mech_xyz(1.76164, 0.055, -1.14244); 
    // Eigen::Vector3d mech_rpy(1.083529, 3.106964, 1.505914); 
    Eigen::Vector3d mech_xyz(0.0, 0.0, 0.0); 
    Eigen::Vector3d mech_rpy(0.0, 0.0, 0.0); 

    if (!getExtrinsics_gt(cam_id, mech_xyz, mech_rpy))
    {
        spdlog::error("[CalibFunc] 无法获取摄像头 {} 的外参参考真值，无法进行标定", cam_id);
        return;
    }
    else
    {
        spdlog::info("[CalibFunc] 摄像头 {} 外参参考真值: x: {}, y: {}, z: {}, roll: {}, pitch: {}, yaw: {}", 
                     cam_id, mech_xyz[0], mech_xyz[1], mech_xyz[2], mech_rpy[0], mech_rpy[1], mech_rpy[2]);
    }

    bool success = false;

    if("ceres" == algorithm_type)
    {
        spdlog::info("[CalibFunc] 开始使用ceres非线性求解器标定摄像头 {} 外参", cam_id);

        // 不带有物理约束的标定，ceres非线性求解器
        success = RefineFromMounting(data_pair.points_2d_, data_pair.points_3d_, 
                        cam_params.intrinsics_.K_, 
                        mech_xyz, mech_rpy, 
                        rvec, tvec);
    }
    else if("ceres_restrain" == algorithm_type)
    {
        spdlog::info("[CalibFunc] 开始使用带约束的ceres非线性求解器标定摄像头 {} 外参", cam_id);

        // 带有物理约束的标定，默认约束±2cm，ceres非线性求解器
        success = RefineWithPhysicalConstraints(data_pair.points_2d_, data_pair.points_3d_, 
                        cam_params.intrinsics_.K_, 
                        mech_xyz, mech_rpy, 
                        rvec, tvec);

    }
    else if("epnp" == algorithm_type)
    {
        spdlog::info("[CalibFunc] 开始使用epnp标定摄像头 {} 外参", cam_id);

        // 调用solvePnP进行标定, 计算得到的是sensor_kit_base_link到camera_link的外参
        success = cv::solvePnP(data_pair.points_3d_, data_pair.points_2d_, cam_params.intrinsics_.K_, cam_params.intrinsics_.dist_, rvec, tvec, false, cv::SOLVEPNP_EPNP);
    }
    else
    {
        spdlog::error("[CalibFunc] 未知的标定算法类型: {}", algorithm_type);
    }


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

        std::cout << "T_liar2camera:\n"
          << std::fixed << std::setprecision(6)
          << T_o2c.matrix() << std::endl;

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
    else {
        spdlog::error("[CalibFunc] 相机: {} 标定失败!", cam_id);
    }
}

/**
 * @brief 标定精修函数
 * @param init_rpy 初始欧拉角 (Roll, Pitch, Yaw) 单位: 弧度
 * @param init_xyz 初始相机在雷达系下的位置 (X, Y, Z) 单位: 米
 * @param K 相机内参
 * @param out_rvec 输出旋转向量 (angle-axis)
 * @param out_tvec 输出平移向量
 */
bool CalibFunc::RefineFromMounting(const std::vector<cv::Point2f>& pts_2d,
                        const std::vector<cv::Point3f>& pts_3d,
                        const cv::Mat& K,
                        const Eigen::Vector3d& mount_xyz,
                        const Eigen::Vector3d& mount_rpy,
                        cv::Mat& out_rvec, cv::Mat& out_tvec) {

    // 1. 构建 Camera to LiDAR 的变换矩阵 T_c2l
    Eigen::AngleAxisd roll(mount_rpy.x(), Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitch(mount_rpy.y(), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yaw(mount_rpy.z(), Eigen::Vector3d::UnitZ());
    Eigen::Matrix3d R_c2l = (yaw * pitch * roll).toRotationMatrix();
    Eigen::Vector3d t_c2l = mount_xyz;

    // 2. 求逆得到 LiDAR to Camera 的初始值 T_l2c
    // R_l2c = R_c2l^T,  t_l2c = -R_c2l^T * t_c2l
    Eigen::Matrix3d R_l2c = R_c2l.transpose();
    Eigen::Vector3d t_l2c = -R_l2c * t_c2l;
    Eigen::Quaterniond q_l2c(R_l2c);

    // 准备优化变量 (Ceres 四元数顺序 w, x, y, z)
    double q_ptr[4] = {q_l2c.w(), q_l2c.x(), q_l2c.y(), q_l2c.z()};
    double t_ptr[3] = {t_l2c.x(), t_l2c.y(), t_l2c.z()};

    // 3. 构造 Ceres 问题
    ceres::Problem problem;
    for (size_t i = 0; i < pts_2d.size(); ++i) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<LidarCamPnPError, 2, 4, 3>(
                new LidarCamPnPError(pts_2d[i].x, pts_2d[i].y, pts_3d[i], K)),
            new ceres::HuberLoss(1.0), q_ptr, t_ptr);
    }

    // 设置四元数流形约束
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
    problem.SetManifold(q_ptr, new ceres::QuaternionManifold());
#else
    problem.SetParameterization(q_ptr, new ceres::EigenQuaternionParameterization());
#endif

    // 注意：这里按照你的要求去掉了平移的 ParameterBound 约束

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 100;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 4. 将结果转回 rvec (旋转向量) 和 tvec
    Eigen::Quaterniond q_final(q_ptr[0], q_ptr[1], q_ptr[2], q_ptr[3]);
    Eigen::AngleAxisd aa_final(q_final);
    Eigen::Vector3d rv = aa_final.axis() * aa_final.angle();

    for(int i=0; i<3; ++i) {
        out_rvec.at<double>(i) = rv[i];
        out_tvec.at<double>(i) = t_ptr[i];
    }

    std::cout << "Optimization Complete. RMSE: " << std::sqrt(summary.final_cost / pts_2d.size()) << std::endl;
    return summary.termination_type == ceres::TerminationType::CONVERGENCE;
}

bool CalibFunc:: RefineWithPhysicalConstraints(const std::vector<cv::Point2f>& pts_2d,
                                   const std::vector<cv::Point3f>& pts_3d,
                                   const cv::Mat& K,
                                   const Eigen::Vector3d& init_mount_xyz, // 初始物理位置
                                   const Eigen::Vector3d& init_mount_rpy, // 初始（不精确）旋转
                                   cv::Mat& out_rvec, cv::Mat& out_tvec) {

    // 1. 构造初始值
    Eigen::AngleAxisd roll(init_mount_rpy.x(), Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitch(init_mount_rpy.y(), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yaw(init_mount_rpy.z(), Eigen::Vector3d::UnitZ());
    // 初始 LiDAR to Camera 旋转 (R_c2l 的转置)
    Eigen::Quaterniond q_l2c((yaw * pitch * roll).toRotationMatrix().transpose());

    double q_ptr[4] = {q_l2c.w(), q_l2c.x(), q_l2c.y(), q_l2c.z()};
    double p_mount[3] = {init_mount_xyz.x(), init_mount_xyz.y(), init_mount_xyz.z()};

    // 2. 构建 Ceres 问题
    ceres::Problem problem;
    for (size_t i = 0; i < pts_2d.size(); ++i) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<MountingDecoupledError, 2, 4, 3>(
                new MountingDecoupledError(pts_2d[i].x, pts_2d[i].y, pts_3d[i], K)),
            new ceres::HuberLoss(1.0), q_ptr, p_mount);
    }

    // 3. 约束设置
#if CERES_VERSION_MAJOR >= 2 && CERES_VERSION_MINOR >= 1
    problem.SetManifold(q_ptr, new ceres::QuaternionManifold());
#else
    problem.SetParameterization(q_ptr, new ceres::EigenQuaternionParameterization());
#endif

    for (int i = 0; i < 3; ++i) {
        problem.SetParameterLowerBound(p_mount, i, p_mount[i] - 0.02);
        problem.SetParameterUpperBound(p_mount, i, p_mount[i] + 0.02);
    }

    // 4. 求解
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 5. 计算并对比结果
    Eigen::Quaterniond q_final(q_ptr[0], q_ptr[1], q_ptr[2], q_ptr[3]);
    Eigen::Matrix3d R_final = q_final.toRotationMatrix();
    Eigen::Vector3d p_mount_final(p_mount[0], p_mount[1], p_mount[2]);
    Eigen::Vector3d t_final = -R_final * p_mount_final;

    // 计算旋转向量
    Eigen::AngleAxisd aa_final(R_final);
    Eigen::Vector3d rv = aa_final.axis() * aa_final.angle();

    // 输出到 OpenCV 矩阵
    for(int i=0; i<3; ++i) {
        out_rvec.at<double>(i) = rv[i];
        out_tvec.at<double>(i) = t_final[i];
    }

    double rmse = std::sqrt(summary.final_cost / pts_2d.size());

    // --- 对比输出逻辑 ---
    spdlog::info("========== Calibration Refinement Report ==========");
    spdlog::info("1. Optimization Status: {}", summary.BriefReport());
    spdlog::info("2. Reprojection RMSE: {:.4f} pixels", rmse);
    
    spdlog::info("\n");
    spdlog::info("3. Mounting Position (Camera in LiDAR frame):");
    spdlog::info("   Initial: [{:.4f}, {:.4f}, {:.4f}]", init_mount_xyz.x(), init_mount_xyz.y(), init_mount_xyz.z());
    spdlog::info("   Refined: [{:.4f}, {:.4f}, {:.4f}]", p_mount_final.x(), p_mount_final.y(), p_mount_final.z());
    
    Eigen::Vector3d delta_p = p_mount_final - init_mount_xyz;
    spdlog::info("   Delta:   [{:.4f}, {:.4f}, {:.4f}] (meters)", delta_p.x(), delta_p.y(), delta_p.z());
    
    spdlog::info("\n");
    spdlog::info("4. Final Extrinsic (LiDAR to Camera):");
    spdlog::info("   rvec: [{:.4f}, {:.4f}, {:.4f}]", out_rvec.at<double>(0), out_rvec.at<double>(1), out_rvec.at<double>(2));
    spdlog::info("   tvec: [{:.4f}, {:.4f}, {:.4f}]", out_tvec.at<double>(0), out_tvec.at<double>(1), out_tvec.at<double>(2));
    spdlog::info("==================================================\n");
    return (summary.termination_type == ceres::TerminationType::CONVERGENCE && rmse < 4.0);
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
