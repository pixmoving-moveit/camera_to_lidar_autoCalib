#ifndef CALIBRATE_H
#define CALIBRATE_H

#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <vector>
#include <string>
#include "base_struct.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <yaml-cpp/yaml.h>
#include "utils.h"

struct CameraParams
{
    int cam_id;
    IntrParams intrinsics_;
    ExtParams extrinsics_;
};

struct DataPair
{
    int cam_id;
    std::vector<cv::Point2f> points_2d_;  // 2D角点
    std::vector<cv::Point3f> points_3d_; // 3D角点
};



class CalibFunc {
public:
    CalibFunc();
    static CalibFunc* getInstance();

    void init_CalibFunc(const std::string& param_path);

    void setDataForCalib(const int cam_id, const std::vector<BoardInfo>& data_list);
    void calibrate(const int cam_id, const std::string& algorithm_type);
    void write_Extrinsics();

    bool getExtrinsics_gt(const int cam_id, Eigen::Vector3d& xyz, Eigen::Vector3d& rpy);
    
    ExtParams getExtParams(const int cam_id);
    IntrParams getIntrParams(const int cam_id);

private:

    bool flag_init = false;

    Utils gt_utils;
    YAML::Node extrinsics_gt;

    // cv::Mat img_;
    std::string config_path;

    std::vector<CameraParams> camera_params_;
    std::vector<DataPair> board_data_;

    void readAllParams(const std::string param_path);
    bool readExtrinsics_gt(const std::string node_key);

    IntrParams read_intrinsicParams(const std::string intrinsics_path);
    ExtParams read_extrinsicParams(const std::string extrinsics_path, const int cam_id);

    bool RefineFromMounting(const std::vector<cv::Point2f>& pts_2d,
                        const std::vector<cv::Point3f>& pts_3d,
                        const cv::Mat& K,
                        const Eigen::Vector3d& mount_xyz,
                        const Eigen::Vector3d& mount_rpy,
                        cv::Mat& out_rvec, cv::Mat& out_tvec);
                        
    bool RefineWithPhysicalConstraints(const std::vector<cv::Point2f>& pts_2d,
                                   const std::vector<cv::Point3f>& pts_3d,
                                   const cv::Mat& K,
                                   const Eigen::Vector3d& init_mount_xyz, // 初始物理位置
                                   const Eigen::Vector3d& init_mount_rpy, // 初始（不精确）旋转
                                   cv::Mat& out_rvec, cv::Mat& out_tvec);

};

#endif // CALIBRATE_H