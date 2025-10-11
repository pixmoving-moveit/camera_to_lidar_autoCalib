#ifndef CALIBRATE_H
#define CALIBRATE_H

#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <vector>
#include <string>
#include "base_struct.h"



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
    void calibrate(const int cam_id);
    void write_Extrinsics();

    ExtParams getExtParams(const int cam_id);
    IntrParams getIntrParams(const int cam_id);

private:

    bool flag_init = false;

    // cv::Mat img_;
    std::string config_path;

    std::vector<CameraParams> camera_params_;
    std::vector<DataPair> board_data_;


    void readParams(std::string param_path);
    void readAllParams(const std::string param_path);
    IntrParams read_intrinsicParams(const std::string intrinsics_path);
    ExtParams read_extrinsicParams(const std::string extrinsics_path, const int cam_id);

    // void projectPointsToImage(const cv::Mat& rvec, const cv::Mat& tvec);
};

#endif // CALIBRATE_H