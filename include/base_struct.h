#ifndef BASE_STRUCT_H
#define BASE_STRUCT_H
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <vector>
#include <iostream>
#include "spd_log.h"

using PointT = pcl::PointXYZI;
using PointCloudT = pcl::PointCloud<PointT>;

struct ExtParams
{
    double x, y, z;       // 平移
    double roll, pitch, yaw; // 旋转（欧拉角）
    ExtParams() : x(0), y(0), z(0), roll(0), pitch(0), yaw(0) {}
};

struct IntrParams
{
    cv::Mat K_;
    cv::Mat dist_;
    IntrParams() : K_(cv::Mat::eye(3, 3, CV_64F)), dist_(cv::Mat::zeros(5, 1, CV_64F)) {}
};

struct ArucoInitParams
{
    std::string path;
    float marker_length; // 单个Aruco标记的边长（米）
    float marker_scale;  // 外围框放大比例
    float marker_shift;  // 外围框平移比例
    int marker_num;
    std::vector<int> camera_id_list;
    std::vector<std::string> camera_name_list;
    std::vector<int> marker_order;
    bool export_yolo_data = false;
};

//图像角点
struct Board2DCorners
{
    bool is_valid = false; //当前信息是否有效
    int board_id; // 标定板ID
    cv::Point2f center;   // 棋盘中心点 (像素坐标)
    cv::Point2f corn_left_top;    // 左上角 (像素坐标)
    cv::Point2f corn_right_top;   // 右上角 (像素坐标)
    cv::Point2f corn_right_bottom; // 右下角 (像素坐标)
    cv::Point2f corn_left_bottom;  // 左下角 (像素坐标)

    Board2DCorners() : center(0,0), corn_left_top(0,0), corn_right_top(0,0), corn_right_bottom(0,0), corn_left_bottom(0,0) {}
    bool isNone() const { return center == cv::Point2f(0,0) && corn_left_top == cv::Point2f(0,0) && corn_right_top == cv::Point2f(0,0) && corn_right_bottom == cv::Point2f(0,0) && corn_left_bottom == cv::Point2f(0,0); }
};

//aruco角点
struct Board3DCenter
{
    int board_id; // 标定板ID
    cv::Point3f center;   // 棋盘中心点 (三维坐标)
    Board3DCenter() : center(0,0,0) {}
};


struct BoardCloudCorners
{
    bool is_valid = false; //当前信息是否有效
    int board_id; // 标定板ID
    PointT center_3d;            // 中心点 (三维坐标)
    PointT corn_left_bottom_3d;  // 左下角 (三维坐标)
    PointT corn_right_bottom_3d; // 右下角 (三维坐标)
    PointT corn_left_top_3d;     // 左上角 (三维坐标)
    PointT corn_right_top_3d;    // 右上角 (三维坐标)
};

struct BoardInfo
{
    int board_id; // 标定板ID
    Board2DCorners corners_2d; // 图像角点
    BoardCloudCorners corners_cloud; // 3D角点
};

struct BoardImgAruco
{
    int cam_id; // 摄像头ID
    int aruco_num; // 检测到的Aruco标记数量
    std::vector<Board2DCorners> boards_2d; // 多个标定板的图像角点
    std::vector<Board3DCenter> boards_3d_center; // 多个标定板的3D中心点
    BoardImgAruco() : cam_id(-1), aruco_num(0) {}
};




#endif // BASE_STRUCT_H