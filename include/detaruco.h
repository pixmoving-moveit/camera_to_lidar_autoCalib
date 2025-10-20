#ifndef DETARUCO_H
#define DETARUCO_H
#include "calibrate.h"
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/aruco.hpp>
#include <opencv4/opencv2/calib3d.hpp>
#include <opencv4/opencv2/dnn.hpp>
#include <vector>
#include <map>
#include <filesystem>
#include <memory>          // <--- 新增
#include <random>
#include"base_struct.h"
#include <onnxruntime_cxx_api.h>
#include <regex>

#define    RET_OK nullptr

#ifdef USE_CUDA
#include <cuda_fp16.h>
#endif

struct ArucoBoardInfo
{
    cv::Mat aruco_img;
    cv::Point2f lt_point; // ArUco图像区域左上角点
    cv::Size orig_size;
};


enum MODEL_TYPE
{
    //FLOAT32 MODEL
    YOLO_DETECT_V8 = 1,
    YOLO_POSE = 2,
    YOLO_CLS = 3,

    //FLOAT16 MODEL
    YOLO_DETECT_V8_HALF = 4,
    YOLO_POSE_V8_HALF = 5,
    YOLO_CLS_HALF = 6
};


typedef struct _DL_INIT_PARAM
{
    std::string modelPath;
    MODEL_TYPE modelType = YOLO_DETECT_V8;
    std::vector<int> imgSize = { 640, 640 };
    float rectConfidenceThreshold = 0.6;
    float iouThreshold = 0.00001;
    int	keyPointsNum = 2;//Note:kpt number for pose
    bool cudaEnable = false;
    int logSeverityLevel = 3;
    int intraOpNumThreads = 1;
} DL_INIT_PARAM;


typedef struct _DL_RESULT
{
    int classId;
    float confidence;
    cv::Rect box;
    std::vector<cv::Point2f> keyPoints;
} DL_RESULT;


class YOLO_V8
{
public:
    YOLO_V8();

    ~YOLO_V8();

public:
    char* CreateSession(DL_INIT_PARAM& iParams);

    char* RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult);

    char* WarmUpSession();

    template<typename N>
    char* TensorProcess(clock_t& starttime_1, cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims,
        std::vector<DL_RESULT>& oResult);

    char* PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg);

    std::vector<std::string> classes{};

private:
    Ort::Env env;
    Ort::Session* session;
    bool cudaEnable;
    Ort::RunOptions options;
    std::vector<const char*> inputNodeNames;
    std::vector<const char*> outputNodeNames;

    MODEL_TYPE modelType;
    std::vector<int> imgSize;
    float rectConfidenceThreshold;
    float iouThreshold;
    float resizeScales;//letterbox scale
};


class DetAruco
{
public:
    DetAruco(/* args */);
    ~DetAruco();
    static DetAruco* getInstance();

    void init_DetAruco(const ArucoInitParams& params);

    std::string image_path_;

    BoardImgAruco detectArucoMarkers(const int cam_id, const IntrParams& intr_params);

    std::map<int, std::string> cam_id_name_map_;

    std::unique_ptr<YOLO_V8> yoloDetector;
    void keepTop3ByScore(std::vector<DL_RESULT>& result);

private:
    /* data */
    bool flag_init = false;

    bool flag_save_yolo_data = false;

    float marker_length_; // ArUco标记边长（米）
    float marker_scale_;  // 外围框放大比例
    float marker_shift_;  // 外围框平移比例
};




#endif // ARUCO_H