#include "detaruco.h"

/**
 * @file detaruco.cpp
 * @brief 用于检测图像中存在的Aruco棋盘并提取其位姿信息的实现文件
 */

#include "detaruco.h"


DetAruco::DetAruco(/* args */)
{
    yoloDetector = std::make_unique<YOLO_V8>();   

}

DetAruco::~DetAruco()
{
    
}

void DetAruco::init_DetAruco(const ArucoInitParams& params)
{
    image_path_ = params.path;
    marker_length_ = params.marker_length;
    marker_scale_ = params.marker_scale;
    marker_shift_ = params.marker_shift;
    flag_save_yolo_data = params.export_yolo_data;


    yoloDetector->classes.push_back("pot"); //只有一个类别
    DL_INIT_PARAM params_yolo;
    params_yolo.rectConfidenceThreshold = 0.6;
    params_yolo.iouThreshold = 0.4;
    params_yolo.modelPath = image_path_ + "/../config/best.onnx";
    params_yolo.imgSize = { 640, 640 };
    params_yolo.modelType = YOLO_DETECT_V8;
    params_yolo.cudaEnable = false;
    yoloDetector->CreateSession(params_yolo);


    int size = params.camera_id_list.size();
    if((size == params.camera_name_list.size()) && size > 1)
    {
        spdlog::info("[DetAruco] 初始化完成, 相机ID列表: {}, 相机名称列表: {}", fmt::join(params.camera_id_list, ", "), fmt::join(params.camera_name_list, ", "));

        for(int i=0;i<size;++i)
        {
            cam_id_name_map_[params.camera_id_list[i]] = params.camera_name_list[i];
        }
    }
    else 
    {
        spdlog::error("相机ID列表和相机名称列表长度不一致, 相机ID列表长度: {}, 相机名称列表长度: {}", size, params.camera_name_list.size());
    }

    flag_init = true;
}

BoardImgAruco DetAruco::detectArucoMarkers(const int cam_id, const IntrParams& intr_params)
{
    BoardImgAruco aruco_info;
    aruco_info.cam_id = cam_id;

    if (!flag_init) {
        spdlog::error("DetAruco未初始化，请先调用init_DetAruco()");
        return aruco_info;
    }

    // 根据cam_id获取对应的图像路径
    std::string imagePath;
    if (cam_id_name_map_.find(cam_id) != cam_id_name_map_.end()) {
        imagePath = image_path_ + "/" + cam_id_name_map_[cam_id];
    } else {
        spdlog::error("无效的相机ID: {}", cam_id);
        return aruco_info;
    }

    // 读取输入图像
    cv::Mat image = cv::imread(imagePath + "/image_undistort.png");
    if (image.empty()) {
        spdlog::error("无法打开图像文件: {}", imagePath);
        return aruco_info;
    }

    // 创建输出图像的副本
    cv::Mat outputImage = image.clone();

    double alpha = 1.1; // 亮度增益
    int beta = 10;      // 亮度偏移
    image.convertTo(image, -1, alpha, beta);

    
    // 创建AprilTag 16h5字典
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_16h5);
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
    
    // 用于存储检测结果
    std::vector<std::vector<cv::Point2f>> markerCorners;
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> rejectedCandidates;
    
    // 检测ArUco标记
    cv::aruco::detectMarkers(image, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);
        
    // 转换相机内参为cv::Mat
    cv::Mat cameraMatrix = intr_params.K_;
    cv::Mat distCoeffs = intr_params.dist_;

    std::vector<ArucoBoardInfo> detectedBoards;
    
    if (!markerIds.empty()) 
    {
        aruco_info.aruco_num = static_cast<int>(markerIds.size());
        aruco_info.boards_2d.resize(aruco_info.aruco_num);
        aruco_info.boards_3d_center.resize(aruco_info.aruco_num);
        
        ArucoBoardInfo board_info;

        // 在图像上绘制检测到的标记
        cv::aruco::drawDetectedMarkers(outputImage, markerCorners, markerIds);
        
        // 存储位姿估计结果
        std::vector<cv::Vec3d> rvecs, tvecs;
        cv::aruco::estimatePoseSingleMarkers(markerCorners, marker_length_, 
            cameraMatrix, distCoeffs, rvecs, tvecs);
        
        // 处理每个检测到的标记
        for (size_t i = 0; i < markerCorners.size(); i++) 
        {
            Board2DCorners board_2d;
            Board2DCorners board_roi;
            Board3DCenter board_3d;

            board_2d.board_id = markerIds[i];
            board_3d.board_id = markerIds[i];
            board_3d.center = cv::Point3f(tvecs[i][2], -tvecs[i][0], tvecs[i][1]);

            spdlog::info("Detected marker ID: {} 坐标: ({:.2f}, {:.2f}, {:.2f})", markerIds[i], board_3d.center.x, board_3d.center.y, board_3d.center.z);
            // 在图像上绘制坐标轴
            // cv::aruco::drawAxis(outputImage, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], marker_length_/4);
            const auto& markers = markerCorners[i];  // 当前标记的四个角点

            cv::Point2f anchor = markers[0];  // 左上角锚点
            
            // 计算外围框的顶点
            std::vector<cv::Point2f> outerPts;

            // 计算外围框点（相对于左上角锚点进行缩放）
            for (int j = 0; j < 4; j++) {
                // 以右下角为锚点，先平移再缩放
                float scale = marker_scale_; // 放大比例
                float shift_rb = marker_shift_; // 右下角平移比例
                // 右下-左上方向向量
                cv::Point2f vec_rb2lt = markers[0] - markers[2];
                // 新右下角锚点，沿右下-左上方向平移
                cv::Point2f anchor_rb = markers[2] + vec_rb2lt * shift_rb;
                // 以新锚点为基准做相似放大
                cv::Point2f vec = markers[j] - anchor_rb;
                cv::Point2f outerPt = anchor_rb + vec * scale;
                outerPts.push_back(outerPt);
            }

            // 保存角点到结构体
            // 注意：ArUco检测的角点顺序是：左上、右上、右下、左下
            board_roi.corn_left_top = outerPts[1];
            board_roi.corn_right_top = outerPts[0];
            board_roi.corn_right_bottom = outerPts[3];
            board_roi.corn_left_bottom = outerPts[2];
            board_roi.board_id = markerIds[i];
            // 计算中心点（四个角点的平均值）
            board_roi.center = (board_roi.corn_left_top + board_roi.corn_right_top + 
                            board_roi.corn_right_bottom + board_roi.corn_left_bottom) * 0.25f;
            
            
            aruco_info.boards_3d_center[i] = board_3d;
            
            // 提取Aruco图像区域
            std::vector<cv::Point2f> quad_pts = {
                    board_roi.corn_left_top,
                    board_roi.corn_right_top,
                    board_roi.corn_right_bottom,
                    board_roi.corn_left_bottom
                };
            cv::RotatedRect minRect = cv::minAreaRect(quad_pts);
            cv::Rect bbox = minRect.boundingRect();
            // 检查边界合法性
            bbox &= cv::Rect(0, 0, image.cols, image.rows);
            // 如果边界宽高不合法，则将其限制为最小值0
            if (bbox.width <= 0) bbox.width = 0;
            if (bbox.height <= 0) bbox.height = 0;
            cv::Mat rect_img = image(bbox).clone();
            
            if(flag_save_yolo_data)
            {
                std::string path = image_path_ + "/yolo_data/" + std::to_string(cam_id)+"_"+ std::to_string(i) + ".png";
                cv::imwrite(path, rect_img);
            }

            cv::resize(rect_img, board_info.aruco_img, cv::Size(640, 640));
            // lt_point = 矩形区域的左上角点
            board_info.lt_point = cv::Point(bbox.tl().x + 1, bbox.tl().y + 1);
            board_info.orig_size = rect_img.size();
            detectedBoards.push_back(board_info);

            
            std::vector<DL_RESULT> res;
            cv::Mat img_show = board_info.aruco_img.clone();
            yoloDetector->RunSession(img_show, res);

            keepTop3ByScore(res); // 保留置信度最高的三个结果

            std::vector<cv::Point2f> corner_points;

            for (auto& re : res)
            {
                cv::RNG rng(cv::getTickCount());
                cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

                // 计算原图上的矩形框位置和大小
                // re.box 是在 640x640 上的，需映射回 rect_img 的原始尺寸，再加左上角偏移
                double scale_x = static_cast<double>(board_info.orig_size.width) / 640.0;
                double scale_y = static_cast<double>(board_info.orig_size.height) / 640.0;
                double x = (re.box.x * scale_x) + board_info.lt_point.x;
                double y = (re.box.y * scale_y) + board_info.lt_point.y;
                double w = re.box.width * scale_x;
                double h = re.box.height * scale_y;
                cv::Rect mapped_box(static_cast<int>(x), static_cast<int>(y), static_cast<int>(w), static_cast<int>(h));

                corner_points.push_back(cv::Point2f(mapped_box.x + mapped_box.width*0.5, mapped_box.y + mapped_box.height*0.5));

                // 绘制到原图 outputImage 上
                cv::rectangle(outputImage, mapped_box, color, 1);

                // 置信度和类别标签
                float confidence = floor(100 * re.confidence) / 100;
                std::cout << std::fixed << std::setprecision(2);
                std::string label = yoloDetector->classes[re.classId] + " " +
                    std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

                // 标签背景
                cv::rectangle(
                    outputImage,
                    cv::Point(x, y - 15),
                    cv::Point(x + label.length() * 5, y),
                    color,
                    cv::FILLED
                );

                // 标签文字
                cv::putText(
                    outputImage,
                    label,
                    cv::Point(x, y - 5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.3,
                    cv::Scalar(0, 0, 0),
                    1
                );
            }

            if(corner_points.size() == 3) //检测到三个圆形锚点
            {
                spdlog::info("检测到三个圆形锚点，开始计算角点位置");
                // 计算三个圆形锚点的最小外接圆
                cv::Point2f center(0, 0);
                for (const auto& pt : corner_points) {
                    center += pt;
                }
                center /= 3;

                std::vector<std::pair<double, cv::Point2f>> angle_points;
                for (const auto& pt : corner_points) {
                    cv::Point2f vec = pt - center;
                    double angle = std::atan2(vec.y, vec.x); // 以X轴为基准，逆时针为正
                    if (angle < 0) angle += 2 * CV_PI; // 转为0~2PI，顺时针排序
                    angle_points.emplace_back(angle, pt);
                }
                // 按角度升序排序（逆时针）
                std::sort(angle_points.begin(), angle_points.end(),
                          [](const std::pair<double, cv::Point2f>& a, const std::pair<double, cv::Point2f>& b) {
                              return a.first > b.first;
                          });
                // 排序后的点
                std::vector<cv::Point2f> sorted_points;
                for (const auto& ap : angle_points) {
                    sorted_points.push_back(ap.second);
                }
                // 重新赋值给 board_roi 的角点，顺序 右上- 左上 - 左下 - 右下
                board_2d.corn_right_top = sorted_points[1];
                board_2d.corn_left_top = sorted_points[0];
                board_2d.corn_right_bottom = sorted_points[2];
                board_2d.corn_left_bottom = markers[2];

            }
            else
            {
                spdlog::error("未检测到三个圆形锚点，无法计算角点位置，检测到 {} 个", corner_points.size());
            }
            aruco_info.boards_2d[i] = board_2d;

            // 绘制外围框
            std::vector<cv::Point> outerPtsInt = {
                cv::Point(static_cast<int>(board_roi.corn_left_top.x), static_cast<int>(board_roi.corn_left_top.y)),
                cv::Point(static_cast<int>(board_roi.corn_right_top.x), static_cast<int>(board_roi.corn_right_top.y)),
                cv::Point(static_cast<int>(board_roi.corn_right_bottom.x), static_cast<int>(board_roi.corn_right_bottom.y)),
                cv::Point(static_cast<int>(board_roi.corn_left_bottom.x), static_cast<int>(board_roi.corn_left_bottom.y))
            };
            
            // 绘制外围框（红色）
            cv::polylines(outputImage, std::vector<std::vector<cv::Point>>{outerPtsInt}, 
                         true, cv::Scalar(0, 0, 255), 1);
            
            // 绘制四个角点（蓝色圆点），顺序：左上、右上、右下、左下
            cv::circle(outputImage, board_2d.corn_right_top, 3, cv::Scalar(0, 50, 0), -1);      // 右上
            cv::circle(outputImage, board_2d.corn_left_top, 3, cv::Scalar(0, 100, 0), -1);     // 左上
            cv::circle(outputImage, board_2d.corn_left_bottom, 3, cv::Scalar(0, 150, 0), -1);  // 左下
            cv::circle(outputImage, board_2d.corn_right_bottom, 3, cv::Scalar(0, 200, 0), -1); // 右下

            // 绘制中心点（绿色）
            cv::circle(outputImage, cv::Point(static_cast<int>(board_2d.center.x), 
                      static_cast<int>(board_2d.center.y)), 2, cv::Scalar(50, 0, 0), -1);

        }
    }
    
    // 保存结果图像
    cv::imwrite(imagePath + "/image_aruco.png", outputImage);

    return aruco_info;
}

void DetAruco::keepTop3ByScore(std::vector<DL_RESULT>& res)
{
    if (res.size() <= 3) return;

    // 1. 计算 score 并排序（降序）
    std::sort(res.begin(), res.end(),
              [](const DL_RESULT& a, const DL_RESULT& b)
              {
                  double scoreA = a.box.area() * a.confidence;
                  double scoreB = b.box.area() * b.confidence;
                  return scoreA > scoreB;          // 从大到小
              });

    // 2. 只保留前 3 个
    res.resize(3);
}

DetAruco* DetAruco::getInstance()
{
    static DetAruco instance;
    return &instance;
}


#define benchmark
#define min(a,b)            (((a) < (b)) ? (a) : (b))
YOLO_V8::YOLO_V8() {

}


YOLO_V8::~YOLO_V8() {
    delete session;
}

#ifdef USE_CUDA
namespace Ort
{
    template<>
    struct TypeToTensorType<half> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; };
}
#endif


template<typename T>
char* BlobFromImage(cv::Mat& iImg, T& iBlob) {
    int channels = iImg.channels();
    int imgHeight = iImg.rows;
    int imgWidth = iImg.cols;

    for (int c = 0; c < channels; c++)
    {
        for (int h = 0; h < imgHeight; h++)
        {
            for (int w = 0; w < imgWidth; w++)
            {
                iBlob[c * imgWidth * imgHeight + h * imgWidth + w] = typename std::remove_pointer<T>::type(
                    (iImg.at<cv::Vec3b>(h, w)[c]) / 255.0f);
            }
        }
    }
    return RET_OK;
}


char* YOLO_V8::PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg)
{
    if (iImg.channels() == 3)
    {
        oImg = iImg.clone();
        cv::cvtColor(oImg, oImg, cv::COLOR_BGR2RGB);
    }
    else
    {
        cv::cvtColor(iImg, oImg, cv::COLOR_GRAY2RGB);
    }

    switch (modelType)
    {
    case YOLO_DETECT_V8:
    case YOLO_POSE:
    case YOLO_DETECT_V8_HALF:
    case YOLO_POSE_V8_HALF://LetterBox
    {
        if (iImg.cols >= iImg.rows)
        {
            resizeScales = iImg.cols / (float)iImgSize.at(0);
            cv::resize(oImg, oImg, cv::Size(iImgSize.at(0), int(iImg.rows / resizeScales)));
        }
        else
        {
            resizeScales = iImg.rows / (float)iImgSize.at(0);
            cv::resize(oImg, oImg, cv::Size(int(iImg.cols / resizeScales), iImgSize.at(1)));
        }
        cv::Mat tempImg = cv::Mat::zeros(iImgSize.at(0), iImgSize.at(1), CV_8UC3);
        oImg.copyTo(tempImg(cv::Rect(0, 0, oImg.cols, oImg.rows)));
        oImg = tempImg;
        break;
    }
    case YOLO_CLS://CenterCrop
    {
        int h = iImg.rows;
        int w = iImg.cols;
        int m = min(h, w);
        int top = (h - m) / 2;
        int left = (w - m) / 2;
        cv::resize(oImg(cv::Rect(left, top, m, m)), oImg, cv::Size(iImgSize.at(0), iImgSize.at(1)));
        break;
    }
    }
    return RET_OK;
}


char* YOLO_V8::CreateSession(DL_INIT_PARAM& iParams) {
    char* Ret = RET_OK;
    std::regex pattern("[\u4e00-\u9fa5]");
    bool result = std::regex_search(iParams.modelPath, pattern);
    if (result)
    {
        spdlog::error("[YOLO_V8]: Your model path is error.Change your model path without chinese characters.");
        return Ret;
    }
    try
    {
        rectConfidenceThreshold = iParams.rectConfidenceThreshold;
        iouThreshold = iParams.iouThreshold;
        imgSize = iParams.imgSize;
        modelType = iParams.modelType;
        cudaEnable = iParams.cudaEnable;
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolo");
        Ort::SessionOptions sessionOption;
        if (iParams.cudaEnable)
        {
            OrtCUDAProviderOptions cudaOption;
            cudaOption.device_id = 0;
            sessionOption.AppendExecutionProvider_CUDA(cudaOption);
        }
        sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        sessionOption.SetIntraOpNumThreads(iParams.intraOpNumThreads);
        sessionOption.SetLogSeverityLevel(iParams.logSeverityLevel);

#ifdef _WIN32
        int ModelPathSize = MultiByteToWideChar(CP_UTF8, 0, iParams.modelPath.c_str(), static_cast<int>(iParams.modelPath.length()), nullptr, 0);
        wchar_t* wide_cstr = new wchar_t[ModelPathSize + 1];
        MultiByteToWideChar(CP_UTF8, 0, iParams.modelPath.c_str(), static_cast<int>(iParams.modelPath.length()), wide_cstr, ModelPathSize);
        wide_cstr[ModelPathSize] = L'\0';
        const wchar_t* modelPath = wide_cstr;
#else
        const char* modelPath = iParams.modelPath.c_str();
#endif // _WIN32

        session = new Ort::Session(env, modelPath, sessionOption);
        Ort::AllocatorWithDefaultOptions allocator;
        size_t inputNodesNum = session->GetInputCount();
        for (size_t i = 0; i < inputNodesNum; i++)
        {
            Ort::AllocatedStringPtr input_node_name = session->GetInputNameAllocated(i, allocator);
            char* temp_buf = new char[50];
            strcpy(temp_buf, input_node_name.get());
            inputNodeNames.push_back(temp_buf);
        }
        size_t OutputNodesNum = session->GetOutputCount();
        for (size_t i = 0; i < OutputNodesNum; i++)
        {
            Ort::AllocatedStringPtr output_node_name = session->GetOutputNameAllocated(i, allocator);
            char* temp_buf = new char[10];
            strcpy(temp_buf, output_node_name.get());
            outputNodeNames.push_back(temp_buf);
        }
        options = Ort::RunOptions{ nullptr };
        WarmUpSession();
        return RET_OK;
    }
    catch (const std::exception& e)
    {
        const char* str1 = "[YOLO_V8]:";
        const char* str2 = e.what();
        std::string result = std::string(str1) + std::string(str2);
        char* merged = new char[result.length() + 1];
        std::strcpy(merged, result.c_str());
        // spdlog::error("{}", merged);
        delete[] merged;
        return "[YOLO_V8]:Create session failed.";
    }

}


char* YOLO_V8::RunSession(cv::Mat& iImg, std::vector<DL_RESULT>& oResult) {
#ifdef benchmark
    clock_t starttime_1 = clock();
#endif // benchmark

    char* Ret = RET_OK;
    cv::Mat processedImg;
    PreProcess(iImg, imgSize, processedImg);
    if (modelType < 4)
    {
        float* blob = new float[processedImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> inputNodeDims = { 1, 3, imgSize.at(0), imgSize.at(1) };
        TensorProcess(starttime_1, iImg, blob, inputNodeDims, oResult);
    }
    else
    {
#ifdef USE_CUDA
        half* blob = new half[processedImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> inputNodeDims = { 1,3,imgSize.at(0),imgSize.at(1) };
        TensorProcess(starttime_1, iImg, blob, inputNodeDims, oResult);
#endif
    }

    return Ret;
}


template<typename N>
char* YOLO_V8::TensorProcess(clock_t& starttime_1, cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims,
    std::vector<DL_RESULT>& oResult) {
    Ort::Value inputTensor = Ort::Value::CreateTensor<typename std::remove_pointer<N>::type>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1),
        inputNodeDims.data(), inputNodeDims.size());
#ifdef benchmark
    clock_t starttime_2 = clock();
#endif // benchmark
    auto outputTensor = session->Run(options, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(),
        outputNodeNames.size());
#ifdef benchmark
    clock_t starttime_3 = clock();
#endif // benchmark

    Ort::TypeInfo typeInfo = outputTensor.front().GetTypeInfo();
    auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outputNodeDims = tensor_info.GetShape();
    auto output = outputTensor.front().GetTensorMutableData<typename std::remove_pointer<N>::type>();
    delete[] blob;
    switch (modelType)
    {
    case YOLO_DETECT_V8:
    case YOLO_DETECT_V8_HALF:
    {
        int signalResultNum = outputNodeDims[1];//84
        int strideNum = outputNodeDims[2];//8400
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        cv::Mat rawData;
        if (modelType == YOLO_DETECT_V8)
        {
            // FP32
            rawData = cv::Mat(signalResultNum, strideNum, CV_32F, output);
        }
        else
        {
            // FP16
            rawData = cv::Mat(signalResultNum, strideNum, CV_16F, output);
            rawData.convertTo(rawData, CV_32F);
        }
        // Note:
        // ultralytics add transpose operator to the output of yolov8 model.which make yolov8/v5/v7 has same shape
        // https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
        rawData = rawData.t();

        float* data = (float*)rawData.data;

        for (int i = 0; i < strideNum; ++i)
        {
            float* classesScores = data + 4;
            cv::Mat scores(1, this->classes.size(), CV_32FC1, classesScores);
            cv::Point class_id;
            double maxClassScore;
            cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
            if (maxClassScore > rectConfidenceThreshold)
            {
                confidences.push_back(maxClassScore);
                class_ids.push_back(class_id.x);
                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                int left = int((x - 0.5 * w) * resizeScales);
                int top = int((y - 0.5 * h) * resizeScales);

                int width = int(w * resizeScales);
                int height = int(h * resizeScales);

                boxes.push_back(cv::Rect(left, top, width, height));
            }
            data += signalResultNum;
        }
        std::vector<int> nmsResult;
        cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold, iouThreshold, nmsResult);
        for (int i = 0; i < nmsResult.size(); ++i)
        {
            int idx = nmsResult[i];
            DL_RESULT result;
            result.classId = class_ids[idx];
            result.confidence = confidences[idx];
            result.box = boxes[idx];
            oResult.push_back(result);
        }

#ifdef benchmark
        clock_t starttime_4 = clock();
        double pre_process_time = (double)(starttime_2 - starttime_1) / CLOCKS_PER_SEC * 1000;
        double process_time = (double)(starttime_3 - starttime_2) / CLOCKS_PER_SEC * 1000;
        double post_process_time = (double)(starttime_4 - starttime_3) / CLOCKS_PER_SEC * 1000;
        if (cudaEnable)
        {
            spdlog::info("[YOLO_V8(CUDA)]: {}ms pre-process, {}ms inference, {}ms post-process.", pre_process_time, process_time, post_process_time);
        }
        else
        {
            spdlog::info("[YOLO_V8(CPU)]: {}ms pre-process, {}ms inference, {}ms post-process.", pre_process_time, process_time, post_process_time);
        }
#endif // benchmark

        break;
    }
    case YOLO_CLS:
    case YOLO_CLS_HALF:
    {
        cv::Mat rawData;
        if (modelType == YOLO_CLS) {
            // FP32
            rawData = cv::Mat(1, this->classes.size(), CV_32F, output);
        } else {
            // FP16
            rawData = cv::Mat(1, this->classes.size(), CV_16F, output);
            rawData.convertTo(rawData, CV_32F);
        }
        float *data = (float *) rawData.data;

        DL_RESULT result;
        for (int i = 0; i < this->classes.size(); i++)
        {
            result.classId = i;
            result.confidence = data[i];
            oResult.push_back(result);
        }
        break;
    }
    default:
        spdlog::error("[YOLO_V8]: Not support model type.");
    }
    return RET_OK;

}


char* YOLO_V8::WarmUpSession() {
    clock_t starttime_1 = clock();
    cv::Mat iImg = cv::Mat(cv::Size(imgSize.at(0), imgSize.at(1)), CV_8UC3);
    cv::Mat processedImg;
    PreProcess(iImg, imgSize, processedImg);
    if (modelType < 4)
    {
        float* blob = new float[iImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> YOLO_input_node_dims = { 1, 3, imgSize.at(0), imgSize.at(1) };
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1),
            YOLO_input_node_dims.data(), YOLO_input_node_dims.size());
        auto output_tensors = session->Run(options, inputNodeNames.data(), &input_tensor, 1, outputNodeNames.data(),
            outputNodeNames.size());
        delete[] blob;
        clock_t starttime_4 = clock();
        double post_process_time = (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
        if (cudaEnable)
        {
            spdlog::info("[YOLO_V8(CUDA)]: Cuda warm-up cost {} ms.", post_process_time);
        }
    }
    else
    {
#ifdef USE_CUDA
        half* blob = new half[iImg.total() * 3];
        BlobFromImage(processedImg, blob);
        std::vector<int64_t> YOLO_input_node_dims = { 1,3,imgSize.at(0),imgSize.at(1) };
        Ort::Value input_tensor = Ort::Value::CreateTensor<half>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1), YOLO_input_node_dims.data(), YOLO_input_node_dims.size());
        auto output_tensors = session->Run(options, inputNodeNames.data(), &input_tensor, 1, outputNodeNames.data(), outputNodeNames.size());
        delete[] blob;
        clock_t starttime_4 = clock();
        double post_process_time = (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
        if (cudaEnable)
        {
            std::cout << "[YOLO_V8(CUDA)]: " << "Cuda warm-up cost " << post_process_time << " ms. " << std::endl;
        }
#endif
    }
    return RET_OK;
}