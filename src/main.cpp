#include <rclcpp/rclcpp.hpp>
#include "rclcomm.h"
#include <memory>

int main(int argc, char * argv[])
{
    // 初始化ROS2
    rclcpp::init(argc, argv);
    
    // 创建节点实例
    auto node = std::make_shared<RclComm>();
    
    // 启动标定
    node->start();

    // auto calibfunc = CalibFunc::getInstance();
    // IntrParams intr_params = calibfunc->getIntrParams(5);   //获取相机cam_id的内参
    // auto detaruco_info = node->detaruco_->detectArucoMarkers(5, intr_params); //检测相机cam_id拍摄到的Aruco标记
    
    // 开始事件循环
    rclcpp::spin(node);
    
    // 清理并退出
    rclcpp::shutdown();
    return 0;
}
