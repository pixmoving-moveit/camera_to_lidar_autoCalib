#include <rclcpp/rclcpp.hpp>
#include "rclcomm.h"
#include <memory>

int main(int argc, char * argv[])
{
    // 初始化ROS2
    rclcpp::init(argc, argv);
    
    // 创建节点实例
    auto node = std::make_shared<RclComm>();
    
    spdlog::info("标定程序启动");
    node->start();

    // 开始事件循环
    rclcpp::spin(node);
    
    // 清理并退出
    rclcpp::shutdown();
    return 0;
}
