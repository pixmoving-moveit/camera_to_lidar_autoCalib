/**
 * @file cornersmatch.cpp
 * @brief 使用匈牙利算法进行3D角点与2D图像特征点匹配的实现文件
 */

#include "cornersmatch.h"
#include <algorithm>
#include <cmath>
#include <utility>


CornersMatch::CornersMatch(/* args */)
{


}

CornersMatch::~CornersMatch()
{
}

void CornersMatch::init_cornersMatch(const ArucoInitParams& params)
{
    marker_num = params.marker_num;
    marker_order = params.marker_order;
    spdlog::info("[CornersMatch] 初始化完成, marker_order: {}", fmt::join(marker_order, ", "));
    
}

// 获取匹配结果,标定板的图像角点与点云3D角点，相互对应
void CornersMatch::getMatchResult(std::vector<BoardInfo>& result)
{
    result.clear();

    // 遍历P_board_corners中的每个2D角点
    for(size_t i = 0; i < P_board_corners.size(); ++i)
    {
        const auto& p_board = P_board_corners[i];
        
        // 在Q_board_corners中寻找对应的board_id
        bool found_match = false;
        for(size_t j = 0; j < Q_board_corners.size(); ++j)
        {
            const auto& q_board = Q_board_corners[j];
            
            // 如果board_id匹配
            if(p_board.board_id == q_board.board_id)
            {
                // 创建BoardInfo并添加到结果中
                BoardInfo board_info;
                board_info.board_id = p_board.board_id;
                board_info.corners_2d = p_board;
                board_info.corners_cloud = q_board;
                
                result.push_back(board_info);
                found_match = true;

                spdlog::info("[CornersMatch] 匹配成功 - board_id: {}", p_board.board_id);
                break; // 找到匹配后跳出内层循环
            }
        }
        
        if(!found_match)
        {
            spdlog::error("[CornersMatch] 警告: 未找到board_id {} 的3D匹配", p_board.board_id);
        }
    }

    spdlog::info("[CornersMatch] 匹配完成，成功匹配 {} 个标定板", result.size());
}

// 设置点云 Q（3D 点）所有通过点云检测到的标定板中心点
void CornersMatch::setBoardPointCloudCenters_Q(std::vector<BoardInfo>& boards)
{
    Q_board_corners.clear();
    
    if(boards.size() == marker_num)
    {
        // 创建索引和角度对的向量用于排序
        std::vector<std::pair<double, int>> angle_index_pairs;
        
        for(size_t i = 0; i < boards.size(); ++i)
        {
            const PointT& center = boards[i].corners_cloud.center_3d;
            
            // 计算与X负半轴的夹角(0-360度)
            // X负半轴方向为(-1, 0)，所以我们计算点(x, y)与(-1, 0)的夹角
            double angle = std::atan2(center.y, -center.x); // atan2返回[-π, π]
            
            // 将角度转换为[0, 2π]范围
            if(angle < 0)
            {
                angle += 2 * M_PI;
            }
            
            // 转换为度数便于理解（可选）
            double angle_degrees = angle * 180.0 / M_PI;
            
            angle_index_pairs.push_back(std::make_pair(angle, i)); 
        }
        
        // 按角度从小到大排序（顺时针方向）
        std::sort(angle_index_pairs.begin(), angle_index_pairs.end());
        
        // 按排序后的顺序填充Q和Q_board_corners，并设置board_id
        for(size_t i = 0; i < angle_index_pairs.size(); ++i)
        {
            int original_index = angle_index_pairs[i].second;   // 获取原始索引
            
            // 添加到Q_board_corners中
            BoardCloudCorners board_corners = boards[original_index].corners_cloud;
            board_corners.board_id = marker_order[i]; // 临时设置为排序后的索引
            Q_board_corners.push_back(board_corners);

            
            // 使用marker_order配置board_id
            if(i < marker_order.size())
            {
                boards[original_index].board_id = marker_order[i];
            }
            else
            {
                spdlog::error("[CornersMatch] marker_order配置不足，索引 {} 超出范围", i);
                boards[original_index].board_id = -1; // 设置为无效ID
            }
        }
        spdlog::info("[CornersMatch] 点云标定板中心点设置完成，共 {} 个", Q_board_corners.size());
    }
    else
    {
        spdlog::error("[CornersMatch] lidar 检测到的tags数量与配置参数不一致，无法进行配准");
        spdlog::error("[CornersMatch] 检测到: {} 个, 期望: {} 个", boards.size(), marker_num);
    }
}

void CornersMatch::setBoardArucoCenters_P(const BoardImgAruco& boards_temp)
{
    if(boards_temp.aruco_num <= marker_num)
    {
        P_board_corners.clear();
        P_board_corners = boards_temp.boards_2d;

    }
    else
    {
        spdlog::error("[CornersMatch] 相机检测到的tags数量超过配置参数，无法进行配准");
        spdlog::error("[CornersMatch] 检测到: {} 个, 期望最多: {} 个", boards_temp.aruco_num, marker_num);
        return;
    }
}


CornersMatch* CornersMatch::getInstance()
{
    static CornersMatch instance;
    return &instance;
}