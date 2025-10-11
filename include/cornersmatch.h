#ifndef CORNERSMATCH_H
#define CORNERSMATCH_H

#include <vector>
#include <cmath>
#include <limits>
#include <iostream>
#include <array>
#include "base_struct.h"

// ---------- 主功能：P→Q 子集匹配 ----------
struct MatchResult {
    double total_dist;
    std::vector<int> q_idx;   // 与 P 逐行对应
};


class CornersMatch
{
private:
    int marker_num;
    std::vector<int> marker_order;

    // ---------- 工具：Hungarian（匈牙利）算法 ----------
    void hungarian(const std::vector< std::vector<double> > & cost, std::vector<int>& assignment);

    // ---------- 主功能：P→Q 子集匹配 ----------
    MatchResult matchSubset3D(const std::vector<cv::Point3f>& P, const std::vector<PointT>& Q, double dummy_scale = 1.2);

    std::vector<BoardCloudCorners> Q_board_corners; // 预存各个标定板的3D角点

    std::vector<Board2DCorners> P_board_corners; // 预存各个标定板的2D角点

public:
    CornersMatch(/* args */);
    ~CornersMatch();

    static CornersMatch* getInstance();

    void init_cornersMatch(const ArucoInitParams& params);

    void getMatchResult(std::vector<BoardInfo>& result);
    void setBoardPointCloudCenters_Q(std::vector<BoardInfo>& boards);
    void setBoardArucoCenters_P(const BoardImgAruco& boards);
};


#endif // CORNERSMATCH_H