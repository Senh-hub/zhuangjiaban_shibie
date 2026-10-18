#pragma once

#include <string>
#include <opencv2/opencv.hpp>

struct XiangjiCanshu
{
    int tupian_kuan = 0;
    int tupian_gao = 0;
    std::string xiangji_name;

    cv::Mat neican_juzhen;
    cv::Mat jibian_canshu;

    double chongtouying_wucha = 0.0;
};

XiangjiCanshu du_xiangji_canshu(const std::string& path);
