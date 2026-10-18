#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

#include "leixing.hpp"
#include "xiangji_canshu.hpp"

class PnpJiesuan
{
public:
    explicit PnpJiesuan(
        const XiangjiCanshu& c
    );

    PnpJieguo jiesuan(
        const Zhuangjiaban& ban
    ) const;

private:
    cv::Mat K_;
    cv::Mat D_;

    std::vector<cv::Point3f> zuo_3d(
        BanType type
    ) const;
};
