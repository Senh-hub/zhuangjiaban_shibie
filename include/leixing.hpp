#pragma once

#include <array>
#include <string>
#include <opencv2/opencv.hpp>

enum class BanType
{
    XIAO,
    DA,
    INVALID
};

struct Dengtiao
{
    cv::RotatedRect rect;
    cv::Point2f top;
    cv::Point2f bottom;

    float chang = 0.0f;
    float kuan = 0.0f;
    float jiaodu = 0.0f;

    // 0 = blue, 1 = red
    int color = -1;
};

struct Zhuangjiaban
{
    Dengtiao left;
    Dengtiao right;

    // 顺序统一为：左上、左下、右下、右上
    std::array<cv::Point2f, 4> dian{};

    cv::Point2f center;
    BanType type = BanType::INVALID;

    std::string number = "?";
    float score = 0.0f;
    int color = -1;

    cv::Mat number_img;
};

struct PnpJieguo
{
    bool ok = false;
    cv::Vec3d rvec{0.0, 0.0, 0.0};
    cv::Vec3d tvec{0.0, 0.0, 0.0};
};
