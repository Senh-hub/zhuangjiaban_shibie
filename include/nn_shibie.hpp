#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include "leixing.hpp"

class NnShibie
{
public:
    NnShibie(
        const std::string& model_path,
        float yuzhi = 0.50f,
        float nms = 0.20f
    );

    bool keyong() const;

    std::vector<Zhuangjiaban> jiance(
        const cv::Mat& tu
    );

private:
    ov::Core core_;
    ov::CompiledModel compiled_;
    ov::InferRequest request_;

    float yuzhi_ = 0.50f;
    float nms_ = 0.20f;
    bool keyong_ = false;

    int input_kuan_ = 640;
    int input_gao_ = 480;

    cv::Mat yuchuli(
        const cv::Mat& tu,
        float& s,
        int& px,
        int& py
    ) const;

    static BanType panduan_type(
        const std::array<cv::Point2f, 4>& p
    );
};
