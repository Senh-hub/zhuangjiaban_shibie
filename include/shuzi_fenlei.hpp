#pragma once

#include <string>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include "leixing.hpp"

class ShuziFenlei
{
public:
    ShuziFenlei(
        const std::string& model_path,
        const std::string& label_path,
        float yuzhi = 0.70f
    );

    bool keyong() const;

    void fenlei(
        const cv::Mat& tu,
        std::vector<Zhuangjiaban>& ban
    );

private:
    cv::dnn::Net net_;
    std::vector<std::string> labels_;
    float yuzhi_ = 0.70f;
    bool keyong_ = false;

    cv::Mat tiqu(
        const cv::Mat& tu,
        const Zhuangjiaban& ban
    ) const;
};
