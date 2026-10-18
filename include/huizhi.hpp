#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

#include "leixing.hpp"

void hua_deng(
    cv::Mat& tu,
    const std::vector<Dengtiao>& deng
);

void hua_ban(
    cv::Mat& tu,
    const std::vector<Zhuangjiaban>& ban,
    const std::vector<PnpJieguo>* pnp = nullptr
);
