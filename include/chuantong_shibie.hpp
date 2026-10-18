#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "leixing.hpp"

class ChuantongShibie
{
public:
    explicit ChuantongShibie(
        const std::string& canshu_path
    );

    // target_color:
    // -1 = 红蓝都检测
    //  0 = 蓝
    //  1 = 红
    std::vector<Zhuangjiaban> jiance(
        const cv::Mat& tu,
        int target_color = -1
    );

    const cv::Mat& get_erzhi() const;
    const std::vector<Dengtiao>& get_deng() const;

private:
    int erzhi_yuzhi_ = 80;
    int color_cha_ = 12;

    // 真值图片可以放大后检测；视频配置不写时默认1
    float suofang_beishu_ = 1.0f;

    float deng_min_bili_ = 0.02f;
    float deng_max_bili_ = 0.50f;
    float deng_max_jiaodu_ = 35.0f;
    float deng_min_chang_ = 4.0f;
    float min_fill_ratio_ = 0.20f;

    float min_deng_bili_ = 0.45f;
    float max_deng_jiaodu_cha_ = 20.0f;
    float max_y_cha_ = 1.0f;

    float xiao_min_juli_ = 0.55f;
    float xiao_max_juli_ = 3.8f;
    float da_min_juli_ = 3.8f;
    float da_max_juli_ = 6.5f;
    float ban_max_jiaodu_ = 32.0f;

    cv::Mat erzhi_;
    std::vector<Dengtiao> deng_;

    cv::Mat yuchuli(
        const cv::Mat& tu
    ) const;

    std::vector<Dengtiao> zhao_deng(
        const cv::Mat& tu,
        const cv::Mat& erzhi
    ) const;

    bool shi_deng(
        const Dengtiao& d
    ) const;

    BanType shi_ban(
        const Dengtiao& a,
        const Dengtiao& b
    ) const;

    bool zhongjian_you_deng(
        const Dengtiao& a,
        const Dengtiao& b,
        const std::vector<Dengtiao>& deng
    ) const;

    float shuzi_zhengju(
        const cv::Mat& tu,
        const Dengtiao& left,
        const Dengtiao& right
    ) const;

    std::vector<Zhuangjiaban> pei_dui(
        const cv::Mat& tu,
        const std::vector<Dengtiao>& deng,
        int target_color
    ) const;
};
