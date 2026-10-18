#include "pnp_jiesuan.hpp"

PnpJiesuan::PnpJiesuan(
    const XiangjiCanshu& c
)
{
    K_ = c.neican_juzhen.clone();
    D_ = c.jibian_canshu.clone();
}

std::vector<cv::Point3f> PnpJiesuan::zuo_3d(
    BanType type
) const
{
    // 单位：米
    float kuan =
        type == BanType::DA ?
        0.225f :
        0.135f;

    float gao = 0.055f;

    float hw = kuan * 0.5f;
    float hh = gao * 0.5f;

    // 对应二维点：
    // 左上、左下、右下、右上
    return {
        {-hw, -hh, 0.0f},
        {-hw,  hh, 0.0f},
        { hw,  hh, 0.0f},
        { hw, -hh, 0.0f}
    };
}

PnpJieguo PnpJiesuan::jiesuan(
    const Zhuangjiaban& ban
) const
{
    PnpJieguo jg;

    if (ban.type == BanType::INVALID)
    {
        return jg;
    }

    std::vector<cv::Point2f> p2(
        ban.dian.begin(),
        ban.dian.end()
    );

    auto p3 =
        zuo_3d(ban.type);

    cv::Vec3d rvec;
    cv::Vec3d tvec;

    bool ok =
        cv::solvePnP(
            p3,
            p2,
            K_,
            D_,
            rvec,
            tvec,
            false,
            cv::SOLVEPNP_IPPE
        );

    jg.ok = ok;

    if (ok)
    {
        jg.rvec = rvec;
        jg.tvec = tvec;
    }

    return jg;
}
