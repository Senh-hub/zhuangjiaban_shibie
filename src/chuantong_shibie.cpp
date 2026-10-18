#include "chuantong_shibie.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

template <typename T>
void du_or(
    const cv::FileStorage& fs,
    const char* key,
    T& value
)
{
    cv::FileNode n = fs[key];

    if (!n.empty())
    {
        n >> value;
    }
}

// 用轮廓拟合真实长轴。
// 比只看 minAreaRect 的 width/height 更适合远距离少像素灯条。
Dengtiao zuo_deng(
    const std::vector<cv::Point>& contour,
    const cv::RotatedRect& rect
)
{
    Dengtiao d;
    d.rect = rect;

    d.kuan = std::min(
        rect.size.width,
        rect.size.height
    );

    cv::Vec4f line;

    cv::fitLine(
        contour,
        line,
        cv::DIST_L2,
        0,
        0.01,
        0.01
    );

    cv::Point2f dir(
        line[0],
        line[1]
    );

    float len =
        std::sqrt(
            dir.x * dir.x +
            dir.y * dir.y
        );

    if (len < 1e-6f)
    {
        return d;
    }

    dir.x /= len;
    dir.y /= len;

    cv::Point2f p0(
        line[2],
        line[3]
    );

    float min_t =
        std::numeric_limits<float>::max();

    float max_t =
        -std::numeric_limits<float>::max();

    for (const auto& p : contour)
    {
        cv::Point2f q(
            static_cast<float>(p.x),
            static_cast<float>(p.y)
        );

        float t =
            (q.x - p0.x) * dir.x +
            (q.y - p0.y) * dir.y;

        min_t = std::min(
            min_t,
            t
        );

        max_t = std::max(
            max_t,
            t
        );
    }

    cv::Point2f a =
        p0 + dir * min_t;

    cv::Point2f b =
        p0 + dir * max_t;

    d.chang =
        static_cast<float>(
            cv::norm(a - b)
        );

    // 极少像素情况下 fitLine 长度可能偏短，
    // 用旋转矩形长边兜底。
    d.chang =
        std::max(
            d.chang,
            std::max(
                rect.size.width,
                rect.size.height
            )
        );

    if (a.y <= b.y)
    {
        d.top = a;
        d.bottom = b;
    }
    else
    {
        d.top = b;
        d.bottom = a;
    }

    // 相对于竖直方向：
    // 0度=竖直，90度=水平
    d.jiaodu =
        static_cast<float>(
            std::atan2(
                std::abs(dir.x),
                std::abs(dir.y) + 1e-6f
            ) * 180.0 / CV_PI
        );

    return d;
}

cv::Rect kuo_roi(
    const cv::Rect& r,
    const cv::Size& size,
    int dx,
    int dy
)
{
    cv::Rect q(
        r.x - dx,
        r.y - dy,
        r.width + dx * 2,
        r.height + dy * 2
    );

    return
        q &
        cv::Rect(
            0,
            0,
            size.width,
            size.height
        );
}

void huifu_deng(
    Dengtiao& d,
    float s
)
{
    if (std::abs(s - 1.0f) < 1e-6f)
    {
        return;
    }

    d.top *= 1.0f / s;
    d.bottom *= 1.0f / s;

    d.rect.center *= 1.0f / s;
    d.rect.size.width /= s;
    d.rect.size.height /= s;

    d.chang /= s;
    d.kuan /= s;
}

void huifu_ban(
    Zhuangjiaban& b,
    float s
)
{
    if (std::abs(s - 1.0f) < 1e-6f)
    {
        return;
    }

    huifu_deng(
        b.left,
        s
    );

    huifu_deng(
        b.right,
        s
    );

    for (auto& p : b.dian)
    {
        p *= 1.0f / s;
    }

    b.center *= 1.0f / s;
}

}

ChuantongShibie::ChuantongShibie(
    const std::string& canshu_path
)
{
    cv::FileStorage fs(
        canshu_path,
        cv::FileStorage::READ
    );

    if (!fs.isOpened())
    {
        throw std::runtime_error(
            "无法打开传统视觉配置: " +
            canshu_path
        );
    }

    du_or(
        fs,
        "erzhi_yuzhi",
        erzhi_yuzhi_
    );

    du_or(
        fs,
        "color_cha",
        color_cha_
    );

    du_or(
        fs,
        "suofang_beishu",
        suofang_beishu_
    );

    du_or(
        fs,
        "deng_min_bili",
        deng_min_bili_
    );

    du_or(
        fs,
        "deng_max_bili",
        deng_max_bili_
    );

    du_or(
        fs,
        "deng_max_jiaodu",
        deng_max_jiaodu_
    );

    du_or(
        fs,
        "deng_min_chang",
        deng_min_chang_
    );

    du_or(
        fs,
        "min_fill_ratio",
        min_fill_ratio_
    );

    du_or(
        fs,
        "min_deng_bili",
        min_deng_bili_
    );

    du_or(
        fs,
        "max_deng_jiaodu_cha",
        max_deng_jiaodu_cha_
    );

    du_or(
        fs,
        "max_y_cha",
        max_y_cha_
    );

    du_or(
        fs,
        "xiao_min_juli",
        xiao_min_juli_
    );

    du_or(
        fs,
        "xiao_max_juli",
        xiao_max_juli_
    );

    du_or(
        fs,
        "da_min_juli",
        da_min_juli_
    );

    du_or(
        fs,
        "da_max_juli",
        da_max_juli_
    );

    du_or(
        fs,
        "ban_max_jiaodu",
        ban_max_jiaodu_
    );

    suofang_beishu_ =
        std::max(
            1.0f,
            suofang_beishu_
        );
}

// 混合预处理：
// 1. 灰度高亮：保留可能过曝成白色的灯条中心。
// 2. 蓝/红通道差：补充较暗但颜色明显的灯条。
// 传统 RM 检测常不能只靠颜色，因为灯条中心可能过曝。
cv::Mat ChuantongShibie::yuchuli(
    const cv::Mat& tu
) const
{
    cv::Mat gray;

    cv::cvtColor(
        tu,
        gray,
        cv::COLOR_BGR2GRAY
    );

    cv::Mat bright;

    cv::threshold(
        gray,
        bright,
        erzhi_yuzhi_,
        255,
        cv::THRESH_BINARY
    );

    std::vector<cv::Mat> ch;

    cv::split(
        tu,
        ch
    );

    cv::Mat blue_diff;
    cv::Mat red_diff;

    cv::subtract(
        ch[0],
        ch[2],
        blue_diff
    );

    cv::subtract(
        ch[2],
        ch[0],
        red_diff
    );

    cv::Mat blue_mask;
    cv::Mat red_mask;

    cv::threshold(
        blue_diff,
        blue_mask,
        color_cha_,
        255,
        cv::THRESH_BINARY
    );

    cv::threshold(
        red_diff,
        red_mask,
        color_cha_,
        255,
        cv::THRESH_BINARY
    );

    cv::Mat color_mask;

    cv::bitwise_or(
        blue_mask,
        red_mask,
        color_mask
    );

    // 颜色区最低亮度要求较宽松。
    cv::Mat color_light;

    cv::threshold(
        gray,
        color_light,
        std::max(
            20,
            erzhi_yuzhi_ / 2
        ),
        255,
        cv::THRESH_BINARY
    );

    cv::bitwise_and(
        color_mask,
        color_light,
        color_mask
    );

    cv::Mat mask;

    cv::bitwise_or(
        bright,
        color_mask,
        mask
    );

    // 只做轻微闭运算，不做 opening，
    // 避免远距离灯条被直接吃掉。
    cv::Mat kernel =
        cv::getStructuringElement(
            cv::MORPH_RECT,
            cv::Size(
                2,
                3
            )
        );

    cv::morphologyEx(
        mask,
        mask,
        cv::MORPH_CLOSE,
        kernel
    );

    return mask;
}

bool ChuantongShibie::shi_deng(
    const Dengtiao& d
) const
{
    if (d.chang < deng_min_chang_)
    {
        return false;
    }

    float bili =
        d.kuan /
        (
            d.chang +
            1e-6f
        );

    if (
        bili < deng_min_bili_ ||
        bili > deng_max_bili_
    )
    {
        return false;
    }

    if (
        d.jiaodu >
        deng_max_jiaodu_
    )
    {
        return false;
    }

    return true;
}

std::vector<Dengtiao>
ChuantongShibie::zhao_deng(
    const cv::Mat& tu,
    const cv::Mat& erzhi
) const
{
    std::vector<
        std::vector<cv::Point>
    > contours;

    cv::findContours(
        erzhi,
        contours,
        cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_SIMPLE
    );

    std::vector<Dengtiao> out;

    for (const auto& contour : contours)
    {
        if (contour.size() < 2)
        {
            continue;
        }

        double area =
            std::abs(
                cv::contourArea(
                    contour
                )
            );

        // 放大检测后面积会平方增长。
        double s2 =
            suofang_beishu_ *
            suofang_beishu_;

        if (
            area < 1.5 * s2 ||
            area > 6000.0 * s2
        )
        {
            continue;
        }

        cv::RotatedRect rect =
            cv::minAreaRect(
                contour
            );

        Dengtiao d =
            zuo_deng(
                contour,
                rect
            );

        if (!shi_deng(d))
        {
            continue;
        }

        float rect_area =
            std::max(
                1.0f,
                d.chang * d.kuan
            );

        float fill =
            static_cast<float>(
                area /
                rect_area
            );

        if (
            fill <
            min_fill_ratio_
        )
        {
            continue;
        }

        cv::Rect br =
            cv::boundingRect(
                contour
            );

        // 扩一点，看灯条外围的颜色光晕；
        // 白色数字通常没有明显蓝/红光晕。
        cv::Rect roi =
            kuo_roi(
                br,
                tu.size(),
                std::max(
                    2,
                    br.width
                ),
                std::max(
                    2,
                    br.height / 4
                )
            );

        if (
            roi.width <= 0 ||
            roi.height <= 0
        )
        {
            continue;
        }

        cv::Scalar mean =
            cv::mean(
                tu(roi)
            );

        double b =
            mean[0];

        double r =
            mean[2];

        double color_strength =
            std::abs(
                b - r
            );

        // 太接近纯白/纯灰：多半是数字或背景。
        if (
            color_strength < 3.0
        )
        {
            continue;
        }

        d.color =
            b > r ?
            0 :
            1;

        out.push_back(
            d
        );
    }

    std::sort(
        out.begin(),
        out.end(),
        [](
            const Dengtiao& a,
            const Dengtiao& b
        )
        {
            return
                a.rect.center.x <
                b.rect.center.x;
        }
    );

    return out;
}

bool ChuantongShibie::zhongjian_you_deng(
    const Dengtiao& a,
    const Dengtiao& b,
    const std::vector<Dengtiao>& deng
) const
{
    float left =
        std::min(
            a.rect.center.x,
            b.rect.center.x
        );

    float right =
        std::max(
            a.rect.center.x,
            b.rect.center.x
        );

    float ave_y =
        (
            a.rect.center.y +
            b.rect.center.y
        ) * 0.5f;

    float ave_len =
        (
            a.chang +
            b.chang
        ) * 0.5f;

    for (const auto& d : deng)
    {
        if (
            &d == &a ||
            &d == &b
        )
        {
            continue;
        }

        if (
            d.color == a.color &&
            d.rect.center.x > left &&
            d.rect.center.x < right &&
            std::abs(
                d.rect.center.y -
                ave_y
            ) < ave_len
        )
        {
            return true;
        }
    }

    return false;
}

BanType ChuantongShibie::shi_ban(
    const Dengtiao& a,
    const Dengtiao& b
) const
{
    if (a.color != b.color)
    {
        return BanType::INVALID;
    }

    float min_len =
        std::min(
            a.chang,
            b.chang
        );

    float max_len =
        std::max(
            a.chang,
            b.chang
        );

    float len_ratio =
        min_len /
        (
            max_len +
            1e-6f
        );

    if (
        len_ratio <
        min_deng_bili_
    )
    {
        return BanType::INVALID;
    }

    if (
        std::abs(
            a.jiaodu -
            b.jiaodu
        ) >
        max_deng_jiaodu_cha_
    )
    {
        return BanType::INVALID;
    }

    float ave_len =
        (
            a.chang +
            b.chang
        ) * 0.5f;

    float dx =
        std::abs(
            a.rect.center.x -
            b.rect.center.x
        );

    float dy =
        std::abs(
            a.rect.center.y -
            b.rect.center.y
        );

    if (
        dx < 1.0f
    )
    {
        return BanType::INVALID;
    }

    if (
        dy /
        (
            ave_len +
            1e-6f
        ) >
        max_y_cha_
    )
    {
        return BanType::INVALID;
    }

    float ban_angle =
        static_cast<float>(
            std::atan2(
                dy,
                dx
            ) *
            180.0 /
            CV_PI
        );

    if (
        ban_angle >
        ban_max_jiaodu_
    )
    {
        return BanType::INVALID;
    }

    float juli =
        dx /
        (
            ave_len +
            1e-6f
        );

    if (
        juli >= xiao_min_juli_ &&
        juli < xiao_max_juli_
    )
    {
        return BanType::XIAO;
    }

    if (
        juli >= da_min_juli_ &&
        juli < da_max_juli_
    )
    {
        return BanType::DA;
    }

    return BanType::INVALID;
}

// 检查两灯条之间有没有白色数字。
// 返回值越大，越像真正装甲板。
float ChuantongShibie::shuzi_zhengju(
    const cv::Mat& tu,
    const Dengtiao& left,
    const Dengtiao& right
) const
{
    float x1 =
        left.rect.center.x;

    float x2 =
        right.rect.center.x;

    if (x1 > x2)
    {
        std::swap(
            x1,
            x2
        );
    }

    float top =
        std::min(
            left.top.y,
            right.top.y
        );

    float bottom =
        std::max(
            left.bottom.y,
            right.bottom.y
        );

    cv::Rect roi(
        static_cast<int>(
            std::round(x1)
        ),
        static_cast<int>(
            std::round(top)
        ),
        std::max(
            1,
            static_cast<int>(
                std::round(
                    x2 - x1
                )
            )
        ),
        std::max(
            1,
            static_cast<int>(
                std::round(
                    bottom - top
                )
            )
        )
    );

    roi &=
        cv::Rect(
            0,
            0,
            tu.cols,
            tu.rows
        );

    if (
        roi.width < 2 ||
        roi.height < 2
    )
    {
        return 0.0f;
    }

    cv::Mat crop =
        tu(roi);

    cv::Mat gray;

    cv::cvtColor(
        crop,
        gray,
        cv::COLOR_BGR2GRAY
    );

    cv::Mat white;

    cv::threshold(
        gray,
        white,
        120,
        255,
        cv::THRESH_BINARY
    );

    float ratio =
        static_cast<float>(
            cv::countNonZero(
                white
            )
        ) /
        static_cast<float>(
            roi.area()
        );

    return
        std::min(
            ratio,
            1.0f
        );
}

std::vector<Zhuangjiaban>
ChuantongShibie::pei_dui(
    const cv::Mat& tu,
    const std::vector<Dengtiao>& deng,
    int target_color
) const
{
    struct Houxuan
    {
        std::size_t i;
        std::size_t j;
        BanType type;
        float score;
    };

    std::vector<Houxuan>
        houxuan;

    for (
        std::size_t i = 0;
        i < deng.size();
        ++i
    )
    {
        for (
            std::size_t j = i + 1;
            j < deng.size();
            ++j
        )
        {
            const auto& a =
                deng[i];

            const auto& b =
                deng[j];

            if (
                a.color != b.color
            )
            {
                continue;
            }

            if (
                target_color != -1 &&
                a.color != target_color
            )
            {
                continue;
            }

            BanType type =
                shi_ban(
                    a,
                    b
                );

            if (
                type ==
                BanType::INVALID
            )
            {
                continue;
            }

            if (
                zhongjian_you_deng(
                    a,
                    b,
                    deng
                )
            )
            {
                continue;
            }

            float min_len =
                std::min(
                    a.chang,
                    b.chang
                );

            float max_len =
                std::max(
                    a.chang,
                    b.chang
                );

            float len_ratio =
                min_len /
                (
                    max_len +
                    1e-6f
                );

            float ave_len =
                (
                    a.chang +
                    b.chang
                ) * 0.5f;

            float dx =
                std::abs(
                    a.rect.center.x -
                    b.rect.center.x
                );

            float dy =
                std::abs(
                    a.rect.center.y -
                    b.rect.center.y
                );

            float dist_ratio =
                dx /
                (
                    ave_len +
                    1e-6f
                );

            float angle_diff =
                std::abs(
                    a.jiaodu -
                    b.jiaodu
                );

            float white_ratio =
                shuzi_zhengju(
                    tu,
                    a,
                    b
                );

            // 小板的典型中心距比约2，大板约4.5。
            float target_dist =
                type == BanType::XIAO ?
                2.0f :
                4.5f;

            // 越小越好。
            // 数字证据越明显，扣分越多（即越优）。
            float score =
                (
                    1.0f -
                    len_ratio
                ) * 2.5f
                +
                (
                    dy /
                    (
                        ave_len +
                        1e-6f
                    )
                ) * 2.0f
                +
                angle_diff /
                20.0f
                +
                std::abs(
                    dist_ratio -
                    target_dist
                ) /
                target_dist
                -
                white_ratio * 2.0f;

            houxuan.push_back(
                {
                    i,
                    j,
                    type,
                    score
                }
            );
        }
    }

    std::sort(
        houxuan.begin(),
        houxuan.end(),
        [](
            const Houxuan& a,
            const Houxuan& b
        )
        {
            return
                a.score <
                b.score;
        }
    );

    std::vector<bool>
        used(
            deng.size(),
            false
        );

    std::vector<Zhuangjiaban>
        out;

    // 一根灯条最多参与一个最终装甲板。
    for (const auto& h : houxuan)
    {
        if (
            used[h.i] ||
            used[h.j]
        )
        {
            continue;
        }

        const Dengtiao* left =
            &deng[h.i];

        const Dengtiao* right =
            &deng[h.j];

        if (
            left->rect.center.x >
            right->rect.center.x
        )
        {
            std::swap(
                left,
                right
            );
        }

        Zhuangjiaban b;

        b.left =
            *left;

        b.right =
            *right;

        b.type =
            h.type;

        b.color =
            left->color;

        // 左上、左下、右下、右上
        b.dian = {
            left->top,
            left->bottom,
            right->bottom,
            right->top
        };

        b.center =
            (
                left->rect.center +
                right->rect.center
            ) * 0.5f;

        b.number =
            "?";

        b.score =
            0.0f;

        out.push_back(
            b
        );

        used[h.i] =
            true;

        used[h.j] =
            true;
    }

    return out;
}

std::vector<Zhuangjiaban>
ChuantongShibie::jiance(
    const cv::Mat& tu,
    int target_color
)
{
    if (tu.empty())
    {
        return {};
    }

    float s =
        std::max(
            1.0f,
            suofang_beishu_
        );

    cv::Mat work;

    if (s > 1.01f)
    {
        cv::resize(
            tu,
            work,
            cv::Size(),
            s,
            s,
            cv::INTER_CUBIC
        );
    }
    else
    {
        work =
            tu;
    }

    cv::Mat work_mask =
        yuchuli(
            work
        );

    std::vector<Dengtiao>
        work_deng =
            zhao_deng(
                work,
                work_mask
            );

    std::vector<Zhuangjiaban>
        ban =
            pei_dui(
                work,
                work_deng,
                target_color
            );

    if (s > 1.01f)
    {
        for (auto& d : work_deng)
        {
            huifu_deng(
                d,
                s
            );
        }

        for (auto& b : ban)
        {
            huifu_ban(
                b,
                s
            );
        }

        cv::resize(
            work_mask,
            erzhi_,
            tu.size(),
            0,
            0,
            cv::INTER_NEAREST
        );
    }
    else
    {
        erzhi_ =
            work_mask;
    }

    deng_ =
        work_deng;

    return ban;
}

const cv::Mat&
ChuantongShibie::get_erzhi() const
{
    return erzhi_;
}

const std::vector<Dengtiao>&
ChuantongShibie::get_deng() const
{
    return deng_;
}
