#include "shuzi_fenlei.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>

ShuziFenlei::ShuziFenlei(
    const std::string& model_path,
    const std::string& label_path,
    float yuzhi
)
: yuzhi_(yuzhi)
{
    if (
        !std::filesystem::exists(model_path) ||
        !std::filesystem::exists(label_path)
    )
    {
        return;
    }

    net_ = cv::dnn::readNetFromONNX(model_path);

    std::ifstream f(label_path);
    std::string line;

    while (std::getline(f, line))
    {
        if (!line.empty())
        {
            labels_.push_back(line);
        }
    }

    keyong_ =
        !net_.empty() &&
        !labels_.empty();
}

bool ShuziFenlei::keyong() const
{
    return keyong_;
}

cv::Mat ShuziFenlei::tiqu(
    const cv::Mat& tu,
    const Zhuangjiaban& ban
) const
{
    const int deng_chang = 12;
    const int warp_gao = 28;

    const int xiao_kuan = 32;
    const int da_kuan = 54;

    const cv::Size roi_size(20, 28);

    int warp_kuan =
        ban.type == BanType::DA ?
        da_kuan :
        xiao_kuan;

    int top_y =
        (warp_gao - deng_chang) / 2 - 1;

    int bottom_y =
        top_y + deng_chang;

    cv::Point2f src[4] = {
        ban.left.bottom,
        ban.left.top,
        ban.right.top,
        ban.right.bottom
    };

    cv::Point2f dst[4] = {
        cv::Point2f(0.0f, static_cast<float>(bottom_y)),
        cv::Point2f(0.0f, static_cast<float>(top_y)),
        cv::Point2f(static_cast<float>(warp_kuan - 1), static_cast<float>(top_y)),
        cv::Point2f(static_cast<float>(warp_kuan - 1), static_cast<float>(bottom_y))
    };

    cv::Mat m =
        cv::getPerspectiveTransform(src, dst);

    cv::Mat number;

    cv::warpPerspective(
        tu,
        number,
        m,
        cv::Size(warp_kuan, warp_gao)
    );

    int x =
        (warp_kuan - roi_size.width) / 2;

    number =
        number(
            cv::Rect(
                x,
                0,
                roi_size.width,
                roi_size.height
            )
        ).clone();

    cv::cvtColor(
        number,
        number,
        cv::COLOR_BGR2GRAY
    );

    cv::threshold(
        number,
        number,
        0,
        255,
        cv::THRESH_BINARY | cv::THRESH_OTSU
    );

    return number;
}

void ShuziFenlei::fenlei(
    const cv::Mat& tu,
    std::vector<Zhuangjiaban>& ban
)
{
    if (!keyong_)
    {
        return;
    }

    for (auto& b : ban)
    {
        b.number_img = tiqu(tu, b);

        cv::Mat input;
        b.number_img.convertTo(
            input,
            CV_32F,
            1.0 / 255.0
        );

        cv::Mat blob;
        cv::dnn::blobFromImage(input, blob);

        net_.setInput(blob);

        cv::Mat out = net_.forward();

        cv::Mat flat =
            out.reshape(1, 1);

        double max_v = 0.0;
        cv::Point max_p;

        cv::minMaxLoc(
            flat,
            nullptr,
            &max_v,
            nullptr,
            &max_p
        );

        // softmax
        cv::Mat exp_v;
        cv::exp(flat - max_v, exp_v);

        double sum =
            cv::sum(exp_v)[0];

        cv::Mat prob =
            exp_v / sum;

        double conf = 0.0;

        cv::minMaxLoc(
            prob,
            nullptr,
            &conf,
            nullptr,
            &max_p
        );

        int id = max_p.x;

        b.score =
        static_cast<float>(conf);



        if (
            id >= 0 &&
            id < static_cast<int>(
                labels_.size()
            )
        )
{
    b.number =
        labels_[id];
}
else
{
    b.number =
        "?";
}
    }
}
