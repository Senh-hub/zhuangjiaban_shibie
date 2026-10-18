#include "huizhi.hpp"

#include <iomanip>
#include <sstream>

void hua_deng(
    cv::Mat& tu,
    const std::vector<Dengtiao>& deng
)
{
    for (const auto& d : deng)
    {
        cv::Scalar color =
            d.color == 0 ?
            cv::Scalar(255, 100, 0) :
            cv::Scalar(0, 100, 255);

        cv::line(
            tu,
            d.top,
            d.bottom,
            color,
            2
        );
    }
}

void hua_ban(
    cv::Mat& tu,
    const std::vector<Zhuangjiaban>& ban,
    const std::vector<PnpJieguo>* pnp
)
{
    for (std::size_t i = 0; i < ban.size(); ++i)
    {
        const auto& b = ban[i];

        for (int k = 0; k < 4; ++k)
        {
            cv::line(
                tu,
                b.dian[k],
                b.dian[(k + 1) % 4],
                cv::Scalar(0, 255, 0),
                2
            );

            cv::circle(
                tu,
                b.dian[k],
                3,
                cv::Scalar(0, 255, 255),
                cv::FILLED
            );
        }

        std::ostringstream ss;

        ss << b.number;

        if (b.score > 0.0f)
        {
            ss
                << " "
                << std::fixed
                << std::setprecision(2)
                << b.score;
        }

        if (
            pnp != nullptr &&
            i < pnp->size() &&
            (*pnp)[i].ok
        )
        {
            const auto& t =
                (*pnp)[i].tvec;

            ss
                << " xyz("
                << std::fixed
                << std::setprecision(2)
                << t[0] << ","
                << t[1] << ","
                << t[2] << ")";
        }

        cv::putText(
            tu,
            ss.str(),
            b.dian[0],
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            cv::Scalar(0, 255, 255),
            2
        );
    }
}
