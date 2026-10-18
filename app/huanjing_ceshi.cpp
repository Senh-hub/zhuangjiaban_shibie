#include <iostream>

#include <opencv2/opencv.hpp>

#include "xiangji_canshu.hpp"

int main()
{
    try
    {
        auto c =
            du_xiangji_canshu(
                "config/xiangji_tupian.yaml"
            );

        std::cout
            << "OpenCV: "
            << CV_VERSION
            << "\n";

        std::cout
            << "xiangji: "
            << c.xiangji_name
            << "\n";

        std::cout
            << "size: "
            << c.tupian_kuan
            << " x "
            << c.tupian_gao
            << "\n";

        std::cout
            << "K:\n"
            << c.neican_juzhen
            << "\n";

        std::cout
            << "D:\n"
            << c.jibian_canshu
            << "\n";

        cv::Mat tu =
            cv::imread(
                "data/zhenzhi/tu1.jpg"
            );

        if (tu.empty())
        {
            std::cout
                << "tu1.jpg 尚未放入 data/zhenzhi/"
                << "\n";
        }
        else
        {
            std::cout
                << "tu1: "
                << tu.cols
                << " x "
                << tu.rows
                << "\n";
        }
    }
    catch (const std::exception& e)
    {
        std::cerr
            << e.what()
            << "\n";

        return 1;
    }

    return 0;
}
