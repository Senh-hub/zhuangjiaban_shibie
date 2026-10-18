#include <filesystem>
#include <iostream>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include "huizhi.hpp"
#include "nn_shibie.hpp"
#include "pnp_jiesuan.hpp"
#include "xiangji_canshu.hpp"

int main()
{
    const std::string model_path =
        "model/armor_v8.onnx";

    NnShibie detector(
        model_path,
        0.50f,
        0.20f
    );

    if (!detector.keyong())
    {
        std::cerr
            << "缺少NN模型: "
            << model_path
            << "\n";

        return 1;
    }

    cv::VideoCapture video(
        "data/shipin/shipin.avi"
    );

    if (!video.isOpened())
    {
        std::cerr
            << "无法打开视频\n";

        return 1;
    }

    auto c =
        du_xiangji_canshu(
            "config/xiangji_tupian.yaml"
        );

    PnpJiesuan pnp(c);

    double fps =
        video.get(cv::CAP_PROP_FPS);

    int kuan =
        static_cast<int>(
            video.get(cv::CAP_PROP_FRAME_WIDTH)
        );

    int gao =
        static_cast<int>(
            video.get(cv::CAP_PROP_FRAME_HEIGHT)
        );

    if (fps <= 0.0)
    {
        fps = 30.0;
    }

    std::filesystem::create_directories(
        "result"
    );

    cv::VideoWriter writer(
        "result/nn_result.mp4",
        cv::VideoWriter::fourcc(
            'm', 'p', '4', 'v'
        ),
        fps,
        cv::Size(kuan, gao)
    );

    cv::Mat frame;
    int frame_id = 0;

    while (video.read(frame))
    {
        ++frame_id;

        auto ban =
            detector.jiance(frame);

        // 当前测试视频只保留红色装甲板
        // NN颜色定义：
        // 0 = 蓝
        // 1 = 红
        // 2 = 白
        // 3 = 紫

        ban.erase(
            std::remove_if(
                ban.begin(),
                ban.end(),

                [](const Zhuangjiaban& b)
                {
                    return b.color != 1;
                }
            ),

            ban.end()
        );

        std::vector<PnpJieguo> pnp_jg;
        pnp_jg.reserve(ban.size());

        for (const auto& b : ban)
        {
            pnp_jg.push_back(
                pnp.jiesuan(b)
            );
        }

        cv::Mat out =
            frame.clone();

        hua_ban(
            out,
            ban,
            &pnp_jg
        );

        if (writer.isOpened())
        {
            writer.write(out);
        }

        if (frame_id % 100 == 0)
        {
            std::cout
                << "frame "
                << frame_id
                << ", armor "
                << ban.size()
                << "\n";
        }
    }

    video.release();
    writer.release();

    std::cout
        << "完成: result/nn_result.mp4\n";

    return 0;
}
