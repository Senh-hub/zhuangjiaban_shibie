#include <algorithm>
#include <filesystem>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "chuantong_shibie.hpp"
#include "huizhi.hpp"
#include "pnp_jiesuan.hpp"
#include "shuzi_fenlei.hpp"
#include "xiangji_canshu.hpp"


int main()
{
    // ========================================================
    // 1. 视频路径
    // ========================================================

    const std::string video_path =
        "data/shipin/shipin.avi";


    cv::VideoCapture video(
        video_path
    );


    if (!video.isOpened())
    {
        std::cerr
            << "无法打开视频: "
            << video_path
            << std::endl;

        return 1;
    }


    // ========================================================
    // 2. 相机参数
    //
    // 注意：
    // 如果视频相机和真值图片相机不同，
    // 后面应该换成视频专用相机参数。
    // ========================================================

    XiangjiCanshu xiangji =
        du_xiangji_canshu(
            "config/xiangji_tupian.yaml"
        );


    // ========================================================
    // 3. 传统视觉检测器
    //
    // 视频继续使用原来的参数。
    // 不要使用真值图片专用参数。
    // ========================================================

    ChuantongShibie detector(
        "config/sibie_canshu.yaml"
    );


    // ========================================================
    // 4. PnP
    // ========================================================

    PnpJiesuan pnp(
        xiangji
    );


    // ========================================================
    // 5. MLP数字分类器
    // ========================================================

    ShuziFenlei classifier(
        "model/mlp.onnx",
        "model/label.txt",
        0.70f
    );


    if (!classifier.keyong())
    {
        std::cerr
            << "警告：MLP数字分类器加载失败"
            << std::endl;
    }


    // ========================================================
    // 6. 视频基本信息
    // ========================================================

    double fps =
        video.get(
            cv::CAP_PROP_FPS
        );


    int kuan =
        static_cast<int>(
            video.get(
                cv::CAP_PROP_FRAME_WIDTH
            )
        );


    int gao =
        static_cast<int>(
            video.get(
                cv::CAP_PROP_FRAME_HEIGHT
            )
        );


    if (fps <= 0.0)
    {
        fps = 30.0;
    }


    std::filesystem::create_directories(
        "result"
    );


    // ========================================================
    // 7. 输出视频
    // ========================================================

    cv::VideoWriter writer(

        "result/chuantong_result.mp4",

        cv::VideoWriter::fourcc(
            'm',
            'p',
            '4',
            'v'
        ),

        fps,

        cv::Size(
            kuan,
            gao
        )
    );


    if (!writer.isOpened())
    {
        std::cerr
            << "无法创建输出视频"
            << std::endl;

        return 1;
    }


    // ========================================================
    // 8. 逐帧处理
    // ========================================================

    cv::Mat frame;

    int frame_id = 0;


    while (
        video.read(
            frame
        )
    )
    {
        ++frame_id;


        // ====================================================
        // A. 传统视觉检测
        //
        // 当前真实视频主要是红色目标，
        // 所以直接检测红色。
        //
        // 0 = 蓝
        // 1 = 红
        // -1 = 红蓝都找
        // ====================================================

        std::vector<Zhuangjiaban> ban =
            detector.jiance(
                frame,
                1
            );


        // ====================================================
        // B. MLP数字分类
        // ====================================================

        if (
            classifier.keyong()
        )
        {
            classifier.fenlei(
                frame,
                ban
            );


            // ================================================
            // 视频最终输出时：
            //
            // ? / negative / 低置信度
            // 不认为是最终装甲板
            // ================================================

            ban.erase(

                std::remove_if(

                    ban.begin(),
                    ban.end(),

                    [](const Zhuangjiaban& b)
                    {
                        return
                            b.number == "?"
                            ||
                            b.number == "negative"
                            ||
                            b.score < 0.70f;
                    }
                ),

                ban.end()
            );
        }


        // ====================================================
        // C. PnP
        // ====================================================

        std::vector<PnpJieguo>
            pnp_jg;


        pnp_jg.reserve(
            ban.size()
        );


        for (
            const auto& b :
            ban
        )
        {
            pnp_jg.push_back(
                pnp.jiesuan(
                    b
                )
            );
        }


        // ====================================================
        // D. 绘制
        // ====================================================

        cv::Mat out =
            frame.clone();


        // 灯条
        hua_deng(
            out,
            detector.get_deng()
        );


        // 装甲板 + 数字 + xyz
        hua_ban(
            out,
            ban,
            &pnp_jg
        );


        // ====================================================
        // E. 写视频
        // ====================================================

        writer.write(
            out
        );


        // ====================================================
        // F. 终端进度
        // ====================================================

        if (
            frame_id % 100
            ==
            0
        )
        {
            std::cout
                << "frame "
                << frame_id
                << ", armor "
                << ban.size()
                << std::endl;
        }
    }


    // ========================================================
    // 9. 释放
    // ========================================================

    video.release();

    writer.release();


    std::cout
        << "完成: result/chuantong_result.mp4"
        << std::endl;


    return 0;
}