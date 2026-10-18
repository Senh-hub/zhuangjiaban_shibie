#include "nn_shibie.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <vector>

#include <opencv2/dnn.hpp>


// ============================================================
// 工具函数
// ============================================================

namespace
{

// 有些模型输出本身已经是 0~1 概率，
// 有些模型可能输出 logit。
// 如果已经在 0~1 范围，直接使用；
// 否则补一次 sigmoid。
float zhuan_score(float v)
{
    if (
        v >= 0.0f &&
        v <= 1.0f
    )
    {
        return v;
    }

    return
        1.0f /
        (
            1.0f +
            std::exp(-v)
        );
}

}


// ============================================================
// 构造函数
//
// 作用：
//
// armor_v8.onnx
//        ↓
// OpenVINO读取模型
//        ↓
// 编译到CPU
//        ↓
// 创建推理请求
// ============================================================

NnShibie::NnShibie(
    const std::string& model_path,
    float yuzhi,
    float nms
)
:
    yuzhi_(yuzhi),
    nms_(nms)
{
    // ========================================================
    // 1. 检查模型文件
    // ========================================================

    if (
        !std::filesystem::exists(
            model_path
        )
    )
    {
        keyong_ = false;

        return;
    }


    // ========================================================
    // 2. OpenVINO读取ONNX模型
    // ========================================================

    std::shared_ptr<ov::Model> model =
        core_.read_model(
            model_path
        );


    // ========================================================
    // 3. 检查输入形状
    //
    // 当前Infantry模型应该：
    //
    // 1 x 3 x 480 x 640
    //
    // N C H W
    // ========================================================

    const ov::Shape input_shape =
        model->input().get_shape();


    if (
        input_shape.size() != 4
    )
    {
        throw std::runtime_error(
            "NN输入维度不是4维"
        );
    }


    if (
        input_shape[1] != 3
    )
    {
        throw std::runtime_error(
            "NN输入通道数不是3"
        );
    }


    input_gao_ =
        static_cast<int>(
            input_shape[2]
        );


    input_kuan_ =
        static_cast<int>(
            input_shape[3]
        );


    // ========================================================
    // 4. 编译模型
    //
    // 先固定使用CPU。
    //
    // 后面如果需要性能优化，
    // 再考虑GPU。
    // ========================================================

    compiled_ =
        core_.compile_model(
            model,
            "CPU"
        );


    // ========================================================
    // 5. 创建推理请求
    // ========================================================

    request_ =
        compiled_.create_infer_request();


    keyong_ = true;
}


// ============================================================
// 判断NN是否可以使用
// ============================================================

bool NnShibie::keyong() const
{
    return keyong_;
}


// ============================================================
// 图像预处理：Letterbox
//
// 原图：
// 例如 1920 × 1080
//
//             ↓
//
// 等比例缩放
//
//             ↓
//
// 640 × 480画布
//
// 不拉伸原图比例。
//
// s  = 缩放比例
// px = 左右填充
// py = 上下填充
//
// 后面模型输出的角点还要靠这三个值恢复到原图。
// ============================================================

cv::Mat NnShibie::yuchuli(
    const cv::Mat& tu,
    float& s,
    int& px,
    int& py
) const
{
    if (tu.empty())
    {
        return {};
    }


    // ========================================================
    // 1. 计算缩放比例
    // ========================================================

    float sx =
        input_kuan_ /
        static_cast<float>(
            tu.cols
        );


    float sy =
        input_gao_ /
        static_cast<float>(
            tu.rows
        );


    s =
        std::min(
            sx,
            sy
        );


    // ========================================================
    // 2. 新尺寸
    // ========================================================

    int new_kuan =
        static_cast<int>(
            std::round(
                tu.cols * s
            )
        );


    int new_gao =
        static_cast<int>(
            std::round(
                tu.rows * s
            )
        );


    // ========================================================
    // 3. 缩放
    // ========================================================

    cv::Mat resize_tu;


    cv::resize(
        tu,
        resize_tu,
        cv::Size(
            new_kuan,
            new_gao
        )
    );


    // ========================================================
    // 4. 计算填充
    // ========================================================

    px =
        (
            input_kuan_ -
            new_kuan
        )
        / 2;


    py =
        (
            input_gao_ -
            new_gao
        )
        / 2;


    // ========================================================
    // 5. 建立640×480背景
    // ========================================================

    cv::Mat canvas(
        input_gao_,
        input_kuan_,
        CV_8UC3,
        cv::Scalar(
            114,
            114,
            114
        )
    );


    // ========================================================
    // 6. 把缩放后的图放进去
    // ========================================================

    resize_tu.copyTo(
        canvas(
            cv::Rect(
                px,
                py,
                new_kuan,
                new_gao
            )
        )
    );


    return canvas;
}


// ============================================================
// 根据四角点比例判断大小装甲板
//
// 这里目前是辅助判断。
//
// 后面如果模型类别本身能够明确决定大小装甲板，
// 可以继续优化。
// ============================================================

BanType NnShibie::panduan_type(
    const std::array<cv::Point2f, 4>& p
)
{
    // ========================================================
    // 四角点顺序：
    //
    // p[0] 左上
    // p[1] 左下
    // p[2] 右下
    // p[3] 右上
    // ========================================================


    // 上下两条横边平均长度
    float kuan =
        0.5f
        *
        (
            static_cast<float>(
                cv::norm(
                    p[0] -
                    p[3]
                )
            )

            +

            static_cast<float>(
                cv::norm(
                    p[1] -
                    p[2]
                )
            )
        );


    // 左右两条竖边平均长度
    float gao =
        0.5f
        *
        (
            static_cast<float>(
                cv::norm(
                    p[0] -
                    p[1]
                )
            )

            +

            static_cast<float>(
                cv::norm(
                    p[3] -
                    p[2]
                )
            )
        );


    float bili =
        kuan /
        (
            gao +
            1e-6f
        );


    if (
        bili >
        3.2f
    )
    {
        return BanType::DA;
    }


    return BanType::XIAO;
}


// ============================================================
// NN检测主函数
// ============================================================

std::vector<Zhuangjiaban>
NnShibie::jiance(
    const cv::Mat& tu
)
{
    if (
        !keyong_ ||
        tu.empty()
    )
    {
        return {};
    }


    // ========================================================
    // 第一部分
    //
    // 原图
    // ↓
    // letterbox
    // ↓
    // 640×480
    // ========================================================

    float s = 1.0f;

    int px = 0;

    int py = 0;


    cv::Mat input =
        yuchuli(
            tu,
            s,
            px,
            py
        );


    if (input.empty())
    {
        return {};
    }


    // ========================================================
    // 第二部分
    //
    // OpenCV Mat
    // BGR
    // HWC
    //
    // 转成：
    //
    // OpenVINO Tensor
    // RGB
    // NCHW
    // float32
    // 0~1
    //
    //
    // Tensor结构：
    //
    // [1][3][H][W]
    //
    // channel 0 = R
    // channel 1 = G
    // channel 2 = B
    // ========================================================


    ov::Shape tensor_shape = {

        1,

        3,

        static_cast<std::size_t>(
            input_gao_
        ),

        static_cast<std::size_t>(
            input_kuan_
        )
    };


    ov::Tensor input_tensor(
        ov::element::f32,
        tensor_shape
    );


    float* data =
        input_tensor.data<float>();


    const int area =
        input_gao_ *
        input_kuan_;


    // ========================================================
    // HWC BGR
    //
    // →
    //
    // CHW RGB
    // ========================================================

    for (
        int y = 0;
        y < input_gao_;
        ++y
    )
    {
        const cv::Vec3b* row =
            input.ptr<cv::Vec3b>(
                y
            );


        for (
            int x = 0;
            x < input_kuan_;
            ++x
        )
        {
            const cv::Vec3b& pixel =
                row[x];


            int id =
                y *
                input_kuan_
                +
                x;


            // OpenCV：
            // pixel[0] B
            // pixel[1] G
            // pixel[2] R


            // R
            data[id] =
                pixel[2]
                /
                255.0f;


            // G
            data[
                area +
                id
            ] =
                pixel[1]
                /
                255.0f;


            // B
            data[
                area * 2 +
                id
            ] =
                pixel[0]
                /
                255.0f;
        }
    }


    // ========================================================
    // 第三部分
    //
    // 把输入送给OpenVINO
    // ========================================================

    request_.set_input_tensor(
        input_tensor
    );


    // ========================================================
    // 真正执行神经网络
    // ========================================================

    request_.infer();


    // ========================================================
    // 第四部分
    //
    // 获取模型输出
    // ========================================================

    ov::Tensor output =
        request_.get_output_tensor();


    ov::Shape shape =
        output.get_shape();


    // ========================================================
    // 预期：
    //
    // [1, 21, 6300]
    //
    // 最后一个6300有可能因为模型版本略有变化，
    // 所以这里只要求：
    //
    // 三维
    // +
    // 第二维 = 21
    // ========================================================

    if (
        shape.size() != 3
    )
    {
        throw std::runtime_error(
            "NN输出不是3维"
        );
    }


    if (
        shape[0] != 1 ||
        shape[1] != 21
    )
    {
        throw std::runtime_error(
            "NN输出不是预期的 [1,21,N]"
        );
    }


    int num =
        static_cast<int>(
            shape[2]
        );


    std::size_t total =
        21
        *
        static_cast<std::size_t>(
            num
        );


    // ========================================================
    // 第五部分
    //
    // OpenVINO模型可能输出：
    //
    // FP32
    //
    // 或者
    //
    // FP16
    //
    // 统一转换成float。
    // ========================================================

    std::vector<float> out_data(
        total
    );


    if (
        output.get_element_type()
        ==
        ov::element::f32
    )
    {
        const float* src =
            output.data<const float>();


        std::copy(
            src,
            src + total,
            out_data.begin()
        );
    }

    else if (
        output.get_element_type()
        ==
        ov::element::f16
    )
    {
        const ov::float16* src =
            output.data<const ov::float16>();


        for (
            std::size_t i = 0;
            i < total;
            ++i
        )
        {
            out_data[i] =
                static_cast<float>(
                    src[i]
                );
        }
    }

    else
    {
        throw std::runtime_error(
            "NN输出数据类型不是FP32或FP16"
        );
    }


    // ========================================================
    // 方便读取：
    //
    // at(通道, 候选编号)
    //
    // 比如：
    //
    // at(0, 100)
    //
    // 表示：
    //
    // 第100个候选的第0通道
    // ========================================================

    auto at =
        [&](int channel, int id)
        {
            return
                out_data[
                    static_cast<std::size_t>(
                        channel
                    )
                    *
                    num
                    +
                    id
                ];
        };


    // ========================================================
    // 第六部分
    //
    // 解析所有候选
    // ========================================================

    std::vector<Zhuangjiaban>
        houxuan;


    std::vector<cv::Rect>
        boxes;


    std::vector<float>
        scores;


    for (
        int i = 0;
        i < num;
        ++i
    )
    {
        // ====================================================
        // 颜色分类
        //
        // 0 蓝
        // 1 红
        // 2 白
        // 3 紫
        // ====================================================

        int color_id = 0;


        float color_score =
            zhuan_score(
                at(
                    0,
                    i
                )
            );


        for (
            int k = 1;
            k < 4;
            ++k
        )
        {
            float sc =
                zhuan_score(
                    at(
                        k,
                        i
                    )
                );


            if (
                sc >
                color_score
            )
            {
                color_score =
                    sc;


                color_id =
                    k;
            }
        }


        // ====================================================
        // 装甲板类别
        //
        // 通道4~12
        // 共9类
        // ====================================================

        int cls_id = 0;


        float cls_score =
            zhuan_score(
                at(
                    4,
                    i
                )
            );


        for (
            int k = 1;
            k < 9;
            ++k
        )
        {
            float sc =
                zhuan_score(
                    at(
                        4 + k,
                        i
                    )
                );


            if (
                sc >
                cls_score
            )
            {
                cls_score =
                    sc;


                cls_id =
                    k;
            }
        }


        // ====================================================
        // 综合置信度
        // ====================================================

        float conf =
            color_score
            *
            cls_score;


        if (
            conf <
            yuzhi_
        )
        {
            continue;
        }


        Zhuangjiaban ban;


        bool inside =
            true;


        // ====================================================
        // 四角点
        //
        // 通道：
        //
        // 13 x1
        // 14 y1
        //
        // 15 x2
        // 16 y2
        //
        // 17 x3
        // 18 y3
        //
        // 19 x4
        // 20 y4
        //
        //
        // 顺序统一：
        //
        // 左上
        // 左下
        // 右下
        // 右上
        // ====================================================

        for (
            int k = 0;
            k < 4;
            ++k
        )
        {
            float x =
                (
                    at(
                        13 + k * 2,
                        i
                    )
                    -
                    static_cast<float>(
                        px
                    )
                )
                /
                s;


            float y =
                (
                    at(
                        14 + k * 2,
                        i
                    )
                    -
                    static_cast<float>(
                        py
                    )
                )
                /
                s;


            ban.dian[k] =
                cv::Point2f(
                    x,
                    y
                );


            // 点完全跑出图像时，
            // 暂时认为这个候选无效。
            if (
                x < 0.0f ||
                x >=
                    static_cast<float>(
                        tu.cols
                    )
                ||
                y < 0.0f ||
                y >=
                    static_cast<float>(
                        tu.rows
                    )
            )
            {
                inside =
                    false;
            }
        }


        if (!inside)
        {
            continue;
        }


        // ====================================================
        // 生成普通矩形
        //
        // NMS使用
        // ====================================================

        std::vector<cv::Point2f> p(
            ban.dian.begin(),
            ban.dian.end()
        );


        cv::Rect box =
            cv::boundingRect(
                p
            );


        if (
            box.width <= 0 ||
            box.height <= 0
        )
        {
            continue;
        }


        // ====================================================
        // 中心点
        // ====================================================

        ban.center =
            (
                ban.dian[0]
                +
                ban.dian[1]
                +
                ban.dian[2]
                +
                ban.dian[3]
            )
            *
            0.25f;


        // ====================================================
        // 大小装甲板
        // ====================================================

        ban.type =
            panduan_type(
                ban.dian
            );


        // ====================================================
        // 结果属性
        // ====================================================

        ban.score =
            conf;


        ban.color =
            color_id;


        // ====================================================
        // 类别名字
        // ====================================================

        static const char*
        labels[9] = {

            "sentry",

            "hero",

            "engineer",

            "3",

            "4",

            "5",

            "outpost",

            "base_bottom",

            "base_top"
        };


        ban.number =
            labels[
                cls_id
            ];


        // ====================================================
        // 保存候选
        // ====================================================

        houxuan.push_back(
            ban
        );


        boxes.push_back(
            box
        );


        scores.push_back(
            conf
        );
    }


    // ========================================================
    // 第七部分：NMS
    //
    // 同一个装甲板可能产生多个非常接近的候选。
    //
    // NMS负责：
    //
    // 保留最好的
    // 删除重复框
    // ========================================================

    std::vector<int> keep;


    cv::dnn::NMSBoxes(
        boxes,
        scores,
        yuzhi_,
        nms_,
        keep
    );


    // ========================================================
    // 第八部分：最终结果
    // ========================================================

    std::vector<Zhuangjiaban>
        out;


    out.reserve(
        keep.size()
    );


    for (
        int id :
        keep
    )
    {
        out.push_back(
            houxuan[id]
        );
    }


    return out;
}