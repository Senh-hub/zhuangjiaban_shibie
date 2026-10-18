#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "chuantong_shibie.hpp"
#include "huizhi.hpp"
#include "nn_shibie.hpp"
#include "pnp_jiesuan.hpp"
#include "shuzi_fenlei.hpp"
#include "xiangji_canshu.hpp"

using namespace std;

struct Zhenzhi
{
    string file;
    string armor_name;
    string number;
    cv::Point3d xyz;
};

cv::Point2f touying_zhongxin(
    const Zhenzhi& z,
    const XiangjiCanshu& c
)
{
    vector<cv::Point3d> p3{
        z.xyz
    };

    vector<cv::Point2d> p2;

    cv::projectPoints(
        p3,
        cv::Vec3d(
            0.0,
            0.0,
            0.0
        ),
        cv::Vec3d(
            0.0,
            0.0,
            0.0
        ),
        c.neican_juzhen,
        c.jibian_canshu,
        p2
    );

    return cv::Point2f(
        static_cast<float>(
            p2[0].x
        ),
        static_cast<float>(
            p2[0].y
        )
    );
}

double pipei_fanwei(
    const Zhenzhi& z,
    const XiangjiCanshu& c
)
{
    double fx =
        c.neican_juzhen.at<double>(
            0,
            0
        );

    // 约两个小装甲板投影宽度，兼容大斜角和关键点偏差。
    double w =
        fx *
        0.135 /
        z.xyz.z;

    return
        std::clamp(
            w * 2.2,
            30.0,
            100.0
        );
}

int zhao_zuijin(
    const vector<Zhuangjiaban>& ban,
    const cv::Point2f& zhongxin,
    double max_dis,
    const set<int>& used,
    double& best_dis
)
{
    int best = -1;
    best_dis = 1e9;

    for (
        int i = 0;
        i < static_cast<int>(
            ban.size()
        );
        ++i
    )
    {
        if (
            used.count(i)
        )
        {
            continue;
        }

        double d =
            cv::norm(
                ban[i].center -
                zhongxin
            );

        if (
            d <= max_dis &&
            d < best_dis
        )
        {
            best_dis = d;
            best = i;
        }
    }

    return best;
}

void hua_raw(
    const cv::Mat& yuan,
    const string& title,
    const vector<Dengtiao>* deng,
    const vector<Zhuangjiaban>& ban,
    const string& path
)
{
    cv::Mat out =
        yuan.clone();

    if (deng != nullptr)
    {
        hua_deng(
            out,
            *deng
        );
    }

    hua_ban(
        out,
        ban,
        nullptr
    );

    cv::putText(
        out,
        title,
        cv::Point(
            20,
            30
        ),
        cv::FONT_HERSHEY_SIMPLEX,
        0.7,
        cv::Scalar(
            0,
            255,
            255
        ),
        2
    );

    cv::imwrite(
        path,
        out
    );
}

void xie_not_found(
    ofstream& csv,
    const string& method,
    const Zhenzhi& z
)
{
    csv
        << z.file << ","
        << method << ","
        << z.armor_name << ","
        << z.number << ","
        << "not_found"
        << ",,,,,"
        << z.xyz.x << ","
        << z.xyz.y << ","
        << z.xyz.z
        << ",,,,\n";
}

void pingce(
    const string& method,
    const cv::Mat& yuan,
    const vector<Zhuangjiaban>& ban,
    const vector<Zhenzhi>& truth,
    const XiangjiCanshu& c,
    const PnpJiesuan& pnp,
    ofstream& csv,
    const string& path
)
{
    cv::Mat out =
        yuan.clone();

    cv::putText(
        out,
        "METHOD: " + method,
        cv::Point(
            20,
            30
        ),
        cv::FONT_HERSHEY_SIMPLEX,
        0.7,
        cv::Scalar(
            0,
            255,
            255
        ),
        2
    );

    set<int> used;

    for (const auto& z : truth)
    {
        cv::Point2f gt =
            touying_zhongxin(
                z,
                c
            );

        cv::drawMarker(
            out,
            gt,
            cv::Scalar(
                255,
                0,
                255
            ),
            cv::MARKER_CROSS,
            14,
            2
        );

        double center_error = 0.0;

        int id =
            zhao_zuijin(
                ban,
                gt,
                pipei_fanwei(
                    z,
                    c
                ),
                used,
                center_error
            );

        if (id < 0)
        {
            cv::putText(
                out,
                z.armor_name +
                " NOT FOUND",
                gt +
                cv::Point2f(
                    8,
                    -8
                ),
                cv::FONT_HERSHEY_SIMPLEX,
                0.42,
                cv::Scalar(
                    0,
                    0,
                    255
                ),
                1
            );

            cout
                << method
                << " | "
                << z.file
                << " | "
                << z.armor_name
                << " | NOT FOUND\n";

            xie_not_found(
                csv,
                method,
                z
            );

            continue;
        }

        used.insert(
            id
        );

        Zhuangjiaban cur =
            ban[id];

        // 这三张图都是3号步兵，使用小装甲板。
        cur.type =
            BanType::XIAO;

        PnpJieguo jg =
            pnp.jiesuan(
                cur
            );

        for (
            int k = 0;
            k < 4;
            ++k
        )
        {
            cv::line(
                out,
                cur.dian[k],
                cur.dian[
                    (k + 1) % 4
                ],
                cv::Scalar(
                    0,
                    255,
                    0
                ),
                2
            );
        }

        if (!jg.ok)
        {
            csv
                << z.file << ","
                << method << ","
                << z.armor_name << ","
                << z.number << ","
                << "pnp_failed,"
                << cur.number << ","
                << cur.score << ","
                << center_error
                << ",,"
                << z.xyz.x << ","
                << z.xyz.y << ","
                << z.xyz.z
                << ",,,,\n";

            continue;
        }

        double dx =
            jg.tvec[0] -
            z.xyz.x;

        double dy =
            jg.tvec[1] -
            z.xyz.y;

        double dz =
            jg.tvec[2] -
            z.xyz.z;

        double err =
            std::sqrt(
                dx * dx +
                dy * dy +
                dz * dz
            );

        ostringstream text;

        text
            << z.armor_name
            << " num="
            << cur.number
            << " err="
            << fixed
            << setprecision(3)
            << err
            << "m";

        cv::putText(
            out,
            text.str(),
            cur.dian[0] +
            cv::Point2f(
                0,
                -7
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.42,
            cv::Scalar(
                0,
                255,
                255
            ),
            1
        );

        cout
            << fixed
            << setprecision(4)
            << "\n"
            << method
            << " | "
            << z.file
            << " | "
            << z.armor_name
            << "\n  number = "
            << cur.number
            << "  score = "
            << cur.score
            << "\n  center error = "
            << center_error
            << " px"
            << "\n  EST  = ("
            << jg.tvec[0] << ", "
            << jg.tvec[1] << ", "
            << jg.tvec[2] << ")"
            << "\n  TRUE = ("
            << z.xyz.x << ", "
            << z.xyz.y << ", "
            << z.xyz.z << ")"
            << "\n  D    = ("
            << dx << ", "
            << dy << ", "
            << dz << ")"
            << "\n  ERR3D = "
            << err
            << " m\n";

        csv
            << z.file << ","
            << method << ","
            << z.armor_name << ","
            << z.number << ","
            << "ok,"
            << cur.number << ","
            << cur.score << ","
            << center_error << ","
            << jg.tvec[0] << ","
            << jg.tvec[1] << ","
            << jg.tvec[2] << ","
            << z.xyz.x << ","
            << z.xyz.y << ","
            << z.xyz.z << ","
            << dx << ","
            << dy << ","
            << dz << ","
            << err << "\n";
    }

    cv::imwrite(
        path,
        out
    );
}

int main()
{
    try
    {
        filesystem::remove_all(
            "result/pingce"
        );

        filesystem::create_directories(
            "result/pingce"
        );

        XiangjiCanshu c =
            du_xiangji_canshu(
                "config/xiangji_tupian.yaml"
            );

        ChuantongShibie chuantong(
            "config/sibie_canshu_tupian.yaml"
        );

        ShuziFenlei classifier(
            "model/mlp.onnx",
            "model/label.txt",
            0.70f
        );

        NnShibie nn(
            "model/armor_v8.onnx",
            0.50f,
            0.20f
        );

        PnpJiesuan pnp(
            c
        );

        if (!classifier.keyong())
        {
            cerr
                << "MLP加载失败\n";

            return 1;
        }

        if (!nn.keyong())
        {
            cerr
                << "NN加载失败\n";

            return 1;
        }

        vector<Zhenzhi> all = {
            {
                "tu1.jpg",
                "armor_1",
                "3",
                {
                    -0.174340,
                     0.169600,
                     3.794632
                }
            },
            {
                "tu1.jpg",
                "armor_4",
                "3",
                {
                     0.119301,
                     0.183937,
                     3.762585
                }
            },

            {
                "tu2.jpg",
                "armor_1",
                "3",
                {
                     0.285747,
                     0.070019,
                     1.384573
                }
            },
            {
                "tu2.jpg",
                "armor_2",
                "3",
                {
                     0.009916,
                     0.097664,
                     1.281592
                }
            },

            {
                "tu3.jpg",
                "armor_3",
                "3",
                {
                     0.189233,
                     0.070265,
                     1.553395
                }
            },
            {
                "tu3.jpg",
                "armor_4",
                "3",
                {
                    -0.092414,
                     0.092079,
                     1.467008
                }
            }
        };

        ofstream csv(
            "result/pingce.csv"
        );

        csv
            << fixed
            << setprecision(6);

        csv
            << "file,method,armor,expected_number,"
            << "status,detected_number,score,"
            << "center_error_px,"
            << "x_est,y_est,z_est,"
            << "x_true,y_true,z_true,"
            << "dx,dy,dz,error3d\n";

        for (
            int i = 1;
            i <= 3;
            ++i
        )
        {
            string name =
                string("tu") +
                to_string(i) +
                ".jpg";

            cv::Mat tu =
                cv::imread(
                    "data/zhenzhi/" +
                    name
                );

            if (tu.empty())
            {
                cerr
                    << "无法读取 "
                    << name
                    << "\n";

                continue;
            }

            vector<Zhenzhi>
                truth;

            for (const auto& z : all)
            {
                if (z.file == name)
                {
                    truth.push_back(
                        z
                    );
                }
            }

            cout
                << "\n====================================\n"
                << "处理 "
                << name
                << "\n====================================\n";

            // ------------------------------------------------
            // 传统视觉
            // 真值仿真图只检测蓝灯，减少场景误检。
            // ------------------------------------------------

            vector<Zhuangjiaban>
                ban_c =
                    chuantong.jiance(
                        tu,
                        0
                    );

            classifier.fenlei(
                tu,
                ban_c
            );

            cout
                << "[traditional]\n"
                << "light count = "
                << chuantong.get_deng().size()
                << "\n"
                << "armor candidate count = "
                << ban_c.size()
                << "\n";

            for (
                int k = 0;
                k < static_cast<int>(
                    ban_c.size()
                );
                ++k
            )
            {
                cout
                    << "candidate "
                    << k
                    << " number="
                    << ban_c[k].number
                    << " score="
                    << ban_c[k].score
                    << " center=("
                    << ban_c[k].center.x
                    << ","
                    << ban_c[k].center.y
                    << ")\n";
            }

            hua_raw(
                tu,
                "TRADITIONAL RAW",
                &chuantong.get_deng(),
                ban_c,
                string(
                    "result/pingce/tu"
                ) +
                to_string(i) +
                "_traditional_raw.jpg"
            );

            pingce(
                "traditional",
                tu,
                ban_c,
                truth,
                c,
                pnp,
                csv,
                string(
                    "result/pingce/tu"
                ) +
                to_string(i) +
                "_traditional.jpg"
            );

            // ------------------------------------------------
            // NN
            // ------------------------------------------------

            vector<Zhuangjiaban>
                ban_nn =
                    nn.jiance(
                        tu
                    );

            cout
                << "\n[nn]\n"
                << "armor candidate count = "
                << ban_nn.size()
                << "\n";

            hua_raw(
                tu,
                "NN RAW",
                nullptr,
                ban_nn,
                string(
                    "result/pingce/tu"
                ) +
                to_string(i) +
                "_nn_raw.jpg"
            );

            pingce(
                "nn",
                tu,
                ban_nn,
                truth,
                c,
                pnp,
                csv,
                string(
                    "result/pingce/tu"
                ) +
                to_string(i) +
                "_nn.jpg"
            );
        }

        cout
            << "\n完成：result/pingce.csv\n";

        return 0;
    }
    catch (
        const exception& e
    )
    {
        cerr
            << "异常："
            << e.what()
            << "\n";

        return 1;
    }
}
