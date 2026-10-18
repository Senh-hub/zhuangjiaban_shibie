#include "xiangji_canshu.hpp"

#include <stdexcept>

XiangjiCanshu du_xiangji_canshu(const std::string& path)
{
    cv::FileStorage fs(path, cv::FileStorage::READ);

    if (!fs.isOpened())
    {
        throw std::runtime_error("无法打开相机配置文件: " + path);
    }

    XiangjiCanshu c;

    fs["tupian_kuan"] >> c.tupian_kuan;
    fs["tupian_gao"] >> c.tupian_gao;
    fs["xiangji_name"] >> c.xiangji_name;
    fs["neican_juzhen"] >> c.neican_juzhen;
    fs["jibian_canshu"] >> c.jibian_canshu;
    fs["chongtouying_wucha"] >> c.chongtouying_wucha;

    if (c.neican_juzhen.empty())
    {
        throw std::runtime_error("neican_juzhen 为空");
    }

    if (c.jibian_canshu.empty())
    {
        throw std::runtime_error("jibian_canshu 为空");
    }

    return c;
}
