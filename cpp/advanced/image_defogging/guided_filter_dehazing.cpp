#include <advanced/filtering/guided_filter.hpp>
#include <advanced/defogging/dark_channel.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace ip101 {
namespace advanced {

// 注意：guided_filter函数已在 advanced_filtering/guided_filter.cpp 中定义
// 这里不再重复定义，直接使用头文件中的声明

void guided_filter_defogging(const cv::Mat& src, cv::Mat& dst, int radius, double eps, double omega, double t0) {
    CV_Assert(!src.empty() && src.channels() == 3);
    CV_Assert(radius > 0 && eps > 0 && omega > 0 && omega <= 1.0 && t0 > 0 && t0 < 1.0);

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 1. 计算暗通道
    cv::Mat dark;
    compute_dark_channel(src, dark, radius / 2); // 使用更小的半径计算暗通道

    // 2. 估计大气光照值
    cv::Vec3d A = estimate_atmospheric_light(src, dark);

    // 3. 估计粗透射率
    cv::Mat transmission = estimate_transmission(src, dark, A, omega);

    // 4. 使用导向滤波细化透射率
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    cv::Mat guidance, transmission_refined;
    gray.convertTo(guidance, CV_64FC1, 1.0 / 255.0);

    // 应用导向滤波
    guided_filter(transmission, guidance, transmission_refined, radius, eps);

    // 5. 限制最小透射率，避免过度去雾
    cv::max(transmission_refined, t0, transmission_refined);

    // 6. 利用大气散射模型恢复图像
    std::vector<cv::Mat> channels;
    cv::split(src, channels);

    std::vector<cv::Mat> recovered_channels(3);

    for (int c = 0; c < 3; c++) {
        // 转换到浮点数
        cv::Mat channel_f;
        channels[c].convertTo(channel_f, CV_64FC1);

        // 应用大气散射模型: J = (I - A) / t + A
        cv::Mat recovered = cv::Mat(src.size(), CV_64FC1);

        #pragma omp parallel for
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                double val = channel_f.at<double>(y, x);
                double t = transmission_refined.at<double>(y, x);

                // 恢复无雾图像
                recovered.at<double>(y, x) = (val - A[c]) / t + A[c];
            }
        }

        // 裁剪到[0, 255]范围
        cv::Mat recovered_8u;
        recovered.convertTo(recovered_8u, CV_8UC1);
        recovered_channels[c] = recovered_8u;
    }

    // 合并通道
    cv::merge(recovered_channels, dst);
}

// 实现Kaiming He的改进算法
void kaiming_he_guided_defogging(const cv::Mat& src, cv::Mat& dst, int radius, double eps, double omega, double t0) {
    CV_Assert(!src.empty() && src.channels() == 3);
    CV_Assert(radius > 0 && eps > 0 && omega > 0 && omega <= 1.0 && t0 > 0 && t0 < 1.0);

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 1. 计算暗通道
    cv::Mat dark;
    compute_dark_channel(src, dark, 15); // 使用固定大小的patch计算暗通道

    // 2. 估计大气光照值
    cv::Vec3d A = estimate_atmospheric_light(src, dark);

    // 3. 估计粗透射率
    cv::Mat transmission = estimate_transmission(src, dark, A, omega);

    // 4. 使用引导图像滤波优化透射率 - 使用原始图像作为引导图像
    cv::Mat src_float;
    src.convertTo(src_float, CV_64FC3, 1.0 / 255.0);

    cv::Mat transmission_refined;
    guided_filter(transmission, src_float, transmission_refined, radius, eps);

    // 5. 限制最小透射率
    cv::max(transmission_refined, t0, transmission_refined);

    // 6. 利用大气散射模型恢复图像
    std::vector<cv::Mat> channels;
    cv::split(src, channels);

    std::vector<cv::Mat> recovered_channels(3);

    for (int c = 0; c < 3; c++) {
        // 转换到浮点数
        cv::Mat channel_f;
        channels[c].convertTo(channel_f, CV_64FC1);

        // 应用大气散射模型: J = (I - A) / t + A
        cv::Mat recovered = cv::Mat(src.size(), CV_64FC1);

        #pragma omp parallel for
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                double val = channel_f.at<double>(y, x);
                double t = transmission_refined.at<double>(y, x);

                // 恢复无雾图像
                recovered.at<double>(y, x) = (val - A[c]) / t + A[c];
            }
        }

        // 裁剪到[0, 255]范围
        cv::Mat recovered_8u;
        recovered.convertTo(recovered_8u, CV_8UC1);
        recovered_channels[c] = recovered_8u;
    }

    // 合并通道
    cv::merge(recovered_channels, dst);
}

} // namespace advanced
} // namespace ip101