#include "real_time_adaptive_contrast.hpp"
#include <cmath>
#include <omp.h>

namespace ip101 {
namespace advanced {

void real_time_adaptive_contrast(const cv::Mat& src, cv::Mat& dst,
                               int window_size, double clip_limit) {
    CV_Assert(!src.empty());

    // 转换到浮点型
    cv::Mat float_src;
    src.convertTo(float_src, CV_32F);

    // 计算局部均值和标准差
    cv::Mat mean, stddev;
    cv::boxFilter(float_src, mean, CV_32F, cv::Size(window_size, window_size),
                 cv::Point(-1,-1), true, cv::BORDER_REFLECT);

    cv::Mat src_squared;
    cv::multiply(float_src, float_src, src_squared);
    cv::Mat mean_squared;
    cv::boxFilter(src_squared, mean_squared, CV_32F, cv::Size(window_size, window_size),
                 cv::Point(-1,-1), true, cv::BORDER_REFLECT);

    cv::Mat variance = mean_squared - mean.mul(mean);
    cv::sqrt(cv::max(variance, 0), stddev);

    // 计算自适应增益
    cv::Mat gain = 1.0 + clip_limit * (stddev / (mean + 1e-6));

    // 应用增益
    cv::Mat result = float_src.mul(gain);

    // 使用OpenMP加速处理
    #pragma omp parallel for
    for(int i = 0; i < result.rows; i++) {
        for(int j = 0; j < result.cols; j++) {
            cv::Vec3f& pixel = result.at<cv::Vec3f>(i, j);
            pixel[0] = cv::saturate_cast<uchar>(pixel[0]);
            pixel[1] = cv::saturate_cast<uchar>(pixel[1]);
            pixel[2] = cv::saturate_cast<uchar>(pixel[2]);
        }
    }

    // 转换回8位图像
    result.convertTo(dst, CV_8UC3);
}

} // namespace advanced
} // namespace ip101