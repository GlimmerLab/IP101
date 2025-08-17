#include <advanced/enhancement/automatic_color_equalization.hpp>
#include <cmath>
#include <omp.h>

namespace ip101 {
namespace advanced {

void automatic_color_equalization(const cv::Mat& src, cv::Mat& dst,
                                double alpha, double beta) {
    CV_Assert(!src.empty() && src.channels() == 3);

    // 转换到LAB颜色空间
    cv::Mat lab;
    cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);

    // 分离通道
    std::vector<cv::Mat> channels;
    cv::split(lab, channels);

    // 对L通道进行自适应对比度增强
    cv::Mat L = channels[0];
    cv::Scalar mean_L, stddev_L;
    cv::meanStdDev(L, mean_L, stddev_L);

    // 计算自适应增益
    double gain = alpha * (255.0 / (stddev_L[0] + beta));

    // 使用OpenMP加速处理
    #pragma omp parallel for
    for(int i = 0; i < L.rows; i++) {
        for(int j = 0; j < L.cols; j++) {
            uchar& pixel = L.at<uchar>(i, j);
            pixel = cv::saturate_cast<uchar>(
                mean_L[0] + gain * (pixel - mean_L[0]));
        }
    }

    // 合并通道
    channels[0] = L;
    cv::merge(channels, lab);

    // 转换回BGR颜色空间
    cv::cvtColor(lab, dst, cv::COLOR_Lab2BGR);

    // 确保结果在有效范围内
    #pragma omp parallel for
    for(int i = 0; i < dst.rows; i++) {
        for(int j = 0; j < dst.cols; j++) {
            cv::Vec3b& pixel = dst.at<cv::Vec3b>(i, j);
            pixel[0] = cv::saturate_cast<uchar>(pixel[0]);
            pixel[1] = cv::saturate_cast<uchar>(pixel[1]);
            pixel[2] = cv::saturate_cast<uchar>(pixel[2]);
        }
    }
}

} // namespace advanced
} // namespace ip101