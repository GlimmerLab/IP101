#include "retinex_msrcr.hpp"
#include <cmath>
#include <omp.h>

namespace ip101 {
namespace advanced {

void retinex_msrcr(const cv::Mat& src, cv::Mat& dst,
                  double sigma1, double sigma2, double sigma3,
                  double alpha, double beta, double gain) {
    CV_Assert(!src.empty() && src.channels() == 3);

    // 转换到对数域
    cv::Mat log_src;
    src.convertTo(log_src, CV_32F);
    log_src += 1.0;
    cv::log(log_src, log_src);

    // 分离通道
    std::vector<cv::Mat> channels;
    cv::split(log_src, channels);

    // 对每个通道进行多尺度Retinex处理
    std::vector<cv::Mat> msr_channels(3);
    #pragma omp parallel for
    for(int c = 0; c < 3; c++) {
        cv::Mat msr = cv::Mat::zeros(src.size(), CV_32F);

        // 多尺度高斯滤波
        for(double sigma : {sigma1, sigma2, sigma3}) {
            cv::Mat gaussian;
            cv::GaussianBlur(channels[c], gaussian, cv::Size(0, 0), sigma);
            msr += log_src - gaussian;
        }
        msr /= 3.0;

        // 颜色恢复
        msr_channels[c] = alpha * (cv::log(channels[c] + 1.0) - cv::log(msr + 1.0)) + beta;
    }

    // 合并通道
    cv::Mat msr_result;
    cv::merge(msr_channels, msr_result);

    // 增益调整和归一化
    msr_result *= gain;
    cv::exp(msr_result, msr_result);
    msr_result -= 1.0;

    // 裁剪到[0, 255]范围
    msr_result.convertTo(dst, CV_8UC3, 255.0);
}

} // namespace advanced
} // namespace ip101