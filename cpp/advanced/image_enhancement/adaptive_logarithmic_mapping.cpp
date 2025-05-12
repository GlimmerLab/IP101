#include "adaptive_logarithmic_mapping.hpp"
#include <cmath>
#include <omp.h>

namespace ip101 {
namespace advanced {

void adaptive_logarithmic_mapping(const cv::Mat& src, cv::Mat& dst,
                                double bias, double max_scale) {
    CV_Assert(!src.empty() && src.channels() == 3);

    // 转换到浮点型
    cv::Mat float_src;
    src.convertTo(float_src, CV_32F);

    // 计算图像的最大亮度值
    double max_val;
    cv::minMaxLoc(float_src, nullptr, &max_val);

    // 计算自适应缩放因子
    double scale = max_scale / std::log10(max_val + 1.0);

    // 应用对数映射
    cv::Mat log_result;
    cv::log(float_src + 1.0, log_result);
    log_result *= scale;

    // 添加偏置并归一化
    log_result += bias;

    // 使用OpenMP加速处理
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < log_result.rows; i++) {
        for(int j = 0; j < log_result.cols; j++) {
            cv::Vec3f& pixel = log_result.at<cv::Vec3f>(i, j);
            pixel[0] = std::min(255.0f, std::max(0.0f, pixel[0]));
            pixel[1] = std::min(255.0f, std::max(0.0f, pixel[1]));
            pixel[2] = std::min(255.0f, std::max(0.0f, pixel[2]));
        }
    }

    log_result.convertTo(dst, CV_8UC3);
}

} // namespace advanced
} // namespace ip101