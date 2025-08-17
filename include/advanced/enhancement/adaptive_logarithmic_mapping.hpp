#ifndef ADAPTIVE_LOGARITHMIC_MAPPING_HPP
#define ADAPTIVE_LOGARITHMIC_MAPPING_HPP

#include <opencv2/opencv.hpp>

namespace ip101 {
namespace advanced {

/**
 * @brief 自适应对数映射算法实现
 * @param src 输入图像
 * @param dst 输出图像
 * @param bias 偏置参数
 * @param max_scale 最大缩放因子
 */
void adaptive_logarithmic_mapping(const cv::Mat& src, cv::Mat& dst,
                                double bias = 0.85, double max_scale = 100.0);

} // namespace advanced
} // namespace ip101

#endif // ADAPTIVE_LOGARITHMIC_MAPPING_HPP