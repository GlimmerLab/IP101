#ifndef FAST_DEFOGGING_HPP
#define FAST_DEFOGGING_HPP

#include <opencv2/opencv.hpp>

namespace ip101 {
namespace advanced {

/**
 * @brief 基于最大对比度先验的快速去雾算法
 * @param src 输入图像
 * @param dst 输出图像
 * @param r 局部区域半径
 * @param lambda 对比度参数
 * @param t0 透射率下限
 */
void fast_max_contrast_defogging(const cv::Mat& src, cv::Mat& dst,
                                int r = 15,
                                double lambda = 0.2,
                                double t0 = 0.1);

/**
 * @brief 基于颜色线性转换的快速去雾算法
 * @param src 输入图像
 * @param dst 输出图像
 * @param alpha 对比度参数
 * @param beta 亮度参数
 */
void color_linear_transform_defogging(const cv::Mat& src, cv::Mat& dst,
                                     double alpha = 1.25,
                                     double beta = 20.0);

/**
 * @brief 基于对数反差增强的快速去雾算法
 * @param src 输入图像
 * @param dst 输出图像
 * @param gamma 伽马校正系数
 * @param gain 增益系数
 */
void logarithmic_enhancement_defogging(const cv::Mat& src, cv::Mat& dst,
                                      double gamma = 0.8,
                                      double gain = 2.0);

} // namespace advanced
} // namespace ip101

#endif // FAST_DEFOGGING_HPP