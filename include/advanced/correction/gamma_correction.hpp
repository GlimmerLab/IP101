#ifndef GAMMA_CORRECTION_HPP
#define GAMMA_CORRECTION_HPP

#include <opencv2/opencv.hpp>

namespace ip101 {
namespace advanced {

/**
 * @brief 标准伽马校正
 * @param src 输入图像
 * @param dst 输出图像
 * @param gamma 伽马值，小于1时增亮，大于1时变暗
 */
void standard_gamma_correction(const cv::Mat& src, cv::Mat& dst,
                              double gamma = 1.0);

/**
 * @brief 二维伽马校正
 * @param src 输入图像
 * @param dst 输出图像
 * @param gamma_dark 暗区伽马值
 * @param gamma_bright 亮区伽马值
 * @param threshold 暗亮区分界阈值 (0-255)
 * @param smooth_factor 平滑过渡系数 (0-1)
 */
void two_dimensional_gamma_correction(const cv::Mat& src, cv::Mat& dst,
                                     double gamma_dark = 0.75,
                                     double gamma_bright = 1.25,
                                     int threshold = 128,
                                     double smooth_factor = 0.5);

/**
 * @brief 自适应伽马校正
 * @param src 输入图像
 * @param dst 输出图像
 * @param blocks 块划分数量
 */
void adaptive_gamma_correction(const cv::Mat& src, cv::Mat& dst,
                              int blocks = 4);

} // namespace advanced
} // namespace ip101

#endif // GAMMA_CORRECTION_HPP