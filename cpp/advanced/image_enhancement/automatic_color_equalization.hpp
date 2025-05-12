#ifndef AUTOMATIC_COLOR_EQUALIZATION_HPP
#define AUTOMATIC_COLOR_EQUALIZATION_HPP

#include <opencv2/opencv.hpp>

namespace ip101 {
namespace advanced {

/**
 * @brief 自动色彩均衡(ACE)算法实现
 * @param src 输入图像
 * @param dst 输出图像
 * @param alpha 对比度增强参数
 * @param beta 平滑参数
 */
void automatic_color_equalization(const cv::Mat& src, cv::Mat& dst,
                                double alpha = 1.0, double beta = 1.0);

} // namespace advanced
} // namespace ip101

#endif // AUTOMATIC_COLOR_EQUALIZATION_HPP