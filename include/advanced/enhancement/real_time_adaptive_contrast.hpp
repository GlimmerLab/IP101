#ifndef REAL_TIME_ADAPTIVE_CONTRAST_HPP
#define REAL_TIME_ADAPTIVE_CONTRAST_HPP

#include <opencv2/opencv.hpp>

namespace ip101 {
namespace advanced {

/**
 * @brief 实时自适应对比度增强算法实现
 * @param src 输入图像
 * @param dst 输出图像
 * @param window_size 局部窗口大小
 * @param clip_limit 对比度限制阈值
 */
void real_time_adaptive_contrast(const cv::Mat& src, cv::Mat& dst,
                               int window_size = 7, double clip_limit = 2.0);

} // namespace advanced
} // namespace ip101

#endif // REAL_TIME_ADAPTIVE_CONTRAST_HPP