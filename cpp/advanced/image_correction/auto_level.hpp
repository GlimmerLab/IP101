#ifndef AUTO_LEVEL_HPP
#define AUTO_LEVEL_HPP

#include <opencv2/opencv.hpp>

namespace ip101 {
namespace advanced {

/**
 * @brief 自动色阶调整算法实现
 * @param src 输入图像
 * @param dst 输出图像
 * @param clip_percent 裁剪百分比，去除最暗和最亮的像素 (0.0-5.0)
 * @param separate_channels 是否单独处理每个通道
 */
void auto_level(const cv::Mat& src, cv::Mat& dst,
               float clip_percent = 0.5f,
               bool separate_channels = true);

/**
 * @brief 自动对比度调整算法实现
 * @param src 输入图像
 * @param dst 输出图像
 * @param clip_percent 裁剪百分比，去除最暗和最亮的像素 (0.0-5.0)
 * @param separate_channels 是否单独处理每个通道
 */
void auto_contrast(const cv::Mat& src, cv::Mat& dst,
                  float clip_percent = 0.5f,
                  bool separate_channels = false);

} // namespace advanced
} // namespace ip101

#endif // AUTO_LEVEL_HPP