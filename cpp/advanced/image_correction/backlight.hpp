#ifndef BACKLIGHT_HPP
#define BACKLIGHT_HPP

#include <opencv2/opencv.hpp>

namespace ip101 {
namespace advanced {

/**
 * @brief 逆光图像恢复 (Inverse Intensity-Range Based Backlight)
 * @param src 输入图像
 * @param dst 输出图像
 * @param gamma 伽马校正参数
 * @param lambda 增强系数
 */
void inrbl_backlight_correction(const cv::Mat& src, cv::Mat& dst,
                               double gamma = 0.6, double lambda = 0.8);

/**
 * @brief 自适应局部对比度增强逆光校正
 * @param src 输入图像
 * @param dst 输出图像
 * @param clip_limit 对比度限制
 * @param grid_size 网格大小
 */
void adaptive_backlight_correction(const cv::Mat& src, cv::Mat& dst,
                                  double clip_limit = 3.0,
                                  const cv::Size& grid_size = cv::Size(8, 8));

/**
 * @brief 基于曝光融合的逆光恢复
 * @param src 输入图像
 * @param dst 输出图像
 * @param num_exposures 合成的曝光数量
 */
void exposure_fusion_backlight_correction(const cv::Mat& src, cv::Mat& dst,
                                         int num_exposures = 3);

} // namespace advanced
} // namespace ip101

#endif // BACKLIGHT_HPP