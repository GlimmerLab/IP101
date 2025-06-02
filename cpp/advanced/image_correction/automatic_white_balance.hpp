#ifndef AUTOMATIC_WHITE_BALANCE_HPP
#define AUTOMATIC_WHITE_BALANCE_HPP

#include <opencv2/opencv.hpp>

namespace ip101 {
namespace advanced {

/**
 * @brief 自动白平衡算法实现
 * @param src 输入图像
 * @param dst 输出图像
 * @param method 白平衡方法 ("gray_world", "perfect_reflector", "retinex")
 */
void automatic_white_balance(const cv::Mat& src, cv::Mat& dst, const std::string& method = "gray_world");

/**
 * @brief 基于灰度世界假设的白平衡
 * @param src 输入图像
 * @param dst 输出图像
 */
void gray_world_white_balance(const cv::Mat& src, cv::Mat& dst);

/**
 * @brief 基于完美反射假设的白平衡
 * @param src 输入图像
 * @param dst 输出图像
 * @param ratio 考虑作为参考白色的最高像素值比例（默认为前5%的高亮区域）
 */
void perfect_reflector_white_balance(const cv::Mat& src, cv::Mat& dst, float ratio = 0.05f);

} // namespace advanced
} // namespace ip101

#endif // AUTOMATIC_WHITE_BALANCE_HPP