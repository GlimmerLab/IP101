#ifndef GUIDED_FILTER_DEFOGGING_HPP
#define GUIDED_FILTER_DEFOGGING_HPP

#include <opencv2/opencv.hpp>

namespace ip101 {
namespace advanced {

/**
 * @brief 导向滤波去雾算法
 * @param src 输入图像
 * @param dst 输出图像
 * @param radius 导向滤波的半径
 * @param eps 导向滤波的正则化参数
 * @param omega 去雾强度参数 (0-1)
 * @param t0 透射率下限
 */
void guided_filter_defogging(const cv::Mat& src, cv::Mat& dst,
                            int radius = 60,
                            double eps = 0.0001,
                            double omega = 0.95,
                            double t0 = 0.1);

/**
 * @brief 导向滤波实现
 * @param p 输入图像
 * @param I 引导图像
 * @param q 输出图像
 * @param radius 滤波半径
 * @param eps 正则化参数
 */
void guided_filter(const cv::Mat& p, const cv::Mat& I, cv::Mat& q,
                  int radius, double eps);

/**
 * @brief 基于Kaiming He的导向滤波去雾改进算法
 * @param src 输入图像
 * @param dst 输出图像
 * @param radius 导向滤波的半径
 * @param eps 导向滤波的正则化参数
 * @param omega 去雾强度参数 (0-1)
 * @param t0 透射率下限
 */
void kaiming_he_guided_defogging(const cv::Mat& src, cv::Mat& dst,
                               int radius = 60,
                               double eps = 0.0001,
                               double omega = 0.95,
                               double t0 = 0.1);

} // namespace advanced
} // namespace ip101

#endif // GUIDED_FILTER_DEFOGGING_HPP