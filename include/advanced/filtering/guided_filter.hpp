#ifndef GUIDED_FILTER_HPP
#define GUIDED_FILTER_HPP

#include <opencv2/opencv.hpp>

namespace ip101 {
namespace advanced {

/**
 * @brief 导向滤波实现
 * @param p 输入图像（需要滤波的图像）
 * @param I 引导图像
 * @param q 输出图像
 * @param radius 滤波半径
 * @param eps 正则化参数
 */
void guided_filter(const cv::Mat& p, const cv::Mat& I, cv::Mat& q,
                  int radius = 10, double eps = 1e-6);

/**
 * @brief 快速导向滤波实现
 * @param p 输入图像（需要滤波的图像）
 * @param I 引导图像
 * @param q 输出图像
 * @param radius 滤波半径
 * @param eps 正则化参数
 * @param s 下采样因子
 */
void fast_guided_filter(const cv::Mat& p, const cv::Mat& I, cv::Mat& q,
                       int radius = 10, double eps = 1e-6, int s = 4);

/**
 * @brief 边缘感知导向滤波
 * @param p 输入图像（需要滤波的图像）
 * @param I 引导图像
 * @param q 输出图像
 * @param radius 滤波半径
 * @param eps 正则化参数
 * @param edge_aware_factor 边缘感知因子
 */
void edge_aware_guided_filter(const cv::Mat& p, const cv::Mat& I, cv::Mat& q,
                             int radius = 10, double eps = 1e-6, double edge_aware_factor = 10.0);

/**
 * @brief 联合双边滤波 - 使用导向滤波实现
 * @param p 输入图像（需要滤波的图像）
 * @param I 引导图像
 * @param q 输出图像
 * @param radius 滤波半径
 * @param sigma_space 空间域标准差
 * @param sigma_range 值域标准差
 */
void joint_bilateral_filter(const cv::Mat& p, const cv::Mat& I, cv::Mat& q,
                           int radius = 10, double sigma_space = 50.0, double sigma_range = 50.0);

} // namespace advanced
} // namespace ip101

#endif // GUIDED_FILTER_HPP