#ifndef MEDIAN_FILTER_DEFOGGING_HPP
#define MEDIAN_FILTER_DEFOGGING_HPP

#include <opencv2/opencv.hpp>

namespace ip101 {
namespace advanced {

/**
 * @brief 中值滤波去雾算法
 * @param src 输入图像
 * @param dst 输出图像
 * @param kernel_size 中值滤波核大小
 * @param omega 去雾强度参数 (0-1)
 * @param t0 透射率下限
 */
void median_filter_defogging(const cv::Mat& src, cv::Mat& dst,
                            int kernel_size = 31,
                            double omega = 0.95,
                            double t0 = 0.1);

/**
 * @brief 基于Meng等人改进的中值滤波去雾算法
 * @param src 输入图像
 * @param dst 输出图像
 * @param kernel_size 中值滤波核大小
 * @param omega 去雾强度参数 (0-1)
 * @param t0 透射率下限
 * @param lambda 正则化参数
 */
void improved_median_filter_defogging(const cv::Mat& src, cv::Mat& dst,
                                     int kernel_size = 31,
                                     double omega = 0.95,
                                     double t0 = 0.1,
                                     double lambda = 0.001);

/**
 * @brief 自适应中值滤波去雾算法
 * @param src 输入图像
 * @param dst 输出图像
 * @param init_size 初始滤波核大小
 * @param max_size 最大滤波核大小
 * @param omega 去雾强度参数 (0-1)
 * @param t0 透射率下限
 */
void adaptive_median_filter_defogging(const cv::Mat& src, cv::Mat& dst,
                                     int init_size = 3,
                                     int max_size = 21,
                                     double omega = 0.95,
                                     double t0 = 0.1);

} // namespace advanced
} // namespace ip101

#endif // MEDIAN_FILTER_DEFOGGING_HPP