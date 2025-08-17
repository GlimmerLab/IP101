#ifndef REALTIME_DEHAZING_HPP
#define REALTIME_DEHAZING_HPP

#include <opencv2/opencv.hpp>

namespace ip101 {
namespace advanced {

/**
 * @brief 实时视频去雾算法
 * @param src 输入图像
 * @param dst 输出图像
 * @param downsample_factor 下采样因子，用于加速计算
 * @param omega 去雾强度参数 (0-1)
 * @param t0 透射率下限
 */
void realtime_dehazing(const cv::Mat& src, cv::Mat& dst,
                      double downsample_factor = 0.25,
                      double omega = 0.95,
                      double t0 = 0.1);

/**
 * @brief 快速大气散射模型
 * @param src 输入图像
 * @param dst 输出图像
 * @param A 大气光照值
 * @param t 透射率图像
 */
void fast_dehazing_model(const cv::Mat& src, cv::Mat& dst,
                        const cv::Vec3d& A,
                        const cv::Mat& t);

/**
 * @brief 实时改进的暗通道先验去雾
 * @param src 输入图像
 * @param dst 输出图像
 * @param radius 局部区域半径
 * @param min_filter_radius 最小值滤波半径
 * @param omega 去雾强度参数 (0-1)
 * @param t0 透射率下限
 */
void realtime_dark_channel_dehazing(const cv::Mat& src, cv::Mat& dst,
                                   int radius = 15,
                                   int min_filter_radius = 7,
                                   double omega = 0.95,
                                   double t0 = 0.1);

/**
 * @brief 快速最小值滤波
 * @param src 输入图像
 * @param dst 输出图像
 * @param radius 滤波半径
 */
void fast_min_filter(const cv::Mat& src, cv::Mat& dst, int radius);

} // namespace advanced
} // namespace ip101

#endif // REALTIME_DEHAZING_HPP