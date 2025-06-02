#ifndef DARK_CHANNEL_HPP
#define DARK_CHANNEL_HPP

#include <opencv2/opencv.hpp>

namespace ip101 {
namespace advanced {

/**
 * @brief 暗通道去雾算法 (Dark Channel Prior)
 * @param src 输入图像
 * @param dst 输出图像
 * @param patch_size 局部区域的大小
 * @param omega 去雾强度参数 (0-1)
 * @param t0 透射率下限
 */
void dark_channel_defogging(const cv::Mat& src, cv::Mat& dst,
                           int patch_size = 15,
                           double omega = 0.95,
                           double t0 = 0.1);

/**
 * @brief 计算暗通道图像
 * @param src 输入彩色图像
 * @param dark 输出暗通道图像
 * @param patch_size 局部区域的大小
 */
void compute_dark_channel(const cv::Mat& src, cv::Mat& dark, int patch_size = 15);

/**
 * @brief 估计大气光照值
 * @param src 输入彩色图像
 * @param dark 暗通道图像
 * @param percent 用于估计的暗通道最亮像素的百分比
 * @return 估计的大气光照值(3通道)
 */
cv::Vec3d estimate_atmospheric_light(const cv::Mat& src, const cv::Mat& dark, double percent = 0.001);

/**
 * @brief 估计透射率图像
 * @param src 输入彩色图像
 * @param dark 暗通道图像
 * @param A 大气光照值
 * @param omega 去雾强度
 * @return 估计的透射率图像
 */
cv::Mat estimate_transmission(const cv::Mat& src, const cv::Mat& dark,
                             const cv::Vec3d& A, double omega);

/**
 * @brief 双边滤波改进的暗通道去雾算法
 * @param src 输入图像
 * @param dst 输出图像
 * @param patch_size 局部区域的大小
 * @param omega 去雾强度参数 (0-1)
 * @param t0 透射率下限
 */
void bilateral_dark_channel_defogging(const cv::Mat& src, cv::Mat& dst,
                                     int patch_size = 15,
                                     double omega = 0.95,
                                     double t0 = 0.1);

} // namespace advanced
} // namespace ip101

#endif // DARK_CHANNEL_HPP