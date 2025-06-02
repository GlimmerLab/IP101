#ifndef HOMOMORPHIC_FILTER_HPP
#define HOMOMORPHIC_FILTER_HPP

#include <opencv2/opencv.hpp>

namespace ip101 {
namespace advanced {

/**
 * @brief 同态滤波算法
 * @param src 输入图像
 * @param dst 输出图像
 * @param gamma_low 低频增益(控制阴影区域)
 * @param gamma_high 高频增益(控制高光区域)
 * @param cutoff 截止频率
 * @param c 锐化因子
 */
void homomorphic_filter(const cv::Mat& src, cv::Mat& dst,
                       double gamma_low = 0.1, double gamma_high = 1.8,
                       double cutoff = 32.0, double c = 1.0);

/**
 * @brief 创建同态滤波的高斯高通滤波器
 * @param size 滤波器大小
 * @param gamma_low 低频增益
 * @param gamma_high 高频增益
 * @param cutoff 截止频率
 * @param c 锐化因子
 * @return 频域滤波器
 */
cv::Mat create_homomorphic_filter(const cv::Size& size,
                                 double gamma_low, double gamma_high,
                                 double cutoff, double c);

/**
 * @brief 傅里叶频谱可视化
 * @param complex_img 复数图像
 * @param dst 可视化结果
 */
void visualize_spectrum(const cv::Mat& complex_img, cv::Mat& dst);

/**
 * @brief 图像的DFT变换及滤波
 * @param src 输入图像
 * @param dst 输出图像
 * @param filter 频域滤波器
 */
void dft_filter(const cv::Mat& src, cv::Mat& dst, const cv::Mat& filter);

/**
 * @brief 增强型同态滤波算法
 * @param src 输入图像
 * @param dst 输出图像
 * @param gamma_low 低频增益(控制阴影区域)
 * @param gamma_high 高频增益(控制高光区域)
 * @param cutoff 截止频率
 * @param c 锐化因子
 * @param alpha 额外增强因子
 */
void enhanced_homomorphic_filter(const cv::Mat& src, cv::Mat& dst,
                               double gamma_low = 0.1, double gamma_high = 1.8,
                               double cutoff = 32.0, double c = 1.0,
                               double alpha = 0.5);

} // namespace advanced
} // namespace ip101

#endif // HOMOMORPHIC_FILTER_HPP