#ifndef RETINEX_MSRCR_HPP
#define RETINEX_MSRCR_HPP

#include <opencv2/opencv.hpp>

namespace ip101 {
namespace advanced {

/**
 * @brief Retinex MSRCR(多尺度Retinex带颜色恢复)算法实现
 * @param src 输入图像
 * @param dst 输出图像
 * @param sigma1 第一个高斯核的标准差
 * @param sigma2 第二个高斯核的标准差
 * @param sigma3 第三个高斯核的标准差
 * @param alpha 颜色恢复参数
 * @param beta 颜色恢复参数
 * @param gain 增益参数
 */
void retinex_msrcr(const cv::Mat& src, cv::Mat& dst,
                  double sigma1 = 15.0, double sigma2 = 80.0, double sigma3 = 250.0,
                  double alpha = 125.0, double beta = 46.0, double gain = 192.0);

} // namespace advanced
} // namespace ip101

#endif // RETINEX_MSRCR_HPP