#ifndef MULTI_SCALE_DETAIL_ENHANCEMENT_HPP
#define MULTI_SCALE_DETAIL_ENHANCEMENT_HPP

#include <opencv2/opencv.hpp>

namespace ip101 {
namespace advanced {

/**
 * @brief 多尺度细节增强算法实现
 * @param src 输入图像
 * @param dst 输出图像
 * @param sigma1 高斯金字塔构建的标准差
 * @param sigma2 细节提取的标准差
 * @param alpha 细节增强强度
 * @param beta 对比度增强因子
 */
void multi_scale_detail_enhancement(const cv::Mat& src, cv::Mat& dst,
                                  double sigma1 = 1.0, double sigma2 = 2.0,
                                  double alpha = 1.5, double beta = 1.2);

} // namespace advanced
} // namespace ip101

#endif // MULTI_SCALE_DETAIL_ENHANCEMENT_HPP