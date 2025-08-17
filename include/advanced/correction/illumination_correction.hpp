#ifndef ILLUMINATION_CORRECTION_HPP
#define ILLUMINATION_CORRECTION_HPP

#include <opencv2/opencv.hpp>

namespace ip101 {
namespace advanced {

/**
 * @brief 光照不均匀校正算法实现
 * @param src 输入图像
 * @param dst 输出图像
 * @param method 校正方法 ("homomorphic", "background_subtraction", "multi_scale")
 */
void illumination_correction(const cv::Mat& src, cv::Mat& dst,
                            const std::string& method = "homomorphic");

/**
 * @brief 同态滤波光照校正
 * @param src 输入图像
 * @param dst 输出图像
 * @param gamma_low 低频系数
 * @param gamma_high 高频系数
 * @param cutoff 截止频率
 */
void homomorphic_illumination_correction(const cv::Mat& src, cv::Mat& dst,
                                        double gamma_low = 0.3,
                                        double gamma_high = 1.5,
                                        double cutoff = 30.0);

/**
 * @brief 背景减除法光照校正
 * @param src 输入图像
 * @param dst 输出图像
 * @param kernel_size 滤波核大小
 * @param resize_factor 缩放因子，用于加速处理
 */
void background_subtraction_correction(const cv::Mat& src, cv::Mat& dst,
                                      int kernel_size = 51,
                                      double resize_factor = 0.5);

/**
 * @brief 多尺度Retinex光照校正
 * @param src 输入图像
 * @param dst 输出图像
 * @param sigma_list 高斯模糊的sigma值列表
 */
void multi_scale_illumination_correction(const cv::Mat& src, cv::Mat& dst,
                                        const std::vector<double>& sigma_list = {15, 80, 250});

} // namespace advanced
} // namespace ip101

#endif // ILLUMINATION_CORRECTION_HPP