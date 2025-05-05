#ifndef IMAGE_ENHANCEMENT_HPP
#define IMAGE_ENHANCEMENT_HPP

#include <opencv2/opencv.hpp>
#include <immintrin.h>  // 用于SIMD指令(AVX2)
#include <omp.h>        // 用于OpenMP并行计算

namespace ip101 {

/**
 * @brief 直方图均衡化
 * @param src 输入图像
 * @return 均衡化后的图像
 */
cv::Mat histogram_equalization(const cv::Mat& src);

/**
 * @brief 伽马变换
 * @param src 输入图像
 * @param gamma 伽马值
 * @return 变换后的图像
 */
cv::Mat gamma_transform(const cv::Mat& src, double gamma);

/**
 * @brief 对比度拉伸
 * @param src 输入图像
 * @param min_val 最小输出值
 * @param max_val 最大输出值
 * @return 拉伸后的图像
 */
cv::Mat contrast_stretching(const cv::Mat& src, double min_val = 0, double max_val = 255);

/**
 * @brief 亮度调整
 * @param src 输入图像
 * @param brightness 亮度调整值(-255到255)
 * @return 调整后的图像
 */
cv::Mat adjust_brightness(const cv::Mat& src, int brightness);

/**
 * @brief 饱和度调整
 * @param src 输入图像
 * @param saturation 饱和度调整值(-100到100)
 * @return 调整后的图像
 */
cv::Mat adjust_saturation(const cv::Mat& src, int saturation);

/**
 * @brief 计算图像直方图
 * @param src 输入图像
 * @return 直方图(256维向量)
 */
std::vector<int> calculate_histogram(const cv::Mat& src);

/**
 * @brief 计算累积分布函数
 * @param histogram 直方图
 * @return 累积分布函数
 */
std::vector<float> calculate_cdf(const std::vector<int>& histogram);

/**
 * @brief 局部直方图均衡化
 * @param src 输入图像
 * @param dst 输出图像
 * @param window_size 局部窗口大小
 */
void local_histogram_equalization(const cv::Mat& src, cv::Mat& dst,
                                int window_size = 3);

/**
 * @brief CLAHE(限制对比度自适应直方图均衡化)
 * @param src 输入图像
 * @param dst 输出图像
 * @param clip_limit 对比度限制阈值
 * @param grid_size 网格大小
 */
void clahe(const cv::Mat& src, cv::Mat& dst,
          double clip_limit = 40.0,
          cv::Size grid_size = cv::Size(8, 8));

} // namespace ip101

#endif // IMAGE_ENHANCEMENT_HPP