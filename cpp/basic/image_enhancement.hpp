#ifndef IMAGE_ENHANCEMENT_HPP
#define IMAGE_ENHANCEMENT_HPP

#include <opencv2/opencv.hpp>
#include <immintrin.h>  // 用于SIMD指令(AVX2)
#include <omp.h>        // 用于OpenMP并行计算

namespace ip101 {

/**
 * @brief 直方图均衡化
 * @param src 输入图像
 * @param dst 输出图像
 * @param method 均衡化方法("global", "adaptive", "clahe")
 * @param clip_limit CLAHE的对比度限制阈值
 * @param grid_size CLAHE的网格大小
 */
void histogram_equalization(const cv::Mat& src, cv::Mat& dst,
                          const std::string& method = "global",
                          double clip_limit = 40.0,
                          cv::Size grid_size = cv::Size(8, 8));

/**
 * @brief 伽马变换
 * @param src 输入图像
 * @param dst 输出图像
 * @param gamma 伽马值
 */
void gamma_correction(const cv::Mat& src, cv::Mat& dst, double gamma);

/**
 * @brief 对比度拉伸
 * @param src 输入图像
 * @param dst 输出图像
 * @param min_val 最小输出值
 * @param max_val 最大输出值
 */
void contrast_stretching(const cv::Mat& src, cv::Mat& dst,
                        double min_val = 0, double max_val = 255);

/**
 * @brief 亮度调整
 * @param src 输入图像
 * @param dst 输出图像
 * @param beta 亮度调整值(-255到255)
 */
void brightness_adjustment(const cv::Mat& src, cv::Mat& dst, double beta);

/**
 * @brief 饱和度调整
 * @param src 输入图像
 * @param dst 输出图像
 * @param saturation 饱和度调整值(0到2)
 */
void saturation_adjustment(const cv::Mat& src, cv::Mat& dst, double saturation);

/**
 * @brief 计算图像直方图
 * @param src 输入图像
 * @param hist 输出直方图
 * @param channel 通道索引(对于彩色图像)
 */
void calculate_histogram(const cv::Mat& src, cv::Mat& hist, int channel = 0);

/**
 * @brief 计算累积分布函数
 * @param histogram 直方图
 * @param cdf 输出累积分布函数
 */
void calculate_cdf(const cv::Mat& hist, cv::Mat& cdf);

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