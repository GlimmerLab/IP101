#ifndef COLOR_OPERATIONS_HPP
#define COLOR_OPERATIONS_HPP

#include <opencv2/opencv.hpp>
#include <immintrin.h>  // 用于SIMD指令(AVX2)
#include <omp.h>        // 用于OpenMP并行计算

namespace ip101 {

/**
 * @brief 通道替换
 * @param src 输入图像
 * @param dst 输出图像
 * @param r_idx R通道映射索引(0-2)
 * @param g_idx G通道映射索引(0-2)
 * @param b_idx B通道映射索引(0-2)
 */
void channel_swap(const cv::Mat& src, cv::Mat& dst,
                 int r_idx = 0, int g_idx = 1, int b_idx = 2);

/**
 * @brief 灰度化
 * @param src 输入图像
 * @param dst 输出图像
 * @param method 灰度化方法("average", "weighted", "max", "min")
 */
void to_gray(const cv::Mat& src, cv::Mat& dst,
            const std::string& method = "weighted");

/**
 * @brief 二值化
 * @param src 输入图像
 * @param dst 输出图像
 * @param threshold 阈值
 * @param max_value 最大值
 * @param method 二值化方法("binary", "binary_inv", "trunc", "tozero", "tozero_inv")
 */
void threshold_image(const cv::Mat& src, cv::Mat& dst,
                    double threshold, double max_value = 255,
                    const std::string& method = "binary");

/**
 * @brief 大津算法自动阈值
 * @param src 输入图像
 * @param dst 输出图像
 * @param max_value 最大值
 * @return 计算得到的最优阈值
 */
double otsu_threshold(const cv::Mat& src, cv::Mat& dst,
                     double max_value = 255);

/**
 * @brief HSV颜色空间转换
 * @param src 输入图像(BGR)
 * @param dst 输出图像(HSV)
 */
void bgr_to_hsv(const cv::Mat& src, cv::Mat& dst);

/**
 * @brief HSV颜色空间转回BGR
 * @param src 输入图像(HSV)
 * @param dst 输出图像(BGR)
 */
void hsv_to_bgr(const cv::Mat& src, cv::Mat& dst);

/**
 * @brief HSV颜色空间调整
 * @param src 输入图像(HSV)
 * @param dst 输出图像(HSV)
 * @param h_offset 色相偏移量(-180到+180)
 * @param s_scale 饱和度缩放因子(0到无穷大)
 * @param v_scale 明度缩放因子(0到无穷大)
 */
void adjust_hsv(const cv::Mat& src, cv::Mat& dst,
               float h_offset = 0, float s_scale = 1, float v_scale = 1);

} // namespace ip101

#endif // COLOR_OPERATIONS_HPP