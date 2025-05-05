#ifndef FILTERING_HPP
#define FILTERING_HPP

#include <opencv2/opencv.hpp>
#include <immintrin.h>  // 用于SIMD指令(AVX2)
#include <omp.h>        // 用于OpenMP并行计算

namespace ip101 {

/**
 * @brief 均值滤波
 * @param src 输入图像
 * @param dst 输出图像
 * @param ksize 滤波器大小
 * @param border_type 边界处理方式
 */
void mean_filter(const cv::Mat& src, cv::Mat& dst,
                int ksize = 3,
                int border_type = cv::BORDER_DEFAULT);

/**
 * @brief 中值滤波
 * @param src 输入图像
 * @param dst 输出图像
 * @param ksize 滤波器大小
 * @param border_type 边界处理方式
 */
void median_filter(const cv::Mat& src, cv::Mat& dst,
                  int ksize = 3,
                  int border_type = cv::BORDER_DEFAULT);

/**
 * @brief 高斯滤波
 * @param src 输入图像
 * @param dst 输出图像
 * @param ksize 滤波器大小
 * @param sigma_x x方向标准差
 * @param sigma_y y方向标准差
 * @param border_type 边界处理方式
 */
void gaussian_filter(const cv::Mat& src, cv::Mat& dst,
                    int ksize = 3,
                    double sigma_x = 1.0,
                    double sigma_y = 1.0,
                    int border_type = cv::BORDER_DEFAULT);

/**
 * @brief 均值池化
 * @param src 输入图像
 * @param dst 输出图像
 * @param ksize 池化窗口大小
 * @param stride 步长
 */
void mean_pooling(const cv::Mat& src, cv::Mat& dst,
                 int ksize = 2,
                 int stride = 2);

/**
 * @brief 最大池化
 * @param src 输入图像
 * @param dst 输出图像
 * @param ksize 池化窗口大小
 * @param stride 步长
 */
void max_pooling(const cv::Mat& src, cv::Mat& dst,
                int ksize = 2,
                int stride = 2);

/**
 * @brief 创建高斯核
 * @param ksize 核大小
 * @param sigma_x x方向标准差
 * @param sigma_y y方向标准差
 * @return 高斯核矩阵
 */
cv::Mat create_gaussian_kernel(int ksize,
                             double sigma_x,
                             double sigma_y);

} // namespace ip101

#endif // FILTERING_HPP