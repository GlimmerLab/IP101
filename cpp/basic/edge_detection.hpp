#ifndef EDGE_DETECTION_HPP
#define EDGE_DETECTION_HPP

#include <opencv2/opencv.hpp>
#include <immintrin.h> // 用于AVX2指令集
#include <omp.h>       // 用于OpenMP多线程

namespace ip101 {

/**
 * @brief 微分滤波器接口
 * @param src 输入图像
 * @param dst 输出图像
 * @param dx x方向的微分阶数
 * @param dy y方向的微分阶数
 * @param ksize 滤波器核大小
 */
void differential_filter(const cv::Mat& src, cv::Mat& dst,
                       int dx, int dy, int ksize = 3);

/**
 * @brief Sobel算子边缘检测
 * @param src 输入图像
 * @param dst 输出图像
 * @param dx x方向的微分阶数
 * @param dy y方向的微分阶数
 * @param ksize 滤波器核大小
 * @param scale 缩放因子
 */
void sobel_filter(const cv::Mat& src, cv::Mat& dst,
                 int dx, int dy, int ksize = 3, double scale = 1.0);

/**
 * @brief Prewitt算子边缘检测
 * @param src 输入图像
 * @param dst 输出图像
 * @param dx x方向的微分阶数
 * @param dy y方向的微分阶数
 */
void prewitt_filter(const cv::Mat& src, cv::Mat& dst,
                   int dx, int dy);

/**
 * @brief Laplacian算子边缘检测
 * @param src 输入图像
 * @param dst 输出图像
 * @param ksize 滤波器核大小
 * @param scale 缩放因子
 */
void laplacian_filter(const cv::Mat& src, cv::Mat& dst,
                     int ksize = 3, double scale = 1.0);

/**
 * @brief 浮雕效果
 * @param src 输入图像
 * @param dst 输出图像
 * @param direction 浮雕方向（0-7，对应8个方向）
 */
void emboss_effect(const cv::Mat& src, cv::Mat& dst,
                  int direction = 0);

/**
 * @brief 综合边缘检测
 * @param src 输入图像
 * @param dst 输出图像
 * @param method 检测方法（"sobel", "prewitt", "laplacian"）
 * @param threshold 边缘阈值
 */
void edge_detection(const cv::Mat& src, cv::Mat& dst,
                   const std::string& method = "sobel",
                   double threshold = 128.0);

} // namespace ip101

#endif // EDGE_DETECTION_HPP