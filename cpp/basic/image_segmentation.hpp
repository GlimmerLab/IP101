#ifndef IMAGE_SEGMENTATION_HPP
#define IMAGE_SEGMENTATION_HPP

#include <opencv2/opencv.hpp>
#include <immintrin.h>  // 用于SIMD指令(AVX2)
#include <omp.h>        // 用于OpenMP并行计算

namespace ip101 {

/**
 * @brief 阈值分割
 * @param src 输入图像
 * @param dst 输出图像
 * @param threshold 阈值
 * @param max_val 最大值
 * @param type 阈值类型
 */
void threshold_segmentation(const cv::Mat& src, cv::Mat& dst,
                          double threshold, double max_val = 255,
                          int type = cv::THRESH_BINARY);

/**
 * @brief K均值分割
 * @param src 输入图像
 * @param dst 输出图像
 * @param k 聚类数
 * @param max_iter 最大迭代次数
 */
void kmeans_segmentation(const cv::Mat& src, cv::Mat& dst,
                        int k = 3, int max_iter = 100);

/**
 * @brief 区域生长分割
 * @param src 输入图像
 * @param dst 输出图像
 * @param seed_points 种子点
 * @param threshold 生长阈值
 */
void region_growing(const cv::Mat& src, cv::Mat& dst,
                   const std::vector<cv::Point>& seed_points,
                   double threshold = 10.0);

/**
 * @brief 分水岭分割
 * @param src 输入图像
 * @param markers 标记图像
 * @param dst 输出图像
 */
void watershed_segmentation(const cv::Mat& src,
                          cv::Mat& markers,
                          cv::Mat& dst);

/**
 * @brief 图割分割
 * @param src 输入图像
 * @param dst 输出图像
 * @param rect 前景矩形框
 */
void graph_cut_segmentation(const cv::Mat& src, cv::Mat& dst,
                          const cv::Rect& rect);

} // namespace ip101

#endif // IMAGE_SEGMENTATION_HPP