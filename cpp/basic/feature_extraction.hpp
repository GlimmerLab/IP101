#ifndef FEATURE_EXTRACTION_HPP
#define FEATURE_EXTRACTION_HPP

#include <opencv2/opencv.hpp>
#include <immintrin.h>  // 用于SIMD指令(AVX2)
#include <omp.h>        // 用于OpenMP并行计算

namespace ip101 {

/**
 * @brief Harris角点检测
 * @param src 输入图像
 * @param dst 输出图像
 * @param block_size 邻域大小
 * @param ksize Sobel算子的孔径参数
 * @param k Harris检测参数
 * @param threshold 角点检测阈值
 */
void harris_corner_detection(const cv::Mat& src, cv::Mat& dst,
                           int block_size = 2, int ksize = 3,
                           double k = 0.04, double threshold = 0.01);

/**
 * @brief SIFT特征提取
 * @param src 输入图像
 * @param dst 输出图像
 * @param nfeatures 提取的特征点数量(0表示提取所有)
 */
void sift_features(const cv::Mat& src, cv::Mat& dst,
                  int nfeatures = 0);

/**
 * @brief SURF特征提取
 * @param src 输入图像
 * @param dst 输出图像
 * @param hessian_threshold Hessian矩阵阈值
 */
void surf_features(const cv::Mat& src, cv::Mat& dst,
                  double hessian_threshold = 100);

/**
 * @brief ORB特征提取
 * @param src 输入图像
 * @param dst 输出图像
 * @param nfeatures 提取的特征点数量
 */
void orb_features(const cv::Mat& src, cv::Mat& dst,
                 int nfeatures = 500);

/**
 * @brief 特征匹配
 * @param src1 第一张图像
 * @param src2 第二张图像
 * @param dst 输出图像
 * @param method 特征提取方法("sift", "surf", "orb")
 */
void feature_matching(const cv::Mat& src1, const cv::Mat& src2,
                     cv::Mat& dst, const std::string& method = "sift");

/**
 * @brief 手动实现的Harris角点检测
 * @param src 输入图像
 * @param dst 输出图像
 * @param k Harris检测参数
 * @param window_size 窗口大小
 * @param threshold 角点检测阈值
 */
void compute_harris_manual(const cv::Mat& src, cv::Mat& dst,
                          double k = 0.04, int window_size = 3,
                          double threshold = 0.01);

} // namespace ip101

#endif // FEATURE_EXTRACTION_HPP