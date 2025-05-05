#ifndef MORPHOLOGY_HPP
#define MORPHOLOGY_HPP

#include <opencv2/opencv.hpp>
#include <immintrin.h>  // 用于SIMD指令(AVX2)
#include <omp.h>        // 用于OpenMP并行计算

namespace ip101 {

/**
 * @brief 膨胀操作
 * @param src 输入图像
 * @param dst 输出图像
 * @param kernel 结构元素
 * @param iterations 迭代次数
 */
void dilate_manual(const cv::Mat& src, cv::Mat& dst,
                  const cv::Mat& kernel = cv::Mat(),
                  int iterations = 1);

/**
 * @brief 腐蚀操作
 * @param src 输入图像
 * @param dst 输出图像
 * @param kernel 结构元素
 * @param iterations 迭代次数
 */
void erode_manual(const cv::Mat& src, cv::Mat& dst,
                 const cv::Mat& kernel = cv::Mat(),
                 int iterations = 1);

/**
 * @brief 开运算
 * @param src 输入图像
 * @param dst 输出图像
 * @param kernel 结构元素
 * @param iterations 迭代次数
 */
void opening_manual(const cv::Mat& src, cv::Mat& dst,
                   const cv::Mat& kernel = cv::Mat(),
                   int iterations = 1);

/**
 * @brief 闭运算
 * @param src 输入图像
 * @param dst 输出图像
 * @param kernel 结构元素
 * @param iterations 迭代次数
 */
void closing_manual(const cv::Mat& src, cv::Mat& dst,
                   const cv::Mat& kernel = cv::Mat(),
                   int iterations = 1);

/**
 * @brief 形态学梯度
 * @param src 输入图像
 * @param dst 输出图像
 * @param kernel 结构元素
 */
void morphological_gradient_manual(const cv::Mat& src, cv::Mat& dst,
                                 const cv::Mat& kernel = cv::Mat());

/**
 * @brief 创建结构元素
 * @param shape 形状类型(MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE)
 * @param ksize 结构元素大小
 * @return 结构元素矩阵
 */
cv::Mat create_kernel(int shape, cv::Size ksize);

} // namespace ip101

#endif // MORPHOLOGY_HPP