#ifndef IMAGE_MATCHING_HPP
#define IMAGE_MATCHING_HPP

#include <opencv2/opencv.hpp>
#include <immintrin.h>  // 用于SIMD指令(AVX2)
#include <omp.h>        // 用于OpenMP并行计算
#include <vector>

namespace ip101 {

/**
 * @brief SSD模板匹配
 * @param src 输入图像
 * @param templ 模板图像
 * @param result 匹配结果
 * @param method 匹配方法(默认TM_SQDIFF)
 */
void ssd_matching(const cv::Mat& src,
                 const cv::Mat& templ,
                 cv::Mat& result,
                 int method = cv::TM_SQDIFF);

/**
 * @brief SAD模板匹配
 * @param src 输入图像
 * @param templ 模板图像
 * @param result 匹配结果
 */
void sad_matching(const cv::Mat& src,
                 const cv::Mat& templ,
                 cv::Mat& result);

/**
 * @brief NCC模板匹配
 * @param src 输入图像
 * @param templ 模板图像
 * @param result 匹配结果
 */
void ncc_matching(const cv::Mat& src,
                 const cv::Mat& templ,
                 cv::Mat& result);

/**
 * @brief ZNCC模板匹配
 * @param src 输入图像
 * @param templ 模板图像
 * @param result 匹配结果
 */
void zncc_matching(const cv::Mat& src,
                  const cv::Mat& templ,
                  cv::Mat& result);

/**
 * @brief 特征点匹配
 * @param src1 第一张图像
 * @param src2 第二张图像
 * @param matches 匹配结果
 * @param keypoints1 第一张图像的特征点
 * @param keypoints2 第二张图像的特征点
 */
void feature_point_matching(const cv::Mat& src1,
                          const cv::Mat& src2,
                          std::vector<cv::DMatch>& matches,
                          std::vector<cv::KeyPoint>& keypoints1,
                          std::vector<cv::KeyPoint>& keypoints2);

/**
 * @brief 绘制匹配结果
 * @param src 输入图像
 * @param templ 模板图像
 * @param result 匹配结果
 * @param method 匹配方法
 * @return 绘制了匹配结果的图像
 */
cv::Mat draw_matching_result(const cv::Mat& src,
                           const cv::Mat& templ,
                           const cv::Mat& result,
                           int method);

} // namespace ip101

#endif // IMAGE_MATCHING_HPP