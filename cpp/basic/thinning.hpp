#ifndef THINNING_HPP
#define THINNING_HPP

#include <opencv2/opencv.hpp>

namespace ip101 {

/**
 * @brief 基本细化算法
 * @param src 输入二值图像
 * @param dst 输出细化图像
 */
void basic_thinning(const cv::Mat& src, cv::Mat& dst);

/**
 * @brief Hilditch细化算法
 * @param src 输入二值图像
 * @param dst 输出细化图像
 */
void hilditch_thinning(const cv::Mat& src, cv::Mat& dst);

/**
 * @brief Zhang-Suen细化算法
 * @param src 输入二值图像
 * @param dst 输出细化图像
 */
void zhang_suen_thinning(const cv::Mat& src, cv::Mat& dst);

/**
 * @brief 骨架提取
 * @param src 输入二值图像
 * @param dst 输出骨架图像
 */
void skeleton_extraction(const cv::Mat& src, cv::Mat& dst);

/**
 * @brief 中轴变换
 * @param src 输入二值图像
 * @param dst 输出中轴图像
 * @param dist_transform 距离变换图像(可选)
 */
void medial_axis_transform(const cv::Mat& src,
                         cv::Mat& dst,
                         cv::Mat& dist_transform);

} // namespace ip101

#endif // THINNING_HPP