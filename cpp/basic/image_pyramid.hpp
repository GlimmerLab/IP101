#ifndef IMAGE_PYRAMID_HPP
#define IMAGE_PYRAMID_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace ip101 {

/**
 * @brief 构建高斯金字塔
 * @param src 输入图像
 * @param num_levels 金字塔层数
 * @return 高斯金字塔图像列表
 */
std::vector<cv::Mat> build_gaussian_pyramid(
    const cv::Mat& src,
    int num_levels);

/**
 * @brief 构建拉普拉斯金字塔
 * @param src 输入图像
 * @param num_levels 金字塔层数
 * @return 拉普拉斯金字塔图像列表
 */
std::vector<cv::Mat> build_laplacian_pyramid(
    const cv::Mat& src,
    int num_levels);

/**
 * @brief 图像融合
 * @param src1 输入图像1
 * @param src2 输入图像2
 * @param mask 融合掩码
 * @param num_levels 金字塔层数
 * @return 融合后的图像
 */
cv::Mat pyramid_blend(
    const cv::Mat& src1,
    const cv::Mat& src2,
    const cv::Mat& mask,
    int num_levels);

/**
 * @brief 构建SIFT尺度空间
 * @param src 输入图像
 * @param num_octaves 组数
 * @param num_scales 每组的尺度数
 * @param sigma 初始高斯模糊尺度
 * @return 尺度空间图像列表
 */
std::vector<std::vector<cv::Mat>> build_sift_scale_space(
    const cv::Mat& src,
    int num_octaves = 4,
    int num_scales = 5,
    float sigma = 1.6f);

/**
 * @brief 显著性检测
 * @param src 输入图像
 * @param num_levels 金字塔层数
 * @return 显著性图
 */
cv::Mat saliency_detection(
    const cv::Mat& src,
    int num_levels = 6);

/**
 * @brief 可视化金字塔
 * @param pyramid 金字塔图像列表
 * @param padding 图像间的间隔
 * @return 可视化结果
 */
cv::Mat visualize_pyramid(
    const std::vector<cv::Mat>& pyramid,
    int padding = 10);

} // namespace ip101

#endif // IMAGE_PYRAMID_HPP