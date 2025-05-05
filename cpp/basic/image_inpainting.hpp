#ifndef IMAGE_INPAINTING_HPP
#define IMAGE_INPAINTING_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace ip101 {

/**
 * @brief 基于扩散的图像修复
 * @param src 输入图像
 * @param mask 待修复区域掩码(255表示需要修复的区域)
 * @param radius 扩散半径
 * @param num_iterations 迭代次数
 * @return 修复后的图像
 */
cv::Mat diffusion_inpaint(
    const cv::Mat& src,
    const cv::Mat& mask,
    int radius = 3,
    int num_iterations = 100);

/**
 * @brief 基于块匹配的图像修复
 * @param src 输入图像
 * @param mask 待修复区域掩码(255表示需要修复的区域)
 * @param patch_size 块大小
 * @param search_area 搜索区域大小
 * @return 修复后的图像
 */
cv::Mat patch_match_inpaint(
    const cv::Mat& src,
    const cv::Mat& mask,
    int patch_size = 9,
    int search_area = 30);

/**
 * @brief 基于快速行进法的图像修复
 * @param src 输入图像
 * @param mask 待修复区域掩码(255表示需要修复的区域)
 * @param radius 修复半径
 * @return 修复后的图像
 */
cv::Mat fast_marching_inpaint(
    const cv::Mat& src,
    const cv::Mat& mask,
    int radius = 3);

/**
 * @brief 基于纹理合成的图像修复
 * @param src 输入图像
 * @param mask 待修复区域掩码(255表示需要修复的区域)
 * @param patch_size 纹理块大小
 * @param overlap 重叠区域大小
 * @return 修复后的图像
 */
cv::Mat texture_synthesis_inpaint(
    const cv::Mat& src,
    const cv::Mat& mask,
    int patch_size = 15,
    int overlap = 4);

/**
 * @brief 基于结构传播的图像修复
 * @param src 输入图像
 * @param mask 待修复区域掩码(255表示需要修复的区域)
 * @param patch_size 块大小
 * @param num_iterations 迭代次数
 * @return 修复后的图像
 */
cv::Mat structure_propagation_inpaint(
    const cv::Mat& src,
    const cv::Mat& mask,
    int patch_size = 9,
    int num_iterations = 10);

} // namespace ip101

#endif // IMAGE_INPAINTING_HPP