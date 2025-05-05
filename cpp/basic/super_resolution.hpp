#ifndef SUPER_RESOLUTION_HPP
#define SUPER_RESOLUTION_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace ip101 {

/**
 * @brief 双三次插值超分辨率
 * @param src 输入低分辨率图像
 * @param scale_factor 放大倍数
 * @return 超分辨率结果
 */
cv::Mat bicubic_sr(
    const cv::Mat& src,
    float scale_factor);

/**
 * @brief 基于稀疏表示的超分辨率
 * @param src 输入低分辨率图像
 * @param scale_factor 放大倍数
 * @param dict_size 字典大小
 * @param patch_size 块大小
 * @return 超分辨率结果
 */
cv::Mat sparse_sr(
    const cv::Mat& src,
    float scale_factor,
    int dict_size = 512,
    int patch_size = 5);

/**
 * @brief 基于深度学习的超分辨率(SRCNN)
 * @param src 输入低分辨率图像
 * @param scale_factor 放大倍数
 * @return 超分辨率结果
 */
cv::Mat srcnn_sr(
    const cv::Mat& src,
    float scale_factor);

/**
 * @brief 多帧超分辨率
 * @param frames 输入低分辨率图像序列
 * @param scale_factor 放大倍数
 * @return 超分辨率结果
 */
cv::Mat multi_frame_sr(
    const std::vector<cv::Mat>& frames,
    float scale_factor);

/**
 * @brief 基于自适应权重的超分辨率
 * @param src 输入低分辨率图像
 * @param scale_factor 放大倍数
 * @param patch_size 块大小
 * @param search_window 搜索窗口大小
 * @return 超分辨率结果
 */
cv::Mat adaptive_weight_sr(
    const cv::Mat& src,
    float scale_factor,
    int patch_size = 5,
    int search_window = 21);

/**
 * @brief 基于迭代反投影的超分辨率
 * @param src 输入低分辨率图像
 * @param scale_factor 放大倍数
 * @param num_iterations 迭代次数
 * @return 超分辨率结果
 */
cv::Mat iterative_backprojection_sr(
    const cv::Mat& src,
    float scale_factor,
    int num_iterations = 30);

/**
 * @brief 基于梯度引导的超分辨率
 * @param src 输入低分辨率图像
 * @param scale_factor 放大倍数
 * @param lambda 梯度权重
 * @return 超分辨率结果
 */
cv::Mat gradient_guided_sr(
    const cv::Mat& src,
    float scale_factor,
    float lambda = 0.1);

/**
 * @brief 基于自相似性的超分辨率
 * @param src 输入低分辨率图像
 * @param scale_factor 放大倍数
 * @param patch_size 块大小
 * @param num_similar 相似块数量
 * @return 超分辨率结果
 */
cv::Mat self_similarity_sr(
    const cv::Mat& src,
    float scale_factor,
    int patch_size = 7,
    int num_similar = 10);

} // namespace ip101

#endif // SUPER_RESOLUTION_HPP