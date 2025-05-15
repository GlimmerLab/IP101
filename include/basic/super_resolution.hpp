#ifndef SUPER_RESOLUTION_HPP
#define SUPER_RESOLUTION_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace ip101 {

/**
 * @brief Bicubic interpolation super resolution
 * @param src Input low-resolution image
 * @param scale_factor Scale factor
 * @return Super resolution result
 */
cv::Mat bicubic_sr(
    const cv::Mat& src,
    float scale_factor);

/**
 * @brief Sparse representation based super resolution
 * @param src Input low-resolution image
 * @param scale_factor Scale factor
 * @param dict_size Dictionary size
 * @param patch_size Patch size
 * @return Super resolution result
 */
cv::Mat sparse_sr(
    const cv::Mat& src,
    float scale_factor,
    int dict_size = 512,
    int patch_size = 5);

/**
 * @brief Deep learning based super resolution (SRCNN)
 * @param src Input low-resolution image
 * @param scale_factor Scale factor
 * @return Super resolution result
 */
cv::Mat srcnn_sr(
    const cv::Mat& src,
    float scale_factor);

/**
 * @brief Multi-frame super resolution
 * @param frames Input low-resolution image sequence
 * @param scale_factor Scale factor
 * @return Super resolution result
 */
cv::Mat multi_frame_sr(
    const std::vector<cv::Mat>& frames,
    float scale_factor);

/**
 * @brief Adaptive weight based super resolution
 * @param src Input low-resolution image
 * @param scale_factor Scale factor
 * @param patch_size Patch size
 * @param search_window Search window size
 * @return Super resolution result
 */
cv::Mat adaptive_weight_sr(
    const cv::Mat& src,
    float scale_factor,
    int patch_size = 5,
    int search_window = 21);

/**
 * @brief Iterative back-projection based super resolution
 * @param src Input low-resolution image
 * @param scale_factor Scale factor
 * @param num_iterations Number of iterations
 * @return Super resolution result
 */
cv::Mat iterative_backprojection_sr(
    const cv::Mat& src,
    float scale_factor,
    int num_iterations = 30);

/**
 * @brief Gradient guided super resolution
 * @param src Input low-resolution image
 * @param scale_factor Scale factor
 * @param lambda Gradient weight
 * @return Super resolution result
 */
cv::Mat gradient_guided_sr(
    const cv::Mat& src,
    float scale_factor,
    float lambda = 0.1);

/**
 * @brief Self-similarity based super resolution
 * @param src Input low-resolution image
 * @param scale_factor Scale factor
 * @param patch_size Patch size
 * @param num_similar Number of similar patches
 * @return Super resolution result
 */
cv::Mat self_similarity_sr(
    const cv::Mat& src,
    float scale_factor,
    int patch_size = 7,
    int num_similar = 10);

} // namespace ip101

#endif // SUPER_RESOLUTION_HPP