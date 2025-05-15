#ifndef IMAGE_PYRAMID_HPP
#define IMAGE_PYRAMID_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace ip101 {

/**
 * @brief Build a Gaussian pyramid
 * @param src Input image
 * @param num_levels Number of pyramid levels
 * @return List of Gaussian pyramid images
 */
std::vector<cv::Mat> build_gaussian_pyramid(
    const cv::Mat& src,
    int num_levels);

/**
 * @brief Build a Laplacian pyramid
 * @param src Input image
 * @param num_levels Number of pyramid levels
 * @return List of Laplacian pyramid images
 */
std::vector<cv::Mat> build_laplacian_pyramid(
    const cv::Mat& src,
    int num_levels);

/**
 * @brief Image blending using pyramid
 * @param src1 Input image 1
 * @param src2 Input image 2
 * @param mask Blending mask
 * @param num_levels Number of pyramid levels
 * @return Blended image
 */
cv::Mat pyramid_blend(
    const cv::Mat& src1,
    const cv::Mat& src2,
    const cv::Mat& mask,
    int num_levels);

/**
 * @brief Build SIFT scale space
 * @param src Input image
 * @param num_octaves Number of octaves
 * @param num_scales Number of scales per octave
 * @param sigma Initial Gaussian blur sigma
 * @return Scale space image list
 */
std::vector<std::vector<cv::Mat>> build_sift_scale_space(
    const cv::Mat& src,
    int num_octaves = 4,
    int num_scales = 5,
    float sigma = 1.6f);

/**
 * @brief Saliency detection
 * @param src Input image
 * @param num_levels Number of pyramid levels
 * @return Saliency map
 */
cv::Mat saliency_detection(
    const cv::Mat& src,
    int num_levels = 6);

/**
 * @brief Visualize pyramid
 * @param pyramid List of pyramid images
 * @param padding Padding between images
 * @return Visualization result
 */
cv::Mat visualize_pyramid(
    const std::vector<cv::Mat>& pyramid,
    int padding = 10);

} // namespace ip101

#endif // IMAGE_PYRAMID_HPP