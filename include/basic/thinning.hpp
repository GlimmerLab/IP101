#ifndef THINNING_HPP
#define THINNING_HPP

#include <opencv2/opencv.hpp>

namespace ip101 {

/**
 * @brief Basic thinning algorithm
 * @param src Input binary image
 * @param dst Output thinned image
 */
void basic_thinning(const cv::Mat& src, cv::Mat& dst);

/**
 * @brief Hilditch thinning algorithm
 * @param src Input binary image
 * @param dst Output thinned image
 */
void hilditch_thinning(const cv::Mat& src, cv::Mat& dst);

/**
 * @brief Zhang-Suen thinning algorithm
 * @param src Input binary image
 * @param dst Output thinned image
 */
void zhang_suen_thinning(const cv::Mat& src, cv::Mat& dst);

/**
 * @brief Skeleton extraction
 * @param src Input binary image
 * @param dst Output skeleton image
 */
void skeleton_extraction(const cv::Mat& src, cv::Mat& dst);

/**
 * @brief Medial axis transform
 * @param src Input binary image
 * @param dst Output medial axis image
 * @param dist_transform Distance transform image (optional)
 */
void medial_axis_transform(const cv::Mat& src,
                         cv::Mat& dst,
                         cv::Mat& dist_transform);

} // namespace ip101

#endif // THINNING_HPP