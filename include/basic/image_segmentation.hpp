#ifndef IMAGE_SEGMENTATION_HPP
#define IMAGE_SEGMENTATION_HPP

#include <opencv2/opencv.hpp>
#include <immintrin.h>  // For SIMD instructions (AVX2)
#include <omp.h>        // For OpenMP parallel computing

namespace ip101 {

/**
 * @brief Threshold segmentation
 * @param src Input image
 * @param dst Output image
 * @param threshold Threshold value
 * @param max_val Maximum value
 * @param type Threshold type
 */
void threshold_segmentation(const cv::Mat& src, cv::Mat& dst,
                          double threshold, double max_val = 255,
                          int type = cv::THRESH_BINARY);

/**
 * @brief K-means segmentation
 * @param src Input image
 * @param dst Output image
 * @param k Number of clusters
 * @param max_iter Maximum iterations
 */
void kmeans_segmentation(const cv::Mat& src, cv::Mat& dst,
                        int k = 3, int max_iter = 100);

/**
 * @brief Region growing segmentation
 * @param src Input image
 * @param dst Output image
 * @param seed_points Seed points
 * @param threshold Growing threshold
 */
void region_growing(const cv::Mat& src, cv::Mat& dst,
                   const std::vector<cv::Point>& seed_points,
                   double threshold = 10.0);

/**
 * @brief Watershed segmentation
 * @param src Input image
 * @param markers Marker image
 * @param dst Output image
 */
void watershed_segmentation(const cv::Mat& src,
                          cv::Mat& markers,
                          cv::Mat& dst);

/**
 * @brief Graph cut segmentation
 * @param src Input image
 * @param dst Output image
 * @param rect Foreground rectangle
 */
void graph_cut_segmentation(const cv::Mat& src, cv::Mat& dst,
                          const cv::Rect& rect);

} // namespace ip101

#endif // IMAGE_SEGMENTATION_HPP