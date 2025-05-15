#ifndef IMAGE_MATCHING_HPP
#define IMAGE_MATCHING_HPP

#include <opencv2/opencv.hpp>
#include <immintrin.h>  // For SIMD instructions (AVX2)
#include <omp.h>        // For OpenMP parallel computing
#include <vector>

namespace ip101 {

/**
 * @brief SSD template matching
 * @param src Input image
 * @param templ Template image
 * @param result Matching result
 * @param method Matching method (default TM_SQDIFF)
 */
void ssd_matching(const cv::Mat& src,
                 const cv::Mat& templ,
                 cv::Mat& result,
                 int method = cv::TM_SQDIFF);

/**
 * @brief SAD template matching
 * @param src Input image
 * @param templ Template image
 * @param result Matching result
 */
void sad_matching(const cv::Mat& src,
                 const cv::Mat& templ,
                 cv::Mat& result);

/**
 * @brief NCC template matching
 * @param src Input image
 * @param templ Template image
 * @param result Matching result
 */
void ncc_matching(const cv::Mat& src,
                 const cv::Mat& templ,
                 cv::Mat& result);

/**
 * @brief ZNCC template matching
 * @param src Input image
 * @param templ Template image
 * @param result Matching result
 */
void zncc_matching(const cv::Mat& src,
                  const cv::Mat& templ,
                  cv::Mat& result);

/**
 * @brief Feature point matching
 * @param src1 First image
 * @param src2 Second image
 * @param matches Matching results
 * @param keypoints1 Feature points of the first image
 * @param keypoints2 Feature points of the second image
 */
void feature_point_matching(const cv::Mat& src1,
                          const cv::Mat& src2,
                          std::vector<cv::DMatch>& matches,
                          std::vector<cv::KeyPoint>& keypoints1,
                          std::vector<cv::KeyPoint>& keypoints2);

/**
 * @brief Draw matching results
 * @param src Input image
 * @param templ Template image
 * @param result Matching result
 * @param method Matching method
 * @return Image with matching results drawn
 */
cv::Mat draw_matching_result(const cv::Mat& src,
                           const cv::Mat& templ,
                           const cv::Mat& result,
                           int method);

} // namespace ip101

#endif // IMAGE_MATCHING_HPP