#ifndef FEATURE_EXTRACTION_HPP
#define FEATURE_EXTRACTION_HPP

#include <opencv2/opencv.hpp>
#include <immintrin.h>  // For SIMD instructions (AVX2)
#include <omp.h>        // For OpenMP parallel computing

namespace ip101 {

/**
 * @brief Harris corner detection
 * @param src Input image
 * @param dst Output image
 * @param block_size Neighborhood size
 * @param ksize Aperture parameter for Sobel operator
 * @param k Harris detector parameter
 * @param threshold Corner detection threshold
 */
void harris_corner_detection(const cv::Mat& src, cv::Mat& dst,
                           int block_size = 2, int ksize = 3,
                           double k = 0.04, double threshold = 0.01);

/**
 * @brief SIFT feature extraction
 * @param src Input image
 * @param dst Output image
 * @param nfeatures Number of features to extract (0 means extract all)
 */
void sift_features(const cv::Mat& src, cv::Mat& dst,
                  int nfeatures = 0);

/**
 * @brief SURF feature extraction (Note: May not be available in some OpenCV builds)
 * @param src Input image
 * @param dst Output image
 * @param hessian_threshold Hessian matrix threshold
 */
void surf_features(const cv::Mat& src, cv::Mat& dst,
                  double hessian_threshold = 100);

/**
 * @brief ORB feature extraction
 * @param src Input image
 * @param dst Output image
 * @param nfeatures Number of features to extract
 */
void orb_features(const cv::Mat& src, cv::Mat& dst,
                 int nfeatures = 500);

/**
 * @brief Feature matching
 * @param src1 First image
 * @param src2 Second image
 * @param dst Output image
 * @param method Feature extraction method ("sift", "surf", "orb")
 */
void feature_matching(const cv::Mat& src1, const cv::Mat& src2,
                     cv::Mat& dst, const std::string& method = "sift");

/**
 * @brief Manually implemented Harris corner detection
 * @param src Input image
 * @param dst Output image
 * @param k Harris detector parameter
 * @param window_size Window size
 * @param threshold Corner detection threshold
 */
void compute_harris_manual(const cv::Mat& src, cv::Mat& dst,
                          double k = 0.04, int window_size = 3,
                          double threshold = 0.01);

} // namespace ip101

#endif // FEATURE_EXTRACTION_HPP