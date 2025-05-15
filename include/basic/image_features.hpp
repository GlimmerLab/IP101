#ifndef IMAGE_FEATURES_HPP
#define IMAGE_FEATURES_HPP

#include <opencv2/opencv.hpp>
#include <immintrin.h>  // For SIMD instructions (AVX2)
#include <omp.h>        // For OpenMP parallel computing
#include <vector>

namespace ip101 {

/**
 * @brief HOG feature extraction
 * @param src Input image
 * @param features Output feature vector
 * @param cell_size Cell size
 * @param block_size Block size
 * @param bin_num Number of bins in orientation histogram
 */
void hog_features(const cv::Mat& src,
                 std::vector<float>& features,
                 int cell_size = 8,
                 int block_size = 2,
                 int bin_num = 9);

/**
 * @brief LBP feature extraction
 * @param src Input image
 * @param dst Output LBP image
 * @param radius LBP radius
 * @param neighbors Number of neighboring points
 */
void lbp_features(const cv::Mat& src,
                 cv::Mat& dst,
                 int radius = 1,
                 int neighbors = 8);

/**
 * @brief Haar feature extraction
 * @param src Input image
 * @param features Output feature vector
 * @param min_size Minimum feature size
 * @param max_size Maximum feature size
 */
void haar_features(const cv::Mat& src,
                  std::vector<float>& features,
                  cv::Size min_size = cv::Size(24, 24),
                  cv::Size max_size = cv::Size(48, 48));

/**
 * @brief Gabor feature extraction
 * @param src Input image
 * @param features Output feature vector
 * @param scales Number of scales
 * @param orientations Number of orientations
 */
void gabor_features(const cv::Mat& src,
                   std::vector<float>& features,
                   int scales = 5,
                   int orientations = 8);

/**
 * @brief Color histogram features
 * @param src Input image
 * @param hist Output histogram
 * @param bins Number of bins for each channel
 */
void color_histogram(const cv::Mat& src,
                    cv::Mat& hist,
                    const std::vector<int>& bins = {8, 8, 8});

/**
 * @brief Create Gabor filter bank
 * @param scales Number of scales
 * @param orientations Number of orientations
 * @param size Filter size
 * @return Gabor filter bank
 */
std::vector<cv::Mat> create_gabor_filters(int scales,
                                        int orientations,
                                        cv::Size size = cv::Size(31, 31));

/**
 * @brief Compute integral image
 * @param src Input image
 * @param integral Output integral image
 */
void compute_integral_image(const cv::Mat& src,
                          cv::Mat& integral);

/**
 * @brief Compute gradient histogram
 * @param magnitude Gradient magnitude
 * @param angle Gradient orientation
 * @param hist Output histogram
 * @param bin_num Number of bins
 */
void compute_gradient_histogram(const cv::Mat& magnitude,
                              const cv::Mat& angle,
                              std::vector<float>& hist,
                              int bin_num = 9);

} // namespace ip101

#endif // IMAGE_FEATURES_HPP