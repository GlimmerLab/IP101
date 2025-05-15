#ifndef IMAGE_ENHANCEMENT_HPP
#define IMAGE_ENHANCEMENT_HPP

#include <opencv2/opencv.hpp>
#include <immintrin.h>  // For SIMD instructions (AVX2)
#include <omp.h>        // For OpenMP parallel computing

namespace ip101 {

/**
 * @brief Histogram equalization
 * @param src Input image
 * @param dst Output image
 * @param method Equalization method ("global", "adaptive", "clahe")
 * @param clip_limit Contrast limit threshold for CLAHE
 * @param grid_size Grid size for CLAHE
 */
void histogram_equalization(const cv::Mat& src, cv::Mat& dst,
                          const std::string& method = "global",
                          double clip_limit = 40.0,
                          cv::Size grid_size = cv::Size(8, 8));

/**
 * @brief Gamma correction
 * @param src Input image
 * @param dst Output image
 * @param gamma Gamma value
 */
void gamma_correction(const cv::Mat& src, cv::Mat& dst, double gamma);

/**
 * @brief Contrast stretching
 * @param src Input image
 * @param dst Output image
 * @param min_val Minimum output value
 * @param max_val Maximum output value
 */
void contrast_stretching(const cv::Mat& src, cv::Mat& dst,
                        double min_val = 0, double max_val = 255);

/**
 * @brief Brightness adjustment
 * @param src Input image
 * @param dst Output image
 * @param beta Brightness adjustment value (-255 to 255)
 */
void brightness_adjustment(const cv::Mat& src, cv::Mat& dst, double beta);

/**
 * @brief Saturation adjustment
 * @param src Input image
 * @param dst Output image
 * @param saturation Saturation adjustment value (0 to 2)
 */
void saturation_adjustment(const cv::Mat& src, cv::Mat& dst, double saturation);

/**
 * @brief Calculate image histogram
 * @param src Input image
 * @param hist Output histogram
 * @param channel Channel index (for color images)
 */
void calculate_histogram(const cv::Mat& src, cv::Mat& hist, int channel = 0);

/**
 * @brief Calculate cumulative distribution function
 * @param histogram Histogram
 * @param cdf Output cumulative distribution function
 */
void calculate_cdf(const cv::Mat& hist, cv::Mat& cdf);

/**
 * @brief Local histogram equalization
 * @param src Input image
 * @param dst Output image
 * @param window_size Local window size
 */
void local_histogram_equalization(const cv::Mat& src, cv::Mat& dst,
                                int window_size = 3);

/**
 * @brief CLAHE (Contrast Limited Adaptive Histogram Equalization)
 * @param src Input image
 * @param dst Output image
 * @param clip_limit Contrast limit threshold
 * @param grid_size Grid size
 */
void clahe(const cv::Mat& src, cv::Mat& dst,
          double clip_limit = 40.0,
          cv::Size grid_size = cv::Size(8, 8));

} // namespace ip101

#endif // IMAGE_ENHANCEMENT_HPP