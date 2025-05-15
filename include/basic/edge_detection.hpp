#ifndef EDGE_DETECTION_HPP
#define EDGE_DETECTION_HPP

#include <opencv2/opencv.hpp>
#include <immintrin.h> // For AVX2 instruction set
#include <omp.h>       // For OpenMP multi-threading

namespace ip101 {

/**
 * @brief Differential filter interface
 * @param src Input image
 * @param dst Output image
 * @param dx Derivative order in x-direction
 * @param dy Derivative order in y-direction
 * @param ksize Filter kernel size
 */
void differential_filter(const cv::Mat& src, cv::Mat& dst,
                       int dx, int dy, int ksize = 3);

/**
 * @brief Sobel operator edge detection
 * @param src Input image
 * @param dst Output image
 * @param dx Derivative order in x-direction
 * @param dy Derivative order in y-direction
 * @param ksize Filter kernel size
 * @param scale Scaling factor
 */
void sobel_filter(const cv::Mat& src, cv::Mat& dst,
                 int dx, int dy, int ksize = 3, double scale = 1.0);

/**
 * @brief Prewitt operator edge detection
 * @param src Input image
 * @param dst Output image
 * @param dx Derivative order in x-direction
 * @param dy Derivative order in y-direction
 */
void prewitt_filter(const cv::Mat& src, cv::Mat& dst,
                   int dx, int dy);

/**
 * @brief Laplacian operator edge detection
 * @param src Input image
 * @param dst Output image
 * @param ksize Filter kernel size
 * @param scale Scaling factor
 */
void laplacian_filter(const cv::Mat& src, cv::Mat& dst,
                     int ksize = 3, double scale = 1.0);

/**
 * @brief Emboss effect
 * @param src Input image
 * @param dst Output image
 * @param direction Emboss direction (0-7, corresponding to 8 directions)
 */
void emboss_effect(const cv::Mat& src, cv::Mat& dst,
                  int direction = 0);

/**
 * @brief Comprehensive edge detection
 * @param src Input image
 * @param dst Output image
 * @param method Detection method ("sobel", "prewitt", "laplacian")
 * @param thresh_val Edge threshold
 */
void edge_detection(const cv::Mat& src, cv::Mat& dst,
                   const std::string& method = "sobel",
                   double thresh_val = 128.0);

} // namespace ip101

#endif // EDGE_DETECTION_HPP