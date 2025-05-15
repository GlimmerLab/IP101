#ifndef MORPHOLOGY_HPP
#define MORPHOLOGY_HPP

#include <opencv2/opencv.hpp>
#include <immintrin.h>  // For SIMD instructions (AVX2)
#include <omp.h>        // For OpenMP parallel computing

namespace ip101 {

/**
 * @brief Dilation operation
 * @param src Input image
 * @param dst Output image
 * @param kernel Structuring element
 * @param iterations Number of iterations
 */
void dilate_manual(const cv::Mat& src, cv::Mat& dst,
                  const cv::Mat& kernel = cv::Mat(),
                  int iterations = 1);

/**
 * @brief Erosion operation
 * @param src Input image
 * @param dst Output image
 * @param kernel Structuring element
 * @param iterations Number of iterations
 */
void erode_manual(const cv::Mat& src, cv::Mat& dst,
                 const cv::Mat& kernel = cv::Mat(),
                 int iterations = 1);

/**
 * @brief Opening operation
 * @param src Input image
 * @param dst Output image
 * @param kernel Structuring element
 * @param iterations Number of iterations
 */
void opening_manual(const cv::Mat& src, cv::Mat& dst,
                   const cv::Mat& kernel = cv::Mat(),
                   int iterations = 1);

/**
 * @brief Closing operation
 * @param src Input image
 * @param dst Output image
 * @param kernel Structuring element
 * @param iterations Number of iterations
 */
void closing_manual(const cv::Mat& src, cv::Mat& dst,
                   const cv::Mat& kernel = cv::Mat(),
                   int iterations = 1);

/**
 * @brief Morphological gradient
 * @param src Input image
 * @param dst Output image
 * @param kernel Structuring element
 */
void morphological_gradient_manual(const cv::Mat& src, cv::Mat& dst,
                                 const cv::Mat& kernel = cv::Mat());

/**
 * @brief Create structuring element
 * @param shape Shape type (MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE)
 * @param ksize Size of the structuring element
 * @return Structuring element matrix
 */
cv::Mat create_kernel(int shape, cv::Size ksize);

} // namespace ip101

#endif // MORPHOLOGY_HPP