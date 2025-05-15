#ifndef FILTERING_HPP
#define FILTERING_HPP

#include <opencv2/opencv.hpp>
#include <immintrin.h>  // For SIMD instructions (AVX2)
#include <omp.h>        // For OpenMP parallel computing

namespace ip101 {

using namespace cv;
using namespace std;

// Original unoptimized implementation functions
/**
 * @brief Mean filter original implementation
 * @param src Input image
 * @param kernelSize Filter kernel size
 * @return Processed image
 */
Mat meanFilter(const Mat& src, int kernelSize);

/**
 * @brief Median filter original implementation
 * @param src Input image
 * @param kernelSize Filter kernel size
 * @return Processed image
 */
Mat medianFilter(const Mat& src, int kernelSize);

/**
 * @brief Gaussian filter original implementation
 * @param src Input image
 * @param kernelSize Filter kernel size
 * @param sigma Standard deviation of Gaussian function
 * @return Processed image
 */
Mat gaussianFilter(const Mat& src, int kernelSize, double sigma);

/**
 * @brief Mean pooling original implementation
 * @param src Input image
 * @param poolSize Pooling size
 * @return Processed image
 */
Mat meanPooling(const Mat& src, int poolSize);

/**
 * @brief Max pooling original implementation
 * @param src Input image
 * @param poolSize Pooling size
 * @return Processed image
 */
Mat maxPooling(const Mat& src, int poolSize);

// Optimized implementation functions
/**
 * @brief Mean filter optimized implementation
 * @param src Input image
 * @param kernelSize Filter kernel size
 * @return Processed image
 */
Mat meanFilter_optimized(const Mat& src, int kernelSize);

/**
 * @brief Median filter optimized implementation
 * @param src Input image
 * @param kernelSize Filter kernel size
 * @return Processed image
 */
Mat medianFilter_optimized(const Mat& src, int kernelSize);

/**
 * @brief Gaussian filter optimized implementation
 * @param src Input image
 * @param kernelSize Filter kernel size
 * @param sigma Standard deviation of Gaussian function
 * @return Processed image
 */
Mat gaussianFilter_optimized(const Mat& src, int kernelSize, double sigma);

/**
 * @brief Mean pooling optimized implementation
 * @param src Input image
 * @param poolSize Pooling size
 * @return Processed image
 */
Mat meanPooling_optimized(const Mat& src, int poolSize);

/**
 * @brief Max pooling optimized implementation
 * @param src Input image
 * @param poolSize Pooling size
 * @return Processed image
 */
Mat maxPooling_optimized(const Mat& src, int poolSize);

/**
 * @brief Create Gaussian kernel
 * @param ksize Kernel size
 * @param sigma_x Standard deviation in x direction
 * @param sigma_y Standard deviation in y direction
 * @return Gaussian kernel matrix
 */
Mat create_gaussian_kernel(int ksize, double sigma_x, double sigma_y);

} // namespace ip101

#endif // FILTERING_HPP