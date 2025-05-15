#ifndef IMAGE_COMPRESSION_HPP
#define IMAGE_COMPRESSION_HPP

#include <opencv2/opencv.hpp>
#include <immintrin.h>  // For SIMD instructions (AVX2)
#include <omp.h>        // For OpenMP parallel computing
#include <vector>
#include <string>

namespace ip101 {

/**
 * @brief Lossless compression (RLE encoding)
 * @param src Input image
 * @param encoded Compressed data
 * @return Compression ratio
 */
double rle_encode(const cv::Mat& src, std::vector<uchar>& encoded);

/**
 * @brief Lossless decompression (RLE decoding)
 * @param encoded Compressed data
 * @param dst Decompressed image
 * @param original_size Original image size
 */
void rle_decode(const std::vector<uchar>& encoded,
                cv::Mat& dst,
                const cv::Size& original_size);

/**
 * @brief JPEG compression
 * @param src Input image
 * @param dst Compressed image
 * @param quality Compression quality (1-100)
 * @return Compression ratio
 */
double jpeg_compress_manual(const cv::Mat& src, cv::Mat& dst,
                          int quality = 80);

/**
 * @brief Fractal compression
 * @param src Input image
 * @param dst Compressed image
 * @param block_size Block size
 * @return Compression ratio
 */
double fractal_compress(const cv::Mat& src, cv::Mat& dst,
                       int block_size = 8);

/**
 * @brief Wavelet compression
 * @param src Input image
 * @param dst Compressed image
 * @param level Wavelet decomposition level
 * @param threshold Threshold
 * @return Compression ratio
 */
double wavelet_compress(const cv::Mat& src, cv::Mat& dst,
                       int level = 3,
                       double threshold = 10.0);

/**
 * @brief Calculate compression ratio
 * @param original_size Original size
 * @param compressed_size Compressed size
 * @return Compression ratio
 */
double compute_compression_ratio(size_t original_size,
                               size_t compressed_size);

/**
 * @brief Calculate image quality metric (PSNR) for compressed images
 * @param original Original image
 * @param compressed Compressed image
 * @return PSNR value
 */
double compute_compression_psnr(const cv::Mat& original,
                   const cv::Mat& compressed);

} // namespace ip101

#endif // IMAGE_COMPRESSION_HPP