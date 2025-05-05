#ifndef IMAGE_COMPRESSION_HPP
#define IMAGE_COMPRESSION_HPP

#include <opencv2/opencv.hpp>
#include <immintrin.h>  // 用于SIMD指令(AVX2)
#include <omp.h>        // 用于OpenMP并行计算
#include <vector>
#include <string>

namespace ip101 {

/**
 * @brief 无损压缩(RLE编码)
 * @param src 输入图像
 * @param encoded 压缩后的数据
 * @return 压缩率
 */
double rle_encode(const cv::Mat& src, std::vector<uchar>& encoded);

/**
 * @brief 无损解压缩(RLE解码)
 * @param encoded 压缩数据
 * @param dst 解压后的图像
 * @param original_size 原始图像大小
 */
void rle_decode(const std::vector<uchar>& encoded,
                cv::Mat& dst,
                const cv::Size& original_size);

/**
 * @brief JPEG压缩
 * @param src 输入图像
 * @param dst 压缩后的图像
 * @param quality 压缩质量(1-100)
 * @return 压缩率
 */
double jpeg_compress_manual(const cv::Mat& src, cv::Mat& dst,
                          int quality = 80);

/**
 * @brief 分形压缩
 * @param src 输入图像
 * @param dst 压缩后的图像
 * @param block_size 分块大小
 * @return 压缩率
 */
double fractal_compress(const cv::Mat& src, cv::Mat& dst,
                       int block_size = 8);

/**
 * @brief 小波压缩
 * @param src 输入图像
 * @param dst 压缩后的图像
 * @param level 小波分解层数
 * @param threshold 阈值
 * @return 压缩率
 */
double wavelet_compress(const cv::Mat& src, cv::Mat& dst,
                       int level = 3,
                       double threshold = 10.0);

/**
 * @brief 计算压缩率
 * @param original_size 原始大小
 * @param compressed_size 压缩后大小
 * @return 压缩率
 */
double compute_compression_ratio(size_t original_size,
                               size_t compressed_size);

/**
 * @brief 计算图像质量评价指标(PSNR)
 * @param original 原始图像
 * @param compressed 压缩图像
 * @return PSNR值
 */
double compute_psnr(const cv::Mat& original,
                   const cv::Mat& compressed);

} // namespace ip101

#endif // IMAGE_COMPRESSION_HPP