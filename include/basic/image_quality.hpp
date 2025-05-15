#ifndef IMAGE_QUALITY_HPP
#define IMAGE_QUALITY_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace ip101 {

/**
 * @brief Calculate Peak Signal-to-Noise Ratio (PSNR)
 * @param src1 Original image
 * @param src2 Comparison image
 * @return PSNR value (dB)
 */
double compute_psnr(
    const cv::Mat& src1,
    const cv::Mat& src2);

/**
 * @brief Calculate Structural Similarity Index (SSIM)
 * @param src1 Original image
 * @param src2 Comparison image
 * @param window_size Local window size
 * @return SSIM value [-1,1]
 */
double compute_ssim(
    const cv::Mat& src1,
    const cv::Mat& src2,
    int window_size = 11);

/**
 * @brief Calculate Mean Square Error (MSE)
 * @param src1 Original image
 * @param src2 Comparison image
 * @return MSE value
 */
double compute_mse(
    const cv::Mat& src1,
    const cv::Mat& src2);

/**
 * @brief Calculate Visual Information Fidelity (VIF)
 * @param src1 Original image
 * @param src2 Comparison image
 * @param num_scales Number of scales
 * @return VIF value [0,1]
 */
double compute_vif(
    const cv::Mat& src1,
    const cv::Mat& src2,
    int num_scales = 4);

/**
 * @brief Calculate Natural Image Quality Evaluator (NIQE)
 * @param src Input image
 * @param patch_size Patch size
 * @return NIQE value (lower is better)
 */
double compute_niqe(
    const cv::Mat& src,
    int patch_size = 96);

/**
 * @brief Calculate Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE)
 * @param src Input image
 * @return BRISQUE value (lower is better)
 */
double compute_brisque(
    const cv::Mat& src);

/**
 * @brief Calculate Multi-Scale Structural Similarity (MS-SSIM)
 * @param src1 Original image
 * @param src2 Comparison image
 * @param num_scales Number of scales
 * @return MS-SSIM value [0,1]
 */
double compute_msssim(
    const cv::Mat& src1,
    const cv::Mat& src2,
    int num_scales = 5);

/**
 * @brief Calculate Feature Similarity (FSIM)
 * @param src1 Original image
 * @param src2 Comparison image
 * @return FSIM value [0,1]
 */
double compute_fsim(
    const cv::Mat& src1,
    const cv::Mat& src2);

} // namespace ip101

#endif // IMAGE_QUALITY_HPP