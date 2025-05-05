#ifndef IMAGE_QUALITY_HPP
#define IMAGE_QUALITY_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace ip101 {

/**
 * @brief 计算峰值信噪比(PSNR)
 * @param src1 原始图像
 * @param src2 对比图像
 * @return PSNR值(dB)
 */
double compute_psnr(
    const cv::Mat& src1,
    const cv::Mat& src2);

/**
 * @brief 计算结构相似性(SSIM)
 * @param src1 原始图像
 * @param src2 对比图像
 * @param window_size 局部窗口大小
 * @return SSIM值[-1,1]
 */
double compute_ssim(
    const cv::Mat& src1,
    const cv::Mat& src2,
    int window_size = 11);

/**
 * @brief 计算均方误差(MSE)
 * @param src1 原始图像
 * @param src2 对比图像
 * @return MSE值
 */
double compute_mse(
    const cv::Mat& src1,
    const cv::Mat& src2);

/**
 * @brief 计算视觉信息保真度(VIF)
 * @param src1 原始图像
 * @param src2 对比图像
 * @param num_scales 尺度数
 * @return VIF值[0,1]
 */
double compute_vif(
    const cv::Mat& src1,
    const cv::Mat& src2,
    int num_scales = 4);

/**
 * @brief 计算无参考图像质量评价指标(NIQE)
 * @param src 输入图像
 * @param patch_size 块大小
 * @return NIQE值(越小越好)
 */
double compute_niqe(
    const cv::Mat& src,
    int patch_size = 96);

/**
 * @brief 计算BRISQUE无参考质量评价指标
 * @param src 输入图像
 * @return BRISQUE值(越小越好)
 */
double compute_brisque(
    const cv::Mat& src);

/**
 * @brief 计算多尺度结构相似性(MS-SSIM)
 * @param src1 原始图像
 * @param src2 对比图像
 * @param num_scales 尺度数
 * @return MS-SSIM值[0,1]
 */
double compute_msssim(
    const cv::Mat& src1,
    const cv::Mat& src2,
    int num_scales = 5);

/**
 * @brief 计算特征相似性(FSIM)
 * @param src1 原始图像
 * @param src2 对比图像
 * @return FSIM值[0,1]
 */
double compute_fsim(
    const cv::Mat& src1,
    const cv::Mat& src2);

} // namespace ip101

#endif // IMAGE_QUALITY_HPP