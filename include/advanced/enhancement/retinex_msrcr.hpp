#ifndef RETINEX_MSRCR_HPP
#define RETINEX_MSRCR_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace ip101 {
namespace advanced {

/**
 * @brief 手动实现高斯核生成
 * @param size 核大小
 * @param sigma 高斯标准差
 * @return 高斯核矩阵
 */
cv::Mat createGaussianKernel(int size, double sigma);

/**
 * @brief 创建一维高斯核（用于分离卷积）
 * @param size 核大小
 * @param sigma 高斯标准差
 * @return 一维高斯核
 */
std::vector<float> createGaussianKernel1D(int size, double sigma);

/**
 * @brief 手动实现卷积操作（优化版）
 * @param src 输入图像
 * @param kernel 卷积核
 * @return 卷积结果
 */
cv::Mat convolve(const cv::Mat& src, const cv::Mat& kernel);

/**
 * @brief 手动实现高斯模糊（使用分离卷积加速）
 * @param src 输入图像
 * @param sigma 高斯标准差
 * @return 模糊后的图像
 */
cv::Mat gaussianBlur(const cv::Mat& src, double sigma);

/**
 * @brief 手动实现对数运算（优化版）
 * @param src 输入图像
 * @return 对数结果
 */
cv::Mat logarithm(const cv::Mat& src);

/**
 * @brief 手动实现指数运算（优化版）
 * @param src 输入图像
 * @return 指数结果
 */
cv::Mat exponential(const cv::Mat& src);

/**
 * @brief 优化的矩阵加法
 * @param src1 输入矩阵1
 * @param src2 输入矩阵2
 * @return 结果矩阵
 */
cv::Mat matAdd(const cv::Mat& src1, const cv::Mat& src2);

/**
 * @brief 优化的矩阵减法
 * @param src1 输入矩阵1
 * @param src2 输入矩阵2
 * @return 结果矩阵
 */
cv::Mat matSubtract(const cv::Mat& src1, const cv::Mat& src2);

/**
 * @brief 优化的矩阵乘法（元素级别）
 * @param src 输入矩阵
 * @param scale 缩放因子
 * @return 结果矩阵
 */
cv::Mat matMultiply(const cv::Mat& src, float scale);

/**
 * @brief Retinex MSRCR(多尺度Retinex带颜色恢复)算法实现
 * @param src 输入图像
 * @param dst 输出图像
 * @param sigma1 第一个高斯核的标准差
 * @param sigma2 第二个高斯核的标准差
 * @param sigma3 第三个高斯核的标准差
 * @param alpha 颜色恢复参数
 * @param beta 颜色恢复参数
 * @param gain 增益参数
 */
void retinex_msrcr(const cv::Mat& src, cv::Mat& dst,
                  double sigma1 = 15.0, double sigma2 = 80.0, double sigma3 = 250.0,
                  double alpha = 125.0, double beta = 46.0, double gain = 192.0);

} // namespace advanced
} // namespace ip101

#endif // RETINEX_MSRCR_HPP