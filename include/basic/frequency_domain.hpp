#ifndef FREQUENCY_DOMAIN_HPP
#define FREQUENCY_DOMAIN_HPP

#include <opencv2/opencv.hpp>
#include <immintrin.h>  // 用于SIMD指令(AVX2)
#include <omp.h>        // 用于OpenMP并行计算
#include <complex>

// Define DCT constants as they might not be available in some OpenCV versions
#ifndef DCT_FORWARD
#define DCT_FORWARD 0
#endif

#ifndef DCT_INVERSE
#define DCT_INVERSE 1
#endif

namespace ip101 {

// Forward declaration for internal helper function
void fft(std::vector<std::complex<double>>& data, int n, bool inverse);

/**
 * @brief 傅里叶变换
 * @param src 输入图像
 * @param dst 输出频谱图像
 * @param flags 变换标志(DFT_FORWARD或DFT_INVERSE)
 */
void fourier_transform_manual(const cv::Mat& src, cv::Mat& dst,
                            int flags = cv::DFT_COMPLEX_OUTPUT);

/**
 * @brief 频域滤波
 * @param src 输入图像
 * @param dst 输出图像
 * @param filter_type 滤波器类型("lowpass", "highpass", "bandpass")
 * @param cutoff_freq 截止频率
 */
void frequency_filter_manual(const cv::Mat& src, cv::Mat& dst,
                           const std::string& filter_type,
                           double cutoff_freq);

/**
 * @brief DCT变换
 * @param src 输入图像
 * @param dst 输出图像
 * @param flags 变换标志(DCT_FORWARD或DCT_INVERSE)
 */
void dct_transform_manual(const cv::Mat& src, cv::Mat& dst,
                         int flags = DCT_FORWARD);

/**
 * @brief 小波变换
 * @param src 输入图像
 * @param dst 输出图像
 * @param wavelet_type 小波类型("haar", "db1", "db2")
 * @param level 分解层数
 */
void wavelet_transform_manual(const cv::Mat& src, cv::Mat& dst,
                            const std::string& wavelet_type = "haar",
                            int level = 1);

/**
 * @brief 创建频域滤波器
 * @param size 滤波器大小
 * @param cutoff_freq 截止频率
 * @param filter_type 滤波器类型
 * @return 滤波器矩阵
 */
cv::Mat create_frequency_filter(const cv::Size& size,
                              double cutoff_freq,
                              const std::string& filter_type);

/**
 * @brief 频谱可视化
 * @param spectrum 频谱
 * @param dst 可视化结果
 */
void visualize_spectrum(const cv::Mat& spectrum, cv::Mat& dst);

} // namespace ip101

#endif // FREQUENCY_DOMAIN_HPP