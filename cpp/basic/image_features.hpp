#ifndef IMAGE_FEATURES_HPP
#define IMAGE_FEATURES_HPP

#include <opencv2/opencv.hpp>
#include <immintrin.h>  // 用于SIMD指令(AVX2)
#include <omp.h>        // 用于OpenMP并行计算
#include <vector>

namespace ip101 {

/**
 * @brief HOG特征提取
 * @param src 输入图像
 * @param features 输出特征向量
 * @param cell_size 单元格大小
 * @param block_size 块大小
 * @param bin_num 方向直方图的bin数量
 */
void hog_features(const cv::Mat& src,
                 std::vector<float>& features,
                 int cell_size = 8,
                 int block_size = 2,
                 int bin_num = 9);

/**
 * @brief LBP特征提取
 * @param src 输入图像
 * @param dst 输出LBP图像
 * @param radius LBP半径
 * @param neighbors 邻域点数
 */
void lbp_features(const cv::Mat& src,
                 cv::Mat& dst,
                 int radius = 1,
                 int neighbors = 8);

/**
 * @brief Haar特征提取
 * @param src 输入图像
 * @param features 输出特征向量
 * @param min_size 最小特征尺寸
 * @param max_size 最大特征尺寸
 */
void haar_features(const cv::Mat& src,
                  std::vector<float>& features,
                  cv::Size min_size = cv::Size(24, 24),
                  cv::Size max_size = cv::Size(48, 48));

/**
 * @brief Gabor特征提取
 * @param src 输入图像
 * @param features 输出特征向量
 * @param scales 尺度数量
 * @param orientations 方向数量
 */
void gabor_features(const cv::Mat& src,
                   std::vector<float>& features,
                   int scales = 5,
                   int orientations = 8);

/**
 * @brief 颜色直方图特征
 * @param src 输入图像
 * @param hist 输出直方图
 * @param bins 每个通道的bin数量
 */
void color_histogram(const cv::Mat& src,
                    cv::Mat& hist,
                    const std::vector<int>& bins = {8, 8, 8});

/**
 * @brief 创建Gabor滤波器组
 * @param scales 尺度数量
 * @param orientations 方向数量
 * @param size 滤波器大小
 * @return Gabor滤波器组
 */
std::vector<cv::Mat> create_gabor_filters(int scales,
                                        int orientations,
                                        cv::Size size = cv::Size(31, 31));

/**
 * @brief 计算积分图像
 * @param src 输入图像
 * @param integral 输出积分图像
 */
void compute_integral_image(const cv::Mat& src,
                          cv::Mat& integral);

/**
 * @brief 计算方向梯度直方图
 * @param magnitude 梯度幅值
 * @param angle 梯度方向
 * @param hist 输出直方图
 * @param bin_num bin数量
 */
void compute_gradient_histogram(const cv::Mat& magnitude,
                              const cv::Mat& angle,
                              std::vector<float>& hist,
                              int bin_num = 9);

} // namespace ip101

#endif // IMAGE_FEATURES_HPP