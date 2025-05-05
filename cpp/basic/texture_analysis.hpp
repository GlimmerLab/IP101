#ifndef TEXTURE_ANALYSIS_HPP
#define TEXTURE_ANALYSIS_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace ip101 {

/**
 * @brief 计算灰度共生矩阵(GLCM)
 * @param src 输入灰度图像
 * @param distance 像素对之间的距离
 * @param angle 角度(0, 45, 90, 135)
 * @return 灰度共生矩阵
 */
cv::Mat compute_glcm(
    const cv::Mat& src,
    int distance = 1,
    int angle = 0);

/**
 * @brief 从GLCM中提取Haralick纹理特征
 * @param glcm 灰度共生矩阵
 * @return 纹理特征向量(对比度、相关性、能量、同质性等)
 */
std::vector<double> extract_haralick_features(
    const cv::Mat& glcm);

/**
 * @brief 计算纹理统计特征
 * @param src 输入灰度图像
 * @param window_size 局部窗口大小
 * @return 特征图像(均值、方差、偏度、峰度)
 */
std::vector<cv::Mat> compute_statistical_features(
    const cv::Mat& src,
    int window_size = 3);

/**
 * @brief 计算局部二值模式(LBP)特征
 * @param src 输入灰度图像
 * @param radius LBP算子半径
 * @param neighbors 邻域采样点数
 * @return LBP特征图像
 */
cv::Mat compute_lbp(
    const cv::Mat& src,
    int radius = 1,
    int neighbors = 8);

/**
 * @brief 生成Gabor滤波器组
 * @param ksize 滤波器大小
 * @param sigma 高斯包络的标准差
 * @param theta 方向角度数
 * @param lambda 正弦波波长
 * @param gamma 空间纵横比
 * @param psi 相位偏移
 * @return Gabor滤波器组
 */
std::vector<cv::Mat> generate_gabor_filters(
    int ksize = 31,
    double sigma = 3.0,
    int theta = 8,
    double lambda = 4.0,
    double gamma = 0.5,
    double psi = 0.0);

/**
 * @brief 提取Gabor纹理特征
 * @param src 输入灰度图像
 * @param filters Gabor滤波器组
 * @return Gabor特征图像组
 */
std::vector<cv::Mat> extract_gabor_features(
    const cv::Mat& src,
    const std::vector<cv::Mat>& filters);

/**
 * @brief 纹理分类
 * @param features 纹理特征向量
 * @param labels 训练样本标签
 * @param train_features 训练样本特征
 * @return 分类结果标签
 */
int classify_texture(
    const std::vector<double>& features,
    const std::vector<int>& labels,
    const std::vector<std::vector<double>>& train_features);

} // namespace ip101

#endif // TEXTURE_ANALYSIS_HPP