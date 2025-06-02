#ifndef CARTOON_EFFECT_HPP
#define CARTOON_EFFECT_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace ip101 {
namespace advanced {

/**
 * @brief 卡通效果参数结构体
 */
struct CartoonParams {
    int edge_size;               // 边缘宽度
    int median_blur_size;        // 中值滤波核大小
    int bilateral_d;             // 双边滤波d参数
    double bilateral_sigma_color; // 双边滤波颜色标准差
    double bilateral_sigma_space; // 双边滤波空间标准差
    int quantize_levels;         // 颜色量化级别

    // 默认构造函数
    CartoonParams() :
        edge_size(1),
        median_blur_size(7),
        bilateral_d(9),
        bilateral_sigma_color(75.0),
        bilateral_sigma_space(75.0),
        quantize_levels(8)
    {}
};

/**
 * @brief 卡通化效果算法
 * @param src 输入图像
 * @param dst 输出图像
 * @param params 卡通效果参数
 */
void cartoon_effect(const cv::Mat& src, cv::Mat& dst, const CartoonParams& params = CartoonParams());

/**
 * @brief 边缘检测
 * @param src 输入图像
 * @param edges 边缘图像
 * @param edge_size 边缘宽度
 */
void detect_edges(const cv::Mat& src, cv::Mat& edges, int edge_size);

/**
 * @brief 颜色量化
 * @param src 输入图像
 * @param dst 输出图像
 * @param levels 量化级别
 */
void color_quantization(const cv::Mat& src, cv::Mat& dst, int levels);

/**
 * @brief 高级卡通化效果（带纹理增强）
 * @param src 输入图像
 * @param dst 输出图像
 * @param params 卡通效果参数
 * @param texture_strength 纹理增强强度
 */
void enhanced_cartoon_effect(const cv::Mat& src, cv::Mat& dst,
                            const CartoonParams& params = CartoonParams(),
                            double texture_strength = 0.5);

} // namespace advanced
} // namespace ip101

#endif // CARTOON_EFFECT_HPP