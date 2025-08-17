#ifndef UNSHARP_MASKING_HPP
#define UNSHARP_MASKING_HPP

#include <opencv2/opencv.hpp>

namespace ip101 {
namespace advanced {

/**
 * @brief 钝化蒙版锐化参数结构体
 */
struct UnsharpMaskingParams {
    double strength;        // 锐化强度，范围[0, 5]
    double threshold;       // 锐化阈值，小于此值的差异不进行锐化，范围[0, 255]
    double radius;          // 高斯模糊半径
    bool adaptive;          // 是否使用自适应锐化，根据区域内容调整锐化强度
    double edge_protect;    // 边缘保护强度，避免过度锐化边缘产生伪影，范围[0, 1]

    // 默认构造函数
    UnsharpMaskingParams() :
        strength(1.5),
        threshold(10),
        radius(1.0),
        adaptive(false),
        edge_protect(0.5)
    {}
};

/**
 * @brief 钝化蒙版锐化算法
 * @param src 输入图像
 * @param dst 输出图像
 * @param params 钝化蒙版参数
 */
void unsharp_masking(const cv::Mat& src, cv::Mat& dst, const UnsharpMaskingParams& params = UnsharpMaskingParams());

/**
 * @brief 基本钝化蒙版锐化
 * @param src 输入图像
 * @param dst 输出图像
 * @param strength 锐化强度，范围[0, 5]
 * @param radius 高斯模糊半径
 */
void basic_unsharp_masking(const cv::Mat& src, cv::Mat& dst, double strength = 1.5, double radius = 1.0);

/**
 * @brief 高频防抖钝化蒙版锐化
 * @param src 输入图像
 * @param dst 输出图像
 * @param strength 锐化强度，范围[0, 5]
 * @param radius 高斯模糊半径
 * @param threshold 锐化阈值，范围[0, 255]
 */
void high_pass_unsharp_masking(const cv::Mat& src, cv::Mat& dst, double strength = 1.5,
                             double radius = 1.0, double threshold = 10);

/**
 * @brief 自适应钝化蒙版锐化
 * @param src 输入图像
 * @param dst 输出图像
 * @param strength 锐化强度，范围[0, 5]
 * @param radius 高斯模糊半径
 * @param edge_protect 边缘保护强度，范围[0, 1]
 */
void adaptive_unsharp_masking(const cv::Mat& src, cv::Mat& dst, double strength = 1.5,
                            double radius = 1.0, double edge_protect = 0.5);

} // namespace advanced
} // namespace ip101

#endif // UNSHARP_MASKING_HPP