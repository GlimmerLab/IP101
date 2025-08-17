#ifndef SPHERIZE_HPP
#define SPHERIZE_HPP

#include <opencv2/opencv.hpp>

namespace ip101 {
namespace advanced {

/**
 * @brief 球面化效果参数结构体
 */
struct SpherizeParams {
    double strength;         // 球面化强度，范围[0, 1]
    double radius;           // 球面效果半径，相对于图像较短边的比例，范围[0, 1]
    cv::Point2f center;      // 球面中心点，默认为图像中心
    bool invert;             // 是否反向球面化（凹陷效果）

    // 默认构造函数
    SpherizeParams() :
        strength(0.5),
        radius(0.8),
        center(cv::Point2f(-1, -1)),  // -1表示使用图像中心
        invert(false)
    {}
};

/**
 * @brief 球面化效果算法
 * @param src 输入图像
 * @param dst 输出图像
 * @param params 球面化参数
 */
void spherize(const cv::Mat& src, cv::Mat& dst, const SpherizeParams& params = SpherizeParams());

/**
 * @brief 向外球面化效果，使图像呈现凸出效果
 * @param src 输入图像
 * @param dst 输出图像
 * @param strength 球面化强度，范围[0, 1]
 * @param center 球面中心点，默认为图像中心
 */
void bulge_effect(const cv::Mat& src, cv::Mat& dst, double strength = 0.5,
                 const cv::Point2f& center = cv::Point2f(-1, -1));

/**
 * @brief 向内球面化效果，使图像呈现凹陷效果
 * @param src 输入图像
 * @param dst 输出图像
 * @param strength 球面化强度，范围[0, 1]
 * @param center 球面中心点，默认为图像中心
 */
void pinch_effect(const cv::Mat& src, cv::Mat& dst, double strength = 0.5,
                 const cv::Point2f& center = cv::Point2f(-1, -1));

/**
 * @brief 鱼眼效果变换
 * @param src 输入图像
 * @param dst 输出图像
 * @param strength 变换强度，范围[0, 1]
 */
void fisheye_effect(const cv::Mat& src, cv::Mat& dst, double strength = 0.5);

/**
 * @brief 双线性插值函数
 * @param img 输入图像
 * @param x x坐标（浮点数）
 * @param y y坐标（浮点数）
 * @return 插值后的像素值
 */
cv::Vec3b interpolate_pixel(const cv::Mat& img, float x, float y);

} // namespace advanced
} // namespace ip101

#endif // SPHERIZE_HPP