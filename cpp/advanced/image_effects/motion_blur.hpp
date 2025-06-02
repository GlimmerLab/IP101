#ifndef MOTION_BLUR_HPP
#define MOTION_BLUR_HPP

#include <opencv2/opencv.hpp>

namespace ip101 {
namespace advanced {

/**
 * @brief 运动模糊参数结构体
 */
struct MotionBlurParams {
    int size;               // 模糊核大小
    double angle;           // 模糊方向角度 (0-360度)
    double strength;        // 模糊强度

    // 默认构造函数
    MotionBlurParams() :
        size(15),
        angle(45.0),
        strength(1.0)
    {}
};

/**
 * @brief 运动模糊特效算法
 * @param src 输入图像
 * @param dst 输出图像
 * @param params 运动模糊参数
 */
void motion_blur(const cv::Mat& src, cv::Mat& dst, const MotionBlurParams& params = MotionBlurParams());

/**
 * @brief 创建运动模糊核
 * @param size 核大小
 * @param angle 模糊方向角度
 * @return 运动模糊核
 */
cv::Mat create_motion_blur_kernel(int size, double angle);

/**
 * @brief 方向性运动模糊 - 沿特定方向的模糊效果
 * @param src 输入图像
 * @param dst 输出图像
 * @param size 模糊核大小
 * @param angle 模糊方向角度
 * @param strength 模糊强度
 */
void directional_motion_blur(const cv::Mat& src, cv::Mat& dst, int size, double angle, double strength);

/**
 * @brief 径向运动模糊 - 从中心点向外扩散的模糊效果
 * @param src 输入图像
 * @param dst 输出图像
 * @param strength 模糊强度 (0.0-1.0)
 * @param center 中心点 (默认为图像中心)
 */
void radial_motion_blur(const cv::Mat& src, cv::Mat& dst, double strength = 0.5,
                       const cv::Point2f& center = cv::Point2f(-1, -1));

/**
 * @brief 旋转运动模糊 - 围绕中心点旋转的模糊效果
 * @param src 输入图像
 * @param dst 输出图像
 * @param strength 模糊强度 (0.0-1.0)
 * @param center 中心点 (默认为图像中心)
 */
void rotational_motion_blur(const cv::Mat& src, cv::Mat& dst, double strength = 0.5,
                           const cv::Point2f& center = cv::Point2f(-1, -1));

/**
 * @brief 缩放运动模糊 - 模拟缩放效果的模糊
 * @param src 输入图像
 * @param dst 输出图像
 * @param strength 模糊强度 (0.0-1.0)
 * @param center 中心点 (默认为图像中心)
 */
void zoom_motion_blur(const cv::Mat& src, cv::Mat& dst, double strength = 0.5,
                     const cv::Point2f& center = cv::Point2f(-1, -1));

} // namespace advanced
} // namespace ip101

#endif // MOTION_BLUR_HPP