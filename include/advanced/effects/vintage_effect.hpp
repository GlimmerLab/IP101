#ifndef VINTAGE_EFFECT_HPP
#define VINTAGE_EFFECT_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace ip101 {
namespace advanced {

/**
 * @brief 老照片特效参数结构体
 */
struct VintageParams {
    double sepia_intensity;      // 褐色调强度
    double noise_level;          // 噪点强度
    double vignette_strength;    // 暗角强度
    double scratch_count;        // 划痕数量
    double scratch_intensity;    // 划痕强度
    bool add_border;             // 是否添加边框

    // 默认构造函数
    VintageParams() :
        sepia_intensity(0.8),
        noise_level(15.0),
        vignette_strength(0.5),
        scratch_count(10),
        scratch_intensity(0.7),
        add_border(true)
    {}
};

/**
 * @brief 老照片特效算法
 * @param src 输入图像
 * @param dst 输出图像
 * @param params 老照片特效参数
 */
void vintage_effect(const cv::Mat& src, cv::Mat& dst, const VintageParams& params = VintageParams());

/**
 * @brief 应用褐色调特效
 * @param src 输入图像
 * @param dst 输出图像
 * @param intensity 褐色调强度
 */
void apply_sepia_tone(const cv::Mat& src, cv::Mat& dst, double intensity);

/**
 * @brief 添加老照片噪点
 * @param src 输入图像
 * @param dst 输出图像
 * @param noise_level 噪点强度
 */
void add_film_grain(const cv::Mat& src, cv::Mat& dst, double noise_level);

/**
 * @brief 添加暗角效果
 * @param src 输入图像
 * @param dst 输出图像
 * @param strength 暗角强度
 */
void add_vignette(const cv::Mat& src, cv::Mat& dst, double strength);

/**
 * @brief 添加划痕效果
 * @param src 输入图像
 * @param dst 输出图像
 * @param count 划痕数量
 * @param intensity 划痕强度
 */
void add_scratches(const cv::Mat& src, cv::Mat& dst, double count, double intensity);

/**
 * @brief 添加老式边框
 * @param src 输入图像
 * @param dst 输出图像
 */
void add_vintage_border(const cv::Mat& src, cv::Mat& dst);

} // namespace advanced
} // namespace ip101

#endif // VINTAGE_EFFECT_HPP