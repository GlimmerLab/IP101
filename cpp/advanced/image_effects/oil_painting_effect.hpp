#ifndef OIL_PAINTING_EFFECT_HPP
#define OIL_PAINTING_EFFECT_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace ip101 {
namespace advanced {

/**
 * @brief 油画效果参数结构体
 */
struct OilPaintingParams {
    int radius;              // 邻域半径
    int levels;              // 色彩强度级别
    int dynamic_ratio;       // 动态范围比例

    // 默认构造函数
    OilPaintingParams() :
        radius(3),
        levels(10),
        dynamic_ratio(15)
    {}
};

/**
 * @brief 油画效果算法
 * @param src 输入图像
 * @param dst 输出图像
 * @param params 油画效果参数
 */
void oil_painting_effect(const cv::Mat& src, cv::Mat& dst, const OilPaintingParams& params = OilPaintingParams());

/**
 * @brief 高级油画效果（带纹理增强）
 * @param src 输入图像
 * @param dst 输出图像
 * @param params 油画效果参数
 * @param texture_strength 笔刷纹理增强强度
 */
void enhanced_oil_painting_effect(const cv::Mat& src, cv::Mat& dst,
                                 const OilPaintingParams& params = OilPaintingParams(),
                                 double texture_strength = 0.5);

/**
 * @brief 生成笔刷纹理
 * @param size 纹理大小
 * @param brush_size 笔刷大小
 * @param brush_density 笔刷密度
 * @param angle 笔刷角度（0-360度）
 * @return 笔刷纹理图像
 */
cv::Mat generate_brush_texture(const cv::Size& size, int brush_size, int brush_density, float angle);

/**
 * @brief 实时油画效果算法（优化版本）
 * @param src 输入图像
 * @param dst 输出图像
 * @param params 油画效果参数
 */
void realtime_oil_painting_effect(const cv::Mat& src, cv::Mat& dst, const OilPaintingParams& params = OilPaintingParams());

} // namespace advanced
} // namespace ip101

#endif // OIL_PAINTING_EFFECT_HPP