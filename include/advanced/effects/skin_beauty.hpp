#ifndef SKIN_BEAUTY_HPP
#define SKIN_BEAUTY_HPP

#include <opencv2/opencv.hpp>

namespace ip101 {
namespace advanced {

/**
 * @brief 磨皮美白参数结构体
 */
struct SkinBeautyParams {
    double smoothing_factor;    // 磨皮强度，范围[0, 1]
    double whitening_factor;    // 美白强度，范围[0, 1]
    double detail_factor;       // 细节保留因子，范围[0, 1]
    int bilateral_size;         // 双边滤波窗口大小
    double bilateral_color;     // 双边滤波色彩空间标准差
    double bilateral_space;     // 双边滤波坐标空间标准差

    // 默认构造函数
    SkinBeautyParams() :
        smoothing_factor(0.5),
        whitening_factor(0.2),
        detail_factor(0.3),
        bilateral_size(9),
        bilateral_color(30.0),
        bilateral_space(7.0)
    {}
};

/**
 * @brief 磨皮美白算法
 * @param src 输入图像
 * @param dst 输出图像
 * @param params 磨皮美白参数
 */
void skin_beauty(const cv::Mat& src, cv::Mat& dst, const SkinBeautyParams& params = SkinBeautyParams());

/**
 * @brief 皮肤检测
 * @param src 输入图像
 * @param skin_mask 输出的皮肤区域掩码
 */
void detect_skin(const cv::Mat& src, cv::Mat& skin_mask);

/**
 * @brief 磨皮效果
 * @param src 输入图像
 * @param dst 输出图像
 * @param strength 磨皮强度
 * @param preserve_detail 是否保留细节
 */
void smooth_skin(const cv::Mat& src, cv::Mat& dst, double strength = 0.5, bool preserve_detail = true);

/**
 * @brief 美白效果
 * @param src 输入图像
 * @param dst 输出图像
 * @param skin_mask 皮肤区域掩码
 * @param strength 美白强度，范围[0, 1]
 */
void whiten_skin(const cv::Mat& src, cv::Mat& dst, const cv::Mat& skin_mask, double strength = 0.3);

/**
 * @brief 改善面部明暗对比
 * @param src 输入图像
 * @param dst 输出图像
 * @param strength 亮度调整强度，范围[0, 1]
 */
void improve_face_lighting(const cv::Mat& src, cv::Mat& dst, double strength = 0.4);

} // namespace advanced
} // namespace ip101

#endif // SKIN_BEAUTY_HPP