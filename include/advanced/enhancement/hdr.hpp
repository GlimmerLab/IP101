#ifndef HDR_HPP
#define HDR_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace ip101 {
namespace advanced {

// ================== 工具函数 ==================
/**
 * @brief 权重函数：像素值越接近中间灰度，权重越高。
 * @param pixel_value 像素值(0-255)
 * @return 权重值(0-1)
 */
float weight_function(uchar pixel_value);

// ================== 标准版接口 ==================
/**
 * @brief 计算相机响应曲线（标准版）
 * @param images 多张不同曝光的图像
 * @param exposure_times 曝光时间数组
 * @param lambda 平滑度参数
 * @param samples 采样点数量
 * @return 响应曲线
 */
cv::Mat calculate_camera_response(const std::vector<cv::Mat>& images,
                                 const std::vector<float>& exposure_times,
                                 float lambda = 10.0f,
                                 int samples = 100);

/**
 * @brief HDR融合（标准版）
 * @param images 多张不同曝光的图像
 * @param exposure_times 曝光时间数组
 * @param response_curve 响应曲线（可选）
 * @return HDR图像
 */
cv::Mat create_hdr(const std::vector<cv::Mat>& images,
                   const std::vector<float>& exposure_times,
                   const cv::Mat& response_curve = cv::Mat());

/**
 * @brief 全局色调映射（Reinhard，标准版）
 * @param hdr_image HDR图像
 * @param key 亮度参数
 * @param white_point 白点参数
 * @return LDR图像
 */
cv::Mat tone_mapping_global(const cv::Mat& hdr_image,
                           float key = 0.18f,
                           float white_point = 1.0f);

/**
 * @brief 局部色调映射（Durand，标准版）
 * @param hdr_image HDR图像
 * @param sigma 高斯滤波标准差
 * @param contrast 对比度参数
 * @return LDR图像
 */
cv::Mat tone_mapping_local(const cv::Mat& hdr_image,
                          float sigma = 2.0f,
                          float contrast = 4.0f);

// ================== 性能优化版接口 ==================
/**
 * @brief 计算相机响应曲线（优化版）
 */
cv::Mat calculate_camera_response_optimized(const std::vector<cv::Mat>& images,
                                            const std::vector<float>& exposure_times,
                                            float lambda = 10.0f,
                                            int samples = 100);
/**
 * @brief HDR融合（优化版）
 */
cv::Mat create_hdr_optimized(const std::vector<cv::Mat>& images,
                             const std::vector<float>& exposure_times,
                             const cv::Mat& response_curve = cv::Mat());
/**
 * @brief 全局色调映射（优化版）
 *        使用OpenMP和SIMD指令优化，优化内存访问模式。
 *        相比标准版本，主要优化点：
 *        1. 使用OpenMP并行化计算
 *        2. 使用SIMD指令优化向量运算
 *        3. 优化内存访问模式，减少缓存未命中
 *        4. 预计算常量，减少重复计算
 *
 * @param hdr_image HDR图像
 * @param key 亮度参数
 * @param white_point 白点参数
 * @return LDR图像
 */
cv::Mat tone_mapping_global_optimized(const cv::Mat& hdr_image,
                                    float key = 0.18f,
                                    float white_point = 1.0f);
/**
 * @brief 局部色调映射（优化版）
 *        使用OpenMP和SIMD指令优化，优化内存访问和计算模式。
 *        相比标准版本，主要优化点：
 *        1. 使用OpenMP并行化计算
 *        2. 使用SIMD指令优化向量运算
 *        3. 优化内存访问模式，减少缓存未命中
 *        4. 预分配内存，避免重复分配
 *        5. 分步计算，提高缓存命中率
 *
 * @param hdr_image HDR图像
 * @param sigma 高斯滤波标准差
 * @param contrast 对比度参数
 * @return LDR图像
 */
cv::Mat tone_mapping_local_optimized(const cv::Mat& hdr_image,
                                   float sigma = 2.0f,
                                   float contrast = 4.0f);

} // namespace advanced
} // namespace ip101

#endif // HDR_HPP