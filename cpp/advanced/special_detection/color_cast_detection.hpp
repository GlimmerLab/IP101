#ifndef COLOR_CAST_DETECTION_HPP
#define COLOR_CAST_DETECTION_HPP

#include <opencv2/opencv.hpp>

namespace ip101 {
namespace advanced {

/**
 * @brief 偏色检测结果结构体
 */
struct ColorCastResult {
    bool has_color_cast;             // 是否存在偏色
    double color_cast_degree;        // 偏色程度，范围[0, 1]
    std::string dominant_color;      // 主要偏色方向（"red", "green", "blue", "yellow", "magenta", "cyan"）
    cv::Vec3f color_cast_vector;     // 偏色向量
    cv::Mat color_distribution_map;  // 颜色分布图

    // 默认构造函数
    ColorCastResult() :
        has_color_cast(false),
        color_cast_degree(0.0),
        dominant_color("none"),
        color_cast_vector(cv::Vec3f(0, 0, 0))
    {}
};

/**
 * @brief 偏色检测参数结构体
 */
struct ColorCastDetectionParams {
    double threshold;                // 偏色判断阈值，大于此值认为存在偏色，范围[0, 1]
    bool use_reference_white;        // 是否使用参考白点
    cv::Vec3f reference_white;       // 参考白点，默认为D65标准光源
    bool analyze_distribution;       // 是否分析颜色分布
    bool auto_white_balance_check;   // 是否进行自动白平衡检查

    // 默认构造函数
    ColorCastDetectionParams() :
        threshold(0.15),
        use_reference_white(false),
        reference_white(cv::Vec3f(0.95f, 1.0f, 1.05f)),  // D65光源
        analyze_distribution(true),
        auto_white_balance_check(true)
    {}
};

/**
 * @brief 偏色检测算法
 * @param src 输入图像
 * @param result 偏色检测结果
 * @param params 偏色检测参数
 */
void detect_color_cast(const cv::Mat& src, ColorCastResult& result,
                     const ColorCastDetectionParams& params = ColorCastDetectionParams());

/**
 * @brief 基于颜色直方图的偏色检测
 * @param src 输入图像
 * @param result 偏色检测结果
 * @param params 偏色检测参数
 */
void detect_color_cast_histogram(const cv::Mat& src, ColorCastResult& result,
                              const ColorCastDetectionParams& params);

/**
 * @brief 基于白平衡假设的偏色检测
 * @param src 输入图像
 * @param result 偏色检测结果
 * @param params 偏色检测参数
 */
void detect_color_cast_white_balance(const cv::Mat& src, ColorCastResult& result,
                                  const ColorCastDetectionParams& params);

/**
 * @brief 生成颜色分布图
 * @param src 输入图像
 * @param distribution_map 输出的颜色分布图
 */
void generate_color_distribution_map(const cv::Mat& src, cv::Mat& distribution_map);

/**
 * @brief 获取偏色的主要方向
 * @param color_vector 偏色向量
 * @return 返回主要偏色方向的名称（"red", "green", "blue", "yellow", "magenta", "cyan"）
 */
std::string get_dominant_color_direction(const cv::Vec3f& color_vector);

/**
 * @brief 在图像上可视化颜色偏移
 * @param src 输入图像
 * @param dst 输出图像
 * @param result 偏色检测结果
 */
void visualize_color_cast(const cv::Mat& src, cv::Mat& dst, const ColorCastResult& result);

} // namespace advanced
} // namespace ip101

#endif // COLOR_CAST_DETECTION_HPP