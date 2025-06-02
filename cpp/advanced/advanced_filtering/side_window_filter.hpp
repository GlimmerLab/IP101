#ifndef SIDE_WINDOW_FILTER_HPP
#define SIDE_WINDOW_FILTER_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace ip101 {
namespace advanced {

/**
 * @brief 侧窗口滤波器类型枚举
 */
enum class SideWindowType {
    BOX,        // 均值滤波
    MEDIAN      // 中值滤波
};

/**
 * @brief 方向枚举
 */
enum class Direction {
    N,  // 北
    NE, // 东北
    E,  // 东
    SE, // 东南
    S,  // 南
    SW, // 西南
    W,  // 西
    NW  // 西北
};

/**
 * @brief Box侧窗口滤波
 * @param src 输入图像
 * @param dst 输出图像
 * @param window_size 窗口大小
 */
void box_side_window_filter(const cv::Mat& src, cv::Mat& dst, int window_size = 5);

/**
 * @brief Median侧窗口滤波
 * @param src 输入图像
 * @param dst 输出图像
 * @param window_size 窗口大小
 */
void median_side_window_filter(const cv::Mat& src, cv::Mat& dst, int window_size = 5);

/**
 * @brief 侧窗口滤波通用函数
 * @param src 输入图像
 * @param dst 输出图像
 * @param window_size 窗口大小
 * @param filter_type 滤波器类型
 */
void side_window_filter(const cv::Mat& src, cv::Mat& dst, int window_size = 5,
                       SideWindowType filter_type = SideWindowType::BOX);

/**
 * @brief 计算各个方向上的梯度幅值
 * @param src 输入图像
 * @param gradients 输出梯度图像数组
 */
void compute_directional_gradients(const cv::Mat& src, std::vector<cv::Mat>& gradients);

/**
 * @brief 根据方向梯度确定最优窗口
 * @param gradients 梯度图像数组
 * @param optimal_dir 最优方向图像
 */
void determine_optimal_window(const std::vector<cv::Mat>& gradients, cv::Mat& optimal_dir);

/**
 * @brief 应用最优窗口滤波
 * @param src 输入图像
 * @param dst 输出图像
 * @param optimal_dir 最优方向图像
 * @param window_size 窗口大小
 * @param filter_type 滤波器类型
 */
void apply_optimal_window_filter(const cv::Mat& src, cv::Mat& dst, const cv::Mat& optimal_dir,
                               int window_size, SideWindowType filter_type);

} // namespace advanced
} // namespace ip101

#endif // SIDE_WINDOW_FILTER_HPP