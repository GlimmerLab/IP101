#ifndef CONNECTED_COMPONENTS_HPP
#define CONNECTED_COMPONENTS_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace ip101 {

/**
 * @brief 连通域标记结构体
 */
struct ConnectedComponent {
    int label;              // 标记值
    int area;              // 面积
    cv::Point centroid;    // 质心
    cv::Rect bbox;         // 外接矩形
    double circularity;    // 圆形度
    double aspect_ratio;   // 长宽比
    double solidity;       // 实心度
};

/**
 * @brief 4连通域标记
 * @param src 输入二值图像
 * @param labels 输出标记图像
 * @return 连通域数量
 */
int label_4connected(const cv::Mat& src, cv::Mat& labels);

/**
 * @brief 8连通域标记
 * @param src 输入二值图像
 * @param labels 输出标记图像
 * @return 连通域数量
 */
int label_8connected(const cv::Mat& src, cv::Mat& labels);

/**
 * @brief 连通域统计
 * @param labels 标记图像
 * @param num_labels 连通域数量
 * @return 连通域属性列表
 */
std::vector<ConnectedComponent> analyze_components(const cv::Mat& labels, int num_labels);

/**
 * @brief 连通域过滤
 * @param labels 标记图像
 * @param stats 连通域属性
 * @param min_area 最小面积
 * @param max_area 最大面积
 * @return 过滤后的标记图像
 */
cv::Mat filter_components(const cv::Mat& labels,
                         const std::vector<ConnectedComponent>& stats,
                         int min_area = 100,
                         int max_area = 10000);

/**
 * @brief 连通域属性计算
 * @param src 原始图像
 * @param labels 标记图像
 * @param stats 连通域属性
 * @return 带标记的彩色图像
 */
cv::Mat draw_components(const cv::Mat& src,
                       const cv::Mat& labels,
                       const std::vector<ConnectedComponent>& stats);

} // namespace ip101

#endif // CONNECTED_COMPONENTS_HPP