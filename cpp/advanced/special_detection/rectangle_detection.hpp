#ifndef RECTANGLE_DETECTION_HPP
#define RECTANGLE_DETECTION_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace ip101 {
namespace advanced {

/**
 * @brief 矩形检测结果结构体
 */
struct RectangleInfo {
    cv::Point2f center;     // 矩形中心
    cv::Size2f size;        // 矩形大小
    float angle;            // 矩形角度（度）
    double confidence;      // 置信度
    std::vector<cv::Point> corners; // 四个角点坐标
};

/**
 * @brief 矩形检测算法
 * @param src 输入图像
 * @param rectangles 检测到的矩形信息
 * @param min_area 最小矩形面积
 * @param max_area 最大矩形面积
 * @param min_aspect_ratio 最小长宽比
 * @param max_aspect_ratio 最大长宽比
 * @param min_confidence 最小置信度
 */
void detect_rectangles(const cv::Mat& src, std::vector<RectangleInfo>& rectangles,
                      double min_area = 1000.0, double max_area = 100000.0,
                      double min_aspect_ratio = 0.5, double max_aspect_ratio = 2.0,
                      double min_confidence = 0.8);

/**
 * @brief 基于轮廓的矩形检测
 * @param src 输入图像
 * @param rectangles 检测到的矩形信息
 * @param min_area 最小矩形面积
 * @param max_area 最大矩形面积
 * @param min_aspect_ratio 最小长宽比
 * @param max_aspect_ratio 最大长宽比
 * @param min_confidence 最小置信度
 */
void contour_based_rectangle_detection(const cv::Mat& src, std::vector<RectangleInfo>& rectangles,
                                     double min_area = 1000.0, double max_area = 100000.0,
                                     double min_aspect_ratio = 0.5, double max_aspect_ratio = 2.0,
                                     double min_confidence = 0.8);

/**
 * @brief 基于霍夫变换的矩形检测
 * @param src 输入图像
 * @param rectangles 检测到的矩形信息
 * @param min_area 最小矩形面积
 * @param max_area 最大矩形面积
 * @param min_confidence 最小置信度
 */
void hough_based_rectangle_detection(const cv::Mat& src, std::vector<RectangleInfo>& rectangles,
                                   double min_area = 1000.0, double max_area = 100000.0,
                                   double min_confidence = 0.8);

/**
 * @brief 绘制检测到的矩形
 * @param image 输入/输出图像
 * @param rectangles 检测到的矩形信息
 * @param color 矩形颜色
 * @param thickness 线宽
 */
void draw_rectangles(cv::Mat& image, const std::vector<RectangleInfo>& rectangles,
                    const cv::Scalar& color = cv::Scalar(0, 255, 0), int thickness = 2);

/**
 * @brief 计算点到线段的距离
 * @param point 点
 * @param line_start 线段起点
 * @param line_end 线段终点
 * @return 点到线段的距离
 */
double point_to_line_distance(const cv::Point2f& point, const cv::Point2f& line_start, const cv::Point2f& line_end);

/**
 * @brief 计算矩形的置信度
 * @param contour 轮廓点集
 * @param corners 矩形四个角点
 * @return 置信度
 */
double calculate_rectangle_confidence(const std::vector<cv::Point>& contour, const std::vector<cv::Point>& corners);

} // namespace advanced
} // namespace ip101

#endif // RECTANGLE_DETECTION_HPP