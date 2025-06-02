#include "rectangle_detection.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace ip101 {
namespace advanced {

double point_to_line_distance(const cv::Point2f& point, const cv::Point2f& line_start, const cv::Point2f& line_end) {
    // 如果起点和终点是同一个点，则计算点到点的距离
    if (line_start == line_end) {
        return cv::norm(point - line_start);
    }

    // 计算线段向量
    cv::Point2f line_vec = line_end - line_start;
    // 计算点到起点的向量
    cv::Point2f point_vec = point - line_start;

    // 计算线段长度的平方
    float line_len_squared = line_vec.x * line_vec.x + line_vec.y * line_vec.y;

    // 计算点在线段上的投影比例
    float projection = (point_vec.x * line_vec.x + point_vec.y * line_vec.y) / line_len_squared;

    // 处理点在线段外的情况
    if (projection < 0.0f) {
        return cv::norm(point - line_start);
    } else if (projection > 1.0f) {
        return cv::norm(point - line_end);
    }

    // 计算投影点
    cv::Point2f projection_point = line_start + projection * line_vec;

    // 返回点到投影点的距离
    return cv::norm(point - projection_point);
}

double calculate_rectangle_confidence(const std::vector<cv::Point>& contour, const std::vector<cv::Point>& corners) {
    if (corners.size() != 4 || contour.empty()) {
        return 0.0;
    }

    // 计算矩形周长
    double rectangle_perimeter = 0.0;
    for (int i = 0; i < 4; i++) {
        rectangle_perimeter += cv::norm(corners[i] - corners[(i + 1) % 4]);
    }

    // 计算轮廓到矩形边的平均距离
    double total_distance = 0.0;
    int num_points = 0;

    for (const auto& point : contour) {
        // 计算点到矩形四条边的最小距离
        double min_distance = std::numeric_limits<double>::max();
        for (int i = 0; i < 4; i++) {
            double dist = point_to_line_distance(point, corners[i], corners[(i + 1) % 4]);
            min_distance = std::min(min_distance, dist);
        }

        total_distance += min_distance;
        num_points++;
    }

    // 计算平均距离
    double avg_distance = total_distance / num_points;

    // 用平均距离与矩形周长的比例作为置信度的逆
    double inverse_confidence = avg_distance / (rectangle_perimeter / 4.0);

    // 将逆置信度转换为置信度，并限制在[0,1]范围内
    double confidence = 1.0 - std::min(1.0, inverse_confidence);

    return confidence;
}

void contour_based_rectangle_detection(const cv::Mat& src, std::vector<RectangleInfo>& rectangles,
                                     double min_area, double max_area,
                                     double min_aspect_ratio, double max_aspect_ratio,
                                     double min_confidence) {
    CV_Assert(!src.empty());

    // 清空输出向量
    rectangles.clear();

    // 转换为灰度图像
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // 使用自适应阈值进行二值化
    cv::Mat binary;
    cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                         cv::THRESH_BINARY_INV, 11, 2);

    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    // 处理每个轮廓
    for (const auto& contour : contours) {
        // 计算轮廓面积
        double area = cv::contourArea(contour);

        // 面积过滤
        if (area < min_area || area > max_area) {
            continue;
        }

        // 获取最小外接矩形
        cv::RotatedRect rotated_rect = cv::minAreaRect(contour);

        // 获取矩形的宽和高
        float width = rotated_rect.size.width;
        float height = rotated_rect.size.height;

        // 确保宽大于高
        if (width < height) {
            std::swap(width, height);
        }

        // 计算长宽比
        float aspect_ratio = width / height;

        // 长宽比过滤
        if (aspect_ratio < min_aspect_ratio || aspect_ratio > max_aspect_ratio) {
            continue;
        }

        // 获取四个角点
        cv::Point2f corners[4];
        rotated_rect.points(corners);

        std::vector<cv::Point> corner_points;
        for (int i = 0; i < 4; i++) {
            corner_points.push_back(cv::Point(static_cast<int>(corners[i].x), static_cast<int>(corners[i].y)));
        }

        // 计算置信度
        double confidence = calculate_rectangle_confidence(contour, corner_points);

        // 置信度过滤
        if (confidence < min_confidence) {
            continue;
        }

        // 创建矩形信息
        RectangleInfo rect_info;
        rect_info.center = rotated_rect.center;
        rect_info.size = rotated_rect.size;
        rect_info.angle = rotated_rect.angle;
        rect_info.confidence = confidence;
        rect_info.corners = corner_points;

        rectangles.push_back(rect_info);
    }
}

void hough_based_rectangle_detection(const cv::Mat& src, std::vector<RectangleInfo>& rectangles,
                                   double min_area, double max_area, double min_confidence) {
    CV_Assert(!src.empty());

    // 清空输出向量
    rectangles.clear();

    // 转换为灰度图像
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // 使用Canny进行边缘检测
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);

    // 霍夫变换检测直线
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI/180, 50, 50, 10);

    // 如果没有检测到足够的线，返回
    if (lines.size() < 4) {
        return;
    }

    // 按照方向对线段进行分组
    std::vector<cv::Vec4i> horizontal_lines;
    std::vector<cv::Vec4i> vertical_lines;

    for (const auto& line : lines) {
        int x1 = line[0];
        int y1 = line[1];
        int x2 = line[2];
        int y2 = line[3];

        double angle = std::abs(std::atan2(y2 - y1, x2 - x1) * 180.0 / CV_PI);

        // 判断线段方向
        if ((angle < 45.0 || angle > 135.0)) {
            horizontal_lines.push_back(line);
        } else {
            vertical_lines.push_back(line);
        }
    }

    // 如果水平线或垂直线数量不足，返回
    if (horizontal_lines.size() < 2 || vertical_lines.size() < 2) {
        return;
    }

    // 尝试组合水平线和垂直线形成矩形
    for (size_t i = 0; i < horizontal_lines.size(); i++) {
        for (size_t j = i + 1; j < horizontal_lines.size(); j++) {
            cv::Vec4i h1 = horizontal_lines[i];
            cv::Vec4i h2 = horizontal_lines[j];

            // 计算两条水平线的距离
            double h_dist = std::abs((h1[1] + h1[3]) / 2.0 - (h2[1] + h2[3]) / 2.0);

            // 如果两条水平线太近或太远，跳过
            if (h_dist < 20 || h_dist > 1000) {
                continue;
            }

            for (size_t k = 0; k < vertical_lines.size(); k++) {
                for (size_t l = k + 1; l < vertical_lines.size(); l++) {
                    cv::Vec4i v1 = vertical_lines[k];
                    cv::Vec4i v2 = vertical_lines[l];

                    // 计算两条垂直线的距离
                    double v_dist = std::abs((v1[0] + v1[2]) / 2.0 - (v2[0] + v2[2]) / 2.0);

                    // 如果两条垂直线太近或太远，跳过
                    if (v_dist < 20 || v_dist > 1000) {
                        continue;
                    }

                    // 计算矩形面积
                    double area = h_dist * v_dist;

                    // 面积过滤
                    if (area < min_area || area > max_area) {
                        continue;
                    }

                    // 计算矩形的四个角点
                    cv::Point2f corners[4];
                    bool valid_corners = true;

                    // 寻找两条水平线和两条垂直线的交点
                    // 上左角
                    if (!cv::solveSubpixelPosition(h1, v1, corners[0])) {
                        valid_corners = false;
                    }

                    // 上右角
                    if (!cv::solveSubpixelPosition(h1, v2, corners[1])) {
                        valid_corners = false;
                    }

                    // 下右角
                    if (!cv::solveSubpixelPosition(h2, v2, corners[2])) {
                        valid_corners = false;
                    }

                    // 下左角
                    if (!cv::solveSubpixelPosition(h2, v1, corners[3])) {
                        valid_corners = false;
                    }

                    if (!valid_corners) {
                        continue;
                    }

                    // 创建矩形轮廓
                    std::vector<cv::Point> rect_contour;
                    for (int c = 0; c < 4; c++) {
                        rect_contour.push_back(cv::Point(static_cast<int>(corners[c].x), static_cast<int>(corners[c].y)));
                    }

                    // 确保轮廓在图像范围内
                    bool outside_image = false;
                    for (const auto& pt : rect_contour) {
                        if (pt.x < 0 || pt.x >= src.cols || pt.y < 0 || pt.y >= src.rows) {
                            outside_image = true;
                            break;
                        }
                    }

                    if (outside_image) {
                        continue;
                    }

                    // 计算矩形的置信度
                    // 这里使用矩形周围的边缘强度作为置信度指标
                    cv::Mat rect_mask = cv::Mat::zeros(edges.size(), CV_8UC1);
                    std::vector<std::vector<cv::Point>> rect_contours = {rect_contour};
                    cv::drawContours(rect_mask, rect_contours, 0, cv::Scalar(255), 2);

                    cv::Mat edge_overlap;
                    cv::bitwise_and(edges, rect_mask, edge_overlap);

                    double edge_pixels = cv::countNonZero(edge_overlap);
                    double rect_perimeter = cv::arcLength(rect_contour, true);
                    double confidence = edge_pixels / rect_perimeter;

                    // 置信度过滤
                    if (confidence < min_confidence) {
                        continue;
                    }

                    // 创建矩形信息
                    RectangleInfo rect_info;

                    // 计算中心点
                    rect_info.center = cv::Point2f(0, 0);
                    for (int c = 0; c < 4; c++) {
                        rect_info.center.x += corners[c].x;
                        rect_info.center.y += corners[c].y;
                    }
                    rect_info.center.x /= 4;
                    rect_info.center.y /= 4;

                    // 计算矩形尺寸和角度
                    rect_info.size = cv::Size2f(v_dist, h_dist);
                    rect_info.angle = 0; // 这里假设矩形是轴对齐的
                    rect_info.confidence = confidence;

                    // 存储角点
                    rect_info.corners.resize(4);
                    for (int c = 0; c < 4; c++) {
                        rect_info.corners[c] = cv::Point(static_cast<int>(corners[c].x), static_cast<int>(corners[c].y));
                    }

                    rectangles.push_back(rect_info);
                }
            }
        }
    }
}

void detect_rectangles(const cv::Mat& src, std::vector<RectangleInfo>& rectangles,
                      double min_area, double max_area,
                      double min_aspect_ratio, double max_aspect_ratio,
                      double min_confidence) {
    // 使用基于轮廓的方法检测矩形
    std::vector<RectangleInfo> contour_rects;
    contour_based_rectangle_detection(src, contour_rects, min_area, max_area, min_aspect_ratio, max_aspect_ratio, min_confidence);

    // 使用基于霍夫变换的方法检测矩形
    std::vector<RectangleInfo> hough_rects;
    hough_based_rectangle_detection(src, hough_rects, min_area, max_area, min_confidence);

    // 合并结果
    rectangles = contour_rects;
    rectangles.insert(rectangles.end(), hough_rects.begin(), hough_rects.end());

    // 如果检测到了重叠的矩形，只保留置信度最高的
    if (rectangles.size() > 1) {
        // 按照置信度排序（降序）
        std::sort(rectangles.begin(), rectangles.end(), [](const RectangleInfo& a, const RectangleInfo& b) {
            return a.confidence > b.confidence;
        });

        std::vector<RectangleInfo> filtered_rects;
        std::vector<bool> is_removed(rectangles.size(), false);

        for (size_t i = 0; i < rectangles.size(); i++) {
            if (is_removed[i]) {
                continue;
            }

            filtered_rects.push_back(rectangles[i]);

            // 检查剩余矩形是否与当前矩形重叠
            for (size_t j = i + 1; j < rectangles.size(); j++) {
                if (is_removed[j]) {
                    continue;
                }

                // 计算两个矩形的交集面积
                cv::Rect rect_i = cv::boundingRect(rectangles[i].corners);
                cv::Rect rect_j = cv::boundingRect(rectangles[j].corners);
                cv::Rect intersection = rect_i & rect_j;

                // 如果交集不为空，计算IoU
                if (intersection.width > 0 && intersection.height > 0) {
                    float intersection_area = intersection.width * intersection.height;
                    float union_area = rect_i.width * rect_i.height + rect_j.width * rect_j.height - intersection_area;
                    float iou = intersection_area / union_area;

                    // 如果IoU大于阈值，则认为矩形重叠
                    if (iou > 0.2) {
                        is_removed[j] = true;
                    }
                }
            }
        }

        rectangles = filtered_rects;
    }
}

void draw_rectangles(cv::Mat& image, const std::vector<RectangleInfo>& rectangles,
                    const cv::Scalar& color, int thickness) {
    for (const auto& rect : rectangles) {
        // 绘制矩形边框
        for (int i = 0; i < 4; i++) {
            cv::line(image, rect.corners[i], rect.corners[(i + 1) % 4], color, thickness);
        }

        // 绘制置信度
        std::string confidence_str = cv::format("%.2f", rect.confidence);
        cv::putText(image, confidence_str, rect.center, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }
}

} // namespace advanced
} // namespace ip101