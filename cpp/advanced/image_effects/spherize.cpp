#include "spherize.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace ip101 {
namespace advanced {

void spherize(const cv::Mat& src, cv::Mat& dst, const SpherizeParams& params) {
    CV_Assert(!src.empty());
    CV_Assert(params.strength >= 0.0 && params.strength <= 1.0);
    CV_Assert(params.radius > 0.0 && params.radius <= 1.0);

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 确定球面中心点
    cv::Point2f center;
    if (params.center.x < 0 || params.center.y < 0) {
        center = cv::Point2f(src.cols / 2.0f, src.rows / 2.0f);
    } else {
        center = params.center;
    }

    // 计算最大半径（图像较短边的一半乘以radius参数）
    float max_radius = params.radius * std::min(center.x, std::min(center.y,
                                              std::min(src.cols - center.x, src.rows - center.y)));

    // 计算变换强度系数
    float strength_factor = params.strength * 2.0f;
    if (params.invert) {
        strength_factor = -strength_factor;
    }

    // 遍历每个像素并应用球面化变换
    #pragma omp parallel for
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            // 计算当前点到中心的距离
            float dx = x - center.x;
            float dy = y - center.y;
            float distance = std::sqrt(dx * dx + dy * dy);

            // 如果在影响范围内，应用球面化变换
            if (distance < max_radius) {
                // 归一化距离
                float normalized_dist = distance / max_radius;

                // 应用球面化公式计算新的归一化距离
                float new_dist;
                if (params.invert) {
                    // 凹陷效果（向内球面化）
                    new_dist = normalized_dist * (1.0f + strength_factor * (1.0f - normalized_dist));
                } else {
                    // 凸出效果（向外球面化）
                    new_dist = normalized_dist * (1.0f - strength_factor * (1.0f - normalized_dist));
                }

                // 计算缩放后的坐标
                float scale = (distance == 0) ? 1.0f : new_dist / normalized_dist;
                float src_x = center.x + dx * scale;
                float src_y = center.y + dy * scale;

                // 确保坐标在有效范围内
                if (src_x >= 0 && src_x < src.cols - 1 && src_y >= 0 && src_y < src.rows - 1) {
                    // 使用双线性插值获取颜色
                    dst.at<cv::Vec3b>(y, x) = interpolate_pixel(src, src_x, src_y);
                } else {
                    // 超出范围的点，填充黑色或原图
                    dst.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
                }
            } else {
                // 范围外的点，直接复制原图
                dst.at<cv::Vec3b>(y, x) = src.at<cv::Vec3b>(y, x);
            }
        }
    }
}

// 辅助函数：双线性插值
cv::Vec3b interpolate_pixel(const cv::Mat& img, float x, float y) {
    int x1 = static_cast<int>(x);
    int y1 = static_cast<int>(y);
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    // 计算权重
    float x_weight = x - x1;
    float y_weight = y - y1;

    // 获取四个相邻像素点
    cv::Vec3b p1 = img.at<cv::Vec3b>(y1, x1);
    cv::Vec3b p2 = img.at<cv::Vec3b>(y1, x2);
    cv::Vec3b p3 = img.at<cv::Vec3b>(y2, x1);
    cv::Vec3b p4 = img.at<cv::Vec3b>(y2, x2);

    // 双线性插值
    cv::Vec3b result;
    for (int i = 0; i < 3; i++) {
        float top = p1[i] * (1 - x_weight) + p2[i] * x_weight;
        float bottom = p3[i] * (1 - x_weight) + p4[i] * x_weight;
        result[i] = cv::saturate_cast<uchar>(top * (1 - y_weight) + bottom * y_weight);
    }

    return result;
}

void bulge_effect(const cv::Mat& src, cv::Mat& dst, double strength, const cv::Point2f& center) {
    // 创建参数结构体
    SpherizeParams params;
    params.strength = strength;
    params.radius = 0.8;  // 默认使用80%的较短边作为半径
    params.center = center;
    params.invert = false;  // 凸出效果

    // 调用通用球面化函数
    spherize(src, dst, params);
}

void pinch_effect(const cv::Mat& src, cv::Mat& dst, double strength, const cv::Point2f& center) {
    // 创建参数结构体
    SpherizeParams params;
    params.strength = strength;
    params.radius = 0.8;  // 默认使用80%的较短边作为半径
    params.center = center;
    params.invert = true;  // 凹陷效果

    // 调用通用球面化函数
    spherize(src, dst, params);
}

void fisheye_effect(const cv::Mat& src, cv::Mat& dst, double strength) {
    CV_Assert(!src.empty());
    CV_Assert(strength >= 0.0 && strength <= 1.0);

    // 创建输出图像
    dst.create(src.size(), src.type());
    dst = cv::Scalar(0, 0, 0);  // 初始化为黑色

    // 图像中心
    cv::Point2f center(src.cols / 2.0f, src.rows / 2.0f);

    // 半径为图像较短边的一半
    float radius = std::min(center.x, center.y);

    // 缩放因子，根据强度调整
    float scale_factor = 1.0f + strength;

    // 遍历输出图像的每个像素
    #pragma omp parallel for
    for (int y = 0; y < dst.rows; y++) {
        for (int x = 0; x < dst.cols; x++) {
            // 计算到中心的相对位置
            float nx = (x - center.x) / radius;
            float ny = (y - center.y) / radius;

            // 到中心的距离
            float r = std::sqrt(nx * nx + ny * ny);

            // 如果在单位圆内
            if (r < 1.0f) {
                // 鱼眼变换公式
                float theta = std::atan2(ny, nx);
                float new_r = std::pow(r, scale_factor);

                // 转换回原始坐标
                float src_x = center.x + radius * new_r * std::cos(theta);
                float src_y = center.y + radius * new_r * std::sin(theta);

                // 检查坐标是否在图像范围内
                if (src_x >= 0 && src_x < src.cols - 1 && src_y >= 0 && src_y < src.rows - 1) {
                    // 使用双线性插值获取颜色
                    dst.at<cv::Vec3b>(y, x) = interpolate_pixel(src, src_x, src_y);
                }
            }
        }
    }
}

} // namespace advanced
} // namespace ip101