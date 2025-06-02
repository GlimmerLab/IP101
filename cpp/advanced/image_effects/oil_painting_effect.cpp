#include "oil_painting_effect.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <omp.h>

namespace ip101 {
namespace advanced {

void oil_painting_effect(const cv::Mat& src, cv::Mat& dst, const OilPaintingParams& params) {
    CV_Assert(!src.empty());
    CV_Assert(params.radius > 0 && params.levels > 0 && params.dynamic_ratio > 0);

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 获取图像宽高和通道数
    int width = src.cols;
    int height = src.rows;
    int channels = src.channels();

    // 油画效果的半径
    int radius = params.radius;

    // 计算颜色强度的级别数量
    int intensity_levels = params.levels;

    // 遍历图像中的每个像素
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // 计算当前像素位置的邻域
            int xmin = std::max(0, x - radius);
            int ymin = std::max(0, y - radius);
            int xmax = std::min(width - 1, x + radius);
            int ymax = std::min(height - 1, y + radius);

            // 统计每个强度级别的累计颜色和计数
            std::vector<cv::Vec3i> intensity_counts(intensity_levels, cv::Vec3i(0, 0, 0));
            std::vector<int> intensity_nums(intensity_levels, 0);

            // 遍历邻域内的每个像素
            for (int ny = ymin; ny <= ymax; ny++) {
                for (int nx = xmin; nx <= xmax; nx++) {
                    // 获取邻域像素
                    cv::Vec3b pixel = src.at<cv::Vec3b>(ny, nx);

                    // 计算灰度强度值并归一化到[0, intensity_levels-1]范围
                    float intensity = (pixel[0] + pixel[1] + pixel[2]) / 3.0f;
                    int level = std::min(intensity_levels - 1, static_cast<int>(intensity * intensity_levels / 255.0f));

                    // 更新对应强度级别的累计颜色和计数
                    intensity_counts[level][0] += pixel[0];
                    intensity_counts[level][1] += pixel[1];
                    intensity_counts[level][2] += pixel[2];
                    intensity_nums[level]++;
                }
            }

            // 找到数量最多的强度级别
            int max_count = 0;
            int max_index = 0;

            for (int i = 0; i < intensity_levels; i++) {
                if (intensity_nums[i] > max_count) {
                    max_count = intensity_nums[i];
                    max_index = i;
                }
            }

            // 如果找到了有效的强度级别，计算该级别的平均颜色
            if (max_count > 0) {
                cv::Vec3b& out_pixel = dst.at<cv::Vec3b>(y, x);

                for (int c = 0; c < channels; c++) {
                    out_pixel[c] = cv::saturate_cast<uchar>(intensity_counts[max_index][c] / max_count);
                }
            }
        }
    }
}

cv::Mat generate_brush_texture(const cv::Size& size, int brush_size, int brush_density, float angle) {
    CV_Assert(size.width > 0 && size.height > 0);
    CV_Assert(brush_size > 0 && brush_density > 0);

    // 创建空白纹理图像
    cv::Mat texture = cv::Mat::zeros(size, CV_8UC1);

    // 随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist_x(0, size.width);
    std::uniform_real_distribution<float> dist_y(0, size.height);
    std::uniform_real_distribution<float> dist_len(brush_size * 0.5f, brush_size * 1.5f);
    std::uniform_real_distribution<float> dist_width(1, brush_size * 0.3f);
    std::uniform_real_distribution<float> dist_alpha(30, 60);

    // 角度转换为弧度
    float radian = angle * CV_PI / 180.0f;

    // 生成笔刷笔触
    for (int i = 0; i < brush_density; i++) {
        // 随机生成笔触起点
        cv::Point2f pt1(dist_x(gen), dist_y(gen));

        // 随机生成笔触长度和宽度
        float length = dist_len(gen);
        float width = dist_width(gen);
        float alpha = dist_alpha(gen);

        // 计算笔触终点（考虑角度）
        cv::Point2f pt2(pt1.x + length * std::cos(radian), pt1.y + length * std::sin(radian));

        // 绘制笔触线条
        cv::line(texture, pt1, pt2, cv::Scalar(alpha), width, cv::LINE_AA);
    }

    // 高斯模糊使笔触更柔和
    cv::GaussianBlur(texture, texture, cv::Size(5, 5), 0);

    return texture;
}

void enhanced_oil_painting_effect(const cv::Mat& src, cv::Mat& dst, const OilPaintingParams& params, double texture_strength) {
    CV_Assert(!src.empty());
    CV_Assert(texture_strength >= 0.0 && texture_strength <= 1.0);

    // 应用基本油画效果
    cv::Mat oil_painted;
    oil_painting_effect(src, oil_painted, params);

    if (texture_strength <= 0.0) {
        // 纹理强度为0，直接返回基本油画效果
        oil_painted.copyTo(dst);
        return;
    }

    // 生成笔刷纹理
    cv::Mat texture = generate_brush_texture(src.size(), params.radius * 3, 500, 45.0f);

    // 转换纹理为浮点型，并归一化到[0,1]范围
    cv::Mat texture_float;
    texture.convertTo(texture_float, CV_32F, 1.0 / 255.0);

    // 将油画图像转换为浮点型
    cv::Mat oil_float;
    oil_painted.convertTo(oil_float, CV_32FC3, 1.0 / 255.0);

    // 划分通道
    std::vector<cv::Mat> channels;
    cv::split(oil_float, channels);

    // 为每个通道应用纹理
    for (int c = 0; c < 3; c++) {
        // 应用纹理：texture_strength控制纹理强度
        channels[c] = channels[c].mul(1.0 - texture_strength + texture_strength * texture_float);
    }

    // 合并通道
    cv::Mat result;
    cv::merge(channels, result);

    // 转换回8位图像
    result.convertTo(dst, CV_8UC3, 255.0);
}

void realtime_oil_painting_effect(const cv::Mat& src, cv::Mat& dst, const OilPaintingParams& params) {
    CV_Assert(!src.empty());
    CV_Assert(params.radius > 0 && params.levels > 0 && params.dynamic_ratio > 0);

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 缩小图像尺寸进行处理，提高速度
    cv::Size small_size(src.cols / 2, src.rows / 2);
    cv::Mat small_src, small_dst;
    cv::resize(src, small_src, small_size, 0, 0, cv::INTER_LINEAR);

    // 在缩小的图像上应用油画效果
    oil_painting_effect(small_src, small_dst, params);

    // 将结果放大回原始尺寸
    cv::resize(small_dst, dst, src.size(), 0, 0, cv::INTER_LINEAR);

    // 使用锐化滤波器增强细节
    cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
                     0, -1, 0,
                     -1, 5, -1,
                     0, -1, 0);
    cv::filter2D(dst, dst, -1, kernel);
}

} // namespace advanced
} // namespace ip101