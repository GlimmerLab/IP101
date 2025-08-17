#include <advanced/effects/motion_blur.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <omp.h>

namespace ip101 {
namespace advanced {

cv::Mat create_motion_blur_kernel(int size, double angle) {
    CV_Assert(size > 0);

    // 确保核大小为奇数
    if (size % 2 == 0) {
        size++;
    }

    cv::Mat kernel = cv::Mat::zeros(size, size, CV_32F);

    // 将角度转换为弧度
    double radian_angle = angle * CV_PI / 180.0;

    // 计算中心点
    cv::Point2f center(size / 2, size / 2);

    // 计算方向向量
    float dx = std::cos(radian_angle);
    float dy = std::sin(radian_angle);

    // 生成核
    float norm_factor = 0.0f;

    for (int i = -size/2; i <= size/2; i++) {
        int x = std::round(center.x + i * dx);
        int y = std::round(center.y + i * dy);

        if (x >= 0 && x < size && y >= 0 && y < size) {
            kernel.at<float>(y, x) = 1.0f;
            norm_factor += 1.0f;
        }
    }

    // 归一化核确保权重和为1
    if (norm_factor > 0) {
        kernel /= norm_factor;
    }

    return kernel;
}

void directional_motion_blur(const cv::Mat& src, cv::Mat& dst, int size, double angle, double strength) {
    CV_Assert(!src.empty());
    CV_Assert(size > 0);
    CV_Assert(strength > 0.0);

    // 创建运动模糊核
    cv::Mat kernel = create_motion_blur_kernel(size, angle);

    // 应用运动模糊滤波
    cv::filter2D(src, dst, -1, kernel);

    // 如果强度不为1.0，则混合原图和模糊图像
    if (std::abs(strength - 1.0) > 1e-6) {
        cv::addWeighted(src, 1.0 - strength, dst, strength, 0.0, dst);
    }
}

void radial_motion_blur(const cv::Mat& src, cv::Mat& dst, double strength, const cv::Point2f& center) {
    CV_Assert(!src.empty());
    CV_Assert(strength >= 0.0 && strength <= 1.0);

    // 确定中心点
    cv::Point2f blur_center;
    if (center.x < 0 || center.y < 0) {
        blur_center = cv::Point2f(src.cols / 2.0f, src.rows / 2.0f);
    } else {
        blur_center = center;
    }

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 径向模糊采样数
    const int num_samples = 15;

    // 计算步长
    float step = strength / (num_samples - 1);

    // 初始化输出图像为0
    dst = cv::Mat::zeros(src.size(), src.type());

    // 累积权重
    float total_weight = 0.0f;

    // 应用径向模糊
    #pragma omp parallel for
    for (int i = 0; i < num_samples; i++) {
        // 计算缩放系数
        float scale = 1.0f - step * i;
        float weight = 1.0f / num_samples;
        total_weight += weight;

        // 创建仿射变换矩阵
        cv::Mat M = cv::Mat::eye(2, 3, CV_32F);
        M.at<float>(0, 0) = scale;
        M.at<float>(1, 1) = scale;
        M.at<float>(0, 2) = blur_center.x * (1.0f - scale);
        M.at<float>(1, 2) = blur_center.y * (1.0f - scale);

        // 应用变换
        cv::Mat temp;
        cv::warpAffine(src, temp, M, src.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        // 累积到输出图像
        #pragma omp critical
        {
            dst += weight * temp;
        }
    }

    // 确保权重和为1
    if (std::abs(total_weight - 1.0f) > 1e-6) {
        dst /= total_weight;
    }
}

void rotational_motion_blur(const cv::Mat& src, cv::Mat& dst, double strength, const cv::Point2f& center) {
    CV_Assert(!src.empty());
    CV_Assert(strength >= 0.0 && strength <= 1.0);

    // 确定中心点
    cv::Point2f rotation_center;
    if (center.x < 0 || center.y < 0) {
        rotation_center = cv::Point2f(src.cols / 2.0f, src.rows / 2.0f);
    } else {
        rotation_center = center;
    }

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 旋转模糊采样数
    const int num_samples = 15;

    // 计算角度步长（最大旋转角度为30度）
    float max_angle = 30.0f * strength;
    float angle_step = max_angle / (num_samples - 1);

    // 初始化输出图像为0
    dst = cv::Mat::zeros(src.size(), src.type());

    // 累积权重
    float total_weight = 0.0f;

    // 应用旋转模糊
    #pragma omp parallel for
    for (int i = 0; i < num_samples; i++) {
        // 计算旋转角度
        float angle = -max_angle / 2.0f + i * angle_step;
        float weight = 1.0f / num_samples;
        total_weight += weight;

        // 创建旋转矩阵
        cv::Mat M = cv::getRotationMatrix2D(rotation_center, angle, 1.0);

        // 应用变换
        cv::Mat temp;
        cv::warpAffine(src, temp, M, src.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        // 累积到输出图像
        #pragma omp critical
        {
            dst += weight * temp;
        }
    }

    // 确保权重和为1
    if (std::abs(total_weight - 1.0f) > 1e-6) {
        dst /= total_weight;
    }
}

void zoom_motion_blur(const cv::Mat& src, cv::Mat& dst, double strength, const cv::Point2f& center) {
    CV_Assert(!src.empty());
    CV_Assert(strength >= 0.0 && strength <= 1.0);

    // 确定中心点
    cv::Point2f zoom_center;
    if (center.x < 0 || center.y < 0) {
        zoom_center = cv::Point2f(src.cols / 2.0f, src.rows / 2.0f);
    } else {
        zoom_center = center;
    }

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 缩放模糊采样数
    const int num_samples = 15;

    // 计算缩放步长
    float max_scale_delta = 0.4f * strength;
    float scale_step = max_scale_delta / (num_samples - 1);

    // 初始化输出图像为0
    dst = cv::Mat::zeros(src.size(), src.type());

    // 累积权重
    float total_weight = 0.0f;

    // 应用缩放模糊
    #pragma omp parallel for
    for (int i = 0; i < num_samples; i++) {
        // 计算缩放系数
        float scale = 1.0f - max_scale_delta / 2.0f + i * scale_step;
        float weight = 1.0f / num_samples;
        total_weight += weight;

        // 创建仿射变换矩阵
        cv::Mat M = cv::Mat::eye(2, 3, CV_32F);
        M.at<float>(0, 0) = scale;
        M.at<float>(1, 1) = scale;
        M.at<float>(0, 2) = zoom_center.x * (1.0f - scale);
        M.at<float>(1, 2) = zoom_center.y * (1.0f - scale);

        // 应用变换
        cv::Mat temp;
        cv::warpAffine(src, temp, M, src.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);

        // 累积到输出图像
        #pragma omp critical
        {
            dst += weight * temp;
        }
    }

    // 确保权重和为1
    if (std::abs(total_weight - 1.0f) > 1e-6) {
        dst /= total_weight;
    }
}

void motion_blur(const cv::Mat& src, cv::Mat& dst, const MotionBlurParams& params) {
    CV_Assert(!src.empty());

    // 应用方向性运动模糊
    directional_motion_blur(src, dst, params.size, params.angle, params.strength);
}

} // namespace advanced
} // namespace ip101