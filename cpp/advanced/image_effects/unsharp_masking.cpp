#include "unsharp_masking.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace ip101 {
namespace advanced {

void unsharp_masking(const cv::Mat& src, cv::Mat& dst, const UnsharpMaskingParams& params) {
    CV_Assert(!src.empty());
    CV_Assert(params.strength >= 0.0 && params.radius > 0.0);
    CV_Assert(params.threshold >= 0.0 && params.threshold <= 255.0);
    CV_Assert(params.edge_protect >= 0.0 && params.edge_protect <= 1.0);

    // 根据参数选择不同的钝化蒙版方法
    if (params.adaptive) {
        adaptive_unsharp_masking(src, dst, params.strength, params.radius, params.edge_protect);
    } else if (params.threshold > 0) {
        high_pass_unsharp_masking(src, dst, params.strength, params.radius, params.threshold);
    } else {
        basic_unsharp_masking(src, dst, params.strength, params.radius);
    }
}

void basic_unsharp_masking(const cv::Mat& src, cv::Mat& dst, double strength, double radius) {
    CV_Assert(!src.empty());
    CV_Assert(strength >= 0.0 && radius > 0.0);

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 转换为浮点型以便精确计算
    cv::Mat src_float;
    src.convertTo(src_float, CV_32FC3);

    // 应用高斯模糊创建低通滤波图像
    cv::Mat blurred;
    cv::GaussianBlur(src_float, blurred, cv::Size(0, 0), radius);

    // 计算高频成分（原图减去模糊图）
    cv::Mat highpass = src_float - blurred;

    // 执行钝化蒙版（原图加上高频成分的一部分）
    cv::Mat sharpened = src_float + strength * highpass;

    // 转换回8位格式并裁剪到有效范围
    sharpened.convertTo(dst, CV_8UC3);
}

void high_pass_unsharp_masking(const cv::Mat& src, cv::Mat& dst, double strength, double radius, double threshold) {
    CV_Assert(!src.empty());
    CV_Assert(strength >= 0.0 && radius > 0.0);
    CV_Assert(threshold >= 0.0 && threshold <= 255.0);

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 转换为浮点型以便精确计算
    cv::Mat src_float;
    src.convertTo(src_float, CV_32FC3);

    // 应用高斯模糊创建低通滤波图像
    cv::Mat blurred;
    cv::GaussianBlur(src_float, blurred, cv::Size(0, 0), radius);

    // 计算高频成分（原图减去模糊图）
    cv::Mat highpass = src_float - blurred;

    // 创建掩码，只在高频成分大于阈值的地方应用锐化
    cv::Mat mask = cv::Mat::zeros(src.size(), CV_32FC3);

    // 阈值处理高频成分
    #pragma omp parallel for
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            cv::Vec3f& hp = highpass.at<cv::Vec3f>(y, x);
            cv::Vec3f& m = mask.at<cv::Vec3f>(y, x);

            // 对每个通道应用阈值
            for (int c = 0; c < 3; c++) {
                float abs_val = std::abs(hp[c]);
                if (abs_val > threshold) {
                    // 阈值以上的差异完全保留
                    m[c] = hp[c];
                } else {
                    // 平滑过渡区域
                    float scale = std::pow(abs_val / threshold, 2.0);
                    m[c] = scale * hp[c];
                }
            }
        }
    }

    // 执行钝化蒙版（原图加上高频成分的一部分，根据掩码）
    cv::Mat sharpened = src_float + strength * mask;

    // 转换回8位格式并裁剪到有效范围
    sharpened.convertTo(dst, CV_8UC3);
}

void adaptive_unsharp_masking(const cv::Mat& src, cv::Mat& dst, double strength, double radius, double edge_protect) {
    CV_Assert(!src.empty());
    CV_Assert(strength >= 0.0 && radius > 0.0);
    CV_Assert(edge_protect >= 0.0 && edge_protect <= 1.0);

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 转换为浮点型以便精确计算
    cv::Mat src_float;
    src.convertTo(src_float, CV_32FC3);

    // 应用高斯模糊创建低通滤波图像
    cv::Mat blurred;
    cv::GaussianBlur(src_float, blurred, cv::Size(0, 0), radius);

    // 计算高频成分（原图减去模糊图）
    cv::Mat highpass = src_float - blurred;

    // 转换为灰度图用于边缘检测
    cv::Mat gray;
    if (src.channels() == 1) {
        src.convertTo(gray, CV_8U);
    } else {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    }

    // 计算边缘强度
    cv::Mat edges;
    cv::Sobel(gray, edges, CV_32F, 1, 1);
    cv::convertScaleAbs(edges, edges);

    // 创建边缘权重掩码（边缘区域权重较小，平滑区域权重较大）
    cv::Mat edge_mask;
    double max_val;
    cv::minMaxLoc(edges, nullptr, &max_val);
    edges.convertTo(edge_mask, CV_32F, 1.0 / max_val);

    // 应用非线性变换，使得强边缘区域锐化程度较低
    #pragma omp parallel for
    for (int y = 0; y < edge_mask.rows; y++) {
        for (int x = 0; x < edge_mask.cols; x++) {
            float& val = edge_mask.at<float>(y, x);
            val = 1.0f - edge_protect * std::pow(val, 0.5f);  // 反转权重，边缘处权重较小
        }
    }

    // 扩展边缘掩码为3通道（如果输入是彩色图像）
    cv::Mat weight_mask;
    if (src.channels() == 3) {
        std::vector<cv::Mat> channels = {edge_mask, edge_mask, edge_mask};
        cv::merge(channels, weight_mask);
    } else {
        weight_mask = edge_mask;
    }

    // 计算自适应锐化图像（原图加上带权重的高频成分）
    cv::Mat sharpened = src_float + strength * highpass.mul(weight_mask);

    // 转换回8位格式并裁剪到有效范围
    sharpened.convertTo(dst, CV_8UC3);
}

} // namespace advanced
} // namespace ip101