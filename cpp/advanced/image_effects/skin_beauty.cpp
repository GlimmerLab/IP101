#include "skin_beauty.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace ip101 {
namespace advanced {

void skin_beauty(const cv::Mat& src, cv::Mat& dst, const SkinBeautyParams& params) {
    CV_Assert(!src.empty());
    CV_Assert(params.smoothing_factor >= 0.0 && params.smoothing_factor <= 1.0);
    CV_Assert(params.whitening_factor >= 0.0 && params.whitening_factor <= 1.0);
    CV_Assert(params.detail_factor >= 0.0 && params.detail_factor <= 1.0);

    // 复制原图
    dst = src.clone();

    // 1. 肤色检测
    cv::Mat skin_mask;
    detect_skin(src, skin_mask);

    // 2. 磨皮处理
    cv::Mat smoothed;
    smooth_skin(src, smoothed, params.smoothing_factor, params.detail_factor > 0);

    // 3. 保留细节
    if (params.detail_factor > 0) {
        // 提取高频细节
        cv::Mat detail;
        cv::Mat blurred;
        cv::GaussianBlur(src, blurred, cv::Size(0, 0), 3.0);
        cv::addWeighted(src, 1.0 + params.detail_factor, blurred, -params.detail_factor, 0, detail);

        // 计算最终磨皮结果: 平滑后的图像加上一定比例的细节
        cv::addWeighted(smoothed, 1.0, detail - smoothed, params.detail_factor, 0, smoothed);
    }

    // 4. 根据皮肤掩码混合原图和磨皮后的图像
    smoothed.copyTo(dst, skin_mask);

    // 5. 美白处理
    if (params.whitening_factor > 0) {
        whiten_skin(dst, dst, skin_mask, params.whitening_factor);
    }

    // 6. 改善面部明暗对比
    improve_face_lighting(dst, dst, 0.3 * params.whitening_factor);
}

void detect_skin(const cv::Mat& src, cv::Mat& skin_mask) {
    CV_Assert(!src.empty());

    // 创建输出掩码
    skin_mask = cv::Mat::zeros(src.size(), CV_8UC1);

    // 转换到YCrCb颜色空间
    cv::Mat ycrcb;
    cv::cvtColor(src, ycrcb, cv::COLOR_BGR2YCrCb);

    // 分离通道
    std::vector<cv::Mat> channels;
    cv::split(ycrcb, channels);

    // 获取Cr和Cb通道
    cv::Mat& cr = channels[1];
    cv::Mat& cb = channels[2];

    // 在YCrCb颜色空间中，肤色像素通常满足以下条件:
    // 133 <= Cr <= 173 and 77 <= Cb <= 127
    #pragma omp parallel for
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            uchar cr_val = cr.at<uchar>(y, x);
            uchar cb_val = cb.at<uchar>(y, x);

            // 使用YCrCb空间的肤色检测规则
            if (cr_val >= 133 && cr_val <= 173 && cb_val >= 77 && cb_val <= 127) {
                skin_mask.at<uchar>(y, x) = 255;
            }
        }
    }

    // 应用形态学操作改善掩码质量
    int morph_size = 3;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                              cv::Size(2 * morph_size + 1, 2 * morph_size + 1),
                                              cv::Point(morph_size, morph_size));

    // 闭运算（先膨胀后腐蚀）填充小洞
    cv::morphologyEx(skin_mask, skin_mask, cv::MORPH_CLOSE, element);

    // 开运算（先腐蚀后膨胀）去除小噪点
    cv::morphologyEx(skin_mask, skin_mask, cv::MORPH_OPEN, element);

    // 高斯模糊平滑边缘
    cv::GaussianBlur(skin_mask, skin_mask, cv::Size(5, 5), 0);
}

void smooth_skin(const cv::Mat& src, cv::Mat& dst, double strength, bool preserve_detail) {
    CV_Assert(!src.empty());
    CV_Assert(strength >= 0.0 && strength <= 1.0);

    // 如果强度为0，直接返回原图
    if (strength <= 0.0) {
        dst = src.clone();
        return;
    }

    // 创建输出图像
    dst = src.clone();

    // 自适应计算滤波参数，根据强度调整
    int d = 7 + static_cast<int>(strength * 10);           // 窗口大小
    double sigma_color = 10.0 + strength * 30.0;          // 颜色相似性标准差
    double sigma_space = 5.0 + strength * 5.0;            // 空间相似性标准差

    // 双边滤波进行平滑，保留边缘
    cv::Mat bilateral;
    cv::bilateralFilter(src, bilateral, d, sigma_color, sigma_space);

    // 高斯滤波进行进一步平滑
    cv::Mat gaussian;
    cv::GaussianBlur(bilateral, gaussian, cv::Size(5, 5), 2.0);

    // 根据强度混合原图、双边滤波和高斯滤波的结果
    double alpha = strength;
    cv::addWeighted(bilateral, alpha, gaussian, 1.0 - alpha, 0, dst);

    // 如果需要保留细节
    if (preserve_detail) {
        // 使用高通滤波提取细节
        cv::Mat high_freq;
        cv::Mat low_freq;
        cv::GaussianBlur(src, low_freq, cv::Size(0, 0), 3.0);
        high_freq = src - low_freq + cv::Scalar(128, 128, 128);  // 128为中性灰色

        // 高频细节的保留强度与磨皮强度成反比
        double detail_strength = 0.3 * (1.0 - strength);
        cv::addWeighted(dst, 1.0, high_freq - cv::Scalar(128, 128, 128), detail_strength, 0, dst);
    }
}

void whiten_skin(const cv::Mat& src, cv::Mat& dst, const cv::Mat& skin_mask, double strength) {
    CV_Assert(!src.empty());
    CV_Assert(!skin_mask.empty() && skin_mask.size() == src.size());
    CV_Assert(strength >= 0.0 && strength <= 1.0);

    // 复制原图
    dst = src.clone();

    // 如果强度为0，直接返回
    if (strength <= 0.0) {
        return;
    }

    // 转换到LAB颜色空间
    cv::Mat lab;
    cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);

    // 分离通道
    std::vector<cv::Mat> channels;
    cv::split(lab, channels);

    // 只调整亮度通道（L通道）
    cv::Mat& l_channel = channels[0];

    // 非线性调整亮度（美白）
    #pragma omp parallel for
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            // 如果是皮肤区域
            if (skin_mask.at<uchar>(y, x) > 128) {
                // 获取原亮度值
                uchar l_val = l_channel.at<uchar>(y, x);

                // 根据原亮度值计算调整系数（较暗区域调整更多）
                double adjust_factor = strength * (1.0 - l_val / 255.0);

                // 非线性增亮，使中等亮度区域增亮更多
                double curve_factor = std::sin((l_val / 255.0) * CV_PI) * 1.5;
                double adjusted_val = l_val + adjust_factor * curve_factor * 50.0;

                // 确保值在有效范围内
                l_channel.at<uchar>(y, x) = cv::saturate_cast<uchar>(adjusted_val);
            }
        }
    }

    // 合并通道
    cv::merge(channels, lab);

    // 转换回BGR颜色空间
    cv::cvtColor(lab, dst, cv::COLOR_Lab2BGR);
}

void improve_face_lighting(const cv::Mat& src, cv::Mat& dst, double strength) {
    CV_Assert(!src.empty());
    CV_Assert(strength >= 0.0 && strength <= 1.0);

    // 如果强度为0，直接返回
    if (strength <= 0.0) {
        dst = src.clone();
        return;
    }

    // 创建输出图像
    dst = src.clone();

    // 转换到YUV颜色空间（只调整Y通道）
    cv::Mat yuv;
    cv::cvtColor(src, yuv, cv::COLOR_BGR2YUV);

    // 分离通道
    std::vector<cv::Mat> channels;
    cv::split(yuv, channels);

    // 获取Y通道（亮度）
    cv::Mat& y_channel = channels[0];

    // 计算亮度直方图
    cv::Mat hist;
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };
    cv::calcHist(&y_channel, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);

    // 使用CLAHE进行自适应直方图均衡化
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(2.0 + strength * 2.0);  // 根据强度调整限制阈值
    clahe->setTilesGridSize(cv::Size(8, 8));    // 设置网格大小

    // 应用CLAHE到亮度通道
    cv::Mat enhanced_y;
    clahe->apply(y_channel, enhanced_y);

    // 混合原始亮度和增强后的亮度
    cv::addWeighted(y_channel, 1.0 - strength, enhanced_y, strength, 0, channels[0]);

    // 合并通道
    cv::merge(channels, yuv);

    // 转换回BGR颜色空间
    cv::cvtColor(yuv, dst, cv::COLOR_YUV2BGR);
}

} // namespace advanced
} // namespace ip101