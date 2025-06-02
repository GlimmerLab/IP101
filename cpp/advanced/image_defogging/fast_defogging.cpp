#include "fast_defogging.hpp"
#include "dark_channel.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace ip101 {
namespace advanced {

void fast_max_contrast_defogging(const cv::Mat& src, cv::Mat& dst, int r, double lambda, double t0) {
    CV_Assert(!src.empty() && src.channels() == 3);
    CV_Assert(r > 0 && lambda > 0 && t0 > 0 && t0 < 1.0);

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 1. 快速计算局部最大对比度
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_64FC1, 1.0/255.0);

    // 计算梯度幅值
    cv::Mat grad_x, grad_y, gradient;
    cv::Sobel(gray, grad_x, CV_64F, 1, 0);
    cv::Sobel(gray, grad_y, CV_64F, 0, 1);
    cv::magnitude(grad_x, grad_y, gradient);

    // 计算局部最大梯度 - 使用滑动窗口的方法
    cv::Mat max_grad = cv::Mat::zeros(gray.size(), CV_64FC1);

    for (int y = 0; y < gray.rows; y++) {
        for (int x = 0; x < gray.cols; x++) {
            // 计算局部窗口范围
            int start_y = std::max(0, y - r);
            int end_y = std::min(gray.rows - 1, y + r);
            int start_x = std::max(0, x - r);
            int end_x = std::min(gray.cols - 1, x + r);

            double max_val = 0;
            for (int i = start_y; i <= end_y; i++) {
                for (int j = start_x; j <= end_x; j++) {
                    max_val = std::max(max_val, gradient.at<double>(i, j));
                }
            }

            max_grad.at<double>(y, x) = max_val;
        }
    }

    // 2. 估计粗透射率
    cv::Mat transmission = 1.0 - lambda * max_grad;

    // 3. 使用中值滤波平滑透射率
    cv::Mat refined_transmission;
    transmission.convertTo(refined_transmission, CV_8UC1, 255.0);
    cv::medianBlur(refined_transmission, refined_transmission, 5); // 使用5x5中值滤波
    refined_transmission.convertTo(refined_transmission, CV_64FC1, 1.0/255.0);

    // 4. 限制最小透射率
    cv::max(refined_transmission, t0, refined_transmission);

    // 5. 估计大气光照值
    cv::Vec3d A;

    // 找到亮度最高的前0.1%的像素
    int num_pixels = static_cast<int>(gray.total() * 0.001);
    num_pixels = std::max(1, num_pixels);

    // 创建亮度与位置的对应关系
    std::vector<std::pair<double, cv::Point>> intensity_pos;
    intensity_pos.reserve(gray.total());

    for (int y = 0; y < gray.rows; y++) {
        for (int x = 0; x < gray.cols; x++) {
            intensity_pos.push_back(std::make_pair(gray.at<double>(y, x), cv::Point(x, y)));
        }
    }

    // 按亮度降序排序
    std::partial_sort(intensity_pos.begin(), intensity_pos.begin() + num_pixels, intensity_pos.end(),
                     [](const std::pair<double, cv::Point>& a, const std::pair<double, cv::Point>& b) {
                         return a.first > b.first;
                     });

    // 计算亮度最高的像素的平均颜色作为大气光照
    double sum_b = 0, sum_g = 0, sum_r = 0;

    for (int i = 0; i < num_pixels; i++) {
        cv::Point p = intensity_pos[i].second;
        cv::Vec3b pixel = src.at<cv::Vec3b>(p);

        sum_b += pixel[0];
        sum_g += pixel[1];
        sum_r += pixel[2];
    }

    A = cv::Vec3d(sum_b / num_pixels, sum_g / num_pixels, sum_r / num_pixels);

    // 6. 恢复图像
    std::vector<cv::Mat> channels;
    cv::split(src, channels);

    std::vector<cv::Mat> recovered_channels(3);

    #pragma omp parallel for
    for (int c = 0; c < 3; c++) {
        // 转换为浮点
        cv::Mat channel_f;
        channels[c].convertTo(channel_f, CV_64FC1);

        // 应用散射模型
        cv::Mat recovered = cv::Mat(src.size(), CV_64FC1);

        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                double val = channel_f.at<double>(y, x);
                double t = refined_transmission.at<double>(y, x);

                // 恢复无雾图像
                recovered.at<double>(y, x) = (val - A[c]) / t + A[c];
            }
        }

        // 裁剪到[0, 255]范围
        recovered.convertTo(recovered_channels[c], CV_8UC1);
    }

    // 合并通道
    cv::merge(recovered_channels, dst);
}

void color_linear_transform_defogging(const cv::Mat& src, cv::Mat& dst, double alpha, double beta) {
    CV_Assert(!src.empty() && src.channels() == 3);
    CV_Assert(alpha > 0);

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 转换为LAB颜色空间
    cv::Mat lab;
    cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);

    // 分离通道
    std::vector<cv::Mat> channels;
    cv::split(lab, channels);

    // 只增强L通道
    cv::Mat& L = channels[0];

    // 线性变换: L' = alpha * L + beta
    L.convertTo(L, CV_8UC1, alpha, beta);

    // 合并通道
    cv::merge(channels, lab);

    // 转回BGR
    cv::cvtColor(lab, dst, cv::COLOR_Lab2BGR);

    // 额外的色彩校正
    cv::Mat hsv;
    cv::cvtColor(dst, hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv, hsv_channels);

    // 增加饱和度
    hsv_channels[1] *= 1.2;

    cv::merge(hsv_channels, hsv);
    cv::cvtColor(hsv, dst, cv::COLOR_HSV2BGR);
}

void logarithmic_enhancement_defogging(const cv::Mat& src, cv::Mat& dst, double gamma, double gain) {
    CV_Assert(!src.empty() && src.channels() == 3);
    CV_Assert(gamma > 0 && gain > 0);

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 分离通道
    std::vector<cv::Mat> channels;
    cv::split(src, channels);

    std::vector<cv::Mat> enhanced_channels(3);

    #pragma omp parallel for
    for (int c = 0; c < 3; c++) {
        // 转换为浮点数
        cv::Mat channel_f;
        channels[c].convertTo(channel_f, CV_32F, 1.0 / 255.0);

        // 对数变换
        cv::Mat log_img;
        cv::log(channel_f + 1.0f, log_img); // 加1避免log(0)

        // 缩放对数图像
        log_img = gain * log_img;

        // 反对数变换
        cv::Mat exp_img;
        cv::exp(log_img, exp_img);
        exp_img -= 1.0f; // 减去之前加的1

        // 伽马校正
        cv::Mat gamma_img;
        cv::pow(exp_img, gamma, gamma_img);

        // 归一化并转回8位图像
        double min_val, max_val;
        cv::minMaxLoc(gamma_img, &min_val, &max_val);
        gamma_img = (gamma_img - min_val) * 255.0 / (max_val - min_val);

        gamma_img.convertTo(enhanced_channels[c], CV_8UC1);
    }

    // 合并通道
    cv::merge(enhanced_channels, dst);

    // 额外的色彩校正
    cv::Mat hsv;
    cv::cvtColor(dst, hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv, hsv_channels);

    // 增加饱和度
    hsv_channels[1] *= 1.1;

    cv::merge(hsv_channels, hsv);
    cv::cvtColor(hsv, dst, cv::COLOR_HSV2BGR);
}

} // namespace advanced
} // namespace ip101