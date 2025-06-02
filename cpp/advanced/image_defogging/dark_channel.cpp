#include "dark_channel.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace ip101 {
namespace advanced {

// 计算暗通道图像
void compute_dark_channel(const cv::Mat& src, cv::Mat& dark, int patch_size) {
    CV_Assert(!src.empty() && src.channels() == 3);
    CV_Assert(patch_size > 0);

    // 初始化暗通道图像
    dark = cv::Mat(src.size(), CV_8UC1, cv::Scalar(255));

    // 分割BGR通道
    std::vector<cv::Mat> channels;
    cv::split(src, channels);

    int radius = patch_size / 2;

    // 对每个像素进行处理
    #pragma omp parallel for
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            // 定义局部区域范围
            int start_y = std::max(0, y - radius);
            int end_y = std::min(src.rows - 1, y + radius);
            int start_x = std::max(0, x - radius);
            int end_x = std::min(src.cols - 1, x + radius);

            // 查找局部区域内三个通道的最小值
            uchar min_val = 255;

            for (int i = start_y; i <= end_y; i++) {
                for (int j = start_x; j <= end_x; j++) {
                    for (int c = 0; c < 3; c++) {
                        min_val = std::min(min_val, channels[c].at<uchar>(i, j));
                    }
                }
            }

            dark.at<uchar>(y, x) = min_val;
        }
    }
}

// 估计大气光照值
cv::Vec3d estimate_atmospheric_light(const cv::Mat& src, const cv::Mat& dark, double percent) {
    CV_Assert(!src.empty() && src.channels() == 3);
    CV_Assert(!dark.empty() && dark.channels() == 1);

    // 计算用于估计的像素数量
    int num_pixels = static_cast<int>(dark.rows * dark.cols * percent);
    num_pixels = std::max(1, num_pixels); // 至少一个像素

    // 创建暗通道值和对应索引的向量
    std::vector<std::pair<uchar, cv::Point>> dark_pixels;
    dark_pixels.reserve(dark.rows * dark.cols);

    for (int y = 0; y < dark.rows; y++) {
        for (int x = 0; x < dark.cols; x++) {
            dark_pixels.push_back(std::make_pair(dark.at<uchar>(y, x), cv::Point(x, y)));
        }
    }

    // 按暗通道值降序排序（越亮的在前面）
    std::partial_sort(dark_pixels.begin(), dark_pixels.begin() + num_pixels, dark_pixels.end(),
                     [](const std::pair<uchar, cv::Point>& a, const std::pair<uchar, cv::Point>& b) {
                         return a.first > b.first;
                     });

    // 在源图像中找到最亮的区域的平均值作为大气光照值
    double sum_b = 0, sum_g = 0, sum_r = 0;

    for (int i = 0; i < num_pixels; i++) {
        cv::Point p = dark_pixels[i].second;
        cv::Vec3b pixel = src.at<cv::Vec3b>(p);

        sum_b += pixel[0];
        sum_g += pixel[1];
        sum_r += pixel[2];
    }

    cv::Vec3d A(sum_b / num_pixels, sum_g / num_pixels, sum_r / num_pixels);
    return A;
}

// 估计透射率图像
cv::Mat estimate_transmission(const cv::Mat& src, const cv::Mat& dark, const cv::Vec3d& A, double omega) {
    CV_Assert(!src.empty() && src.channels() == 3);
    CV_Assert(!dark.empty() && dark.channels() == 1);

    // 创建归一化的图像
    cv::Mat normalized = cv::Mat(src.size(), CV_64FC3);

    // 归一化图像 (I / A)
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);

            normalized.at<cv::Vec3d>(y, x)[0] = static_cast<double>(pixel[0]) / A[0];
            normalized.at<cv::Vec3d>(y, x)[1] = static_cast<double>(pixel[1]) / A[1];
            normalized.at<cv::Vec3d>(y, x)[2] = static_cast<double>(pixel[2]) / A[2];
        }
    }

    // 计算归一化图像的暗通道
    cv::Mat norm_dark;
    cv::Mat norm_dark_8u;

    normalized.convertTo(norm_dark_8u, CV_8UC3, 255.0);
    compute_dark_channel(norm_dark_8u, norm_dark);

    // 计算透射率图像 t(x) = 1 - omega * J_dark(x)
    cv::Mat transmission = cv::Mat(src.size(), CV_64FC1);

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            double dark_val = static_cast<double>(norm_dark.at<uchar>(y, x)) / 255.0;
            transmission.at<double>(y, x) = 1.0 - omega * dark_val;
        }
    }

    return transmission;
}

// 主去雾函数
void dark_channel_defogging(const cv::Mat& src, cv::Mat& dst, int patch_size, double omega, double t0) {
    CV_Assert(!src.empty() && src.channels() == 3);
    CV_Assert(patch_size > 0 && omega > 0 && omega <= 1.0 && t0 > 0 && t0 < 1.0);

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 1. 计算暗通道
    cv::Mat dark;
    compute_dark_channel(src, dark, patch_size);

    // 2. 估计大气光照值
    cv::Vec3d A = estimate_atmospheric_light(src, dark);

    // 3. 估计透射率
    cv::Mat transmission = estimate_transmission(src, dark, A, omega);

    // 4. 优化透射率（使用导向滤波或者双边滤波）
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    cv::Mat refined_transmission;
    cv::Mat guidance = gray;
    guidance.convertTo(guidance, CV_64FC1, 1.0 / 255.0);

    // 手动实现简化版导向滤波
    int r = std::max(10, patch_size); // 滤波半径
    double eps = 1e-6; // 正则化参数

    cv::Mat N = cv::Mat(cv::Size(src.cols, src.rows), CV_64FC1, cv::Scalar(1.0));
    cv::Mat mean_I, mean_p, mean_Ip, mean_II;

    // 计算均值
    cv::boxFilter(guidance, mean_I, CV_64FC1, cv::Size(r, r), cv::Point(-1, -1), true);
    cv::boxFilter(transmission, mean_p, CV_64FC1, cv::Size(r, r), cv::Point(-1, -1), true);

    // 计算协方差
    cv::Mat Ip = guidance.mul(transmission);
    cv::boxFilter(Ip, mean_Ip, CV_64FC1, cv::Size(r, r), cv::Point(-1, -1), true);

    // 计算方差
    cv::Mat I_squared = guidance.mul(guidance);
    cv::boxFilter(I_squared, mean_II, CV_64FC1, cv::Size(r, r), cv::Point(-1, -1), true);

    // 计算协方差
    cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);

    // 计算方差
    cv::Mat var_I = mean_II - mean_I.mul(mean_I);

    // 计算系数a和b
    cv::Mat a = cov_Ip / (var_I + eps);
    cv::Mat b = mean_p - a.mul(mean_I);

    // 对a和b进行均值滤波
    cv::Mat mean_a, mean_b;
    cv::boxFilter(a, mean_a, CV_64FC1, cv::Size(r, r), cv::Point(-1, -1), true);
    cv::boxFilter(b, mean_b, CV_64FC1, cv::Size(r, r), cv::Point(-1, -1), true);

    // 计算输出
    refined_transmission = mean_a.mul(guidance) + mean_b;

    // 5. 限制最小透射率，避免过度去雾
    cv::max(refined_transmission, t0, refined_transmission);

    // 6. 利用大气散射模型恢复图像
    std::vector<cv::Mat> channels;
    cv::split(src, channels);

    std::vector<cv::Mat> recovered_channels(3);

    for (int c = 0; c < 3; c++) {
        // 转换到浮点数
        cv::Mat channel_f;
        channels[c].convertTo(channel_f, CV_64FC1);

        // 应用大气散射模型: J = (I - A) / t + A
        cv::Mat recovered = cv::Mat(src.size(), CV_64FC1);

        #pragma omp parallel for
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                double val = channel_f.at<double>(y, x);
                double t = refined_transmission.at<double>(y, x);

                // 恢复无雾图像
                recovered.at<double>(y, x) = (val - A[c]) / t + A[c];
            }
        }

        // 裁剪到[0, 255]范围
        cv::Mat recovered_8u;
        recovered.convertTo(recovered_8u, CV_8UC1);
        recovered_channels[c] = recovered_8u;
    }

    // 合并通道
    cv::merge(recovered_channels, dst);
}

// 双边滤波改进的暗通道去雾算法
void bilateral_dark_channel_defogging(const cv::Mat& src, cv::Mat& dst, int patch_size, double omega, double t0) {
    CV_Assert(!src.empty() && src.channels() == 3);
    CV_Assert(patch_size > 0 && omega > 0 && omega <= 1.0 && t0 > 0 && t0 < 1.0);

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 1. 计算暗通道
    cv::Mat dark;
    compute_dark_channel(src, dark, patch_size);

    // 2. 估计大气光照值
    cv::Vec3d A = estimate_atmospheric_light(src, dark);

    // 3. 估计透射率
    cv::Mat transmission = estimate_transmission(src, dark, A, omega);

    // 4. 使用双边滤波优化透射率
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    cv::Mat trans_8u;
    transmission.convertTo(trans_8u, CV_8UC1, 255.0);

    cv::Mat refined_transmission;
    cv::bilateralFilter(trans_8u, refined_transmission, patch_size, 50, 50);
    refined_transmission.convertTo(refined_transmission, CV_64FC1, 1.0 / 255.0);

    // 5. 限制最小透射率，避免过度去雾
    cv::max(refined_transmission, t0, refined_transmission);

    // 6. 利用大气散射模型恢复图像
    std::vector<cv::Mat> channels;
    cv::split(src, channels);

    std::vector<cv::Mat> recovered_channels(3);

    for (int c = 0; c < 3; c++) {
        // 转换到浮点数
        cv::Mat channel_f;
        channels[c].convertTo(channel_f, CV_64FC1);

        // 应用大气散射模型: J = (I - A) / t + A
        cv::Mat recovered = cv::Mat(src.size(), CV_64FC1);

        #pragma omp parallel for
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                double val = channel_f.at<double>(y, x);
                double t = refined_transmission.at<double>(y, x);

                // 恢复无雾图像
                recovered.at<double>(y, x) = (val - A[c]) / t + A[c];
            }
        }

        // 裁剪到[0, 255]范围
        cv::Mat recovered_8u;
        recovered.convertTo(recovered_8u, CV_8UC1);
        recovered_channels[c] = recovered_8u;
    }

    // 合并通道
    cv::merge(recovered_channels, dst);
}

} // namespace advanced
} // namespace ip101