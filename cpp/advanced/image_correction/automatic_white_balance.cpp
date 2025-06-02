#include "automatic_white_balance.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace ip101 {
namespace advanced {

void automatic_white_balance(const cv::Mat& src, cv::Mat& dst, const std::string& method) {
    CV_Assert(!src.empty() && src.channels() == 3);

    if (method == "gray_world") {
        gray_world_white_balance(src, dst);
    } else if (method == "perfect_reflector") {
        perfect_reflector_white_balance(src, dst);
    } else if (method == "retinex") {
        // 简易的单尺度Retinex白平衡实现
        cv::Mat log_src, log_dst;
        src.convertTo(log_src, CV_32F);
        log_src += 1.0f; // 避免log(0)

        std::vector<cv::Mat> channels;
        cv::split(log_src, channels);

        for (int i = 0; i < 3; i++) {
            cv::log(channels[i], channels[i]);

            // 高斯滤波作为照明估计
            cv::Mat illumination;
            cv::GaussianBlur(channels[i], illumination, cv::Size(0, 0), 15);

            // 去除照明分量
            channels[i] = channels[i] - illumination;

            // 还原
            cv::exp(channels[i], channels[i]);
        }

        cv::merge(channels, log_dst);
        log_dst -= 1.0f;

        // 归一化到[0, 255]
        double min_val, max_val;
        cv::minMaxLoc(log_dst, &min_val, &max_val);
        log_dst = (log_dst - min_val) * 255.0 / (max_val - min_val);

        log_dst.convertTo(dst, CV_8UC3);
    } else {
        throw std::invalid_argument("Unsupported white balance method: " + method);
    }
}

void gray_world_white_balance(const cv::Mat& src, cv::Mat& dst) {
    CV_Assert(!src.empty() && src.channels() == 3);

    // 分离通道
    std::vector<cv::Mat> bgr_channels;
    cv::split(src, bgr_channels);

    // 计算每个通道的平均值
    double b_avg = 0, g_avg = 0, r_avg = 0;
    double pixel_count = src.rows * src.cols;

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            b_avg = cv::sum(bgr_channels[0])[0] / pixel_count;
        }
        #pragma omp section
        {
            g_avg = cv::sum(bgr_channels[1])[0] / pixel_count;
        }
        #pragma omp section
        {
            r_avg = cv::sum(bgr_channels[2])[0] / pixel_count;
        }
    }

    // 计算所有通道的平均值作为参考灰度值
    double avg = (b_avg + g_avg + r_avg) / 3.0;

    // 计算每个通道的缩放系数
    double kb = avg / b_avg;
    double kg = avg / g_avg;
    double kr = avg / r_avg;

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 并行应用白平衡
    #pragma omp parallel for
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);

            // 应用缩放系数，保证值在0-255范围内
            dst.at<cv::Vec3b>(y, x)[0] = cv::saturate_cast<uchar>(pixel[0] * kb);
            dst.at<cv::Vec3b>(y, x)[1] = cv::saturate_cast<uchar>(pixel[1] * kg);
            dst.at<cv::Vec3b>(y, x)[2] = cv::saturate_cast<uchar>(pixel[2] * kr);
        }
    }
}

void perfect_reflector_white_balance(const cv::Mat& src, cv::Mat& dst, float ratio) {
    CV_Assert(!src.empty() && src.channels() == 3);
    CV_Assert(ratio > 0 && ratio < 1);

    // 分离通道
    std::vector<cv::Mat> bgr_channels;
    cv::split(src, bgr_channels);

    // 计算需要考虑的像素数量
    int pixelCount = src.rows * src.cols;
    int considerPixels = static_cast<int>(pixelCount * ratio);

    // 为每个通道创建直方图
    const int histSize = 256;
    std::vector<int> b_hist(histSize, 0);
    std::vector<int> g_hist(histSize, 0);
    std::vector<int> r_hist(histSize, 0);

    // 统计直方图
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);
            b_hist[pixel[0]]++;
            g_hist[pixel[1]]++;
            r_hist[pixel[2]]++;
        }
    }

    // 找到每个通道的阈值
    int b_thresh = 255, g_thresh = 255, r_thresh = 255;
    int b_count = 0, g_count = 0, r_count = 0;

    for (int i = 255; i >= 0; i--) {
        b_count += b_hist[i];
        if (b_count >= considerPixels) {
            b_thresh = i;
            break;
        }
    }

    for (int i = 255; i >= 0; i--) {
        g_count += g_hist[i];
        if (g_count >= considerPixels) {
            g_thresh = i;
            break;
        }
    }

    for (int i = 255; i >= 0; i--) {
        r_count += r_hist[i];
        if (r_count >= considerPixels) {
            r_thresh = i;
            break;
        }
    }

    // 计算通道均值
    double b_avg = 0, g_avg = 0, r_avg = 0;
    int valid_count = 0;

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);
            if (pixel[0] >= b_thresh && pixel[1] >= g_thresh && pixel[2] >= r_thresh) {
                b_avg += pixel[0];
                g_avg += pixel[1];
                r_avg += pixel[2];
                valid_count++;
            }
        }
    }

    // 防止除零错误
    if (valid_count == 0) {
        valid_count = 1;
    }

    b_avg /= valid_count;
    g_avg /= valid_count;
    r_avg /= valid_count;

    // 计算缩放系数
    double kb = 255.0 / b_avg;
    double kg = 255.0 / g_avg;
    double kr = 255.0 / r_avg;

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 应用白平衡
    #pragma omp parallel for
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);

            // 应用缩放系数，保证值在0-255范围内
            dst.at<cv::Vec3b>(y, x)[0] = cv::saturate_cast<uchar>(pixel[0] * kb);
            dst.at<cv::Vec3b>(y, x)[1] = cv::saturate_cast<uchar>(pixel[1] * kg);
            dst.at<cv::Vec3b>(y, x)[2] = cv::saturate_cast<uchar>(pixel[2] * kr);
        }
    }
}

} // namespace advanced
} // namespace ip101