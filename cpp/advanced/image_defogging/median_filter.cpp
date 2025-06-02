#include "median_filter.hpp"
#include "dark_channel.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace ip101 {
namespace advanced {

// 手动实现中值滤波（简化版，效率较低但易于理解）
void custom_median_filter(const cv::Mat& src, cv::Mat& dst, int kernel_size) {
    CV_Assert(!src.empty());
    CV_Assert(kernel_size > 0 && kernel_size % 2 == 1); // 必须是奇数

    dst.create(src.size(), src.type());

    int radius = kernel_size / 2;

    // 对每个像素应用中值滤波
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            // 确定窗口范围
            int start_y = std::max(0, y - radius);
            int end_y = std::min(src.rows - 1, y + radius);
            int start_x = std::max(0, x - radius);
            int end_x = std::min(src.cols - 1, x + radius);

            // 收集窗口内的值
            std::vector<double> values;

            for (int i = start_y; i <= end_y; i++) {
                for (int j = start_x; j <= end_x; j++) {
                    double val = src.type() == CV_64FC1 ?
                                 src.at<double>(i, j) :
                                 static_cast<double>(src.at<uchar>(i, j));
                    values.push_back(val);
                }
            }

            // 计算中值
            std::sort(values.begin(), values.end());
            double median = values[values.size() / 2];

            // 赋值到目标图像
            if (src.type() == CV_64FC1) {
                dst.at<double>(y, x) = median;
            } else {
                dst.at<uchar>(y, x) = cv::saturate_cast<uchar>(median);
            }
        }
    }
}

// 自适应中值滤波算法
void custom_adaptive_median_filter(const cv::Mat& src, cv::Mat& dst, int init_size, int max_size) {
    CV_Assert(!src.empty());
    CV_Assert(init_size > 0 && init_size % 2 == 1); // 必须是奇数
    CV_Assert(max_size >= init_size && max_size % 2 == 1); // 必须是奇数且大于等于init_size

    dst.create(src.size(), src.type());

    // 对每个像素应用自适应中值滤波
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            double pixel_value = src.type() == CV_64FC1 ?
                               src.at<double>(y, x) :
                               static_cast<double>(src.at<uchar>(y, x));

            // 从初始大小开始，逐渐增加窗口大小
            int current_size = init_size;
            bool found_median = false;
            double median = 0;

            while (current_size <= max_size && !found_median) {
                int radius = current_size / 2;

                // 确定窗口范围
                int start_y = std::max(0, y - radius);
                int end_y = std::min(src.rows - 1, y + radius);
                int start_x = std::max(0, x - radius);
                int end_x = std::min(src.cols - 1, x + radius);

                // 收集窗口内的值
                std::vector<double> values;

                for (int i = start_y; i <= end_y; i++) {
                    for (int j = start_x; j <= end_x; j++) {
                        double val = src.type() == CV_64FC1 ?
                                     src.at<double>(i, j) :
                                     static_cast<double>(src.at<uchar>(i, j));
                        values.push_back(val);
                    }
                }

                // 排序并计算统计量
                std::sort(values.begin(), values.end());
                median = values[values.size() / 2];
                double min_val = values.front();
                double max_val = values.back();

                // 计算噪声检测的标准差
                double sum = 0, sum_squared = 0;
                for (double val : values) {
                    sum += val;
                    sum_squared += val * val;
                }
                double mean = sum / values.size();
                double variance = (sum_squared / values.size()) - (mean * mean);
                double std_dev = std::sqrt(variance);

                // 噪声检测
                if ((std::abs(pixel_value - median) / std_dev) > 2.0) {
                    // 可能是噪声，继续增大窗口
                    current_size += 2;
                } else {
                    // 不是噪声，使用该中值
                    found_median = true;
                }
            }

            // 如果没有找到合适的中值，使用最大窗口的中值
            if (!found_median) {
                int radius = max_size / 2;

                // 确定窗口范围
                int start_y = std::max(0, y - radius);
                int end_y = std::min(src.rows - 1, y + radius);
                int start_x = std::max(0, x - radius);
                int end_x = std::min(src.cols - 1, x + radius);

                // 收集窗口内的值
                std::vector<double> values;

                for (int i = start_y; i <= end_y; i++) {
                    for (int j = start_x; j <= end_x; j++) {
                        double val = src.type() == CV_64FC1 ?
                                     src.at<double>(i, j) :
                                     static_cast<double>(src.at<uchar>(i, j));
                        values.push_back(val);
                    }
                }

                std::sort(values.begin(), values.end());
                median = values[values.size() / 2];
            }

            // 赋值到目标图像
            if (src.type() == CV_64FC1) {
                dst.at<double>(y, x) = median;
            } else {
                dst.at<uchar>(y, x) = cv::saturate_cast<uchar>(median);
            }
        }
    }
}

void median_filter_defogging(const cv::Mat& src, cv::Mat& dst, int kernel_size, double omega, double t0) {
    CV_Assert(!src.empty() && src.channels() == 3);
    CV_Assert(kernel_size > 0 && kernel_size % 2 == 1); // 必须是奇数
    CV_Assert(omega > 0 && omega <= 1.0 && t0 > 0 && t0 < 1.0);

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 1. 计算暗通道
    cv::Mat dark;
    compute_dark_channel(src, dark, 15); // 使用固定的暗通道窗口大小

    // 2. 估计大气光照值
    cv::Vec3d A = estimate_atmospheric_light(src, dark);

    // 3. 估计粗透射率
    cv::Mat transmission = estimate_transmission(src, dark, A, omega);

    // 4. 应用中值滤波改进透射率图
    cv::Mat refined_transmission;

    // 使用OpenCV的中值滤波函数
    cv::medianBlur(transmission, refined_transmission, kernel_size);

    // 5. 限制最小透射率，避免过度去雾
    cv::max(refined_transmission, t0, refined_transmission);

    // 6. 应用大气散射模型恢复图像
    std::vector<cv::Mat> channels;
    cv::split(src, channels);

    std::vector<cv::Mat> recovered_channels(3);

    #pragma omp parallel for
    for (int c = 0; c < 3; c++) {
        // 转换到浮点数
        cv::Mat channel_f;
        channels[c].convertTo(channel_f, CV_64FC1);

        // 应用大气散射模型: J = (I - A) / t + A
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
        cv::Mat recovered_8u;
        recovered.convertTo(recovered_8u, CV_8UC1);
        recovered_channels[c] = recovered_8u;
    }

    // 合并通道
    cv::merge(recovered_channels, dst);
}

void improved_median_filter_defogging(const cv::Mat& src, cv::Mat& dst, int kernel_size, double omega, double t0, double lambda) {
    CV_Assert(!src.empty() && src.channels() == 3);
    CV_Assert(kernel_size > 0 && kernel_size % 2 == 1); // 必须是奇数
    CV_Assert(omega > 0 && omega <= 1.0 && t0 > 0 && t0 < 1.0 && lambda > 0);

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 1. 计算暗通道
    cv::Mat dark;
    compute_dark_channel(src, dark, 15);

    // 2. 估计大气光照值
    cv::Vec3d A = estimate_atmospheric_light(src, dark);

    // 3. 估计粗透射率
    cv::Mat transmission = estimate_transmission(src, dark, A, omega);

    // 4. 计算亮度图像
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_64FC1, 1.0/255.0);

    // 5. 计算梯度的幅值
    cv::Mat grad_x, grad_y, gradient;
    cv::Sobel(gray, grad_x, CV_64F, 1, 0);
    cv::Sobel(gray, grad_y, CV_64F, 0, 1);
    cv::magnitude(grad_x, grad_y, gradient);

    // 6. 应用边缘保持的中值滤波
    // 首先使用OpenCV的中值滤波
    cv::Mat median_trans;
    transmission.convertTo(median_trans, CV_8UC1, 255.0);
    cv::medianBlur(median_trans, median_trans, kernel_size);
    median_trans.convertTo(median_trans, CV_64FC1, 1.0/255.0);

    // 7. 使用梯度信息修正透射率
    cv::Mat refined_transmission = transmission.clone();

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            double grad = gradient.at<double>(y, x);
            double alpha = std::exp(-lambda * grad); // 梯度越大，alpha越小

            refined_transmission.at<double>(y, x) =
                alpha * median_trans.at<double>(y, x) + (1.0 - alpha) * transmission.at<double>(y, x);
        }
    }

    // 8. 限制最小透射率
    cv::max(refined_transmission, t0, refined_transmission);

    // 9. 应用大气散射模型恢复图像
    std::vector<cv::Mat> channels;
    cv::split(src, channels);

    std::vector<cv::Mat> recovered_channels(3);

    #pragma omp parallel for
    for (int c = 0; c < 3; c++) {
        // 转换到浮点数
        cv::Mat channel_f;
        channels[c].convertTo(channel_f, CV_64FC1);

        // 应用大气散射模型: J = (I - A) / t + A
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
        cv::Mat recovered_8u;
        recovered.convertTo(recovered_8u, CV_8UC1);
        recovered_channels[c] = recovered_8u;
    }

    // 合并通道
    cv::merge(recovered_channels, dst);

    // 10. 对结果进行额外的色彩增强
    cv::Mat hsv;
    cv::cvtColor(dst, hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv, hsv_channels);

    // 增加饱和度
    hsv_channels[1] *= 1.1;

    cv::merge(hsv_channels, hsv);
    cv::cvtColor(hsv, dst, cv::COLOR_HSV2BGR);
}

void adaptive_median_filter_defogging(const cv::Mat& src, cv::Mat& dst, int init_size, int max_size, double omega, double t0) {
    CV_Assert(!src.empty() && src.channels() == 3);
    CV_Assert(init_size > 0 && init_size % 2 == 1); // 必须是奇数
    CV_Assert(max_size >= init_size && max_size % 2 == 1); // 必须是奇数且大于等于init_size
    CV_Assert(omega > 0 && omega <= 1.0 && t0 > 0 && t0 < 1.0);

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 1. 计算暗通道
    cv::Mat dark;
    compute_dark_channel(src, dark, 15); // 使用固定的暗通道窗口大小

    // 2. 估计大气光照值
    cv::Vec3d A = estimate_atmospheric_light(src, dark);

    // 3. 估计粗透射率
    cv::Mat transmission = estimate_transmission(src, dark, A, omega);

    // 4. 应用自适应中值滤波改进透射率图
    cv::Mat refined_transmission = transmission.clone();
    custom_adaptive_median_filter(transmission, refined_transmission, init_size, max_size);

    // 5. 限制最小透射率，避免过度去雾
    cv::max(refined_transmission, t0, refined_transmission);

    // 6. 应用大气散射模型恢复图像
    std::vector<cv::Mat> channels;
    cv::split(src, channels);

    std::vector<cv::Mat> recovered_channels(3);

    #pragma omp parallel for
    for (int c = 0; c < 3; c++) {
        // 转换到浮点数
        cv::Mat channel_f;
        channels[c].convertTo(channel_f, CV_64FC1);

        // 应用大气散射模型: J = (I - A) / t + A
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
        cv::Mat recovered_8u;
        recovered.convertTo(recovered_8u, CV_8UC1);
        recovered_channels[c] = recovered_8u;
    }

    // 合并通道
    cv::merge(recovered_channels, dst);
}

} // namespace advanced
} // namespace ip101