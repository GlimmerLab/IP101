#include <advanced/defogging/realtime_dehazing.hpp>
#include <advanced/defogging/dark_channel.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <deque>
#include <omp.h>

namespace ip101 {
namespace advanced {

// 快速最小值滤波 - 使用滑动窗口法
void fast_min_filter(const cv::Mat& src, cv::Mat& dst, int radius) {
    CV_Assert(!src.empty());
    CV_Assert(radius > 0);

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 水平方向最小值滤波
    cv::Mat temp(src.size(), src.type());
    int window_size = 2 * radius + 1;

    // 水平滤波
    for (int y = 0; y < src.rows; y++) {
        // 使用双端队列来存储窗口内的最小值候选
        std::deque<int> min_deque;

        // 初始化第一个窗口
        for (int x = 0; x < std::min(window_size, src.cols); x++) {
            while (!min_deque.empty() && src.at<uchar>(y, x) < src.at<uchar>(y, min_deque.back())) {
                min_deque.pop_back();
            }
            min_deque.push_back(x);
        }

        // 滑动窗口，计算每个位置的最小值
        for (int x = 0; x < src.cols; x++) {
            // 添加新元素到窗口
            int right = x + radius;
            if (right < src.cols) {
                while (!min_deque.empty() && src.at<uchar>(y, right) < src.at<uchar>(y, min_deque.back())) {
                    min_deque.pop_back();
                }
                min_deque.push_back(right);
            }

            // 移除超出窗口的元素
            if (!min_deque.empty() && min_deque.front() < x - radius) {
                min_deque.pop_front();
            }

            // 当前窗口的最小值
            temp.at<uchar>(y, x) = src.at<uchar>(y, min_deque.front());
        }
    }

    // 垂直滤波
    for (int x = 0; x < temp.cols; x++) {
        // 使用双端队列来存储窗口内的最小值候选
        std::deque<int> min_deque;

        // 初始化第一个窗口
        for (int y = 0; y < std::min(window_size, temp.rows); y++) {
            while (!min_deque.empty() && temp.at<uchar>(y, x) < temp.at<uchar>(min_deque.back(), x)) {
                min_deque.pop_back();
            }
            min_deque.push_back(y);
        }

        // 滑动窗口，计算每个位置的最小值
        for (int y = 0; y < temp.rows; y++) {
            // 添加新元素到窗口
            int bottom = y + radius;
            if (bottom < temp.rows) {
                while (!min_deque.empty() && temp.at<uchar>(bottom, x) < temp.at<uchar>(min_deque.back(), x)) {
                    min_deque.pop_back();
                }
                min_deque.push_back(bottom);
            }

            // 移除超出窗口的元素
            if (!min_deque.empty() && min_deque.front() < y - radius) {
                min_deque.pop_front();
            }

            // 当前窗口的最小值
            dst.at<uchar>(y, x) = temp.at<uchar>(min_deque.front(), x);
        }
    }
}

// 快速计算暗通道
void fast_dark_channel(const cv::Mat& src, cv::Mat& dark, int radius) {
    CV_Assert(!src.empty() && src.channels() == 3);
    CV_Assert(radius > 0);

    // 计算每个像素的BGR通道最小值
    cv::Mat min_channel(src.size(), CV_8UC1);

    #pragma omp parallel for
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);
            min_channel.at<uchar>(y, x) = std::min(std::min(pixel[0], pixel[1]), pixel[2]);
        }
    }

    // 应用最小值滤波，得到暗通道
    fast_min_filter(min_channel, dark, radius);
}

// 实现高效的暗通道去雾算法
void realtime_dark_channel_dehazing(const cv::Mat& src, cv::Mat& dst, int radius, int min_filter_radius, double omega, double t0) {
    CV_Assert(!src.empty() && src.channels() == 3);
    CV_Assert(radius > 0 && min_filter_radius > 0);
    CV_Assert(omega > 0 && omega <= 1.0 && t0 > 0 && t0 < 1.0);

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 1. 快速计算暗通道
    cv::Mat dark;
    fast_dark_channel(src, dark, min_filter_radius);

    // 2. 估计大气光照值
    cv::Vec3d A = estimate_atmospheric_light(src, dark);

    // 3. 估计粗透射率
    cv::Mat transmission = cv::Mat(src.size(), CV_64FC1);

    for (int y = 0; y < dark.rows; y++) {
        for (int x = 0; x < dark.cols; x++) {
            double dark_val = static_cast<double>(dark.at<uchar>(y, x)) / 255.0;
            transmission.at<double>(y, x) = 1.0 - omega * dark_val;
        }
    }

    // 4. 应用双边滤波优化透射率
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    cv::Mat trans_8u;
    transmission.convertTo(trans_8u, CV_8UC1, 255.0);

    cv::Mat refined_transmission;
    cv::bilateralFilter(trans_8u, refined_transmission, radius, 50.0, 50.0);
    refined_transmission.convertTo(refined_transmission, CV_64FC1, 1.0 / 255.0);

    // 5. 限制最小透射率
    cv::max(refined_transmission, t0, refined_transmission);

    // 6. 恢复去雾图像
    fast_dehazing_model(src, dst, A, refined_transmission);
}

// 实时视频去雾算法
void realtime_dehazing(const cv::Mat& src, cv::Mat& dst, double downsample_factor, double omega, double t0) {
    CV_Assert(!src.empty() && src.channels() == 3);
    CV_Assert(downsample_factor > 0 && downsample_factor <= 1.0);
    CV_Assert(omega > 0 && omega <= 1.0 && t0 > 0 && t0 < 1.0);

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 1. 下采样以加速处理
    cv::Mat small_img;
    cv::Size down_size(static_cast<int>(src.cols * downsample_factor),
                      static_cast<int>(src.rows * downsample_factor));

    if (downsample_factor < 1.0) {
        cv::resize(src, small_img, down_size, 0, 0, cv::INTER_LINEAR);
    } else {
        small_img = src.clone();
    }

    // 2. 在低分辨率图像上执行去雾
    cv::Mat dehazed_small;
    realtime_dark_channel_dehazing(small_img, dehazed_small, 7, 3, omega, t0);

    // 3. 上采样回原始大小
    if (downsample_factor < 1.0) {
        cv::resize(dehazed_small, dst, src.size(), 0, 0, cv::INTER_LINEAR);
    } else {
        dehazed_small.copyTo(dst);
    }

    // 4. 执行额外的色彩校正
    cv::Mat hsv;
    cv::cvtColor(dst, hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);

    // 提高饱和度，使恢复的图像更有活力
    channels[1] *= 1.1;

    cv::merge(channels, hsv);
    cv::cvtColor(hsv, dst, cv::COLOR_HSV2BGR);
}

// 快速应用大气散射模型
void fast_dehazing_model(const cv::Mat& src, cv::Mat& dst, const cv::Vec3d& A, const cv::Mat& t) {
    CV_Assert(!src.empty() && src.channels() == 3);
    CV_Assert(!t.empty() && t.size() == src.size() && t.type() == CV_64FC1);

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 使用SIMD优化的方式应用去雾模型
    #pragma omp parallel for
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);
            double trans = t.at<double>(y, x);

            // 按通道应用去雾模型: J = (I - A) / t + A
            for (int c = 0; c < 3; c++) {
                double val = static_cast<double>(pixel[c]);
                double recovered = (val - A[c]) / trans + A[c];
                dst.at<cv::Vec3b>(y, x)[c] = cv::saturate_cast<uchar>(recovered);
            }
        }
    }
}

} // namespace advanced
} // namespace ip101