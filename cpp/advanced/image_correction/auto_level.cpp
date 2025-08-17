#include <advanced/correction/auto_level.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace ip101 {
namespace advanced {

namespace {
    // 计算累积直方图并找到指定百分比对应的阈值
    std::pair<int, int> computeClippingThresholds(const cv::Mat& channel, float clip_percent) {
        const int histSize = 256;
        const float* ranges[] = { nullptr };

        // 手动计算直方图
        int histogram[histSize] = {0};
        for (int y = 0; y < channel.rows; y++) {
            for (int x = 0; x < channel.cols; x++) {
                int val = channel.at<uchar>(y, x);
                histogram[val]++;
            }
        }

        // 计算总像素数
        int total_pixels = channel.rows * channel.cols;

        // 计算裁剪阈值对应的像素数
        int clip_pixels = static_cast<int>(total_pixels * clip_percent / 100.0f);

        // 查找最小阈值
        int low_threshold = 0;
        int accumulated = 0;
        for (int i = 0; i < histSize; i++) {
            accumulated += histogram[i];
            if (accumulated > clip_pixels) {
                low_threshold = i;
                break;
            }
        }

        // 查找最大阈值
        int high_threshold = 255;
        accumulated = 0;
        for (int i = histSize - 1; i >= 0; i--) {
            accumulated += histogram[i];
            if (accumulated > clip_pixels) {
                high_threshold = i;
                break;
            }
        }

        return std::make_pair(low_threshold, high_threshold);
    }
}  // namespace

void auto_level(const cv::Mat& src, cv::Mat& dst, float clip_percent, bool separate_channels) {
    CV_Assert(!src.empty());
    CV_Assert(clip_percent >= 0.0f && clip_percent <= 5.0f);

    if (src.channels() == 1) {
        // 灰度图像处理
        dst.create(src.size(), src.type());

        auto thresholds = computeClippingThresholds(src, clip_percent);
        int low_threshold = thresholds.first;
        int high_threshold = thresholds.second;

        // 计算缩放因子
        double scale = 255.0 / (high_threshold - low_threshold);

        // 应用线性拉伸
        #pragma omp parallel for
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                int val = src.at<uchar>(y, x);
                val = cv::saturate_cast<uchar>((val - low_threshold) * scale);
                dst.at<uchar>(y, x) = val;
            }
        }
    } else {
        // 彩色图像处理
        dst.create(src.size(), src.type());

        if (separate_channels) {
            // 分离通道，分别处理
            std::vector<cv::Mat> channels;
            cv::split(src, channels);

            #pragma omp parallel for
            for (int c = 0; c < src.channels(); c++) {
                auto thresholds = computeClippingThresholds(channels[c], clip_percent);
                int low_threshold = thresholds.first;
                int high_threshold = thresholds.second;

                double scale = 255.0 / (high_threshold - low_threshold);

                for (int y = 0; y < channels[c].rows; y++) {
                    for (int x = 0; x < channels[c].cols; x++) {
                        int val = channels[c].at<uchar>(y, x);
                        val = cv::saturate_cast<uchar>((val - low_threshold) * scale);
                        channels[c].at<uchar>(y, x) = val;
                    }
                }
            }

            cv::merge(channels, dst);
        } else {
            // 计算全局最小值和最大值
            cv::Mat gray;
            cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

            auto thresholds = computeClippingThresholds(gray, clip_percent);
            int low_threshold = thresholds.first;
            int high_threshold = thresholds.second;

            double scale = 255.0 / (high_threshold - low_threshold);

            #pragma omp parallel for
            for (int y = 0; y < src.rows; y++) {
                for (int x = 0; x < src.cols; x++) {
                    cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);
                    for (int c = 0; c < 3; c++) {
                        int val = pixel[c];
                        val = cv::saturate_cast<uchar>((val - low_threshold) * scale);
                        dst.at<cv::Vec3b>(y, x)[c] = val;
                    }
                }
            }
        }
    }
}

void auto_contrast(const cv::Mat& src, cv::Mat& dst, float clip_percent, bool separate_channels) {
    CV_Assert(!src.empty());
    CV_Assert(clip_percent >= 0.0f && clip_percent <= 5.0f);

    // 创建目标图像
    dst.create(src.size(), src.type());

    if (src.channels() == 1) {
        // 灰度图像处理
        // 计算均值和标准差
        cv::Scalar mean, stddev;
        cv::meanStdDev(src, mean, stddev);

        // 计算拉伸参数
        double mean_val = mean[0];
        double std_val = stddev[0];

        // 定义对比度增强系数
        double alpha = 128.0 / std_val;

        // 应用对比度增强
        #pragma omp parallel for
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                int val = src.at<uchar>(y, x);
                val = cv::saturate_cast<uchar>(alpha * (val - mean_val) + 128);
                dst.at<uchar>(y, x) = val;
            }
        }
    } else {
        // 彩色图像处理
        if (separate_channels) {
            // 分离通道，分别处理
            std::vector<cv::Mat> channels;
            cv::split(src, channels);

            #pragma omp parallel for
            for (int c = 0; c < src.channels(); c++) {
                cv::Scalar mean, stddev;
                cv::meanStdDev(channels[c], mean, stddev);

                double mean_val = mean[0];
                double std_val = stddev[0];
                double alpha = 128.0 / std_val;

                for (int y = 0; y < channels[c].rows; y++) {
                    for (int x = 0; x < channels[c].cols; x++) {
                        int val = channels[c].at<uchar>(y, x);
                        val = cv::saturate_cast<uchar>(alpha * (val - mean_val) + 128);
                        channels[c].at<uchar>(y, x) = val;
                    }
                }
            }

            cv::merge(channels, dst);
        } else {
            // 全局处理，保持颜色
            cv::Mat ycrcb;
            cv::cvtColor(src, ycrcb, cv::COLOR_BGR2YCrCb);

            std::vector<cv::Mat> channels;
            cv::split(ycrcb, channels);

            // 只增强亮度通道的对比度
            cv::Scalar mean, stddev;
            cv::meanStdDev(channels[0], mean, stddev);

            double mean_val = mean[0];
            double std_val = stddev[0];
            double alpha = 128.0 / std_val;

            #pragma omp parallel for
            for (int y = 0; y < channels[0].rows; y++) {
                for (int x = 0; x < channels[0].cols; x++) {
                    int val = channels[0].at<uchar>(y, x);
                    val = cv::saturate_cast<uchar>(alpha * (val - mean_val) + 128);
                    channels[0].at<uchar>(y, x) = val;
                }
            }

            cv::merge(channels, ycrcb);
            cv::cvtColor(ycrcb, dst, cv::COLOR_YCrCb2BGR);
        }
    }
}

} // namespace advanced
} // namespace ip101