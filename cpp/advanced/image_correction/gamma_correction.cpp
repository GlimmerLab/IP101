#include "gamma_correction.hpp"
#include <vector>
#include <cmath>
#include <omp.h>

namespace ip101 {
namespace advanced {

// 创建伽马校正查找表
void create_gamma_lut(double gamma, uchar* lut) {
    for (int i = 0; i < 256; i++) {
        double val = static_cast<double>(i) / 255.0;
        val = std::pow(val, gamma);
        lut[i] = cv::saturate_cast<uchar>(val * 255.0);
    }
}

void standard_gamma_correction(const cv::Mat& src, cv::Mat& dst, double gamma) {
    CV_Assert(!src.empty());
    CV_Assert(gamma > 0.0);

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 创建查找表
    uchar lut[256];
    create_gamma_lut(gamma, lut);

    // 应用gamma校正
    if (src.channels() == 1) {
        // 灰度图像
        #pragma omp parallel for
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                uchar val = src.at<uchar>(y, x);
                dst.at<uchar>(y, x) = lut[val];
            }
        }
    } else {
        // 彩色图像
        #pragma omp parallel for
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);
                for (int c = 0; c < 3; c++) {
                    pixel[c] = lut[pixel[c]];
                }
                dst.at<cv::Vec3b>(y, x) = pixel;
            }
        }
    }
}

void two_dimensional_gamma_correction(const cv::Mat& src, cv::Mat& dst,
                                     double gamma_dark, double gamma_bright,
                                     int threshold, double smooth_factor) {
    CV_Assert(!src.empty());
    CV_Assert(gamma_dark > 0.0 && gamma_bright > 0.0);
    CV_Assert(threshold >= 0 && threshold <= 255);
    CV_Assert(smooth_factor >= 0.0 && smooth_factor <= 1.0);

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 创建两个查找表：暗区和亮区
    uchar lut_dark[256];
    uchar lut_bright[256];
    create_gamma_lut(gamma_dark, lut_dark);
    create_gamma_lut(gamma_bright, lut_bright);

    // 计算平滑过渡宽度
    int transition_width = static_cast<int>(255.0 * smooth_factor);
    int start_transition = std::max(0, threshold - transition_width / 2);
    int end_transition = std::min(255, threshold + transition_width / 2);

    // 创建混合查找表
    uchar lut_blended[256];
    for (int i = 0; i < 256; i++) {
        if (i <= start_transition) {
            // 暗区
            lut_blended[i] = lut_dark[i];
        } else if (i >= end_transition) {
            // 亮区
            lut_blended[i] = lut_bright[i];
        } else {
            // 平滑过渡区
            double weight = static_cast<double>(i - start_transition) / (end_transition - start_transition);
            lut_blended[i] = cv::saturate_cast<uchar>((1.0 - weight) * lut_dark[i] + weight * lut_bright[i]);
        }
    }

    // 应用校正
    if (src.channels() == 1) {
        // 灰度图像
        #pragma omp parallel for
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                uchar val = src.at<uchar>(y, x);
                dst.at<uchar>(y, x) = lut_blended[val];
            }
        }
    } else {
        // 彩色图像
        cv::Mat ycrcb;
        cv::cvtColor(src, ycrcb, cv::COLOR_BGR2YCrCb);

        std::vector<cv::Mat> channels;
        cv::split(ycrcb, channels);

        // 仅对亮度通道应用gamma校正
        #pragma omp parallel for
        for (int y = 0; y < channels[0].rows; y++) {
            for (int x = 0; x < channels[0].cols; x++) {
                uchar val = channels[0].at<uchar>(y, x);
                channels[0].at<uchar>(y, x) = lut_blended[val];
            }
        }

        cv::merge(channels, ycrcb);
        cv::cvtColor(ycrcb, dst, cv::COLOR_YCrCb2BGR);
    }
}

void adaptive_gamma_correction(const cv::Mat& src, cv::Mat& dst, int blocks) {
    CV_Assert(!src.empty());
    CV_Assert(blocks > 0);

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 转换为YCrCb颜色空间处理
    cv::Mat ycrcb;
    if (src.channels() == 3) {
        cv::cvtColor(src, ycrcb, cv::COLOR_BGR2YCrCb);
    } else {
        ycrcb = src.clone();
    }

    std::vector<cv::Mat> channels;
    cv::split(ycrcb, channels);

    // 仅处理亮度通道
    cv::Mat& Y = channels[0];

    // 计算块大小
    int block_height = Y.rows / blocks;
    int block_width = Y.cols / blocks;

    // 为每个块创建掩码和gamma值
    std::vector<cv::Mat> masks;
    std::vector<double> gammas;

    // 对每个块计算适当的gamma值
    for (int i = 0; i < blocks; i++) {
        for (int j = 0; j < blocks; j++) {
            // 计算块区域
            int start_y = i * block_height;
            int end_y = (i == blocks - 1) ? Y.rows : (i + 1) * block_height;
            int start_x = j * block_width;
            int end_x = (j == blocks - 1) ? Y.cols : (j + 1) * block_width;

            cv::Rect block_rect(start_x, start_y, end_x - start_x, end_y - start_y);
            cv::Mat block = Y(block_rect);

            // 计算块的平均亮度
            double mean_val = cv::mean(block)[0] / 255.0;

            // 根据平均亮度计算gamma值
            // 暗区使用小于1的gamma增亮，亮区使用大于1的gamma减暗
            double gamma = 0.0;
            if (mean_val < 0.5) {
                // 暗区，增亮
                gamma = 0.5 + 0.5 * mean_val; // gamma在0.5到0.75之间
            } else {
                // 亮区，减暗
                gamma = 1.0 + (mean_val - 0.5) * 1.0; // gamma在1.0到1.5之间
            }

            // 创建该块的掩码
            cv::Mat mask = cv::Mat::zeros(Y.size(), CV_8UC1);
            mask(block_rect) = 255;

            // 高斯模糊掩码边缘，使过渡平滑
            cv::GaussianBlur(mask, mask, cv::Size(61, 61), 30);

            // 归一化掩码 (0-1)
            mask.convertTo(mask, CV_32F, 1.0 / 255.0);

            masks.push_back(mask);
            gammas.push_back(gamma);
        }
    }

    // 应用自适应gamma校正
    cv::Mat result = cv::Mat::zeros(Y.size(), CV_32F);
    cv::Mat Y_float;
    Y.convertTo(Y_float, CV_32F, 1.0 / 255.0);

    // 对每个块应用gamma校正并加权融合
    for (size_t k = 0; k < masks.size(); k++) {
        cv::Mat gamma_result;

        // 应用gamma校正
        cv::pow(Y_float, gammas[k], gamma_result);

        // 加权融合
        result += gamma_result.mul(masks[k]);
    }

    // 转回8位图像
    result = result * 255.0;
    result.convertTo(Y, CV_8UC1);

    // 合并通道
    cv::merge(channels, ycrcb);

    // 转回原始颜色空间
    if (src.channels() == 3) {
        cv::cvtColor(ycrcb, dst, cv::COLOR_YCrCb2BGR);
    } else {
        ycrcb.copyTo(dst);
    }
}

} // namespace advanced
} // namespace ip101