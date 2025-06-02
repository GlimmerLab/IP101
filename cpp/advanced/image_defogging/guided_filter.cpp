#include "guided_filter.hpp"
#include "dark_channel.hpp"
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace ip101 {
namespace advanced {

// 导向滤波的实现
void guided_filter(const cv::Mat& p, const cv::Mat& I, cv::Mat& q, int radius, double eps) {
    CV_Assert(!p.empty() && !I.empty());
    CV_Assert(p.size() == I.size() && p.type() == CV_64FC1);
    CV_Assert(I.type() == CV_64FC1 || I.type() == CV_64FC3);

    int h = I.rows;
    int w = I.cols;

    q.create(p.size(), p.type());

    // 均值滤波器
    cv::Mat mean_I, mean_p, mean_Ip, mean_II;

    if (I.channels() == 1) {
        // 单通道导向图像

        // 计算I的均值
        cv::boxFilter(I, mean_I, CV_64FC1, cv::Size(radius, radius), cv::Point(-1, -1), true);

        // 计算p的均值
        cv::boxFilter(p, mean_p, CV_64FC1, cv::Size(radius, radius), cv::Point(-1, -1), true);

        // 计算I*p的均值
        cv::Mat Ip = I.mul(p);
        cv::boxFilter(Ip, mean_Ip, CV_64FC1, cv::Size(radius, radius), cv::Point(-1, -1), true);

        // 计算I*I的均值
        cv::Mat II = I.mul(I);
        cv::boxFilter(II, mean_II, CV_64FC1, cv::Size(radius, radius), cv::Point(-1, -1), true);

        // 计算协方差
        cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);

        // 计算I的方差
        cv::Mat var_I = mean_II - mean_I.mul(mean_I);

        // 计算a, b
        cv::Mat a = cov_Ip / (var_I + eps);
        cv::Mat b = mean_p - a.mul(mean_I);

        // 计算a和b的均值
        cv::Mat mean_a, mean_b;
        cv::boxFilter(a, mean_a, CV_64FC1, cv::Size(radius, radius), cv::Point(-1, -1), true);
        cv::boxFilter(b, mean_b, CV_64FC1, cv::Size(radius, radius), cv::Point(-1, -1), true);

        // 计算输出q = mean_a * I + mean_b
        q = mean_a.mul(I) + mean_b;
    } else {
        // 三通道导向图像
        std::vector<cv::Mat> I_channels;
        cv::split(I, I_channels);

        // 为每个通道计算均值
        std::vector<cv::Mat> mean_I_channels(3);
        for (int i = 0; i < 3; i++) {
            cv::boxFilter(I_channels[i], mean_I_channels[i], CV_64FC1, cv::Size(radius, radius), cv::Point(-1, -1), true);
        }

        // 计算p的均值
        cv::boxFilter(p, mean_p, CV_64FC1, cv::Size(radius, radius), cv::Point(-1, -1), true);

        // 计算协方差矩阵
        std::vector<cv::Mat> Ip_channels(3);
        std::vector<cv::Mat> mean_Ip_channels(3);

        for (int i = 0; i < 3; i++) {
            Ip_channels[i] = I_channels[i].mul(p);
            cv::boxFilter(Ip_channels[i], mean_Ip_channels[i], CV_64FC1, cv::Size(radius, radius), cv::Point(-1, -1), true);
        }

        // 计算I的方差和协方差
        std::vector<cv::Mat> var_I_channels(6); // 对于3通道，我们有6个不同的方差/协方差项

        // 计算对角项 (方差)
        for (int i = 0; i < 3; i++) {
            var_I_channels[i] = cv::Mat(h, w, CV_64FC1);
            cv::Mat II = I_channels[i].mul(I_channels[i]);
            cv::Mat mean_II;
            cv::boxFilter(II, mean_II, CV_64FC1, cv::Size(radius, radius), cv::Point(-1, -1), true);
            var_I_channels[i] = mean_II - mean_I_channels[i].mul(mean_I_channels[i]); // Var(I)
        }

        // 计算非对角项 (协方差)
        int idx = 3;
        for (int i = 0; i < 3; i++) {
            for (int j = i + 1; j < 3; j++) {
                var_I_channels[idx] = cv::Mat(h, w, CV_64FC1);
                cv::Mat IJ = I_channels[i].mul(I_channels[j]);
                cv::Mat mean_IJ;
                cv::boxFilter(IJ, mean_IJ, CV_64FC1, cv::Size(radius, radius), cv::Point(-1, -1), true);
                var_I_channels[idx] = mean_IJ - mean_I_channels[i].mul(mean_I_channels[j]); // Cov(I_i, I_j)
                idx++;
            }
        }

        // 计算协方差 cov(I, p)
        std::vector<cv::Mat> cov_Ip_channels(3);
        for (int i = 0; i < 3; i++) {
            cov_Ip_channels[i] = mean_Ip_channels[i] - mean_I_channels[i].mul(mean_p);
        }

        // 对每个窗口求解线性系统
        std::vector<cv::Mat> a_channels(3);
        cv::Mat b = cv::Mat::zeros(h, w, CV_64FC1);

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                // 构建协方差矩阵
                cv::Mat Sigma(3, 3, CV_64FC1);
                Sigma.at<double>(0, 0) = var_I_channels[0].at<double>(y, x) + eps;
                Sigma.at<double>(1, 1) = var_I_channels[1].at<double>(y, x) + eps;
                Sigma.at<double>(2, 2) = var_I_channels[2].at<double>(y, x) + eps;
                Sigma.at<double>(0, 1) = var_I_channels[3].at<double>(y, x);
                Sigma.at<double>(1, 0) = var_I_channels[3].at<double>(y, x);
                Sigma.at<double>(0, 2) = var_I_channels[4].at<double>(y, x);
                Sigma.at<double>(2, 0) = var_I_channels[4].at<double>(y, x);
                Sigma.at<double>(1, 2) = var_I_channels[5].at<double>(y, x);
                Sigma.at<double>(2, 1) = var_I_channels[5].at<double>(y, x);

                // 构建协方差向量
                cv::Mat cov_vec(3, 1, CV_64FC1);
                cov_vec.at<double>(0, 0) = cov_Ip_channels[0].at<double>(y, x);
                cov_vec.at<double>(1, 0) = cov_Ip_channels[1].at<double>(y, x);
                cov_vec.at<double>(2, 0) = cov_Ip_channels[2].at<double>(y, x);

                // 求解线性系统 a = Sigma^(-1) * cov_vec
                cv::Mat a_vec = Sigma.inv() * cov_vec;

                if (a_channels.empty()) {
                    for (int i = 0; i < 3; i++) {
                        a_channels.push_back(cv::Mat::zeros(h, w, CV_64FC1));
                    }
                }

                // 保存a的值
                a_channels[0].at<double>(y, x) = a_vec.at<double>(0, 0);
                a_channels[1].at<double>(y, x) = a_vec.at<double>(1, 0);
                a_channels[2].at<double>(y, x) = a_vec.at<double>(2, 0);

                // 计算b
                double b_val = mean_p.at<double>(y, x) -
                    (a_vec.at<double>(0, 0) * mean_I_channels[0].at<double>(y, x) +
                     a_vec.at<double>(1, 0) * mean_I_channels[1].at<double>(y, x) +
                     a_vec.at<double>(2, 0) * mean_I_channels[2].at<double>(y, x));

                b.at<double>(y, x) = b_val;
            }
        }

        // 计算a和b的均值
        std::vector<cv::Mat> mean_a_channels(3);
        for (int i = 0; i < 3; i++) {
            cv::boxFilter(a_channels[i], mean_a_channels[i], CV_64FC1, cv::Size(radius, radius), cv::Point(-1, -1), true);
        }

        cv::Mat mean_b;
        cv::boxFilter(b, mean_b, CV_64FC1, cv::Size(radius, radius), cv::Point(-1, -1), true);

        // 计算输出q
        q = mean_b.clone();
        for (int i = 0; i < 3; i++) {
            q += mean_a_channels[i].mul(I_channels[i]);
        }
    }
}

void guided_filter_defogging(const cv::Mat& src, cv::Mat& dst, int radius, double eps, double omega, double t0) {
    CV_Assert(!src.empty() && src.channels() == 3);
    CV_Assert(radius > 0 && eps > 0 && omega > 0 && omega <= 1.0 && t0 > 0 && t0 < 1.0);

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 1. 计算暗通道
    cv::Mat dark;
    compute_dark_channel(src, dark, radius / 2); // 使用更小的半径计算暗通道

    // 2. 估计大气光照值
    cv::Vec3d A = estimate_atmospheric_light(src, dark);

    // 3. 估计粗透射率
    cv::Mat transmission = estimate_transmission(src, dark, A, omega);

    // 4. 使用导向滤波细化透射率
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    cv::Mat guidance, transmission_refined;
    gray.convertTo(guidance, CV_64FC1, 1.0 / 255.0);

    // 应用导向滤波
    guided_filter(transmission, guidance, transmission_refined, radius, eps);

    // 5. 限制最小透射率，避免过度去雾
    cv::max(transmission_refined, t0, transmission_refined);

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
                double t = transmission_refined.at<double>(y, x);

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

// 实现Kaiming He的改进算法
void kaiming_he_guided_defogging(const cv::Mat& src, cv::Mat& dst, int radius, double eps, double omega, double t0) {
    CV_Assert(!src.empty() && src.channels() == 3);
    CV_Assert(radius > 0 && eps > 0 && omega > 0 && omega <= 1.0 && t0 > 0 && t0 < 1.0);

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 1. 计算暗通道
    cv::Mat dark;
    compute_dark_channel(src, dark, 15); // 使用固定大小的patch计算暗通道

    // 2. 估计大气光照值
    cv::Vec3d A = estimate_atmospheric_light(src, dark);

    // 3. 估计粗透射率
    cv::Mat transmission = estimate_transmission(src, dark, A, omega);

    // 4. 使用引导图像滤波优化透射率 - 使用原始图像作为引导图像
    cv::Mat src_float;
    src.convertTo(src_float, CV_64FC3, 1.0 / 255.0);

    cv::Mat transmission_refined;
    guided_filter(transmission, src_float, transmission_refined, radius, eps);

    // 5. 限制最小透射率
    cv::max(transmission_refined, t0, transmission_refined);

    // 6. 应用雾效应成像模型恢复图像
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
                double t = transmission_refined.at<double>(y, x);

                // 恢复无雾图像
                recovered.at<double>(y, x) = (val - A[c]) / std::max(t, 0.1); // 使用较大的最小值防止过度放大
            }
        }

        // 裁剪到[0, 255]范围
        cv::Mat recovered_8u;
        recovered.convertTo(recovered_8u, CV_8UC1);
        recovered_channels[c] = recovered_8u;
    }

    // 合并通道
    cv::merge(recovered_channels, dst);

    // 执行额外的色彩校正 - 简单提升饱和度
    cv::Mat hsv;
    cv::cvtColor(dst, hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv, hsv_channels);

    // 适度提高饱和度
    hsv_channels[1] *= 1.2;

    cv::merge(hsv_channels, hsv);
    cv::cvtColor(hsv, dst, cv::COLOR_HSV2BGR);
}

} // namespace advanced
} // namespace ip101