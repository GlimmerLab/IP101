#include <advanced/filtering/guided_filter.hpp>
#include <vector>
#include <cmath>
#include <omp.h>

namespace ip101 {
namespace advanced {

void guided_filter(const cv::Mat& p, const cv::Mat& I, cv::Mat& q, int radius, double eps) {
    CV_Assert(!p.empty() && !I.empty());
    CV_Assert(p.size() == I.size());
    CV_Assert(radius > 0 && eps > 0);

    // 输入图像类型转换到浮点型
    cv::Mat p_float, I_float;
    p.convertTo(p_float, CV_64FC1);

    if (I.type() == CV_8UC3) {
        I.convertTo(I_float, CV_64FC3, 1.0/255.0);
    } else if (I.type() == CV_8UC1) {
        I.convertTo(I_float, CV_64FC1, 1.0/255.0);
    } else {
        I_float = I.clone();
    }

    int h = p.rows;
    int w = p.cols;

    // 输出图像初始化
    q.create(p.size(), p_float.type());

    // 计算窗口像素数
    int win_size = (2 * radius + 1) * (2 * radius + 1);

    if (I_float.channels() == 1) {
        // 单通道引导图像

        // 均值滤波器
        cv::Mat mean_I, mean_p, mean_Ip, mean_II;

        // 使用盒式滤波器计算局部均值
        cv::boxFilter(I_float, mean_I, CV_64FC1, cv::Size(2*radius+1, 2*radius+1), cv::Point(-1,-1), true);
        cv::boxFilter(p_float, mean_p, CV_64FC1, cv::Size(2*radius+1, 2*radius+1), cv::Point(-1,-1), true);

        // 计算I*p和I*I的均值
        cv::Mat Ip = I_float.mul(p_float);
        cv::boxFilter(Ip, mean_Ip, CV_64FC1, cv::Size(2*radius+1, 2*radius+1), cv::Point(-1,-1), true);

        cv::Mat II = I_float.mul(I_float);
        cv::boxFilter(II, mean_II, CV_64FC1, cv::Size(2*radius+1, 2*radius+1), cv::Point(-1,-1), true);

        // 计算局部线性系数a和b
        cv::Mat var_I = mean_II - mean_I.mul(mean_I);  // 方差
        cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p); // 协方差

        cv::Mat a = cov_Ip / (var_I + eps);
        cv::Mat b = mean_p - a.mul(mean_I);

        // 对系数a和b进行滤波
        cv::Mat mean_a, mean_b;
        cv::boxFilter(a, mean_a, CV_64FC1, cv::Size(2*radius+1, 2*radius+1), cv::Point(-1,-1), true);
        cv::boxFilter(b, mean_b, CV_64FC1, cv::Size(2*radius+1, 2*radius+1), cv::Point(-1,-1), true);

        // 生成输出图像
        q = mean_a.mul(I_float) + mean_b;

    } else {
        // 三通道引导图像
        std::vector<cv::Mat> I_channels;
        cv::split(I_float, I_channels);

        // 存储均值
        cv::Mat mean_I_r, mean_I_g, mean_I_b, mean_p;
        cv::boxFilter(I_channels[0], mean_I_r, CV_64FC1, cv::Size(2*radius+1, 2*radius+1), cv::Point(-1,-1), true);
        cv::boxFilter(I_channels[1], mean_I_g, CV_64FC1, cv::Size(2*radius+1, 2*radius+1), cv::Point(-1,-1), true);
        cv::boxFilter(I_channels[2], mean_I_b, CV_64FC1, cv::Size(2*radius+1, 2*radius+1), cv::Point(-1,-1), true);
        cv::boxFilter(p_float, mean_p, CV_64FC1, cv::Size(2*radius+1, 2*radius+1), cv::Point(-1,-1), true);

        // 计算各种协方差
        cv::Mat I_r_p = I_channels[0].mul(p_float);
        cv::Mat I_g_p = I_channels[1].mul(p_float);
        cv::Mat I_b_p = I_channels[2].mul(p_float);

        cv::Mat mean_I_r_p, mean_I_g_p, mean_I_b_p;
        cv::boxFilter(I_r_p, mean_I_r_p, CV_64FC1, cv::Size(2*radius+1, 2*radius+1), cv::Point(-1,-1), true);
        cv::boxFilter(I_g_p, mean_I_g_p, CV_64FC1, cv::Size(2*radius+1, 2*radius+1), cv::Point(-1,-1), true);
        cv::boxFilter(I_b_p, mean_I_b_p, CV_64FC1, cv::Size(2*radius+1, 2*radius+1), cv::Point(-1,-1), true);

        // 计算I各通道间的协方差
        cv::Mat I_rr = I_channels[0].mul(I_channels[0]);
        cv::Mat I_rg = I_channels[0].mul(I_channels[1]);
        cv::Mat I_rb = I_channels[0].mul(I_channels[2]);
        cv::Mat I_gg = I_channels[1].mul(I_channels[1]);
        cv::Mat I_gb = I_channels[1].mul(I_channels[2]);
        cv::Mat I_bb = I_channels[2].mul(I_channels[2]);

        cv::Mat mean_I_rr, mean_I_rg, mean_I_rb, mean_I_gg, mean_I_gb, mean_I_bb;
        cv::boxFilter(I_rr, mean_I_rr, CV_64FC1, cv::Size(2*radius+1, 2*radius+1), cv::Point(-1,-1), true);
        cv::boxFilter(I_rg, mean_I_rg, CV_64FC1, cv::Size(2*radius+1, 2*radius+1), cv::Point(-1,-1), true);
        cv::boxFilter(I_rb, mean_I_rb, CV_64FC1, cv::Size(2*radius+1, 2*radius+1), cv::Point(-1,-1), true);
        cv::boxFilter(I_gg, mean_I_gg, CV_64FC1, cv::Size(2*radius+1, 2*radius+1), cv::Point(-1,-1), true);
        cv::boxFilter(I_gb, mean_I_gb, CV_64FC1, cv::Size(2*radius+1, 2*radius+1), cv::Point(-1,-1), true);
        cv::boxFilter(I_bb, mean_I_bb, CV_64FC1, cv::Size(2*radius+1, 2*radius+1), cv::Point(-1,-1), true);

        // 求解线性方程组，获取系数a和b
        cv::Mat a_r = cv::Mat::zeros(p.size(), CV_64FC1);
        cv::Mat a_g = cv::Mat::zeros(p.size(), CV_64FC1);
        cv::Mat a_b = cv::Mat::zeros(p.size(), CV_64FC1);
        cv::Mat b = cv::Mat::zeros(p.size(), CV_64FC1);

        #pragma omp parallel for
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                // 构建协方差矩阵Sigma
                cv::Matx33d Sigma;
                Sigma(0,0) = mean_I_rr.at<double>(y,x) - mean_I_r.at<double>(y,x) * mean_I_r.at<double>(y,x) + eps;
                Sigma(0,1) = mean_I_rg.at<double>(y,x) - mean_I_r.at<double>(y,x) * mean_I_g.at<double>(y,x);
                Sigma(0,2) = mean_I_rb.at<double>(y,x) - mean_I_r.at<double>(y,x) * mean_I_b.at<double>(y,x);
                Sigma(1,0) = Sigma(0,1);
                Sigma(1,1) = mean_I_gg.at<double>(y,x) - mean_I_g.at<double>(y,x) * mean_I_g.at<double>(y,x) + eps;
                Sigma(1,2) = mean_I_gb.at<double>(y,x) - mean_I_g.at<double>(y,x) * mean_I_b.at<double>(y,x);
                Sigma(2,0) = Sigma(0,2);
                Sigma(2,1) = Sigma(1,2);
                Sigma(2,2) = mean_I_bb.at<double>(y,x) - mean_I_b.at<double>(y,x) * mean_I_b.at<double>(y,x) + eps;

                // 构建向量cov_Ip
                cv::Matx31d cov_Ip;
                cov_Ip(0,0) = mean_I_r_p.at<double>(y,x) - mean_I_r.at<double>(y,x) * mean_p.at<double>(y,x);
                cov_Ip(1,0) = mean_I_g_p.at<double>(y,x) - mean_I_g.at<double>(y,x) * mean_p.at<double>(y,x);
                cov_Ip(2,0) = mean_I_b_p.at<double>(y,x) - mean_I_b.at<double>(y,x) * mean_p.at<double>(y,x);

                // 解线性方程组
                cv::Matx31d a = Sigma.solve(cov_Ip);

                a_r.at<double>(y,x) = a(0,0);
                a_g.at<double>(y,x) = a(1,0);
                a_b.at<double>(y,x) = a(2,0);

                b.at<double>(y,x) = mean_p.at<double>(y,x)
                                    - a(0,0) * mean_I_r.at<double>(y,x)
                                    - a(1,0) * mean_I_g.at<double>(y,x)
                                    - a(2,0) * mean_I_b.at<double>(y,x);
            }
        }

        // 对系数a和b进行滤波
        cv::Mat mean_a_r, mean_a_g, mean_a_b, mean_b;
        cv::boxFilter(a_r, mean_a_r, CV_64FC1, cv::Size(2*radius+1, 2*radius+1), cv::Point(-1,-1), true);
        cv::boxFilter(a_g, mean_a_g, CV_64FC1, cv::Size(2*radius+1, 2*radius+1), cv::Point(-1,-1), true);
        cv::boxFilter(a_b, mean_a_b, CV_64FC1, cv::Size(2*radius+1, 2*radius+1), cv::Point(-1,-1), true);
        cv::boxFilter(b, mean_b, CV_64FC1, cv::Size(2*radius+1, 2*radius+1), cv::Point(-1,-1), true);

        // 生成输出图像 q = a_r*I_r + a_g*I_g + a_b*I_b + b
        q = mean_a_r.mul(I_channels[0]) + mean_a_g.mul(I_channels[1])
            + mean_a_b.mul(I_channels[2]) + mean_b;
    }
}

void fast_guided_filter(const cv::Mat& p, const cv::Mat& I, cv::Mat& q,
                        int radius, double eps, int s) {
    CV_Assert(!p.empty() && !I.empty());
    CV_Assert(p.size() == I.size());
    CV_Assert(radius > 0 && eps > 0 && s > 0);

    // 下采样
    cv::Mat I_sub, p_sub;
    cv::resize(I, I_sub, cv::Size(), 1.0/s, 1.0/s, cv::INTER_LINEAR);
    cv::resize(p, p_sub, cv::Size(), 1.0/s, 1.0/s, cv::INTER_LINEAR);

    // 在下采样图像上应用导向滤波
    cv::Mat q_sub;
    int r_sub = radius / s;
    if (r_sub < 1) r_sub = 1;

    guided_filter(p_sub, I_sub, q_sub, r_sub, eps);

    // 上采样回原始分辨率
    cv::resize(q_sub, q, p.size(), 0, 0, cv::INTER_LINEAR);
}

void edge_aware_guided_filter(const cv::Mat& p, const cv::Mat& I, cv::Mat& q,
                            int radius, double eps, double edge_aware_factor) {
    CV_Assert(!p.empty() && !I.empty());
    CV_Assert(p.size() == I.size());
    CV_Assert(radius > 0 && eps > 0 && edge_aware_factor > 0);

    // 输入图像类型转换到浮点型
    cv::Mat p_float, I_float;
    p.convertTo(p_float, CV_64FC1);

    if (I.type() == CV_8UC3) {
        I.convertTo(I_float, CV_64FC3, 1.0/255.0);
    } else if (I.type() == CV_8UC1) {
        I.convertTo(I_float, CV_64FC1, 1.0/255.0);
    } else {
        I_float = I.clone();
    }

    // 计算引导图像的梯度幅值
    cv::Mat gray;
    if (I_float.channels() == 3) {
        cv::cvtColor(I_float, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = I_float.clone();
    }

    cv::Mat gradient_x, gradient_y, gradient_magnitude;
    cv::Sobel(gray, gradient_x, CV_64F, 1, 0, 3);
    cv::Sobel(gray, gradient_y, CV_64F, 0, 1, 3);
    cv::magnitude(gradient_x, gradient_y, gradient_magnitude);

    // 根据梯度幅值自适应调整eps
    cv::Mat adaptive_eps = cv::Mat::zeros(p.size(), CV_64FC1);

    double max_grad;
    cv::minMaxLoc(gradient_magnitude, nullptr, &max_grad);

    #pragma omp parallel for
    for (int y = 0; y < p.rows; y++) {
        for (int x = 0; x < p.cols; x++) {
            double grad = gradient_magnitude.at<double>(y, x);
            double factor = 1.0 + edge_aware_factor * (grad / max_grad);
            adaptive_eps.at<double>(y, x) = eps * factor * factor;
        }
    }

    // 应用空间变化的eps导向滤波
    // 由于标准导向滤波不支持空间变化的eps，我们分块处理
    int block_size = 50; // 50x50的块
    q = cv::Mat::zeros(p.size(), p_float.type());

    #pragma omp parallel for
    for (int y = 0; y < p.rows; y += block_size) {
        for (int x = 0; x < p.cols; x += block_size) {
            // 确定当前块的大小
            int block_h = std::min(block_size, p.rows - y);
            int block_w = std::min(block_size, p.cols - x);

            // 提取当前块
            cv::Rect roi(x, y, block_w, block_h);
            cv::Mat p_block = p_float(roi);
            cv::Mat I_block = I_float(roi);

            // 获取该块的eps平均值
            cv::Scalar mean_eps = cv::mean(adaptive_eps(roi));

            // 使用该eps值对块进行导向滤波
            cv::Mat q_block;
            guided_filter(p_block, I_block, q_block, radius, mean_eps[0]);

            // 将结果复制到输出图像
            q_block.copyTo(q(roi));
        }
    }
}

void joint_bilateral_filter(const cv::Mat& p, const cv::Mat& I, cv::Mat& q,
                           int radius, double sigma_space, double sigma_range) {
    CV_Assert(!p.empty() && !I.empty());
    CV_Assert(p.size() == I.size());
    CV_Assert(radius > 0 && sigma_space > 0 && sigma_range > 0);

    // 输入图像类型转换到浮点型
    cv::Mat p_float, I_float;
    p.convertTo(p_float, CV_64FC1);

    if (I.type() == CV_8UC3) {
        I.convertTo(I_float, CV_64FC3, 1.0/255.0);
    } else if (I.type() == CV_8UC1) {
        I.convertTo(I_float, CV_64FC1, 1.0/255.0);
    } else {
        I_float = I.clone();
    }

    // 初始化输出
    q.create(p.size(), p_float.type());
    q = cv::Mat::zeros(p.size(), p_float.type());

    // 预计算空间权重（高斯核）
    cv::Mat spatial_weight = cv::Mat::zeros(2*radius+1, 2*radius+1, CV_64FC1);
    double sigma_space2_inv = 1.0 / (2.0 * sigma_space * sigma_space);

    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            double dist2 = i*i + j*j;
            int idx_i = i + radius;
            int idx_j = j + radius;
            spatial_weight.at<double>(idx_i, idx_j) = std::exp(-dist2 * sigma_space2_inv);
        }
    }

    // 预计算sigma_range的平方的倒数
    double sigma_range2_inv = 1.0 / (2.0 * sigma_range * sigma_range);

    // 手动实现联合双边滤波
    int h = p.rows;
    int w = p.cols;

    if (I_float.channels() == 1) {
        // 单通道引导图像
        #pragma omp parallel for
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double center_intensity = I_float.at<double>(y, x);
                double sum = 0.0;
                double weight_sum = 0.0;

                // 遍历窗口内的像素
                for (int i = -radius; i <= radius; i++) {
                    int yi = y + i;
                    if (yi < 0 || yi >= h) continue;

                    for (int j = -radius; j <= radius; j++) {
                        int xj = x + j;
                        if (xj < 0 || xj >= w) continue;

                        // 计算空间权重
                        double s_weight = spatial_weight.at<double>(i+radius, j+radius);

                        // 计算范围权重
                        double neighbor_intensity = I_float.at<double>(yi, xj);
                        double intensity_diff = center_intensity - neighbor_intensity;
                        double r_weight = std::exp(-intensity_diff * intensity_diff * sigma_range2_inv);

                        // 联合权重
                        double weight = s_weight * r_weight;

                        // 累加权重和加权值
                        sum += weight * p_float.at<double>(yi, xj);
                        weight_sum += weight;
                    }
                }

                // 归一化
                if (weight_sum > 0.0) {
                    q.at<double>(y, x) = sum / weight_sum;
                } else {
                    q.at<double>(y, x) = p_float.at<double>(y, x);
                }
            }
        }
    } else {
        // 三通道引导图像
        #pragma omp parallel for
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                cv::Vec3d center_intensity = I_float.at<cv::Vec3d>(y, x);
                double sum = 0.0;
                double weight_sum = 0.0;

                // 遍历窗口内的像素
                for (int i = -radius; i <= radius; i++) {
                    int yi = y + i;
                    if (yi < 0 || yi >= h) continue;

                    for (int j = -radius; j <= radius; j++) {
                        int xj = x + j;
                        if (xj < 0 || xj >= w) continue;

                        // 计算空间权重
                        double s_weight = spatial_weight.at<double>(i+radius, j+radius);

                        // 计算范围权重
                        cv::Vec3d neighbor_intensity = I_float.at<cv::Vec3d>(yi, xj);
                        cv::Vec3d intensity_diff = center_intensity - neighbor_intensity;
                        double r_weight = std::exp(-(intensity_diff[0]*intensity_diff[0] +
                                                   intensity_diff[1]*intensity_diff[1] +
                                                   intensity_diff[2]*intensity_diff[2]) * sigma_range2_inv);

                        // 联合权重
                        double weight = s_weight * r_weight;

                        // 累加权重和加权值
                        sum += weight * p_float.at<double>(yi, xj);
                        weight_sum += weight;
                    }
                }

                // 归一化
                if (weight_sum > 0.0) {
                    q.at<double>(y, x) = sum / weight_sum;
                } else {
                    q.at<double>(y, x) = p_float.at<double>(y, x);
                }
            }
        }
    }
}

} // namespace advanced
} // namespace ip101