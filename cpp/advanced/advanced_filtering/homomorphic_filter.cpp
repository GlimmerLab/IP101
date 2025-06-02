#include "homomorphic_filter.hpp"
#include <vector>
#include <cmath>
#include <omp.h>

namespace ip101 {
namespace advanced {

cv::Mat create_homomorphic_filter(const cv::Size& size, double gamma_low, double gamma_high,
                                 double cutoff, double c) {
    CV_Assert(size.width > 0 && size.height > 0);
    CV_Assert(gamma_low >= 0 && gamma_high >= 0);
    CV_Assert(cutoff > 0 && c > 0);

    // 创建频域高斯高通滤波器
    cv::Mat filter = cv::Mat::zeros(size, CV_32F);

    // 计算中心点
    cv::Point2f center(static_cast<float>(size.width) / 2.0f, static_cast<float>(size.height) / 2.0f);

    // 计算D0^2
    double d0_squared = cutoff * cutoff;

    // 构建滤波器: (gamma_high - gamma_low) * (1 - exp(-c * (D^2 / D0^2))) + gamma_low
    for (int y = 0; y < size.height; y++) {
        for (int x = 0; x < size.width; x++) {
            // 计算到中心的距离平方
            double dx = x - center.x;
            double dy = y - center.y;
            double d_squared = dx * dx + dy * dy;

            // 计算滤波器值
            double h = (gamma_high - gamma_low) * (1.0 - std::exp(-c * d_squared / d0_squared)) + gamma_low;
            filter.at<float>(y, x) = static_cast<float>(h);
        }
    }

    return filter;
}

void dft_filter(const cv::Mat& src, cv::Mat& dst, const cv::Mat& filter) {
    CV_Assert(!src.empty() && !filter.empty());
    CV_Assert(src.type() == CV_32F);
    CV_Assert(filter.type() == CV_32F);

    // 优化DFT尺寸
    int m = cv::getOptimalDFTSize(src.rows);
    int n = cv::getOptimalDFTSize(src.cols);

    // 创建填充后的图像
    cv::Mat padded;
    cv::copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    // 为DFT创建复数图像
    cv::Mat planes[] = {padded, cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complex_img;
    cv::merge(planes, 2, complex_img);

    // 执行正向DFT
    cv::dft(complex_img, complex_img);

    // 应用滤波器 (频域乘法)
    cv::Mat filter_planes[] = {filter, filter}; // 实部和虚部使用相同的滤波器
    cv::Mat filter_complex;
    cv::merge(filter_planes, 2, filter_complex);

    // 进行频域滤波
    cv::mulSpectrums(complex_img, filter_complex, complex_img, 0);

    // 执行逆DFT
    cv::idft(complex_img, complex_img);

    // 分离结果的实部和虚部
    cv::split(complex_img, planes);

    // 实部包含结果
    dst = planes[0];

    // 将结果裁剪回原始大小
    dst = dst(cv::Rect(0, 0, src.cols, src.rows));
}

void visualize_spectrum(const cv::Mat& complex_img, cv::Mat& dst) {
    CV_Assert(!complex_img.empty() && complex_img.channels() == 2);

    // 分离实部和虚部
    cv::Mat planes[2];
    cv::split(complex_img, planes);

    // 计算幅度谱
    cv::Mat magnitude;
    cv::magnitude(planes[0], planes[1], magnitude);

    // 对数变换，扩展动态范围
    magnitude += 1.0; // 避免log(0)
    cv::log(magnitude, magnitude);

    // 归一化
    cv::normalize(magnitude, dst, 0, 255, cv::NORM_MINMAX);
    dst.convertTo(dst, CV_8U);
}

void homomorphic_filter(const cv::Mat& src, cv::Mat& dst, double gamma_low, double gamma_high,
                       double cutoff, double c) {
    CV_Assert(!src.empty());

    // 创建输出图像
    dst.create(src.size(), src.type());

    if (src.channels() == 1) {
        // 灰度图像处理

        // 1. 将图像转换为浮点型并加1(避免log(0))
        cv::Mat src_float;
        src.convertTo(src_float, CV_32F, 1.0, 1.0);

        // 2. 对数变换
        cv::Mat log_image;
        cv::log(src_float, log_image);

        // 3. 创建同态滤波器
        cv::Mat filter = create_homomorphic_filter(log_image.size(), gamma_low, gamma_high, cutoff, c);

        // 4. 应用DFT滤波
        cv::Mat filtered;
        dft_filter(log_image, filtered, filter);

        // 5. 指数变换
        cv::Mat exp_image;
        cv::exp(filtered, exp_image);

        // 6. 减去之前加的1
        exp_image -= 1.0;

        // 7. 归一化并转回8位图像
        double min_val, max_val;
        cv::minMaxLoc(exp_image, &min_val, &max_val);

        // 拉伸到原始范围
        cv::Mat dst_float;
        exp_image.convertTo(dst_float, CV_32F, 255.0 / (max_val - min_val), -min_val * 255.0 / (max_val - min_val));

        // 转回8位
        dst_float.convertTo(dst, CV_8U);

    } else if (src.channels() == 3) {
        // 彩色图像处理 - 转到YCrCb空间，只对Y通道做同态滤波

        // 转换颜色空间
        cv::Mat ycrcb;
        cv::cvtColor(src, ycrcb, cv::COLOR_BGR2YCrCb);

        // 分离通道
        std::vector<cv::Mat> channels;
        cv::split(ycrcb, channels);

        // 对亮度通道进行同态滤波
        cv::Mat y_filtered;
        homomorphic_filter(channels[0], y_filtered, gamma_low, gamma_high, cutoff, c);

        // 替换亮度通道
        channels[0] = y_filtered;

        // 合并通道
        cv::merge(channels, ycrcb);

        // 转回BGR
        cv::cvtColor(ycrcb, dst, cv::COLOR_YCrCb2BGR);
    }
}

void enhanced_homomorphic_filter(const cv::Mat& src, cv::Mat& dst, double gamma_low, double gamma_high,
                               double cutoff, double c, double alpha) {
    CV_Assert(!src.empty());
    CV_Assert(alpha >= 0.0 && alpha <= 1.0);

    // 1. 应用常规同态滤波
    cv::Mat homomorphic_result;
    homomorphic_filter(src, homomorphic_result, gamma_low, gamma_high, cutoff, c);

    if (alpha == 0.0) {
        // 如果alpha为0，直接返回常规同态滤波结果
        homomorphic_result.copyTo(dst);
        return;
    }

    // 2. 额外的边缘增强处理
    cv::Mat enhanced;

    if (src.channels() == 1) {
        // 灰度图像处理

        // 2.1 使用USM(非锐化掩模)增强边缘
        cv::Mat blurred;
        cv::GaussianBlur(homomorphic_result, blurred, cv::Size(0, 0), 3.0);

        // 计算锐化图像 = 原图 + alpha * (原图 - 模糊图)
        enhanced = homomorphic_result * (1.0 + alpha) - blurred * alpha;

        // 确保结果在有效范围内
        cv::normalize(enhanced, enhanced, 0, 255, cv::NORM_MINMAX);
        enhanced.convertTo(dst, CV_8U);

    } else {
        // 彩色图像处理

        // 转换到LAB色彩空间
        cv::Mat lab;
        cv::cvtColor(homomorphic_result, lab, cv::COLOR_BGR2Lab);

        // 分离通道
        std::vector<cv::Mat> lab_channels;
        cv::split(lab, lab_channels);

        // 对亮度通道进行边缘增强
        cv::Mat l_enhanced;
        cv::Mat blurred;
        cv::GaussianBlur(lab_channels[0], blurred, cv::Size(0, 0), 3.0);

        // 计算锐化亮度通道
        lab_channels[0] = lab_channels[0] * (1.0 + alpha) - blurred * alpha;

        // 合并通道
        cv::merge(lab_channels, lab);

        // 转回BGR
        cv::cvtColor(lab, dst, cv::COLOR_Lab2BGR);
    }
}

} // namespace advanced
} // namespace ip101