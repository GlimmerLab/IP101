#include <advanced/correction/illumination_correction.hpp>
#include <vector>
#include <cmath>
#include <omp.h>

namespace ip101 {
namespace advanced {

void illumination_correction(const cv::Mat& src, cv::Mat& dst, const std::string& method) {
    CV_Assert(!src.empty());

    if (method == "homomorphic") {
        homomorphic_illumination_correction(src, dst);
    } else if (method == "background_subtraction") {
        background_subtraction_correction(src, dst);
    } else if (method == "multi_scale") {
        multi_scale_illumination_correction(src, dst);
    } else {
        throw std::invalid_argument("Unsupported illumination correction method: " + method);
    }
}

void homomorphic_illumination_correction(const cv::Mat& src, cv::Mat& dst,
                                        double gamma_low, double gamma_high, double cutoff) {
    CV_Assert(!src.empty());

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 转换为浮点数据类型
    cv::Mat log_img;
    if (src.channels() == 1) {
        // 灰度图像
        src.convertTo(log_img, CV_32F, 1.0, 1.0); // 加1避免log(0)
        cv::log(log_img, log_img);

        // 进行DFT
        cv::Mat padded;
        int m = cv::getOptimalDFTSize(log_img.rows);
        int n = cv::getOptimalDFTSize(log_img.cols);
        cv::copyMakeBorder(log_img, padded, 0, m - log_img.rows, 0, n - log_img.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

        cv::Mat planes[] = {padded, cv::Mat::zeros(padded.size(), CV_32F)};
        cv::Mat complex_img;
        cv::merge(planes, 2, complex_img);

        cv::dft(complex_img, complex_img);

        // 分离实部和虚部以进行滤波
        cv::split(complex_img, planes);

        // 构建滤波器
        cv::Mat filter = cv::Mat(complex_img.size(), CV_32F);
        double d0_squared = cutoff * cutoff;
        cv::Point center(filter.cols / 2, filter.rows / 2);

        for (int y = 0; y < filter.rows; y++) {
            for (int x = 0; x < filter.cols; x++) {
                // 计算到中心的距离平方
                double d_squared = std::pow(y - center.y, 2) + std::pow(x - center.x, 2);

                // 构建高斯同态滤波器
                filter.at<float>(y, x) = (gamma_high - gamma_low) * (1 - std::exp(-d_squared / (2 * d0_squared))) + gamma_low;
            }
        }

        // 重新排列滤波器，使DC位于角落
        cv::Mat tmp = filter.clone();
        int cx = filter.cols / 2;
        int cy = filter.rows / 2;

        // 重新排列四个象限
        cv::Mat q0(tmp, cv::Rect(0, 0, cx, cy));     // 左上
        cv::Mat q1(tmp, cv::Rect(cx, 0, cx, cy));    // 右上
        cv::Mat q2(tmp, cv::Rect(0, cy, cx, cy));    // 左下
        cv::Mat q3(tmp, cv::Rect(cx, cy, cx, cy));   // 右下

        cv::Mat tmp0, tmp1, tmp2, tmp3;
        q0.copyTo(tmp0);
        q1.copyTo(tmp1);
        q2.copyTo(tmp2);
        q3.copyTo(tmp3);

        tmp3.copyTo(cv::Mat(filter, cv::Rect(0, 0, cx, cy)));        // 右下 -> 左上
        tmp2.copyTo(cv::Mat(filter, cv::Rect(cx, 0, cx, cy)));       // 左下 -> 右上
        tmp1.copyTo(cv::Mat(filter, cv::Rect(0, cy, cx, cy)));       // 右上 -> 左下
        tmp0.copyTo(cv::Mat(filter, cv::Rect(cx, cy, cx, cy)));      // 左上 -> 右下

        // 应用滤波器
        cv::Mat filtered_planes[2];
        filtered_planes[0] = planes[0].mul(filter);
        filtered_planes[1] = planes[1].mul(filter);

        // 合并实部和虚部
        cv::merge(filtered_planes, 2, complex_img);

        // 执行逆DFT
        cv::idft(complex_img, complex_img);

        // 分离实部
        cv::split(complex_img, planes);

        // 指数恢复
        cv::exp(planes[0], planes[0]);
        planes[0] -= 1.0; // 减去之前加的1

        // 规范化并转回8位图像
        cv::normalize(planes[0], planes[0], 0, 255, cv::NORM_MINMAX);
        planes[0].convertTo(dst, CV_8U);
    } else {
        // 彩色图像，分离通道处理
        std::vector<cv::Mat> channels;
        cv::split(src, channels);

        std::vector<cv::Mat> output_channels(channels.size());

        #pragma omp parallel for
        for (int i = 0; i < channels.size(); i++) {
            homomorphic_illumination_correction(channels[i], output_channels[i],
                                              gamma_low, gamma_high, cutoff);
        }

        cv::merge(output_channels, dst);
    }
}

void background_subtraction_correction(const cv::Mat& src, cv::Mat& dst,
                                      int kernel_size, double resize_factor) {
    CV_Assert(!src.empty());
    CV_Assert(kernel_size > 0 && kernel_size % 2 == 1); // 必须是正奇数
    CV_Assert(resize_factor > 0.0 && resize_factor <= 1.0);

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 对于彩色图像，仅处理亮度通道
    if (src.channels() == 3) {
        cv::Mat ycrcb;
        cv::cvtColor(src, ycrcb, cv::COLOR_BGR2YCrCb);

        std::vector<cv::Mat> channels;
        cv::split(ycrcb, channels);

        // 仅对亮度通道进行光照校正
        cv::Mat y_corrected;
        background_subtraction_correction(channels[0], y_corrected, kernel_size, resize_factor);

        // 替换亮度通道
        channels[0] = y_corrected;

        // 合并通道
        cv::merge(channels, ycrcb);

        // 转回BGR
        cv::cvtColor(ycrcb, dst, cv::COLOR_YCrCb2BGR);
        return;
    }

    // 转换为浮点型
    cv::Mat float_img;
    src.convertTo(float_img, CV_32F);

    // 调整大小以加速处理
    cv::Mat resized;
    cv::Size new_size(static_cast<int>(float_img.cols * resize_factor),
                     static_cast<int>(float_img.rows * resize_factor));
    cv::resize(float_img, resized, new_size);

    // 应用大核高斯模糊来估计背景照明
    cv::Mat background;
    int adjusted_kernel_size = static_cast<int>(kernel_size * resize_factor);
    if (adjusted_kernel_size % 2 == 0) {
        adjusted_kernel_size++; // 确保是奇数
    }
    cv::GaussianBlur(resized, background, cv::Size(adjusted_kernel_size, adjusted_kernel_size), 0);

    // 恢复原始尺寸
    cv::resize(background, background, float_img.size());

    // 背景减除
    cv::Mat result = float_img.clone();
    for (int y = 0; y < result.rows; y++) {
        for (int x = 0; x < result.cols; x++) {
            float correction = 128.0f / (background.at<float>(y, x) + 1e-6f); // 防止除零
            result.at<float>(y, x) *= correction;
        }
    }

    // 规范化并转回8位图像
    cv::normalize(result, result, 0, 255, cv::NORM_MINMAX);
    result.convertTo(dst, CV_8U);
}

void multi_scale_illumination_correction(const cv::Mat& src, cv::Mat& dst,
                                        const std::vector<double>& sigma_list) {
    CV_Assert(!src.empty());
    CV_Assert(!sigma_list.empty());

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 转换为32位浮点数
    cv::Mat float_img;
    src.convertTo(float_img, CV_32F);
    float_img += 1.0f; // 避免log(0)

    if (src.channels() == 1) {
        // 灰度图像
        cv::Mat log_img;
        cv::log(float_img, log_img);

        // 多尺度处理
        cv::Mat msr = cv::Mat::zeros(log_img.size(), CV_32F);

        for (double sigma : sigma_list) {
            cv::Mat gaussian_img;
            cv::GaussianBlur(log_img, gaussian_img, cv::Size(0, 0), sigma);
            msr += log_img - gaussian_img;
        }

        // 取平均
        msr /= sigma_list.size();

        // 恢复
        cv::Mat exp_img;
        cv::exp(msr, exp_img);
        exp_img -= 1.0f; // 减去之前加的1

        // 规范化并转回8位图像
        cv::normalize(exp_img, exp_img, 0, 255, cv::NORM_MINMAX);
        exp_img.convertTo(dst, CV_8U);
    } else {
        // 彩色图像处理
        std::vector<cv::Mat> channels;
        cv::split(float_img, channels);

        std::vector<cv::Mat> log_channels(channels.size());
        std::vector<cv::Mat> msr_channels(channels.size());

        // 对每个通道进行log变换
        for (int i = 0; i < channels.size(); i++) {
            cv::log(channels[i], log_channels[i]);
            msr_channels[i] = cv::Mat::zeros(log_channels[i].size(), CV_32F);
        }

        // 多尺度处理
        for (double sigma : sigma_list) {
            for (int i = 0; i < channels.size(); i++) {
                cv::Mat gaussian_img;
                cv::GaussianBlur(log_channels[i], gaussian_img, cv::Size(0, 0), sigma);
                msr_channels[i] += log_channels[i] - gaussian_img;
            }
        }

        // 取平均并恢复
        std::vector<cv::Mat> result_channels(channels.size());

        for (int i = 0; i < channels.size(); i++) {
            msr_channels[i] /= sigma_list.size();
            cv::exp(msr_channels[i], result_channels[i]);
            result_channels[i] -= 1.0f; // 减去之前加的1

            // 规范化
            cv::normalize(result_channels[i], result_channels[i], 0, 255, cv::NORM_MINMAX);
            result_channels[i].convertTo(result_channels[i], CV_8U);
        }

        // 合并通道
        cv::merge(result_channels, dst);
    }
}

} // namespace advanced
} // namespace ip101