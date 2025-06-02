#include "backlight.hpp"
#include <vector>
#include <cmath>
#include <omp.h>

namespace ip101 {
namespace advanced {

void inrbl_backlight_correction(const cv::Mat& src, cv::Mat& dst, double gamma, double lambda) {
    CV_Assert(!src.empty());
    CV_Assert(gamma > 0.0 && lambda > 0.0);

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 转换为YUV颜色空间，仅对Y通道进行处理
    cv::Mat yuv;
    if (src.channels() == 3) {
        cv::cvtColor(src, yuv, cv::COLOR_BGR2YUV);
    } else {
        yuv = src.clone();
    }

    std::vector<cv::Mat> channels;
    cv::split(yuv, channels);

    // 提取Y通道
    cv::Mat& Y = channels[0];

    // 计算亮度反比
    cv::Mat inv_Y;
    cv::subtract(cv::Scalar::all(255), Y, inv_Y);

    // 计算gamma校正后的亮度
    cv::Mat gamma_Y;
    Y.convertTo(gamma_Y, CV_32F, 1.0/255.0);

    // 手动实现gamma校正
    for (int y = 0; y < gamma_Y.rows; y++) {
        for (int x = 0; x < gamma_Y.cols; x++) {
            gamma_Y.at<float>(y, x) = std::pow(gamma_Y.at<float>(y, x), gamma);
        }
    }

    gamma_Y.convertTo(gamma_Y, CV_8U, 255.0);

    // 计算增强Y通道
    cv::Mat enhanced_Y = Y.clone();

    // 应用INRBL算法
    for (int y = 0; y < Y.rows; y++) {
        for (int x = 0; x < Y.cols; x++) {
            // 计算增强系数
            float alpha = static_cast<float>(inv_Y.at<uchar>(y, x)) / 255.0f;
            alpha = lambda * alpha + (1.0f - lambda) * 0.5f; // 调整增强系数的范围

            // 应用增强
            int val = cv::saturate_cast<uchar>(Y.at<uchar>(y, x) * (1.0f - alpha) + gamma_Y.at<uchar>(y, x) * alpha);
            enhanced_Y.at<uchar>(y, x) = val;
        }
    }

    // 更新Y通道
    channels[0] = enhanced_Y;

    // 合并通道
    cv::merge(channels, yuv);

    // 转回原始颜色空间
    if (src.channels() == 3) {
        cv::cvtColor(yuv, dst, cv::COLOR_YUV2BGR);
    } else {
        yuv.copyTo(dst);
    }
}

// 自定义CLAHE实现
void apply_clahe(const cv::Mat& src, cv::Mat& dst, double clip_limit, const cv::Size& grid_size) {
    CV_Assert(!src.empty() && src.type() == CV_8UC1);

    // 为了简单起见，我们使用OpenCV的CLAHE实现
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(clip_limit);
    clahe->setTilesGridSize(grid_size);
    clahe->apply(src, dst);
}

void adaptive_backlight_correction(const cv::Mat& src, cv::Mat& dst, double clip_limit, const cv::Size& grid_size) {
    CV_Assert(!src.empty());

    // 创建输出图像
    dst.create(src.size(), src.type());

    if (src.channels() == 1) {
        // 直接应用CLAHE
        apply_clahe(src, dst, clip_limit, grid_size);
    } else {
        // 转换到LAB颜色空间
        cv::Mat lab;
        cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);

        // 分离通道
        std::vector<cv::Mat> lab_channels;
        cv::split(lab, lab_channels);

        // 仅对亮度通道应用CLAHE
        apply_clahe(lab_channels[0], lab_channels[0], clip_limit, grid_size);

        // 合并通道
        cv::merge(lab_channels, lab);

        // 转回BGR
        cv::cvtColor(lab, dst, cv::COLOR_Lab2BGR);
    }

    // 额外的色彩增强
    cv::Mat hsv;
    cv::cvtColor(dst, hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv, hsv_channels);

    // 略微增加饱和度
    hsv_channels[1] *= 1.2;

    cv::merge(hsv_channels, hsv);
    cv::cvtColor(hsv, dst, cv::COLOR_HSV2BGR);
}

void exposure_fusion_backlight_correction(const cv::Mat& src, cv::Mat& dst, int num_exposures) {
    CV_Assert(!src.empty());
    CV_Assert(num_exposures >= 2);

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 生成不同曝光补偿的图像
    std::vector<cv::Mat> exposures;
    exposures.reserve(num_exposures);

    double gamma_range = 1.5;  // 伽马范围，调整曝光

    for (int i = 0; i < num_exposures; i++) {
        double gamma = 0.5 + i * (gamma_range / (num_exposures - 1));

        cv::Mat exposure = cv::Mat::zeros(src.size(), src.type());

        // 伽马校正生成不同曝光
        src.convertTo(exposure, CV_32F, 1.0/255.0);

        #pragma omp parallel for
        for (int y = 0; y < exposure.rows; y++) {
            for (int x = 0; x < exposure.cols; x++) {
                if (src.channels() == 1) {
                    float val = exposure.at<float>(y, x);
                    exposure.at<float>(y, x) = std::pow(val, gamma);
                } else {
                    cv::Vec3f pixel = exposure.at<cv::Vec3f>(y, x);
                    for (int c = 0; c < 3; c++) {
                        pixel[c] = std::pow(pixel[c], gamma);
                    }
                    exposure.at<cv::Vec3f>(y, x) = pixel;
                }
            }
        }

        exposure.convertTo(exposure, CV_8U, 255.0);
        exposures.push_back(exposure);
    }

    // 计算每个曝光图像的权重
    std::vector<cv::Mat> weights;
    weights.reserve(num_exposures);

    for (int i = 0; i < num_exposures; i++) {
        cv::Mat exposure = exposures[i];

        // 计算对比度权重
        cv::Mat contrast;
        cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0, -1, 4, -1, 0, -1, 0);
        if (exposure.channels() == 1) {
            cv::filter2D(exposure, contrast, CV_32F, kernel);
            cv::convertScaleAbs(contrast, contrast);
        } else {
            std::vector<cv::Mat> channels;
            cv::split(exposure, channels);

            std::vector<cv::Mat> contrast_channels;
            for (int c = 0; c < channels.size(); c++) {
                cv::Mat temp;
                cv::filter2D(channels[c], temp, CV_32F, kernel);
                cv::convertScaleAbs(temp, temp);
                contrast_channels.push_back(temp);
            }

            cv::merge(contrast_channels, contrast);
        }

        // 计算饱和度权重
        cv::Mat saturation;
        if (exposure.channels() == 3) {
            std::vector<cv::Mat> channels;
            cv::split(exposure, channels);

            cv::Mat mean = (channels[0] + channels[1] + channels[2]) / 3.0;

            cv::Mat variance = cv::Mat::zeros(exposure.size(), CV_32F);
            for (int c = 0; c < 3; c++) {
                cv::Mat diff;
                cv::absdiff(channels[c], mean, diff);
                diff.convertTo(diff, CV_32F);
                variance += diff;
            }

            variance /= 3.0;
            cv::convertScaleAbs(variance, saturation);
        } else {
            saturation = cv::Mat::ones(exposure.size(), CV_8U) * 255;
        }

        // 计算适当曝光权重
        cv::Mat well_exposed;
        cv::Mat gauss = cv::Mat::zeros(exposure.size(), CV_32F);

        if (exposure.channels() == 1) {
            exposure.convertTo(gauss, CV_32F, 1.0/255.0);

            for (int y = 0; y < gauss.rows; y++) {
                for (int x = 0; x < gauss.cols; x++) {
                    float val = gauss.at<float>(y, x);
                    gauss.at<float>(y, x) = std::exp(-12.5f * std::pow(val - 0.5f, 2));
                }
            }
        } else {
            exposure.convertTo(gauss, CV_32F, 1.0/255.0);

            for (int y = 0; y < gauss.rows; y++) {
                for (int x = 0; x < gauss.cols; x++) {
                    cv::Vec3f pixel = gauss.at<cv::Vec3f>(y, x);
                    float val = (pixel[0] + pixel[1] + pixel[2]) / 3.0f;
                    gauss.at<cv::Vec3f>(y, x) = cv::Vec3f(
                        std::exp(-12.5f * std::pow(val - 0.5f, 2)));
                }
            }
        }

        cv::convertScaleAbs(gauss, well_exposed, 255.0);

        // 组合所有权重
        cv::Mat weight;
        if (exposure.channels() == 1) {
            weight = contrast.mul(well_exposed);
        } else {
            weight = contrast.mul(saturation).mul(well_exposed);
        }

        // 添加小常数避免可能的除零
        weight += 1e-6;

        weights.push_back(weight);
    }

    // 归一化权重
    cv::Mat weight_sum = weights[0].clone();
    for (int i = 1; i < num_exposures; i++) {
        weight_sum += weights[i];
    }

    for (int i = 0; i < num_exposures; i++) {
        weights[i] /= weight_sum;
    }

    // 融合曝光
    dst = cv::Mat::zeros(src.size(), CV_32F);

    for (int i = 0; i < num_exposures; i++) {
        cv::Mat temp;
        exposures[i].convertTo(temp, CV_32F);

        if (src.channels() == 1) {
            for (int y = 0; y < temp.rows; y++) {
                for (int x = 0; x < temp.cols; x++) {
                    dst.at<float>(y, x) += temp.at<float>(y, x) * weights[i].at<float>(y, x);
                }
            }
        } else {
            for (int y = 0; y < temp.rows; y++) {
                for (int x = 0; x < temp.cols; x++) {
                    cv::Vec3f pixel = temp.at<cv::Vec3f>(y, x);
                    cv::Vec3f weight = weights[i].at<cv::Vec3f>(y, x);

                    dst.at<cv::Vec3f>(y, x)[0] += pixel[0] * weight[0];
                    dst.at<cv::Vec3f>(y, x)[1] += pixel[1] * weight[1];
                    dst.at<cv::Vec3f>(y, x)[2] += pixel[2] * weight[2];
                }
            }
        }
    }

    // 转回8位
    dst.convertTo(dst, CV_8U);
}

} // namespace advanced
} // namespace ip101