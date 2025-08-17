#include <advanced/effects/cartoon_effect.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace ip101 {
namespace advanced {

void detect_edges(const cv::Mat& src, cv::Mat& edges, int edge_size) {
    CV_Assert(!src.empty());

    // 转换为灰度图像
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // 使用中值滤波减少噪声
    cv::Mat blurred;
    cv::medianBlur(gray, blurred, 5);

    // 使用自适应阈值进行边缘检测
    cv::Mat thresh;
    cv::adaptiveThreshold(blurred, thresh, 255, cv::ADAPTIVE_THRESH_MEAN_C,
                         cv::THRESH_BINARY_INV, 9, 2);

    // 扩张边缘
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(edge_size, edge_size));
    cv::dilate(thresh, edges, kernel);
}

void color_quantization(const cv::Mat& src, cv::Mat& dst, int levels) {
    CV_Assert(!src.empty() && levels > 0);

    // 确保图像是BGR格式
    cv::Mat bgr;
    if (src.channels() == 3) {
        bgr = src.clone();
    } else {
        cv::cvtColor(src, bgr, cv::COLOR_GRAY2BGR);
    }

    // 计算每个通道的量化因子
    double factor = 255.0 / levels;

    // 创建输出图像
    dst.create(bgr.size(), bgr.type());

    // 对每个像素进行量化
    #pragma omp parallel for
    for (int y = 0; y < bgr.rows; y++) {
        for (int x = 0; x < bgr.cols; x++) {
            cv::Vec3b& pixel = bgr.at<cv::Vec3b>(y, x);
            cv::Vec3b& output_pixel = dst.at<cv::Vec3b>(y, x);

            // 对每个通道进行量化
            for (int c = 0; c < 3; c++) {
                // 量化处理：将像素值归一化到[0, levels-1]，然后再乘以factor得到量化后的值
                int index = static_cast<int>(pixel[c] / factor);
                output_pixel[c] = static_cast<uchar>(std::min(255, static_cast<int>(index * factor + factor / 2)));
            }
        }
    }
}

void cartoon_effect(const cv::Mat& src, cv::Mat& dst, const CartoonParams& params) {
    CV_Assert(!src.empty());

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 1. 边缘检测
    cv::Mat edges;
    detect_edges(src, edges, params.edge_size);

    // 2. 使用中值滤波平滑图像
    cv::Mat smoothed;
    cv::medianBlur(src, smoothed, params.median_blur_size);

    // 3. 使用双边滤波进一步平滑同时保留边缘
    cv::Mat bilateral;
    cv::bilateralFilter(smoothed, bilateral, params.bilateral_d,
                       params.bilateral_sigma_color, params.bilateral_sigma_space);

    // 4. 颜色量化
    cv::Mat quantized;
    color_quantization(bilateral, quantized, params.quantize_levels);

    // 5. 将黑色边缘与量化后的图像合并
    #pragma omp parallel for
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            if (edges.at<uchar>(y, x) > 0) {
                // 如果是边缘，设置为黑色
                dst.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
            } else {
                // 否则使用量化后的颜色
                dst.at<cv::Vec3b>(y, x) = quantized.at<cv::Vec3b>(y, x);
            }
        }
    }
}

void enhanced_cartoon_effect(const cv::Mat& src, cv::Mat& dst, const CartoonParams& params, double texture_strength) {
    CV_Assert(!src.empty());
    CV_Assert(texture_strength >= 0.0 && texture_strength <= 1.0);

    // 1. 使用基本卡通效果算法
    cv::Mat basic_cartoon;
    cartoon_effect(src, basic_cartoon, params);

    if (texture_strength <= 0.0) {
        // 如果纹理增强强度为0，直接返回基本卡通效果
        basic_cartoon.copyTo(dst);
        return;
    }

    // 2. 纹理提取
    cv::Mat gray, texture;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // 使用高斯差分(DoG)提取纹理
    cv::Mat blur1, blur2;
    cv::GaussianBlur(gray, blur1, cv::Size(3, 3), 1.0);
    cv::GaussianBlur(gray, blur2, cv::Size(9, 9), 3.0);

    // DoG结果
    cv::Mat dog = blur1 - blur2;

    // 归一化DoG结果
    cv::normalize(dog, texture, 0, 255, cv::NORM_MINMAX);
    texture.convertTo(texture, CV_8U);

    // 3. 调整纹理强度
    cv::Mat weightedTexture;
    texture.convertTo(weightedTexture, CV_32F, texture_strength);

    // 4. 将纹理合并到卡通图像中
    cv::Mat cartoon_f;
    basic_cartoon.convertTo(cartoon_f, CV_32FC3);

    cv::Mat textureRGB[3] = {weightedTexture, weightedTexture, weightedTexture};
    cv::Mat textureRGBMerged;
    cv::merge(textureRGB, 3, textureRGBMerged);

    // 合并纹理
    cv::Mat result;
    cv::multiply(cartoon_f, (1.0 + textureRGBMerged / 255.0), result);

    // 转换回8位图像
    result.convertTo(dst, CV_8UC3);

    // 确保边缘仍然是黑色
    #pragma omp parallel for
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            if (basic_cartoon.at<cv::Vec3b>(y, x) == cv::Vec3b(0, 0, 0)) {
                dst.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
            }
        }
    }
}

} // namespace advanced
} // namespace ip101