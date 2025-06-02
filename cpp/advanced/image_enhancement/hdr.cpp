/**
 * @file hdr.cpp
 * @brief HDR（高动态范围）图像融合与色调映射算法实现（含性能优化版）
 * @author GlimmerLab
 * @date 2024-06-09
 *
 * HDR算法就像一位调和光影的画家，把多张不同曝光的照片，融合成一幅明暗有度、细节丰富的画卷。
 */

#include "hdr.hpp"
#include <cmath>
#include <random>
#include <algorithm>
#include <omp.h>

namespace ip101 {
namespace advanced {

// ================== 工具函数 ==================

/**
 * @brief 权重函数：像素值越接近中间灰度，权重越高。
 *        极端像素（过曝/欠曝）权重为0。
 */
float weight_function(uchar pixel_value) {
    return pixel_value > 0 && pixel_value < 255 ?
           1.0f - std::abs(pixel_value - 128.0f) / 128.0f :
           0.0f;
}

// ================== 标准版实现 ==================

/**
 * @brief 计算相机响应曲线（标准版）
 *        通过采样点和曝光时间，推算出每个通道的响应曲线。
 */
cv::Mat calculate_camera_response(const std::vector<cv::Mat>& images,
                                 const std::vector<float>& exposure_times,
                                 float lambda,
                                 int samples) {
    int num_images = static_cast<int>(images.size());
    int height = images[0].rows;
    int width = images[0].cols;
    int channels = images[0].channels();
    std::vector<cv::Point> sample_points;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist_height(0, height - 1);
    std::uniform_int_distribution<> dist_width(0, width - 1);
    int valid_samples = 0;
    int max_attempts = samples * 10;
    int attempts = 0;
    // 随机采样，避免极端像素
    while (valid_samples < samples && attempts < max_attempts) {
        cv::Point pt(dist_width(gen), dist_height(gen));
        bool valid = true;
        for (const auto& img : images) {
            cv::Vec3b pixel = img.at<cv::Vec3b>(pt);
            if (pixel[0] <= 5 || pixel[0] >= 250 ||
                pixel[1] <= 5 || pixel[1] >= 250 ||
                pixel[2] <= 5 || pixel[2] >= 250) {
                valid = false;
                break;
            }
        }
        if (valid) {
            sample_points.push_back(pt);
            valid_samples++;
        }
        attempts++;
    }
    cv::Mat response_curve(256, channels, CV_32F);
    #pragma omp parallel for
    for (int c = 0; c < channels; c++) {
        int num_equations = sample_points.size() * num_images + 254 + 1;
        cv::Mat A = cv::Mat::zeros(num_equations, 256, CV_32F);
        cv::Mat b = cv::Mat::zeros(num_equations, 1, CV_32F);
        int eq_idx = 0;
        for (size_t i = 0; i < sample_points.size(); i++) {
            for (int j = 0; j < num_images; j++) {
                uchar z = images[j].at<cv::Vec3b>(sample_points[i])[c];
                float w = weight_function(z);
                A.at<float>(eq_idx, z) = w;
                A.at<float>(eq_idx, 128) = -w;
                b.at<float>(eq_idx, 0) = w * std::log(exposure_times[j]);
                eq_idx++;
            }
        }
        // 平滑约束，防止曲线剧烈波动
        for (int i = 0; i < 254; i++) {
            float w = weight_function(i+1);
            A.at<float>(eq_idx, i) = lambda * w;
            A.at<float>(eq_idx, i+1) = -2 * lambda * w;
            A.at<float>(eq_idx, i+2) = lambda * w;
            eq_idx++;
        }
        // 固定中间值，避免平凡解
        A.at<float>(eq_idx, 128) = 1.0f;
        b.at<float>(eq_idx, 0) = 0.0f;
        cv::Mat x;
        cv::solve(A, b, x, cv::DECOMP_SVD);
        for (int i = 0; i < 256; i++) {
            response_curve.at<float>(i, c) = x.at<float>(i, 0);
        }
    }
    return response_curve;
}

/**
 * @brief HDR融合（标准版）
 *        多张不同曝光的照片融合为一张高动态范围图像。
 */
cv::Mat create_hdr(const std::vector<cv::Mat>& images,
                   const std::vector<float>& exposure_times,
                   const cv::Mat& response_curve) {
    int height = images[0].rows;
    int width = images[0].cols;
    int channels = images[0].channels();
    cv::Mat camera_response = response_curve.empty() ?
        calculate_camera_response(images, exposure_times) : response_curve;
    cv::Mat hdr_image(height, width, CV_32FC3, cv::Scalar(0, 0, 0));
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum_weights[3] = {0.0f, 0.0f, 0.0f};
            float pixel_values[3] = {0.0f, 0.0f, 0.0f};
            for (size_t i = 0; i < images.size(); i++) {
                const cv::Vec3b& pixel = images[i].at<cv::Vec3b>(y, x);
                for (int c = 0; c < channels; c++) {
                    float weight = weight_function(pixel[c]);
                    float radiance = camera_response.at<float>(pixel[c], c) - std::log(exposure_times[i]);
                    pixel_values[c] += weight * radiance;
                    sum_weights[c] += weight;
                }
            }
            for (int c = 0; c < channels; c++) {
                if (sum_weights[c] > 0.0f)
                    hdr_image.at<cv::Vec3f>(y, x)[c] = std::exp(pixel_values[c] / sum_weights[c]);
            }
        }
    }
    return hdr_image;
}

/**
 * @brief 全局色调映射（Reinhard，标准版）
 *        压缩动态范围，保留细节。
 */
cv::Mat tone_mapping_global(const cv::Mat& hdr_image, float key, float white_point) {
    double sum_log_luminance = 0.0;
    double epsilon = 1e-6;
    int valid_pixels = 0;
    #pragma omp parallel for reduction(+:sum_log_luminance, valid_pixels)
    for (int y = 0; y < hdr_image.rows; y++) {
        for (int x = 0; x < hdr_image.cols; x++) {
            cv::Vec3f pixel = hdr_image.at<cv::Vec3f>(y, x);
            float luminance = 0.2126f * pixel[2] + 0.7152f * pixel[1] + 0.0722f * pixel[0];
            if (luminance > epsilon) {
                sum_log_luminance += std::log(luminance);
                valid_pixels++;
            }
        }
    }
    float log_average_luminance = valid_pixels > 0 ?
                                std::exp(sum_log_luminance / valid_pixels) :
                                epsilon;
    float scale_factor = key / log_average_luminance;
    float Lwhite2 = white_point * white_point;
    cv::Mat ldr_image(hdr_image.size(), CV_8UC3);
    #pragma omp parallel for
    for (int y = 0; y < hdr_image.rows; y++) {
        for (int x = 0; x < hdr_image.cols; x++) {
            cv::Vec3f hdr_pixel = hdr_image.at<cv::Vec3f>(y, x);
            cv::Vec3b ldr_pixel;
            float luminance = 0.2126f * hdr_pixel[2] + 0.7152f * hdr_pixel[1] + 0.0722f * hdr_pixel[0];
            float scaled_luminance = scale_factor * luminance;
            float mapped_luminance = (scaled_luminance * (1.0f + scaled_luminance / Lwhite2)) / (1.0f + scaled_luminance);
            for (int c = 0; c < 3; c++) {
                float color_ratio = luminance > epsilon ? hdr_pixel[c] / luminance : 1.0f;
                float mapped_value = std::min(255.0f, 255.0f * color_ratio * mapped_luminance);
                ldr_pixel[c] = static_cast<uchar>(std::max(0.0f, mapped_value));
            }
            ldr_image.at<cv::Vec3b>(y, x) = ldr_pixel;
        }
    }
    return ldr_image;
}

/**
 * @brief 局部色调映射（Durand，标准版）
 *        分离基础层和细节层，压缩基础层动态范围。
 */
cv::Mat tone_mapping_local(const cv::Mat& hdr_image, float sigma, float contrast) {
    cv::Mat ldr_image(hdr_image.size(), CV_8UC3);
    cv::Mat luminance(hdr_image.size(), CV_32F);
    #pragma omp parallel for
    for (int y = 0; y < hdr_image.rows; y++) {
        for (int x = 0; x < hdr_image.cols; x++) {
            cv::Vec3f pixel = hdr_image.at<cv::Vec3f>(y, x);
            luminance.at<float>(y, x) = 0.2126f * pixel[2] + 0.7152f * pixel[1] + 0.0722f * pixel[0];
        }
    }
    cv::Mat log_luminance;
    cv::log(luminance + 1e-6f, log_luminance);
    cv::Mat base_layer;
    cv::bilateralFilter(log_luminance, base_layer, -1, sigma * 3, sigma);
    cv::Mat detail_layer = log_luminance - base_layer;
    float log_min, log_max;
    cv::minMaxLoc(base_layer, &log_min, &log_max);
    cv::Mat compressed_base = (base_layer - log_max) * contrast / (log_max - log_min);
    cv::Mat log_output = compressed_base + detail_layer;
    cv::Mat output_luminance;
    cv::exp(log_output, output_luminance);
    #pragma omp parallel for
    for (int y = 0; y < hdr_image.rows; y++) {
        for (int x = 0; x < hdr_image.cols; x++) {
            cv::Vec3f hdr_pixel = hdr_image.at<cv::Vec3f>(y, x);
            cv::Vec3b ldr_pixel;
            float original_luminance = luminance.at<float>(y, x);
            float mapped_luminance = output_luminance.at<float>(y, x);
            for (int c = 0; c < 3; c++) {
                float color_ratio = original_luminance > 1e-6f ? hdr_pixel[c] / original_luminance : 1.0f;
                float mapped_value = std::min(255.0f, 255.0f * color_ratio * mapped_luminance);
                ldr_pixel[c] = static_cast<uchar>(std::max(0.0f, mapped_value));
            }
            ldr_image.at<cv::Vec3b>(y, x) = ldr_pixel;
        }
    }
    return ldr_image;
}

// ================== 优化版实现 ==================

/**
 * @brief 计算相机响应曲线（优化版）
 *        采样点生成和通道处理均并行。
 */
cv::Mat calculate_camera_response_optimized(const std::vector<cv::Mat>& images,
                                            const std::vector<float>& exposure_times,
                                            float lambda,
                                            int samples) {
    int num_images = static_cast<int>(images.size());
    int height = images[0].rows;
    int width = images[0].cols;
    int channels = images[0].channels();
    std::vector<cv::Point> sample_points(samples);
    #pragma omp parallel for
    for (int i = 0; i < samples; ++i) {
        std::mt19937 gen(i * 1234567 + static_cast<unsigned>(time(nullptr)));
        std::uniform_int_distribution<> dist_height(0, height - 1);
        std::uniform_int_distribution<> dist_width(0, width - 1);
        int tries = 0;
        while (tries < 20) {
            int y = dist_height(gen);
            int x = dist_width(gen);
            bool valid = true;
            for (const auto& img : images) {
                cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);
                if (pixel[0] <= 5 || pixel[0] >= 250 ||
                    pixel[1] <= 5 || pixel[1] >= 250 ||
                    pixel[2] <= 5 || pixel[2] >= 250) {
                    valid = false;
                    break;
                }
            }
            if (valid) {
                sample_points[i] = cv::Point(x, y);
                break;
            }
            tries++;
        }
    }
    cv::Mat response_curve(256, channels, CV_32F);
    #pragma omp parallel for
    for (int c = 0; c < channels; c++) {
        int num_equations = samples * num_images + 254 + 1;
        cv::Mat A = cv::Mat::zeros(num_equations, 256, CV_32F);
        cv::Mat b = cv::Mat::zeros(num_equations, 1, CV_32F);
        int eq_idx = 0;
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < num_images; j++) {
                uchar z = images[j].at<cv::Vec3b>(sample_points[i])[c];
                float w = weight_function(z);
                A.at<float>(eq_idx, z) = w;
                A.at<float>(eq_idx, 128) = -w;
                b.at<float>(eq_idx, 0) = w * std::log(exposure_times[j]);
                eq_idx++;
            }
        }
        for (int i = 0; i < 254; i++) {
            float w = weight_function(i+1);
            A.at<float>(eq_idx, i) = lambda * w;
            A.at<float>(eq_idx, i+1) = -2 * lambda * w;
            A.at<float>(eq_idx, i+2) = lambda * w;
            eq_idx++;
        }
        A.at<float>(eq_idx, 128) = 1.0f;
        b.at<float>(eq_idx, 0) = 0.0f;
        cv::Mat x;
        cv::solve(A, b, x, cv::DECOMP_SVD);
        for (int i = 0; i < 256; i++) {
            response_curve.at<float>(i, c) = x.at<float>(i, 0);
        }
    }
    return response_curve;
}

/**
 * @brief HDR融合（优化版）
 *        像素并行，通道SIMD优化。
 */
cv::Mat create_hdr_optimized(const std::vector<cv::Mat>& images,
                             const std::vector<float>& exposure_times,
                             const cv::Mat& response_curve) {
    int height = images[0].rows;
    int width = images[0].cols;
    int channels = images[0].channels();
    cv::Mat camera_response = response_curve.empty() ?
        calculate_camera_response_optimized(images, exposure_times) : response_curve;
    cv::Mat hdr_image(height, width, CV_32FC3, cv::Scalar(0, 0, 0));
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum_weights[3] = {0.0f, 0.0f, 0.0f};
            float pixel_values[3] = {0.0f, 0.0f, 0.0f};
            for (size_t i = 0; i < images.size(); i++) {
                const cv::Vec3b& pixel = images[i].at<cv::Vec3b>(y, x);
                #pragma omp simd
                for (int c = 0; c < channels; c++) {
                    float weight = weight_function(pixel[c]);
                    float radiance = camera_response.at<float>(pixel[c], c) - std::log(exposure_times[i]);
                    pixel_values[c] += weight * radiance;
                    sum_weights[c] += weight;
                }
            }
            #pragma omp simd
            for (int c = 0; c < channels; c++) {
                if (sum_weights[c] > 0.0f)
                    hdr_image.at<cv::Vec3f>(y, x)[c] = std::exp(pixel_values[c] / sum_weights[c]);
            }
        }
    }
    return hdr_image;
}

/**
 * @brief 全局色调映射（优化版）
 *        使用OpenMP和SIMD优化，优化内存访问模式。
 */
cv::Mat tone_mapping_global_optimized(const cv::Mat& hdr_image, float key, float white_point) {
    int height = hdr_image.rows;
    int width = hdr_image.cols;
    cv::Mat ldr_image(height, width, CV_8UC3);

    // 预计算常量
    const float epsilon = 1e-6f;
    const float Lwhite2 = white_point * white_point;

    // 计算对数平均亮度（并行优化）
    double sum_log_luminance = 0.0;
    int valid_pixels = 0;

    #pragma omp parallel for reduction(+:sum_log_luminance, valid_pixels)
    for (int y = 0; y < height; y++) {
        const float* row_ptr = hdr_image.ptr<float>(y);
        for (int x = 0; x < width; x++) {
            float luminance = 0.2126f * row_ptr[x*3+2] +
                            0.7152f * row_ptr[x*3+1] +
                            0.0722f * row_ptr[x*3];
            if (luminance > epsilon) {
                sum_log_luminance += std::log(luminance);
                valid_pixels++;
            }
        }
    }

    float log_average_luminance = valid_pixels > 0 ?
                                std::exp(sum_log_luminance / valid_pixels) :
                                epsilon;
    float scale_factor = key / log_average_luminance;

    // 应用色调映射（并行优化）
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        const float* src_row = hdr_image.ptr<float>(y);
        uchar* dst_row = ldr_image.ptr<uchar>(y);

        #pragma omp simd
        for (int x = 0; x < width; x++) {
            float luminance = 0.2126f * src_row[x*3+2] +
                            0.7152f * src_row[x*3+1] +
                            0.0722f * src_row[x*3];

            float scaled_luminance = scale_factor * luminance;
            float mapped_luminance = (scaled_luminance * (1.0f + scaled_luminance / Lwhite2)) /
                                   (1.0f + scaled_luminance);

            // 保持色彩，修改亮度
            for (int c = 0; c < 3; c++) {
                float color_ratio = luminance > epsilon ? src_row[x*3+c] / luminance : 1.0f;
                float mapped_value = 255.0f * color_ratio * mapped_luminance;
                dst_row[x*3+c] = static_cast<uchar>(std::max(0.0f, std::min(255.0f, mapped_value)));
            }
        }
    }

    return ldr_image;
}

/**
 * @brief 局部色调映射（优化版）
 *        使用OpenMP和SIMD优化，优化内存访问和计算模式。
 */
cv::Mat tone_mapping_local_optimized(const cv::Mat& hdr_image, float sigma, float contrast) {
    int height = hdr_image.rows;
    int width = hdr_image.cols;
    cv::Mat ldr_image(height, width, CV_8UC3);

    // 预分配内存
    cv::Mat luminance(height, width, CV_32F);
    cv::Mat log_luminance(height, width, CV_32F);
    cv::Mat base_layer(height, width, CV_32F);
    cv::Mat detail_layer(height, width, CV_32F);
    cv::Mat output_luminance(height, width, CV_32F);

    const float epsilon = 1e-6f;

    // 计算亮度通道（并行优化）
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        const float* src_row = hdr_image.ptr<float>(y);
        float* lum_row = luminance.ptr<float>(y);

        #pragma omp simd
        for (int x = 0; x < width; x++) {
            lum_row[x] = 0.2126f * src_row[x*3+2] +
                        0.7152f * src_row[x*3+1] +
                        0.0722f * src_row[x*3];
        }
    }

    // 对亮度取对数（并行优化）
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        float* log_row = log_luminance.ptr<float>(y);
        const float* lum_row = luminance.ptr<float>(y);

        #pragma omp simd
        for (int x = 0; x < width; x++) {
            log_row[x] = std::log(lum_row[x] + epsilon);
        }
    }

    // 使用双边滤波分离基础层和细节层
    cv::bilateralFilter(log_luminance, base_layer, -1, sigma * 3, sigma);

    // 计算细节层（并行优化）
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        float* detail_row = detail_layer.ptr<float>(y);
        const float* log_row = log_luminance.ptr<float>(y);
        const float* base_row = base_layer.ptr<float>(y);

        #pragma omp simd
        for (int x = 0; x < width; x++) {
            detail_row[x] = log_row[x] - base_row[x];
        }
    }

    // 压缩基础层的动态范围
    float log_min, log_max;
    cv::minMaxLoc(base_layer, &log_min, &log_max);
    float range = log_max - log_min;

    // 重建压缩后的亮度（并行优化）
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        float* output_row = output_luminance.ptr<float>(y);
        const float* base_row = base_layer.ptr<float>(y);
        const float* detail_row = detail_layer.ptr<float>(y);

        #pragma omp simd
        for (int x = 0; x < width; x++) {
            float compressed_base = (base_row[x] - log_max) * contrast / range;
            output_row[x] = std::exp(compressed_base + detail_row[x]);
        }
    }

    // 应用色调映射（并行优化）
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        const float* src_row = hdr_image.ptr<float>(y);
        const float* lum_row = luminance.ptr<float>(y);
        const float* output_row = output_luminance.ptr<float>(y);
        uchar* dst_row = ldr_image.ptr<uchar>(y);

        #pragma omp simd
        for (int x = 0; x < width; x++) {
            float original_luminance = lum_row[x];
            float mapped_luminance = output_row[x];

            for (int c = 0; c < 3; c++) {
                float color_ratio = original_luminance > epsilon ?
                                  src_row[x*3+c] / original_luminance : 1.0f;
                float mapped_value = 255.0f * color_ratio * mapped_luminance;
                dst_row[x*3+c] = static_cast<uchar>(std::max(0.0f, std::min(255.0f, mapped_value)));
            }
        }
    }

    return ldr_image;
}

} // namespace advanced
} // namespace ip101
