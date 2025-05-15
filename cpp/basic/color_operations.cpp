/**
 * 颜色操作相关问题：
 * 1. 通道替换 - 将RGB通道顺序改为BGR
 * 2. 灰度化 - 将彩色图像转换为灰度图像
 * 3. 二值化 - 将灰度图像转换为二值图像
 * 4. 大津算法 - 自适应二值化
 * 5. HSV变换 - 将RGB图像转换为HSV色彩空间
 */

#include <basic/color_operations.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <thread>
#include <immintrin.h>
#include <cmath>

using namespace cv;
using namespace std;

namespace ip101 {

namespace {
// 内部常量定义
constexpr int CACHE_LINE = 64;    // CPU缓存行大小(字节)
constexpr int SIMD_WIDTH = 32;    // AVX2 SIMD向量宽度(字节)
constexpr int BLOCK_SIZE = 16;    // 分块处理大小

// 灰度化权重
constexpr float GRAY_WEIGHT_R = 0.299f;
constexpr float GRAY_WEIGHT_G = 0.587f;
constexpr float GRAY_WEIGHT_B = 0.114f;

// 内存对齐辅助函数
inline uchar* alignPtr(uchar* ptr, size_t align = CACHE_LINE) {
    return (uchar*)(((size_t)ptr + align - 1) & ~(align - 1));
}

// SIMD优化的灰度化计算
inline void gray_simd(const uchar* src, uchar* dst, int width) {
    __m256 weight_r = _mm256_set1_ps(GRAY_WEIGHT_R);
    __m256 weight_g = _mm256_set1_ps(GRAY_WEIGHT_G);
    __m256 weight_b = _mm256_set1_ps(GRAY_WEIGHT_B);

    for (int col = 0; col < width; col += 8) {
        // 加载8个像素的BGR值
        __m256i bgr = _mm256_loadu_si256((__m256i*)(src + col * 3));

        // 分离BGR通道
        __m256 b = _mm256_cvtepi32_ps(_mm256_and_si256(bgr, _mm256_set1_epi32(0xFF)));
        __m256 g = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(bgr, 8), _mm256_set1_epi32(0xFF)));
        __m256 r = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(bgr, 16), _mm256_set1_epi32(0xFF)));

        // 计算加权和
        __m256 gray = _mm256_add_ps(
            _mm256_mul_ps(r, weight_r),
            _mm256_add_ps(
                _mm256_mul_ps(g, weight_g),
                _mm256_mul_ps(b, weight_b)
            )
        );

        // 转换回uchar并存储
        __m256i gray_i = _mm256_cvtps_epi32(gray);
        _mm256_storeu_si256((__m256i*)(dst + col), gray_i);
    }
}

} // anonymous namespace

void channel_swap(const Mat& src, Mat& dst, int r_idx, int g_idx, int b_idx) {
    CV_Assert(!src.empty() && src.type() == CV_8UC3);
    CV_Assert(r_idx >= 0 && r_idx < 3 && g_idx >= 0 && g_idx < 3 && b_idx >= 0 && b_idx < 3);

    dst.create(src.size(), src.type());

    #pragma omp parallel for
    for (int y = 0; y < src.rows; ++y) {
        for (int col = 0; col < src.cols; ++col) {
            const Vec3b& pixel = src.at<Vec3b>(y, col);
            dst.at<Vec3b>(y, col) = Vec3b(pixel[b_idx], pixel[g_idx], pixel[r_idx]);
        }
    }
}

void to_gray(const Mat& src, Mat& dst, const std::string& method) {
    CV_Assert(!src.empty() && src.type() == CV_8UC3);

    dst.create(src.size(), CV_8UC1);

    if (method == "weighted") {
        #pragma omp parallel for
        for (int y = 0; y < src.rows; ++y) {
            for (int col = 0; col < src.cols; ++col) {
                const Vec3b& pixel = src.at<Vec3b>(y, col);
                dst.at<uchar>(y, col) = saturate_cast<uchar>(
                    GRAY_WEIGHT_B * pixel[0] +
                    GRAY_WEIGHT_G * pixel[1] +
                    GRAY_WEIGHT_R * pixel[2]
                );
            }
        }
    } else if (method == "average") {
        #pragma omp parallel for
        for (int y = 0; y < src.rows; ++y) {
            for (int col = 0; col < src.cols; ++col) {
                const Vec3b& pixel = src.at<Vec3b>(y, col);
                dst.at<uchar>(y, col) = saturate_cast<uchar>(
                    (pixel[0] + pixel[1] + pixel[2]) / 3.0f
                );
            }
        }
    } else if (method == "max") {
        #pragma omp parallel for
        for (int y = 0; y < src.rows; ++y) {
            for (int col = 0; col < src.cols; ++col) {
                const Vec3b& pixel = src.at<Vec3b>(y, col);
                dst.at<uchar>(y, col) = std::max({pixel[0], pixel[1], pixel[2]});
            }
        }
    } else if (method == "min") {
        #pragma omp parallel for
        for (int y = 0; y < src.rows; ++y) {
            for (int col = 0; col < src.cols; ++col) {
                const Vec3b& pixel = src.at<Vec3b>(y, col);
                dst.at<uchar>(y, col) = std::min({pixel[0], pixel[1], pixel[2]});
            }
        }
    } else {
        throw std::invalid_argument("Unsupported grayscale method: " + method);
    }
}

void threshold_image(const Mat& src, Mat& dst, double threshold, double max_value, const std::string& method) {
    CV_Assert(!src.empty() && (src.type() == CV_8UC1 || src.type() == CV_8UC3));

    // 如果是彩色图像，先转换为灰度图
    Mat gray;
    if (src.type() == CV_8UC3) {
        to_gray(src, gray);
    } else {
        gray = src;
    }

    dst.create(gray.size(), CV_8UC1);

    int thresh_type;
    if (method == "binary") {
        thresh_type = THRESH_BINARY;
    } else if (method == "binary_inv") {
        thresh_type = THRESH_BINARY_INV;
    } else if (method == "trunc") {
        thresh_type = THRESH_TRUNC;
    } else if (method == "tozero") {
        thresh_type = THRESH_TOZERO;
    } else if (method == "tozero_inv") {
        thresh_type = THRESH_TOZERO_INV;
    } else {
        throw std::invalid_argument("Unsupported threshold method: " + method);
    }

    cv::threshold(gray, dst, threshold, max_value, thresh_type);
}

double otsu_threshold(const Mat& src, Mat& dst, double max_value) {
    CV_Assert(!src.empty() && (src.type() == CV_8UC1 || src.type() == CV_8UC3));

    // 如果是彩色图像，先转换为灰度图
    Mat gray;
    if (src.type() == CV_8UC3) {
        to_gray(src, gray);
    } else {
        gray = src;
    }

    // 计算直方图
    int histogram[256] = {0};
    for (int y = 0; y < gray.rows; ++y) {
        for (int col = 0; col < gray.cols; ++col) {
            histogram[gray.at<uchar>(y, col)]++;
        }
    }

    // 计算总像素数
    int total = gray.rows * gray.cols;

    // 计算最优阈值
    double sum = 0;
    for (int i = 0; i < 256; ++i) {
        sum += i * histogram[i];
    }

    double sumB = 0;
    int wB = 0;
    int wF = 0;
    double maxVariance = 0;
    double threshold = 0;

    for (int t = 0; t < 256; ++t) {
        wB += histogram[t];
        if (wB == 0) continue;

        wF = total - wB;
        if (wF == 0) break;

        sumB += t * histogram[t];
        double mB = sumB / wB;
        double mF = (sum - sumB) / wF;

        double variance = wB * wF * (mB - mF) * (mB - mF);
        if (variance > maxVariance) {
            maxVariance = variance;
            threshold = t;
        }
    }

    // 应用阈值
    cv::threshold(gray, dst, threshold, max_value, THRESH_BINARY);

    return threshold;
}

void bgr_to_hsv(const Mat& src, Mat& dst) {
    CV_Assert(!src.empty() && src.type() == CV_8UC3);

    dst.create(src.size(), CV_8UC3);

    #pragma omp parallel for
    for (int y = 0; y < src.rows; ++y) {
        for (int col = 0; col < src.cols; ++col) {
            const Vec3b& bgr = src.at<Vec3b>(y, col);
            float b = bgr[0] / 255.0f;
            float g = bgr[1] / 255.0f;
            float r = bgr[2] / 255.0f;

            float max_val = std::max({r, g, b});
            float min_val = std::min({r, g, b});
            float diff = max_val - min_val;

            // 计算H
            float h = 0;
            if (diff > 0) {
                if (max_val == r) {
                    h = 60.0f * (g - b) / diff;
                } else if (max_val == g) {
                    h = 60.0f * (b - r) / diff + 120.0f;
                } else {
                    h = 60.0f * (r - g) / diff + 240.0f;
                }
            }
            if (h < 0) h += 360.0f;

            // 计算S
            float s = max_val > 0 ? diff / max_val : 0;

            // 计算V
            float v = max_val;

            // 转换到OpenCV的HSV范围
            dst.at<Vec3b>(y, col) = Vec3b(
                saturate_cast<uchar>(h / 2.0f),      // H: 0-180
                saturate_cast<uchar>(s * 255.0f),    // S: 0-255
                saturate_cast<uchar>(v * 255.0f)     // V: 0-255
            );
        }
    }
}

void hsv_to_bgr(const Mat& src, Mat& dst) {
    CV_Assert(!src.empty() && src.type() == CV_8UC3);

    dst.create(src.size(), CV_8UC3);

    #pragma omp parallel for
    for (int y = 0; y < src.rows; ++y) {
        for (int col = 0; col < src.cols; ++col) {
            const Vec3b& hsv = src.at<Vec3b>(y, col);
            float h = hsv[0] * 2.0f;          // H: 0-360
            float s = hsv[1] / 255.0f;        // S: 0-1
            float v = hsv[2] / 255.0f;        // V: 0-1

            float c = v * s;
            float m = v - c;
            float x = c * (1.0f - std::abs(std::fmod(h / 60.0f, 2.0f) - 1.0f));

            float r = 0, g = 0, b = 0;
            if (h < 60.0f) {
                r = c; g = x; b = 0;
            } else if (h < 120.0f) {
                r = x; g = c; b = 0;
            } else if (h < 180.0f) {
                r = 0; g = c; b = x;
            } else if (h < 240.0f) {
                r = 0; g = x; b = c;
            } else if (h < 300.0f) {
                r = x; g = 0; b = c;
            } else {
                r = c; g = 0; b = x;
            }

            dst.at<Vec3b>(y, col) = Vec3b(
                saturate_cast<uchar>((b + m) * 255.0f),
                saturate_cast<uchar>((g + m) * 255.0f),
                saturate_cast<uchar>((r + m) * 255.0f)
            );
        }
    }
}

void adjust_hsv(const Mat& src, Mat& dst, float h_offset, float s_scale, float v_scale) {
    CV_Assert(!src.empty() && src.type() == CV_8UC3);

    dst.create(src.size(), CV_8UC3);

    #pragma omp parallel for
    for (int y = 0; y < src.rows; ++y) {
        for (int col = 0; col < src.cols; ++col) {
            const Vec3b& hsv = src.at<Vec3b>(y, col);

            // 调整H (注意OpenCV中H的范围是0-180)
            float h = hsv[0] * 2.0f + h_offset;  // 转换到0-360范围
            h = std::fmod(h + 360.0f, 360.0f);   // 处理环绕
            h = h / 2.0f;                        // 转回0-180范围

            // 调整S
            float s = hsv[1] * s_scale;

            // 调整V
            float v = hsv[2] * v_scale;

            dst.at<Vec3b>(y, col) = Vec3b(
                saturate_cast<uchar>(h),
                saturate_cast<uchar>(s),
                saturate_cast<uchar>(v)
            );
        }
    }
}

} // namespace ip101