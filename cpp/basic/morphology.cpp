#include "morphology.hpp"
#include <algorithm>

namespace ip101 {

using namespace cv;

namespace {
// 内部常量定义
constexpr int CACHE_LINE = 64;    // CPU缓存行大小(字节)
constexpr int SIMD_WIDTH = 32;    // AVX2 SIMD向量宽度(字节)
constexpr int BLOCK_SIZE = 16;    // 分块处理大小

// 内存对齐辅助函数
inline uchar* alignPtr(uchar* ptr, size_t align = CACHE_LINE) {
    return (uchar*)(((size_t)ptr + align - 1) & -align);
}

// 获取默认结构元素
Mat getDefaultKernel() {
    return Mat::ones(3, 3, CV_8UC1);
}

} // anonymous namespace

Mat create_kernel(int shape, Size ksize) {
    Mat kernel = Mat::zeros(ksize, CV_8UC1);
    int center_x = ksize.width / 2;
    int center_y = ksize.height / 2;

    switch (shape) {
        case MORPH_RECT:
            kernel = Mat::ones(ksize, CV_8UC1);
            break;

        case MORPH_CROSS:
            for (int i = 0; i < ksize.height; i++) {
                kernel.at<uchar>(i, center_x) = 1;
            }
            for (int j = 0; j < ksize.width; j++) {
                kernel.at<uchar>(center_y, j) = 1;
            }
            break;

        case MORPH_ELLIPSE: {
            float rx = (ksize.width - 1) / 2.0f;
            float ry = (ksize.height - 1) / 2.0f;
            float rx2 = rx * rx;
            float ry2 = ry * ry;

            for (int y = 0; y < ksize.height; y++) {
                for (int x = 0; x < ksize.width; x++) {
                    float dx = (x - center_x);
                    float dy = (y - center_y);
                    if ((dx * dx) / rx2 + (dy * dy) / ry2 <= 1.0f) {
                        kernel.at<uchar>(y, x) = 1;
                    }
                }
            }
            break;
        }
    }

    return kernel;
}

void dilate_manual(const Mat& src, Mat& dst,
                  const Mat& kernel, int iterations) {
    CV_Assert(!src.empty());

    // 使用默认3x3结构元素
    Mat k = kernel.empty() ? getDefaultKernel() : kernel;
    int kh = k.rows;
    int kw = k.cols;
    int kcy = kh / 2;
    int kcx = kw / 2;

    // 创建临时图像
    Mat temp;
    src.copyTo(temp);
    dst = src.clone();

    // 迭代处理
    for (int iter = 0; iter < iterations; iter++) {
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                uchar maxVal = 0;

                // 在结构元素范围内查找最大值
                for (int ky = 0; ky < kh; ky++) {
                    int sy = y + ky - kcy;
                    if (sy < 0 || sy >= src.rows) continue;

                    for (int kx = 0; kx < kw; kx++) {
                        int sx = x + kx - kcx;
                        if (sx < 0 || sx >= src.cols) continue;

                        if (k.at<uchar>(ky, kx)) {
                            maxVal = std::max(maxVal, temp.at<uchar>(sy, sx));
                        }
                    }
                }

                dst.at<uchar>(y, x) = maxVal;
            }
        }

        if (iter < iterations - 1) {
            dst.copyTo(temp);
        }
    }
}

void erode_manual(const Mat& src, Mat& dst,
                 const Mat& kernel, int iterations) {
    CV_Assert(!src.empty());

    // 使用默认3x3结构元素
    Mat k = kernel.empty() ? getDefaultKernel() : kernel;
    int kh = k.rows;
    int kw = k.cols;
    int kcy = kh / 2;
    int kcx = kw / 2;

    // 创建临时图像
    Mat temp;
    src.copyTo(temp);
    dst = src.clone();

    // 迭代处理
    for (int iter = 0; iter < iterations; iter++) {
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                uchar minVal = 255;

                // 在结构元素范围内查找最小值
                for (int ky = 0; ky < kh; ky++) {
                    int sy = y + ky - kcy;
                    if (sy < 0 || sy >= src.rows) continue;

                    for (int kx = 0; kx < kw; kx++) {
                        int sx = x + kx - kcx;
                        if (sx < 0 || sx >= src.cols) continue;

                        if (k.at<uchar>(ky, kx)) {
                            minVal = std::min(minVal, temp.at<uchar>(sy, sx));
                        }
                    }
                }

                dst.at<uchar>(y, x) = minVal;
            }
        }

        if (iter < iterations - 1) {
            dst.copyTo(temp);
        }
    }
}

void opening_manual(const Mat& src, Mat& dst,
                   const Mat& kernel, int iterations) {
    Mat temp;
    erode_manual(src, temp, kernel, iterations);
    dilate_manual(temp, dst, kernel, iterations);
}

void closing_manual(const Mat& src, Mat& dst,
                   const Mat& kernel, int iterations) {
    Mat temp;
    dilate_manual(src, temp, kernel, iterations);
    erode_manual(temp, dst, kernel, iterations);
}

void morphological_gradient_manual(const Mat& src, Mat& dst,
                                 const Mat& kernel) {
    Mat dilated, eroded;
    dilate_manual(src, dilated, kernel);
    erode_manual(src, eroded, kernel);

    // 计算形态学梯度
    dst.create(src.size(), CV_8UC1);
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            dst.at<uchar>(y, x) = saturate_cast<uchar>(
                dilated.at<uchar>(y, x) - eroded.at<uchar>(y, x)
            );
        }
    }
}

} // namespace ip101