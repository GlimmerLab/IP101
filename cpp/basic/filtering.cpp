#include <basic/filtering.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <immintrin.h> // for SSE/AVX
#include <thread>

// 添加M_PI的定义
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ip101 {

using namespace cv;
using namespace std;

// 常量定义
constexpr int CACHE_LINE = 64;  // 缓存行大小
constexpr int SIMD_WIDTH = 32;  // AVX2 宽度
constexpr int BLOCK_SIZE = 16;  // 分块大小

/**
 * @brief 均值滤波实现
 * @param src 输入图像
 * @param kernelSize 滤波核大小
 * @return 处理后的图像
 */
Mat meanFilter(const Mat& src, int kernelSize) {
    Mat dst = src.clone();
    int halfKernel = kernelSize / 2;

    for(int y = halfKernel; y < src.rows - halfKernel; y++) {
        for(int x = halfKernel; x < src.cols - halfKernel; x++) {
            int sum = 0;
            // 邻居聚会时间！
            for(int i = -halfKernel; i <= halfKernel; i++) {
                for(int j = -halfKernel; j <= halfKernel; j++) {
                    sum += src.at<uchar>(y + i, x + j);
                }
            }
            // 取个平均，和谐相处
            dst.at<uchar>(y, x) = sum / (kernelSize * kernelSize);
        }
    }
    return dst;
}

/**
 * @brief 中值滤波实现
 * @param src 输入图像
 * @param kernelSize 滤波核大小
 * @return 处理后的图像
 */
Mat medianFilter(const Mat& src, int kernelSize) {
    Mat dst = src.clone();
    int halfKernel = kernelSize / 2;
    vector<uchar> neighbors;  // 用来存放邻居们的"投票"

    for(int y = halfKernel; y < src.rows - halfKernel; y++) {
        for(int x = halfKernel; x < src.cols - halfKernel; x++) {
            neighbors.clear();
            // 收集邻居们的意见
            for(int i = -halfKernel; i <= halfKernel; i++) {
                for(int j = -halfKernel; j <= halfKernel; j++) {
                    neighbors.push_back(src.at<uchar>(y + i, x + j));
                }
            }
            // 排序，取中位数（最公平的决定！）
            sort(neighbors.begin(), neighbors.end());
            dst.at<uchar>(y, x) = neighbors[neighbors.size() / 2];
        }
    }
    return dst;
}

/**
 * @brief 高斯滤波实现
 * @param src 输入图像
 * @param kernelSize 滤波核大小
 * @param sigma 高斯函数的标准差
 * @return 处理后的图像
 */
Mat gaussianFilter(const Mat& src, int kernelSize, double sigma) {
    Mat dst = src.clone();
    int halfKernel = kernelSize / 2;

    // 先计算高斯核（权重矩阵）
    vector<vector<double>> kernel(kernelSize, vector<double>(kernelSize));
    double sum = 0.0;

    for(int i = -halfKernel; i <= halfKernel; i++) {
        for(int j = -halfKernel; j <= halfKernel; j++) {
            kernel[i + halfKernel][j + halfKernel] =
                exp(-(i*i + j*j)/(2*sigma*sigma)) / (2*M_PI*sigma*sigma);
            sum += kernel[i + halfKernel][j + halfKernel];
        }
    }

    // 归一化，确保权重和为1
    for(int i = 0; i < kernelSize; i++) {
        for(int j = 0; j < kernelSize; j++) {
            kernel[i][j] /= sum;
        }
    }

    // 应用滤波器
    for(int y = halfKernel; y < src.rows - halfKernel; y++) {
        for(int x = halfKernel; x < src.cols - halfKernel; x++) {
            double pixelValue = 0.0;
            // 加权求和，近亲远疏
            for(int i = -halfKernel; i <= halfKernel; i++) {
                for(int j = -halfKernel; j <= halfKernel; j++) {
                    pixelValue += src.at<uchar>(y + i, x + j) *
                                 kernel[i + halfKernel][j + halfKernel];
                }
            }
            dst.at<uchar>(y, x) = static_cast<uchar>(pixelValue);
        }
    }
    return dst;
}

/**
 * @brief 均值池化实现
 * @param src 输入图像
 * @param poolSize 池化大小
 * @return 处理后的图像
 */
Mat meanPooling(const Mat& src, int poolSize) {
    int newRows = src.rows / poolSize;
    int newCols = src.cols / poolSize;
    Mat dst(newRows, newCols, src.type());

    for(int y = 0; y < newRows; y++) {
        for(int x = 0; x < newCols; x++) {
            int sum = 0;
            // 计算一个池化区域的平均值
            for(int i = 0; i < poolSize; i++) {
                for(int j = 0; j < poolSize; j++) {
                    sum += src.at<uchar>(y*poolSize + i, x*poolSize + j);
                }
            }
            dst.at<uchar>(y, x) = sum / (poolSize * poolSize);
        }
    }
    return dst;
}

/**
 * @brief 最大池化实现
 * @param src 输入图像
 * @param poolSize 池化大小
 * @return 处理后的图像
 */
Mat maxPooling(const Mat& src, int poolSize) {
    int newRows = src.rows / poolSize;
    int newCols = src.cols / poolSize;
    Mat dst(newRows, newCols, src.type());

    for(int y = 0; y < newRows; y++) {
        for(int x = 0; x < newCols; x++) {
            uchar maxVal = 0;
            // 找出区域内的最大值
            for(int i = 0; i < poolSize; i++) {
                for(int j = 0; j < poolSize; j++) {
                    maxVal = max(maxVal,
                               src.at<uchar>(y*poolSize + i, x*poolSize + j));
                }
            }
            dst.at<uchar>(y, x) = maxVal;
        }
    }
    return dst;
}

// 内存对齐辅助函数
inline uchar* alignPtr(uchar* ptr, size_t align = CACHE_LINE) {
    return (uchar*)(((size_t)ptr + align - 1) & ~(align - 1));
}

// 使用AVX2优化的均值滤波实现
Mat meanFilter_optimized(const Mat& src, int kernelSize) {
    CV_Assert(!src.empty() && src.type() == CV_8UC1);
    Mat dst = src.clone();
    int halfKernel = kernelSize / 2;

    // 创建扩展图像
    Mat padded;
    copyMakeBorder(src, padded, halfKernel, halfKernel, halfKernel, halfKernel, BORDER_REFLECT);

    // 预计算除法因子
    const __m256i div_factor = _mm256_set1_epi16(kernelSize * kernelSize);

    #pragma omp parallel for
    for (int y = 0; y < src.rows; y++) {
        uchar* dstRow = dst.ptr<uchar>(y);

        // 使用AVX2优化
        for (int x = 0; x <= src.cols - 32; x += 32) {
            __m256i sum = _mm256_setzero_si256();

            // 在核窗口内累加
            for (int ky = -halfKernel; ky <= halfKernel; ky++) {
                const uchar* srcPtr = padded.ptr<uchar>(y + halfKernel + ky) + x + halfKernel;
                for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                    // 加载32个像素
                    __m256i pixels = _mm256_loadu_si256((__m256i*)(srcPtr + kx));
                    // 转换为16位整数
                    __m256i pixels_16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(pixels, 0));
                    // 累加
                    sum = _mm256_add_epi16(sum, pixels_16);
                }
            }

            // 执行除法
            __m256i result = _mm256_div_epi16(sum, div_factor);

            // 转换回8位并存储
            __m128i result_8bit = _mm_packus_epi16(_mm256_extracti128_si256(result, 0),
                                                  _mm256_extracti128_si256(result, 1));
            _mm_storeu_si128((__m128i*)(dstRow + x), result_8bit);
        }

        // 处理剩余像素
        for (int x = (src.cols/32)*32; x < src.cols; x++) {
            int sum = 0;
            for (int ky = -halfKernel; ky <= halfKernel; ky++) {
                const uchar* srcPtr = padded.ptr<uchar>(y + halfKernel + ky);
                for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                    sum += srcPtr[x + halfKernel + kx];
                }
            }
            dstRow[x] = static_cast<uchar>(sum / (kernelSize * kernelSize));
        }
    }

    return dst;
}

// 使用直方图优化的中值滤波实现
Mat medianFilter_optimized(const Mat& src, int kernelSize) {
    CV_Assert(!src.empty() && src.type() == CV_8UC1);
    Mat dst = src.clone();
    int halfKernel = kernelSize / 2;
    int windowSize = kernelSize * kernelSize;
    int medianPos = windowSize / 2;

    // 创建扩展图像
    Mat padded;
    copyMakeBorder(src, padded, halfKernel, halfKernel, halfKernel, halfKernel, BORDER_REFLECT);

    // 使用直方图方法
    vector<int> histogram(256);

    #pragma omp parallel for private(histogram)
    for (int y = 0; y < src.rows; y++) {
        uchar* dstRow = dst.ptr<uchar>(y);

        // 使用AVX2优化
        for (int x = 0; x <= src.cols - 32; x += 32) {
            // 重置直方图
            fill(histogram.begin(), histogram.end(), 0);

            // 收集窗口内的像素值
            for (int ky = -halfKernel; ky <= halfKernel; ky++) {
                const uchar* srcPtr = padded.ptr<uchar>(y + halfKernel + ky) + x + halfKernel;
                for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                    // 使用AVX2加载8个像素
                    __m256i pixels = _mm256_loadu_si256((__m256i*)(srcPtr + kx));
                    // 更新直方图
                    for (int i = 0; i < 32; i++) {
                        histogram[((uchar*)&pixels)[i]]++;
                    }
                }
            }

            // 计算中值
            int count = 0;
            for (int i = 0; i < 256; i++) {
                count += histogram[i];
                if (count > medianPos) {
                    // 使用AVX2存储结果
                    __m256i result = _mm256_set1_epi8(static_cast<uchar>(i));
                    _mm256_storeu_si256((__m256i*)(dstRow + x), result);
                    break;
                }
            }
        }

        // 处理剩余像素
        for (int x = (src.cols/32)*32; x < src.cols; x++) {
            // 重置直方图
            fill(histogram.begin(), histogram.end(), 0);

            // 收集窗口内的像素值
            for (int ky = -halfKernel; ky <= halfKernel; ky++) {
                const uchar* srcPtr = padded.ptr<uchar>(y + halfKernel + ky);
                for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                    histogram[srcPtr[x + halfKernel + kx]]++;
                }
            }

            // 计算中值
            int count = 0;
            for (int i = 0; i < 256; i++) {
                count += histogram[i];
                if (count > medianPos) {
                    dstRow[x] = static_cast<uchar>(i);
                    break;
                }
            }
        }
    }

    return dst;
}

// 修复在gaussianFilter_optimized中的错误代码
void fix_gaussian_filter_simd(const uchar* srcPtr, __m256d& sum, const __m256d& kernel) {
    // 加载8个像素
    __m128i pixels_low = _mm_loadu_si128((__m128i*)srcPtr);
    // 转换为double (只处理前4个像素)
    __m128i pixels_int = _mm_cvtepu8_epi32(pixels_low);
    __m256d pixels_d = _mm256_cvtepi32_pd(pixels_int);
    // 累加
    sum = _mm256_add_pd(sum, _mm256_mul_pd(pixels_d, kernel));
}

// 使用AVX2优化的高斯滤波实现
Mat gaussianFilter_optimized(const Mat& src, int kernelSize, double sigma) {
    CV_Assert(!src.empty() && src.type() == CV_8UC1);
    Mat dst = src.clone();
    int halfKernel = kernelSize / 2;

    // 创建扩展图像
    Mat padded;
    copyMakeBorder(src, padded, halfKernel, halfKernel, halfKernel, halfKernel, BORDER_REFLECT);

    // 预计算高斯核
    vector<double> kernel1D(kernelSize);
    double sum = 0.0;
    for (int i = 0; i < kernelSize; i++) {
        double x = i - halfKernel;
        kernel1D[i] = exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel1D[i];
    }
    // 归一化
    for (int i = 0; i < kernelSize; i++) {
        kernel1D[i] /= sum;
    }

    // 使用OpenMP并行计算
    #pragma omp parallel for
    for (int y = 0; y < src.rows; y++) {
        uchar* dstRow = dst.ptr<uchar>(y);
        const uchar* paddedRow = padded.ptr<uchar>(y + halfKernel);

        // 处理所有像素
        for (int x = 0; x < src.cols; x++) {
            double sum = 0.0;
            for (int ky = -halfKernel; ky <= halfKernel; ky++) {
                const uchar* srcPtr = padded.ptr<uchar>(y + halfKernel + ky);
                for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                    sum += srcPtr[x + halfKernel + kx] *
                           kernel1D[ky + halfKernel] * kernel1D[kx + halfKernel];
                }
            }
            dstRow[x] = static_cast<uchar>(sum);
        }
    }

    return dst;
}

// 使用SIMD和分块优化的均值池化实现
Mat meanPooling_optimized(const Mat& src, int poolSize) {
    int newRows = src.rows / poolSize;
    int newCols = src.cols / poolSize;
    Mat dst(newRows, newCols, src.type());

    // 使用AVX2处理32个像素
    #pragma omp parallel for
    for(int y = 0; y < newRows; y++) {
        for(int x = 0; x < newCols; x += 32) {
            if(x + 32 <= newCols) {
                __m256i sum = _mm256_setzero_si256();

                for(int py = 0; py < poolSize; py++) {
                    const uchar* srcPtr = src.ptr<uchar>(y * poolSize + py);
                    for(int px = 0; px < poolSize; px++) {
                        __m256i pixels = _mm256_loadu_si256((__m256i*)(srcPtr + (x * poolSize + px)));
                        sum = _mm256_add_epi16(sum, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)srcPtr)));
                    }
                }

                // 执行平均
                __m256i result = _mm256_srli_epi16(sum, static_cast<int>(log2(poolSize * poolSize)));

                // 存储结果
                __m128i result_8bit = _mm_packus_epi16(_mm256_extracti128_si256(result, 0),
                                                      _mm256_extracti128_si256(result, 1));
                _mm_storeu_si128((__m128i*)(dst.ptr<uchar>(y) + x), result_8bit);
            } else {
                // 处理剩余像素
                for(int rx = x; rx < newCols; rx++) {
                    int sum = 0;
                    for(int py = 0; py < poolSize; py++) {
                        const uchar* srcPtr = src.ptr<uchar>(y * poolSize + py);
                        for(int px = 0; px < poolSize; px++) {
                            sum += srcPtr[rx * poolSize + px];
                        }
                    }
                    dst.at<uchar>(y, rx) = static_cast<uchar>(sum / (poolSize * poolSize));
                }
            }
        }
    }

    return dst;
}

// 使用SIMD和分块优化的最大池化实现
Mat maxPooling_optimized(const Mat& src, int poolSize) {
    int newRows = src.rows / poolSize;
    int newCols = src.cols / poolSize;
    Mat dst(newRows, newCols, src.type());

    #pragma omp parallel for
    for(int y = 0; y < newRows; y++) {
        for(int x = 0; x < newCols; x += 16) {
            if(x + 16 <= newCols) {
                __m128i maxVal = _mm_setzero_si128();

                for(int py = 0; py < poolSize; py++) {
                    const uchar* srcPtr = src.ptr<uchar>(y * poolSize + py);
                    for(int px = 0; px < poolSize; px++) {
                        __m128i pixels = _mm_loadu_si128((__m128i*)(srcPtr + (x * poolSize + px)));
                        maxVal = _mm_max_epu8(maxVal, pixels);
                    }
                }

                _mm_storeu_si128((__m128i*)(dst.ptr<uchar>(y) + x), maxVal);
            } else {
                // 处理剩余像素
                for(int rx = x; rx < newCols; rx++) {
                    uchar maxVal = 0;
                    for(int py = 0; py < poolSize; py++) {
                        const uchar* srcPtr = src.ptr<uchar>(y * poolSize + py);
                        for(int px = 0; px < poolSize; px++) {
                            maxVal = max(maxVal, srcPtr[rx * poolSize + px]);
                        }
                    }
                    dst.at<uchar>(y, rx) = maxVal;
                }
            }
        }
    }

    return dst;
}

// 创建高斯核的函数
Mat create_gaussian_kernel(int ksize, double sigma_x, double sigma_y) {
    Mat kernel(ksize, ksize, CV_64F);
    int halfKernel = ksize / 2;
    double sum = 0.0;

    for (int y = -halfKernel; y <= halfKernel; y++) {
        for (int x = -halfKernel; x <= halfKernel; x++) {
            double value = exp(-(x*x/(2*sigma_x*sigma_x) + y*y/(2*sigma_y*sigma_y)));
            kernel.at<double>(y + halfKernel, x + halfKernel) = value;
            sum += value;
        }
    }

    // 归一化
    kernel /= sum;
    return kernel;
}

} // namespace ip101