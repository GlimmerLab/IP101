#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <immintrin.h> // for SSE/AVX
#include <thread>

using namespace cv;
using namespace std;
using namespace std::chrono;

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
    return (uchar*)(((size_t)ptr + align - 1) & -align);
}

// 使用AVX2优化的均值滤波实现
Mat meanFilter_optimized(const Mat& src, int kernelSize) {
    Mat dst(src.size(), src.type());
    int halfKernel = kernelSize / 2;
    const int width = src.cols;
    const int height = src.rows;

    // 为边界处理创建扩展图像
    Mat padded;
    copyMakeBorder(src, padded, halfKernel, halfKernel, halfKernel, halfKernel, BORDER_REFLECT);

    // 预计算除法因子
    const __m256i div_factor = _mm256_set1_epi16(kernelSize * kernelSize);

    #pragma omp parallel for
    for(int y = 0; y < height; y++) {
        uchar* dstRow = dst.ptr<uchar>(y);

        // 每次处理32个像素
        for(int x = 0; x <= width - 32; x += 32) {
            __m256i sum = _mm256_setzero_si256();

            // 在核窗口内累加
            for(int ky = -halfKernel; ky <= halfKernel; ky++) {
                const uchar* srcPtr = padded.ptr<uchar>(y + halfKernel + ky) + x + halfKernel;
                for(int kx = -halfKernel; kx <= halfKernel; kx++) {
                    __m256i src_pixels = _mm256_loadu_si256((__m256i*)(srcPtr + kx));
                    sum = _mm256_add_epi16(sum, _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)srcPtr)));
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
        for(int x = (width/32)*32; x < width; x++) {
            int sum = 0;
            for(int ky = -halfKernel; ky <= halfKernel; ky++) {
                const uchar* srcPtr = padded.ptr<uchar>(y + halfKernel + ky);
                for(int kx = -halfKernel; kx <= halfKernel; kx++) {
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
    Mat dst(src.size(), src.type());
    int halfKernel = kernelSize / 2;
    int windowSize = kernelSize * kernelSize;
    int medianPos = windowSize / 2;

    // 创建扩展图像
    Mat padded;
    copyMakeBorder(src, padded, halfKernel, halfKernel, halfKernel, halfKernel, BORDER_REFLECT);

    // 使用直方图方法
    vector<int> histogram(256);

    #pragma omp parallel for private(histogram)
    for(int y = 0; y < src.rows; y++) {
        uchar* dstRow = dst.ptr<uchar>(y);

        for(int x = 0; x < src.cols; x += BLOCK_SIZE) {
            const int blockEnd = min(x + BLOCK_SIZE, src.cols);

            for(int bx = x; bx < blockEnd; bx++) {
                // 清空直方图
                fill(histogram.begin(), histogram.end(), 0);

                // 构建直方图
                for(int ky = -halfKernel; ky <= halfKernel; ky++) {
                    const uchar* srcPtr = padded.ptr<uchar>(y + halfKernel + ky);
                    for(int kx = -halfKernel; kx <= halfKernel; kx++) {
                        histogram[srcPtr[bx + halfKernel + kx]]++;
                    }
                }

                // 查找中位数
                int count = 0;
                for(int i = 0; i < 256; i++) {
                    count += histogram[i];
                    if(count > medianPos) {
                        dstRow[bx] = static_cast<uchar>(i);
                        break;
                    }
                }
            }
        }
    }

    return dst;
}

// 使用分离卷积和SIMD优化的高斯滤波实现
Mat gaussianFilter_optimized(const Mat& src, int kernelSize, double sigma) {
    Mat dst(src.size(), src.type());
    int halfKernel = kernelSize / 2;

    // 预计算高斯核并转换为定点数
    vector<int> kernel1D(kernelSize);
    const int FIXED_POINT_SCALE = 1 << 14;
    float sum = 0.0f;

    for(int i = 0; i < kernelSize; i++) {
        float x = i - halfKernel;
        kernel1D[i] = static_cast<int>(exp(-(x*x)/(2*sigma*sigma)) * FIXED_POINT_SCALE);
        sum += kernel1D[i];
    }

    // 归一化
    for(int i = 0; i < kernelSize; i++) {
        kernel1D[i] = static_cast<int>((kernel1D[i] * FIXED_POINT_SCALE) / sum);
    }

    // 创建临时缓冲区，包含填充
    Mat temp(src.rows, src.cols + 2 * halfKernel, CV_16S);

    // 水平方向卷积（使用SSE4）
    #pragma omp parallel for
    for(int y = 0; y < src.rows; y++) {
        const uchar* srcRow = src.ptr<uchar>(y);
        short* tempRow = temp.ptr<short>(y) + halfKernel;

        // 使用SSE4处理每8个像素
        for(int x = 0; x <= src.cols - 8; x += 8) {
            __m128i sum = _mm_setzero_si128();

            for(int k = -halfKernel; k <= halfKernel; k++) {
                __m128i pixels = _mm_loadl_epi64((__m128i*)(srcRow + x + k));
                __m128i kernel_val = _mm_set1_epi16(kernel1D[k + halfKernel]);
                sum = _mm_add_epi16(sum, _mm_mullo_epi16(_mm_cvtepu8_epi16(pixels), kernel_val));
            }

            _mm_storeu_si128((__m128i*)(tempRow + x), sum);
        }

        // 处理剩余像素
        for(int x = (src.cols/8)*8; x < src.cols; x++) {
            int sum = 0;
            for(int k = -halfKernel; k <= halfKernel; k++) {
                sum += srcRow[x + k] * kernel1D[k + halfKernel];
            }
            tempRow[x] = static_cast<short>(sum >> 14);
        }
    }

    // 垂直方向卷积
    #pragma omp parallel for
    for(int y = halfKernel; y < src.rows - halfKernel; y++) {
        uchar* dstRow = dst.ptr<uchar>(y);

        for(int x = 0; x < src.cols; x += 8) {
            __m128i sum = _mm_setzero_si128();

            for(int k = -halfKernel; k <= halfKernel; k++) {
                const short* tempPtr = temp.ptr<short>(y + k) + x;
                __m128i pixels = _mm_loadu_si128((__m128i*)tempPtr);
                __m128i kernel_val = _mm_set1_epi16(kernel1D[k + halfKernel]);
                sum = _mm_add_epi32(sum, _mm_madd_epi16(pixels, kernel_val));
            }

            // 转换回8位
            sum = _mm_srai_epi32(sum, 14);
            __m128i result = _mm_packus_epi16(_mm_packs_epi32(sum, sum), _mm_setzero_si128());
            _mm_storel_epi64((__m128i*)(dstRow + x), result);
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
    #pragma omp parallel for collapse(2)
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

    #pragma omp parallel for collapse(2)
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

// 性能测试函数
void performanceTest(const Mat& img, int kernelSize) {
    cout << "\n性能测试报告 (kernel size = " << kernelSize << "):\n";
    cout << "----------------------------------------\n";

    const int REPEAT_COUNT = 10;  // 重复测试次数
    vector<double> times_custom(REPEAT_COUNT);
    vector<double> times_opencv(REPEAT_COUNT);

    // 均值滤波测试
    cout << "\n均值滤波测试:\n";

    // 预热
    Mat result1 = meanFilter_optimized(img, kernelSize);
    Mat result2;
    blur(img, result2, Size(kernelSize, kernelSize));

    // 测试优化版本
    for(int i = 0; i < REPEAT_COUNT; i++) {
        auto start = high_resolution_clock::now();
        result1 = meanFilter_optimized(img, kernelSize);
        auto end = high_resolution_clock::now();
        times_custom[i] = duration_cast<microseconds>(end - start).count() / 1000.0;
    }

    // 测试OpenCV版本
    for(int i = 0; i < REPEAT_COUNT; i++) {
        auto start = high_resolution_clock::now();
        blur(img, result2, Size(kernelSize, kernelSize));
        auto end = high_resolution_clock::now();
        times_opencv[i] = duration_cast<microseconds>(end - start).count() / 1000.0;
    }

    // 计算统计数据
    sort(times_custom.begin(), times_custom.end());
    sort(times_opencv.begin(), times_opencv.end());

    double median_custom = times_custom[REPEAT_COUNT/2];
    double median_opencv = times_opencv[REPEAT_COUNT/2];

    cout << "优化版本耗时(中位数): " << median_custom << "ms\n";
    cout << "OpenCV版本耗时(中位数): " << median_opencv << "ms\n";
    cout << "性能比: " << median_opencv/median_custom << "x\n";

    // 计算结果差异
    Mat diff;
    absdiff(result1, result2, diff);
    double maxDiff;
    minMaxLoc(diff, nullptr, &maxDiff);
    cout << "结果最大差异: " << maxDiff << "\n";

    // 其他滤波器的测试类似...
    // 为简洁起见，省略了中值滤波和高斯滤波的重复测试代码
}

// 使用示例
int main() {
    // 读取图像
    Mat img = imread("input.jpg", IMREAD_GRAYSCALE);
    if(img.empty()) {
        cout << "Error: Could not read the image." << endl;
        return -1;
    }

    // 设置OpenMP线程数
    int num_threads = thread::hardware_concurrency();
    omp_set_num_threads(num_threads);
    cout << "使用 " << num_threads << " 个线程进行并行计算\n";

    // 进行性能测试
    vector<int> kernelSizes = {3, 5, 7};
    for(int kernelSize : kernelSizes) {
        performanceTest(img, kernelSize);
    }

    // 显示结果对比
    int testKernelSize = 3;

    Mat meanResult_opencv, meanResult_custom;
    blur(img, meanResult_opencv, Size(testKernelSize, testKernelSize));
    meanResult_custom = meanFilter_optimized(img, testKernelSize);

    Mat medianResult_opencv, medianResult_custom;
    medianBlur(img, medianResult_opencv, testKernelSize);
    medianResult_custom = medianFilter_optimized(img, testKernelSize);

    Mat gaussianResult_opencv, gaussianResult_custom;
    GaussianBlur(img, gaussianResult_opencv, Size(testKernelSize, testKernelSize), 1.0);
    gaussianResult_custom = gaussianFilter_optimized(img, testKernelSize, 1.0);

    // 显示结果
    imshow("Original", img);
    imshow("Mean Filter (OpenCV)", meanResult_opencv);
    imshow("Mean Filter (Custom)", meanResult_custom);
    imshow("Median Filter (OpenCV)", medianResult_opencv);
    imshow("Median Filter (Custom)", medianResult_custom);
    imshow("Gaussian Filter (OpenCV)", gaussianResult_opencv);
    imshow("Gaussian Filter (Custom)", gaussianResult_custom);

    waitKey(0);
    return 0;
}