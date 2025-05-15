#include <basic/edge_detection.hpp>
#include <immintrin.h>  // 用于SIMD指令(AVX2)
#include <omp.h>        // 用于OpenMP并行计算
#include <cmath>

namespace ip101 {

using namespace cv;  // 在命名空间开始处声明使用cv命名空间

namespace {
// 内部常量定义
constexpr int CACHE_LINE = 64;    // CPU缓存行大小(字节)
constexpr int SIMD_WIDTH = 32;    // AVX2 SIMD向量宽度(字节)
constexpr int BLOCK_SIZE = 16;    // 分块处理大小，用于优化缓存访问

/**
 * @brief 内存对齐辅助函数
 * @param ptr 需要对齐的指针
 * @param align 对齐字节数，默认为缓存行大小
 * @return 对齐后的指针
 * @note 用于确保内存访问对齐到缓存行边界，提高内存访问效率
 */
inline uchar* alignPtr(uchar* ptr, size_t align = CACHE_LINE) {
    return (uchar*)(((size_t)ptr + align - 1) & ~(align - 1));
}

/**
 * @brief SIMD优化的边缘检测核心计算函数
 * @param padded 填充后的输入图像
 * @param result 输出结果图像
 * @param kernel_x x方向卷积核
 * @param kernel_y y方向卷积核
 * @param y 当前处理的y坐标
 * @param x 当前处理的x坐标
 * @param kernel_size 卷积核大小
 * @note 使用AVX2指令集优化3x3卷积运算，对于其他大小使用普通实现
 */
void process_block_simd(const Mat& padded, Mat& result,
                       const Mat& kernel_x, const Mat& kernel_y,
                       int y, int x, int kernel_size) {
    float gx = 0.0f, gy = 0.0f;

    // 使用AVX2进行并行计算
    if (kernel_size == 3) {
        // 初始化向量寄存器
        __m256 sum_x = _mm256_setzero_ps();  // 8个单精度浮点数的向量
        __m256 sum_y = _mm256_setzero_ps();

        // 3x3卷积核计算
        for (int ky = 0; ky < 3; ++ky) {
            for (int kx = 0; kx < 3; ++kx) {
                float val = padded.at<uchar>(y + ky, x + kx);
                float kx_val = kernel_x.at<float>(ky, kx);
                float ky_val = kernel_y.at<float>(ky, kx);

                // 向量乘法和累加
                sum_x = _mm256_add_ps(sum_x, _mm256_mul_ps(_mm256_set1_ps(val), _mm256_set1_ps(kx_val)));
                sum_y = _mm256_add_ps(sum_y, _mm256_mul_ps(_mm256_set1_ps(val), _mm256_set1_ps(ky_val)));
            }
        }

        // 水平相加得到最终结果
        float gx_arr[8], gy_arr[8];
        _mm256_store_ps(gx_arr, sum_x);
        _mm256_store_ps(gy_arr, sum_y);

        for (int i = 0; i < 8; ++i) {
            gx += gx_arr[i];
            gy += gy_arr[i];
        }
    } else {
        // 对于非3x3的kernel使用普通实现
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                float val = padded.at<uchar>(y + ky, x + kx);
                gx += val * kernel_x.at<float>(ky, kx);
                gy += val * kernel_y.at<float>(ky, kx);
            }
        }
    }

    // 计算梯度幅值并饱和到uchar范围
    result.at<uchar>(y, x) = saturate_cast<uchar>(std::sqrt(gx * gx + gy * gy));
}

// 内存对齐辅助函数
inline size_t get_aligned_size(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

// SIMD优化的像素处理
inline void process_pixels_avx2(__m256i* dst, const __m256i* src, const float* kernel, int kernel_size) {
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < kernel_size; ++i) {
        __m256 val = _mm256_cvtepi32_ps(_mm256_loadu_si256(src + i));
        __m256 k = _mm256_broadcast_ss(&kernel[i]);
        sum = _mm256_fmadd_ps(val, k, sum);
    }
    *dst = _mm256_cvtps_epi32(sum);
}

} // anonymous namespace

/**
 * @brief 微分滤波边缘检测实现
 */
void differential_filter(const cv::Mat& src, cv::Mat& dst, int dx, int dy, int ksize) {
    CV_Assert(!src.empty() && src.type() == CV_8UC1);

    dst = Mat::zeros(src.size(), CV_8UC1);
    int pad = ksize / 2;

    // 边缘填充，使用边缘像素值填充
    Mat padded;
    copyMakeBorder(src, padded, pad, pad, pad, pad, BORDER_REPLICATE);

    // 定义微分算子
    Mat kernel_x = (Mat_<float>(3, 3) << 0, 0, 0, -1, 0, 1, 0, 0, 0);
    Mat kernel_y = (Mat_<float>(3, 3) << 0, -1, 0, 0, 0, 0, 0, 1, 0);

    // 使用OpenMP进行并行计算
    #pragma omp parallel for
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            process_block_simd(padded, dst, kernel_x, kernel_y, y, x, ksize);
        }
    }
}

/**
 * @brief Sobel算子边缘检测实现
 */
void sobel_filter(const cv::Mat& src, cv::Mat& dst, int dx, int dy, int ksize, double scale) {
    CV_Assert(!src.empty() && src.type() == CV_8UC1);

    dst = Mat::zeros(src.size(), CV_8UC1);
    int pad = ksize / 2;

    // 边缘填充
    Mat padded;
    copyMakeBorder(src, padded, pad, pad, pad, pad, BORDER_REPLICATE);

    // 定义Sobel算子
    Mat kernel_x = (Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    Mat kernel_y = (Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

    // 使用OpenMP进行并行计算
    #pragma omp parallel for
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            process_block_simd(padded, dst, kernel_x, kernel_y, y, x, ksize);
        }
    }

    // 应用缩放因子
    if (scale != 1.0) {
        dst = dst * scale;
    }
}

/**
 * @brief Prewitt算子边缘检测实现
 */
void prewitt_filter(const cv::Mat& src, cv::Mat& dst, int dx, int dy) {
    CV_Assert(!src.empty() && src.type() == CV_8UC1);

    dst = Mat::zeros(src.size(), CV_8UC1);
    int ksize = 3; // Prewitt算子固定为3x3
    int pad = ksize / 2;

    // 边缘填充
    Mat padded;
    copyMakeBorder(src, padded, pad, pad, pad, pad, BORDER_REPLICATE);

    // 定义Prewitt算子
    Mat kernel_x = (Mat_<float>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
    Mat kernel_y = (Mat_<float>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);

    // 使用OpenMP进行并行计算
    #pragma omp parallel for
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            process_block_simd(padded, dst, kernel_x, kernel_y, y, x, ksize);
        }
    }
}

/**
 * @brief Laplacian算子边缘检测实现
 */
void laplacian_filter(const cv::Mat& src, cv::Mat& dst, int ksize, double scale) {
    CV_Assert(!src.empty() && src.type() == CV_8UC1);

    dst = Mat::zeros(src.size(), CV_8UC1);
    int pad = ksize / 2;

    // 边缘填充
    Mat padded;
    copyMakeBorder(src, padded, pad, pad, pad, pad, BORDER_REPLICATE);

    // 定义Laplacian算子
    Mat kernel = (Mat_<float>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
    Mat kernel_x = kernel.clone(); // 为了兼容process_block_simd函数
    Mat kernel_y = kernel.clone();

    // 使用OpenMP进行并行计算
    #pragma omp parallel for
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            float sum = 0.0f;

            // 对于非3x3的kernel使用普通实现
            for (int ky = 0; ky < ksize; ++ky) {
                for (int kx = 0; kx < ksize; ++kx) {
                    float val = padded.at<uchar>(y + ky, x + kx);
                    sum += val * kernel.at<float>(ky % 3, kx % 3); // 使用模运算确保索引在有效范围内
                }
            }

            // 取绝对值并饱和到uchar范围
            dst.at<uchar>(y, x) = saturate_cast<uchar>(std::abs(sum) * scale);
        }
    }
}

/**
 * @brief 浮雕效果实现
 */
void emboss_effect(const cv::Mat& src, cv::Mat& dst, int direction) {
    CV_Assert(!src.empty() && src.type() == CV_8UC1);

    dst = Mat::zeros(src.size(), CV_8UC1);
    int ksize = 3; // 浮雕效果固定使用3x3卷积核
    int pad = ksize / 2;
    int offset = 128; // 默认偏移值

    // 边缘填充
    Mat padded;
    copyMakeBorder(src, padded, pad, pad, pad, pad, BORDER_REPLICATE);

    // 根据方向选择浮雕算子
    Mat kernel;
    switch (direction) {
        case 0: // 默认方向（右下）
            kernel = (Mat_<float>(3, 3) << 2, 0, 0, 0, -1, 0, 0, 0, -1);
            break;
        case 1: // 右
            kernel = (Mat_<float>(3, 3) << 0, 0, 2, 0, -1, 0, 0, 0, -1);
            break;
        case 2: // 右上
            kernel = (Mat_<float>(3, 3) << 0, 0, 2, 0, -1, 0, -1, 0, 0);
            break;
        case 3: // 上
            kernel = (Mat_<float>(3, 3) << 0, 2, 0, 0, -1, 0, 0, -1, 0);
            break;
        case 4: // 左上
            kernel = (Mat_<float>(3, 3) << 2, 0, 0, 0, -1, 0, 0, 0, -1);
            kernel = kernel.t(); // 转置
            break;
        case 5: // 左
            kernel = (Mat_<float>(3, 3) << 0, 0, -1, 0, -1, 0, 2, 0, 0);
            break;
        case 6: // 左下
            kernel = (Mat_<float>(3, 3) << -1, 0, 0, 0, -1, 0, 0, 0, 2);
            break;
        case 7: // 下
            kernel = (Mat_<float>(3, 3) << 0, -1, 0, 0, -1, 0, 0, 2, 0);
            break;
        default:
            kernel = (Mat_<float>(3, 3) << 2, 0, 0, 0, -1, 0, 0, 0, -1);
            break;
    }

    // 使用OpenMP进行并行计算
    #pragma omp parallel for
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            float sum = 0.0f;

            // 计算卷积
            for (int ky = 0; ky < ksize; ++ky) {
                for (int kx = 0; kx < ksize; ++kx) {
                    float val = padded.at<uchar>(y + ky, x + kx);
                    sum += val * kernel.at<float>(ky, kx);
                }
            }

            // 添加偏移并饱和到uchar范围
            dst.at<uchar>(y, x) = saturate_cast<uchar>(sum + offset);
        }
    }
}

/**
 * @brief 综合边缘检测实现
 */
void edge_detection(const cv::Mat& src, cv::Mat& dst, const std::string& method, double thresh_val) {
    CV_Assert(!src.empty());

    // 转换为灰度图
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // 根据选择的方法进行边缘检测
    Mat result;
    if (method == "sobel") {
        sobel_filter(gray, result, 1, 1, 3, 1.0); // dx=1, dy=1, ksize=3, scale=1.0
    } else if (method == "prewitt") {
        prewitt_filter(gray, result, 1, 1); // dx=1, dy=1
    } else if (method == "laplacian") {
        laplacian_filter(gray, result, 3, 1.0); // ksize=3, scale=1.0
    } else {
        throw std::invalid_argument("Unsupported method: " + method);
    }

    // 二值化处理
    threshold(result, dst, thresh_val, 255, THRESH_BINARY);
}

} // namespace ip101