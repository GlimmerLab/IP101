#include "retinex_msrcr.hpp"
#include <cmath>
#include <omp.h>
#include <algorithm>
#include <vector>

namespace ip101 {
namespace advanced {

/**
 * @brief 手动实现高斯核生成
 * @param size 核大小
 * @param sigma 高斯标准差
 * @return 高斯核矩阵
 */
cv::Mat createGaussianKernel(int size, double sigma) {
    // 确保核大小为奇数
    size = size | 1;  // 如果是偶数，加1使其变为奇数
    cv::Mat kernel(size, size, CV_32F);
    int center = size / 2;
    double sum = 0.0;
    double twoSigmaSquare = 2.0 * sigma * sigma;

    // 预先计算指数部分，避免重复计算
    std::vector<double> expValues(size);
    for (int i = 0; i < size; i++) {
        double x = i - center;
        expValues[i] = std::exp(-(x*x) / twoSigmaSquare);
    }

    // 计算高斯核
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int i = 0; i < size; i++) {
        float* kernelRow = kernel.ptr<float>(i);
        double y = i - center;
        double expY = expValues[i];

        for (int j = 0; j < size; j++) {
            double expX = expValues[j];
            // 利用高斯核的可分离性质
            kernelRow[j] = static_cast<float>(expY * expX);
            sum += kernelRow[j];
        }
    }

    // 归一化
    if (sum != 0) {
        float scale = static_cast<float>(1.0 / sum);
        kernel *= scale;
    }
    return kernel;
}

/**
 * @brief 创建一维高斯核（用于分离卷积）
 * @param size 核大小
 * @param sigma 高斯标准差
 * @return 一维高斯核
 */
std::vector<float> createGaussianKernel1D(int size, double sigma) {
    std::vector<float> kernel(size);
    int center = size / 2;
    double sum = 0.0;
    double twoSigmaSquare = 2.0 * sigma * sigma;

    for (int i = 0; i < size; i++) {
        double x = i - center;
        kernel[i] = std::exp(-(x*x) / twoSigmaSquare);
        sum += kernel[i];
    }

    // 归一化
    if (sum != 0) {
        float scale = static_cast<float>(1.0 / sum);
        for (int i = 0; i < size; i++) {
            kernel[i] *= scale;
        }
    }

    return kernel;
}

/**
 * @brief 手动实现卷积操作（优化版）
 * @param src 输入图像
 * @param kernel 卷积核
 * @return 卷积结果
 */
cv::Mat convolve(const cv::Mat& src, const cv::Mat& kernel) {
    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
    int ksize = kernel.rows;
    int kRadius = ksize / 2;

    // 对图像进行卷积，使用并行处理
    #pragma omp parallel for schedule(dynamic)
    for (int i = kRadius; i < src.rows - kRadius; i++) {
        const float* kernelPtr = kernel.ptr<float>(0);

        for (int j = kRadius; j < src.cols - kRadius; j++) {
            float sum = 0.0f;

            // 优化内存访问模式，减少缓存未命中
            for (int ki = 0; ki < ksize; ki++) {
                const float* srcRow = src.ptr<float>(i + ki - kRadius);
                const float* kernelRow = kernel.ptr<float>(ki);

                for (int kj = 0; kj < ksize; kj++) {
                    sum += srcRow[j + kj - kRadius] * kernelRow[kj];
                }
            }

            dst.at<float>(i, j) = sum;
        }
    }

    // 处理边界（简单复制边缘像素）
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            for (int i = 0; i < src.rows; i++) {
                for (int j = 0; j < kRadius; j++) {
                    dst.at<float>(i, j) = src.at<float>(i, kRadius);
                }
            }
        }

        #pragma omp section
        {
            for (int i = 0; i < src.rows; i++) {
                for (int j = 0; j < kRadius; j++) {
                    dst.at<float>(i, src.cols - 1 - j) = src.at<float>(i, src.cols - 1 - kRadius);
                }
            }
        }

        #pragma omp section
        {
            for (int j = 0; j < src.cols; j++) {
                for (int i = 0; i < kRadius; i++) {
                    dst.at<float>(i, j) = src.at<float>(kRadius, j);
                }
            }
        }

        #pragma omp section
        {
            for (int j = 0; j < src.cols; j++) {
                for (int i = 0; i < kRadius; i++) {
                    dst.at<float>(src.rows - 1 - i, j) = src.at<float>(src.rows - 1 - kRadius, j);
                }
            }
        }
    }

    return dst;
}

/**
 * @brief 手动实现高斯模糊（使用分离卷积加速）
 * @param src 输入图像
 * @param sigma 高斯标准差
 * @return 模糊后的图像
 */
cv::Mat gaussianBlur(const cv::Mat& src, double sigma) {
    // 根据sigma计算合适的核大小（一般取6*sigma）
    int ksize = std::max(3, (int)(6 * sigma + 1));
    ksize = ksize | 1;  // 确保为奇数

    // 创建1D高斯核（用于分离卷积）
    std::vector<float> kernel1D = createGaussianKernel1D(ksize, sigma);
    int kRadius = ksize / 2;

    // 先进行水平方向卷积
    cv::Mat temp = cv::Mat::zeros(src.size(), src.type());

    // 水平方向卷积 - 并行处理
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < src.rows; i++) {
        const float* srcRow = src.ptr<float>(i);
        float* tempRow = temp.ptr<float>(i);

        // 处理非边界区域
        for (int j = kRadius; j < src.cols - kRadius; j++) {
            float sum = 0.0f;

            // 使用连续内存访问模式
            for (int k = -kRadius; k <= kRadius; k++) {
                sum += srcRow[j + k] * kernel1D[k + kRadius];
            }

            tempRow[j] = sum;
        }

        // 处理左边界
        for (int j = 0; j < kRadius; j++) {
            tempRow[j] = tempRow[kRadius];
        }

        // 处理右边界
        for (int j = src.cols - kRadius; j < src.cols; j++) {
            tempRow[j] = tempRow[src.cols - kRadius - 1];
        }
    }

    // 垂直方向卷积
    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());

    // 垂直方向卷积 - 并行处理
    #pragma omp parallel for schedule(dynamic, 16)
    for (int j = 0; j < src.cols; j++) {
        // 处理非边界区域
        for (int i = kRadius; i < src.rows - kRadius; i++) {
            float sum = 0.0f;

            // 优化内存访问，但垂直方向访问不连续
            for (int k = -kRadius; k <= kRadius; k++) {
                sum += temp.ptr<float>(i + k)[j] * kernel1D[k + kRadius];
            }

            dst.ptr<float>(i)[j] = sum;
        }

        // 处理上边界
        for (int i = 0; i < kRadius; i++) {
            dst.ptr<float>(i)[j] = dst.ptr<float>(kRadius)[j];
        }

        // 处理下边界
        for (int i = src.rows - kRadius; i < src.rows; i++) {
            dst.ptr<float>(i)[j] = dst.ptr<float>(src.rows - kRadius - 1)[j];
        }
    }

    return dst;
}

/**
 * @brief 手动实现对数运算（优化版）
 * @param src 输入图像
 * @return 对数结果
 */
cv::Mat logarithm(const cv::Mat& src) {
    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());

    // 使用并行处理
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < src.rows; i++) {
        const float* srcRow = src.ptr<float>(i);
        float* dstRow = dst.ptr<float>(i);

        // 连续内存访问，提高缓存命中率
        for (int j = 0; j < src.cols; j++) {
            dstRow[j] = std::log(srcRow[j]);
        }
    }

    return dst;
}

/**
 * @brief 手动实现指数运算（优化版）
 * @param src 输入图像
 * @return 指数结果
 */
cv::Mat exponential(const cv::Mat& src) {
    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());

    // 使用并行处理
    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < src.rows; i++) {
        const float* srcRow = src.ptr<float>(i);
        float* dstRow = dst.ptr<float>(i);

        // 连续内存访问，提高缓存命中率
        for (int j = 0; j < src.cols; j++) {
            dstRow[j] = std::exp(srcRow[j]);
        }
    }

    return dst;
}

/**
 * @brief 优化的矩阵加法
 * @param src1 输入矩阵1
 * @param src2 输入矩阵2
 * @return 结果矩阵
 */
cv::Mat matAdd(const cv::Mat& src1, const cv::Mat& src2) {
    CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());
    cv::Mat dst = cv::Mat::zeros(src1.size(), src1.type());

    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < src1.rows; i++) {
        const float* src1Row = src1.ptr<float>(i);
        const float* src2Row = src2.ptr<float>(i);
        float* dstRow = dst.ptr<float>(i);

        // 使用向量化操作处理每一行
        for (int j = 0; j < src1.cols; j++) {
            dstRow[j] = src1Row[j] + src2Row[j];
        }
    }

    return dst;
}

/**
 * @brief 优化的矩阵减法
 * @param src1 输入矩阵1
 * @param src2 输入矩阵2
 * @return 结果矩阵
 */
cv::Mat matSubtract(const cv::Mat& src1, const cv::Mat& src2) {
    CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());
    cv::Mat dst = cv::Mat::zeros(src1.size(), src1.type());

    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < src1.rows; i++) {
        const float* src1Row = src1.ptr<float>(i);
        const float* src2Row = src2.ptr<float>(i);
        float* dstRow = dst.ptr<float>(i);

        // 使用向量化操作处理每一行
        for (int j = 0; j < src1.cols; j++) {
            dstRow[j] = src1Row[j] - src2Row[j];
        }
    }

    return dst;
}

/**
 * @brief 优化的矩阵乘法（元素级别）
 * @param src 输入矩阵
 * @param scale 缩放因子
 * @return 结果矩阵
 */
cv::Mat matMultiply(const cv::Mat& src, float scale) {
    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());

    #pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < src.rows; i++) {
        const float* srcRow = src.ptr<float>(i);
        float* dstRow = dst.ptr<float>(i);

        // 使用向量化操作处理每一行
        for (int j = 0; j < src.cols; j++) {
            dstRow[j] = srcRow[j] * scale;
        }
    }

    return dst;
}

void retinex_msrcr(const cv::Mat& src, cv::Mat& dst,
                  double sigma1, double sigma2, double sigma3,
                  double alpha, double beta, double gain) {
    CV_Assert(!src.empty() && src.channels() == 3);

    // 设置OpenMP线程数量，根据可用处理器核心数
    int numThreads = std::min(12, omp_get_max_threads());
    omp_set_num_threads(numThreads);

    // 转换到对数域
    cv::Mat log_src;
    src.convertTo(log_src, CV_32F);
    log_src += 1.0;
    // 使用手动实现的对数函数
    log_src = logarithm(log_src);

    // 分离通道
    std::vector<cv::Mat> channels;
    cv::split(log_src, channels);

    // 预先计算高斯模糊结果，避免重复计算
    std::vector<std::vector<cv::Mat>> gaussian_blurred(3);
    for (int c = 0; c < 3; c++) {
        gaussian_blurred[c].resize(3);
    }

    // 并行计算所有通道的高斯模糊
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int c = 0; c < 3; c++) {
        for (int s = 0; s < 3; s++) {
            double sigma = (s == 0) ? sigma1 : ((s == 1) ? sigma2 : sigma3);
            gaussian_blurred[c][s] = gaussianBlur(channels[c], sigma);
        }
    }

    // 对每个通道进行多尺度Retinex处理
    std::vector<cv::Mat> msr_channels(3);
    #pragma omp parallel for
    for(int c = 0; c < 3; c++) {
        cv::Mat msr = cv::Mat::zeros(src.size(), CV_32F);

        // 多尺度高斯滤波（使用预计算结果）
        for(int s = 0; s < 3; s++) {
            // 使用优化的矩阵减法
            cv::Mat diff = matSubtract(channels[c], gaussian_blurred[c][s]);
            // 使用优化的矩阵加法
            msr = matAdd(msr, diff);
        }

        // 取平均
        msr = matMultiply(msr, 1.0f/3.0f);

        // 颜色恢复（使用手动实现的对数函数和优化的矩阵操作）
        cv::Mat temp1 = matAdd(channels[c], cv::Mat::ones(channels[c].size(), CV_32F));
        cv::Mat temp2 = matAdd(msr, cv::Mat::ones(msr.size(), CV_32F));
        cv::Mat log_temp1 = logarithm(temp1);
        cv::Mat log_temp2 = logarithm(temp2);
        cv::Mat diff = matSubtract(log_temp1, log_temp2);
        msr_channels[c] = matAdd(matMultiply(diff, alpha), cv::Mat::ones(diff.size(), CV_32F) * beta);
    }

    // 合并通道
    cv::Mat msr_result;
    cv::merge(msr_channels, msr_result);

    // 增益调整和归一化
    msr_result = matMultiply(msr_result, gain);
    // 使用手动实现的指数函数
    msr_result = exponential(msr_result);
    msr_result -= 1.0;

    // 裁剪到[0, 255]范围
    #pragma omp parallel for
    for (int i = 0; i < msr_result.rows; i++) {
        float* row = msr_result.ptr<float>(i);
        for (int j = 0; j < msr_result.cols; j++) {
            for (int c = 0; c < 3; c++) {
                int idx = j * 3 + c;
                row[idx] = std::min(std::max(row[idx], 0.0f), 1.0f);
            }
        }
    }

    msr_result.convertTo(dst, CV_8UC3, 255.0);
}

} // namespace advanced
} // namespace ip101