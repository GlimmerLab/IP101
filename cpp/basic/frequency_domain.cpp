#include "frequency_domain.hpp"
#include <cmath>

namespace ip101 {

using namespace cv;
using namespace std;

namespace {
// 内部常量定义
constexpr int CACHE_LINE = 64;    // CPU缓存行大小(字节)
constexpr int SIMD_WIDTH = 32;    // AVX2 SIMD向量宽度(字节)
constexpr int BLOCK_SIZE = 16;    // 分块处理大小
constexpr double PI = 3.14159265358979323846;

// 内存对齐辅助函数
inline uchar* alignPtr(uchar* ptr, size_t align = CACHE_LINE) {
    return (uchar*)(((size_t)ptr + align - 1) & -align);
}

// 获取最优DFT尺寸
int getOptimalDFTSize(int size) {
    int result = 1;
    while (result < size) {
        result *= 2;
    }
    return result;
}

// 计算复数的幅值
inline double magnitude(const complex<double>& c) {
    return abs(c);
}

// 计算复数的相位
inline double phase(const complex<double>& c) {
    return arg(c);
}

} // anonymous namespace

void fourier_transform_manual(const Mat& src, Mat& dst, int flags) {
    CV_Assert(!src.empty());

    // 转换为灰度图
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // 扩展图像到最优DFT尺寸
    Mat padded;
    int m = getOptimalDFTSize(gray.rows);
    int n = getOptimalDFTSize(gray.cols);
    copyMakeBorder(gray, padded, 0, m - gray.rows, 0, n - gray.cols,
                   BORDER_CONSTANT, Scalar::all(0));

    // 创建复数矩阵
    vector<vector<complex<double>>> complexImg(m, vector<complex<double>>(n));

    // 转换为复数并乘以(-1)^(x+y)以移动频谱中心
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < m; y++) {
        for (int x = 0; x < n; x++) {
            double val = padded.at<uchar>(y, x);
            double sign = ((x + y) % 2 == 0) ? 1.0 : -1.0;
            complexImg[y][x] = sign * complex<double>(val, 0);
        }
    }

    // 行方向FFT
    #pragma omp parallel for
    for (int y = 0; y < m; y++) {
        fft(complexImg[y], n, flags == DFT_INVERSE);
    }

    // 转置矩阵
    vector<vector<complex<double>>> transposed(n, vector<complex<double>>(m));
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < m; y++) {
        for (int x = 0; x < n; x++) {
            transposed[x][y] = complexImg[y][x];
        }
    }

    // 列方向FFT
    #pragma omp parallel for
    for (int x = 0; x < n; x++) {
        fft(transposed[x], m, flags == DFT_INVERSE);
    }

    // 转回原始方向
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < m; y++) {
        for (int x = 0; x < n; x++) {
            complexImg[y][x] = transposed[x][y];
        }
    }

    // 创建输出矩阵
    if (flags == DFT_COMPLEX_OUTPUT) {
        vector<Mat> planes = {
            Mat::zeros(m, n, CV_64F),
            Mat::zeros(m, n, CV_64F)
        };

        #pragma omp parallel for collapse(2)
        for (int y = 0; y < m; y++) {
            for (int x = 0; x < n; x++) {
                planes[0].at<double>(y, x) = complexImg[y][x].real();
                planes[1].at<double>(y, x) = complexImg[y][x].imag();
            }
        }

        merge(planes, dst);
    } else {
        dst.create(m, n, CV_64F);
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < m; y++) {
            for (int x = 0; x < n; x++) {
                dst.at<double>(y, x) = magnitude(complexImg[y][x]);
            }
        }
    }
}

Mat create_frequency_filter(const Size& size,
                          double cutoff_freq,
                          const string& filter_type) {
    Mat filter = Mat::zeros(size, CV_64F);
    Point center(size.width/2, size.height/2);
    double radius2 = cutoff_freq * cutoff_freq;

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < size.height; y++) {
        for (int x = 0; x < size.width; x++) {
            double distance2 = pow(x - center.x, 2) + pow(y - center.y, 2);

            if (filter_type == "lowpass") {
                filter.at<double>(y, x) = exp(-distance2 / (2 * radius2));
            } else if (filter_type == "highpass") {
                filter.at<double>(y, x) = 1.0 - exp(-distance2 / (2 * radius2));
            } else if (filter_type == "bandpass") {
                double r1 = radius2 * 0.5;  // 内圈半径
                double r2 = radius2 * 2.0;  // 外圈半径
                if (distance2 >= r1 && distance2 <= r2) {
                    filter.at<double>(y, x) = 1.0;
                }
            }
        }
    }

    return filter;
}

void frequency_filter_manual(const Mat& src, Mat& dst,
                           const string& filter_type,
                           double cutoff_freq) {
    // 执行傅里叶变换
    Mat dft_result;
    fourier_transform_manual(src, dft_result, DFT_COMPLEX_OUTPUT);

    // 创建滤波器
    Mat filter = create_frequency_filter(dft_result.size(), cutoff_freq, filter_type);

    // 应用滤波器
    vector<Mat> planes;
    split(dft_result, planes);

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < dft_result.rows; y++) {
        for (int x = 0; x < dft_result.cols; x++) {
            double f = filter.at<double>(y, x);
            planes[0].at<double>(y, x) *= f;
            planes[1].at<double>(y, x) *= f;
        }
    }

    merge(planes, dft_result);

    // 执行逆傅里叶变换
    fourier_transform_manual(dft_result, dst, DFT_INVERSE);
}

void dct_transform_manual(const Mat& src, Mat& dst, int flags) {
    CV_Assert(!src.empty());

    // 转换为灰度图并归一化
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }
    gray.convertTo(gray, CV_64F);

    int m = gray.rows;
    int n = gray.cols;
    dst.create(m, n, CV_64F);

    if (flags == DCT_FORWARD) {
        #pragma omp parallel for collapse(2)
        for (int u = 0; u < m; u++) {
            for (int v = 0; v < n; v++) {
                double cu = (u == 0) ? 1.0/sqrt(2.0) : 1.0;
                double cv = (v == 0) ? 1.0/sqrt(2.0) : 1.0;
                double sum = 0.0;

                for (int x = 0; x < m; x++) {
                    for (int y = 0; y < n; y++) {
                        double val = gray.at<double>(x, y);
                        double cos1 = cos((2*x + 1) * u * PI / (2*m));
                        double cos2 = cos((2*y + 1) * v * PI / (2*n));
                        sum += val * cos1 * cos2;
                    }
                }

                dst.at<double>(u, v) = cu * cv * sum * 2.0/sqrt(m*n);
            }
        }
    } else {  // DCT_INVERSE
        #pragma omp parallel for collapse(2)
        for (int x = 0; x < m; x++) {
            for (int y = 0; y < n; y++) {
                double sum = 0.0;

                for (int u = 0; u < m; u++) {
                    for (int v = 0; v < n; v++) {
                        double cu = (u == 0) ? 1.0/sqrt(2.0) : 1.0;
                        double cv = (v == 0) ? 1.0/sqrt(2.0) : 1.0;
                        double val = gray.at<double>(u, v);
                        double cos1 = cos((2*x + 1) * u * PI / (2*m));
                        double cos2 = cos((2*y + 1) * v * PI / (2*n));
                        sum += cu * cv * val * cos1 * cos2;
                    }
                }

                dst.at<double>(x, y) = sum * 2.0/sqrt(m*n);
            }
        }
    }
}

void wavelet_transform_manual(const Mat& src, Mat& dst,
                            const string& wavelet_type,
                            int level) {
    CV_Assert(!src.empty());

    // 转换为灰度图
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }
    gray.convertTo(gray, CV_64F);

    // 创建小波滤波器
    vector<double> lowpass, highpass;
    if (wavelet_type == "haar") {
        lowpass = {1/sqrt(2), 1/sqrt(2)};
        highpass = {1/sqrt(2), -1/sqrt(2)};
    } else if (wavelet_type == "db1") {
        lowpass = {1/sqrt(2), 1/sqrt(2)};
        highpass = {-1/sqrt(2), 1/sqrt(2)};
    } else {
        // 默认使用Haar小波
        lowpass = {1/sqrt(2), 1/sqrt(2)};
        highpass = {1/sqrt(2), -1/sqrt(2)};
    }

    dst = gray.clone();
    int rows = dst.rows;
    int cols = dst.cols;

    // 多层小波分解
    for (int l = 0; l < level; l++) {
        Mat temp = dst(Rect(0, 0, cols, rows)).clone();

        // 行变换
        #pragma omp parallel for
        for (int y = 0; y < rows; y++) {
            vector<double> row(cols);
            for (int x = 0; x < cols; x += 2) {
                double v1 = temp.at<double>(y, x);
                double v2 = temp.at<double>(y, x+1);

                // 低频部分
                row[x/2] = v1 * lowpass[0] + v2 * lowpass[1];
                // 高频部分
                row[cols/2 + x/2] = v1 * highpass[0] + v2 * highpass[1];
            }

            for (int x = 0; x < cols; x++) {
                dst.at<double>(y, x) = row[x];
            }
        }

        // 列变换
        temp = dst(Rect(0, 0, cols, rows)).clone();
        #pragma omp parallel for
        for (int x = 0; x < cols; x++) {
            vector<double> col(rows);
            for (int y = 0; y < rows; y += 2) {
                double v1 = temp.at<double>(y, x);
                double v2 = temp.at<double>(y+1, x);

                // 低频部分
                col[y/2] = v1 * lowpass[0] + v2 * lowpass[1];
                // 高频部分
                col[rows/2 + y/2] = v1 * highpass[0] + v2 * highpass[1];
            }

            for (int y = 0; y < rows; y++) {
                dst.at<double>(y, x) = col[y];
            }
        }

        rows /= 2;
        cols /= 2;
    }
}

void visualize_spectrum(const Mat& spectrum, Mat& dst) {
    CV_Assert(spectrum.type() == CV_64FC2);

    vector<Mat> planes;
    split(spectrum, planes);

    // 计算幅度谱
    Mat magnitude;
    magnitude.create(spectrum.size(), CV_64F);

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < spectrum.rows; y++) {
        for (int x = 0; x < spectrum.cols; x++) {
            double real = planes[0].at<double>(y, x);
            double imag = planes[1].at<double>(y, x);
            magnitude.at<double>(y, x) = log(1 + sqrt(real*real + imag*imag));
        }
    }

    // 归一化到0-255
    normalize(magnitude, dst, 0, 255, NORM_MINMAX);
    dst.convertTo(dst, CV_8U);
}

// 辅助函数：一维FFT
void fft(vector<complex<double>>& data, int n, bool inverse) {
    int bits = 0;
    while ((1 << bits) < n) bits++;

    // 位反转排序
    for (int i = 1; i < n; i++) {
        int j = 0;
        for (int k = 0; k < bits; k++) {
            if (i & (1 << k)) {
                j |= (1 << (bits - 1 - k));
            }
        }
        if (i < j) {
            swap(data[i], data[j]);
        }
    }

    // FFT计算
    for (int len = 2; len <= n; len *= 2) {
        double angle = 2 * PI / len * (inverse ? 1 : -1);
        complex<double> wlen(cos(angle), sin(angle));

        #pragma omp parallel for
        for (int i = 0; i < n; i += len) {
            complex<double> w(1);
            for (int j = 0; j < len/2; j++) {
                complex<double> u = data[i + j];
                complex<double> v = data[i + j + len/2] * w;
                data[i + j] = u + v;
                data[i + j + len/2] = u - v;
                w *= wlen;
            }
        }
    }

    if (inverse) {
        for (int i = 0; i < n; i++) {
            data[i] /= n;
        }
    }
}

} // namespace ip101