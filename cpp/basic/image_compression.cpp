#include "image_compression.hpp"
#include <cmath>
#include <algorithm>

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

// JPEG量化表(亮度)
const int JPEG_LUMINANCE_QUANTIZATION[64] = {
    16,  11,  10,  16,  24,  40,  51,  61,
    12,  12,  14,  19,  26,  58,  60,  55,
    14,  13,  16,  24,  40,  57,  69,  56,
    14,  17,  22,  29,  51,  87,  80,  62,
    18,  22,  37,  56,  68, 109, 103,  77,
    24,  35,  55,  64,  81, 104, 113,  92,
    49,  64,  78,  87, 103, 121, 120, 101,
    72,  92,  95,  98, 112, 100, 103,  99
};

// JPEG量化表(色度)
const int JPEG_CHROMINANCE_QUANTIZATION[64] = {
    17,  18,  24,  47,  99,  99,  99,  99,
    18,  21,  26,  66,  99,  99,  99,  99,
    24,  26,  56,  99,  99,  99,  99,  99,
    47,  66,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99
};

// DCT基函数查找表
Mat DCT_BASIS[64];

// 初始化DCT基函数查找表
void initDCTBasis() {
    static bool initialized = false;
    if (initialized) return;

    for (int u = 0; u < 8; u++) {
        for (int v = 0; v < 8; v++) {
            Mat basis = Mat::zeros(8, 8, CV_64F);
            double cu = (u == 0) ? 1.0/sqrt(2.0) : 1.0;
            double cv = (v == 0) ? 1.0/sqrt(2.0) : 1.0;

            for (int x = 0; x < 8; x++) {
                for (int y = 0; y < 8; y++) {
                    basis.at<double>(x, y) = cu * cv * 0.25 *
                        cos((2*x + 1) * u * PI / 16.0) *
                        cos((2*y + 1) * v * PI / 16.0);
                }
            }

            DCT_BASIS[u*8 + v] = basis;
        }
    }

    initialized = true;
}

// 用于小波变换的滤波器系数
const double WAVELET_LO_D[] = {0.7071067811865476, 0.7071067811865476};  // 低通分解
const double WAVELET_HI_D[] = {-0.7071067811865476, 0.7071067811865476}; // 高通分解
const double WAVELET_LO_R[] = {0.7071067811865476, 0.7071067811865476};  // 低通重构
const double WAVELET_HI_R[] = {0.7071067811865476, -0.7071067811865476}; // 高通重构

// 分形压缩的块结构
struct FractalBlock {
    Point position;      // 块位置
    Size size;          // 块大小
    double scale;       // 缩放系数
    double offset;      // 偏移系数
    Point domain_pos;   // 对应的定义域块位置
};

// 一维小波变换
void wavelet_transform_1d(vector<double>& data, bool inverse = false) {
    int n = data.size();
    if (n < 2) return;

    vector<double> temp(n);
    const double* lo_filter = inverse ? WAVELET_LO_R : WAVELET_LO_D;
    const double* hi_filter = inverse ? WAVELET_HI_R : WAVELET_HI_D;

    if (!inverse) {
        // 分解
        for (int i = 0; i < n/2; i++) {
            int j = i * 2;
            temp[i] = data[j] * lo_filter[0] + data[j+1] * lo_filter[1];
            temp[i + n/2] = data[j] * hi_filter[0] + data[j+1] * hi_filter[1];
        }
    } else {
        // 重构
        for (int i = 0; i < n/2; i++) {
            int j = i * 2;
            temp[j] = (data[i] * lo_filter[0] + data[i + n/2] * hi_filter[0]) / 2;
            temp[j+1] = (data[i] * lo_filter[1] + data[i + n/2] * hi_filter[1]) / 2;
        }
    }

    data = temp;
}

// 计算块的均值和方差
void compute_block_statistics(const Mat& block, double& mean, double& variance) {
    mean = 0;
    variance = 0;
    int count = 0;

    for (int i = 0; i < block.rows; i++) {
        for (int j = 0; j < block.cols; j++) {
            double val = block.at<uchar>(i, j);
            mean += val;
            variance += val * val;
            count++;
        }
    }

    mean /= count;
    variance = variance/count - mean*mean;
}

} // anonymous namespace

double rle_encode(const Mat& src, vector<uchar>& encoded) {
    CV_Assert(!src.empty());

    // 转换为灰度图
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    encoded.clear();
    encoded.reserve(gray.total());

    uchar current = gray.at<uchar>(0, 0);
    int count = 1;

    // RLE编码
    for (int i = 1; i < gray.total(); i++) {
        uchar pixel = gray.at<uchar>(i / gray.cols, i % gray.cols);

        if (pixel == current && count < 255) {
            count++;
        } else {
            encoded.push_back(current);
            encoded.push_back(count);
            current = pixel;
            count = 1;
        }
    }

    // 处理最后一组
    encoded.push_back(current);
    encoded.push_back(count);

    return compute_compression_ratio(gray.total(), encoded.size());
}

void rle_decode(const vector<uchar>& encoded,
                Mat& dst,
                const Size& original_size) {
    dst = Mat::zeros(original_size, CV_8UC1);

    int idx = 0;
    int pos = 0;

    while (idx < encoded.size()) {
        uchar value = encoded[idx++];
        uchar count = encoded[idx++];

        for (int i = 0; i < count; i++) {
            dst.at<uchar>(pos / original_size.width,
                         pos % original_size.width) = value;
            pos++;
        }
    }
}

double jpeg_compress_manual(const Mat& src, Mat& dst,
                          int quality) {
    CV_Assert(!src.empty());
    initDCTBasis();

    // 转换为YCrCb颜色空间
    Mat ycrcb;
    cvtColor(src, ycrcb, COLOR_BGR2YCrCb);
    vector<Mat> channels;
    split(ycrcb, channels);

    // 调整量化表根据质量参数
    double scale = quality < 50 ? 5000.0/quality : 200.0 - 2*quality;
    Mat qy = Mat(8, 8, CV_32S);
    Mat qc = Mat(8, 8, CV_32S);

    for (int i = 0; i < 64; i++) {
        qy.at<int>(i/8, i%8) = max(1,
            (JPEG_LUMINANCE_QUANTIZATION[i] * scale + 50) / 100);
        qc.at<int>(i/8, i%8) = max(1,
            (JPEG_CHROMINANCE_QUANTIZATION[i] * scale + 50) / 100);
    }

    // 处理每个通道
    vector<Mat> compressed_channels;
    for (int ch = 0; ch < 3; ch++) {
        Mat& channel = channels[ch];
        Mat padded;

        // 填充到8的倍数
        int rows = ((channel.rows + 7) / 8) * 8;
        int cols = ((channel.cols + 7) / 8) * 8;
        copyMakeBorder(channel, padded, 0, rows - channel.rows,
                      0, cols - channel.cols,
                      BORDER_REPLICATE);

        Mat compressed = Mat::zeros(padded.size(), CV_64F);
        Mat& q = (ch == 0) ? qy : qc;

        // 分块DCT变换和量化
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < rows; i += 8) {
            for (int j = 0; j < cols; j += 8) {
                Mat block = padded(Rect(j, i, 8, 8));
                Mat dct_block = Mat::zeros(8, 8, CV_64F);

                // DCT变换
                for (int u = 0; u < 8; u++) {
                    for (int v = 0; v < 8; v++) {
                        double sum = 0;
                        for (int x = 0; x < 8; x++) {
                            for (int y = 0; y < 8; y++) {
                                sum += block.at<uchar>(x, y) *
                                      DCT_BASIS[u*8 + v].at<double>(x, y);
                            }
                        }
                        dct_block.at<double>(u, v) = sum;
                    }
                }

                // 量化
                for (int x = 0; x < 8; x++) {
                    for (int y = 0; y < 8; y++) {
                        dct_block.at<double>(x, y) =
                            round(dct_block.at<double>(x, y) /
                                  q.at<int>(x, y));
                    }
                }

                dct_block.copyTo(compressed(Rect(j, i, 8, 8)));
            }
        }

        compressed_channels.push_back(compressed);
    }

    // 合并通道
    merge(compressed_channels, dst);

    // 计算压缩率
    return compute_compression_ratio(src.total() * src.elemSize(),
                                   dst.total() * dst.elemSize());
}

double fractal_compress(const Mat& src, Mat& dst, int block_size) {
    CV_Assert(!src.empty());

    // 转换为灰度图
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // 调整图像大小为block_size的倍数
    int rows = ((gray.rows + block_size - 1) / block_size) * block_size;
    int cols = ((gray.cols + block_size - 1) / block_size) * block_size;
    Mat padded;
    copyMakeBorder(gray, padded, 0, rows - gray.rows, 0, cols - gray.cols, BORDER_REPLICATE);

    vector<FractalBlock> blocks;
    const int domain_step = block_size / 2;  // 定义域块步长

    // 使用OpenMP加速块匹配过程
    #pragma omp parallel
    {
        vector<FractalBlock> local_blocks;

        #pragma omp for collapse(2) schedule(dynamic)
        for (int i = 0; i < rows; i += block_size) {
            for (int j = 0; j < cols; j += block_size) {
                Rect range_rect(j, i, block_size, block_size);
                Mat range_block = padded(range_rect);

                double best_error = numeric_limits<double>::max();
                FractalBlock best_match;
                best_match.position = Point(j, i);
                best_match.size = Size(block_size, block_size);

                // 在定义域中搜索最佳匹配
                for (int di = 0; di < rows - block_size*2; di += domain_step) {
                    for (int dj = 0; dj < cols - block_size*2; dj += domain_step) {
                        Mat domain_block = padded(Rect(dj, di, block_size*2, block_size*2));
                        Mat domain_small;
                        resize(domain_block, domain_small, Size(block_size, block_size));

                        double domain_mean, range_mean;
                        double domain_var, range_var;
                        compute_block_statistics(domain_small, domain_mean, domain_var);
                        compute_block_statistics(range_block, range_mean, range_var);

                        if (domain_var < 1e-6) continue;  // 跳过平坦区域

                        // 计算缩放和偏移系数
                        double scale = sqrt(range_var / domain_var);
                        double offset = range_mean - scale * domain_mean;

                        // 计算误差
                        Mat predicted = domain_small * scale + offset;
                        Mat diff = predicted - range_block;
                        double error = norm(diff, NORM_L2SQR) / (block_size * block_size);

                        if (error < best_error) {
                            best_error = error;
                            best_match.scale = scale;
                            best_match.offset = offset;
                            best_match.domain_pos = Point(dj, di);
                        }
                    }
                }

                #pragma omp critical
                blocks.push_back(best_match);
            }
        }
    }

    // 重构图像
    dst = Mat::zeros(padded.size(), CV_8UC1);
    for (const auto& block : blocks) {
        Mat domain_block = padded(Rect(block.domain_pos.x, block.domain_pos.y,
                                     block_size*2, block_size*2));
        Mat domain_small;
        resize(domain_block, domain_small, block.size);

        Mat range_block = domain_small * block.scale + block.offset;
        range_block.copyTo(dst(Rect(block.position.x, block.position.y,
                               block.size.width, block.size.height)));
    }

    // 裁剪回原始大小
    dst = dst(Rect(0, 0, src.cols, src.rows));

    // 计算压缩率（每个块存储5个double：位置x,y，scale，offset，domain_pos x,y）
    size_t compressed_size = blocks.size() * (sizeof(double) * 5);
    return compute_compression_ratio(src.total(), compressed_size);
}

double wavelet_compress(const Mat& src, Mat& dst, int level, double threshold) {
    CV_Assert(!src.empty());

    // 转换为灰度图并转换为浮点型
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }
    Mat float_img;
    gray.convertTo(float_img, CV_64F);

    // 确保图像尺寸是2的幂次
    int max_dim = max(float_img.rows, float_img.cols);
    int pad_size = 1;
    while (pad_size < max_dim) pad_size *= 2;

    Mat padded;
    copyMakeBorder(float_img, padded, 0, pad_size - float_img.rows,
                   0, pad_size - float_img.cols, BORDER_REFLECT);

    int rows = padded.rows;
    int cols = padded.cols;
    Mat temp = padded.clone();

    // 前向小波变换
    for (int l = 0; l < level; l++) {
        // 水平方向变换
        #pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            vector<double> row(cols);
            for (int j = 0; j < cols; j++) {
                row[j] = temp.at<double>(i, j);
            }
            wavelet_transform_1d(row);
            for (int j = 0; j < cols; j++) {
                temp.at<double>(i, j) = row[j];
            }
        }

        // 垂直方向变换
        #pragma omp parallel for
        for (int j = 0; j < cols; j++) {
            vector<double> col(rows);
            for (int i = 0; i < rows; i++) {
                col[i] = temp.at<double>(i, j);
            }
            wavelet_transform_1d(col);
            for (int i = 0; i < rows; i++) {
                temp.at<double>(i, j) = col[i];
            }
        }

        rows /= 2;
        cols /= 2;
    }

    // 阈值处理
    double max_coef = 0;
    for (int i = 0; i < temp.rows; i++) {
        for (int j = 0; j < temp.cols; j++) {
            max_coef = max(max_coef, abs(temp.at<double>(i, j)));
        }
    }

    double thresh = max_coef * threshold / 100.0;
    int nonzero_count = 0;

    #pragma omp parallel for reduction(+:nonzero_count)
    for (int i = 0; i < temp.rows; i++) {
        for (int j = 0; j < temp.cols; j++) {
            double& val = temp.at<double>(i, j);
            if (abs(val) < thresh) {
                val = 0;
            } else {
                nonzero_count++;
            }
        }
    }

    // 反向小波变换
    rows = temp.rows;
    cols = temp.cols;
    for (int l = level - 1; l >= 0; l--) {
        rows = temp.rows >> l;
        cols = temp.cols >> l;

        // 垂直方向逆变换
        #pragma omp parallel for
        for (int j = 0; j < cols; j++) {
            vector<double> col(rows);
            for (int i = 0; i < rows; i++) {
                col[i] = temp.at<double>(i, j);
            }
            wavelet_transform_1d(col, true);
            for (int i = 0; i < rows; i++) {
                temp.at<double>(i, j) = col[i];
            }
        }

        // 水平方向逆变换
        #pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            vector<double> row(cols);
            for (int j = 0; j < cols; j++) {
                row[j] = temp.at<double>(i, j);
            }
            wavelet_transform_1d(row, true);
            for (int j = 0; j < cols; j++) {
                temp.at<double>(i, j) = row[j];
            }
        }
    }

    // 裁剪回原始大小并转换回8位图像
    Mat result = temp(Rect(0, 0, src.cols, src.rows));
    normalize(result, result, 0, 255, NORM_MINMAX);
    result.convertTo(dst, CV_8UC1);

    // 计算压缩率（只存储非零系数）
    size_t compressed_size = nonzero_count * (sizeof(double) + sizeof(int) * 2);  // 值和位置
    return compute_compression_ratio(src.total(), compressed_size);
}

double compute_compression_ratio(size_t original_size, size_t compressed_size) {
    return compressed_size > 0 ?
           static_cast<double>(compressed_size) / original_size : 0.0;
}

double compute_psnr(const Mat& original, const Mat& compressed) {
    CV_Assert(!original.empty() && !compressed.empty());
    CV_Assert(original.size() == compressed.size());

    // 转换为灰度图
    Mat gray1, gray2;
    if (original.channels() == 3) {
        cvtColor(original, gray1, COLOR_BGR2GRAY);
    } else {
        gray1 = original.clone();
    }
    if (compressed.channels() == 3) {
        cvtColor(compressed, gray2, COLOR_BGR2GRAY);
    } else {
        gray2 = compressed.clone();
    }

    // 转换为浮点型
    Mat float1, float2;
    gray1.convertTo(float1, CV_64F);
    gray2.convertTo(float2, CV_64F);

    // 计算MSE
    Mat diff = float1 - float2;
    double mse = 0.0;

    // 使用SIMD和OpenMP优化MSE计算
    #pragma omp parallel for reduction(+:mse)
    for (int i = 0; i < diff.rows; i++) {
        double* row = diff.ptr<double>(i);
        __m256d sum_vec = _mm256_setzero_pd();

        // SIMD向量化
        for (int j = 0; j < diff.cols - 3; j += 4) {
            __m256d vec = _mm256_loadu_pd(row + j);
            sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(vec, vec));
        }

        // 处理剩余元素
        double sum[4];
        _mm256_storeu_pd(sum, sum_vec);
        mse += sum[0] + sum[1] + sum[2] + sum[3];

        for (int j = diff.cols - diff.cols % 4; j < diff.cols; j++) {
            mse += row[j] * row[j];
        }
    }

    mse /= (diff.total());

    // 计算PSNR
    return mse > 0 ? 10.0 * log10(255.0 * 255.0 / mse) : INFINITY;
}

} // namespace ip101