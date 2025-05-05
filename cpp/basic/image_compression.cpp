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

double fractal_compress(const Mat& src, Mat& dst,
                       int block_size) {
    CV_Assert(!src.empty());

    // 转换为灰度图
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    int rows = gray.rows;
    int cols = gray.cols;

    // 创建下采样图像
    Mat downsampled;
    resize(gray, downsampled, Size(cols/2, rows/2));

    // 分块处理
    vector<Point> range_blocks;  // 范围块位置
    vector<Point> domain_blocks; // 域块位置
    vector<double> scales;       // 缩放因子
    vector<double> offsets;      // 偏移量

    for (int y = 0; y < rows - block_size; y += block_size) {
        for (int x = 0; x < cols - block_size; x += block_size) {
            Mat range = gray(Rect(x, y, block_size, block_size));
            double min_error = DBL_MAX;
            Point best_pos;
            double best_scale = 0;
            double best_offset = 0;

            // 在域块中搜索最佳匹配
            for (int dy = 0; dy < rows/2 - block_size; dy += block_size/2) {
                for (int dx = 0; dx < cols/2 - block_size; dx += block_size/2) {
                    Mat domain = downsampled(Rect(dx, dy, block_size/2, block_size/2));
                    Mat resized;
                    resize(domain, resized, Size(block_size, block_size));

                    // 计算最佳线性变换参数
                    Scalar mean_r, mean_d, stddev_r, stddev_d;
                    meanStdDev(range, mean_r, stddev_r);
                    meanStdDev(resized, mean_d, stddev_d);

                    double scale = stddev_r[0] / (stddev_d[0] + 1e-6);
                    double offset = mean_r[0] - scale * mean_d[0];

                    // 计算误差
                    Mat transformed = scale * resized + offset;
                    Mat diff = transformed - range;
                    double error = norm(diff);

                    if (error < min_error) {
                        min_error = error;
                        best_pos = Point(dx, dy);
                        best_scale = scale;
                        best_offset = offset;
                    }
                }
            }

            range_blocks.push_back(Point(x, y));
            domain_blocks.push_back(best_pos);
            scales.push_back(best_scale);
            offsets.push_back(best_offset);
        }
    }

    // 重建图像
    dst = Mat::zeros(gray.size(), CV_8UC1);

    for (size_t i = 0; i < range_blocks.size(); i++) {
        Point range_pos = range_blocks[i];
        Point domain_pos = domain_blocks[i];
        double scale = scales[i];
        double offset = offsets[i];

        Mat domain = downsampled(Rect(domain_pos.x, domain_pos.y,
                                    block_size/2, block_size/2));
        Mat resized;
        resize(domain, resized, Size(block_size, block_size));

        Mat transformed = scale * resized + offset;
        transformed.copyTo(dst(Rect(range_pos.x, range_pos.y,
                                  block_size, block_size)));
    }

    // 计算压缩率
    size_t compressed_size = (range_blocks.size() *
                            (sizeof(Point) * 2 + sizeof(double) * 2));
    return compute_compression_ratio(gray.total(), compressed_size);
}

double wavelet_compress(const Mat& src, Mat& dst,
                       int level,
                       double threshold) {
    CV_Assert(!src.empty());

    // 转换为灰度图并转换为浮点型
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }
    gray.convertTo(gray, CV_64F);

    // Haar小波变换
    Mat transformed = gray.clone();
    int rows = transformed.rows;
    int cols = transformed.cols;

    for (int l = 0; l < level; l++) {
        Mat temp = transformed(Rect(0, 0, cols, rows)).clone();

        // 行变换
        #pragma omp parallel for
        for (int y = 0; y < rows; y++) {
            vector<double> row(cols);
            for (int x = 0; x < cols; x += 2) {
                double v1 = temp.at<double>(y, x);
                double v2 = temp.at<double>(y, x+1);

                row[x/2] = (v1 + v2) / sqrt(2.0);
                row[cols/2 + x/2] = (v1 - v2) / sqrt(2.0);
            }

            for (int x = 0; x < cols; x++) {
                transformed.at<double>(y, x) = row[x];
            }
        }

        // 列变换
        temp = transformed(Rect(0, 0, cols, rows)).clone();
        #pragma omp parallel for
        for (int x = 0; x < cols; x++) {
            vector<double> col(rows);
            for (int y = 0; y < rows; y += 2) {
                double v1 = temp.at<double>(y, x);
                double v2 = temp.at<double>(y+1, x);

                col[y/2] = (v1 + v2) / sqrt(2.0);
                col[rows/2 + y/2] = (v1 - v2) / sqrt(2.0);
            }

            for (int y = 0; y < rows; y++) {
                transformed.at<double>(y, x) = col[y];
            }
        }

        rows /= 2;
        cols /= 2;
    }

    // 阈值处理
    Mat compressed = transformed.clone();
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < transformed.rows; y++) {
        for (int x = 0; x < transformed.cols; x++) {
            if (abs(transformed.at<double>(y, x)) < threshold) {
                compressed.at<double>(y, x) = 0;
            }
        }
    }

    // 逆变换
    dst = compressed.clone();
    rows = dst.rows;
    cols = dst.cols;

    for (int l = level-1; l >= 0; l--) {
        rows = dst.rows >> l;
        cols = dst.cols >> l;

        Mat temp = dst(Rect(0, 0, cols, rows)).clone();

        // 列逆变换
        #pragma omp parallel for
        for (int x = 0; x < cols; x++) {
            vector<double> col(rows);
            for (int y = 0; y < rows/2; y++) {
                double a = temp.at<double>(y, x);
                double d = temp.at<double>(y + rows/2, x);

                col[2*y] = (a + d) / sqrt(2.0);
                col[2*y + 1] = (a - d) / sqrt(2.0);
            }

            for (int y = 0; y < rows; y++) {
                dst.at<double>(y, x) = col[y];
            }
        }

        // 行逆变换
        temp = dst(Rect(0, 0, cols, rows)).clone();
        #pragma omp parallel for
        for (int y = 0; y < rows; y++) {
            vector<double> row(cols);
            for (int x = 0; x < cols/2; x++) {
                double a = temp.at<double>(y, x);
                double d = temp.at<double>(y, x + cols/2);

                row[2*x] = (a + d) / sqrt(2.0);
                row[2*x + 1] = (a - d) / sqrt(2.0);
            }

            for (int x = 0; x < cols; x++) {
                dst.at<double>(y, x) = row[x];
            }
        }
    }

    // 转换回8位图像
    dst.convertTo(dst, CV_8U);

    // 计算压缩率
    int nonzero = countNonZero(compressed);
    return compute_compression_ratio(src.total(),
                                   nonzero * sizeof(double));
}

double compute_compression_ratio(size_t original_size,
                               size_t compressed_size) {
    return static_cast<double>(original_size) / compressed_size;
}

double compute_psnr(const Mat& original,
                   const Mat& compressed) {
    Mat diff;
    absdiff(original, compressed, diff);
    diff.convertTo(diff, CV_64F);
    diff = diff.mul(diff);

    double mse = sum(diff)[0] / (original.total() * original.channels());
    double psnr = 20 * log10(255.0) - 10 * log10(mse);

    return psnr;
}

} // namespace ip101