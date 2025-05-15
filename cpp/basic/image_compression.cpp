#include <basic/image_compression.hpp>
#include <cmath>
#include <algorithm>

namespace ip101 {

using namespace cv;
using namespace std;

namespace {
// Internal constants
constexpr int CACHE_LINE = 64;    // CPU cache line size (bytes)
constexpr int SIMD_WIDTH = 32;    // AVX2 SIMD vector width (bytes)
constexpr int BLOCK_SIZE = 16;    // Block processing size
constexpr double PI = 3.14159265358979323846;

// Memory alignment helper function
inline uchar* alignPtr(uchar* ptr, size_t align = CACHE_LINE) {
    return (uchar*)(((size_t)ptr + align - 1) & ~(align - 1));
}

// JPEG luminance quantization table
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

// JPEG chrominance quantization table
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

// DCT basis function lookup table
Mat DCT_BASIS[64];

// Initialize DCT basis function lookup table
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

// Wavelet filter coefficients
const double WAVELET_LO_D[] = {0.7071067811865476, 0.7071067811865476};  // Low-pass decomposition
const double WAVELET_HI_D[] = {-0.7071067811865476, 0.7071067811865476}; // High-pass decomposition
const double WAVELET_LO_R[] = {0.7071067811865476, 0.7071067811865476};  // Low-pass reconstruction
const double WAVELET_HI_R[] = {0.7071067811865476, -0.7071067811865476}; // High-pass reconstruction

// Fractal compression block structure
struct FractalBlock {
    Point position;     // Block position
    Size size;          // Block size
    double scale;       // Scale factor
    double offset;      // Offset factor
    Point domain_pos;   // Corresponding domain block position
};

// One-dimensional wavelet transform
void wavelet_transform_1d(vector<double>& data, bool inverse = false) {
    int n = static_cast<int>(data.size());
    if (n < 2) return;

    vector<double> temp(n);
    const double* lo_filter = inverse ? WAVELET_LO_R : WAVELET_LO_D;
    const double* hi_filter = inverse ? WAVELET_HI_R : WAVELET_HI_D;

    if (!inverse) {
        // Decomposition
        for (int i = 0; i < n/2; i++) {
            int j = i * 2;
            temp[i] = data[j] * lo_filter[0] + data[j+1] * lo_filter[1];
            temp[i + n/2] = data[j] * hi_filter[0] + data[j+1] * hi_filter[1];
        }
    } else {
        // Reconstruction
        for (int i = 0; i < n/2; i++) {
            int j = i * 2;
            temp[j] = (data[i] * lo_filter[0] + data[i + n/2] * hi_filter[0]) / 2;
            temp[j+1] = (data[i] * lo_filter[1] + data[i + n/2] * hi_filter[1]) / 2;
        }
    }

    data = temp;
}

// Compute block mean and variance
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

    // Convert to grayscale
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

    // RLE encoding
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

    // Process the last group
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

    // Convert to YCrCb color space
    Mat ycrcb;
    cvtColor(src, ycrcb, COLOR_BGR2YCrCb);
    vector<Mat> channels;
    split(ycrcb, channels);

    // Adjust quantization tables based on quality parameter
    double scale = quality < 50 ? 5000.0/quality : 200.0 - 2*quality;
    Mat qy = Mat(8, 8, CV_32S);
    Mat qc = Mat(8, 8, CV_32S);

    for (int i = 0; i < 64; i++) {
        int q_value1 = static_cast<int>((JPEG_LUMINANCE_QUANTIZATION[i] * scale + 50) / 100);
        qy.at<int>(i/8, i%8) = std::max(1, q_value1);

        int q_value2 = static_cast<int>((JPEG_CHROMINANCE_QUANTIZATION[i] * scale + 50) / 100);
        qc.at<int>(i/8, i%8) = std::max(1, q_value2);
    }

    // Process each channel
    vector<Mat> compressed_channels;
    for (int ch = 0; ch < 3; ch++) {
        Mat& channel = channels[ch];
        Mat padded;

        // Pad to multiple of 8
        int rows = ((channel.rows + 7) / 8) * 8;
        int cols = ((channel.cols + 7) / 8) * 8;
        copyMakeBorder(channel, padded, 0, rows - channel.rows,
                      0, cols - channel.cols,
                      BORDER_REPLICATE);

        Mat compressed = Mat::zeros(padded.size(), CV_64F);
        Mat& q = (ch == 0) ? qy : qc;

        // Block-wise DCT transform and quantization
        #pragma omp parallel for
        for (int i = 0; i < rows; i += 8) {
            for (int j = 0; j < cols; j += 8) {
                Mat block = padded(Rect(j, i, 8, 8));
                Mat dct_block = Mat::zeros(8, 8, CV_64F);

                // DCT transform
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

                // Quantization
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

    // Merge channels
    merge(compressed_channels, dst);

    // Calculate compression ratio
    return compute_compression_ratio(src.total() * src.elemSize(),
                                   dst.total() * dst.elemSize());
}

double fractal_compress(const Mat& src, Mat& dst, int block_size) {
    CV_Assert(!src.empty());

    // Convert to grayscale
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // Adjust image size to be multiple of block_size
    int rows = ((gray.rows + block_size - 1) / block_size) * block_size;
    int cols = ((gray.cols + block_size - 1) / block_size) * block_size;
    Mat padded;
    copyMakeBorder(gray, padded, 0, rows - gray.rows, 0, cols - gray.cols, BORDER_REPLICATE);

    vector<FractalBlock> blocks;
    const int domain_step = block_size / 2;  // Domain block step size

    // Use OpenMP to accelerate block matching process
    #pragma omp parallel
    {
        vector<FractalBlock> local_blocks;

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < rows; i += block_size) {
            for (int j = 0; j < cols; j += block_size) {
                Rect range_rect(j, i, block_size, block_size);
                Mat range_block = padded(range_rect);

                double best_error = numeric_limits<double>::max();
                FractalBlock best_match;
                best_match.position = Point(j, i);
                best_match.size = Size(block_size, block_size);

                // Search for best match in domain
                for (int di = 0; di < rows - block_size*2; di += domain_step) {
                    for (int dj = 0; dj < cols - block_size*2; dj += domain_step) {
                        Mat domain_block = padded(Rect(dj, di, block_size*2, block_size*2));
                        Mat domain_small;
                        resize(domain_block, domain_small, Size(block_size, block_size));

                        double domain_mean, range_mean;
                        double domain_var, range_var;
                        compute_block_statistics(domain_small, domain_mean, domain_var);
                        compute_block_statistics(range_block, range_mean, range_var);

                        if (domain_var < 1e-6) continue;  // Skip flat areas

                        // Calculate scale and offset coefficients
                        double scale = sqrt(range_var / domain_var);
                        double offset = range_mean - scale * domain_mean;

                        // Calculate error
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

    // Reconstruct image
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

    // Crop back to original size
    dst = dst(Rect(0, 0, src.cols, src.rows));

    // Calculate compression ratio (each block stores 5 doubles: position x,y, scale, offset, domain_pos x,y)
    size_t compressed_size = blocks.size() * (sizeof(double) * 5);
    return compute_compression_ratio(src.total(), compressed_size);
}

double wavelet_compress(const Mat& src, Mat& dst, int level, double threshold) {
    CV_Assert(!src.empty());

    // Convert to grayscale and to floating point
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }
    Mat float_img;
    gray.convertTo(float_img, CV_64F);

    // Ensure image dimensions are powers of 2
    int max_dim = max(float_img.rows, float_img.cols);
    int pad_size = 1;
    while (pad_size < max_dim) pad_size *= 2;

    Mat padded;
    copyMakeBorder(float_img, padded, 0, pad_size - float_img.rows,
                   0, pad_size - float_img.cols, BORDER_REFLECT);

    int rows = padded.rows;
    int cols = padded.cols;
    Mat temp = padded.clone();

    // Forward wavelet transform
    for (int l = 0; l < level; l++) {
        // Horizontal transform
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

        // Vertical transform
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

    // Threshold processing
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

    // Inverse wavelet transform
    rows = temp.rows;
    cols = temp.cols;
    for (int l = level - 1; l >= 0; l--) {
        rows = temp.rows >> l;
        cols = temp.cols >> l;

        // Vertical inverse transform
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

        // Horizontal inverse transform
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

    // Crop back to original size and convert back to 8-bit image
    Mat result = temp(Rect(0, 0, src.cols, src.rows));
    normalize(result, result, 0, 255, NORM_MINMAX);
    result.convertTo(dst, CV_8UC1);

    // Calculate compression ratio (only store non-zero coefficients)
    size_t compressed_size = nonzero_count * (sizeof(double) + sizeof(int) * 2);  // Value and position
    return compute_compression_ratio(src.total(), compressed_size);
}

double compute_compression_ratio(size_t original_size, size_t compressed_size) {
    return compressed_size > 0 ?
           static_cast<double>(compressed_size) / original_size : 0.0;
}

double compute_compression_psnr(const Mat& original, const Mat& compressed) {
    CV_Assert(!original.empty() && !compressed.empty());
    CV_Assert(original.size() == compressed.size());

    // Convert to grayscale
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

    // Convert to floating point
    Mat float1, float2;
    gray1.convertTo(float1, CV_64F);
    gray2.convertTo(float2, CV_64F);

    // Calculate MSE
    Mat diff = float1 - float2;
    double mse = 0.0;

    // Use SIMD and OpenMP to optimize MSE calculation
    #pragma omp parallel for reduction(+:mse)
    for (int i = 0; i < diff.rows; i++) {
        double* row = diff.ptr<double>(i);
        __m256d sum_vec = _mm256_setzero_pd();

        // SIMD vectorization
        for (int j = 0; j < diff.cols - 3; j += 4) {
            __m256d vec = _mm256_loadu_pd(row + j);
            sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(vec, vec));
        }

        // Process remaining elements
        double sum[4];
        _mm256_storeu_pd(sum, sum_vec);
        mse += sum[0] + sum[1] + sum[2] + sum[3];

        for (int j = diff.cols - diff.cols % 4; j < diff.cols; j++) {
            mse += row[j] * row[j];
        }
    }

    mse /= (diff.total());

    // Calculate PSNR
    return mse > 0 ? 10.0 * log10(255.0 * 255.0 / mse) : INFINITY;
}

} // namespace ip101