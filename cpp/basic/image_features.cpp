#include "image_features.hpp"
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

// 计算梯度
void compute_gradient(const Mat& src, Mat& magnitude, Mat& angle) {
    Mat grad_x, grad_y;
    Sobel(src, grad_x, CV_32F, 1, 0, 3);
    Sobel(src, grad_y, CV_32F, 0, 1, 3);

    magnitude.create(src.size(), CV_32F);
    angle.create(src.size(), CV_32F);

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            float gx = grad_x.at<float>(y, x);
            float gy = grad_y.at<float>(y, x);
            magnitude.at<float>(y, x) = sqrt(gx * gx + gy * gy);
            angle.at<float>(y, x) = atan2(gy, gx);
        }
    }
}

} // anonymous namespace

void hog_features(const Mat& src,
                 vector<float>& features,
                 int cell_size,
                 int block_size,
                 int bin_num) {
    CV_Assert(!src.empty());

    // 转换为灰度图
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // 计算梯度
    Mat magnitude, angle;
    compute_gradient(gray, magnitude, angle);

    // 计算cell直方图
    int cell_rows = gray.rows / cell_size;
    int cell_cols = gray.cols / cell_size;
    vector<vector<vector<float>>> cell_hists(cell_rows,
        vector<vector<float>>(cell_cols, vector<float>(bin_num, 0)));

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < gray.rows - cell_size; y += cell_size) {
        for (int x = 0; x < gray.cols - cell_size; x += cell_size) {
            vector<float> hist(bin_num, 0);

            // 计算cell内的梯度直方图
            for (int cy = 0; cy < cell_size; cy++) {
                for (int cx = 0; cx < cell_size; cx++) {
                    float mag = magnitude.at<float>(y + cy, x + cx);
                    float ang = angle.at<float>(y + cy, x + cx);
                    if (ang < 0) ang += PI;

                    float bin_size = PI / bin_num;
                    int bin = static_cast<int>(ang / bin_size);
                    if (bin >= bin_num) bin = bin_num - 1;

                    hist[bin] += mag;
                }
            }

            cell_hists[y/cell_size][x/cell_size] = hist;
        }
    }

    // 计算block特征
    features.clear();
    for (int y = 0; y <= cell_rows - block_size; y++) {
        for (int x = 0; x <= cell_cols - block_size; x++) {
            vector<float> block_feat;
            float norm = 0;

            // 收集block内的所有cell直方图
            for (int by = 0; by < block_size; by++) {
                for (int bx = 0; bx < block_size; bx++) {
                    const auto& hist = cell_hists[y + by][x + bx];
                    block_feat.insert(block_feat.end(), hist.begin(), hist.end());
                    for (float val : hist) {
                        norm += val * val;
                    }
                }
            }

            // L2归一化
            norm = sqrt(norm + 1e-6);
            for (float& val : block_feat) {
                val /= norm;
            }

            features.insert(features.end(), block_feat.begin(), block_feat.end());
        }
    }
}

void lbp_features(const Mat& src,
                 Mat& dst,
                 int radius,
                 int neighbors) {
    CV_Assert(!src.empty());

    // 转换为灰度图
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    dst = Mat::zeros(gray.size(), CV_8U);

    #pragma omp parallel for collapse(2)
    for (int y = radius; y < gray.rows - radius; y++) {
        for (int x = radius; x < gray.cols - radius; x++) {
            uchar center = gray.at<uchar>(y, x);
            uchar code = 0;

            for (int n = 0; n < neighbors; n++) {
                double theta = 2.0 * PI * n / neighbors;
                int rx = static_cast<int>(x + radius * cos(theta) + 0.5);
                int ry = static_cast<int>(y - radius * sin(theta) + 0.5);

                code |= (gray.at<uchar>(ry, rx) >= center) << n;
            }

            dst.at<uchar>(y, x) = code;
        }
    }
}

void haar_features(const Mat& src,
                  vector<float>& features,
                  Size min_size,
                  Size max_size) {
    CV_Assert(!src.empty());

    // 转换为灰度图
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // 计算积分图
    Mat integral;
    compute_integral_image(gray, integral);

    features.clear();

    // 计算不同尺寸的Haar特征
    for (int h = min_size.height; h <= max_size.height; h += 4) {
        for (int w = min_size.width; w <= max_size.width; w += 4) {
            // 垂直边缘特征
            for (int y = 0; y <= gray.rows - h; y++) {
                for (int x = 0; x <= gray.cols - w; x++) {
                    int w2 = w / 2;
                    float left = integral.at<double>(y + h, x + w2) +
                               integral.at<double>(y, x) -
                               integral.at<double>(y, x + w2) -
                               integral.at<double>(y + h, x);

                    float right = integral.at<double>(y + h, x + w) +
                                integral.at<double>(y, x + w2) -
                                integral.at<double>(y, x + w) -
                                integral.at<double>(y + h, x + w2);

                    features.push_back(right - left);
                }
            }

            // 水平边缘特征
            for (int y = 0; y <= gray.rows - h; y++) {
                for (int x = 0; x <= gray.cols - w; x++) {
                    int h2 = h / 2;
                    float top = integral.at<double>(y + h2, x + w) +
                              integral.at<double>(y, x) -
                              integral.at<double>(y, x + w) -
                              integral.at<double>(y + h2, x);

                    float bottom = integral.at<double>(y + h, x + w) +
                                 integral.at<double>(y + h2, x) -
                                 integral.at<double>(y + h2, x + w) -
                                 integral.at<double>(y + h, x);

                    features.push_back(bottom - top);
                }
            }
        }
    }
}

vector<Mat> create_gabor_filters(int scales,
                               int orientations,
                               Size size) {
    vector<Mat> filters;
    double sigma = 1.0;
    double lambda = 4.0;
    double gamma = 0.5;
    double psi = 0;

    for (int s = 0; s < scales; s++) {
        for (int o = 0; o < orientations; o++) {
            Mat kernel = Mat::zeros(size, CV_32F);
            double theta = o * PI / orientations;
            double sigma_x = sigma;
            double sigma_y = sigma / gamma;

            for (int y = -size.height/2; y <= size.height/2; y++) {
                for (int x = -size.width/2; x <= size.width/2; x++) {
                    double x_theta = x * cos(theta) + y * sin(theta);
                    double y_theta = -x * sin(theta) + y * cos(theta);

                    double gaussian = exp(-0.5 * (x_theta * x_theta / (sigma_x * sigma_x) +
                                                y_theta * y_theta / (sigma_y * sigma_y)));
                    double wave = cos(2 * PI * x_theta / lambda + psi);

                    kernel.at<float>(y + size.height/2, x + size.width/2) =
                        gaussian * wave;
                }
            }

            filters.push_back(kernel);
        }

        sigma *= 2;
        lambda *= 2;
    }

    return filters;
}

void gabor_features(const Mat& src,
                   vector<float>& features,
                   int scales,
                   int orientations) {
    CV_Assert(!src.empty());

    // 转换为灰度图
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }
    gray.convertTo(gray, CV_32F);

    // 创建Gabor滤波器组
    vector<Mat> filters = create_gabor_filters(scales, orientations);

    features.clear();

    // 应用滤波器并提取特征
    for (const Mat& filter : filters) {
        Mat response;
        filter2D(gray, response, CV_32F, filter);

        // 计算响应的统计特征
        Scalar mean, stddev;
        meanStdDev(response, mean, stddev);

        features.push_back(mean[0]);
        features.push_back(stddev[0]);
    }
}

void color_histogram(const Mat& src,
                    Mat& hist,
                    const vector<int>& bins) {
    CV_Assert(!src.empty() && src.channels() == 3);

    // 计算每个通道的直方图范围
    vector<float> ranges[] = {
        vector<float>(bins[0] + 1),
        vector<float>(bins[1] + 1),
        vector<float>(bins[2] + 1)
    };

    for (int i = 0; i < 3; i++) {
        float step = 256.0f / bins[i];
        for (int j = 0; j <= bins[i]; j++) {
            ranges[i][j] = j * step;
        }
    }

    // 分离通道
    vector<Mat> channels;
    split(src, channels);

    // 计算3D直方图
    int dims[] = {bins[0], bins[1], bins[2]};
    hist = Mat::zeros(3, dims, CV_32F);

    #pragma omp parallel for collapse(3)
    for (int b = 0; b < bins[0]; b++) {
        for (int g = 0; g < bins[1]; g++) {
            for (int r = 0; r < bins[2]; r++) {
                float count = 0;

                for (int y = 0; y < src.rows; y++) {
                    for (int x = 0; x < src.cols; x++) {
                        uchar b_val = channels[0].at<uchar>(y, x);
                        uchar g_val = channels[1].at<uchar>(y, x);
                        uchar r_val = channels[2].at<uchar>(y, x);

                        if (b_val >= ranges[0][b] && b_val < ranges[0][b+1] &&
                            g_val >= ranges[1][g] && g_val < ranges[1][g+1] &&
                            r_val >= ranges[2][r] && r_val < ranges[2][r+1]) {
                            count++;
                        }
                    }
                }

                hist.at<float>(b, g, r) = count;
            }
        }
    }

    // 归一化
    normalize(hist, hist, 1, 0, NORM_L1);
}

void compute_integral_image(const Mat& src,
                          Mat& integral) {
    CV_Assert(!src.empty());

    integral.create(src.rows + 1, src.cols + 1, CV_64F);
    integral = Scalar(0);

    // 计算积分图
    for (int y = 0; y < src.rows; y++) {
        double row_sum = 0;
        for (int x = 0; x < src.cols; x++) {
            row_sum += src.at<uchar>(y, x);
            integral.at<double>(y + 1, x + 1) =
                row_sum + integral.at<double>(y, x + 1);
        }
    }
}

void compute_gradient_histogram(const Mat& magnitude,
                              const Mat& angle,
                              vector<float>& hist,
                              int bin_num) {
    hist.resize(bin_num);
    fill(hist.begin(), hist.end(), 0);

    float bin_size = PI / bin_num;

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < magnitude.rows; y++) {
        for (int x = 0; x < magnitude.cols; x++) {
            float mag = magnitude.at<float>(y, x);
            float ang = angle.at<float>(y, x);
            if (ang < 0) ang += PI;

            int bin = static_cast<int>(ang / bin_size);
            if (bin >= bin_num) bin = bin_num - 1;

            #pragma omp atomic
            hist[bin] += mag;
        }
    }

    // 归一化
    float sum = 0;
    for (float val : hist) {
        sum += val * val;
    }
    sum = sqrt(sum + 1e-6);

    for (float& val : hist) {
        val /= sum;
    }
}

} // namespace ip101