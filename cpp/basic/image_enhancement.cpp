#include <basic/image_enhancement.hpp>
#include <cmath>
#include <vector>

namespace ip101 {

using namespace cv;
using namespace std;

namespace {
// Internal constants
constexpr int CACHE_LINE = 64;    // CPU cache line size (bytes)
constexpr int SIMD_WIDTH = 32;    // AVX2 SIMD vector width (bytes)
constexpr int BLOCK_SIZE = 16;    // Block processing size

// Memory alignment helper function
inline uchar* alignPtr(uchar* ptr, size_t align = CACHE_LINE) {
    return (uchar*)(((size_t)ptr + align - 1) & ~(align - 1));
}

// SIMD optimized histogram calculation
inline void calculate_histogram_simd(const uchar* src, int* hist, int width) {
    alignas(32) int local_hist[256] = {0};

    for (int x = 0; x < width; x += 32) {
        __m256i pixels = _mm256_loadu_si256((__m256i*)(src + x));

        // Process 32 pixels with AVX2 instructions
        for (int i = 0; i < 32; i++) {
            uchar pixel_val = src[x + i];
            local_hist[pixel_val]++;
        }
    }

    // Accumulate to global histogram
    for (int i = 0; i < 256; i++) {
        hist[i] += local_hist[i];
    }
}

} // anonymous namespace

void calculate_histogram(const Mat& src, Mat& hist, int channel) {
    CV_Assert(!src.empty() && (src.type() == CV_8UC1 || src.type() == CV_8UC3));
    CV_Assert(channel >= 0 && channel < src.channels());

    hist = Mat::zeros(256, 1, CV_32S);
    int* hist_data = hist.ptr<int>(0);

    if (src.type() == CV_8UC1) {
        #pragma omp parallel
        {
            alignas(32) int local_hist[256] = {0};

            #pragma omp for nowait
            for (int y = 0; y < src.rows; y++) {
                const uchar* row = src.ptr<uchar>(y);
                calculate_histogram_simd(row, local_hist, src.cols);
            }

            #pragma omp critical
            {
                for (int i = 0; i < 256; i++) {
                    hist_data[i] += local_hist[i];
                }
            }
        }
    } else {
        #pragma omp parallel
        {
            alignas(32) int local_hist[256] = {0};

            #pragma omp for nowait
            for (int y = 0; y < src.rows; y++) {
                const uchar* row = src.ptr<uchar>(y);
                for (int x = 0; x < src.cols; x++) {
                    local_hist[row[x * 3 + channel]]++;
                }
            }

            #pragma omp critical
            {
                for (int i = 0; i < 256; i++) {
                    hist_data[i] += local_hist[i];
                }
            }
        }
    }
}

void calculate_cdf(const Mat& hist, Mat& cdf) {
    CV_Assert(!hist.empty() && hist.type() == CV_32S);

    cdf = Mat::zeros(256, 1, CV_32S);
    int sum = 0;

    for (int i = 0; i < 256; i++) {
        sum += hist.at<int>(i);
        cdf.at<int>(i) = sum;
    }
}

void histogram_equalization(const Mat& src, Mat& dst,
                          const std::string& method,
                          double clip_limit,
                          Size grid_size) {
    CV_Assert(!src.empty());

    if (method == "global") {
        if (src.channels() == 1) {
            // Global histogram equalization for single-channel image
            Mat hist, cdf;
            calculate_histogram(src, hist);
            calculate_cdf(hist, cdf);

            // Normalize CDF
            double scale = 255.0 / (src.rows * src.cols);

            dst.create(src.size(), src.type());

            #pragma omp parallel for
            for (int y = 0; y < src.rows; y++) {
                for (int x = 0; x < src.cols; x++) {
                    dst.at<uchar>(y, x) = saturate_cast<uchar>(
                        cdf.at<int>(src.at<uchar>(y, x)) * scale);
                }
            }
        } else {
            // Convert to HSV space for processing
            Mat hsv;
            cvtColor(src, hsv, COLOR_BGR2HSV);

            vector<Mat> channels;
            split(hsv, channels);

            // Only equalize the V channel
            Mat hist, cdf;
            calculate_histogram(channels[2], hist);
            calculate_cdf(hist, cdf);

            double scale = 255.0 / (src.rows * src.cols);

            #pragma omp parallel for
            for (int y = 0; y < src.rows; y++) {
                for (int x = 0; x < src.cols; x++) {
                    channels[2].at<uchar>(y, x) = saturate_cast<uchar>(
                        cdf.at<int>(channels[2].at<uchar>(y, x)) * scale);
                }
            }

            merge(channels, hsv);
            cvtColor(hsv, dst, COLOR_HSV2BGR);
        }
    } else if (method == "adaptive") {
        local_histogram_equalization(src, dst);
    } else if (method == "clahe") {
        clahe(src, dst, clip_limit, grid_size);
    } else {
        throw std::invalid_argument("Unsupported histogram equalization method: " + method);
    }
}

void local_histogram_equalization(const Mat& src, Mat& dst, int window_size) {
    CV_Assert(!src.empty());
    CV_Assert(window_size % 2 == 1);  // Ensure window size is odd

    dst.create(src.size(), src.type());
    int radius = window_size / 2;

    // Handle boundaries
    Mat padded;
    copyMakeBorder(src, padded, radius, radius, radius, radius, BORDER_REFLECT);

    if (src.channels() == 1) {
        #pragma omp parallel for
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                // Extract local window
                Mat window = padded(
                    Range(y, y + window_size),
                    Range(x, x + window_size)
                );

                // Calculate local histogram and CDF
                Mat hist, cdf;
                calculate_histogram(window, hist);
                calculate_cdf(hist, cdf);

                // Normalize CDF
                double scale = 255.0 / (window_size * window_size);
                dst.at<uchar>(y, x) = saturate_cast<uchar>(
                    cdf.at<int>(src.at<uchar>(y, x)) * scale);
            }
        }
    } else {
        // Convert to HSV space
        Mat hsv;
        cvtColor(src, hsv, COLOR_BGR2HSV);
        vector<Mat> channels;
        split(hsv, channels);

        // Apply local histogram equalization to V channel
        Mat v_padded;
        copyMakeBorder(channels[2], v_padded, radius, radius, radius, radius, BORDER_REFLECT);

        #pragma omp parallel for
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                Mat window = v_padded(
                    Range(y, y + window_size),
                    Range(x, x + window_size)
                );

                Mat hist, cdf;
                calculate_histogram(window, hist);
                calculate_cdf(hist, cdf);

                double scale = 255.0 / (window_size * window_size);
                channels[2].at<uchar>(y, x) = saturate_cast<uchar>(
                    cdf.at<int>(channels[2].at<uchar>(y, x)) * scale);
            }
        }

        merge(channels, hsv);
        cvtColor(hsv, dst, COLOR_HSV2BGR);
    }
}

void clahe(const Mat& src, Mat& dst, double clip_limit, Size grid_size) {
    CV_Assert(!src.empty());

    if (src.channels() == 1) {
        dst.create(src.size(), src.type());

        // Calculate grid size
        int grid_width = src.cols / grid_size.width;
        int grid_height = src.rows / grid_size.height;

        // Calculate histogram for each grid
        vector<vector<Mat>> grid_hists(grid_size.height, vector<Mat>(grid_size.width));

        #pragma omp parallel for collapse(2)
        for (int gy = 0; gy < grid_size.height; gy++) {
            for (int gx = 0; gx < grid_size.width; gx++) {
                // Extract grid region
                int y0 = gy * grid_height;
                int x0 = gx * grid_width;
                int y1 = (gy == grid_size.height - 1) ? src.rows : y0 + grid_height;
                int x1 = (gx == grid_size.width - 1) ? src.cols : x0 + grid_width;

                Mat grid = src(Range(y0, y1), Range(x0, x1));

                // Calculate histogram
                calculate_histogram(grid, grid_hists[gy][gx]);

                // Apply contrast limit
                int clip = static_cast<int>(clip_limit * grid_width * grid_height / 256);
                if (clip > 0) {
                    int* hist_data = grid_hists[gy][gx].ptr<int>(0);
                    int excess = 0;
                    for (int i = 0; i < 256; i++) {
                        if (hist_data[i] > clip) {
                            excess += hist_data[i] - clip;
                            hist_data[i] = clip;
                        }
                    }

                    // Redistribute excess
                    int redistrib = excess / 256;
                    int mod = excess % 256;
                    for (int i = 0; i < 256; i++) {
                        hist_data[i] += redistrib;
                        if (i < mod) hist_data[i]++;
                    }
                }
            }
        }

        // Create CDFs for each grid
        vector<vector<Mat>> grid_cdfs(grid_size.height, vector<Mat>(grid_size.width));
        for (int gy = 0; gy < grid_size.height; gy++) {
            for (int gx = 0; gx < grid_size.width; gx++) {
                calculate_cdf(grid_hists[gy][gx], grid_cdfs[gy][gx]);
            }
        }

        // Bilinear interpolation for each pixel
        #pragma omp parallel for
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                // Calculate grid position and weights
                float gx = (x * grid_size.width) / static_cast<float>(src.cols) - 0.5f;
                float gy = (y * grid_size.height) / static_cast<float>(src.rows) - 0.5f;

                int gx0 = std::max(0, std::min(grid_size.width - 1, static_cast<int>(floor(gx))));
                int gy0 = std::max(0, std::min(grid_size.height - 1, static_cast<int>(floor(gy))));
                int gx1 = std::min(grid_size.width - 1, gx0 + 1);
                int gy1 = std::min(grid_size.height - 1, gy0 + 1);

                float wx = gx - gx0;
                float wy = gy - gy0;

                // Get pixel value
                uchar pixel = src.at<uchar>(y, x);

                // Bilinear interpolation
                float val00 = grid_cdfs[gy0][gx0].at<int>(pixel);
                float val01 = grid_cdfs[gy0][gx1].at<int>(pixel);
                float val10 = grid_cdfs[gy1][gx0].at<int>(pixel);
                float val11 = grid_cdfs[gy1][gx1].at<int>(pixel);

                float val = (1 - wy) * ((1 - wx) * val00 + wx * val01) +
                           wy * ((1 - wx) * val10 + wx * val11);

                // Normalize
                dst.at<uchar>(y, x) = saturate_cast<uchar>(val * 255.0f / (grid_width * grid_height));
            }
        }
    } else {
        // Convert to HSV space
        Mat hsv;
        cvtColor(src, hsv, COLOR_BGR2HSV);
        vector<Mat> channels;
        split(hsv, channels);

        // Apply CLAHE to V channel
        Mat v_dst;
        clahe(channels[2], v_dst, clip_limit, grid_size);
        v_dst.copyTo(channels[2]);

        merge(channels, hsv);
        cvtColor(hsv, dst, COLOR_HSV2BGR);
    }
}

void gamma_correction(const Mat& src, Mat& dst, double gamma) {
    CV_Assert(!src.empty());

    // Create lookup table
    uchar lut[256];
    for (int i = 0; i < 256; i++) {
        lut[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    }

    dst.create(src.size(), src.type());

    if (src.channels() == 1) {
        #pragma omp parallel for
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                dst.at<uchar>(y, x) = lut[src.at<uchar>(y, x)];
            }
        }
    } else {
        #pragma omp parallel for
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                const Vec3b& pixel = src.at<Vec3b>(y, x);
                dst.at<Vec3b>(y, x) = Vec3b(lut[pixel[0]], lut[pixel[1]], lut[pixel[2]]);
            }
        }
    }
}

void contrast_stretching(const Mat& src, Mat& dst,
                        double min_out, double max_out) {
    CV_Assert(!src.empty());

    // Find min and max values of input image
    double min_val, max_val;
    minMaxLoc(src, &min_val, &max_val);

    dst.create(src.size(), src.type());
    double scale = (max_out - min_out) / (max_val - min_val);

    if (src.channels() == 1) {
        #pragma omp parallel for
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                dst.at<uchar>(y, x) = saturate_cast<uchar>(
                    (src.at<uchar>(y, x) - min_val) * scale + min_out);
            }
        }
    } else {
        #pragma omp parallel for
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                const Vec3b& pixel = src.at<Vec3b>(y, x);
                dst.at<Vec3b>(y, x) = Vec3b(
                    saturate_cast<uchar>((pixel[0] - min_val) * scale + min_out),
                    saturate_cast<uchar>((pixel[1] - min_val) * scale + min_out),
                    saturate_cast<uchar>((pixel[2] - min_val) * scale + min_out)
                );
            }
        }
    }
}

void brightness_adjustment(const Mat& src, Mat& dst, double beta) {
    CV_Assert(!src.empty());

    dst.create(src.size(), src.type());

    if (src.channels() == 1) {
        #pragma omp parallel for
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                dst.at<uchar>(y, x) = saturate_cast<uchar>(src.at<uchar>(y, x) + beta);
            }
        }
    } else {
        #pragma omp parallel for
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                const Vec3b& pixel = src.at<Vec3b>(y, x);
                dst.at<Vec3b>(y, x) = Vec3b(
                    saturate_cast<uchar>(pixel[0] + beta),
                    saturate_cast<uchar>(pixel[1] + beta),
                    saturate_cast<uchar>(pixel[2] + beta)
                );
            }
        }
    }
}

void saturation_adjustment(const Mat& src, Mat& dst, double saturation) {
    CV_Assert(!src.empty() && src.type() == CV_8UC3);

    // Convert to HSV space
    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);

    vector<Mat> channels;
    split(hsv, channels);

    // Adjust saturation channel
    #pragma omp parallel for
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            channels[1].at<uchar>(y, x) = saturate_cast<uchar>(
                channels[1].at<uchar>(y, x) * saturation);
        }
    }

    merge(channels, hsv);
    cvtColor(hsv, dst, COLOR_HSV2BGR);
}

} // namespace ip101