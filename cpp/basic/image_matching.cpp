#include <basic/image_matching.hpp>
#include <cmath>

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

// SIMD optimized version of SSD computation
void compute_ssd_simd(const Mat& src, const Mat& templ, Mat& result) {
    int h = templ.rows;
    int w = templ.cols;
    int H = src.rows;
    int W = src.cols;

    result.create(H-h+1, W-w+1, CV_32F);
    result = Scalar(0);

    #pragma omp parallel for
    for (int y = 0; y < H-h+1; y++) {
        for (int x = 0; x < W-w+1; x++) {
            float sum = 0;
            for (int i = 0; i < h; i++) {
                const uchar* src_ptr = src.ptr<uchar>(y+i) + x;
                const uchar* templ_ptr = templ.ptr<uchar>(i);

                // Use AVX2 for vectorized computation
                for (int j = 0; j < w; j += 8) {
                    __m256i src_vec = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(src_ptr+j)));
                    __m256i templ_vec = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(templ_ptr+j)));
                    __m256i diff = _mm256_sub_epi32(src_vec, templ_vec);
                    __m256i square = _mm256_mullo_epi32(diff, diff);

                    float temp[8];
                    _mm256_storeu_ps(temp, _mm256_cvtepi32_ps(square));
                    for (int k = 0; k < 8 && j+k < w; k++) {
                        sum += temp[k];
                    }
                }
            }
            result.at<float>(y, x) = sum;
        }
    }
}

// SIMD optimized version of SAD computation
void compute_sad_simd(const Mat& src, const Mat& templ, Mat& result) {
    int h = templ.rows;
    int w = templ.cols;
    int H = src.rows;
    int W = src.cols;

    result.create(H-h+1, W-w+1, CV_32F);
    result = Scalar(0);

    #pragma omp parallel for
    for (int y = 0; y < H-h+1; y++) {
        for (int x = 0; x < W-w+1; x++) {
            float sum = 0;
            for (int i = 0; i < h; i++) {
                const uchar* src_ptr = src.ptr<uchar>(y+i) + x;
                const uchar* templ_ptr = templ.ptr<uchar>(i);

                // Use AVX2 for vectorized computation
                for (int j = 0; j < w; j += 8) {
                    __m256i src_vec = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(src_ptr+j)));
                    __m256i templ_vec = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(templ_ptr+j)));
                    __m256i diff = _mm256_abs_epi32(_mm256_sub_epi32(src_vec, templ_vec));

                    float temp[8];
                    _mm256_storeu_ps(temp, _mm256_cvtepi32_ps(diff));
                    for (int k = 0; k < 8 && j+k < w; k++) {
                        sum += temp[k];
                    }
                }
            }
            result.at<float>(y, x) = sum;
        }
    }
}

// SIMD optimized version of NCC computation
void compute_ncc_simd(const Mat& src, const Mat& templ, Mat& result) {
    int h = templ.rows;
    int w = templ.cols;
    int H = src.rows;
    int W = src.cols;

    result.create(H-h+1, W-w+1, CV_32F);
    result = Scalar(0);

    // Calculate template norm
    float templ_norm = 0;
    for (int i = 0; i < h; i++) {
        const uchar* templ_ptr = templ.ptr<uchar>(i);
        for (int j = 0; j < w; j++) {
            templ_norm += templ_ptr[j] * templ_ptr[j];
        }
    }
    templ_norm = sqrt(templ_norm);

    #pragma omp parallel for
    for (int y = 0; y < H-h+1; y++) {
        for (int x = 0; x < W-w+1; x++) {
            float window_norm = 0;
            float dot_product = 0;

            for (int i = 0; i < h; i++) {
                const uchar* src_ptr = src.ptr<uchar>(y+i) + x;
                const uchar* templ_ptr = templ.ptr<uchar>(i);

                // Use AVX2 for vectorized computation
                for (int j = 0; j < w; j += 8) {
                    __m256i src_vec = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(src_ptr+j)));
                    __m256i templ_vec = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(templ_ptr+j)));

                    // Calculate dot product
                    __m256i product = _mm256_mullo_epi32(src_vec, templ_vec);
                    float temp[8];
                    _mm256_storeu_ps(temp, _mm256_cvtepi32_ps(product));
                    for (int k = 0; k < 8 && j+k < w; k++) {
                        dot_product += temp[k];
                    }

                    // Calculate window norm
                    __m256i square = _mm256_mullo_epi32(src_vec, src_vec);
                    _mm256_storeu_ps(temp, _mm256_cvtepi32_ps(square));
                    for (int k = 0; k < 8 && j+k < w; k++) {
                        window_norm += temp[k];
                    }
                }
            }

            window_norm = sqrt(window_norm);
            if (window_norm > 0 && templ_norm > 0) {
                result.at<float>(y, x) = dot_product / (window_norm * templ_norm);
            }
        }
    }
}

// SIMD optimized version of ZNCC computation
void compute_zncc_simd(const Mat& src, const Mat& templ, Mat& result) {
    int h = templ.rows;
    int w = templ.cols;
    int H = src.rows;
    int W = src.cols;

    result.create(H-h+1, W-w+1, CV_32F);
    result = Scalar(0);

    // Calculate template mean and standard deviation
    float templ_mean = 0;
    float templ_std = 0;
    for (int i = 0; i < h; i++) {
        const uchar* templ_ptr = templ.ptr<uchar>(i);
        for (int j = 0; j < w; j++) {
            templ_mean += templ_ptr[j];
        }
    }
    templ_mean /= (h * w);

    for (int i = 0; i < h; i++) {
        const uchar* templ_ptr = templ.ptr<uchar>(i);
        for (int j = 0; j < w; j++) {
            float diff = templ_ptr[j] - templ_mean;
            templ_std += diff * diff;
        }
    }
    templ_std = sqrt(templ_std / (h * w));

    #pragma omp parallel for
    for (int y = 0; y < H-h+1; y++) {
        for (int x = 0; x < W-w+1; x++) {
            // Calculate window mean and standard deviation
            float window_mean = 0;
            float window_std = 0;
            float zncc = 0;

            for (int i = 0; i < h; i++) {
                const uchar* src_ptr = src.ptr<uchar>(y+i) + x;
                for (int j = 0; j < w; j++) {
                    window_mean += src_ptr[j];
                }
            }
            window_mean /= (h * w);

            for (int i = 0; i < h; i++) {
                const uchar* src_ptr = src.ptr<uchar>(y+i) + x;
                const uchar* templ_ptr = templ.ptr<uchar>(i);
                for (int j = 0; j < w; j++) {
                    float src_diff = src_ptr[j] - window_mean;
                    float templ_diff = templ_ptr[j] - templ_mean;
                    window_std += src_diff * src_diff;
                    zncc += src_diff * templ_diff;
                }
            }
            window_std = sqrt(window_std / (h * w));

            if (window_std > 0 && templ_std > 0) {
                result.at<float>(y, x) = zncc / (window_std * templ_std * h * w);
            }
        }
    }
}

} // anonymous namespace

void ssd_matching(const Mat& src, const Mat& templ, Mat& result, int method) {
    CV_Assert(!src.empty() && !templ.empty());
    CV_Assert(src.type() == CV_8UC1 && templ.type() == CV_8UC1);

    compute_ssd_simd(src, templ, result);
}

void sad_matching(const Mat& src, const Mat& templ, Mat& result) {
    CV_Assert(!src.empty() && !templ.empty());
    CV_Assert(src.type() == CV_8UC1 && templ.type() == CV_8UC1);

    compute_sad_simd(src, templ, result);
}

void ncc_matching(const Mat& src, const Mat& templ, Mat& result) {
    CV_Assert(!src.empty() && !templ.empty());
    CV_Assert(src.type() == CV_8UC1 && templ.type() == CV_8UC1);

    compute_ncc_simd(src, templ, result);
}

void zncc_matching(const Mat& src, const Mat& templ, Mat& result) {
    CV_Assert(!src.empty() && !templ.empty());
    CV_Assert(src.type() == CV_8UC1 && templ.type() == CV_8UC1);

    compute_zncc_simd(src, templ, result);
}

void feature_point_matching(const Mat& src1, const Mat& src2,
                          vector<DMatch>& matches,
                          vector<KeyPoint>& keypoints1,
                          vector<KeyPoint>& keypoints2) {
    CV_Assert(!src1.empty() && !src2.empty());
    CV_Assert(src1.type() == CV_8UC1 && src2.type() == CV_8UC1);

    // Use SIFT feature detector and descriptor
    Ptr<Feature2D> sift = SIFT::create();
    Mat descriptors1, descriptors2;

    // Detect keypoints and compute descriptors
    sift->detectAndCompute(src1, noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(src2, noArray(), keypoints2, descriptors2);

    // Use FLANN matcher for feature matching
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    vector<vector<DMatch>> knn_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

    // Filter good matches using Lowe's ratio test
    const float ratio_thresh = 0.7f;
    matches.clear();
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            matches.push_back(knn_matches[i][0]);
        }
    }
}

Mat draw_matching_result(const Mat& src, const Mat& templ, const Mat& result, int method) {
    CV_Assert(!src.empty() && !templ.empty() && !result.empty());

    Mat img_color;
    if (src.channels() == 1) {
        cvtColor(src, img_color, COLOR_GRAY2BGR);
    } else {
        img_color = src.clone();
    }

    double min_val, max_val;
    Point min_loc, max_loc;
    minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);

    Point match_loc;
    if (method == TM_SQDIFF || method == TM_SQDIFF_NORMED) {
        match_loc = min_loc;
    } else {
        match_loc = max_loc;
    }

    rectangle(img_color, match_loc,
             Point(match_loc.x + templ.cols, match_loc.y + templ.rows),
             Scalar(0, 0, 255), 2);

    return img_color;
}

} // namespace ip101