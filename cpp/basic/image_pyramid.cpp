#include <basic/image_pyramid.hpp>
#include <omp.h>

namespace ip101 {

using namespace cv;
using namespace std;

namespace {
// Internal constants
constexpr int CACHE_LINE = 64;    // CPU cache line size (bytes)
constexpr int BLOCK_SIZE = 16;    // Block processing size

// Generate Gaussian kernel
Mat create_gaussian_kernel(float sigma) {
    int kernel_size = static_cast<int>(2 * ceil(3 * sigma) + 1);
    Mat kernel(kernel_size, kernel_size, CV_32F);
    float sum = 0.0f;

    int center = kernel_size / 2;
    float sigma2 = 2 * sigma * sigma;

    #pragma omp parallel for reduction(+:sum)
    for (int y = 0; y < kernel_size; y++) {
        for (int x = 0; x < kernel_size; x++) {
            float value = exp(-((x - center) * (x - center) +
                              (y - center) * (y - center)) / sigma2);
            kernel.at<float>(y, x) = value;
            sum += value;
        }
    }

    // Normalize
    kernel /= sum;
    return kernel;
}

// Optimized Gaussian blur implementation
void gaussian_blur_simd(const Mat& src, Mat& dst, float sigma) {
    Mat kernel = create_gaussian_kernel(sigma);
    int kernel_size = kernel.rows;
    int radius = kernel_size / 2;

    dst.create(src.size(), CV_32F);

    // Horizontal pass
    Mat temp(src.size(), CV_32F);
    #pragma omp parallel for
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            float sum = 0.0f;
            for (int i = -radius; i <= radius; i++) {
                int xx = x + i;
                if (xx < 0) xx = 0;
                if (xx >= src.cols) xx = src.cols - 1;
                sum += src.at<float>(y, xx) * kernel.at<float>(0, i + radius);
            }
            temp.at<float>(y, x) = sum;
        }
    }

    // Vertical pass
    #pragma omp parallel for
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            float sum = 0.0f;
            for (int i = -radius; i <= radius; i++) {
                int yy = y + i;
                if (yy < 0) yy = 0;
                if (yy >= src.rows) yy = src.rows - 1;
                sum += temp.at<float>(yy, x) * kernel.at<float>(i + radius, 0);
            }
            dst.at<float>(y, x) = sum;
        }
    }
}

} // anonymous namespace

vector<Mat> build_gaussian_pyramid(const Mat& src, int num_levels) {
    vector<Mat> pyramid;
    pyramid.reserve(num_levels);

    // Convert to float
    Mat current;
    src.convertTo(current, CV_32F, 1.0/255.0);
    pyramid.push_back(current);

    // Build pyramid
    for (int i = 1; i < num_levels; i++) {
        Mat next;
        // Gaussian blur
        gaussian_blur_simd(current, next, 1.0);

        // Downsample
        pyrDown(next, next);
        pyramid.push_back(next);
        current = next;
    }

    return pyramid;
}

vector<Mat> build_laplacian_pyramid(const Mat& src, int num_levels) {
    vector<Mat> gaussian_pyramid = build_gaussian_pyramid(src, num_levels);
    vector<Mat> laplacian_pyramid(num_levels);

    // Build Laplacian pyramid
    for (int i = 0; i < num_levels - 1; i++) {
        Mat up_level;
        pyrUp(gaussian_pyramid[i + 1], up_level, gaussian_pyramid[i].size());
        subtract(gaussian_pyramid[i], up_level, laplacian_pyramid[i]);
    }

    // Use Gaussian pyramid result for the top level
    laplacian_pyramid[num_levels - 1] = gaussian_pyramid[num_levels - 1];

    return laplacian_pyramid;
}

Mat pyramid_blend(const Mat& src1, const Mat& src2,
                 const Mat& mask, int num_levels) {
    // Build Laplacian pyramids for both images
    vector<Mat> lap1 = build_laplacian_pyramid(src1, num_levels);
    vector<Mat> lap2 = build_laplacian_pyramid(src2, num_levels);

    // Build Gaussian pyramid for the mask
    vector<Mat> gauss_mask = build_gaussian_pyramid(mask, num_levels);

    // Blend at each level
    vector<Mat> blend_pyramid(num_levels);
    #pragma omp parallel for
    for (int i = 0; i < num_levels; i++) {
        blend_pyramid[i] = lap1[i].mul(gauss_mask[i]) +
                          lap2[i].mul(1.0 - gauss_mask[i]);
    }

    // Reconstruct blended image
    Mat result = blend_pyramid[num_levels - 1];
    for (int i = num_levels - 2; i >= 0; i--) {
        pyrUp(result, result, blend_pyramid[i].size());
        result += blend_pyramid[i];
    }

    // Convert back to 8-bit image
    result.convertTo(result, CV_8U, 255.0);
    return result;
}

vector<vector<Mat>> build_sift_scale_space(
    const Mat& src, int num_octaves, int num_scales, float sigma) {

    vector<vector<Mat>> scale_space(num_octaves);
    for (auto& octave : scale_space) {
        octave.resize(num_scales);
    }

    // Initialize first level of first octave
    Mat base;
    src.convertTo(base, CV_32F, 1.0/255.0);
    gaussian_blur_simd(base, scale_space[0][0], sigma);

    // Build scale space
    float k = pow(2.0f, 1.0f / (num_scales - 3));
    for (int o = 0; o < num_octaves; o++) {
        for (int s = 1; s < num_scales; s++) {
            float sig = sigma * pow(k, s);
            gaussian_blur_simd(scale_space[o][s-1],
                             scale_space[o][s], sig);
        }

        if (o < num_octaves - 1) {
            // Downsample for the base image of next octave
            pyrDown(scale_space[o][num_scales-1],
                   scale_space[o+1][0]);
        }
    }

    return scale_space;
}

Mat saliency_detection(const Mat& src, int num_levels) {
    // Build Gaussian pyramid
    vector<Mat> pyramid = build_gaussian_pyramid(src, num_levels);

    // Calculate saliency map
    Mat saliency = Mat::zeros(src.size(), CV_32F);

    #pragma omp parallel for
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            float center_value = src.at<float>(y, x);
            float sum_diff = 0.0f;

            // Calculate differences with other scales
            for (int l = 1; l < num_levels; l++) {
                Mat& level = pyramid[l];
                float scale = static_cast<float>(src.rows) / level.rows;
                int py = static_cast<int>(y / scale);
                int px = static_cast<int>(x / scale);

                if (py >= level.rows) py = level.rows - 1;
                if (px >= level.cols) px = level.cols - 1;

                float surround_value = level.at<float>(py, px);
                sum_diff += abs(center_value - surround_value);
            }

            saliency.at<float>(y, x) = sum_diff / (num_levels - 1);
        }
    }

    // Normalize
    normalize(saliency, saliency, 0, 1, NORM_MINMAX);
    saliency.convertTo(saliency, CV_8U, 255);

    return saliency;
}

Mat visualize_pyramid(const vector<Mat>& pyramid, int padding) {
    // Calculate total width and maximum height
    int total_width = 0;
    int max_height = 0;
    for (const auto& level : pyramid) {
        total_width += level.cols + padding;
        max_height = max(max_height, level.rows);
    }
    total_width -= padding;  // No padding needed for the last one

    // Create display image
    Mat display = Mat::zeros(max_height, total_width, CV_8UC3);

    // Fill in each level
    int x_offset = 0;
    for (const auto& level : pyramid) {
        Mat color_level;
        if (level.channels() == 1) {
            cvtColor(level, color_level, COLOR_GRAY2BGR);
        } else {
            color_level = level.clone();
        }

        // Copy to display image
        Rect roi(x_offset, 0, level.cols, level.rows);
        color_level.copyTo(display(roi));

        // Update offset
        x_offset += level.cols + padding;
    }

    return display;
}

} // namespace ip101