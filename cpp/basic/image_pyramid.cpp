#include "image_pyramid.hpp"
#include <omp.h>

namespace ip101 {

using namespace cv;
using namespace std;

namespace {
// 内部常量定义
constexpr int CACHE_LINE = 64;    // CPU缓存行大小(字节)
constexpr int BLOCK_SIZE = 16;    // 分块处理大小

// 高斯核生成函数
Mat create_gaussian_kernel(float sigma) {
    int kernel_size = static_cast<int>(2 * ceil(3 * sigma) + 1);
    Mat kernel(kernel_size, kernel_size, CV_32F);
    float sum = 0.0f;

    int center = kernel_size / 2;
    float sigma2 = 2 * sigma * sigma;

    #pragma omp parallel for collapse(2) reduction(+:sum)
    for (int y = 0; y < kernel_size; y++) {
        for (int x = 0; x < kernel_size; x++) {
            float value = exp(-((x - center) * (x - center) +
                              (y - center) * (y - center)) / sigma2);
            kernel.at<float>(y, x) = value;
            sum += value;
        }
    }

    // 归一化
    kernel /= sum;
    return kernel;
}

// 高斯滤波优化实现
void gaussian_blur_simd(const Mat& src, Mat& dst, float sigma) {
    Mat kernel = create_gaussian_kernel(sigma);
    int kernel_size = kernel.rows;
    int radius = kernel_size / 2;

    dst.create(src.size(), CV_32F);

    // 水平方向滤波
    Mat temp(src.size(), CV_32F);
    #pragma omp parallel for collapse(2)
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

    // 垂直方向滤波
    #pragma omp parallel for collapse(2)
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

    // 转换为浮点型
    Mat current;
    src.convertTo(current, CV_32F, 1.0/255.0);
    pyramid.push_back(current);

    // 构建金字塔
    for (int i = 1; i < num_levels; i++) {
        Mat next;
        // 高斯滤波
        gaussian_blur_simd(current, next, 1.0);

        // 降采样
        pyrDown(next, next);
        pyramid.push_back(next);
        current = next;
    }

    return pyramid;
}

vector<Mat> build_laplacian_pyramid(const Mat& src, int num_levels) {
    vector<Mat> gaussian_pyramid = build_gaussian_pyramid(src, num_levels);
    vector<Mat> laplacian_pyramid(num_levels);

    // 构建拉普拉斯金字塔
    for (int i = 0; i < num_levels - 1; i++) {
        Mat up_level;
        pyrUp(gaussian_pyramid[i + 1], up_level, gaussian_pyramid[i].size());
        subtract(gaussian_pyramid[i], up_level, laplacian_pyramid[i]);
    }

    // 最顶层直接使用高斯金字塔的结果
    laplacian_pyramid[num_levels - 1] = gaussian_pyramid[num_levels - 1];

    return laplacian_pyramid;
}

Mat pyramid_blend(const Mat& src1, const Mat& src2,
                 const Mat& mask, int num_levels) {
    // 构建两个图像的拉普拉斯金字塔
    vector<Mat> lap1 = build_laplacian_pyramid(src1, num_levels);
    vector<Mat> lap2 = build_laplacian_pyramid(src2, num_levels);

    // 构建掩码的高斯金字塔
    vector<Mat> gauss_mask = build_gaussian_pyramid(mask, num_levels);

    // 在每一层进行融合
    vector<Mat> blend_pyramid(num_levels);
    #pragma omp parallel for
    for (int i = 0; i < num_levels; i++) {
        blend_pyramid[i] = lap1[i].mul(gauss_mask[i]) +
                          lap2[i].mul(1.0 - gauss_mask[i]);
    }

    // 重建融合图像
    Mat result = blend_pyramid[num_levels - 1];
    for (int i = num_levels - 2; i >= 0; i--) {
        pyrUp(result, result, blend_pyramid[i].size());
        result += blend_pyramid[i];
    }

    // 转换回8位图像
    result.convertTo(result, CV_8U, 255.0);
    return result;
}

vector<vector<Mat>> build_sift_scale_space(
    const Mat& src, int num_octaves, int num_scales, float sigma) {

    vector<vector<Mat>> scale_space(num_octaves);
    for (auto& octave : scale_space) {
        octave.resize(num_scales);
    }

    // 初始化第一组第一层
    Mat base;
    src.convertTo(base, CV_32F, 1.0/255.0);
    gaussian_blur_simd(base, scale_space[0][0], sigma);

    // 构建尺度空间
    float k = pow(2.0f, 1.0f / (num_scales - 3));
    for (int o = 0; o < num_octaves; o++) {
        for (int s = 1; s < num_scales; s++) {
            float sig = sigma * pow(k, s);
            gaussian_blur_simd(scale_space[o][s-1],
                             scale_space[o][s], sig);
        }

        if (o < num_octaves - 1) {
            // 降采样作为下一组的基础图像
            pyrDown(scale_space[o][num_scales-1],
                   scale_space[o+1][0]);
        }
    }

    return scale_space;
}

Mat saliency_detection(const Mat& src, int num_levels) {
    // 构建高斯金字塔
    vector<Mat> pyramid = build_gaussian_pyramid(src, num_levels);

    // 计算显著性图
    Mat saliency = Mat::zeros(src.size(), CV_32F);

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            float center_value = src.at<float>(y, x);
            float sum_diff = 0.0f;

            // 计算与其他尺度的差异
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

    // 归一化
    normalize(saliency, saliency, 0, 1, NORM_MINMAX);
    saliency.convertTo(saliency, CV_8U, 255);

    return saliency;
}

Mat visualize_pyramid(const vector<Mat>& pyramid, int padding) {
    // 计算总宽度和最大高度
    int total_width = 0;
    int max_height = 0;
    for (const auto& level : pyramid) {
        total_width += level.cols + padding;
        max_height = max(max_height, level.rows);
    }
    total_width -= padding;  // 最后一个不需要padding

    // 创建显示图像
    Mat display = Mat::zeros(max_height, total_width, CV_8UC3);

    // 填充每一层
    int x_offset = 0;
    for (const auto& level : pyramid) {
        Mat color_level;
        if (level.channels() == 1) {
            cvtColor(level, color_level, COLOR_GRAY2BGR);
        } else {
            color_level = level.clone();
        }

        // 复制到显示图像
        Rect roi(x_offset, 0, level.cols, level.rows);
        color_level.copyTo(display(roi));

        // 更新偏移
        x_offset += level.cols + padding;
    }

    return display;
}

} // namespace ip101