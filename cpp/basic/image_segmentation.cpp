#include "image_segmentation.hpp"
#include <queue>
#include <random>
#include <algorithm>

namespace ip101 {

using namespace cv;

namespace {
// 内部常量定义
constexpr int CACHE_LINE = 64;    // CPU缓存行大小(字节)
constexpr int SIMD_WIDTH = 32;    // AVX2 SIMD向量宽度(字节)
constexpr int BLOCK_SIZE = 16;    // 分块处理大小

// 内存对齐辅助函数
inline uchar* alignPtr(uchar* ptr, size_t align = CACHE_LINE) {
    return (uchar*)(((size_t)ptr + align - 1) & -align);
}

// 计算两个颜色的欧氏距离
inline double colorDistance(const Vec3b& c1, const Vec3b& c2) {
    double diff0 = c1[0] - c2[0];
    double diff1 = c1[1] - c2[1];
    double diff2 = c1[2] - c2[2];
    return std::sqrt(diff0 * diff0 + diff1 * diff1 + diff2 * diff2);
}

} // anonymous namespace

void threshold_segmentation(const Mat& src, Mat& dst,
                          double threshold, double max_val,
                          int type) {
    CV_Assert(!src.empty());

    // 转换为灰度图
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    dst.create(gray.size(), CV_8UC1);

    // 使用OpenMP并行处理
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < gray.rows; y++) {
        for (int x = 0; x < gray.cols; x++) {
            uchar pixel = gray.at<uchar>(y, x);
            switch (type) {
                case THRESH_BINARY:
                    dst.at<uchar>(y, x) = pixel > threshold ? max_val : 0;
                    break;
                case THRESH_BINARY_INV:
                    dst.at<uchar>(y, x) = pixel > threshold ? 0 : max_val;
                    break;
                case THRESH_TRUNC:
                    dst.at<uchar>(y, x) = pixel > threshold ? threshold : pixel;
                    break;
                case THRESH_TOZERO:
                    dst.at<uchar>(y, x) = pixel > threshold ? pixel : 0;
                    break;
                case THRESH_TOZERO_INV:
                    dst.at<uchar>(y, x) = pixel > threshold ? 0 : pixel;
                    break;
            }
        }
    }
}

void kmeans_segmentation(const Mat& src, Mat& dst,
                        int k, int max_iter) {
    CV_Assert(!src.empty() && src.channels() == 3);

    // 将图像转换为浮点型数据
    Mat data;
    src.convertTo(data, CV_32F);
    data = data.reshape(1, src.rows * src.cols);

    // 随机初始化聚类中心
    std::vector<Vec3f> centers(k);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, src.rows * src.cols - 1);
    for (int i = 0; i < k; i++) {
        int idx = dis(gen);
        centers[i] = Vec3f(data.at<float>(idx, 0),
                          data.at<float>(idx, 1),
                          data.at<float>(idx, 2));
    }

    // K均值迭代
    std::vector<int> labels(src.rows * src.cols);
    for (int iter = 0; iter < max_iter; iter++) {
        // 分配标签
        #pragma omp parallel for
        for (int i = 0; i < src.rows * src.cols; i++) {
            float min_dist = FLT_MAX;
            int min_center = 0;
            Vec3f pixel(data.at<float>(i, 0),
                       data.at<float>(i, 1),
                       data.at<float>(i, 2));

            for (int j = 0; j < k; j++) {
                float dist = norm(pixel - centers[j]);
                if (dist < min_dist) {
                    min_dist = dist;
                    min_center = j;
                }
            }
            labels[i] = min_center;
        }

        // 更新聚类中心
        std::vector<Vec3f> new_centers(k, Vec3f(0, 0, 0));
        std::vector<int> counts(k, 0);

        #pragma omp parallel for
        for (int i = 0; i < src.rows * src.cols; i++) {
            int label = labels[i];
            Vec3f pixel(data.at<float>(i, 0),
                       data.at<float>(i, 1),
                       data.at<float>(i, 2));

            #pragma omp atomic
            new_centers[label][0] += pixel[0];
            #pragma omp atomic
            new_centers[label][1] += pixel[1];
            #pragma omp atomic
            new_centers[label][2] += pixel[2];
            #pragma omp atomic
            counts[label]++;
        }

        for (int i = 0; i < k; i++) {
            if (counts[i] > 0) {
                centers[i] = new_centers[i] / counts[i];
            }
        }
    }

    // 生成结果图像
    dst.create(src.size(), CV_8UC3);
    #pragma omp parallel for
    for (int i = 0; i < src.rows * src.cols; i++) {
        int y = i / src.cols;
        int x = i % src.cols;
        Vec3f center = centers[labels[i]];
        dst.at<Vec3b>(y, x) = Vec3b(saturate_cast<uchar>(center[0]),
                                   saturate_cast<uchar>(center[1]),
                                   saturate_cast<uchar>(center[2]));
    }
}

void region_growing(const Mat& src, Mat& dst,
                   const std::vector<Point>& seed_points,
                   double threshold) {
    CV_Assert(!src.empty() && !seed_points.empty());

    // 初始化结果图像
    dst = Mat::zeros(src.size(), CV_8UC1);

    // 对每个种子点进行区域生长
    for (const auto& seed : seed_points) {
        if (dst.at<uchar>(seed) > 0) continue;  // 跳过已处理的点

        std::queue<Point> points;
        points.push(seed);
        dst.at<uchar>(seed) = 255;

        Vec3b seed_color = src.at<Vec3b>(seed);

        while (!points.empty()) {
            Point current = points.front();
            points.pop();

            // 检查8邻域
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    Point neighbor(current.x + dx, current.y + dy);

                    if (neighbor.x >= 0 && neighbor.x < src.cols &&
                        neighbor.y >= 0 && neighbor.y < src.rows &&
                        dst.at<uchar>(neighbor) == 0) {

                        Vec3b neighbor_color = src.at<Vec3b>(neighbor);
                        double distance = colorDistance(seed_color, neighbor_color);

                        if (distance <= threshold) {
                            points.push(neighbor);
                            dst.at<uchar>(neighbor) = 255;
                        }
                    }
                }
            }
        }
    }
}

void watershed_segmentation(const Mat& src,
                          Mat& markers,
                          Mat& dst) {
    CV_Assert(!src.empty() && !markers.empty());

    // 转换标记图像为32位整型
    Mat markers32;
    markers.convertTo(markers32, CV_32S);

    // 应用分水岭算法
    watershed(src, markers32);

    // 生成结果图像
    dst = src.clone();
    for (int y = 0; y < markers32.rows; y++) {
        for (int x = 0; x < markers32.cols; x++) {
            int marker = markers32.at<int>(y, x);
            if (marker == -1) {  // 边界
                dst.at<Vec3b>(y, x) = Vec3b(0, 0, 255);
            }
        }
    }

    // 更新标记图像
    markers32.convertTo(markers, CV_8U);
}

void graph_cut_segmentation(const Mat& src, Mat& dst,
                          const Rect& rect) {
    CV_Assert(!src.empty());

    // 创建掩码
    Mat mask = Mat::zeros(src.size(), CV_8UC1);
    mask(rect) = GC_PR_FGD;  // 矩形区域内为可能前景

    // 创建临时数组
    Mat bgdModel, fgdModel;

    // 应用GrabCut算法
    grabCut(src, mask, rect, bgdModel, fgdModel, 5, GC_INIT_WITH_RECT);

    // 生成结果图像
    dst = src.clone();
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            if (mask.at<uchar>(y, x) == GC_BGD ||
                mask.at<uchar>(y, x) == GC_PR_BGD) {
                dst.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
            }
        }
    }
}

} // namespace ip101