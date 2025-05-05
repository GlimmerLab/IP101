#include "thinning.hpp"
#include <omp.h>

namespace ip101 {

using namespace cv;
using namespace std;

namespace {
// 内部常量定义
constexpr int CACHE_LINE = 64;    // CPU缓存行大小(字节)
constexpr int BLOCK_SIZE = 16;    // 分块处理大小

// 检查点p是否为边界点
inline bool is_boundary(const Mat& img, int y, int x) {
    if (img.at<uchar>(y, x) == 0) return false;

    // 检查8邻域是否存在背景点
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dy == 0 && dx == 0) continue;
            int ny = y + dy;
            int nx = x + dx;
            if (ny >= 0 && ny < img.rows && nx >= 0 && nx < img.cols) {
                if (img.at<uchar>(ny, nx) == 0) return true;
            }
        }
    }
    return false;
}

// 计算连通分量数
inline int count_transitions(const Mat& img, int y, int x) {
    int count = 0;
    vector<uchar> values = {
        img.at<uchar>(y-1, x),   // P2
        img.at<uchar>(y-1, x+1), // P3
        img.at<uchar>(y, x+1),   // P4
        img.at<uchar>(y+1, x+1), // P5
        img.at<uchar>(y+1, x),   // P6
        img.at<uchar>(y+1, x-1), // P7
        img.at<uchar>(y, x-1),   // P8
        img.at<uchar>(y-1, x-1), // P9
        img.at<uchar>(y-1, x)    // P2
    };

    for (size_t i = 0; i < values.size() - 1; i++) {
        if (values[i] == 0 && values[i+1] == 255) count++;
    }
    return count;
}

// 计算8邻域非零像素数
inline int count_nonzero_neighbors(const Mat& img, int y, int x) {
    int count = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dy == 0 && dx == 0) continue;
            int ny = y + dy;
            int nx = x + dx;
            if (ny >= 0 && ny < img.rows && nx >= 0 && nx < img.cols) {
                if (img.at<uchar>(ny, nx) > 0) count++;
            }
        }
    }
    return count;
}

} // anonymous namespace

void basic_thinning(const Mat& src, Mat& dst) {
    CV_Assert(!src.empty() && src.type() == CV_8UC1);

    src.copyTo(dst);
    bool has_changed;

    do {
        has_changed = false;
        Mat tmp = dst.clone();

        #pragma omp parallel for collapse(2)
        for (int y = 1; y < dst.rows - 1; y++) {
            for (int x = 1; x < dst.cols - 1; x++) {
                if (tmp.at<uchar>(y, x) == 0) continue;

                // 检查是否为边界点
                if (!is_boundary(tmp, y, x)) continue;

                // 计算P2到P9的值
                int p2 = tmp.at<uchar>(y-1, x) > 0;
                int p3 = tmp.at<uchar>(y-1, x+1) > 0;
                int p4 = tmp.at<uchar>(y, x+1) > 0;
                int p5 = tmp.at<uchar>(y+1, x+1) > 0;
                int p6 = tmp.at<uchar>(y+1, x) > 0;
                int p7 = tmp.at<uchar>(y+1, x-1) > 0;
                int p8 = tmp.at<uchar>(y, x-1) > 0;
                int p9 = tmp.at<uchar>(y-1, x-1) > 0;

                // 条件1: 2 <= B(P1) <= 6
                int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                if (B < 2 || B > 6) continue;

                // 条件2: A(P1) = 1
                int A = count_transitions(tmp, y, x);
                if (A != 1) continue;

                // 条件3和4
                if ((p2 * p4 * p6 == 0) && (p4 * p6 * p8 == 0)) {
                    dst.at<uchar>(y, x) = 0;
                    has_changed = true;
                }
            }
        }
    } while (has_changed);
}

void hilditch_thinning(const Mat& src, Mat& dst) {
    CV_Assert(!src.empty() && src.type() == CV_8UC1);

    src.copyTo(dst);
    bool has_changed;

    do {
        has_changed = false;
        Mat tmp = dst.clone();

        #pragma omp parallel for collapse(2)
        for (int y = 1; y < dst.rows - 1; y++) {
            for (int x = 1; x < dst.cols - 1; x++) {
                if (tmp.at<uchar>(y, x) == 0) continue;

                // 计算Hilditch算法的条件
                int B = count_nonzero_neighbors(tmp, y, x);
                if (B < 2 || B > 6) continue;

                int A = count_transitions(tmp, y, x);
                if (A != 1) continue;

                // 计算连通性
                int conn = 0;
                if (tmp.at<uchar>(y-1, x) > 0 && tmp.at<uchar>(y-1, x+1) > 0) conn++;
                if (tmp.at<uchar>(y-1, x+1) > 0 && tmp.at<uchar>(y, x+1) > 0) conn++;
                if (tmp.at<uchar>(y, x+1) > 0 && tmp.at<uchar>(y+1, x+1) > 0) conn++;
                if (tmp.at<uchar>(y+1, x+1) > 0 && tmp.at<uchar>(y+1, x) > 0) conn++;
                if (tmp.at<uchar>(y+1, x) > 0 && tmp.at<uchar>(y+1, x-1) > 0) conn++;
                if (tmp.at<uchar>(y+1, x-1) > 0 && tmp.at<uchar>(y, x-1) > 0) conn++;
                if (tmp.at<uchar>(y, x-1) > 0 && tmp.at<uchar>(y-1, x-1) > 0) conn++;
                if (tmp.at<uchar>(y-1, x-1) > 0 && tmp.at<uchar>(y-1, x) > 0) conn++;

                if (conn == 1) {
                    dst.at<uchar>(y, x) = 0;
                    has_changed = true;
                }
            }
        }
    } while (has_changed);
}

void zhang_suen_thinning(const Mat& src, Mat& dst) {
    CV_Assert(!src.empty() && src.type() == CV_8UC1);

    src.copyTo(dst);
    bool has_changed;

    do {
        has_changed = false;

        // 第一次迭代
        Mat tmp = dst.clone();
        #pragma omp parallel for collapse(2)
        for (int y = 1; y < dst.rows - 1; y++) {
            for (int x = 1; x < dst.cols - 1; x++) {
                if (tmp.at<uchar>(y, x) == 0) continue;

                int B = count_nonzero_neighbors(tmp, y, x);
                if (B < 2 || B > 6) continue;

                int A = count_transitions(tmp, y, x);
                if (A != 1) continue;

                // Zhang-Suen条件1
                if (tmp.at<uchar>(y-1, x) * tmp.at<uchar>(y, x+1) * tmp.at<uchar>(y+1, x) == 0 &&
                    tmp.at<uchar>(y, x+1) * tmp.at<uchar>(y+1, x) * tmp.at<uchar>(y, x-1) == 0) {
                    dst.at<uchar>(y, x) = 0;
                    has_changed = true;
                }
            }
        }

        // 第二次迭代
        tmp = dst.clone();
        #pragma omp parallel for collapse(2)
        for (int y = 1; y < dst.rows - 1; y++) {
            for (int x = 1; x < dst.cols - 1; x++) {
                if (tmp.at<uchar>(y, x) == 0) continue;

                int B = count_nonzero_neighbors(tmp, y, x);
                if (B < 2 || B > 6) continue;

                int A = count_transitions(tmp, y, x);
                if (A != 1) continue;

                // Zhang-Suen条件2
                if (tmp.at<uchar>(y-1, x) * tmp.at<uchar>(y, x+1) * tmp.at<uchar>(y, x-1) == 0 &&
                    tmp.at<uchar>(y-1, x) * tmp.at<uchar>(y+1, x) * tmp.at<uchar>(y, x-1) == 0) {
                    dst.at<uchar>(y, x) = 0;
                    has_changed = true;
                }
            }
        }
    } while (has_changed);
}

void skeleton_extraction(const Mat& src, Mat& dst) {
    CV_Assert(!src.empty() && src.type() == CV_8UC1);

    // 使用距离变换和局部最大值提取骨架
    Mat dist;
    distanceTransform(src, dist, DIST_L2, DIST_MASK_PRECISE);

    dst = Mat::zeros(src.size(), CV_8UC1);

    #pragma omp parallel for collapse(2)
    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            if (src.at<uchar>(y, x) == 0) continue;

            // 检查是否为局部最大值
            float center = dist.at<float>(y, x);
            bool is_local_max = true;

            for (int dy = -1; dy <= 1 && is_local_max; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dy == 0 && dx == 0) continue;
                    if (dist.at<float>(y+dy, x+dx) > center) {
                        is_local_max = false;
                        break;
                    }
                }
            }

            if (is_local_max) {
                dst.at<uchar>(y, x) = 255;
            }
        }
    }
}

void medial_axis_transform(const Mat& src, Mat& dst, Mat& dist_transform) {
    CV_Assert(!src.empty() && src.type() == CV_8UC1);

    // 计算距离变换
    distanceTransform(src, dist_transform, DIST_L2, DIST_MASK_PRECISE);

    // 提取中轴
    dst = Mat::zeros(src.size(), CV_8UC1);

    #pragma omp parallel for collapse(2)
    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            if (src.at<uchar>(y, x) == 0) continue;

            float center = dist_transform.at<float>(y, x);
            bool is_medial_axis = false;

            // 检查梯度方向
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dy == 0 && dx == 0) continue;
                    float neighbor = dist_transform.at<float>(y+dy, x+dx);

                    // 如果存在相反方向的相同距离值，则为中轴点
                    if (abs(center - neighbor) < 1e-5) {
                        int opposite_y = y - dy;
                        int opposite_x = x - dx;
                        if (opposite_y >= 0 && opposite_y < src.rows &&
                            opposite_x >= 0 && opposite_x < src.cols) {
                            float opposite = dist_transform.at<float>(opposite_y, opposite_x);
                            if (abs(center - opposite) < 1e-5) {
                                is_medial_axis = true;
                                break;
                            }
                        }
                    }
                }
                if (is_medial_axis) break;
            }

            if (is_medial_axis) {
                dst.at<uchar>(y, x) = 255;
            }
        }
    }
}

} // namespace ip101