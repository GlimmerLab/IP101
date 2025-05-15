#include <basic/thinning.hpp>
#include <omp.h>

namespace ip101 {

using namespace cv;
using namespace std;

namespace {
// Internal constant definitions
constexpr int CACHE_LINE = 64;    // CPU cache line size (bytes)
constexpr int BLOCK_SIZE = 16;    // Block processing size

// Check if point p is a boundary point
inline bool is_boundary(const Mat& img, int y, int x) {
    if (img.at<uchar>(y, x) == 0) return false;

    // Check if there is a background point in 8-neighborhood
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

// Count the number of connected components
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

// Count non-zero pixels in 8-neighborhood
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

        #pragma omp parallel for
        for (int y = 1; y < dst.rows - 1; y++) {
            for (int x = 1; x < dst.cols - 1; x++) {
                if (tmp.at<uchar>(y, x) == 0) continue;

                // Check if it is a boundary point
                if (!is_boundary(tmp, y, x)) continue;

                // Calculate values for P2 to P9
                int p2 = tmp.at<uchar>(y-1, x) > 0;
                int p3 = tmp.at<uchar>(y-1, x+1) > 0;
                int p4 = tmp.at<uchar>(y, x+1) > 0;
                int p5 = tmp.at<uchar>(y+1, x+1) > 0;
                int p6 = tmp.at<uchar>(y+1, x) > 0;
                int p7 = tmp.at<uchar>(y+1, x-1) > 0;
                int p8 = tmp.at<uchar>(y, x-1) > 0;
                int p9 = tmp.at<uchar>(y-1, x-1) > 0;

                // Condition 1: 2 <= B(P1) <= 6
                int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                if (B < 2 || B > 6) continue;

                // Condition 2: A(P1) = 1
                int A = count_transitions(tmp, y, x);
                if (A != 1) continue;

                // Conditions 3 and 4
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

        #pragma omp parallel for
        for (int y = 1; y < dst.rows - 1; y++) {
            for (int x = 1; x < dst.cols - 1; x++) {
                if (tmp.at<uchar>(y, x) == 0) continue;

                // Calculate Hilditch algorithm conditions
                int B = count_nonzero_neighbors(tmp, y, x);
                if (B < 2 || B > 6) continue;

                int A = count_transitions(tmp, y, x);
                if (A != 1) continue;

                // Calculate connectivity
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

        // First iteration
        Mat tmp = dst.clone();
        #pragma omp parallel for
        for (int y = 1; y < dst.rows - 1; y++) {
            for (int x = 1; x < dst.cols - 1; x++) {
                if (tmp.at<uchar>(y, x) == 0) continue;

                int B = count_nonzero_neighbors(tmp, y, x);
                if (B < 2 || B > 6) continue;

                int A = count_transitions(tmp, y, x);
                if (A != 1) continue;

                // Zhang-Suen condition 1
                if (tmp.at<uchar>(y-1, x) * tmp.at<uchar>(y, x+1) * tmp.at<uchar>(y+1, x) == 0 &&
                    tmp.at<uchar>(y, x+1) * tmp.at<uchar>(y+1, x) * tmp.at<uchar>(y, x-1) == 0) {
                    dst.at<uchar>(y, x) = 0;
                    has_changed = true;
                }
            }
        }

        // Second iteration
        tmp = dst.clone();
        #pragma omp parallel for
        for (int y = 1; y < dst.rows - 1; y++) {
            for (int x = 1; x < dst.cols - 1; x++) {
                if (tmp.at<uchar>(y, x) == 0) continue;

                int B = count_nonzero_neighbors(tmp, y, x);
                if (B < 2 || B > 6) continue;

                int A = count_transitions(tmp, y, x);
                if (A != 1) continue;

                // Zhang-Suen condition 2
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

    // Extract skeleton using distance transform and local maxima
    Mat dist;
    distanceTransform(src, dist, DIST_L2, DIST_MASK_PRECISE);

    dst = Mat::zeros(src.size(), CV_8UC1);

    #pragma omp parallel for
    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            if (src.at<uchar>(y, x) == 0) continue;

            // Check if it is a local maximum
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

    // Calculate distance transform
    distanceTransform(src, dist_transform, DIST_L2, DIST_MASK_PRECISE);

    // Extract medial axis
    dst = Mat::zeros(src.size(), CV_8UC1);

    #pragma omp parallel for
    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            if (src.at<uchar>(y, x) == 0) continue;

            float center = dist_transform.at<float>(y, x);
            bool is_medial_axis = false;

            // Check gradient direction
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dy == 0 && dx == 0) continue;
                    float neighbor = dist_transform.at<float>(y+dy, x+dx);

                    // If there is the same distance value in the opposite direction, it is a medial axis point
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