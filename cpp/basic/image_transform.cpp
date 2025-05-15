#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <immintrin.h> // for SSE/AVX
#include <thread>
#include <numeric>     // for std::accumulate
#include <type_traits> // for std::is_same_v
#include <basic/image_transform.hpp>

using namespace cv;
using namespace std;
using namespace std::chrono;

namespace ip101 {

namespace {
// Internal constants
constexpr int CACHE_LINE = 64;    // CPU cache line size (bytes)
constexpr int BLOCK_SIZE = 16;    // Block processing size

// Bilinear interpolation
template<typename T>
T bilinear_interpolation(const Mat& src, float x, float y) {
    int x1 = static_cast<int>(x);
    int y1 = static_cast<int>(y);
    int x2 = x1 + 1;
    int y2 = y1 + 1;

    float dx = x - x1;
    float dy = y - y1;

    T v1 = src.at<T>(y1, x1);
    T v2 = src.at<T>(y1, x2);
    T v3 = src.at<T>(y2, x1);
    T v4 = src.at<T>(y2, x2);

    if constexpr (std::is_same_v<T, uchar>) {
        float v12 = static_cast<float>(v1) + dx * (static_cast<float>(v2) - static_cast<float>(v1));
        float v34 = static_cast<float>(v3) + dx * (static_cast<float>(v4) - static_cast<float>(v3));
        return static_cast<T>(v12 + dy * (v34 - v12));
    } else {
        T v12 = v1 + dx * (v2 - v1);
        T v34 = v3 + dx * (v4 - v3);
        return v12 + dy * (v34 - v12);
    }
}

} // anonymous namespace

/**
 * @brief Affine transformation implementation
 * @param src Input image
 * @param src_points Three points in source image
 * @param dst_points Three points in target image
 * @return Transformed image
 */
Mat affineTransform(const Mat& src, const vector<Point2f>& src_points,
                   const vector<Point2f>& dst_points) {
    Mat dst = src.clone();
    int width = src.cols;
    int height = src.rows;

    // Calculate affine transform matrix
    Mat M = getAffineTransform(src_points, dst_points);

    // Apply transform
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            float new_x = static_cast<float>(M.at<double>(0,0) * x + M.at<double>(0,1) * y + M.at<double>(0,2));
            float new_y = static_cast<float>(M.at<double>(1,0) * x + M.at<double>(1,1) * y + M.at<double>(1,2));

            if(new_x >= 0 && new_x < width && new_y >= 0 && new_y < height) {
                dst.at<Vec3b>(y,x) = src.at<Vec3b>(static_cast<int>(new_y), static_cast<int>(new_x));
            }
        }
    }
    return dst;
}

/**
 * @brief SIMD optimized affine transformation implementation
 */
Mat affineTransform_optimized(const Mat& src, const vector<Point2f>& src_points,
                            const vector<Point2f>& dst_points) {
    Mat dst = src.clone();
    int width = src.cols;
    int height = src.rows;

    // Calculate affine transform matrix
    Mat M = getAffineTransform(src_points, dst_points);

    // Convert transform matrix to SIMD-friendly format
    __m256 m00 = _mm256_set1_ps(static_cast<float>(M.at<double>(0,0)));
    __m256 m01 = _mm256_set1_ps(static_cast<float>(M.at<double>(0,1)));
    __m256 m02 = _mm256_set1_ps(static_cast<float>(M.at<double>(0,2)));
    __m256 m10 = _mm256_set1_ps(static_cast<float>(M.at<double>(1,0)));
    __m256 m11 = _mm256_set1_ps(static_cast<float>(M.at<double>(1,1)));
    __m256 m12 = _mm256_set1_ps(static_cast<float>(M.at<double>(1,2)));

    #pragma omp parallel for
    for(int y = 0; y < height; y++) {
        for(int x = 0; x <= width - 8; x += 8) {
            // Create x coordinate vector
            __m256 x_vec = _mm256_set_ps(static_cast<float>(x+7), static_cast<float>(x+6), static_cast<float>(x+5), static_cast<float>(x+4),
                                          static_cast<float>(x+3), static_cast<float>(x+2), static_cast<float>(x+1), static_cast<float>(x));
            __m256 y_vec = _mm256_set1_ps(static_cast<float>(y));

            // Calculate new coordinates
            __m256 new_x = _mm256_fmadd_ps(m00, x_vec,
                          _mm256_fmadd_ps(m01, y_vec, m02));
            __m256 new_y = _mm256_fmadd_ps(m10, x_vec,
                          _mm256_fmadd_ps(m11, y_vec, m12));

            // Check bounds and copy pixels
            for(int i = 0; i < 8; i++) {
                float nx = ((float*)&new_x)[i];
                float ny = ((float*)&new_y)[i];
                if(nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    dst.at<Vec3b>(y,x+i) = src.at<Vec3b>(static_cast<int>(ny), static_cast<int>(nx));
                }
            }
        }
    }
    return dst;
}

/**
 * @brief Perspective transformation implementation
 * @param src Input image
 * @param src_points Four points in source image
 * @param dst_points Four points in target image
 * @return Transformed image
 */
Mat perspectiveTransform(const Mat& src, const vector<Point2f>& src_points,
                        const vector<Point2f>& dst_points) {
    Mat dst = src.clone();
    int width = src.cols;
    int height = src.rows;

    // Calculate perspective transform matrix
    Mat M = getPerspectiveTransform(src_points, dst_points);

    // Apply transform
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            float denominator = static_cast<float>(M.at<double>(2,0) * x + M.at<double>(2,1) * y + M.at<double>(2,2));
            if(denominator != 0) {
                float new_x = static_cast<float>((M.at<double>(0,0) * x + M.at<double>(0,1) * y + M.at<double>(0,2)) / denominator);
                float new_y = static_cast<float>((M.at<double>(1,0) * x + M.at<double>(1,1) * y + M.at<double>(1,2)) / denominator);

                if(new_x >= 0 && new_x < width && new_y >= 0 && new_y < height) {
                    dst.at<Vec3b>(y,x) = src.at<Vec3b>(static_cast<int>(new_y), static_cast<int>(new_x));
                }
            }
        }
    }
    return dst;
}

/**
 * @brief Image rotation implementation
 * @param src Input image
 * @param angle Rotation angle (degrees)
 * @param center Rotation center point
 * @return Rotated image
 */
Mat rotateImage(const Mat& src, double angle, Point2f center) {
    Mat dst = src.clone();
    int width = src.cols;
    int height = src.rows;

    // If center point not specified, use image center
    if(center.x == -1 || center.y == -1) {
        center = Point2f(static_cast<float>(width)/2.0f, static_cast<float>(height)/2.0f);
    }

    // Calculate rotation matrix
    Mat M = getRotationMatrix2D(center, angle, 1.0);

    // Apply transform
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            float new_x = static_cast<float>(M.at<double>(0,0) * x + M.at<double>(0,1) * y + M.at<double>(0,2));
            float new_y = static_cast<float>(M.at<double>(1,0) * x + M.at<double>(1,1) * y + M.at<double>(1,2));

            if(new_x >= 0 && new_x < width && new_y >= 0 && new_y < height) {
                dst.at<Vec3b>(y,x) = src.at<Vec3b>(static_cast<int>(new_y), static_cast<int>(new_x));
            }
        }
    }
    return dst;
}

/**
 * @brief Image scaling implementation
 * @param src Input image
 * @param scale_x X-direction scaling factor
 * @param scale_y Y-direction scaling factor
 * @return Scaled image
 */
Mat scaleImage(const Mat& src, double scale_x, double scale_y) {
    int new_width = static_cast<int>(src.cols * scale_x);
    int new_height = static_cast<int>(src.rows * scale_y);
    Mat dst(new_height, new_width, src.type());

    // Apply scaling
    for(int y = 0; y < new_height; y++) {
        for(int x = 0; x < new_width; x++) {
            int src_x = static_cast<int>(x / scale_x);
            int src_y = static_cast<int>(y / scale_y);

            if(src_x >= 0 && src_x < src.cols && src_y >= 0 && src_y < src.rows) {
                dst.at<Vec3b>(y,x) = src.at<Vec3b>(src_y, src_x);
            }
        }
    }
    return dst;
}

/**
 * @brief Image translation implementation
 * @param src Input image
 * @param tx X-direction translation amount
 * @param ty Y-direction translation amount
 * @return Translated image
 */
Mat translateImage(const Mat& src, int tx, int ty) {
    Mat dst = src.clone();
    int width = src.cols;
    int height = src.rows;

    // Apply translation
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            int new_x = x - tx;
            int new_y = y - ty;

            if(new_x >= 0 && new_x < width && new_y >= 0 && new_y < height) {
                dst.at<Vec3b>(y,x) = src.at<Vec3b>(new_y, new_x);
            }
        }
    }
    return dst;
}

/**
 * @brief Image mirroring implementation
 * @param src Input image
 * @param direction Mirror direction (0:horizontal, 1:vertical)
 * @return Mirrored image
 */
Mat mirrorImage(const Mat& src, int direction) {
    Mat dst = src.clone();
    int width = src.cols;
    int height = src.rows;

    // Apply mirroring
    if(direction == 0) { // Horizontal mirroring
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width/2; x++) {
                dst.at<Vec3b>(y,x) = src.at<Vec3b>(y,width-1-x);
                dst.at<Vec3b>(y,width-1-x) = src.at<Vec3b>(y,x);
            }
        }
    } else { // Vertical mirroring
        for(int y = 0; y < height/2; y++) {
            for(int x = 0; x < width; x++) {
                dst.at<Vec3b>(y,x) = src.at<Vec3b>(height-1-y,x);
                dst.at<Vec3b>(height-1-y,x) = src.at<Vec3b>(y,x);
            }
        }
    }
    return dst;
}

Mat affine_transform(const Mat& src, const Mat& M, const Size& size) {
    Mat dst(size, src.type());

    // Get transformation matrix elements
    float m00 = M.at<float>(0,0);
    float m01 = M.at<float>(0,1);
    float m02 = M.at<float>(0,2);
    float m10 = M.at<float>(1,0);
    float m11 = M.at<float>(1,1);
    float m12 = M.at<float>(1,2);

    #pragma omp parallel for
    for(int y = 0; y < dst.rows; y++) {
        for(int x = 0; x < dst.cols; x++) {
            // Calculate source image coordinates
            float src_x = m00 * x + m01 * y + m02;
            float src_y = m10 * x + m11 * y + m12;

            // Boundary check
            if(src_x >= 0 && src_x < src.cols-1 &&
               src_y >= 0 && src_y < src.rows-1) {
                if(src.type() == CV_8UC3) {
                    Vec3b p = bilinear_interpolation<Vec3b>(src, src_x, src_y);
                    dst.at<Vec3b>(y,x) = p;
                } else {
                    uchar p = bilinear_interpolation<uchar>(src, src_x, src_y);
                    dst.at<uchar>(y,x) = p;
                }
            }
        }
    }

    return dst;
}

Mat perspective_transform(const Mat& src, const Mat& M, const Size& size) {
    Mat dst(size, src.type());

    // Get transformation matrix elements
    float m00 = M.at<float>(0,0);
    float m01 = M.at<float>(0,1);
    float m02 = M.at<float>(0,2);
    float m10 = M.at<float>(1,0);
    float m11 = M.at<float>(1,1);
    float m12 = M.at<float>(1,2);
    float m20 = M.at<float>(2,0);
    float m21 = M.at<float>(2,1);
    float m22 = M.at<float>(2,2);

    #pragma omp parallel for
    for(int y = 0; y < dst.rows; y++) {
        for(int x = 0; x < dst.cols; x++) {
            // Calculate source image coordinates
            float denominator = m20 * x + m21 * y + m22;
            float src_x = (m00 * x + m01 * y + m02) / denominator;
            float src_y = (m10 * x + m11 * y + m12) / denominator;

            // Boundary check
            if(src_x >= 0 && src_x < src.cols-1 &&
               src_y >= 0 && src_y < src.rows-1) {
                if(src.type() == CV_8UC3) {
                    Vec3b p = bilinear_interpolation<Vec3b>(src, src_x, src_y);
                    dst.at<Vec3b>(y,x) = p;
                } else {
                    uchar p = bilinear_interpolation<uchar>(src, src_x, src_y);
                    dst.at<uchar>(y,x) = p;
                }
            }
        }
    }

    return dst;
}

Mat rotate(const Mat& src, double angle, const Point2f& center, double scale) {
    // Calculate rotation center
    Point2f center_point = center;
    if(center.x < 0 || center.y < 0) {
        center_point = Point2f(src.cols/2.0f, src.rows/2.0f);
    }

    // Calculate rotation matrix
    Mat M = getRotationMatrix2D(center_point, angle, scale);

    // Calculate rotated image size
    double alpha = angle * CV_PI / 180.0;
    double cos_alpha = fabs(cos(alpha));
    double sin_alpha = fabs(sin(alpha));

    int new_w = static_cast<int>(src.cols * cos_alpha + src.rows * sin_alpha);
    int new_h = static_cast<int>(src.cols * sin_alpha + src.rows * cos_alpha);

    // Adjust rotation center
    M.at<double>(0,2) += (new_w/2.0 - center_point.x);
    M.at<double>(1,2) += (new_h/2.0 - center_point.y);

    return affine_transform(src, M, Size(new_w, new_h));
}

Mat resize(const Mat& src, const Size& size, int interpolation) {
    Mat dst(size, src.type());

    float scale_x = static_cast<float>(src.cols) / size.width;
    float scale_y = static_cast<float>(src.rows) / size.height;

    #pragma omp parallel for
    for(int y = 0; y < dst.rows; y++) {
        for(int x = 0; x < dst.cols; x++) {
            float src_x = x * scale_x;
            float src_y = y * scale_y;

            if(src_x >= 0 && src_x < src.cols-1 &&
               src_y >= 0 && src_y < src.rows-1) {
                if(src.type() == CV_8UC3) {
                    Vec3b p = bilinear_interpolation<Vec3b>(src, src_x, src_y);
                    dst.at<Vec3b>(y,x) = p;
                } else {
                    uchar p = bilinear_interpolation<uchar>(src, src_x, src_y);
                    dst.at<uchar>(y,x) = p;
                }
            }
        }
    }

    return dst;
}

Mat translate(const Mat& src, double dx, double dy) {
    Mat M = (Mat_<float>(2,3) << 1, 0, dx, 0, 1, dy);
    return affine_transform(src, M, src.size());
}

Mat mirror(const Mat& src, int flip_code) {
    Mat dst(src.size(), src.type());

    if(flip_code == 0) { // Vertical flip
        #pragma omp parallel for
        for(int y = 0; y < src.rows; y++) {
            for(int x = 0; x < src.cols; x++) {
                dst.at<Vec3b>(y,x) = src.at<Vec3b>(src.rows-1-y,x);
            }
        }
    }
    else if(flip_code > 0) { // Horizontal flip
        #pragma omp parallel for
        for(int y = 0; y < src.rows; y++) {
            for(int x = 0; x < src.cols; x++) {
                dst.at<Vec3b>(y,x) = src.at<Vec3b>(y,src.cols-1-x);
            }
        }
    }
    else { // Both directions flip
        #pragma omp parallel for
        for(int y = 0; y < src.rows; y++) {
            for(int x = 0; x < src.cols; x++) {
                dst.at<Vec3b>(y,x) = src.at<Vec3b>(src.rows-1-y,src.cols-1-x);
            }
        }
    }

    return dst;
}

// 直接使用OpenCV的实现作为我们的API接口
Mat get_affine_transform(const std::vector<Point2f>& src_points,
                           const std::vector<Point2f>& dst_points) {
    return getAffineTransform(src_points, dst_points);
}

Mat get_perspective_transform(const std::vector<Point2f>& src_points,
                                const std::vector<Point2f>& dst_points) {
    return getPerspectiveTransform(src_points, dst_points);
}

Mat get_rotation_matrix(const Point2f& center,
                          double angle,
                          double scale) {
    return getRotationMatrix2D(center, angle, scale);
}

} // namespace ip101