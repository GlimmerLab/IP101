#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <immintrin.h> // for SSE/AVX
#include <thread>
#include "image_transform.hpp"

using namespace cv;
using namespace std;
using namespace std::chrono;

// 常量定义
constexpr int CACHE_LINE = 64;  // 缓存行大小
constexpr int SIMD_WIDTH = 32;  // AVX2 宽度
constexpr int BLOCK_SIZE = 16;  // 分块大小

namespace ip101 {

namespace {
// 内部常量定义
constexpr int CACHE_LINE = 64;    // CPU缓存行大小(字节)
constexpr int BLOCK_SIZE = 16;    // 分块处理大小

// 双线性插值
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

    T v12 = v1 + dx * (v2 - v1);
    T v34 = v3 + dx * (v4 - v3);
    return v12 + dy * (v34 - v12);
}

} // anonymous namespace

/**
 * @brief 仿射变换实现
 * @param src 输入图像
 * @param src_points 源图像中的三个点坐标
 * @param dst_points 目标图像中的三个点坐标
 * @return 变换后的图像
 */
Mat affineTransform(const Mat& src, const vector<Point2f>& src_points,
                   const vector<Point2f>& dst_points) {
    Mat dst = src.clone();
    int width = src.cols;
    int height = src.rows;

    // 计算仿射变换矩阵
    Mat M = getAffineTransform(src_points, dst_points);

    // 应用变换
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            float new_x = M.at<double>(0,0) * x + M.at<double>(0,1) * y + M.at<double>(0,2);
            float new_y = M.at<double>(1,0) * x + M.at<double>(1,1) * y + M.at<double>(1,2);

            if(new_x >= 0 && new_x < width && new_y >= 0 && new_y < height) {
                dst.at<Vec3b>(y,x) = src.at<Vec3b>(new_y, new_x);
            }
        }
    }
    return dst;
}

/**
 * @brief 使用SIMD优化的仿射变换实现
 */
Mat affineTransform_optimized(const Mat& src, const vector<Point2f>& src_points,
                            const vector<Point2f>& dst_points) {
    Mat dst = src.clone();
    int width = src.cols;
    int height = src.rows;

    // 计算仿射变换矩阵
    Mat M = getAffineTransform(src_points, dst_points);

    // 将变换矩阵转换为SIMD友好的格式
    __m256 m00 = _mm256_set1_ps(M.at<double>(0,0));
    __m256 m01 = _mm256_set1_ps(M.at<double>(0,1));
    __m256 m02 = _mm256_set1_ps(M.at<double>(0,2));
    __m256 m10 = _mm256_set1_ps(M.at<double>(1,0));
    __m256 m11 = _mm256_set1_ps(M.at<double>(1,1));
    __m256 m12 = _mm256_set1_ps(M.at<double>(1,2));

    #pragma omp parallel for
    for(int y = 0; y < height; y++) {
        for(int x = 0; x <= width - 8; x += 8) {
            // 创建x坐标向量
            __m256 x_vec = _mm256_set_ps(x+7, x+6, x+5, x+4, x+3, x+2, x+1, x);
            __m256 y_vec = _mm256_set1_ps(y);

            // 计算新坐标
            __m256 new_x = _mm256_fmadd_ps(m00, x_vec,
                          _mm256_fmadd_ps(m01, y_vec, m02));
            __m256 new_y = _mm256_fmadd_ps(m10, x_vec,
                          _mm256_fmadd_ps(m11, y_vec, m12));

            // 检查边界并复制像素
            for(int i = 0; i < 8; i++) {
                float nx = ((float*)&new_x)[i];
                float ny = ((float*)&new_y)[i];
                if(nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    dst.at<Vec3b>(y,x+i) = src.at<Vec3b>(ny, nx);
                }
            }
        }
    }
    return dst;
}

/**
 * @brief 透视变换实现
 * @param src 输入图像
 * @param src_points 源图像中的四个点坐标
 * @param dst_points 目标图像中的四个点坐标
 * @return 变换后的图像
 */
Mat perspectiveTransform(const Mat& src, const vector<Point2f>& src_points,
                        const vector<Point2f>& dst_points) {
    Mat dst = src.clone();
    int width = src.cols;
    int height = src.rows;

    // 计算透视变换矩阵
    Mat M = getPerspectiveTransform(src_points, dst_points);

    // 应用变换
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            float denominator = M.at<double>(2,0) * x + M.at<double>(2,1) * y + M.at<double>(2,2);
            if(denominator != 0) {
                float new_x = (M.at<double>(0,0) * x + M.at<double>(0,1) * y + M.at<double>(0,2)) / denominator;
                float new_y = (M.at<double>(1,0) * x + M.at<double>(1,1) * y + M.at<double>(1,2)) / denominator;

                if(new_x >= 0 && new_x < width && new_y >= 0 && new_y < height) {
                    dst.at<Vec3b>(y,x) = src.at<Vec3b>(new_y, new_x);
                }
            }
        }
    }
    return dst;
}

/**
 * @brief 图像旋转实现
 * @param src 输入图像
 * @param angle 旋转角度（度）
 * @param center 旋转中心点
 * @return 旋转后的图像
 */
Mat rotateImage(const Mat& src, double angle, Point2f center = Point2f(-1,-1)) {
    Mat dst = src.clone();
    int width = src.cols;
    int height = src.rows;

    // 如果未指定中心点，使用图像中心
    if(center.x == -1 || center.y == -1) {
        center = Point2f(width/2, height/2);
    }

    // 计算旋转矩阵
    Mat M = getRotationMatrix2D(center, angle, 1.0);

    // 应用变换
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            float new_x = M.at<double>(0,0) * x + M.at<double>(0,1) * y + M.at<double>(0,2);
            float new_y = M.at<double>(1,0) * x + M.at<double>(1,1) * y + M.at<double>(1,2);

            if(new_x >= 0 && new_x < width && new_y >= 0 && new_y < height) {
                dst.at<Vec3b>(y,x) = src.at<Vec3b>(new_y, new_x);
            }
        }
    }
    return dst;
}

/**
 * @brief 图像缩放实现
 * @param src 输入图像
 * @param scale_x x方向缩放比例
 * @param scale_y y方向缩放比例
 * @return 缩放后的图像
 */
Mat scaleImage(const Mat& src, double scale_x, double scale_y) {
    int new_width = src.cols * scale_x;
    int new_height = src.rows * scale_y;
    Mat dst(new_height, new_width, src.type());

    // 应用缩放
    for(int y = 0; y < new_height; y++) {
        for(int x = 0; x < new_width; x++) {
            int src_x = x / scale_x;
            int src_y = y / scale_y;

            if(src_x >= 0 && src_x < src.cols && src_y >= 0 && src_y < src.rows) {
                dst.at<Vec3b>(y,x) = src.at<Vec3b>(src_y, src_x);
            }
        }
    }
    return dst;
}

/**
 * @brief 图像平移实现
 * @param src 输入图像
 * @param tx x方向平移量
 * @param ty y方向平移量
 * @return 平移后的图像
 */
Mat translateImage(const Mat& src, int tx, int ty) {
    Mat dst = src.clone();
    int width = src.cols;
    int height = src.rows;

    // 应用平移
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
 * @brief 图像镜像实现
 * @param src 输入图像
 * @param direction 镜像方向（0:水平，1:垂直）
 * @return 镜像后的图像
 */
Mat mirrorImage(const Mat& src, int direction = 0) {
    Mat dst = src.clone();
    int width = src.cols;
    int height = src.rows;

    // 应用镜像
    if(direction == 0) { // 水平镜像
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width/2; x++) {
                dst.at<Vec3b>(y,x) = src.at<Vec3b>(y,width-1-x);
                dst.at<Vec3b>(y,width-1-x) = src.at<Vec3b>(y,x);
            }
        }
    } else { // 垂直镜像
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

    // 获取变换矩阵元素
    float m00 = M.at<float>(0,0);
    float m01 = M.at<float>(0,1);
    float m02 = M.at<float>(0,2);
    float m10 = M.at<float>(1,0);
    float m11 = M.at<float>(1,1);
    float m12 = M.at<float>(1,2);

    #pragma omp parallel for collapse(2)
    for(int y = 0; y < dst.rows; y++) {
        for(int x = 0; x < dst.cols; x++) {
            // 计算原图像坐标
            float src_x = m00 * x + m01 * y + m02;
            float src_y = m10 * x + m11 * y + m12;

            // 边界检查
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

    // 获取变换矩阵元素
    float m00 = M.at<float>(0,0);
    float m01 = M.at<float>(0,1);
    float m02 = M.at<float>(0,2);
    float m10 = M.at<float>(1,0);
    float m11 = M.at<float>(1,1);
    float m12 = M.at<float>(1,2);
    float m20 = M.at<float>(2,0);
    float m21 = M.at<float>(2,1);
    float m22 = M.at<float>(2,2);

    #pragma omp parallel for collapse(2)
    for(int y = 0; y < dst.rows; y++) {
        for(int x = 0; x < dst.cols; x++) {
            // 计算原图像坐标
            float denominator = m20 * x + m21 * y + m22;
            float src_x = (m00 * x + m01 * y + m02) / denominator;
            float src_y = (m10 * x + m11 * y + m12) / denominator;

            // 边界检查
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
    // 计算旋转中心
    Point2f center_point = center;
    if(center.x < 0 || center.y < 0) {
        center_point = Point2f(src.cols/2.0f, src.rows/2.0f);
    }

    // 计算旋转矩阵
    Mat M = getRotationMatrix2D(center_point, angle, scale);

    // 计算旋转后的图像大小
    double alpha = angle * CV_PI / 180.0;
    double cos_alpha = fabs(cos(alpha));
    double sin_alpha = fabs(sin(alpha));

    int new_w = static_cast<int>(src.cols * cos_alpha + src.rows * sin_alpha);
    int new_h = static_cast<int>(src.cols * sin_alpha + src.rows * cos_alpha);

    // 调整旋转中心
    M.at<double>(0,2) += (new_w/2.0 - center_point.x);
    M.at<double>(1,2) += (new_h/2.0 - center_point.y);

    return affine_transform(src, M, Size(new_w, new_h));
}

Mat resize(const Mat& src, const Size& size, int interpolation) {
    Mat dst(size, src.type());

    float scale_x = static_cast<float>(src.cols) / size.width;
    float scale_y = static_cast<float>(src.rows) / size.height;

    #pragma omp parallel for collapse(2)
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

    if(flip_code == 0) { // 垂直翻转
        #pragma omp parallel for
        for(int y = 0; y < src.rows; y++) {
            for(int x = 0; x < src.cols; x++) {
                dst.at<Vec3b>(y,x) = src.at<Vec3b>(src.rows-1-y,x);
            }
        }
    }
    else if(flip_code > 0) { // 水平翻转
        #pragma omp parallel for
        for(int y = 0; y < src.rows; y++) {
            for(int x = 0; x < src.cols; x++) {
                dst.at<Vec3b>(y,x) = src.at<Vec3b>(y,src.cols-1-x);
            }
        }
    }
    else { // 双向翻转
        #pragma omp parallel for
        for(int y = 0; y < src.rows; y++) {
            for(int x = 0; x < src.cols; x++) {
                dst.at<Vec3b>(y,x) = src.at<Vec3b>(src.rows-1-y,src.cols-1-x);
            }
        }
    }

    return dst;
}

} // namespace ip101

// 性能测试函数
void performanceTest(const Mat& img) {
    const int REPEAT_COUNT = 100;
    vector<double> times_custom(REPEAT_COUNT);
    vector<double> times_opencv(REPEAT_COUNT);

    // 测试仿射变换
    vector<Point2f> src_points = {Point2f(0,0), Point2f(1,0), Point2f(0,1)};
    vector<Point2f> dst_points = {Point2f(0,0), Point2f(1,0), Point2f(0,1)};

    for(int i = 0; i < REPEAT_COUNT; i++) {
        auto start = high_resolution_clock::now();
        Mat result = ip101::affineTransform(img, src_points, dst_points);
        auto end = high_resolution_clock::now();
        times_custom[i] = duration_cast<microseconds>(end - start).count();

        start = high_resolution_clock::now();
        warpAffine(img, result, ip101::getAffineTransform(src_points, dst_points), img.size());
        end = high_resolution_clock::now();
        times_opencv[i] = duration_cast<microseconds>(end - start).count();
    }

    // 计算平均时间
    double avg_custom = accumulate(times_custom.begin(), times_custom.end(), 0.0) / REPEAT_COUNT;
    double avg_opencv = accumulate(times_opencv.begin(), times_opencv.end(), 0.0) / REPEAT_COUNT;

    cout << "仿射变换性能测试结果：" << endl;
    cout << "自定义实现平均时间: " << avg_custom << " 微秒" << endl;
    cout << "OpenCV实现平均时间: " << avg_opencv << " 微秒" << endl;
    cout << "性能提升: " << (avg_opencv - avg_custom) / avg_opencv * 100 << "%" << endl;
}

int main(int argc, char** argv) {
    if(argc != 2) {
        cout << "Usage: " << argv[0] << " <image_path>" << endl;
        return -1;
    }

    Mat img = imread(argv[1]);
    if(img.empty()) {
        cout << "无法读取图像: " << argv[1] << endl;
        return -1;
    }

    // 运行性能测试
    performanceTest(img);

    return 0;
}