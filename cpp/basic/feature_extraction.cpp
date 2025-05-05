#include "feature_extraction.hpp"
#include <cmath>
#include <vector>

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

// 高斯核生成
void createGaussianKernel(Mat& kernel, int ksize, double sigma) {
    kernel.create(ksize, ksize, CV_64F);
    double sum = 0;
    int center = ksize / 2;

    for (int y = 0; y < ksize; y++) {
        for (int x = 0; x < ksize; x++) {
            double x2 = (x - center) * (x - center);
            double y2 = (y - center) * (y - center);
            double value = exp(-(x2 + y2) / (2 * sigma * sigma));
            kernel.at<double>(y, x) = value;
            sum += value;
        }
    }

    // 归一化
    kernel /= sum;
}

} // anonymous namespace

void compute_harris_manual(const Mat& src, Mat& dst,
                          double k, int window_size,
                          double threshold) {
    CV_Assert(!src.empty() && src.type() == CV_8UC1);

    // 计算图像梯度
    Mat Ix, Iy;
    Sobel(src, Ix, CV_64F, 1, 0, 3);
    Sobel(src, Iy, CV_64F, 0, 1, 3);

    // 计算梯度的乘积
    Mat Ixx, Ixy, Iyy;
    Ixx = Ix.mul(Ix);
    Ixy = Ix.mul(Iy);
    Iyy = Iy.mul(Iy);

    // 创建高斯核
    Mat gaussian_kernel;
    createGaussianKernel(gaussian_kernel, window_size, 1.0);

    // 对梯度乘积进行高斯滤波
    Mat Sxx, Sxy, Syy;
    filter2D(Ixx, Sxx, -1, gaussian_kernel);
    filter2D(Ixy, Sxy, -1, gaussian_kernel);
    filter2D(Iyy, Syy, -1, gaussian_kernel);

    // 计算Harris响应
    Mat det = Sxx.mul(Syy) - Sxy.mul(Sxy);
    Mat trace = Sxx + Syy;
    Mat harris_response = det - k * trace.mul(trace);

    // 阈值处理
    double max_val;
    minMaxLoc(harris_response, nullptr, &max_val);
    threshold *= max_val;

    // 创建输出图像
    dst = Mat::zeros(src.size(), CV_8UC1);
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            if (harris_response.at<double>(y, x) > threshold) {
                dst.at<uchar>(y, x) = 255;
            }
        }
    }
}

void harris_corner_detection(const Mat& src, Mat& dst,
                           int block_size, int ksize,
                           double k, double threshold) {
    CV_Assert(!src.empty());

    // 转换为灰度图
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // 使用OpenCV的Harris角点检测
    Mat corners;
    cornerHarris(gray, corners, block_size, ksize, k);

    // 阈值处理
    Mat corners_norm;
    normalize(corners, corners_norm, 0, 255, NORM_MINMAX, CV_8UC1);

    // 在原图上标记角点
    dst = src.clone();
    for (int y = 0; y < corners_norm.rows; y++) {
        for (int x = 0; x < corners_norm.cols; x++) {
            if (corners_norm.at<uchar>(y, x) > threshold) {
                circle(dst, Point(x, y), 5, Scalar(0, 0, 255), 2);
            }
        }
    }
}

void sift_features(const Mat& src, Mat& dst, int nfeatures) {
    CV_Assert(!src.empty());

    // 转换为灰度图
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // 创建SIFT对象
    Ptr<SIFT> sift = SIFT::create(nfeatures);

    // 检测关键点和描述子
    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    sift->detectAndCompute(gray, Mat(), keypoints, descriptors);

    // 在原图上绘制关键点
    drawKeypoints(src, keypoints, dst, Scalar(0, 255, 0),
                 DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}

void surf_features(const Mat& src, Mat& dst, double hessian_threshold) {
    CV_Assert(!src.empty());

    // 转换为灰度图
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // 创建SURF对象
    Ptr<SURF> surf = SURF::create(hessian_threshold);

    // 检测关键点和描述子
    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    surf->detectAndCompute(gray, Mat(), keypoints, descriptors);

    // 在原图上绘制关键点
    drawKeypoints(src, keypoints, dst, Scalar(0, 255, 0),
                 DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}

void orb_features(const Mat& src, Mat& dst, int nfeatures) {
    CV_Assert(!src.empty());

    // 转换为灰度图
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // 创建ORB对象
    Ptr<ORB> orb = ORB::create(nfeatures);

    // 检测关键点和描述子
    std::vector<KeyPoint> keypoints;
    Mat descriptors;
    orb->detectAndCompute(gray, Mat(), keypoints, descriptors);

    // 在原图上绘制关键点
    drawKeypoints(src, keypoints, dst, Scalar(0, 255, 0),
                 DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}

void feature_matching(const Mat& src1, const Mat& src2,
                     Mat& dst, const std::string& method) {
    CV_Assert(!src1.empty() && !src2.empty());

    // 转换为灰度图
    Mat gray1, gray2;
    if (src1.channels() == 3) {
        cvtColor(src1, gray1, COLOR_BGR2GRAY);
    } else {
        gray1 = src1.clone();
    }
    if (src2.channels() == 3) {
        cvtColor(src2, gray2, COLOR_BGR2GRAY);
    } else {
        gray2 = src2.clone();
    }

    // 创建特征检测器
    Ptr<Feature2D> detector;
    if (method == "sift") {
        detector = SIFT::create();
    } else if (method == "surf") {
        detector = SURF::create();
    } else if (method == "orb") {
        detector = ORB::create();
    } else {
        throw std::invalid_argument("Unsupported feature detection method: " + method);
    }

    // 检测关键点和描述子
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute(gray1, Mat(), keypoints1, descriptors1);
    detector->detectAndCompute(gray2, Mat(), keypoints2, descriptors2);

    // 创建特征匹配器
    Ptr<DescriptorMatcher> matcher;
    if (method == "sift" || method == "surf") {
        matcher = BFMatcher::create(NORM_L2);
    } else {
        matcher = BFMatcher::create(NORM_HAMMING);
    }

    // 进行特征匹配
    std::vector<DMatch> matches;
    matcher->match(descriptors1, descriptors2, matches);

    // 绘制匹配结果
    drawMatches(src1, keypoints1, src2, keypoints2, matches, dst,
               Scalar::all(-1), Scalar::all(-1),
               std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
}

} // namespace ip101