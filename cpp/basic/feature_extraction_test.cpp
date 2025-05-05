#include "feature_extraction.hpp"
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace ip101;

// 性能测试函数
template<typename Func>
double measure_time(Func func, const string& name) {
    auto start = chrono::high_resolution_clock::now();
    func();
    auto end = chrono::high_resolution_clock::now();
    double time = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;
    cout << name << " 耗时: " << time << " ms" << endl;
    return time;
}

int main() {
    // 读取测试图像
    Mat src = imread("test.jpg");
    if (src.empty()) {
        cerr << "无法读取测试图像" << endl;
        return -1;
    }

    // 测试Harris角点检测
    Mat harris_result;
    measure_time([&]() {
        harris_corner_detection(src, harris_result, 2, 3, 0.04, 0.01);
    }, "Harris角点检测");
    imwrite("harris_result.jpg", harris_result);

    // 测试手动实现的Harris角点检测
    Mat harris_manual_result;
    measure_time([&]() {
        compute_harris_manual(src, harris_manual_result, 0.04, 3, 0.01);
    }, "手动Harris角点检测");
    imwrite("harris_manual_result.jpg", harris_manual_result);

    // 测试SIFT特征提取
    Mat sift_result;
    measure_time([&]() {
        sift_features(src, sift_result, 1000);
    }, "SIFT特征提取");
    imwrite("sift_result.jpg", sift_result);

    // 测试SURF特征提取
    Mat surf_result;
    measure_time([&]() {
        surf_features(src, surf_result, 100);
    }, "SURF特征提取");
    imwrite("surf_result.jpg", surf_result);

    // 测试ORB特征提取
    Mat orb_result;
    measure_time([&]() {
        orb_features(src, orb_result, 1000);
    }, "ORB特征提取");
    imwrite("orb_result.jpg", orb_result);

    // 测试特征匹配
    Mat src2 = imread("test2.jpg");
    if (src2.empty()) {
        cerr << "无法读取第二张测试图像" << endl;
        return -1;
    }

    Mat match_result;
    measure_time([&]() {
        feature_matching(src, src2, match_result, "sift");
    }, "SIFT特征匹配");
    imwrite("match_result.jpg", match_result);

    return 0;
}