#include "image_segmentation.hpp"
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

    // 1. 测试阈值分割
    cout << "\n=== 阈值分割测试 ===" << endl;
    Mat threshold_result, opencv_threshold;

    double manual_time = measure_time([&]() {
        threshold_segmentation(src, threshold_result, 128);
    }, "手动实现的阈值分割");

    double opencv_time = measure_time([&]() {
        threshold(src, opencv_threshold, 128, 255, THRESH_BINARY);
    }, "OpenCV阈值分割");

    cout << "性能比: " << opencv_time / manual_time << endl;
    imwrite("threshold_manual.jpg", threshold_result);
    imwrite("threshold_opencv.jpg", opencv_threshold);

    // 2. 测试K均值分割
    cout << "\n=== K均值分割测试 ===" << endl;
    Mat kmeans_result, opencv_kmeans;

    manual_time = measure_time([&]() {
        kmeans_segmentation(src, kmeans_result, 3);
    }, "手动实现的K均值分割");

    opencv_time = measure_time([&]() {
        Mat data = src.reshape(1, src.rows * src.cols);
        data.convertTo(data, CV_32F);
        Mat labels, centers;
        kmeans(data, 3, labels,
               TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 100, 0.1),
               3, KMEANS_PP_CENTERS, centers);
        centers = centers.reshape(3, centers.rows);
        Mat result(src.size(), src.type());
        for (int i = 0; i < src.rows * src.cols; i++) {
            int y = i / src.cols;
            int x = i % src.cols;
            int label = labels.at<int>(i);
            result.at<Vec3b>(y, x) = centers.at<Vec3b>(label);
        }
        opencv_kmeans = result;
    }, "OpenCV K均值分割");

    cout << "性能比: " << opencv_time / manual_time << endl;
    imwrite("kmeans_manual.jpg", kmeans_result);
    imwrite("kmeans_opencv.jpg", opencv_kmeans);

    // 3. 测试区域生长
    cout << "\n=== 区域生长测试 ===" << endl;
    Mat region_result;
    vector<Point> seeds = {Point(src.cols/2, src.rows/2)};

    manual_time = measure_time([&]() {
        region_growing(src, region_result, seeds);
    }, "区域生长分割");

    imwrite("region_growing.jpg", region_result);

    // 4. 测试分水岭分割
    cout << "\n=== 分水岭分割测试 ===" << endl;
    Mat markers = Mat::zeros(src.size(), CV_8UC1);
    rectangle(markers, Point(src.cols/4, src.rows/4),
             Point(src.cols*3/4, src.rows*3/4), Scalar(255), -1);
    Mat watershed_result;

    manual_time = measure_time([&]() {
        watershed_segmentation(src, markers, watershed_result);
    }, "分水岭分割");

    imwrite("watershed.jpg", watershed_result);

    // 5. 测试图割分割
    cout << "\n=== 图割分割测试 ===" << endl;
    Mat graphcut_result;
    Rect rect(src.cols/4, src.rows/4, src.cols/2, src.rows/2);

    manual_time = measure_time([&]() {
        graph_cut_segmentation(src, graphcut_result, rect);
    }, "图割分割");

    imwrite("graphcut.jpg", graphcut_result);

    return 0;
}