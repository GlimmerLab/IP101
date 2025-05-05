#include "morphology.hpp"
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
    Mat src = imread("test.jpg", IMREAD_GRAYSCALE);
    if (src.empty()) {
        cerr << "无法读取测试图像" << endl;
        return -1;
    }

    // 创建不同形状的结构元素
    vector<pair<int, string>> kernel_types = {
        {MORPH_RECT, "矩形"},
        {MORPH_CROSS, "十字形"},
        {MORPH_ELLIPSE, "椭圆形"}
    };

    for (const auto& kernel_type : kernel_types) {
        cout << "\n=== 测试" << kernel_type.second << "结构元素 ===" << endl;
        Mat kernel = create_kernel(kernel_type.first, Size(5, 5));
        Mat opencv_kernel = getStructuringElement(kernel_type.first, Size(5, 5));

        // 1. 测试膨胀操作
        cout << "\n--- 膨胀操作 ---" << endl;
        Mat dilate_result, opencv_dilate;

        double manual_time = measure_time([&]() {
            dilate_manual(src, dilate_result, kernel);
        }, "手动实现的膨胀");

        double opencv_time = measure_time([&]() {
            dilate(src, opencv_dilate, opencv_kernel);
        }, "OpenCV膨胀");

        cout << "性能比: " << opencv_time / manual_time << endl;
        imwrite("dilate_manual_" + kernel_type.second + ".jpg", dilate_result);
        imwrite("dilate_opencv_" + kernel_type.second + ".jpg", opencv_dilate);

        // 2. 测试腐蚀操作
        cout << "\n--- 腐蚀操作 ---" << endl;
        Mat erode_result, opencv_erode;

        manual_time = measure_time([&]() {
            erode_manual(src, erode_result, kernel);
        }, "手动实现的腐蚀");

        opencv_time = measure_time([&]() {
            erode(src, opencv_erode, opencv_kernel);
        }, "OpenCV腐蚀");

        cout << "性能比: " << opencv_time / manual_time << endl;
        imwrite("erode_manual_" + kernel_type.second + ".jpg", erode_result);
        imwrite("erode_opencv_" + kernel_type.second + ".jpg", opencv_erode);

        // 3. 测试开运算
        cout << "\n--- 开运算 ---" << endl;
        Mat opening_result, opencv_opening;

        manual_time = measure_time([&]() {
            opening_manual(src, opening_result, kernel);
        }, "手动实现的开运算");

        opencv_time = measure_time([&]() {
            morphologyEx(src, opencv_opening, MORPH_OPEN, opencv_kernel);
        }, "OpenCV开运算");

        cout << "性能比: " << opencv_time / manual_time << endl;
        imwrite("opening_manual_" + kernel_type.second + ".jpg", opening_result);
        imwrite("opening_opencv_" + kernel_type.second + ".jpg", opencv_opening);

        // 4. 测试闭运算
        cout << "\n--- 闭运算 ---" << endl;
        Mat closing_result, opencv_closing;

        manual_time = measure_time([&]() {
            closing_manual(src, closing_result, kernel);
        }, "手动实现的闭运算");

        opencv_time = measure_time([&]() {
            morphologyEx(src, opencv_closing, MORPH_CLOSE, opencv_kernel);
        }, "OpenCV闭运算");

        cout << "性能比: " << opencv_time / manual_time << endl;
        imwrite("closing_manual_" + kernel_type.second + ".jpg", closing_result);
        imwrite("closing_opencv_" + kernel_type.second + ".jpg", opencv_closing);

        // 5. 测试形态学梯度
        cout << "\n--- 形态学梯度 ---" << endl;
        Mat gradient_result, opencv_gradient;

        manual_time = measure_time([&]() {
            morphological_gradient_manual(src, gradient_result, kernel);
        }, "手动实现的形态学梯度");

        opencv_time = measure_time([&]() {
            morphologyEx(src, opencv_gradient, MORPH_GRADIENT, opencv_kernel);
        }, "OpenCV形态学梯度");

        cout << "性能比: " << opencv_time / manual_time << endl;
        imwrite("gradient_manual_" + kernel_type.second + ".jpg", gradient_result);
        imwrite("gradient_opencv_" + kernel_type.second + ".jpg", opencv_gradient);
    }

    return 0;
}