#include "image_transform.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;
using namespace ip101;

// 性能测试函数
void test_performance(const Mat& src, const string& method_name) {
    auto start = chrono::high_resolution_clock::now();

    if (method_name == "Affine Transform") {
        Mat M = (Mat_<float>(2,3) << 1, 0.5, 100, 0.5, 1, 50);
        Mat result = affine_transform(src, M, src.size());
        imshow("Affine Transform", result);
    }
    else if (method_name == "Perspective Transform") {
        Mat M = (Mat_<float>(3,3) << 1, 0.2, 0, 0.2, 1, 0, 0.001, 0.001, 1);
        Mat result = perspective_transform(src, M, src.size());
        imshow("Perspective Transform", result);
    }
    else if (method_name == "Rotation") {
        Mat result = rotate(src, 45);
        imshow("Rotation", result);
    }
    else if (method_name == "Resize") {
        Mat result = resize(src, Size(src.cols/2, src.rows/2));
        imshow("Resize", result);
    }
    else if (method_name == "Translation") {
        Mat result = translate(src, 100, 50);
        imshow("Translation", result);
    }
    else if (method_name == "Mirror") {
        Mat result = mirror(src, 1);
        imshow("Mirror", result);
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << method_name << " 处理耗时: " << duration.count() << "ms" << endl;

    waitKey(0);
}

// 与OpenCV实现对比
void compare_with_opencv(const Mat& src, const string& method_name) {
    // 测试我们的实现
    auto start_ours = chrono::high_resolution_clock::now();
    Mat result_ours;

    if (method_name == "Affine Transform") {
        Mat M = (Mat_<float>(2,3) << 1, 0.5, 100, 0.5, 1, 50);
        result_ours = affine_transform(src, M, src.size());
    }
    else if (method_name == "Perspective Transform") {
        Mat M = (Mat_<float>(3,3) << 1, 0.2, 0, 0.2, 1, 0, 0.001, 0.001, 1);
        result_ours = perspective_transform(src, M, src.size());
    }
    else if (method_name == "Rotation") {
        result_ours = rotate(src, 45);
    }
    else if (method_name == "Resize") {
        result_ours = resize(src, Size(src.cols/2, src.rows/2));
    }
    else if (method_name == "Translation") {
        result_ours = translate(src, 100, 50);
    }
    else if (method_name == "Mirror") {
        result_ours = mirror(src, 1);
    }

    auto end_ours = chrono::high_resolution_clock::now();
    auto duration_ours = chrono::duration_cast<chrono::milliseconds>(end_ours - start_ours);

    // 测试OpenCV实现
    auto start_opencv = chrono::high_resolution_clock::now();
    Mat result_opencv;

    if (method_name == "Affine Transform") {
        Mat M = (Mat_<float>(2,3) << 1, 0.5, 100, 0.5, 1, 50);
        warpAffine(src, result_opencv, M, src.size());
    }
    else if (method_name == "Perspective Transform") {
        Mat M = (Mat_<float>(3,3) << 1, 0.2, 0, 0.2, 1, 0, 0.001, 0.001, 1);
        warpPerspective(src, result_opencv, M, src.size());
    }
    else if (method_name == "Rotation") {
        Mat M = getRotationMatrix2D(Point2f(src.cols/2, src.rows/2), 45, 1.0);
        warpAffine(src, result_opencv, M, src.size());
    }
    else if (method_name == "Resize") {
        resize(src, result_opencv, Size(src.cols/2, src.rows/2));
    }
    else if (method_name == "Translation") {
        Mat M = (Mat_<float>(2,3) << 1, 0, 100, 0, 1, 50);
        warpAffine(src, result_opencv, M, src.size());
    }
    else if (method_name == "Mirror") {
        flip(src, result_opencv, 1);
    }

    auto end_opencv = chrono::high_resolution_clock::now();
    auto duration_opencv = chrono::duration_cast<chrono::milliseconds>(end_opencv - start_opencv);

    cout << method_name << " 性能对比:" << endl;
    cout << "我们的实现: " << duration_ours.count() << "ms" << endl;
    cout << "OpenCV实现: " << duration_opencv.count() << "ms" << endl;
    cout << "加速比: " << (float)duration_opencv.count() / duration_ours.count() << "x" << endl;

    // 显示对比结果
    Mat comparison;
    hconcat(vector<Mat>{result_ours, result_opencv}, comparison);
    imshow(method_name + " 对比 (左: 我们的实现, 右: OpenCV实现)", comparison);
    waitKey(0);
}

int main() {
    // 读取测试图像
    Mat src = imread("test_images/lena.jpg");
    if (src.empty()) {
        cerr << "无法读取测试图像" << endl;
        return -1;
    }

    // 测试各种方法
    vector<string> methods = {
        "Affine Transform",
        "Perspective Transform",
        "Rotation",
        "Resize",
        "Translation",
        "Mirror"
    };

    for (const auto& method : methods) {
        cout << "\n测试 " << method << ":" << endl;
        test_performance(src, method);
        compare_with_opencv(src, method);
    }

    return 0;
}