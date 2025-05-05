#include "image_pyramid.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;
using namespace ip101;

// 性能测试函数
void test_performance(const Mat& src, const string& method_name) {
    auto start = chrono::high_resolution_clock::now();

    if (method_name == "Gaussian Pyramid") {
        vector<Mat> pyramid = build_gaussian_pyramid(src, 4);
        Mat vis = visualize_pyramid(pyramid);
        imshow("高斯金字塔", vis);
    } else if (method_name == "Laplacian Pyramid") {
        vector<Mat> pyramid = build_laplacian_pyramid(src, 4);
        Mat vis = visualize_pyramid(pyramid);
        imshow("拉普拉斯金字塔", vis);
    } else if (method_name == "SIFT Scale Space") {
        vector<vector<Mat>> scale_space = build_sift_scale_space(src);
        // 显示每个组的所有尺度
        for (size_t o = 0; o < scale_space.size(); o++) {
            Mat vis = visualize_pyramid(scale_space[o]);
            imshow("SIFT尺度空间 - 组" + to_string(o), vis);
        }
    } else if (method_name == "Saliency") {
        Mat saliency = saliency_detection(src);
        imshow("显著性检测结果", saliency);
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
    vector<Mat> pyramid_ours;

    if (method_name == "Gaussian Pyramid") {
        pyramid_ours = build_gaussian_pyramid(src, 4);
    } else if (method_name == "Laplacian Pyramid") {
        pyramid_ours = build_laplacian_pyramid(src, 4);
    } else if (method_name == "Saliency") {
        result_ours = saliency_detection(src);
    }

    auto end_ours = chrono::high_resolution_clock::now();
    auto duration_ours = chrono::duration_cast<chrono::milliseconds>(end_ours - start_ours);

    // 测试OpenCV实现
    auto start_opencv = chrono::high_resolution_clock::now();
    Mat result_opencv;
    vector<Mat> pyramid_opencv;

    if (method_name == "Gaussian Pyramid") {
        Mat current = src.clone();
        pyramid_opencv.push_back(current);
        for (int i = 0; i < 3; i++) {
            Mat next;
            pyrDown(current, next);
            pyramid_opencv.push_back(next);
            current = next;
        }
    } else if (method_name == "Laplacian Pyramid") {
        Mat current = src.clone();
        vector<Mat> gauss_pyr;
        gauss_pyr.push_back(current);
        for (int i = 0; i < 3; i++) {
            Mat next;
            pyrDown(current, next);
            gauss_pyr.push_back(next);
            current = next;
        }

        pyramid_opencv.resize(4);
        for (int i = 0; i < 3; i++) {
            Mat up;
            pyrUp(gauss_pyr[i + 1], up, gauss_pyr[i].size());
            subtract(gauss_pyr[i], up, pyramid_opencv[i]);
        }
        pyramid_opencv[3] = gauss_pyr[3];
    } else if (method_name == "Saliency") {
        Ptr<saliency::StaticSaliencySpectralResidual> saliency =
            saliency::StaticSaliencySpectralResidual::create();
        saliency->computeSaliency(src, result_opencv);
    }

    auto end_opencv = chrono::high_resolution_clock::now();
    auto duration_opencv = chrono::duration_cast<chrono::milliseconds>(end_opencv - start_opencv);

    cout << method_name << " 性能对比:" << endl;
    cout << "我们的实现: " << duration_ours.count() << "ms" << endl;
    cout << "OpenCV实现: " << duration_opencv.count() << "ms" << endl;
    cout << "加速比: " << (float)duration_opencv.count() / duration_ours.count() << "x" << endl;

    // 显示对比结果
    if (method_name == "Gaussian Pyramid" || method_name == "Laplacian Pyramid") {
        Mat vis_ours = visualize_pyramid(pyramid_ours);
        Mat vis_opencv = visualize_pyramid(pyramid_opencv);
        Mat comparison;
        vconcat(vector<Mat>{vis_ours, vis_opencv}, comparison);
        imshow(method_name + " 对比 (上: 我们的实现, 下: OpenCV实现)", comparison);
    } else if (method_name == "Saliency") {
        Mat comparison;
        hconcat(vector<Mat>{result_ours, result_opencv}, comparison);
        imshow(method_name + " 对比 (左: 我们的实现, 右: OpenCV实现)", comparison);
    }

    waitKey(0);
}

// 测试图像融合
void test_image_blend(const Mat& src1, const Mat& src2, const Mat& mask) {
    cout << "\n测试图像融合:" << endl;

    auto start = chrono::high_resolution_clock::now();
    Mat result = pyramid_blend(src1, src2, mask, 4);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

    cout << "融合耗时: " << duration.count() << "ms" << endl;

    // 显示结果
    Mat vis;
    hconcat(vector<Mat>{src1, src2, result}, vis);
    imshow("图像融合 (左: 图像1, 中: 图像2, 右: 融合结果)", vis);
    waitKey(0);
}

int main() {
    // 读取测试图像
    Mat src = imread("test_images/lena.jpg", IMREAD_GRAYSCALE);
    if (src.empty()) {
        cerr << "无法读取测试图像" << endl;
        return -1;
    }

    // 测试各种方法
    vector<string> methods = {
        "Gaussian Pyramid",
        "Laplacian Pyramid",
        "SIFT Scale Space",
        "Saliency"
    };

    for (const auto& method : methods) {
        cout << "\n测试 " << method << ":" << endl;
        test_performance(src, method);
        if (method != "SIFT Scale Space") {
            compare_with_opencv(src, method);
        }
    }

    // 测试图像融合
    Mat src1 = imread("test_images/apple.jpg");
    Mat src2 = imread("test_images/orange.jpg");
    Mat mask = Mat::zeros(src1.size(), CV_32F);
    mask(Rect(0, 0, mask.cols/2, mask.rows)) = 1.0;

    if (!src1.empty() && !src2.empty()) {
        test_image_blend(src1, src2, mask);
    }

    return 0;
}