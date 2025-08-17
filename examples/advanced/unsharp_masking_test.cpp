#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include "advanced/effects/unsharp_masking.hpp"

using namespace cv;
using namespace std;

// 创建输出目录
void create_output_directories() {
    filesystem::create_directories("output/unsharp_masking");
}

// 性能测试
void performance_test(const Mat& src) {
    cout << "\n--- 性能测试 ---" << endl;

    Mat dst;
    int iterations = 10;

    // 测试IP101算法性能
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        ip101::advanced::UnsharpMaskingParams params;
        params.strength = 1.5;
        params.radius = 0.5;
        params.threshold = 0;
        ip101::advanced::unsharp_masking(src, dst, params);
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    double avg_time = duration.count() / (double)iterations;
    double fps = 1000000.0 / avg_time;

    cout << "IP101非锐化掩膜 - 平均时间: " << fixed << setprecision(2)
         << avg_time / 1000.0 << " ms, FPS: " << fps << endl;

    // 测试OpenCV对比算法（使用拉普拉斯算子作为对比）
    Mat opencv_result;
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        Mat gray, laplacian;
        cvtColor(src, gray, COLOR_BGR2GRAY);
        Laplacian(gray, laplacian, CV_8U);
        cvtColor(laplacian, opencv_result, COLOR_GRAY2BGR);
    }
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);

    avg_time = duration.count() / (double)iterations;
    fps = 1000000.0 / avg_time;

    cout << "OpenCV拉普拉斯算子 - 平均时间: " << fixed << setprecision(2)
         << avg_time / 1000.0 << " ms, FPS: " << fps << endl;
}

// 参数效果测试
void parameter_effect_test(const Mat& src) {
    cout << "\n--- 参数效果测试 ---" << endl;

    vector<double> amount_values = {0.5, 1.0, 1.5, 2.0, 2.5};
    vector<double> radius_values = {0.3, 0.5, 0.7, 1.0, 1.5};
    vector<double> threshold_values = {0, 10, 20, 30, 50};

    // 测试强度参数
    cout << "测试强度参数效果..." << endl;
    for (size_t i = 0; i < amount_values.size(); i++) {
        Mat result;
        ip101::advanced::UnsharpMaskingParams params;
        params.strength = amount_values[i];
        params.radius = 0.5;
        params.threshold = 0;
        ip101::advanced::unsharp_masking(src, result, params);
        string filename = "output/unsharp_masking/amount_" + to_string(i) + ".jpg";
        imwrite(filename, result);
    }

    // 测试半径参数
    cout << "测试半径参数效果..." << endl;
    for (size_t i = 0; i < radius_values.size(); i++) {
        Mat result;
        ip101::advanced::UnsharpMaskingParams params;
        params.strength = 1.5;
        params.radius = radius_values[i];
        params.threshold = 0;
        ip101::advanced::unsharp_masking(src, result, params);
        string filename = "output/unsharp_masking/radius_" + to_string(i) + ".jpg";
        imwrite(filename, result);
    }

    // 测试阈值参数
    cout << "测试阈值参数效果..." << endl;
    for (size_t i = 0; i < threshold_values.size(); i++) {
        Mat result;
        ip101::advanced::UnsharpMaskingParams params;
        params.strength = 1.5;
        params.radius = 0.5;
        params.threshold = threshold_values[i];
        ip101::advanced::unsharp_masking(src, result, params);
        string filename = "output/unsharp_masking/threshold_" + to_string(i) + ".jpg";
        imwrite(filename, result);
    }
}

// 可视化测试
void visualization_test(const Mat& src) {
    cout << "\n--- 可视化测试 ---" << endl;

    // 应用非锐化掩膜
    Mat dst_unsharp;
    ip101::advanced::UnsharpMaskingParams params;
    params.strength = 1.5;
    params.radius = 0.5;
    params.threshold = 0;
    ip101::advanced::unsharp_masking(src, dst_unsharp, params);

    // 保存结果
    imwrite("output/unsharp_masking/original.jpg", src);
    imwrite("output/unsharp_masking/unsharp_masked.jpg", dst_unsharp);

    // 创建OpenCV对比结果
    Mat opencv_result;
    Mat gray, laplacian;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    Laplacian(gray, laplacian, CV_8U);
    cvtColor(laplacian, opencv_result, COLOR_GRAY2BGR);
    imwrite("output/unsharp_masking/opencv_comparison.jpg", opencv_result);

    // 创建对比图像
    vector<Mat> images = {src, dst_unsharp, opencv_result};
    Mat comparison;
    hconcat(images, comparison);

    // 添加标题
    vector<string> titles = {"Original", "Unsharp Masking", "OpenCV Comparison"};
    int font_face = FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.8;
    Scalar color(255, 255, 255);
    int thickness = 2;

    for (size_t i = 0; i < titles.size(); i++) {
        int x = i * src.cols + 10;
        putText(comparison, titles[i], Point(x, 30), font_face, font_scale, color, thickness);
    }

    imwrite("output/unsharp_masking/comparison.jpg", comparison);
    cout << "Comparison image saved to: output/unsharp_masking/comparison.jpg" << endl;
}

// 特殊场景测试
void special_scenario_test(const Mat& src) {
    cout << "\n--- 特殊场景测试 ---" << endl;

    // 创建模糊图像
    Mat blurred = src.clone();
    GaussianBlur(blurred, blurred, Size(5, 5), 1.0);
    imwrite("output/unsharp_masking/blurred.jpg", blurred);

    // 应用非锐化掩膜
    Mat blurred_result;
    ip101::advanced::UnsharpMaskingParams params;
    params.strength = 1.5;
    params.radius = 0.5;
    params.threshold = 0;
    ip101::advanced::unsharp_masking(blurred, blurred_result, params);
    imwrite("output/unsharp_masking/blurred_result.jpg", blurred_result);

    // 创建噪声图像
    Mat noisy = src.clone();
    Mat noise = Mat::zeros(noisy.size(), CV_8UC3);
    randn(noise, 0, 15);
    noisy = noisy + noise;
    imwrite("output/unsharp_masking/noisy.jpg", noisy);

    // 应用非锐化掩膜
    Mat noisy_result;
    ip101::advanced::UnsharpMaskingParams params2;
    params2.strength = 1.5;
    params2.radius = 0.5;
    params2.threshold = 0;
    ip101::advanced::unsharp_masking(noisy, noisy_result, params2);
    imwrite("output/unsharp_masking/noisy_result.jpg", noisy_result);

    // 创建低对比度图像
    Mat low_contrast = src.clone();
    low_contrast.convertTo(low_contrast, -1, 0.5, 50);
    imwrite("output/unsharp_masking/low_contrast.jpg", low_contrast);

    // 应用非锐化掩膜
    Mat low_contrast_result;
    ip101::advanced::UnsharpMaskingParams params3;
    params3.strength = 1.5;
    params3.radius = 0.5;
    params3.threshold = 0;
    ip101::advanced::unsharp_masking(low_contrast, low_contrast_result, params3);
    imwrite("output/unsharp_masking/low_contrast_result.jpg", low_contrast_result);
}

// 质量评估
void quality_assessment(const Mat& src) {
    cout << "\n--- 质量评估 ---" << endl;

    // 应用非锐化掩膜
    Mat dst_unsharp;
    ip101::advanced::UnsharpMaskingParams params4;
    params4.strength = 1.5;
    params4.radius = 0.5;
    params4.threshold = 0;
    ip101::advanced::unsharp_masking(src, dst_unsharp, params4);

    // 创建OpenCV对比结果
    Mat opencv_result;
    Mat gray, laplacian;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    Laplacian(gray, laplacian, CV_8U);
    cvtColor(laplacian, opencv_result, COLOR_GRAY2BGR);

    // 计算锐化效果
    Mat gray_src, gray_result, gray_opencv;
    cvtColor(src, gray_src, COLOR_BGR2GRAY);
    cvtColor(dst_unsharp, gray_result, COLOR_BGR2GRAY);
    cvtColor(opencv_result, gray_opencv, COLOR_BGR2GRAY);

    // 计算拉普拉斯算子响应（锐度度量）
    Mat laplacian_src, laplacian_result, laplacian_opencv;
    Laplacian(gray_src, laplacian_src, CV_64F);
    Laplacian(gray_result, laplacian_result, CV_64F);
    Laplacian(gray_opencv, laplacian_opencv, CV_64F);

    Scalar mean_lap_src, std_lap_src, mean_lap_result, std_lap_result, mean_lap_opencv, std_lap_opencv;
    meanStdDev(laplacian_src, mean_lap_src, std_lap_src);
    meanStdDev(laplacian_result, mean_lap_result, std_lap_result);
    meanStdDev(laplacian_opencv, mean_lap_opencv, std_lap_opencv);

    cout << "Original Laplacian response - Mean: " << mean_lap_src[0] << ", Std Dev: " << std_lap_src[0] << endl;
    cout << "Unsharp masking Laplacian response - Mean: " << mean_lap_result[0] << ", Std Dev: " << std_lap_result[0] << endl;
    cout << "OpenCV Laplacian response - Mean: " << mean_lap_opencv[0] << ", Std Dev: " << std_lap_opencv[0] << endl;
    cout << "Sharpening enhancement factor (Unsharp): " << std_lap_result[0] / std_lap_src[0] << endl;
    cout << "Sharpening enhancement factor (OpenCV): " << std_lap_opencv[0] / std_lap_src[0] << endl;

    // 计算PSNR
    Mat diff;
    absdiff(src, dst_unsharp, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);

    double mse = mean(diff)[0];
    double psnr = 10.0 * log10((255.0 * 255.0) / mse);
    cout << "PSNR: " << fixed << setprecision(2) << psnr << " dB" << endl;

    // 计算直方图相似度
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};

    Mat hist_src, hist_result, hist_opencv;
    calcHist(&gray_src, 1, 0, Mat(), hist_src, 1, &histSize, &histRange);
    calcHist(&gray_result, 1, 0, Mat(), hist_result, 1, &histSize, &histRange);
    calcHist(&gray_opencv, 1, 0, Mat(), hist_opencv, 1, &histSize, &histRange);

    double similarity_unsharp = compareHist(hist_src, hist_result, HISTCMP_CORREL);
    double similarity_opencv = compareHist(hist_src, hist_opencv, HISTCMP_CORREL);

    cout << "Histogram similarity (Unsharp): " << fixed << setprecision(3) << similarity_unsharp << endl;
    cout << "Histogram similarity (OpenCV): " << similarity_opencv << endl;

    // 计算边缘增强效果
    Mat edges_src, edges_result, edges_opencv;
    Canny(gray_src, edges_src, 50, 150);
    Canny(gray_result, edges_result, 50, 150);
    Canny(gray_opencv, edges_opencv, 50, 150);

    Scalar mean_edges_src, std_edges_src, mean_edges_result, std_edges_result;
    meanStdDev(edges_src, mean_edges_src, std_edges_src);
    meanStdDev(edges_result, mean_edges_result, std_edges_result);

    cout << "Original edge density: " << mean_edges_src[0] << endl;
    cout << "Unsharp masking edge density: " << mean_edges_result[0] << endl;
    cout << "Edge enhancement factor: " << mean_edges_result[0] / mean_edges_src[0] << endl;

    // 计算颜色保持度
    Scalar mean_src, std_src, mean_result, std_result;
    meanStdDev(src, mean_src, std_src);
    meanStdDev(dst_unsharp, mean_result, std_result);

    cout << "Color preservation - B: " << abs(mean_result[0] - mean_src[0]) / mean_src[0] * 100 << "%" << endl;
    cout << "Color preservation - G: " << abs(mean_result[1] - mean_src[1]) / mean_src[1] * 100 << "%" << endl;
    cout << "Color preservation - R: " << abs(mean_result[2] - mean_src[2]) / mean_src[2] * 100 << "%" << endl;
}

int main() {
    cout << "=== Unsharp Masking Algorithm Test ===" << endl;

    // 创建输出目录
    create_output_directories();

    // 加载测试图像
    Mat src = imread("assets/imori.jpg");
    if (src.empty()) {
        cerr << "Cannot load test image" << endl;
        return -1;
    }

    cout << "Image size: " << src.cols << "x" << src.rows << endl;

    // 执行各项测试
    performance_test(src);
    parameter_effect_test(src);
    visualization_test(src);
    special_scenario_test(src);
    quality_assessment(src);

    cout << "\n=== Test Completed ===" << endl;
    cout << "All results have been saved to output/unsharp_masking/ directory" << endl;

    return 0;
}
