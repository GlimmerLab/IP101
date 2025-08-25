#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include "advanced/effects/vintage_effect.hpp"
#include "system_info.hpp"

using namespace cv;
using namespace std;
using namespace ip101::utils;

// 打印测试环境信息
void print_test_header() {
    cout << "=== Test Environment Information ===" << endl;
    auto sys_info = SystemInfo::getSystemInfo();
    cout << SystemInfo::formatSystemInfo(sys_info) << endl;
    cout << "=====================================" << endl;
}

// 创建输出目录
void create_output_directories() {
    filesystem::create_directories("output/vintage_effect");
}

// Performance test
void performance_test(const Mat& src) {
    cout << "=== Vintage Effect Performance Test ===" << endl;

    Mat dst;
    int iterations = 10;

    // 测试IP101算法性能
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        ip101::advanced::VintageParams params;
        params.sepia_intensity = 0.8;
        params.noise_level = 0.6;
        params.vignette_strength = 0.4;
        params.scratch_intensity = 0.3;
        ip101::advanced::vintage_effect(src, dst, params);
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    double avg_time = duration.count() / (double)iterations;
    double fps = 1000000.0 / avg_time;

    cout << "IP101 Vintage Effect - Average Time: " << fixed << setprecision(2)
         << avg_time / 1000.0 << " ms, FPS: " << fps << endl;

    // Test OpenCV comparison algorithm (using color mapping as comparison)
    Mat opencv_result;
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        applyColorMap(src, opencv_result, COLORMAP_OCEAN);
    }
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);

    avg_time = duration.count() / (double)iterations;
    fps = 1000000.0 / avg_time;

    cout << "OpenCV Color Mapping - Average Time: " << fixed << setprecision(2)
         << avg_time / 1000.0 << " ms, FPS: " << fps << endl;
}

// Parameter effect test
void parameter_effect_test(const Mat& src) {
    cout << "\n--- Parameter Effect Test ---" << endl;

    vector<double> sepia_strengths = {0.3, 0.5, 0.7, 0.8, 0.9};
    vector<double> grain_strengths = {0.2, 0.4, 0.6, 0.8, 1.0};
    vector<double> vignette_strengths = {0.1, 0.3, 0.5, 0.7, 0.9};
    vector<double> scratch_strengths = {0.1, 0.3, 0.5, 0.7, 0.9};

    // Test sepia tone parameters
    cout << "Testing sepia tone parameter effects..." << endl;
    for (size_t i = 0; i < sepia_strengths.size(); i++) {
        Mat result;
        ip101::advanced::VintageParams params;
        params.sepia_intensity = sepia_strengths[i];
        params.noise_level = 0.6;
        params.vignette_strength = 0.4;
        params.scratch_intensity = 0.3;
        ip101::advanced::vintage_effect(src, result, params);
        string filename = "output/vintage_effect/sepia_" + to_string(i) + ".jpg";
        imwrite(filename, result);
    }

    // Test grain parameters
    cout << "Testing grain parameter effects..." << endl;
    for (size_t i = 0; i < grain_strengths.size(); i++) {
        Mat result;
        ip101::advanced::VintageParams params;
        params.sepia_intensity = 0.8;
        params.noise_level = grain_strengths[i];
        params.vignette_strength = 0.4;
        params.scratch_intensity = 0.3;
        ip101::advanced::vintage_effect(src, result, params);
        string filename = "output/vintage_effect/grain_" + to_string(i) + ".jpg";
        imwrite(filename, result);
    }

    // Test vignette parameters
    cout << "Testing vignette parameter effects..." << endl;
    for (size_t i = 0; i < vignette_strengths.size(); i++) {
        Mat result;
        ip101::advanced::VintageParams params;
        params.sepia_intensity = 0.8;
        params.noise_level = 0.6;
        params.vignette_strength = vignette_strengths[i];
        params.scratch_intensity = 0.3;
        ip101::advanced::vintage_effect(src, result, params);
        string filename = "output/vintage_effect/vignette_" + to_string(i) + ".jpg";
        imwrite(filename, result);
    }

    // Test scratch parameters
    cout << "Testing scratch parameter effects..." << endl;
    for (size_t i = 0; i < scratch_strengths.size(); i++) {
        Mat result;
        ip101::advanced::VintageParams params;
        params.sepia_intensity = 0.8;
        params.noise_level = 0.6;
        params.vignette_strength = 0.4;
        params.scratch_intensity = scratch_strengths[i];
        ip101::advanced::vintage_effect(src, result, params);
        string filename = "output/vintage_effect/scratch_" + to_string(i) + ".jpg";
        imwrite(filename, result);
    }
}

// Visualization test
void visualization_test(const Mat& src) {
    cout << "\n--- Visualization Test ---" << endl;

    // Apply vintage effect
    Mat dst_vintage;
    ip101::advanced::VintageParams params_viz;
    params_viz.sepia_intensity = 0.8;
    params_viz.noise_level = 0.6;
    params_viz.vignette_strength = 0.4;
    params_viz.scratch_intensity = 0.3;
    ip101::advanced::vintage_effect(src, dst_vintage, params_viz);

    // Save results
    imwrite("output/vintage_effect/original.jpg", src);
    imwrite("output/vintage_effect/vintage.jpg", dst_vintage);

    // Create OpenCV comparison result
    Mat opencv_result;
    applyColorMap(src, opencv_result, COLORMAP_OCEAN);
    imwrite("output/vintage_effect/opencv_comparison.jpg", opencv_result);

    // Create comparison image
    // Ensure all images have the same size for hconcat
    Mat dst_vintage_resized, opencv_result_resized;
    resize(dst_vintage, dst_vintage_resized, src.size());
    resize(opencv_result, opencv_result_resized, src.size());

    vector<Mat> images = {src, dst_vintage_resized, opencv_result_resized};
    Mat comparison;
    hconcat(images, comparison);

    // Add titles
    vector<string> titles = {"Original", "Vintage Effect", "OpenCV Comparison"};
    int font_face = FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.8;
    Scalar color(255, 255, 255);
    int thickness = 2;

    for (size_t i = 0; i < titles.size(); i++) {
        int x = i * src.cols + 10;
        putText(comparison, titles[i], Point(x, 30), font_face, font_scale, color, thickness);
    }

    imwrite("output/vintage_effect/comparison.jpg", comparison);
    cout << "Comparison image saved to: output/vintage_effect/comparison.jpg" << endl;
}

// Special scenario test
void special_scenario_test(const Mat& src) {
    cout << "\n--- Special Scenario Test ---" << endl;

    // Test different vintage styles
    vector<vector<double>> vintage_styles = {
        {0.9, 0.8, 0.6, 0.5},  // Strong vintage
        {0.7, 0.5, 0.3, 0.2},  // Medium vintage
        {0.5, 0.3, 0.2, 0.1},  // Light vintage
        {0.3, 0.2, 0.1, 0.05}  // Subtle vintage
    };
    vector<string> style_names = {"Strong_Vintage", "Medium_Vintage", "Light_Vintage", "Subtle_Vintage"};

    for (size_t i = 0; i < vintage_styles.size(); i++) {
        Mat result;
        ip101::advanced::VintageParams params_style;
        params_style.sepia_intensity = vintage_styles[i][0];
        params_style.noise_level = vintage_styles[i][1];
        params_style.vignette_strength = vintage_styles[i][2];
        params_style.scratch_intensity = vintage_styles[i][3];
        ip101::advanced::vintage_effect(src, result, params_style);
        string filename = "output/vintage_effect/" + style_names[i] + ".jpg";
        imwrite(filename, result);
    }

    // Test effects under different lighting conditions
    // Dark image
    Mat dark = src.clone();
    dark.convertTo(dark, -1, 0.5, 0);
    imwrite("output/vintage_effect/dark.jpg", dark);

    Mat dark_result;
    ip101::advanced::VintageParams params_dark;
    params_dark.sepia_intensity = 0.8;
    params_dark.noise_level = 0.6;
    params_dark.vignette_strength = 0.4;
    params_dark.scratch_intensity = 0.3;
    ip101::advanced::vintage_effect(dark, dark_result, params_dark);
    imwrite("output/vintage_effect/dark_result.jpg", dark_result);

    // Bright image
    Mat bright = src.clone();
    bright.convertTo(bright, -1, 1.5, 50);
    imwrite("output/vintage_effect/bright.jpg", bright);

    Mat bright_result;
    ip101::advanced::VintageParams params2;
    params2.sepia_intensity = 0.8;
    params2.noise_level = 0.6;
    params2.vignette_strength = 0.4;
    params2.scratch_intensity = 0.3;
    ip101::advanced::vintage_effect(bright, bright_result, params2);
    imwrite("output/vintage_effect/bright_result.jpg", bright_result);
}

// Quality assessment
void quality_assessment(const Mat& src) {
    cout << "\n--- Quality Assessment ---" << endl;

    // Apply vintage effect
    Mat dst_vintage;
    ip101::advanced::VintageParams params_qa;
    params_qa.sepia_intensity = 0.8;
    params_qa.noise_level = 0.6;
    params_qa.vignette_strength = 0.4;
    params_qa.scratch_intensity = 0.3;
    ip101::advanced::vintage_effect(src, dst_vintage, params_qa);

    // Create OpenCV comparison result
    Mat opencv_result;
    applyColorMap(src, opencv_result, COLORMAP_OCEAN);

    // Calculate color change effects
    Mat gray_src, gray_result, gray_opencv;
    cvtColor(src, gray_src, COLOR_BGR2GRAY);
    cvtColor(dst_vintage, gray_result, COLOR_BGR2GRAY);
    cvtColor(opencv_result, gray_opencv, COLOR_BGR2GRAY);

    // Calculate color statistics
    Scalar mean_src, std_src, mean_result, std_result, mean_opencv, std_opencv;
    meanStdDev(src, mean_src, std_src);
    meanStdDev(dst_vintage, mean_result, std_result);
    meanStdDev(opencv_result, mean_opencv, std_opencv);

    cout << "Original Image Color Stats - B: " << mean_src[0] << ", G: " << mean_src[1] << ", R: " << mean_src[2] << endl;
    cout << "Vintage Effect Color Stats - B: " << mean_result[0] << ", G: " << mean_result[1] << ", R: " << mean_result[2] << endl;
    cout << "OpenCV Color Mapping Stats - B: " << mean_opencv[0] << ", G: " << mean_opencv[1] << ", R: " << mean_opencv[2] << endl;

    // Calculate color change degree
    cout << "Color Change Degree - B: " << abs(mean_result[0] - mean_src[0]) / mean_src[0] * 100 << "%" << endl;
    cout << "Color Change Degree - G: " << abs(mean_result[1] - mean_src[1]) / mean_src[1] * 100 << "%" << endl;
    cout << "Color Change Degree - R: " << abs(mean_result[2] - mean_src[2]) / mean_src[2] * 100 << "%" << endl;

    // Calculate PSNR
    Mat diff;
    // Ensure both images have the same size for absdiff
    Mat dst_vintage_resized;
    resize(dst_vintage, dst_vintage_resized, src.size());
    absdiff(src, dst_vintage_resized, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);

    double mse = mean(diff)[0];
    double psnr = 10.0 * log10((255.0 * 255.0) / mse);
    cout << "PSNR: " << fixed << setprecision(2) << psnr << " dB" << endl;

    // Calculate histogram similarity
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};

    Mat hist_src, hist_result, hist_opencv;
    calcHist(&gray_src, 1, 0, Mat(), hist_src, 1, &histSize, &histRange);
    calcHist(&gray_result, 1, 0, Mat(), hist_result, 1, &histSize, &histRange);
    calcHist(&gray_opencv, 1, 0, Mat(), hist_opencv, 1, &histSize, &histRange);

    double similarity_vintage = compareHist(hist_src, hist_result, HISTCMP_CORREL);
    double similarity_opencv = compareHist(hist_src, hist_opencv, HISTCMP_CORREL);

    cout << "Histogram Similarity (Vintage Effect): " << fixed << setprecision(3) << similarity_vintage << endl;
    cout << "Histogram Similarity (OpenCV): " << similarity_opencv << endl;

    // Calculate contrast change
    Scalar mean_gray_src, std_gray_src, mean_gray_result, std_gray_result;
    meanStdDev(gray_src, mean_gray_src, std_gray_src);
    meanStdDev(gray_result, mean_gray_result, std_gray_result);

    cout << "Original Image Contrast: " << std_gray_src[0] << endl;
    cout << "Vintage Effect Contrast: " << std_gray_result[0] << endl;
    cout << "Contrast Change Rate: " << (std_gray_result[0] - std_gray_src[0]) / std_gray_src[0] * 100 << "%" << endl;

    // Calculate brightness change
    cout << "Brightness Change: " << (mean_gray_result[0] - mean_gray_src[0]) / mean_gray_src[0] * 100 << "%" << endl;

    // Calculate saturation change (through color channel standard deviation)
    double saturation_src = (std_src[0] + std_src[1] + std_src[2]) / 3.0;
    double saturation_result = (std_result[0] + std_result[1] + std_result[2]) / 3.0;

    cout << "Saturation Change: " << (saturation_result - saturation_src) / saturation_src * 100 << "%" << endl;
}

int main(int argc, char** argv) {
    cout << "=== Vintage Effect Algorithm Test ===" << endl;

    // Print test environment information
    print_test_header();

    // Create output directories
    create_output_directories();

    string image_path = (argc > 1) ? argv[1] : "assets/imori.jpg";
    Mat src = imread(image_path);
    if (src.empty()) {
        cerr << "Error: Cannot load image " << image_path << endl;
        return -1;
    }

    cout << "Image size: " << src.cols << "x" << src.rows << endl;

    // Execute all tests
    performance_test(src);
    parameter_effect_test(src);
    visualization_test(src);
    special_scenario_test(src);
    quality_assessment(src);

    cout << "\n=== Test Completed ===" << endl;
    cout << "All results saved to output/vintage_effect/ directory" << endl;

    return 0;
}
