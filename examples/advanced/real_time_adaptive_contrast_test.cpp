#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include "advanced/enhancement/real_time_adaptive_contrast.hpp"

using namespace cv;
using namespace std;

// Create output directories
void create_output_directories() {
    filesystem::create_directories("output/real_time_adaptive_contrast");
}

// Performance test
void performance_test(const Mat& src) {
    cout << "\n--- Performance Test ---" << endl;

    Mat dst;
    int iterations = 10;

    // Test IP101 algorithm performance
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        ip101::advanced::real_time_adaptive_contrast(src, dst, 7, 2.0);
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    double avg_time = duration.count() / (double)iterations;
    double fps = 1000000.0 / avg_time;

    cout << "IP101 Real-time Adaptive Contrast - Average time: " << fixed << setprecision(2)
         << avg_time / 1000.0 << " ms, FPS: " << fps << endl;

    // Test OpenCV CLAHE algorithm performance
    Mat opencv_result;
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(7, 7));
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        clahe->apply(src, opencv_result);
    }
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);

    avg_time = duration.count() / (double)iterations;
    fps = 1000000.0 / avg_time;

    cout << "OpenCV CLAHE - Average time: " << fixed << setprecision(2)
         << avg_time / 1000.0 << " ms, FPS: " << fps << endl;
}

// Parameter effect test
void parameter_effect_test(const Mat& src) {
    cout << "\n--- Parameter Effect Test ---" << endl;

    vector<int> window_sizes = {3, 5, 7, 9, 11};
    vector<double> clip_limits = {1.0, 1.5, 2.0, 2.5, 3.0};

    // Test window size parameters
    cout << "Testing window size parameter effects..." << endl;
    for (size_t i = 0; i < window_sizes.size(); i++) {
        Mat result;
        ip101::advanced::real_time_adaptive_contrast(src, result, window_sizes[i], 2.0);
        string filename = "output/real_time_adaptive_contrast/window_" + to_string(i) + ".jpg";
        imwrite(filename, result);
    }

    // Test clip limit parameters
    cout << "Testing clip limit parameter effects..." << endl;
    for (size_t i = 0; i < clip_limits.size(); i++) {
        Mat result;
        ip101::advanced::real_time_adaptive_contrast(src, result, 7, clip_limits[i]);
        string filename = "output/real_time_adaptive_contrast/clip_" + to_string(i) + ".jpg";
        imwrite(filename, result);
    }
}

// Visualization test
void visualization_test(const Mat& src) {
        cout << "\n--- Visualization Test ---" << endl;

    // Apply real-time adaptive contrast
    Mat dst_adaptive_contrast;
    ip101::advanced::real_time_adaptive_contrast(src, dst_adaptive_contrast, 7, 2.0);

    // 创建OpenCV对比结果
    Mat opencv_result;
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(7, 7));
    clahe->apply(src, opencv_result);

    // Save results
    imwrite("output/real_time_adaptive_contrast/original.jpg", src);
    imwrite("output/real_time_adaptive_contrast/enhanced.jpg", dst_adaptive_contrast);
    imwrite("output/real_time_adaptive_contrast/opencv_comparison.jpg", opencv_result);

    // Create comparison image
    vector<Mat> images = {src, dst_adaptive_contrast, opencv_result};
    Mat comparison;
    hconcat(images, comparison);

    // Add titles
    vector<string> titles = {"Original", "Real-time Adaptive Contrast", "OpenCV CLAHE"};
    int font_face = FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.8;
    Scalar color(255, 255, 255);
    int thickness = 2;

    for (size_t i = 0; i < titles.size(); i++) {
        int x = i * src.cols + 10;
        putText(comparison, titles[i], Point(x, 30), font_face, font_scale, color, thickness);
    }

    imwrite("output/real_time_adaptive_contrast/comparison.jpg", comparison);
    cout << "Comparison image saved to: output/real_time_adaptive_contrast/comparison.jpg" << endl;
}

// Special scenario test
void special_scenario_test(const Mat& src) {
    cout << "\n--- Special Scenario Test ---" << endl;

    // Create low contrast image
    Mat low_contrast = src.clone();
    low_contrast.convertTo(low_contrast, -1, 0.3, 50);
    imwrite("output/real_time_adaptive_contrast/low_contrast.jpg", low_contrast);

    // 应用实时自适应对比度
    Mat low_contrast_result;
    ip101::advanced::real_time_adaptive_contrast(low_contrast, low_contrast_result, 7, 2.0);
    imwrite("output/real_time_adaptive_contrast/low_contrast_result.jpg", low_contrast_result);

    // Create dark image
    Mat dark_image = src.clone();
    dark_image.convertTo(dark_image, -1, 0.4, 0);
    imwrite("output/real_time_adaptive_contrast/dark_image.jpg", dark_image);

    // 应用实时自适应对比度
    Mat dark_result;
    ip101::advanced::real_time_adaptive_contrast(dark_image, dark_result, 7, 2.0);
    imwrite("output/real_time_adaptive_contrast/dark_result.jpg", dark_result);

    // Create high contrast image
    Mat high_contrast = src.clone();
    high_contrast.convertTo(high_contrast, -1, 2.0, -50);
    imwrite("output/real_time_adaptive_contrast/high_contrast.jpg", high_contrast);

    // 应用实时自适应对比度
    Mat high_contrast_result;
    ip101::advanced::real_time_adaptive_contrast(high_contrast, high_contrast_result, 7, 2.0);
    imwrite("output/real_time_adaptive_contrast/high_contrast_result.jpg", high_contrast_result);
}

// Quality assessment
void quality_assessment(const Mat& src) {
    cout << "\n--- Quality Assessment ---" << endl;

    // 应用实时自适应对比度
    Mat dst_adaptive_contrast;
    ip101::advanced::real_time_adaptive_contrast(src, dst_adaptive_contrast, 7, 2.0);

    // 创建OpenCV对比结果
    Mat opencv_result;
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(7, 7));
    clahe->apply(src, opencv_result);

    // Calculate contrast enhancement effect
    Mat gray_src, gray_result, gray_clahe;
    cvtColor(src, gray_src, COLOR_BGR2GRAY);
    cvtColor(dst_adaptive_contrast, gray_result, COLOR_BGR2GRAY);
    cvtColor(opencv_result, gray_clahe, COLOR_BGR2GRAY);

    Scalar mean_src, std_src, mean_result, std_result, mean_clahe, std_clahe;
    meanStdDev(gray_src, mean_src, std_src);
    meanStdDev(gray_result, mean_result, std_result);
    meanStdDev(gray_clahe, mean_clahe, std_clahe);

    cout << "Original - Mean: " << mean_src[0] << ", Std Dev: " << std_src[0] << endl;
    cout << "Real-time Adaptive Contrast - Mean: " << mean_result[0] << ", Std Dev: " << std_result[0] << endl;
    cout << "OpenCV CLAHE - Mean: " << mean_clahe[0] << ", Std Dev: " << std_clahe[0] << endl;
    cout << "Contrast enhancement factor (Custom): " << std_result[0] / std_src[0] << endl;
    cout << "Contrast enhancement factor (CLAHE): " << std_clahe[0] / std_src[0] << endl;

    // Calculate PSNR
    Mat diff;
    absdiff(src, dst_adaptive_contrast, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);

    double mse = mean(diff)[0];
    double psnr = 10.0 * log10((255.0 * 255.0) / mse);
    cout << "PSNR: " << fixed << setprecision(2) << psnr << " dB" << endl;

    // Calculate histogram similarity
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};

    Mat hist_src, hist_result, hist_clahe;
    calcHist(&gray_src, 1, 0, Mat(), hist_src, 1, &histSize, &histRange);
    calcHist(&gray_result, 1, 0, Mat(), hist_result, 1, &histSize, &histRange);
    calcHist(&gray_clahe, 1, 0, Mat(), hist_clahe, 1, &histSize, &histRange);

    double similarity_custom = compareHist(hist_src, hist_result, HISTCMP_CORREL);
    double similarity_clahe = compareHist(hist_src, hist_clahe, HISTCMP_CORREL);

    cout << "Histogram similarity (Custom): " << fixed << setprecision(3) << similarity_custom << endl;
    cout << "Histogram similarity (CLAHE): " << similarity_clahe << endl;
}

int main(int argc, char** argv) {
    cout << "=== Real-time Adaptive Contrast Algorithm Test ===" << endl;

    // Create output directory
    create_output_directories();

    string image_path = (argc > 1) ? argv[1] : "assets/imori.jpg";
    Mat src = imread(image_path);
    if (src.empty()) {
        cerr << "Error: Cannot load image " << image_path << endl;
        return -1;
    }

    cout << "Image size: " << src.cols << "x" << src.rows << endl;

    // Execute various tests
    performance_test(src);
    parameter_effect_test(src);
    visualization_test(src);
    special_scenario_test(src);
    quality_assessment(src);

    cout << "\n=== Test Completed ===" << endl;
    cout << "All results have been saved to output/real_time_adaptive_contrast/ directory" << endl;

    return 0;
}
