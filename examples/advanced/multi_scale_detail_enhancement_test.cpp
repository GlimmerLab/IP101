#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include "advanced/enhancement/multi_scale_detail_enhancement.hpp"

using namespace cv;
using namespace std;

// Create output directories
void create_output_directories() {
    filesystem::create_directories("output/multi_scale_detail_enhancement");
}

// Performance test
void performance_test(const Mat& src) {
    cout << "\n--- Performance Test ---" << endl;

    Mat dst;
    int iterations = 10;

    // Test IP101 algorithm performance
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        ip101::advanced::multi_scale_detail_enhancement(src, dst, 1.0, 2.0, 1.5, 1.2);
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    double avg_time = duration.count() / (double)iterations;
    double fps = 1000000.0 / avg_time;

    cout << "IP101 Multi-scale Detail Enhancement - Average time: " << fixed << setprecision(2)
         << avg_time / 1000.0 << " ms, FPS: " << fps << endl;

    // Test OpenCV comparison algorithm (using bilateral filter as comparison)
    Mat opencv_result;
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        bilateralFilter(src, opencv_result, 15, 80, 80);
    }
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);

    avg_time = duration.count() / (double)iterations;
    fps = 1000000.0 / avg_time;

    cout << "OpenCV Bilateral Filter - Average time: " << fixed << setprecision(2)
         << avg_time / 1000.0 << " ms, FPS: " << fps << endl;
}

// Parameter effect test
void parameter_effect_test(const Mat& src) {
    cout << "\n--- Parameter Effect Test ---" << endl;

    vector<double> sigma_values = {0.5, 1.0, 1.5, 2.0, 2.5};
    vector<double> strength_values = {1.0, 1.5, 2.0, 2.5, 3.0};
    vector<double> detail_values = {1.0, 1.2, 1.5, 1.8, 2.0};
    vector<double> contrast_values = {1.0, 1.1, 1.2, 1.3, 1.4};

    // Test sigma parameter
    cout << "Testing sigma parameter effects..." << endl;
    for (size_t i = 0; i < sigma_values.size(); i++) {
        Mat result;
        ip101::advanced::multi_scale_detail_enhancement(src, result, sigma_values[i], 2.0, 1.5, 1.2);
        string filename = "output/multi_scale_detail_enhancement/sigma_" + to_string(i) + ".jpg";
        imwrite(filename, result);
    }

    // Test strength parameter
    cout << "Testing strength parameter effects..." << endl;
    for (size_t i = 0; i < strength_values.size(); i++) {
        Mat result;
        ip101::advanced::multi_scale_detail_enhancement(src, result, 1.0, strength_values[i], 1.5, 1.2);
        string filename = "output/multi_scale_detail_enhancement/strength_" + to_string(i) + ".jpg";
        imwrite(filename, result);
    }

    // Test detail parameter
    cout << "Testing detail parameter effects..." << endl;
    for (size_t i = 0; i < detail_values.size(); i++) {
        Mat result;
        ip101::advanced::multi_scale_detail_enhancement(src, result, 1.0, 2.0, detail_values[i], 1.2);
        string filename = "output/multi_scale_detail_enhancement/detail_" + to_string(i) + ".jpg";
        imwrite(filename, result);
    }

    // Test contrast parameter
    cout << "Testing contrast parameter effects..." << endl;
    for (size_t i = 0; i < contrast_values.size(); i++) {
        Mat result;
        ip101::advanced::multi_scale_detail_enhancement(src, result, 1.0, 2.0, 1.5, contrast_values[i]);
        string filename = "output/multi_scale_detail_enhancement/contrast_" + to_string(i) + ".jpg";
        imwrite(filename, result);
    }
}

// Visualization test
void visualization_test(const Mat& src) {
    cout << "\n--- Visualization Test ---" << endl;

    // Apply multi-scale detail enhancement
    Mat dst_multi_scale;
    ip101::advanced::multi_scale_detail_enhancement(src, dst_multi_scale, 1.0, 2.0, 1.5, 1.2);

    // Save results
    imwrite("output/multi_scale_detail_enhancement/original.jpg", src);
    imwrite("output/multi_scale_detail_enhancement/enhanced.jpg", dst_multi_scale);

    // Create OpenCV comparison result
    Mat opencv_result;
    bilateralFilter(src, opencv_result, 15, 80, 80);
    imwrite("output/multi_scale_detail_enhancement/opencv_comparison.jpg", opencv_result);

    // Create comparison image
    vector<Mat> images = {src, dst_multi_scale, opencv_result};
    Mat comparison;
    hconcat(images, comparison);

    // Add titles
    vector<string> titles = {"Original", "Multi-scale Detail Enhancement", "OpenCV Comparison"};
    int font_face = FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.8;
    Scalar color(255, 255, 255);
    int thickness = 2;

    for (size_t i = 0; i < titles.size(); i++) {
        int x = i * src.cols + 10;
        putText(comparison, titles[i], Point(x, 30), font_face, font_scale, color, thickness);
    }

    imwrite("output/multi_scale_detail_enhancement/comparison.jpg", comparison);
    cout << "Comparison image saved to: output/multi_scale_detail_enhancement/comparison.jpg" << endl;
}

// Special scenario test
void special_scenario_test(const Mat& src) {
    cout << "\n--- Special Scenario Test ---" << endl;

    // Create blurred image
    Mat blurred = src.clone();
    GaussianBlur(blurred, blurred, Size(15, 15), 3.0);
    imwrite("output/multi_scale_detail_enhancement/blurred.jpg", blurred);

    // Apply multi-scale detail enhancement
    Mat blurred_result;
    ip101::advanced::multi_scale_detail_enhancement(blurred, blurred_result, 1.0, 2.0, 1.5, 1.2);
    imwrite("output/multi_scale_detail_enhancement/blurred_result.jpg", blurred_result);

    // Create low contrast image
    Mat low_contrast = src.clone();
    low_contrast.convertTo(low_contrast, -1, 0.5, 50);
    imwrite("output/multi_scale_detail_enhancement/low_contrast.jpg", low_contrast);

    // Apply multi-scale detail enhancement
    Mat low_contrast_result;
    ip101::advanced::multi_scale_detail_enhancement(low_contrast, low_contrast_result, 1.0, 2.0, 1.5, 1.2);
    imwrite("output/multi_scale_detail_enhancement/low_contrast_result.jpg", low_contrast_result);

    // Create noisy image
    Mat noisy = src.clone();
    Mat noise = Mat::zeros(noisy.size(), CV_8UC3);
    randn(noise, 0, 15);
    noisy = noisy + noise;
    imwrite("output/multi_scale_detail_enhancement/noisy.jpg", noisy);

    // Apply multi-scale detail enhancement
    Mat noisy_result;
    ip101::advanced::multi_scale_detail_enhancement(noisy, noisy_result, 1.0, 2.0, 1.5, 1.2);
    imwrite("output/multi_scale_detail_enhancement/noisy_result.jpg", noisy_result);
}

// Quality assessment
void quality_assessment(const Mat& src) {
    cout << "\n--- Quality Assessment ---" << endl;

    // Apply multi-scale detail enhancement
    Mat dst_multi_scale;
    ip101::advanced::multi_scale_detail_enhancement(src, dst_multi_scale, 1.0, 2.0, 1.5, 1.2);

    // Calculate detail enhancement effect
    Mat gray_src, gray_result;
    cvtColor(src, gray_src, COLOR_BGR2GRAY);
    cvtColor(dst_multi_scale, gray_result, COLOR_BGR2GRAY);

    // Calculate Laplacian operator response (detail measure)
    Mat laplacian_src, laplacian_result;
    Laplacian(gray_src, laplacian_src, CV_64F);
    Laplacian(gray_result, laplacian_result, CV_64F);

    Scalar mean_lap_src, std_lap_src, mean_lap_result, std_lap_result;
    meanStdDev(laplacian_src, mean_lap_src, std_lap_src);
    meanStdDev(laplacian_result, mean_lap_result, std_lap_result);

    cout << "Original Laplacian response - Mean: " << mean_lap_src[0] << ", Std Dev: " << std_lap_src[0] << endl;
    cout << "Processed Laplacian response - Mean: " << mean_lap_result[0] << ", Std Dev: " << std_lap_result[0] << endl;
    cout << "Detail enhancement factor: " << std_lap_result[0] / std_lap_src[0] << endl;

    // Calculate contrast enhancement effect
    Scalar mean_src, std_src, mean_result, std_result;
    meanStdDev(gray_src, mean_src, std_src);
    meanStdDev(gray_result, mean_result, std_result);

    cout << "Original - Mean: " << mean_src[0] << ", Std Dev: " << std_src[0] << endl;
    cout << "Processed - Mean: " << mean_result[0] << ", Std Dev: " << std_result[0] << endl;
    cout << "Contrast enhancement factor: " << std_result[0] / std_src[0] << endl;

    // Calculate PSNR
    Mat diff;
    absdiff(src, dst_multi_scale, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);

    double mse = mean(diff)[0];
    double psnr = 10.0 * log10((255.0 * 255.0) / mse);
    cout << "PSNR: " << fixed << setprecision(2) << psnr << " dB" << endl;
}

// Histogram analysis
void histogram_analysis(const Mat& src) {
    cout << "\n--- Histogram Analysis ---" << endl;

    // Apply multi-scale detail enhancement
    Mat dst_multi_scale;
    ip101::advanced::multi_scale_detail_enhancement(src, dst_multi_scale, 1.0, 2.0, 1.5, 1.2);

    // Calculate histograms of original and enhanced images
    vector<Mat> bgr_planes;
    split(src, bgr_planes);

    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};

    Mat b_hist, g_hist, r_hist;
    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange);

    // Calculate histogram of enhanced image
    vector<Mat> bgr_planes_result;
    split(dst_multi_scale, bgr_planes_result);

    Mat b_hist_result, g_hist_result, r_hist_result;
    calcHist(&bgr_planes_result[0], 1, 0, Mat(), b_hist_result, 1, &histSize, &histRange);
    calcHist(&bgr_planes_result[1], 1, 0, Mat(), g_hist_result, 1, &histSize, &histRange);
    calcHist(&bgr_planes_result[2], 1, 0, Mat(), r_hist_result, 1, &histSize, &histRange);

    // Calculate histogram similarity
    double b_similarity = compareHist(b_hist, b_hist_result, HISTCMP_CORREL);
    double g_similarity = compareHist(g_hist, g_hist_result, HISTCMP_CORREL);
    double r_similarity = compareHist(r_hist, r_hist_result, HISTCMP_CORREL);

    cout << "Histogram similarity - B: " << fixed << setprecision(3) << b_similarity
         << ", G: " << g_similarity << ", R: " << r_similarity << endl;

    // Calculate dynamic range
    Mat gray_src, gray_result;
    cvtColor(src, gray_src, COLOR_BGR2GRAY);
    cvtColor(dst_multi_scale, gray_result, COLOR_BGR2GRAY);

    double min_val_src, max_val_src, min_val_result, max_val_result;
    minMaxLoc(gray_src, &min_val_src, &max_val_src);
    minMaxLoc(gray_result, &min_val_result, &max_val_result);

    cout << "Original dynamic range: " << (max_val_src - min_val_src) << endl;
    cout << "Processed dynamic range: " << (max_val_result - min_val_result) << endl;
}

int main(int argc, char** argv) {
    cout << "=== Multi-scale Detail Enhancement Algorithm Test ===" << endl;

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
    histogram_analysis(src);

    cout << "\n=== Test Completed ===" << endl;
    cout << "All results have been saved to output/multi_scale_detail_enhancement/ directory" << endl;

    return 0;
}
