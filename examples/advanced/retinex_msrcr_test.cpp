#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include "advanced/enhancement/retinex_msrcr.hpp"

using namespace cv;
using namespace std;

// 创建输出目录
void create_output_directories() {
    filesystem::create_directories("output/retinex_msrcr");
}

// 性能测试
void performance_test(const Mat& src) {
    cout << "\n--- Performance Test ---" << endl;

    Mat dst;
    int iterations = 10;

    // 测试IP101算法性能
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        ip101::advanced::retinex_msrcr(src, dst, 15, 80, 200, 0.01, 0.01, 0.01);
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    double avg_time = duration.count() / (double)iterations;
    double fps = 1000000.0 / avg_time;

    cout << "IP101 Retinex MSRCR - Average Time: " << fixed << setprecision(2)
         << avg_time / 1000.0 << " ms, FPS: " << fps << endl;

    // 测试OpenCV对比算法（使用直方图均衡化作为对比）
    Mat opencv_result;
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        equalizeHist(src, opencv_result);
    }
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);

    avg_time = duration.count() / (double)iterations;
    fps = 1000000.0 / avg_time;

    cout << "OpenCV Histogram Equalization - Average Time: " << fixed << setprecision(2)
         << avg_time / 1000.0 << " ms, FPS: " << fps << endl;
}

// 参数效果测试
void parameter_effect_test(const Mat& src) {
    cout << "\n--- Parameter Effect Test ---" << endl;

    vector<int> sigma_values = {5, 10, 15, 20, 25};
    vector<int> gain_values = {50, 80, 120, 150, 200};
    vector<int> offset_values = {100, 150, 200, 250, 300};
    vector<double> alpha_values = {0.005, 0.01, 0.02, 0.05, 0.1};
    vector<double> beta_values = {0.005, 0.01, 0.02, 0.05, 0.1};
    vector<double> gamma_values = {0.005, 0.01, 0.02, 0.05, 0.1};

    // Test sigma parameters
    cout << "Testing sigma parameter effects..." << endl;
    for (size_t i = 0; i < sigma_values.size(); i++) {
        Mat result;
        ip101::advanced::retinex_msrcr(src, result, sigma_values[i], 80, 200, 0.01, 0.01, 0.01);
        string filename = "output/retinex_msrcr/sigma_" + to_string(i) + ".jpg";
        imwrite(filename, result);
    }

    // Test gain parameters
    cout << "Testing gain parameter effects..." << endl;
    for (size_t i = 0; i < gain_values.size(); i++) {
        Mat result;
        ip101::advanced::retinex_msrcr(src, result, 15, gain_values[i], 200, 0.01, 0.01, 0.01);
        string filename = "output/retinex_msrcr/gain_" + to_string(i) + ".jpg";
        imwrite(filename, result);
    }

    // Test offset parameters
    cout << "Testing offset parameter effects..." << endl;
    for (size_t i = 0; i < offset_values.size(); i++) {
        Mat result;
        ip101::advanced::retinex_msrcr(src, result, 15, 80, offset_values[i], 0.01, 0.01, 0.01);
        string filename = "output/retinex_msrcr/offset_" + to_string(i) + ".jpg";
        imwrite(filename, result);
    }

    // Test alpha parameters
    cout << "Testing alpha parameter effects..." << endl;
    for (size_t i = 0; i < alpha_values.size(); i++) {
        Mat result;
        ip101::advanced::retinex_msrcr(src, result, 15, 80, 200, alpha_values[i], 0.01, 0.01);
        string filename = "output/retinex_msrcr/alpha_" + to_string(i) + ".jpg";
        imwrite(filename, result);
    }

    // Test beta parameters
    cout << "Testing beta parameter effects..." << endl;
    for (size_t i = 0; i < beta_values.size(); i++) {
        Mat result;
        ip101::advanced::retinex_msrcr(src, result, 15, 80, 200, 0.01, beta_values[i], 0.01);
        string filename = "output/retinex_msrcr/beta_" + to_string(i) + ".jpg";
        imwrite(filename, result);
    }

    // Test gamma parameters
    cout << "Testing gamma parameter effects..." << endl;
    for (size_t i = 0; i < gamma_values.size(); i++) {
        Mat result;
        ip101::advanced::retinex_msrcr(src, result, 15, 80, 200, 0.01, 0.01, gamma_values[i]);
        string filename = "output/retinex_msrcr/gamma_" + to_string(i) + ".jpg";
        imwrite(filename, result);
    }
}

// 可视化测试
void visualization_test(const Mat& src) {
    cout << "\n--- Visualization Test ---" << endl;

    // 应用Retinex MSRCR
    Mat dst_retinex;
    ip101::advanced::retinex_msrcr(src, dst_retinex, 15, 80, 200, 0.01, 0.01, 0.01);

    // 保存结果
    imwrite("output/retinex_msrcr/original.jpg", src);
    imwrite("output/retinex_msrcr/enhanced.jpg", dst_retinex);

    // 创建OpenCV对比结果
    Mat opencv_result;
    equalizeHist(src, opencv_result);
    imwrite("output/retinex_msrcr/opencv_comparison.jpg", opencv_result);

    // 创建对比图像
    // Ensure all images have the same size for hconcat
    Mat dst_retinex_resized, opencv_result_resized;
    resize(dst_retinex, dst_retinex_resized, src.size());
    resize(opencv_result, opencv_result_resized, src.size());

    vector<Mat> images = {src, dst_retinex_resized, opencv_result_resized};
    Mat comparison;
    hconcat(images, comparison);

    // Add titles
    vector<string> titles = {"Original", "Retinex MSRCR", "OpenCV Comparison"};
    int font_face = FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.8;
    Scalar color(255, 255, 255);
    int thickness = 2;

    for (size_t i = 0; i < titles.size(); i++) {
        int x = i * src.cols + 10;
        putText(comparison, titles[i], Point(x, 30), font_face, font_scale, color, thickness);
    }

    imwrite("output/retinex_msrcr/comparison.jpg", comparison);
    cout << "Comparison image saved to: output/retinex_msrcr/comparison.jpg" << endl;
}

// 特殊场景测试
void special_scenario_test(const Mat& src) {
    cout << "\n--- Special Scenario Test ---" << endl;

    // 创建暗图像
    Mat dark = src.clone();
    dark.convertTo(dark, -1, 0.3, 0);
    imwrite("output/retinex_msrcr/dark.jpg", dark);

    // 应用Retinex MSRCR
    Mat dark_result;
    ip101::advanced::retinex_msrcr(dark, dark_result, 15, 80, 200, 0.01, 0.01, 0.01);
    imwrite("output/retinex_msrcr/dark_result.jpg", dark_result);

    // 创建过曝图像
    Mat overexposed = src.clone();
    overexposed.convertTo(overexposed, -1, 2.0, 50);
    imwrite("output/retinex_msrcr/overexposed.jpg", overexposed);

    // 应用Retinex MSRCR
    Mat overexposed_result;
    ip101::advanced::retinex_msrcr(overexposed, overexposed_result, 15, 80, 200, 0.01, 0.01, 0.01);
    imwrite("output/retinex_msrcr/overexposed_result.jpg", overexposed_result);

    // 创建低对比度图像
    Mat low_contrast = src.clone();
    low_contrast.convertTo(low_contrast, -1, 0.5, 50);
    imwrite("output/retinex_msrcr/low_contrast.jpg", low_contrast);

    // 应用Retinex MSRCR
    Mat low_contrast_result;
    ip101::advanced::retinex_msrcr(low_contrast, low_contrast_result, 15, 80, 200, 0.01, 0.01, 0.01);
    imwrite("output/retinex_msrcr/low_contrast_result.jpg", low_contrast_result);

    // 创建噪声图像
    Mat noisy = src.clone();
    Mat noise = Mat::zeros(noisy.size(), CV_8UC3);
    randn(noise, 0, 15);
    noisy = noisy + noise;
    imwrite("output/retinex_msrcr/noisy.jpg", noisy);

    // 应用Retinex MSRCR
    Mat noisy_result;
    ip101::advanced::retinex_msrcr(noisy, noisy_result, 15, 80, 200, 0.01, 0.01, 0.01);
    imwrite("output/retinex_msrcr/noisy_result.jpg", noisy_result);
}

// 质量评估
void quality_assessment(const Mat& src) {
    cout << "\n--- Quality Assessment ---" << endl;

    // 应用Retinex MSRCR
    Mat dst_retinex;
    ip101::advanced::retinex_msrcr(src, dst_retinex, 15, 80, 200, 0.01, 0.01, 0.01);

    // 创建OpenCV对比结果
    Mat opencv_result;
    equalizeHist(src, opencv_result);

    // 计算对比度增强效果
    Mat gray_src, gray_result, gray_opencv;
    cvtColor(src, gray_src, COLOR_BGR2GRAY);
    cvtColor(dst_retinex, gray_result, COLOR_BGR2GRAY);
    cvtColor(opencv_result, gray_opencv, COLOR_BGR2GRAY);

    Scalar mean_src, std_src, mean_result, std_result, mean_opencv, std_opencv;
    meanStdDev(gray_src, mean_src, std_src);
    meanStdDev(gray_result, mean_result, std_result);
    meanStdDev(gray_opencv, mean_opencv, std_opencv);

    cout << "Original - Mean: " << mean_src[0] << ", Std: " << std_src[0] << endl;
    cout << "Retinex MSRCR - Mean: " << mean_result[0] << ", Std: " << std_result[0] << endl;
    cout << "OpenCV Histogram Equalization - Mean: " << mean_opencv[0] << ", Std: " << std_opencv[0] << endl;
    cout << "Contrast Enhancement Ratio (Retinex): " << std_result[0] / std_src[0] << endl;
    cout << "Contrast Enhancement Ratio (OpenCV): " << std_opencv[0] / std_src[0] << endl;

    // 计算PSNR
    Mat diff;
    // Ensure both images have the same size for absdiff
    Mat dst_retinex_resized;
    resize(dst_retinex, dst_retinex_resized, src.size());
    absdiff(src, dst_retinex_resized, diff);
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

    double similarity_retinex = compareHist(hist_src, hist_result, HISTCMP_CORREL);
    double similarity_opencv = compareHist(hist_src, hist_opencv, HISTCMP_CORREL);

    cout << "Histogram Similarity (Retinex): " << fixed << setprecision(3) << similarity_retinex << endl;
    cout << "Histogram Similarity (OpenCV): " << similarity_opencv << endl;

    // 计算动态范围
    double min_val_src, max_val_src, min_val_result, max_val_result;
    minMaxLoc(gray_src, &min_val_src, &max_val_src);
    minMaxLoc(gray_result, &min_val_result, &max_val_result);

    cout << "Original Dynamic Range: " << (max_val_src - min_val_src) << endl;
    cout << "Processed Dynamic Range: " << (max_val_result - min_val_result) << endl;
}

int main(int argc, char** argv) {
    cout << "=== Retinex MSRCR Algorithm Test ===" << endl;

    // 创建输出目录
    create_output_directories();

    string image_path = (argc > 1) ? argv[1] : "assets/imori.jpg";
    Mat src = imread(image_path);
    if (src.empty()) {
        cerr << "Error: Cannot load image " << image_path << endl;
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
    cout << "All results saved to output/retinex_msrcr/ directory" << endl;

    return 0;
}
