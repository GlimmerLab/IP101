#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include "advanced/effects/skin_beauty.hpp"

using namespace cv;
using namespace std;

// 创建输出目录
void create_output_directories() {
    filesystem::create_directories("output/skin_beauty");
}

// 性能测试
void performance_test(const Mat& src) {
    cout << "=== Skin Beauty Performance Test ===" << endl;

    Mat dst;
    int iterations = 10;

    // 测试IP101算法性能
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        ip101::advanced::SkinBeautyParams params;
        params.smoothing_factor = 0.8;
        params.whitening_factor = 0.6;
        params.detail_factor = 0.4;
        ip101::advanced::skin_beauty(src, dst, params);
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    double avg_time = duration.count() / (double)iterations;
    double fps = 1000000.0 / avg_time;

    cout << "IP101 Skin Beauty - Average Time: " << fixed << setprecision(2)
         << avg_time / 1000.0 << " ms, FPS: " << fps << endl;

    // 测试OpenCV对比算法（使用双边滤波作为对比）
    Mat opencv_result;
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        bilateralFilter(src, opencv_result, 15, 80, 80);
    }
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);

    avg_time = duration.count() / (double)iterations;
    fps = 1000000.0 / avg_time;

    cout << "OpenCV Bilateral Filter - Average Time: " << fixed << setprecision(2)
         << avg_time / 1000.0 << " ms, FPS: " << fps << endl;
}

// 参数效果测试
void parameter_effect_test(const Mat& src) {
    cout << "\n--- Parameter Effect Test ---" << endl;

    vector<double> smooth_strengths = {0.3, 0.5, 0.7, 0.8, 0.9};
    vector<double> whiten_strengths = {0.2, 0.4, 0.6, 0.8, 1.0};
    vector<double> lighting_strengths = {0.1, 0.3, 0.5, 0.7, 0.9};

    // Test smoothing strength parameters
    cout << "Testing smoothing strength parameter effects..." << endl;
    for (size_t i = 0; i < smooth_strengths.size(); i++) {
        Mat result;
        ip101::advanced::SkinBeautyParams params;
        params.smoothing_factor = smooth_strengths[i];
        params.whitening_factor = 0.6;
        params.detail_factor = 0.4;
        ip101::advanced::skin_beauty(src, result, params);
        string filename = "output/skin_beauty/smooth_" + to_string(i) + ".jpg";
        imwrite(filename, result);
    }

    // Test whitening strength parameters
    cout << "Testing whitening strength parameter effects..." << endl;
    for (size_t i = 0; i < whiten_strengths.size(); i++) {
        Mat result;
        ip101::advanced::SkinBeautyParams params;
        params.smoothing_factor = 0.8;
        params.whitening_factor = whiten_strengths[i];
        params.detail_factor = 0.4;
        ip101::advanced::skin_beauty(src, result, params);
        string filename = "output/skin_beauty/whiten_" + to_string(i) + ".jpg";
        imwrite(filename, result);
    }

    // Test lighting enhancement parameters
    cout << "Testing lighting enhancement parameter effects..." << endl;
    for (size_t i = 0; i < lighting_strengths.size(); i++) {
        Mat result;
        ip101::advanced::SkinBeautyParams params;
        params.smoothing_factor = 0.8;
        params.whitening_factor = 0.6;
        params.detail_factor = lighting_strengths[i];
        ip101::advanced::skin_beauty(src, result, params);
        string filename = "output/skin_beauty/lighting_" + to_string(i) + ".jpg";
        imwrite(filename, result);
    }
}

// 可视化测试
void visualization_test(const Mat& src) {
    cout << "\n--- Visualization Test ---" << endl;

    // 应用皮肤美化
    Mat dst_skin_beauty;
    ip101::advanced::SkinBeautyParams params;
    params.smoothing_factor = 0.8;
    params.whitening_factor = 0.6;
    params.detail_factor = 0.4;
    ip101::advanced::skin_beauty(src, dst_skin_beauty, params);

    // 保存结果
    imwrite("output/skin_beauty/original.jpg", src);
    imwrite("output/skin_beauty/beautified.jpg", dst_skin_beauty);

    // 创建OpenCV对比结果
    Mat opencv_result;
    bilateralFilter(src, opencv_result, 15, 80, 80);
    imwrite("output/skin_beauty/opencv_comparison.jpg", opencv_result);

    // 创建对比图像
    // Ensure all images have the same size for hconcat
    Mat dst_skin_beauty_resized, opencv_result_resized;
    resize(dst_skin_beauty, dst_skin_beauty_resized, src.size());
    resize(opencv_result, opencv_result_resized, src.size());

    vector<Mat> images = {src, dst_skin_beauty_resized, opencv_result_resized};
    Mat comparison;
    hconcat(images, comparison);

    // 添加标题
    vector<string> titles = {"Original", "Skin Beauty", "OpenCV Comparison"};
    int font_face = FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.8;
    Scalar color(255, 255, 255);
    int thickness = 2;

    for (size_t i = 0; i < titles.size(); i++) {
        int x = i * src.cols + 10;
        putText(comparison, titles[i], Point(x, 30), font_face, font_scale, color, thickness);
    }

    imwrite("output/skin_beauty/comparison.jpg", comparison);
    cout << "Comparison image saved to: output/skin_beauty/comparison.jpg" << endl;
}

// 特殊场景测试
void special_scenario_test(const Mat& src) {
    cout << "\n--- Special Scenario Test ---" << endl;

    // 创建暗图像
    Mat dark = src.clone();
    dark.convertTo(dark, -1, 0.5, 0);
    imwrite("output/skin_beauty/dark.jpg", dark);

    // 应用皮肤美化
    Mat dark_result;
    ip101::advanced::SkinBeautyParams params;
    params.smoothing_factor = 0.8;
    params.whitening_factor = 0.6;
    params.detail_factor = 0.4;
    ip101::advanced::skin_beauty(dark, dark_result, params);
    imwrite("output/skin_beauty/dark_result.jpg", dark_result);

    // 创建过曝图像
    Mat overexposed = src.clone();
    overexposed.convertTo(overexposed, -1, 1.5, 50);
    imwrite("output/skin_beauty/overexposed.jpg", overexposed);

    // 应用皮肤美化
    Mat overexposed_result;
    ip101::advanced::SkinBeautyParams params_overexposed;
    params_overexposed.smoothing_factor = 0.8;
    params_overexposed.whitening_factor = 0.6;
    params_overexposed.detail_factor = 0.4;
    ip101::advanced::skin_beauty(overexposed, overexposed_result, params_overexposed);
    imwrite("output/skin_beauty/overexposed_result.jpg", overexposed_result);

    // 创建噪声图像
    Mat noisy = src.clone();
    Mat noise = Mat::zeros(noisy.size(), CV_8UC3);
    randn(noise, 0, 15);
    noisy = noisy + noise;
    imwrite("output/skin_beauty/noisy.jpg", noisy);

    // 应用皮肤美化
    Mat noisy_result;
    ip101::advanced::SkinBeautyParams params_noisy;
    params_noisy.smoothing_factor = 0.8;
    params_noisy.whitening_factor = 0.6;
    params_noisy.detail_factor = 0.4;
    ip101::advanced::skin_beauty(noisy, noisy_result, params_noisy);
    imwrite("output/skin_beauty/noisy_result.jpg", noisy_result);
}

// 质量评估
void quality_assessment(const Mat& src) {
    cout << "\n--- Quality Assessment ---" << endl;

    // 应用皮肤美化
    Mat dst_skin_beauty;
    ip101::advanced::SkinBeautyParams params;
    params.smoothing_factor = 0.8;
    params.whitening_factor = 0.6;
    params.detail_factor = 0.4;
    ip101::advanced::skin_beauty(src, dst_skin_beauty, params);

    // 创建OpenCV对比结果
    Mat opencv_result;
    bilateralFilter(src, opencv_result, 15, 80, 80);

    // 计算平滑度效果
    Mat gray_src, gray_result, gray_opencv;
    cvtColor(src, gray_src, COLOR_BGR2GRAY);
    cvtColor(dst_skin_beauty, gray_result, COLOR_BGR2GRAY);
    cvtColor(opencv_result, gray_opencv, COLOR_BGR2GRAY);

    // 计算拉普拉斯算子响应（细节度量）
    Mat laplacian_src, laplacian_result, laplacian_opencv;
    Laplacian(gray_src, laplacian_src, CV_64F);
    Laplacian(gray_result, laplacian_result, CV_64F);
    Laplacian(gray_opencv, laplacian_opencv, CV_64F);

    Scalar mean_lap_src, std_lap_src, mean_lap_result, std_lap_result, mean_lap_opencv, std_lap_opencv;
    meanStdDev(laplacian_src, mean_lap_src, std_lap_src);
    meanStdDev(laplacian_result, mean_lap_result, std_lap_result);
    meanStdDev(laplacian_opencv, mean_lap_opencv, std_lap_opencv);

    cout << "Original Laplacian Response - Mean: " << mean_lap_src[0] << ", Std: " << std_lap_src[0] << endl;
    cout << "Skin Beauty Laplacian Response - Mean: " << mean_lap_result[0] << ", Std: " << std_lap_result[0] << endl;
    cout << "OpenCV Bilateral Filter Laplacian Response - Mean: " << mean_lap_opencv[0] << ", Std: " << std_lap_opencv[0] << endl;
    cout << "Smoothness (Skin Beauty): " << std_lap_result[0] / std_lap_src[0] << endl;
    cout << "Smoothness (OpenCV): " << std_lap_opencv[0] / std_lap_src[0] << endl;

    // 计算PSNR
    Mat diff;
    // Ensure both images have the same size for absdiff
    Mat dst_skin_beauty_resized;
    resize(dst_skin_beauty, dst_skin_beauty_resized, src.size());
    absdiff(src, dst_skin_beauty_resized, diff);
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

    double similarity_beauty = compareHist(hist_src, hist_result, HISTCMP_CORREL);
    double similarity_opencv = compareHist(hist_src, hist_opencv, HISTCMP_CORREL);

    cout << "Histogram Similarity (Skin Beauty): " << fixed << setprecision(3) << similarity_beauty << endl;
    cout << "Histogram Similarity (OpenCV): " << similarity_opencv << endl;

    // 计算颜色保持度
    Scalar mean_src, std_src, mean_result, std_result;
    meanStdDev(src, mean_src, std_src);
    meanStdDev(dst_skin_beauty, mean_result, std_result);

    cout << "Color Preservation - B: " << abs(mean_result[0] - mean_src[0]) / mean_src[0] * 100 << "%" << endl;
    cout << "Color Preservation - G: " << abs(mean_result[1] - mean_src[1]) / mean_src[1] * 100 << "%" << endl;
    cout << "Color Preservation - R: " << abs(mean_result[2] - mean_src[2]) / mean_src[2] * 100 << "%" << endl;

    // 计算亮度变化
    Scalar mean_gray_src, mean_gray_result;
    meanStdDev(gray_src, mean_gray_src, std_src);
    meanStdDev(gray_result, mean_gray_result, std_result);

    cout << "Brightness Change: " << (mean_gray_result[0] - mean_gray_src[0]) / mean_gray_src[0] * 100 << "%" << endl;
}

int main(int argc, char** argv) {
    cout << "=== Skin Beauty Algorithm Test ===" << endl;

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
    cout << "All results saved to output/skin_beauty/ directory" << endl;

    return 0;
}
