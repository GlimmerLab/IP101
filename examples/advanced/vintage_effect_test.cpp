#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include "advanced/effects/vintage_effect.hpp"

using namespace cv;
using namespace std;

// 创建输出目录
void create_output_directories() {
    filesystem::create_directories("output/vintage_effect");
}

// 性能测试
void performance_test(const Mat& src) {
    cout << "=== 复古效果性能测试 ===" << endl;

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

    cout << "IP101复古效果 - 平均时间: " << fixed << setprecision(2)
         << avg_time / 1000.0 << " ms, FPS: " << fps << endl;

    // 测试OpenCV对比算法（使用颜色映射作为对比）
    Mat opencv_result;
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        applyColorMap(src, opencv_result, COLORMAP_OCEAN);
    }
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);

    avg_time = duration.count() / (double)iterations;
    fps = 1000000.0 / avg_time;

    cout << "OpenCV颜色映射 - 平均时间: " << fixed << setprecision(2)
         << avg_time / 1000.0 << " ms, FPS: " << fps << endl;
}

// 参数效果测试
void parameter_effect_test(const Mat& src) {
    cout << "\n--- 参数效果测试 ---" << endl;

    vector<double> sepia_strengths = {0.3, 0.5, 0.7, 0.8, 0.9};
    vector<double> grain_strengths = {0.2, 0.4, 0.6, 0.8, 1.0};
    vector<double> vignette_strengths = {0.1, 0.3, 0.5, 0.7, 0.9};
    vector<double> scratch_strengths = {0.1, 0.3, 0.5, 0.7, 0.9};

    // 测试棕褐色调参数
    cout << "测试棕褐色调参数效果..." << endl;
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

    // 测试颗粒感参数
    cout << "测试颗粒感参数效果..." << endl;
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

    // 测试暗角效果参数
    cout << "测试暗角效果参数效果..." << endl;
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

    // 测试划痕效果参数
    cout << "测试划痕效果参数效果..." << endl;
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

// 可视化测试
void visualization_test(const Mat& src) {
    cout << "\n--- 可视化测试 ---" << endl;

    // 应用复古效果
    Mat dst_vintage;
    ip101::advanced::VintageParams params_viz;
    params_viz.sepia_intensity = 0.8;
    params_viz.noise_level = 0.6;
    params_viz.vignette_strength = 0.4;
    params_viz.scratch_intensity = 0.3;
    ip101::advanced::vintage_effect(src, dst_vintage, params_viz);

    // 保存结果
    imwrite("output/vintage_effect/original.jpg", src);
    imwrite("output/vintage_effect/vintage.jpg", dst_vintage);

    // 创建OpenCV对比结果
    Mat opencv_result;
    applyColorMap(src, opencv_result, COLORMAP_OCEAN);
    imwrite("output/vintage_effect/opencv_comparison.jpg", opencv_result);

    // 创建对比图像
    vector<Mat> images = {src, dst_vintage, opencv_result};
    Mat comparison;
    hconcat(images, comparison);

    // 添加标题
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

// 特殊场景测试
void special_scenario_test(const Mat& src) {
    cout << "\n--- 特殊场景测试 ---" << endl;

    // 测试不同复古风格
    vector<vector<double>> vintage_styles = {
        {0.9, 0.8, 0.6, 0.5},  // 强烈复古
        {0.7, 0.5, 0.3, 0.2},  // 中等复古
        {0.5, 0.3, 0.2, 0.1},  // 轻微复古
        {0.3, 0.2, 0.1, 0.05}  // 淡雅复古
    };
    vector<string> style_names = {"强烈复古", "中等复古", "轻微复古", "淡雅复古"};

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

    // 测试不同光照条件下的效果
    // 暗图像
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

    // 亮图像
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

// 质量评估
void quality_assessment(const Mat& src) {
    cout << "\n--- 质量评估 ---" << endl;

    // 应用复古效果
    Mat dst_vintage;
    ip101::advanced::VintageParams params_qa;
    params_qa.sepia_intensity = 0.8;
    params_qa.noise_level = 0.6;
    params_qa.vignette_strength = 0.4;
    params_qa.scratch_intensity = 0.3;
    ip101::advanced::vintage_effect(src, dst_vintage, params_qa);

    // 创建OpenCV对比结果
    Mat opencv_result;
    applyColorMap(src, opencv_result, COLORMAP_OCEAN);

    // 计算颜色变化效果
    Mat gray_src, gray_result, gray_opencv;
    cvtColor(src, gray_src, COLOR_BGR2GRAY);
    cvtColor(dst_vintage, gray_result, COLOR_BGR2GRAY);
    cvtColor(opencv_result, gray_opencv, COLOR_BGR2GRAY);

    // 计算颜色统计
    Scalar mean_src, std_src, mean_result, std_result, mean_opencv, std_opencv;
    meanStdDev(src, mean_src, std_src);
    meanStdDev(dst_vintage, mean_result, std_result);
    meanStdDev(opencv_result, mean_opencv, std_opencv);

    cout << "原图颜色统计 - B: " << mean_src[0] << ", G: " << mean_src[1] << ", R: " << mean_src[2] << endl;
    cout << "复古效果颜色统计 - B: " << mean_result[0] << ", G: " << mean_result[1] << ", R: " << mean_result[2] << endl;
    cout << "OpenCV颜色映射统计 - B: " << mean_opencv[0] << ", G: " << mean_opencv[1] << ", R: " << mean_opencv[2] << endl;

    // 计算颜色变化程度
    cout << "颜色变化程度 - B: " << abs(mean_result[0] - mean_src[0]) / mean_src[0] * 100 << "%" << endl;
    cout << "颜色变化程度 - G: " << abs(mean_result[1] - mean_src[1]) / mean_src[1] * 100 << "%" << endl;
    cout << "颜色变化程度 - R: " << abs(mean_result[2] - mean_src[2]) / mean_src[2] * 100 << "%" << endl;

    // 计算PSNR
    Mat diff;
    absdiff(src, dst_vintage, diff);
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

    double similarity_vintage = compareHist(hist_src, hist_result, HISTCMP_CORREL);
    double similarity_opencv = compareHist(hist_src, hist_opencv, HISTCMP_CORREL);

    cout << "直方图相似度(复古效果): " << fixed << setprecision(3) << similarity_vintage << endl;
    cout << "直方图相似度(OpenCV): " << similarity_opencv << endl;

    // 计算对比度变化
    Scalar mean_gray_src, std_gray_src, mean_gray_result, std_gray_result;
    meanStdDev(gray_src, mean_gray_src, std_gray_src);
    meanStdDev(gray_result, mean_gray_result, std_gray_result);

    cout << "原图对比度: " << std_gray_src[0] << endl;
    cout << "复古效果对比度: " << std_gray_result[0] << endl;
    cout << "对比度变化率: " << (std_gray_result[0] - std_gray_src[0]) / std_gray_src[0] * 100 << "%" << endl;

    // 计算亮度变化
    cout << "亮度变化: " << (mean_gray_result[0] - mean_gray_src[0]) / mean_gray_src[0] * 100 << "%" << endl;

    // 计算饱和度变化（通过颜色通道标准差）
    double saturation_src = (std_src[0] + std_src[1] + std_src[2]) / 3.0;
    double saturation_result = (std_result[0] + std_result[1] + std_result[2]) / 3.0;

    cout << "饱和度变化: " << (saturation_result - saturation_src) / saturation_src * 100 << "%" << endl;
}

int main() {
    cout << "=== 复古效果算法测试 ===" << endl;

    // 创建输出目录
    create_output_directories();

    // 加载测试图像
    Mat src = imread("assets/imori.jpg");
    if (src.empty()) {
        cerr << "无法加载测试图像" << endl;
        return -1;
    }

    cout << "图像尺寸: " << src.cols << "x" << src.rows << endl;

    // 执行各项测试
    performance_test(src);
    parameter_effect_test(src);
    visualization_test(src);
    special_scenario_test(src);
    quality_assessment(src);

    cout << "\n=== 测试完成 ===" << endl;
    cout << "所有结果已保存到 output/vintage_effect/ 目录" << endl;

    return 0;
}
