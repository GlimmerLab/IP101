#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <filesystem>

// 包含运动模糊效果算法头文件
#include "advanced/effects/motion_blur.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    cout << "=== 运动模糊效果算法测试 ===" << endl;

    // 加载测试图像
    string image_path = (argc > 1) ? argv[1] : "assets/imori.jpg";
    Mat src = imread(image_path);

    if (src.empty()) {
        cerr << "错误：无法加载图像 " << image_path << endl;
        return -1;
    }

    cout << "图像尺寸: " << src.size() << endl;

    // 创建输出目录
    filesystem::create_directories("output/motion_blur");

    // ==================== 性能测试 ====================
    cout << "\n--- 性能测试 ---" << endl;

    Mat dst_motion_blur, dst_directional, dst_radial, dst_rotational, dst_zoom;

    // 测试基本运动模糊算法
    ip101::advanced::MotionBlurParams params;
    params.size = 15;
    params.angle = 45.0;
    params.strength = 1.0;

    auto start = chrono::high_resolution_clock::now();
    ip101::advanced::motion_blur(src, dst_motion_blur, params);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "基本运动模糊算法耗时: " << duration.count() << " 微秒" << endl;

    // 测试方向性运动模糊
    start = chrono::high_resolution_clock::now();
    ip101::advanced::directional_motion_blur(src, dst_directional, 15, 45.0, 1.0);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "方向性运动模糊算法耗时: " << duration.count() << " 微秒" << endl;

    // 测试径向运动模糊
    start = chrono::high_resolution_clock::now();
    ip101::advanced::radial_motion_blur(src, dst_radial, 0.5);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "径向运动模糊算法耗时: " << duration.count() << " 微秒" << endl;

    // 测试旋转运动模糊
    start = chrono::high_resolution_clock::now();
    ip101::advanced::rotational_motion_blur(src, dst_rotational, 0.5);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "旋转运动模糊算法耗时: " << duration.count() << " 微秒" << endl;

    // 测试缩放运动模糊
    start = chrono::high_resolution_clock::now();
    ip101::advanced::zoom_motion_blur(src, dst_zoom, 0.5);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "缩放运动模糊算法耗时: " << duration.count() << " 微秒" << endl;

    // ==================== 参数效果测试 ====================
    cout << "\n--- 参数效果测试 ---" << endl;

    // 测试不同模糊核大小
    vector<int> sizes = {5, 10, 15, 20, 25};
    vector<double> angles = {0.0, 45.0, 90.0, 135.0, 180.0};

    for (int size : sizes) {
        for (double angle : angles) {
            Mat result;
            ip101::advanced::directional_motion_blur(src, result, size, angle, 1.0);
            string filename = "output/motion_blur/size" + to_string(size) +
                            "_angle" + to_string(angle).substr(0, 3) + ".jpg";
            imwrite(filename, result);
        }
    }

    // 测试不同模糊强度
    vector<double> strengths = {0.2, 0.5, 0.8, 1.0, 1.5};

    for (double strength : strengths) {
        Mat result;
        ip101::advanced::directional_motion_blur(src, result, 15, 45.0, strength);
        string filename = "output/motion_blur/strength" + to_string(strength).substr(0, 3) + ".jpg";
        imwrite(filename, result);
    }

    // 测试不同中心点的径向模糊
    vector<Point2f> centers = {
        Point2f(src.cols/4, src.rows/4),
        Point2f(src.cols/2, src.rows/2),
        Point2f(3*src.cols/4, 3*src.rows/4)
    };

    for (size_t i = 0; i < centers.size(); i++) {
        Mat result;
        ip101::advanced::radial_motion_blur(src, result, 0.5, centers[i]);
        string filename = "output/motion_blur/radial_center" + to_string(i) + ".jpg";
        imwrite(filename, result);
    }

    // ==================== 可视化结果 ====================
    cout << "\n--- 可视化结果 ---" << endl;

    // 创建对比图像
    vector<Mat> images = {src, dst_motion_blur, dst_directional, dst_radial, dst_rotational, dst_zoom};
    vector<string> titles = {"Original", "Basic Motion Blur", "Directional Blur", "Radial Blur", "Rotational Blur", "Zoom Blur"};

    Mat comparison;
    hconcat(images, comparison);

    // 添加标题
    int font_face = FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.6;
    int thickness = 2;
    Scalar color(255, 255, 255);

    for (size_t i = 0; i < titles.size(); i++) {
        int x = i * src.cols + 10;
        putText(comparison, titles[i], Point(x, 30), font_face, font_scale, color, thickness);
    }

    imwrite("output/motion_blur/comparison.jpg", comparison);
    cout << "Comparison image saved to: output/motion_blur/comparison.jpg" << endl;

    // ==================== 特殊场景测试 ====================
    cout << "\n--- 特殊场景测试 ---" << endl;

    // 创建包含强边缘的测试图像
    Mat edge_test = Mat::zeros(src.size(), CV_8UC3);
    // 添加水平线
    line(edge_test, Point(0, src.rows/3), Point(src.cols, src.rows/3), Scalar(255, 255, 255), 3);
    line(edge_test, Point(0, 2*src.rows/3), Point(src.cols, 2*src.rows/3), Scalar(255, 255, 255), 3);
    // 添加垂直线
    line(edge_test, Point(src.cols/3, 0), Point(src.cols/3, src.rows), Scalar(255, 255, 255), 3);
    line(edge_test, Point(2*src.cols/3, 0), Point(2*src.cols/3, src.rows), Scalar(255, 255, 255), 3);

    imwrite("output/motion_blur/edge_test.jpg", edge_test);

    // 应用不同方向的运动模糊
    Mat edge_horizontal, edge_vertical, edge_diagonal;
    ip101::advanced::directional_motion_blur(edge_test, edge_horizontal, 20, 0.0, 1.0);
    ip101::advanced::directional_motion_blur(edge_test, edge_vertical, 20, 90.0, 1.0);
    ip101::advanced::directional_motion_blur(edge_test, edge_diagonal, 20, 45.0, 1.0);

    imwrite("output/motion_blur/edge_horizontal.jpg", edge_horizontal);
    imwrite("output/motion_blur/edge_vertical.jpg", edge_vertical);
    imwrite("output/motion_blur/edge_diagonal.jpg", edge_diagonal);

    // 测试高对比度图像
    Mat high_contrast = src.clone();
    high_contrast.convertTo(high_contrast, -1, 1.5, -30);
    imwrite("output/motion_blur/high_contrast.jpg", high_contrast);

    Mat high_contrast_result;
    ip101::advanced::directional_motion_blur(high_contrast, high_contrast_result, 15, 45.0, 1.0);
    imwrite("output/motion_blur/high_contrast_result.jpg", high_contrast_result);

    // ==================== 质量评估 ====================
    cout << "\n--- 质量评估 ---" << endl;

    // 计算模糊效果
    Mat gray_src, gray_result;
    cvtColor(src, gray_src, COLOR_BGR2GRAY);
    cvtColor(dst_motion_blur, gray_result, COLOR_BGR2GRAY);

    // 计算拉普拉斯算子响应（清晰度度量）
    Mat laplacian_src, laplacian_result;
    Laplacian(gray_src, laplacian_src, CV_64F);
    Laplacian(gray_result, laplacian_result, CV_64F);

    Scalar mean_lap_src, std_lap_src, mean_lap_result, std_lap_result;
    meanStdDev(laplacian_src, mean_lap_src, std_lap_src);
    meanStdDev(laplacian_result, mean_lap_result, std_lap_result);

    cout << "Original Laplacian response - Mean: " << mean_lap_src[0] << ", Std Dev: " << std_lap_src[0] << endl;
    cout << "Motion blur Laplacian response - Mean: " << mean_lap_result[0] << ", Std Dev: " << std_lap_result[0] << endl;
    cout << "Sharpness reduction factor: " << std_lap_src[0] / std_lap_result[0] << endl;

    // 计算PSNR
    Mat diff;
    absdiff(src, dst_motion_blur, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);

    double mse = mean(diff)[0];
    double psnr = 10.0 * log10((255.0 * 255.0) / mse);
    cout << "PSNR: " << fixed << setprecision(2) << psnr << " dB" << endl;

    // ==================== 直方图分析 ====================
    cout << "\n--- 直方图分析 ---" << endl;

    // 计算原图和模糊后图像的直方图
    vector<Mat> bgr_planes;
    split(src, bgr_planes);

    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};

    Mat b_hist, g_hist, r_hist;
    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange);

    // 计算模糊后图像的直方图
    vector<Mat> bgr_planes_result;
    split(dst_motion_blur, bgr_planes_result);

    Mat b_hist_result, g_hist_result, r_hist_result;
    calcHist(&bgr_planes_result[0], 1, 0, Mat(), b_hist_result, 1, &histSize, &histRange);
    calcHist(&bgr_planes_result[1], 1, 0, Mat(), g_hist_result, 1, &histSize, &histRange);
    calcHist(&bgr_planes_result[2], 1, 0, Mat(), r_hist_result, 1, &histSize, &histRange);

    // 计算直方图相似度
    double b_similarity = compareHist(b_hist, b_hist_result, HISTCMP_CORREL);
    double g_similarity = compareHist(g_hist, g_hist_result, HISTCMP_CORREL);
    double r_similarity = compareHist(r_hist, r_hist_result, HISTCMP_CORREL);

    cout << "Histogram similarity - B: " << fixed << setprecision(3) << b_similarity
         << ", G: " << g_similarity << ", R: " << r_similarity << endl;

    // ==================== 运动模糊核分析 ====================
    cout << "\n--- 运动模糊核分析 ---" << endl;

    // 创建不同参数的运动模糊核
    vector<int> kernel_sizes = {5, 10, 15};
    vector<double> kernel_angles = {0.0, 45.0, 90.0};

    for (int size : kernel_sizes) {
        for (double angle : kernel_angles) {
            Mat kernel = ip101::advanced::create_motion_blur_kernel(size, angle);

            // 归一化核用于可视化
            Mat kernel_vis;
            kernel.convertTo(kernel_vis, CV_8U, 255.0);
            applyColorMap(kernel_vis, kernel_vis, COLORMAP_JET);

            string filename = "output/motion_blur/kernel_size" + to_string(size) +
                            "_angle" + to_string(angle).substr(0, 3) + ".jpg";
            imwrite(filename, kernel_vis);
        }
    }

    // ==================== 保存结果 ====================
    cout << "\n--- 保存结果 ---" << endl;

    imwrite("output/motion_blur/original.jpg", src);
    imwrite("output/motion_blur/motion_blur_result.jpg", dst_motion_blur);
    imwrite("output/motion_blur/directional_result.jpg", dst_directional);
    imwrite("output/motion_blur/radial_result.jpg", dst_radial);
    imwrite("output/motion_blur/rotational_result.jpg", dst_rotational);
    imwrite("output/motion_blur/zoom_result.jpg", dst_zoom);

    cout << "All results have been saved to output/motion_blur/ directory" << endl;
    cout << "Motion blur effect algorithm test completed!" << endl;

    return 0;
}
