#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include "advanced/effects/spherize.hpp"

using namespace cv;
using namespace std;

// 创建输出目录
void create_output_directories() {
    filesystem::create_directories("output/spherize");
}

// 性能测试
void performance_test(const Mat& src) {
    cout << "\n--- Performance Test ---" << endl;

    Mat dst;
    int iterations = 10;

    // 测试IP101算法性能
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        ip101::advanced::SpherizeParams params;
        params.strength = 0.5;
        params.center = Point2f(src.cols/2, src.rows/2);
        ip101::advanced::spherize(src, dst, params);
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    double avg_time = duration.count() / (double)iterations;
    double fps = 1000000.0 / avg_time;

    cout << "IP101 Spherize Effect - Average Time: " << fixed << setprecision(2)
         << avg_time / 1000.0 << " ms, FPS: " << fps << endl;

    // 测试OpenCV对比算法（使用仿射变换作为对比）
    Mat opencv_result;
    Mat transform_matrix = getRotationMatrix2D(Point2f(src.cols/2, src.rows/2), 0, 1.0);
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        warpAffine(src, opencv_result, transform_matrix, src.size());
    }
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);

    avg_time = duration.count() / (double)iterations;
    fps = 1000000.0 / avg_time;

    cout << "OpenCV Affine Transform - Average Time: " << fixed << setprecision(2)
         << avg_time / 1000.0 << " ms, FPS: " << fps << endl;
}

// 参数效果测试
void parameter_effect_test(const Mat& src) {
    cout << "\n--- Parameter Effect Test ---" << endl;

    vector<double> strength_values = {0.1, 0.3, 0.5, 0.7, 0.9};
    vector<Point2f> center_points = {
        Point2f(src.cols/4, src.rows/4),
        Point2f(src.cols/2, src.rows/4),
        Point2f(3*src.cols/4, src.rows/4),
        Point2f(src.cols/4, src.rows/2),
        Point2f(src.cols/2, src.rows/2)
    };

    // 测试强度参数
    cout << "Testing strength parameter effects..." << endl;
    for (size_t i = 0; i < strength_values.size(); i++) {
        Mat result;
        ip101::advanced::SpherizeParams params;
        params.strength = strength_values[i];
        params.center = Point2f(src.cols/2, src.rows/2);
        ip101::advanced::spherize(src, result, params);
        string filename = "output/spherize/strength_" + to_string(i) + ".jpg";
        imwrite(filename, result);
    }

    // 测试中心点参数
    cout << "Testing center point parameter effects..." << endl;
    for (size_t i = 0; i < center_points.size(); i++) {
        Mat result;
        ip101::advanced::SpherizeParams params;
        params.strength = 0.5;
        params.center = center_points[i];
        ip101::advanced::spherize(src, result, params);
        string filename = "output/spherize/center_" + to_string(i) + ".jpg";
        imwrite(filename, result);
    }
}

// 可视化测试
void visualization_test(const Mat& src) {
    cout << "\n--- Visualization Test ---" << endl;

    // 应用球面化效果
    Mat dst_spherize;
    ip101::advanced::SpherizeParams params;
    params.strength = 0.5;
    params.center = Point2f(src.cols/2, src.rows/2);
    ip101::advanced::spherize(src, dst_spherize, params);

    // 保存结果
    imwrite("output/spherize/original.jpg", src);
    imwrite("output/spherize/spherized.jpg", dst_spherize);

    // 创建OpenCV对比结果
    Mat opencv_result;
    Mat transform_matrix = getRotationMatrix2D(Point2f(src.cols/2, src.rows/2), 0, 1.0);
    warpAffine(src, opencv_result, transform_matrix, src.size());
    imwrite("output/spherize/opencv_comparison.jpg", opencv_result);

    // 创建对比图像
    // Ensure all images have the same size for hconcat
    Mat dst_spherize_resized, opencv_result_resized;
    resize(dst_spherize, dst_spherize_resized, src.size());
    resize(opencv_result, opencv_result_resized, src.size());

    vector<Mat> images = {src, dst_spherize_resized, opencv_result_resized};
    Mat comparison;
    hconcat(images, comparison);

    // 添加标题
    vector<string> titles = {"Original", "Spherize Effect", "OpenCV Comparison"};
    int font_face = FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.8;
    Scalar color(255, 255, 255);
    int thickness = 2;

    for (size_t i = 0; i < titles.size(); i++) {
        int x = i * src.cols + 10;
        putText(comparison, titles[i], Point(x, 30), font_face, font_scale, color, thickness);
    }

    imwrite("output/spherize/comparison.jpg", comparison);
    cout << "Comparison image saved to: output/spherize/comparison.jpg" << endl;
}

// 特殊场景测试
void special_scenario_test(const Mat& src) {
    cout << "\n--- Special Scenario Test ---" << endl;

    // 测试不同强度级别
    vector<double> strengths = {0.2, 0.5, 0.8};
    vector<string> effect_names = {"Light Spherize", "Medium Spherize", "Strong Spherize"};

    for (size_t i = 0; i < strengths.size(); i++) {
        Mat result;
        ip101::advanced::SpherizeParams params;
        params.strength = strengths[i];
        params.center = Point2f(src.cols/2, src.rows/2);
        ip101::advanced::spherize(src, result, params);
        string filename = "output/spherize/" + effect_names[i] + ".jpg";
        imwrite(filename, result);
    }

    // 测试不同中心点位置
    vector<Point2f> centers = {
        Point2f(0, 0),                    // 左上角
        Point2f(src.cols, 0),             // 右上角
        Point2f(0, src.rows),             // 左下角
        Point2f(src.cols, src.rows)       // 右下角
    };
    vector<string> center_names = {"Top Left", "Top Right", "Bottom Left", "Bottom Right"};

    for (size_t i = 0; i < centers.size(); i++) {
        Mat result;
        ip101::advanced::SpherizeParams params;
        params.strength = 0.5;
        params.center = centers[i];
        ip101::advanced::spherize(src, result, params);
        string filename = "output/spherize/center_" + center_names[i] + ".jpg";
        imwrite(filename, result);
    }
}

// 质量评估
void quality_assessment(const Mat& src) {
    cout << "\n--- Quality Assessment ---" << endl;

    // 应用球面化效果
    Mat dst_spherize;
    ip101::advanced::SpherizeParams params;
    params.strength = 0.5;
    params.center = Point2f(src.cols/2, src.rows/2);
    ip101::advanced::spherize(src, dst_spherize, params);

    // 创建OpenCV对比结果
    Mat opencv_result;
    Mat transform_matrix = getRotationMatrix2D(Point2f(src.cols/2, src.rows/2), 0, 1.0);
    warpAffine(src, opencv_result, transform_matrix, src.size());

    // 计算几何变形程度
    Mat gray_src, gray_result, gray_opencv;
    cvtColor(src, gray_src, COLOR_BGR2GRAY);
    cvtColor(dst_spherize, gray_result, COLOR_BGR2GRAY);
    cvtColor(opencv_result, gray_opencv, COLOR_BGR2GRAY);

    // 计算边缘密度变化
    Mat edges_src, edges_result, edges_opencv;
    Canny(gray_src, edges_src, 50, 150);
    Canny(gray_result, edges_result, 50, 150);
    Canny(gray_opencv, edges_opencv, 50, 150);

    Scalar mean_edges_src, std_edges_src, mean_edges_result, std_edges_result;
    meanStdDev(edges_src, mean_edges_src, std_edges_src);
    meanStdDev(edges_result, mean_edges_result, std_edges_result);

    cout << "Original edge density: " << mean_edges_src[0] << endl;
    cout << "Spherized edge density: " << mean_edges_result[0] << endl;
    cout << "Edge density change rate: " << (mean_edges_result[0] - mean_edges_src[0]) / mean_edges_src[0] * 100 << "%" << endl;

    // 计算PSNR
    Mat diff;
    // Ensure both images have the same size for absdiff
    Mat dst_spherize_resized;
    resize(dst_spherize, dst_spherize_resized, src.size());
    absdiff(src, dst_spherize_resized, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);

    double mse = mean(diff)[0];
    double psnr = 10.0 * log10((255.0 * 255.0) / mse);
    cout << "PSNR: " << fixed << setprecision(2) << psnr << " dB" << endl;

    // 计算直方图相似度
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};

    Mat hist_src, hist_result;
    calcHist(&gray_src, 1, 0, Mat(), hist_src, 1, &histSize, &histRange);
    calcHist(&gray_result, 1, 0, Mat(), hist_result, 1, &histSize, &histRange);

    double similarity = compareHist(hist_src, hist_result, HISTCMP_CORREL);
    cout << "Histogram similarity: " << fixed << setprecision(3) << similarity << endl;

    // 计算颜色保持度
    Scalar mean_src, std_src, mean_result, std_result;
    meanStdDev(src, mean_src, std_src);
    meanStdDev(dst_spherize, mean_result, std_result);

    cout << "Color Preservation - B: " << abs(mean_result[0] - mean_src[0]) / mean_src[0] * 100 << "%" << endl;
    cout << "Color Preservation - G: " << abs(mean_result[1] - mean_src[1]) / mean_src[1] * 100 << "%" << endl;
    cout << "Color Preservation - R: " << abs(mean_result[2] - mean_src[2]) / mean_src[2] * 100 << "%" << endl;

    // 计算几何变形度量
    Mat laplacian_src, laplacian_result;
    Laplacian(gray_src, laplacian_src, CV_64F);
    Laplacian(gray_result, laplacian_result, CV_64F);

    Scalar mean_lap_src, std_lap_src, mean_lap_result, std_lap_result;
    meanStdDev(laplacian_src, mean_lap_src, std_lap_src);
    meanStdDev(laplacian_result, mean_lap_result, std_lap_result);

    cout << "Geometric Distortion Degree: " << abs(std_lap_result[0] - std_lap_src[0]) / std_lap_src[0] * 100 << "%" << endl;
}

int main(int argc, char** argv) {
    cout << "=== Spherize Effect Algorithm Test ===" << endl;

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
    cout << "All results have been saved to output/spherize/ directory" << endl;

    return 0;
}
