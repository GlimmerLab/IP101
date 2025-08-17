#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include "advanced/detection/rectangle_detection.hpp"

using namespace cv;
using namespace std;

// 创建输出目录
void create_output_directories() {
    filesystem::create_directories("output/rectangle_detection");
}

// 性能测试
void performance_test(const Mat& src) {
    cout << "\n--- 性能测试 ---" << endl;

    vector<ip101::advanced::RectangleInfo> rectangles;
    int iterations = 10;

    // 测试IP101算法性能
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        rectangles.clear();
        ip101::advanced::detect_rectangles(src, rectangles, 1000.0, 100000.0, 0.5, 2.0, 0.8);
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    double avg_time = duration.count() / (double)iterations;
    double fps = 1000000.0 / avg_time;

    cout << "IP101 Rectangle Detection - Average Time: " << fixed << setprecision(2)
         << avg_time / 1000.0 << " ms, FPS: " << fps << endl;
    cout << "Detected " << rectangles.size() << " rectangles" << endl;

    // Test OpenCV comparison algorithm (using contour detection as comparison)
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        contours.clear();
        findContours(gray, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    }
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);

    avg_time = duration.count() / (double)iterations;
    fps = 1000000.0 / avg_time;

    cout << "OpenCV Contour Detection - Average Time: " << fixed << setprecision(2)
         << avg_time / 1000.0 << " ms, FPS: " << fps << endl;
    cout << "Detected " << contours.size() << " contours" << endl;
}

// 参数效果测试
void parameter_effect_test(const Mat& src) {
    cout << "\n--- Parameter Effect Test ---" << endl;

    vector<double> min_areas = {500.0, 1000.0, 2000.0, 5000.0, 10000.0};
    vector<double> max_areas = {50000.0, 100000.0, 200000.0, 500000.0, 1000000.0};
    vector<double> min_aspect_ratios = {0.3, 0.5, 0.7, 1.0, 1.5};
    vector<double> max_aspect_ratios = {1.5, 2.0, 3.0, 5.0, 10.0};
    vector<double> min_confidences = {0.5, 0.6, 0.7, 0.8, 0.9};

    // Test minimum area parameter
    cout << "Testing minimum area parameter effect..." << endl;
    for (size_t i = 0; i < min_areas.size(); i++) {
        vector<ip101::advanced::RectangleInfo> rectangles;
        ip101::advanced::detect_rectangles(src, rectangles, min_areas[i], 100000.0, 0.5, 2.0, 0.8);

        Mat result = src.clone();
        ip101::advanced::draw_rectangles(result, rectangles, Scalar(0, 255, 0), 2);
        string filename = "output/rectangle_detection/min_area_" + to_string(i) + ".jpg";
        imwrite(filename, result);
    }

    // Test maximum area parameter
    cout << "Testing maximum area parameter effect..." << endl;
    for (size_t i = 0; i < max_areas.size(); i++) {
        vector<ip101::advanced::RectangleInfo> rectangles;
        ip101::advanced::detect_rectangles(src, rectangles, 1000.0, max_areas[i], 0.5, 2.0, 0.8);

        Mat result = src.clone();
        ip101::advanced::draw_rectangles(result, rectangles, Scalar(0, 255, 0), 2);
        string filename = "output/rectangle_detection/max_area_" + to_string(i) + ".jpg";
        imwrite(filename, result);
    }

    // Test aspect ratio parameter
    cout << "Testing aspect ratio parameter effect..." << endl;
    for (size_t i = 0; i < min_aspect_ratios.size(); i++) {
        vector<ip101::advanced::RectangleInfo> rectangles;
        ip101::advanced::detect_rectangles(src, rectangles, 1000.0, 100000.0, min_aspect_ratios[i], 2.0, 0.8);

        Mat result = src.clone();
        ip101::advanced::draw_rectangles(result, rectangles, Scalar(0, 255, 0), 2);
        string filename = "output/rectangle_detection/aspect_ratio_" + to_string(i) + ".jpg";
        imwrite(filename, result);
    }

    // Test confidence parameter
    cout << "Testing confidence parameter effect..." << endl;
    for (size_t i = 0; i < min_confidences.size(); i++) {
        vector<ip101::advanced::RectangleInfo> rectangles;
        ip101::advanced::detect_rectangles(src, rectangles, 1000.0, 100000.0, 0.5, 2.0, min_confidences[i]);

        Mat result = src.clone();
        ip101::advanced::draw_rectangles(result, rectangles, Scalar(0, 255, 0), 2);
        string filename = "output/rectangle_detection/confidence_" + to_string(i) + ".jpg";
        imwrite(filename, result);
    }
}

// 可视化测试
void visualization_test(const Mat& src) {
    cout << "\n--- Visualization Test ---" << endl;

    // Apply rectangle detection
    vector<ip101::advanced::RectangleInfo> rectangles;
    ip101::advanced::detect_rectangles(src, rectangles, 1000.0, 100000.0, 0.5, 2.0, 0.8);

    // Save results
    imwrite("output/rectangle_detection/original.jpg", src);

    Mat result = src.clone();
    ip101::advanced::draw_rectangles(result, rectangles, Scalar(0, 255, 0), 2);
    imwrite("output/rectangle_detection/detected.jpg", result);

    // 创建OpenCV对比结果
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(gray, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat opencv_result = src.clone();
    for (const auto& contour : contours) {
        if (contourArea(contour) > 1000) {
            RotatedRect rect = minAreaRect(contour);
            Point2f rect_points[4];
            rect.points(rect_points);
            for (int j = 0; j < 4; j++) {
                line(opencv_result, rect_points[j], rect_points[(j + 1) % 4], Scalar(255, 0, 0), 2);
            }
        }
    }
    imwrite("output/rectangle_detection/opencv_comparison.jpg", opencv_result);

    // 创建对比图像
    vector<Mat> images = {src, result, opencv_result};
    Mat comparison;
    hconcat(images, comparison);

    // Add titles
    vector<string> titles = {"Original", "IP101 Rectangle Detection", "OpenCV Comparison"};
    int font_face = FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.8;
    Scalar color(255, 255, 255);
    int thickness = 2;

    for (size_t i = 0; i < titles.size(); i++) {
        int x = i * src.cols + 10;
        putText(comparison, titles[i], Point(x, 30), font_face, font_scale, color, thickness);
    }

    imwrite("output/rectangle_detection/comparison.jpg", comparison);
    cout << "Comparison image saved to: output/rectangle_detection/comparison.jpg" << endl;
    cout << "Detected " << rectangles.size() << " rectangles" << endl;
}

// 特殊场景测试
void special_scenario_test(const Mat& src) {
    cout << "\n--- Special Scenario Test ---" << endl;

    // Create noisy image
    Mat noisy = src.clone();
    Mat noise = Mat::zeros(noisy.size(), CV_8UC3);
    randn(noise, 0, 20);
    noisy = noisy + noise;
    imwrite("output/rectangle_detection/noisy.jpg", noisy);

    // Apply rectangle detection
    vector<ip101::advanced::RectangleInfo> noisy_rectangles;
    ip101::advanced::detect_rectangles(noisy, noisy_rectangles, 1000.0, 100000.0, 0.5, 2.0, 0.8);

    Mat noisy_result = noisy.clone();
    ip101::advanced::draw_rectangles(noisy_result, noisy_rectangles, Scalar(0, 255, 0), 2);
    imwrite("output/rectangle_detection/noisy_result.jpg", noisy_result);

    // Create blurred image
    Mat blurred = src.clone();
    GaussianBlur(blurred, blurred, Size(5, 5), 1.0);
    imwrite("output/rectangle_detection/blurred.jpg", blurred);

    // Apply rectangle detection
    vector<ip101::advanced::RectangleInfo> blurred_rectangles;
    ip101::advanced::detect_rectangles(blurred, blurred_rectangles, 1000.0, 100000.0, 0.5, 2.0, 0.8);

    Mat blurred_result = blurred.clone();
    ip101::advanced::draw_rectangles(blurred_result, blurred_rectangles, Scalar(0, 255, 0), 2);
    imwrite("output/rectangle_detection/blurred_result.jpg", blurred_result);

    // Create low contrast image
    Mat low_contrast = src.clone();
    low_contrast.convertTo(low_contrast, -1, 0.5, 50);
    imwrite("output/rectangle_detection/low_contrast.jpg", low_contrast);

    // Apply rectangle detection
    vector<ip101::advanced::RectangleInfo> low_contrast_rectangles;
    ip101::advanced::detect_rectangles(low_contrast, low_contrast_rectangles, 1000.0, 100000.0, 0.5, 2.0, 0.8);

    Mat low_contrast_result = low_contrast.clone();
    ip101::advanced::draw_rectangles(low_contrast_result, low_contrast_rectangles, Scalar(0, 255, 0), 2);
    imwrite("output/rectangle_detection/low_contrast_result.jpg", low_contrast_result);
}

// 质量评估
void quality_assessment(const Mat& src) {
    cout << "\n--- Quality Assessment ---" << endl;

    // Apply rectangle detection
    vector<ip101::advanced::RectangleInfo> rectangles;
    ip101::advanced::detect_rectangles(src, rectangles, 1000.0, 100000.0, 0.5, 2.0, 0.8);

    cout << "Detected " << rectangles.size() << " rectangles" << endl;

    // Analyze detection results
    if (!rectangles.empty()) {
        double total_confidence = 0.0;
        double total_area = 0.0;
        double total_aspect_ratio = 0.0;

        for (const auto& rect : rectangles) {
            total_confidence += rect.confidence;
            total_area += rect.size.width * rect.size.height;
            total_aspect_ratio += rect.size.width / rect.size.height;
        }

        double avg_confidence = total_confidence / rectangles.size();
        double avg_area = total_area / rectangles.size();
        double avg_aspect_ratio = total_aspect_ratio / rectangles.size();

        cout << "Average confidence: " << fixed << setprecision(3) << avg_confidence << endl;
        cout << "Average area: " << fixed << setprecision(2) << avg_area << endl;
        cout << "Average aspect ratio: " << fixed << setprecision(3) << avg_aspect_ratio << endl;

        // Calculate detection density
        double image_area = src.cols * src.rows;
        double detection_density = rectangles.size() / (image_area / 10000.0); // Detection count per 10000 pixels
        cout << "Detection density: " << fixed << setprecision(3) << detection_density << " per 10000 pixels" << endl;
    }

    // Compare with OpenCV
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(gray, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    int opencv_rect_count = 0;
    for (const auto& contour : contours) {
        if (contourArea(contour) > 1000) {
            opencv_rect_count++;
        }
    }

    cout << "OpenCV detected " << opencv_rect_count << " candidate rectangles" << endl;
    cout << "Detection rate difference: " << (rectangles.size() - opencv_rect_count) << endl;
}

int main() {
    cout << "=== Rectangle Detection Algorithm Test ===" << endl;

    // Create output directories
    create_output_directories();

    // Load test image
    Mat src = imread("assets/imori.jpg");
    if (src.empty()) {
        cerr << "Unable to load test image" << endl;
        return -1;
    }

    cout << "Image size: " << src.cols << "x" << src.rows << endl;

    // Execute various tests
    performance_test(src);
    parameter_effect_test(src);
    visualization_test(src);
    special_scenario_test(src);
    quality_assessment(src);

    cout << "\n=== Test Complete ===" << endl;
    cout << "All results saved to output/rectangle_detection/ directory" << endl;

    return 0;
}
