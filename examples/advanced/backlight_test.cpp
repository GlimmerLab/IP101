#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "advanced/correction/backlight.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    cout << "=== Backlight Correction Algorithm Test ===" << endl;

    filesystem::create_directories("output/backlight");

    string image_path = (argc > 1) ? argv[1] : "assets/imori.jpg";
    Mat src = imread(image_path);
    if (src.empty()) {
        cerr << "Error: Cannot load image " << image_path << endl;
        return -1;
    }

    int rows = src.rows;
    int cols = src.cols;

    // 性能测试
    Mat inrbl_result, adaptive_result, exposure_result;

    auto start = chrono::high_resolution_clock::now();
    ip101::advanced::inrbl_backlight_correction(src, inrbl_result, 0.6, 0.8);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "INRBL Algorithm - Time: " << duration.count() << " microseconds" << endl;

    start = chrono::high_resolution_clock::now();
    ip101::advanced::adaptive_backlight_correction(src, adaptive_result, 3.0, Size(8, 8));
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Adaptive Algorithm - Time: " << duration.count() << " microseconds" << endl;

    start = chrono::high_resolution_clock::now();
    ip101::advanced::exposure_fusion_backlight_correction(src, exposure_result, 3);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Exposure Fusion Algorithm - Time: " << duration.count() << " microseconds" << endl;

    // 保存结果
    imwrite("output/backlight/inrbl_result.jpg", inrbl_result);
    imwrite("output/backlight/adaptive_result.jpg", adaptive_result);
    imwrite("output/backlight/exposure_result.jpg", exposure_result);

    // 创建对比图
    Mat comparison;
    hconcat(src, inrbl_result, comparison);
    hconcat(comparison, adaptive_result, comparison);
    hconcat(comparison, exposure_result, comparison);
    imwrite("output/backlight/comparison.jpg", comparison);

    cout << "Test completed, results saved to output/backlight/ directory" << endl;
    return 0;
}
