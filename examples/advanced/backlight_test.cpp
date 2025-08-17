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

int main() {
    cout << "=== 逆光校正算法测试 ===" << endl;

    filesystem::create_directories("output/backlight");

    Mat src = imread("test_images/test_image.jpg");
    if (src.empty()) {
        src = Mat::zeros(480, 640, CV_8UC3);
        src.setTo(Scalar(100, 150, 200));
    }

    int rows = src.rows;
    int cols = src.cols;

    // 性能测试
    Mat inrbl_result, adaptive_result, exposure_result;

    auto start = chrono::high_resolution_clock::now();
    ip101::advanced::inrbl_backlight_correction(src, inrbl_result, 0.6, 0.8);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "INRBL算法 - 耗时: " << duration.count() << " 微秒" << endl;

    start = chrono::high_resolution_clock::now();
    ip101::advanced::adaptive_backlight_correction(src, adaptive_result, 3.0, Size(8, 8));
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "自适应算法 - 耗时: " << duration.count() << " 微秒" << endl;

    start = chrono::high_resolution_clock::now();
    ip101::advanced::exposure_fusion_backlight_correction(src, exposure_result, 3);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "曝光融合算法 - 耗时: " << duration.count() << " 微秒" << endl;

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

    cout << "测试完成，结果已保存到 output/backlight/ 目录" << endl;
    return 0;
}
