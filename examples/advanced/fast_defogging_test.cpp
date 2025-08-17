#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "advanced/defogging/fast_defogging.hpp"

using namespace std;
using namespace cv;

int main() {
    cout << "=== 快速去雾算法测试 ===" << endl;
    
    filesystem::create_directories("output/fast_defogging");
    
    Mat src = imread("test_images/test_image.jpg");
    if (src.empty()) {
        src = Mat::zeros(480, 640, CV_8UC3);
        src.setTo(Scalar(100, 150, 200));
    }
    
    int rows = src.rows;
    int cols = src.cols;
    
    // 创建雾霾测试图像
    Mat hazy_image = src.clone();
    Mat depth_map = Mat::zeros(rows, cols, CV_32F);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double distance = sqrt((i - rows/2) * (i - rows/2) + (j - cols/2) * (j - cols/2));
            double max_distance = sqrt((rows/2) * (rows/2) + (cols/2) * (cols/2));
            depth_map.at<float>(i, j) = distance / max_distance;
        }
    }
    
    double beta = 0.8;
    Vec3d A(255, 255, 255);
    
    Mat hazy_float;
    hazy_image.convertTo(hazy_float, CV_32F);
    vector<Mat> channels(3);
    split(hazy_float, channels);
    
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                float t = exp(-beta * depth_map.at<float>(i, j));
                channels[c].at<float>(i, j) = channels[c].at<float>(i, j) * t + A[c] * (1 - t);
            }
        }
    }
    
    merge(channels, hazy_float);
    hazy_float.convertTo(hazy_image, CV_8U);
    
    // 性能测试
    Mat dst_max_contrast, dst_color_linear, dst_logarithmic;
    
    auto start = chrono::high_resolution_clock::now();
    ip101::advanced::fast_max_contrast_defogging(hazy_image, dst_max_contrast, 15, 0.8, 0.1);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "最大对比度去雾算法 - 耗时: " << duration.count() << " 微秒" << endl;
    
    start = chrono::high_resolution_clock::now();
    ip101::advanced::color_linear_transform_defogging(hazy_image, dst_color_linear, 0.8, 0.1);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "颜色线性变换去雾算法 - 耗时: " << duration.count() << " 微秒" << endl;
    
    start = chrono::high_resolution_clock::now();
    ip101::advanced::logarithmic_enhancement_defogging(hazy_image, dst_logarithmic, 0.8, 1.2);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "对数增强去雾算法 - 耗时: " << duration.count() << " 微秒" << endl;
    
    // 保存结果
    imwrite("output/fast_defogging/hazy_original.jpg", hazy_image);
    imwrite("output/fast_defogging/max_contrast_result.jpg", dst_max_contrast);
    imwrite("output/fast_defogging/color_linear_result.jpg", dst_color_linear);
    imwrite("output/fast_defogging/logarithmic_result.jpg", dst_logarithmic);
    
    // 创建对比图
    Mat comparison;
    hconcat(hazy_image, dst_max_contrast, comparison);
    hconcat(comparison, dst_color_linear, comparison);
    hconcat(comparison, dst_logarithmic, comparison);
    imwrite("output/fast_defogging/comparison.jpg", comparison);
    
    cout << "测试完成，结果已保存到 output/fast_defogging/ 目录" << endl;
    return 0;
}
