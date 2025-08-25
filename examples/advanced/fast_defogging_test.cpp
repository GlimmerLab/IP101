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

int main(int argc, char** argv) {
    cout << "=== Fast Defogging Algorithm Test ===" << endl;

    filesystem::create_directories("output/fast_defogging");

    string image_path = (argc > 1) ? argv[1] : "assets/imori.jpg";
    Mat src = imread(image_path);
    if (src.empty()) {
        cerr << "Error: Cannot load image " << image_path << endl;
        return -1;
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
    cout << "Maximum Contrast Defogging Algorithm - Time: " << duration.count() << " microseconds" << endl;

    start = chrono::high_resolution_clock::now();
    ip101::advanced::color_linear_transform_defogging(hazy_image, dst_color_linear, 0.8, 0.1);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Color Linear Transform Defogging Algorithm - Time: " << duration.count() << " microseconds" << endl;

    start = chrono::high_resolution_clock::now();
    ip101::advanced::logarithmic_enhancement_defogging(hazy_image, dst_logarithmic, 0.8, 1.2);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Logarithmic Enhancement Defogging Algorithm - Time: " << duration.count() << " microseconds" << endl;

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

    cout << "Test completed, results saved to output/fast_defogging/ directory" << endl;
    return 0;
}
