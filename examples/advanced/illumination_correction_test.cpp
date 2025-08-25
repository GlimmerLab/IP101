#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "advanced/correction/illumination_correction.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    cout << "=== Illumination Correction Algorithm Test ===" << endl;

    filesystem::create_directories("output/illumination_correction");

    string image_path = (argc > 1) ? argv[1] : "assets/imori.jpg";
    Mat src = imread(image_path);
    if (src.empty()) {
        cerr << "Error: Cannot load image " << image_path << endl;
        return -1;
    }

    int rows = src.rows;
    int cols = src.cols;

    // 创建光照不均匀测试图像
    Mat uneven_image = src.clone();
    Mat illumination_mask = Mat::zeros(rows, cols, CV_32F);
    Point center(cols / 2, rows / 2);
    double max_distance = sqrt(center.x * center.x + center.y * center.y);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double distance = sqrt((i - center.y) * (i - center.y) + (j - center.x) * (j - center.x));
            float factor = 0.3f + 0.7f * (1.0f - distance / max_distance);
            illumination_mask.at<float>(i, j) = factor;
        }
    }

    Mat uneven_float;
    uneven_image.convertTo(uneven_float, CV_32F);
    vector<Mat> channels(3);
    split(uneven_float, channels);

    for (int c = 0; c < 3; ++c) {
        channels[c] = channels[c].mul(illumination_mask);
    }

    merge(channels, uneven_float);
    uneven_float.convertTo(uneven_image, CV_8U);

    // 性能测试
    Mat dst_homomorphic, dst_background, dst_multiscale;

    auto start = chrono::high_resolution_clock::now();
    ip101::advanced::homomorphic_illumination_correction(uneven_image, dst_homomorphic, 0.3, 1.5, 30.0);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Homomorphic Filter Algorithm - Time: " << duration.count() << " microseconds" << endl;

    start = chrono::high_resolution_clock::now();
    ip101::advanced::background_subtraction_correction(uneven_image, dst_background, 51, 0.5);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Background Subtraction Algorithm - Time: " << duration.count() << " microseconds" << endl;

    start = chrono::high_resolution_clock::now();
    ip101::advanced::multi_scale_illumination_correction(uneven_image, dst_multiscale, {15, 80, 250});
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Multi-scale Algorithm - Time: " << duration.count() << " microseconds" << endl;

    // 保存结果
    imwrite("output/illumination_correction/uneven_original.jpg", uneven_image);
    imwrite("output/illumination_correction/homomorphic_result.jpg", dst_homomorphic);
    imwrite("output/illumination_correction/background_result.jpg", dst_background);
    imwrite("output/illumination_correction/multiscale_result.jpg", dst_multiscale);

    // 创建对比图
    Mat comparison;
    hconcat(uneven_image, dst_homomorphic, comparison);
    hconcat(comparison, dst_background, comparison);
    hconcat(comparison, dst_multiscale, comparison);
    imwrite("output/illumination_correction/comparison.jpg", comparison);

    cout << "Test completed, results saved to output/illumination_correction/ directory" << endl;
    return 0;
}
