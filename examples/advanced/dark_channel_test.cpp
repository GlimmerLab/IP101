#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "advanced/defogging/dark_channel.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    cout << "=== Dark Channel Defogging Algorithm Test ===" << endl;

    filesystem::create_directories("output/dark_channel");

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
    Mat dst_dark_channel, dst_bilateral;

    auto start = chrono::high_resolution_clock::now();
    ip101::advanced::dark_channel_defogging(hazy_image, dst_dark_channel, 15, 0.95, 0.1);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Dark Channel Defogging Algorithm - Time: " << duration.count() << " microseconds" << endl;

    start = chrono::high_resolution_clock::now();
    ip101::advanced::bilateral_dark_channel_defogging(hazy_image, dst_bilateral, 15, 0.95, 0.1);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Bilateral Filter Dark Channel Defogging Algorithm - Time: " << duration.count() << " microseconds" << endl;

    // 保存结果
    imwrite("output/dark_channel/hazy_original.jpg", hazy_image);
    imwrite("output/dark_channel/dark_channel_result.jpg", dst_dark_channel);
    imwrite("output/dark_channel/bilateral_result.jpg", dst_bilateral);

    // 创建对比图
    Mat comparison;
    hconcat(hazy_image, dst_dark_channel, comparison);
    hconcat(comparison, dst_bilateral, comparison);
    imwrite("output/dark_channel/comparison.jpg", comparison);

    cout << "Test completed, results saved to output/dark_channel/ directory" << endl;
    return 0;
}
