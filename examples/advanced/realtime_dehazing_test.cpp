#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "advanced/defogging/realtime_dehazing.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    cout << "=== Real-time Defogging Algorithm Test ===" << endl;

    filesystem::create_directories("output/realtime_dehazing");

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
                float transmission = exp(-beta * depth_map.at<float>(i, j));
                channels[c].at<float>(i, j) = channels[c].at<float>(i, j) * transmission + A[c] * (1 - transmission);
            }
        }
    }

    merge(channels, hazy_float);
    hazy_float.convertTo(hazy_image, CV_8U);

    // 性能测试
    Mat dst_realtime, dst_fast, dst_dark_channel;

    auto start = chrono::high_resolution_clock::now();
    ip101::advanced::realtime_dehazing(hazy_image, dst_realtime, 0.25, 0.95, 0.1);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Real-time Defogging Algorithm - Time: " << duration.count() << " microseconds" << endl;

    start = chrono::high_resolution_clock::now();
    Vec3d A_fast(255, 255, 255);
    Mat transmission_map = Mat::ones(hazy_image.size(), CV_32F) * 0.8;
    ip101::advanced::fast_dehazing_model(hazy_image, dst_fast, A_fast, transmission_map);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Fast Defogging Algorithm - Time: " << duration.count() << " microseconds" << endl;

    start = chrono::high_resolution_clock::now();
    ip101::advanced::realtime_dark_channel_dehazing(hazy_image, dst_dark_channel, 15, 7, 0.95, 0.1);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Dark Channel Defogging Algorithm - Time: " << duration.count() << " microseconds" << endl;

    // 保存结果
    imwrite("output/realtime_dehazing/hazy_original.jpg", hazy_image);
    imwrite("output/realtime_dehazing/realtime_result.jpg", dst_realtime);
    imwrite("output/realtime_dehazing/fast_result.jpg", dst_fast);
    imwrite("output/realtime_dehazing/dark_channel_result.jpg", dst_dark_channel);

    // 创建对比图 - 简化版本，避免变量声明问题
    Mat comparison;
    hconcat(hazy_image, dst_realtime, comparison);
    hconcat(comparison, dst_fast, comparison);
    hconcat(comparison, dst_dark_channel, comparison);
    imwrite("output/realtime_dehazing/comparison.jpg", comparison);

    cout << "Test completed, results saved to output/realtime_dehazing/ directory" << endl;
    return 0;
}
