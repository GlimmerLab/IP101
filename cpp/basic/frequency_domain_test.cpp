#include "frequency_domain.hpp"
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace ip101;

// 性能测试函数
template<typename Func>
double measure_time(Func func, const string& name) {
    auto start = chrono::high_resolution_clock::now();
    func();
    auto end = chrono::high_resolution_clock::now();
    double time = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;
    cout << name << " 耗时: " << time << " ms" << endl;
    return time;
}

int main() {
    // 读取测试图像
    Mat src = imread("test.jpg");
    if (src.empty()) {
        cerr << "无法读取测试图像" << endl;
        return -1;
    }

    // 1. 测试傅里叶变换
    cout << "\n=== 傅里叶变换测试 ===" << endl;
    Mat dft_result, opencv_dft;

    double manual_time = measure_time([&]() {
        fourier_transform_manual(src, dft_result);
    }, "手动实现的傅里叶变换");

    double opencv_time = measure_time([&]() {
        Mat padded;
        int m = getOptimalDFTSize(src.rows);
        int n = getOptimalDFTSize(src.cols);
        copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols,
                      BORDER_CONSTANT, Scalar::all(0));
        dft(padded, opencv_dft, DFT_COMPLEX_OUTPUT);
    }, "OpenCV傅里叶变换");

    cout << "性能比: " << opencv_time / manual_time << endl;

    // 可视化频谱
    Mat dft_vis, opencv_dft_vis;
    visualize_spectrum(dft_result, dft_vis);
    visualize_spectrum(opencv_dft, opencv_dft_vis);
    imwrite("dft_manual.jpg", dft_vis);
    imwrite("dft_opencv.jpg", opencv_dft_vis);

    // 2. 测试频域滤波
    cout << "\n=== 频域滤波测试 ===" << endl;
    vector<string> filter_types = {"lowpass", "highpass", "bandpass"};

    for (const auto& filter_type : filter_types) {
        cout << "\n--- " << filter_type << "滤波 ---" << endl;
        Mat filter_result;

        manual_time = measure_time([&]() {
            frequency_filter_manual(src, filter_result, filter_type, 30.0);
        }, "手动实现的" + filter_type + "滤波");

        imwrite("filter_" + filter_type + ".jpg", filter_result);
    }

    // 3. 测试DCT变换
    cout << "\n=== DCT变换测试 ===" << endl;
    Mat dct_result, opencv_dct;

    manual_time = measure_time([&]() {
        dct_transform_manual(src, dct_result);
    }, "手动实现的DCT变换");

    opencv_time = measure_time([&]() {
        Mat gray;
        cvtColor(src, gray, COLOR_BGR2GRAY);
        gray.convertTo(gray, CV_64F);
        dct(gray, opencv_dct);
    }, "OpenCV DCT变换");

    cout << "性能比: " << opencv_time / manual_time << endl;

    // 归一化并保存结果
    normalize(dct_result, dct_result, 0, 255, NORM_MINMAX);
    normalize(opencv_dct, opencv_dct, 0, 255, NORM_MINMAX);
    dct_result.convertTo(dct_result, CV_8U);
    opencv_dct.convertTo(opencv_dct, CV_8U);
    imwrite("dct_manual.jpg", dct_result);
    imwrite("dct_opencv.jpg", opencv_dct);

    // 4. 测试小波变换
    cout << "\n=== 小波变换测试 ===" << endl;
    vector<string> wavelet_types = {"haar", "db1"};

    for (const auto& wavelet_type : wavelet_types) {
        cout << "\n--- " << wavelet_type << "小波 ---" << endl;
        Mat wavelet_result;

        manual_time = measure_time([&]() {
            wavelet_transform_manual(src, wavelet_result, wavelet_type, 3);
        }, "手动实现的" + wavelet_type + "小波变换");

        // 归一化并保存结果
        normalize(wavelet_result, wavelet_result, 0, 255, NORM_MINMAX);
        wavelet_result.convertTo(wavelet_result, CV_8U);
        imwrite("wavelet_" + wavelet_type + ".jpg", wavelet_result);
    }

    return 0;
}