#include <basic/morphology.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;
using namespace ip101;

// Performance test function
template<typename Func>
double measure_time(Func func, const string& name) {
    auto start = chrono::high_resolution_clock::now();
    func();
    auto end = chrono::high_resolution_clock::now();
    double time = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;
    cout << name << " time: " << time << " ms" << endl;
    return time;
}

int main() {
    // Read test image
    Mat src = imread("test.jpg", IMREAD_GRAYSCALE);
    if (src.empty()) {
        cerr << "Cannot read test image" << endl;
        return -1;
    }

    // Create different kernel shapes
    vector<pair<int, string>> kernel_types = {
        {MORPH_RECT, "Rectangle"},
        {MORPH_CROSS, "Cross"},
        {MORPH_ELLIPSE, "Ellipse"}
    };

    for (const auto& kernel_type : kernel_types) {
        cout << "\n=== Testing " << kernel_type.second << " kernel ===" << endl;
        Mat kernel = create_kernel(kernel_type.first, Size(5, 5));
        Mat opencv_kernel = getStructuringElement(kernel_type.first, Size(5, 5));

        // 1. Test Dilation
        cout << "\n--- Dilation ---" << endl;
        Mat dilate_result, opencv_dilate;

        double manual_time = measure_time([&]() {
            dilate_manual(src, dilate_result, kernel);
        }, "Manual dilation");

        double opencv_time = measure_time([&]() {
            dilate(src, opencv_dilate, opencv_kernel);
        }, "OpenCV dilation");

        cout << "Performance ratio: " << opencv_time / manual_time << endl;
        imwrite("dilate_manual_" + kernel_type.second + ".jpg", dilate_result);
        imwrite("dilate_opencv_" + kernel_type.second + ".jpg", opencv_dilate);

        // 2. Test Erosion
        cout << "\n--- Erosion ---" << endl;
        Mat erode_result, opencv_erode;

        manual_time = measure_time([&]() {
            erode_manual(src, erode_result, kernel);
        }, "Manual erosion");

        opencv_time = measure_time([&]() {
            erode(src, opencv_erode, opencv_kernel);
        }, "OpenCV erosion");

        cout << "Performance ratio: " << opencv_time / manual_time << endl;
        imwrite("erode_manual_" + kernel_type.second + ".jpg", erode_result);
        imwrite("erode_opencv_" + kernel_type.second + ".jpg", opencv_erode);

        // 3. Test Opening
        cout << "\n--- Opening ---" << endl;
        Mat opening_result, opencv_opening;

        manual_time = measure_time([&]() {
            opening_manual(src, opening_result, kernel);
        }, "Manual opening");

        opencv_time = measure_time([&]() {
            morphologyEx(src, opencv_opening, MORPH_OPEN, opencv_kernel);
        }, "OpenCV opening");

        cout << "Performance ratio: " << opencv_time / manual_time << endl;
        imwrite("opening_manual_" + kernel_type.second + ".jpg", opening_result);
        imwrite("opening_opencv_" + kernel_type.second + ".jpg", opencv_opening);

        // 4. Test Closing
        cout << "\n--- Closing ---" << endl;
        Mat closing_result, opencv_closing;

        manual_time = measure_time([&]() {
            closing_manual(src, closing_result, kernel);
        }, "Manual closing");

        opencv_time = measure_time([&]() {
            morphologyEx(src, opencv_closing, MORPH_CLOSE, opencv_kernel);
        }, "OpenCV closing");

        cout << "Performance ratio: " << opencv_time / manual_time << endl;
        imwrite("closing_manual_" + kernel_type.second + ".jpg", closing_result);
        imwrite("closing_opencv_" + kernel_type.second + ".jpg", opencv_closing);

        // 5. Test Morphological Gradient
        cout << "\n--- Morphological Gradient ---" << endl;
        Mat gradient_result, opencv_gradient;

        manual_time = measure_time([&]() {
            morphological_gradient_manual(src, gradient_result, kernel);
        }, "Manual gradient");

        opencv_time = measure_time([&]() {
            morphologyEx(src, opencv_gradient, MORPH_GRADIENT, opencv_kernel);
        }, "OpenCV gradient");

        cout << "Performance ratio: " << opencv_time / manual_time << endl;
        imwrite("gradient_manual_" + kernel_type.second + ".jpg", gradient_result);
        imwrite("gradient_opencv_" + kernel_type.second + ".jpg", opencv_gradient);
    }

    return 0;
}

