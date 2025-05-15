#include <basic/feature_extraction.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;
using namespace ip101;

// Performance testing function
template<typename Func>
double measure_time(Func func, const string& name) {
    auto start = chrono::high_resolution_clock::now();
    func();
    auto end = chrono::high_resolution_clock::now();
    double time = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000.0;
    cout << name << " execution time: " << time << " ms" << endl;
    return time;
}

int main() {
    // Read test image
    Mat src = imread("test.jpg");
    if (src.empty()) {
        cerr << "Cannot read test image" << endl;
        return -1;
    }

    // Test Harris corner detection
    Mat harris_result;
    measure_time([&]() {
        harris_corner_detection(src, harris_result, 2, 3, 0.04, 0.01);
    }, "Harris corner detection");
    imwrite("harris_result.jpg", harris_result);

    // Test manually implemented Harris corner detection
    Mat harris_manual_result;
    measure_time([&]() {
        compute_harris_manual(src, harris_manual_result, 0.04, 3, 0.01);
    }, "Manual Harris corner detection");
    imwrite("harris_manual_result.jpg", harris_manual_result);

    // Test SIFT feature extraction
    Mat sift_result;
    measure_time([&]() {
        sift_features(src, sift_result, 1000);
    }, "SIFT feature extraction");
    imwrite("sift_result.jpg", sift_result);

    // Test SURF feature extraction
    Mat surf_result;
    measure_time([&]() {
        surf_features(src, surf_result, 100);
    }, "SURF feature extraction");
    imwrite("surf_result.jpg", surf_result);

    // Test ORB feature extraction
    Mat orb_result;
    measure_time([&]() {
        orb_features(src, orb_result, 1000);
    }, "ORB feature extraction");
    imwrite("orb_result.jpg", orb_result);

    // Test feature matching
    Mat src2 = imread("test2.jpg");
    if (src2.empty()) {
        cerr << "Cannot read second test image" << endl;
        return -1;
    }

    Mat match_result;
    measure_time([&]() {
        feature_matching(src, src2, match_result, "sift");
    }, "SIFT feature matching");
    imwrite("match_result.jpg", match_result);

    return 0;
}

