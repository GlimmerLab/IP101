#include <basic/frequency_domain.hpp>
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

    // 1. Test Fourier transform
    cout << "\n=== Fourier Transform Test ===" << endl;
    Mat dft_result, opencv_dft;

    double manual_time = measure_time([&]() {
        fourier_transform_manual(src, dft_result);
    }, "Manual Fourier transform");

    double opencv_time = measure_time([&]() {
        Mat padded;
        int m = getOptimalDFTSize(src.rows);
        int n = getOptimalDFTSize(src.cols);
        copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols,
                      BORDER_CONSTANT, Scalar::all(0));
        dft(padded, opencv_dft, DFT_COMPLEX_OUTPUT);
    }, "OpenCV Fourier transform");

    cout << "Performance ratio: " << opencv_time / manual_time << endl;

    // Visualize spectrum
    Mat dft_vis, opencv_dft_vis;
    visualize_spectrum(dft_result, dft_vis);
    visualize_spectrum(opencv_dft, opencv_dft_vis);
    imwrite("dft_manual.jpg", dft_vis);
    imwrite("dft_opencv.jpg", opencv_dft_vis);

    // 2. Test frequency domain filtering
    cout << "\n=== Frequency Domain Filtering Test ===" << endl;
    vector<string> filter_types = {"lowpass", "highpass", "bandpass"};

    for (const auto& filter_type : filter_types) {
        cout << "\n--- " << filter_type << " filter ---" << endl;
        Mat filter_result;

        manual_time = measure_time([&]() {
            frequency_filter_manual(src, filter_result, filter_type, 30.0);
        }, "Manual " + filter_type + " filtering");

        imwrite("filter_" + filter_type + ".jpg", filter_result);
    }

    // 3. Test DCT transform
    cout << "\n=== DCT Transform Test ===" << endl;
    Mat dct_result, opencv_dct;

    manual_time = measure_time([&]() {
        dct_transform_manual(src, dct_result);
    }, "Manual DCT transform");

    opencv_time = measure_time([&]() {
        Mat gray;
        cvtColor(src, gray, COLOR_BGR2GRAY);
        gray.convertTo(gray, CV_64F);
        dct(gray, opencv_dct);
    }, "OpenCV DCT transform");

    cout << "Performance ratio: " << opencv_time / manual_time << endl;

    // Normalize and save results
    normalize(dct_result, dct_result, 0, 255, NORM_MINMAX);
    normalize(opencv_dct, opencv_dct, 0, 255, NORM_MINMAX);
    dct_result.convertTo(dct_result, CV_8U);
    opencv_dct.convertTo(opencv_dct, CV_8U);
    imwrite("dct_manual.jpg", dct_result);
    imwrite("dct_opencv.jpg", opencv_dct);

    // 4. Test wavelet transform
    cout << "\n=== Wavelet Transform Test ===" << endl;
    vector<string> wavelet_types = {"haar", "db1"};

    for (const auto& wavelet_type : wavelet_types) {
        cout << "\n--- " << wavelet_type << " wavelet ---" << endl;
        Mat wavelet_result;

        manual_time = measure_time([&]() {
            wavelet_transform_manual(src, wavelet_result, wavelet_type, 3);
        }, "Manual " + wavelet_type + " wavelet transform");

        // Normalize and save results
        normalize(wavelet_result, wavelet_result, 0, 255, NORM_MINMAX);
        wavelet_result.convertTo(wavelet_result, CV_8U);
        imwrite("wavelet_" + wavelet_type + ".jpg", wavelet_result);
    }

    return 0;
}

