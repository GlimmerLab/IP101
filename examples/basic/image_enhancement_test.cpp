#include <basic/image_enhancement.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

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

// Display histogram of an image
void display_histogram(const Mat& img, const string& window_name) {
    // Calculate histogram
    Mat hist;
    calculate_histogram(img, hist);

    // Normalize histogram for visualization
    Mat hist_img(256, 300, CV_8UC1, Scalar(0));
    double max_val = 0;
    for (int i = 0; i < 256; i++) {
        if (hist.at<int>(i) > max_val) max_val = hist.at<int>(i);
    }

    if (max_val > 0) {
        for (int i = 0; i < 256; i++) {
            int height = cvRound((double)hist.at<int>(i) * 250 / max_val);
            line(hist_img, Point(i, 250), Point(i, 250 - height), Scalar(255));
        }
    }

    imshow(window_name + " - Histogram", hist_img);
}

// Test histogram equalization methods
void test_histogram_equalization(const Mat& src) {
    cout << "\n=== Histogram Equalization Test ===" << endl;

    // Test global histogram equalization
    Mat global_eq;
    double time1 = measure_time([&]() {
        histogram_equalization(src, global_eq, "global");
    }, "Global histogram equalization");

    // Test adaptive histogram equalization
    Mat adaptive_eq;
    double time2 = measure_time([&]() {
        histogram_equalization(src, adaptive_eq, "adaptive");
    }, "Adaptive histogram equalization");

    // Test CLAHE
    Mat clahe_eq;
    double time3 = measure_time([&]() {
        histogram_equalization(src, clahe_eq, "clahe", 40.0, Size(8, 8));
    }, "CLAHE");

    // Display results
    imshow("Original", src);
    display_histogram(src, "Original");

    imshow("Global HE", global_eq);
    display_histogram(global_eq, "Global HE");

    imshow("Adaptive HE", adaptive_eq);
    display_histogram(adaptive_eq, "Adaptive HE");

    imshow("CLAHE", clahe_eq);
    display_histogram(clahe_eq, "CLAHE");

    // Save results
    imwrite("global_eq.jpg", global_eq);
    imwrite("adaptive_eq.jpg", adaptive_eq);
    imwrite("clahe_eq.jpg", clahe_eq);

    waitKey(0);
    destroyAllWindows();
}

// Test gamma correction
void test_gamma_correction(const Mat& src) {
    cout << "\n=== Gamma Correction Test ===" << endl;

    vector<double> gamma_values = {0.4, 0.67, 1.0, 1.5, 2.2};
    vector<Mat> results;

    for (double gamma : gamma_values) {
        Mat result;
        double time = measure_time([&]() {
            gamma_correction(src, result, gamma);
        }, "Gamma correction (gamma = " + to_string(gamma) + ")");

        results.push_back(result);
        imshow("Gamma = " + to_string(gamma), result);
        imwrite("gamma_" + to_string(gamma) + ".jpg", result);
    }

    imshow("Original", src);
    waitKey(0);
    destroyAllWindows();
}

// Test contrast stretching
void test_contrast_stretching(const Mat& src) {
    cout << "\n=== Contrast Stretching Test ===" << endl;

    // Standard contrast stretching
    Mat standard_result;
    double time1 = measure_time([&]() {
        contrast_stretching(src, standard_result);
    }, "Standard contrast stretching");

    // Custom range contrast stretching
    Mat custom_result;
    double time2 = measure_time([&]() {
        contrast_stretching(src, custom_result, 50, 200);
    }, "Custom range contrast stretching");

    imshow("Original", src);
    imshow("Standard Contrast Stretching", standard_result);
    imshow("Custom Contrast Stretching", custom_result);

    imwrite("contrast_standard.jpg", standard_result);
    imwrite("contrast_custom.jpg", custom_result);

    waitKey(0);
    destroyAllWindows();
}

// Test brightness adjustment
void test_brightness_adjustment(const Mat& src) {
    cout << "\n=== Brightness Adjustment Test ===" << endl;

    vector<double> beta_values = {-50, -25, 0, 25, 50};
    vector<Mat> results;

    for (double beta : beta_values) {
        Mat result;
        double time = measure_time([&]() {
            brightness_adjustment(src, result, beta);
        }, "Brightness adjustment (beta = " + to_string(beta) + ")");

        results.push_back(result);
        imshow("Brightness = " + to_string(beta), result);
        imwrite("brightness_" + to_string(int(beta)) + ".jpg", result);
    }

    imshow("Original", src);
    waitKey(0);
    destroyAllWindows();
}

// Test saturation adjustment
void test_saturation_adjustment(const Mat& src) {
    cout << "\n=== Saturation Adjustment Test ===" << endl;

    // Check if image is grayscale
    if (src.channels() == 1) {
        cout << "Saturation adjustment requires a color image." << endl;
        return;
    }

    vector<double> saturation_values = {0.0, 0.5, 1.0, 1.5, 2.0};
    vector<Mat> results;

    for (double saturation : saturation_values) {
        Mat result;
        double time = measure_time([&]() {
            saturation_adjustment(src, result, saturation);
        }, "Saturation adjustment (saturation = " + to_string(saturation) + ")");

        results.push_back(result);
        imshow("Saturation = " + to_string(saturation), result);
        imwrite("saturation_" + to_string(saturation) + ".jpg", result);
    }

    imshow("Original", src);
    waitKey(0);
    destroyAllWindows();
}

// Compare with OpenCV's built-in functions
void compare_with_opencv(const Mat& src) {
    cout << "\n=== Comparison with OpenCV ===" << endl;

    // Our implementation of histogram equalization
    Mat our_hist_eq;
    double our_time = measure_time([&]() {
        histogram_equalization(src, our_hist_eq);
    }, "Our histogram equalization");

    // OpenCV's histogram equalization
    Mat opencv_hist_eq;
    double opencv_time = measure_time([&]() {
        if (src.channels() == 1) {
            equalizeHist(src, opencv_hist_eq);
        } else {
            Mat hsv;
            cvtColor(src, hsv, COLOR_BGR2HSV);
            vector<Mat> channels;
            split(hsv, channels);
            equalizeHist(channels[2], channels[2]);
            merge(channels, hsv);
            cvtColor(hsv, opencv_hist_eq, COLOR_HSV2BGR);
        }
    }, "OpenCV histogram equalization");

    cout << "Speed ratio: " << opencv_time / our_time << "x" << endl;

    // Compare results
    imshow("Original", src);
    imshow("Our HE", our_hist_eq);
    imshow("OpenCV HE", opencv_hist_eq);

    waitKey(0);
    destroyAllWindows();
}

int main(int argc, char* argv[]) {
    // Read test image
    string image_path = "test.jpg";
    if (argc > 1) {
        image_path = argv[1];
    }

    Mat src = imread(image_path);
    if (src.empty()) {
        cerr << "Cannot read image: " << image_path << endl;
        return -1;
    }

    // Resize large images for display
    if (src.cols > 1200 || src.rows > 800) {
        double scale = min(1200.0 / src.cols, 800.0 / src.rows);
        resize(src, src, Size(), scale, scale, INTER_AREA);
    }

    // Run tests
    test_histogram_equalization(src);
    test_gamma_correction(src);
    test_contrast_stretching(src);
    test_brightness_adjustment(src);
    test_saturation_adjustment(src);
    compare_with_opencv(src);

    cout << "\nAll tests completed successfully!" << endl;

    return 0;
}