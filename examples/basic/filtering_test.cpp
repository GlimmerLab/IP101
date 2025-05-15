#include <basic/filtering.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <omp.h>

using namespace cv;
using namespace std;

// Performance testing helper function
template<typename Func>
double measure_time(Func&& func, int iterations = 10) {
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration_cast<chrono::milliseconds>(end - start).count() / static_cast<double>(iterations);
}

void test_performance(const Mat& src) {
    // Ensure input image is grayscale
    Mat src_gray;
    if (src.type() != CV_8UC1) {
        cvtColor(src, src_gray, COLOR_BGR2GRAY);
    } else {
        src_gray = src;
    }

    Mat dst, opencv_dst;
    int kernelSize = 3;  // Default filter size
    // Test mean filtering
    cout << "Mean Filtering Performance Test:\n";

    // Test optimized version
    double optimized_time = measure_time([&]() {
        dst = ip101::meanFilter_optimized(src_gray, kernelSize);
    });

    // Test original version
    double original_time = measure_time([&]() {
        dst = ip101::meanFilter(src_gray, kernelSize);
    });

    // Test OpenCV version
    double opencv_time = measure_time([&]() {
        blur(src_gray, opencv_dst, Size(kernelSize, kernelSize));
    });

    cout << "Original version: " << original_time << "ms\n";
    cout << "Optimized version: " << optimized_time << "ms\n";
    cout << "OpenCV implementation: " << opencv_time << "ms\n";
    cout << "Optimization ratio (original/optimized): " << original_time / optimized_time << "x\n";
    cout << "OpenCV ratio (OpenCV/optimized): " << opencv_time / optimized_time << "x\n\n";

    // Test median filtering
    cout << "Median Filtering Performance Test:\n";

    // Test optimized version
    optimized_time = measure_time([&]() {
        dst = ip101::medianFilter_optimized(src_gray, kernelSize);
    });

    // Test original version
    original_time = measure_time([&]() {
        dst = ip101::medianFilter(src_gray, kernelSize);
    });

    // Test OpenCV version
    opencv_time = measure_time([&]() {
        medianBlur(src_gray, opencv_dst, kernelSize);
    });

    cout << "Original version: " << original_time << "ms\n";
    cout << "Optimized version: " << optimized_time << "ms\n";
    cout << "OpenCV implementation: " << opencv_time << "ms\n";
    cout << "Optimization ratio (original/optimized): " << original_time / optimized_time << "x\n";
    cout << "OpenCV ratio (OpenCV/optimized): " << opencv_time / optimized_time << "x\n\n";

    // Test Gaussian filtering
    cout << "Gaussian Filtering Performance Test:\n";
    double sigma = 1.0;  // Default standard deviation
    // Test optimized version
    optimized_time = measure_time([&]() {
        dst = ip101::gaussianFilter_optimized(src_gray, kernelSize, sigma);
    });

    // Test original version
    original_time = measure_time([&]() {
        dst = ip101::gaussianFilter(src_gray, kernelSize, sigma);
    });

    // Test OpenCV version
    opencv_time = measure_time([&]() {
        GaussianBlur(src_gray, opencv_dst, Size(kernelSize, kernelSize), sigma);
    });

    cout << "Original version: " << original_time << "ms\n";
    cout << "Optimized version: " << optimized_time << "ms\n";
    cout << "OpenCV implementation: " << opencv_time << "ms\n";
    cout << "Optimization ratio (original/optimized): " << original_time / optimized_time << "x\n";
    cout << "OpenCV ratio (OpenCV/optimized): " << opencv_time / optimized_time << "x\n\n";

    // Test mean pooling
    cout << "Mean Pooling Performance Test:\n";
    int poolSize = 2;  // Default pooling size

    // Test optimized version
    optimized_time = measure_time([&]() {
        dst = ip101::meanPooling_optimized(src_gray, poolSize);
    });

    // Test original version
    original_time = measure_time([&]() {
        dst = ip101::meanPooling(src_gray, poolSize);
    });

    // Test OpenCV version
    opencv_time = measure_time([&]() {
        resize(src_gray, opencv_dst, Size(), 0.5, 0.5, INTER_AREA);
    });

    cout << "Original version: " << original_time << "ms\n";
    cout << "Optimized version: " << optimized_time << "ms\n";
    cout << "OpenCV implementation: " << opencv_time << "ms\n";
    cout << "Optimization ratio (original/optimized): " << original_time / optimized_time << "x\n";
    cout << "OpenCV ratio (OpenCV/optimized): " << opencv_time / optimized_time << "x\n\n";

    // Test max pooling
    cout << "Max Pooling Performance Test:\n";

    // Test optimized version
    optimized_time = measure_time([&]() {
        dst = ip101::maxPooling_optimized(src_gray, poolSize);
    });

    // Test original version
    original_time = measure_time([&]() {
        dst = ip101::maxPooling(src_gray, poolSize);
    });

    cout << "Original version: " << original_time << "ms\n";
    cout << "Optimized version: " << optimized_time << "ms\n";
    cout << "Optimization ratio (original/optimized): " << original_time / optimized_time << "x\n\n";
    // OpenCV doesn't have a direct max pooling implementation for comparison
}

void visualize_results(const Mat& src) {
    // Ensure input image is grayscale
    Mat src_gray;
    if (src.type() != CV_8UC1) {
        cvtColor(src, src_gray, COLOR_BGR2GRAY);
    } else {
        src_gray = src;
    }

    Mat results[10];

    int kernelSize = 3;
    double sigma = 1.0;
    int poolSize = 2;

    // Test original and optimized implementations
    results[0] = ip101::meanFilter(src_gray, kernelSize);
    results[1] = ip101::meanFilter_optimized(src_gray, kernelSize);

    results[2] = ip101::medianFilter(src_gray, kernelSize);
    results[3] = ip101::medianFilter_optimized(src_gray, kernelSize);

    results[4] = ip101::gaussianFilter(src_gray, kernelSize, sigma);
    results[5] = ip101::gaussianFilter_optimized(src_gray, kernelSize, sigma);

    results[6] = ip101::meanPooling(src_gray, poolSize);
    results[7] = ip101::meanPooling_optimized(src_gray, poolSize);

    results[8] = ip101::maxPooling(src_gray, poolSize);
    results[9] = ip101::maxPooling_optimized(src_gray, poolSize);

    // Create result display window
    Mat display;
    int gap = 10;
    int cols = 5;
    int rows = 2;
    display.create(src_gray.rows * rows + gap * (rows - 1),
                  src_gray.cols * cols + gap * (cols - 1),
                  src.type());
    display.setTo(0);

    // Organize display layout
    Mat roi;
    string labels[10] = {
        "Mean Filter (Original)", "Mean Filter (Optimized)",
        "Median Filter (Original)", "Median Filter (Optimized)",
        "Gaussian Filter (Original)", "Gaussian Filter (Optimized)",
        "Mean Pooling (Original)", "Mean Pooling (Optimized)",
        "Max Pooling (Original)", "Max Pooling (Optimized)"
    };

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int idx = i * cols + j;
            if (idx < 10) {
                roi = display(Rect(j * (src_gray.cols + gap),
                                     i * (src_gray.rows + gap),
                                     src_gray.cols, src_gray.rows));

                // If single channel image, convert to 3-channel for display
                if (results[idx].channels() == 1) {
                    cvtColor(results[idx], roi, COLOR_GRAY2BGR);
                } else {
                    results[idx].copyTo(roi);
                }

                // Add label
                putText(display, labels[idx],
                          Point(j * (src_gray.cols + gap) + 10, i * (src_gray.rows + gap) + 30),
                          FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
            }
        }
    }

    // Display results
    imshow("Filtering Operations Comparison", display);
    waitKey(0);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <image_path>\n";
        return -1;
    }

    // Read image
    Mat src = imread(argv[1], IMREAD_GRAYSCALE);
    if (src.empty()) {
        cout << "Cannot read image: " << argv[1] << "\n";
        return -1;
    }

    // Set OpenMP thread count
    int num_threads = omp_get_num_procs();
    omp_set_num_threads(num_threads);
    cout << "Using " << num_threads << " threads for parallel computing\n";

    // Run performance tests
    cout << "Image size: " << src.cols << "x" << src.rows << "\n\n";
    test_performance(src);

    // Visualize results
    visualize_results(src);

    return 0;
}

