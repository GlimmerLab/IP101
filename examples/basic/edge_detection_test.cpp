#include <basic/edge_detection.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

// Performance testing helper function
template<typename Func>
double measure_time(Func&& func, int iterations = 10) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / static_cast<double>(iterations);
}

void test_performance(const cv::Mat& src) {
    cv::Mat dst, opencv_dst;

    // Test Sobel operator
    std::cout << "Sobel Operator Performance Test:\n";
    double custom_time = measure_time([&]() {
        ip101::sobel_filter(src, dst, 1, 1, 3, 1.0);
    });
    double opencv_time = measure_time([&]() {
        cv::Sobel(src, opencv_dst, CV_8U, 1, 1);
    });
    std::cout << "Custom implementation: " << custom_time << "ms\n";
    std::cout << "OpenCV implementation: " << opencv_time << "ms\n";
    std::cout << "Performance ratio: " << opencv_time / custom_time << "\n\n";

    // Test Prewitt operator
    std::cout << "Prewitt Operator Performance Test:\n";
    custom_time = measure_time([&]() {
        ip101::prewitt_filter(src, dst, 1, 1);
    });
    std::cout << "Custom implementation: " << custom_time << "ms\n\n";

    // Test Laplacian operator
    std::cout << "Laplacian Operator Performance Test:\n";
    custom_time = measure_time([&]() {
        ip101::laplacian_filter(src, dst, 3, 1.0);
    });
    opencv_time = measure_time([&]() {
        cv::Laplacian(src, opencv_dst, CV_8U);
    });
    std::cout << "Custom implementation: " << custom_time << "ms\n";
    std::cout << "OpenCV implementation: " << opencv_time << "ms\n";
    std::cout << "Performance ratio: " << opencv_time / custom_time << "\n\n";
}

void visualize_results(const cv::Mat& src) {
    cv::Mat results[5];

    // Apply different edge detection methods
    ip101::sobel_filter(src, results[0], 1, 1, 3, 1.0);
    ip101::prewitt_filter(src, results[1], 1, 1);
    ip101::laplacian_filter(src, results[2], 3, 1.0);
    ip101::differential_filter(src, results[3], 1, 1, 3);
    ip101::emboss_effect(src, results[4], 0);

    // Create result display window
    cv::Mat display;
    int gap = 10;
    int cols = 3;
    int rows = 2;
    display.create(src.rows * rows + gap * (rows - 1),
                  src.cols * cols + gap * (cols - 1),
                  CV_8UC1);
    display.setTo(0);

    // Organize display layout
    cv::Mat roi;
    int idx = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols && idx < 5; j++) {
            roi = display(cv::Rect(j * (src.cols + gap),
                                 i * (src.rows + gap),
                                 src.cols, src.rows));
            results[idx++].copyTo(roi);
        }
    }

    // Display results
    cv::imshow("Edge Detection Results Comparison", display);
    cv::waitKey(0);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <image_path>\n";
        return -1;
    }

    // Read image
    cv::Mat src = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cout << "Cannot read image: " << argv[1] << "\n";
        return -1;
    }

    // Run performance tests
    std::cout << "Image size: " << src.cols << "x" << src.rows << "\n\n";
    test_performance(src);

    // Visualize results
    visualize_results(src);

    return 0;
}

