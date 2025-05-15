#include <basic/color_operations.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

// Performance test helper function
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

    // Test channel swap
    std::cout << "Channel Swap Performance Test:" << std::endl;
    double custom_time = measure_time([&]() {
        ip101::channel_swap(src, dst, 2, 1, 0);  // BGR -> RGB
    });
    double opencv_time = measure_time([&]() {
        cv::cvtColor(src, opencv_dst, cv::COLOR_BGR2RGB);
    });
    std::cout << "Custom Implementation: " << custom_time << "ms" << std::endl;
    std::cout << "OpenCV Implementation: " << opencv_time << "ms" << std::endl;
    std::cout << "Performance Ratio: " << opencv_time / custom_time << std::endl << std::endl;

    // Test grayscale conversion
    std::cout << "Grayscale Conversion Performance Test:" << std::endl;
    custom_time = measure_time([&]() {
        ip101::to_gray(src, dst, "weighted");
    });
    opencv_time = measure_time([&]() {
        cv::cvtColor(src, opencv_dst, cv::COLOR_BGR2GRAY);
    });
    std::cout << "Custom Implementation: " << custom_time << "ms" << std::endl;
    std::cout << "OpenCV Implementation: " << opencv_time << "ms" << std::endl;
    std::cout << "Performance Ratio: " << opencv_time / custom_time << std::endl << std::endl;

    // Test thresholding
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    std::cout << "Thresholding Performance Test:" << std::endl;
    custom_time = measure_time([&]() {
        ip101::threshold_image(gray, dst, 128);
    });
    opencv_time = measure_time([&]() {
        cv::threshold(gray, opencv_dst, 128, 255, cv::THRESH_BINARY);
    });
    std::cout << "Custom Implementation: " << custom_time << "ms" << std::endl;
    std::cout << "OpenCV Implementation: " << opencv_time << "ms" << std::endl;
    std::cout << "Performance Ratio: " << opencv_time / custom_time << std::endl << std::endl;

    // Test Otsu thresholding
    std::cout << "Otsu Thresholding Performance Test:" << std::endl;
    custom_time = measure_time([&]() {
        ip101::otsu_threshold(gray, dst);
    });
    opencv_time = measure_time([&]() {
        cv::threshold(gray, opencv_dst, 0, 255, cv::THRESH_OTSU);
    });
    std::cout << "Custom Implementation: " << custom_time << "ms" << std::endl;
    std::cout << "OpenCV Implementation: " << opencv_time << "ms" << std::endl;
    std::cout << "Performance Ratio: " << opencv_time / custom_time << std::endl << std::endl;

    // Test HSV conversion
    std::cout << "HSV Conversion Performance Test:" << std::endl;
    custom_time = measure_time([&]() {
        ip101::bgr_to_hsv(src, dst);
    });
    opencv_time = measure_time([&]() {
        cv::cvtColor(src, opencv_dst, cv::COLOR_BGR2HSV);
    });
    std::cout << "Custom Implementation: " << custom_time << "ms" << std::endl;
    std::cout << "OpenCV Implementation: " << opencv_time << "ms" << std::endl;
    std::cout << "Performance Ratio: " << opencv_time / custom_time << std::endl << std::endl;
}

void visualize_results(const cv::Mat& src) {
    cv::Mat results[6];

    // Apply different color operations
    ip101::channel_swap(src, results[0], 2, 1, 0);  // BGR -> RGB
    ip101::to_gray(src, results[1], "weighted");
    ip101::threshold_image(results[1], results[2], 128);
    ip101::otsu_threshold(results[1], results[3]);
    ip101::bgr_to_hsv(src, results[4]);

    // HSV adjustment test
    cv::Mat hsv;
    ip101::bgr_to_hsv(src, hsv);
    ip101::adjust_hsv(hsv, hsv, 60, 1.2f, 1.1f);  // Hue shift by 60 degrees, Saturation +20%, Value +10%
    ip101::hsv_to_bgr(hsv, results[5]);

    // Create result display window
    cv::Mat display;
    int gap = 10;
    int cols = 3;
    int rows = 2;
    display.create(src.rows * rows + gap * (rows - 1),
                  src.cols * cols + gap * (cols - 1),
                  src.type());
    display.setTo(0);

    // Organize display layout
    cv::Mat roi;
    int idx = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols && idx < 6; j++) {
            roi = display(cv::Rect(j * (src.cols + gap),
                                 i * (src.rows + gap),
                                 src.cols, src.rows));
            if (results[idx].channels() == 1) {
                cv::cvtColor(results[idx], roi, cv::COLOR_GRAY2BGR);
            } else {
                results[idx].copyTo(roi);
            }
            idx++;
        }
    }

    // Display results
    cv::imshow("Color Operations Results", display);
    cv::waitKey(0);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    // Read image
    cv::Mat src = cv::imread(argv[1]);
    if (src.empty()) {
        std::cout << "Could not read image: " << argv[1] << std::endl;
        return -1;
    }

    // Run performance test
    std::cout << "Image size: " << src.cols << "x" << src.rows << std::endl << std::endl;
    test_performance(src);

    // Visualize results
    visualize_results(src);

    return 0;
}

