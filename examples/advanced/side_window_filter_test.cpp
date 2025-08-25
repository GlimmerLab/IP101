#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <filesystem>

// IP101 头文件
#include "advanced/filtering/side_window_filter.hpp"

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
    cv::Mat dst;

    std::cout << "Side Window Filter Performance Test:" << std::endl;
    double custom_time = measure_time([&]() {
        ip101::advanced::side_window_filter(src, dst, 5, ip101::advanced::SideWindowType::BOX);
    });
    std::cout << "Side Window Filter: " << custom_time << "ms" << std::endl;

    // 与OpenCV双边滤波对比
    cv::Mat opencv_dst;
    double opencv_time = measure_time([&]() {
        cv::bilateralFilter(src, opencv_dst, 5, 50, 50);
    });
    std::cout << "OpenCV Bilateral Filter: " << opencv_time << "ms" << std::endl;
    std::cout << "Performance Ratio: " << opencv_time / custom_time << std::endl << std::endl;
}

void visualize_results(const cv::Mat& src) {
    cv::Mat results[6];

    // 应用不同的侧窗滤波参数
    ip101::advanced::side_window_filter(src, results[0], 3, ip101::advanced::SideWindowType::BOX);
    ip101::advanced::side_window_filter(src, results[1], 5, ip101::advanced::SideWindowType::BOX);
    ip101::advanced::side_window_filter(src, results[2], 7, ip101::advanced::SideWindowType::BOX);
    ip101::advanced::side_window_filter(src, results[3], 9, ip101::advanced::SideWindowType::BOX);
    ip101::advanced::side_window_filter(src, results[4], 11, ip101::advanced::SideWindowType::BOX);
    ip101::advanced::side_window_filter(src, results[5], 15, ip101::advanced::SideWindowType::BOX);

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
            results[idx].copyTo(roi);
            idx++;
        }
    }

    // Display results
    cv::imshow("Side Window Filter Results", display);
    cv::waitKey(0);
}

void test_parameter_effects(const cv::Mat& src) {
    cv::Mat dst;

    std::cout << "\nParameter Effects Test:" << std::endl;

    // 测试不同半径
    std::vector<int> radii = {3, 5, 7, 9, 11, 15};
    for (int radius : radii) {
        auto start = std::chrono::high_resolution_clock::now();
        ip101::advanced::side_window_filter(src, dst, radius, ip101::advanced::SideWindowType::BOX);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Radius " << radius << ": " << duration.count() << "ms" << std::endl;
        cv::imwrite("side_window_radius_" + std::to_string(radius) + ".jpg", dst);
    }

    // 测试不同滤波器类型
    std::vector<ip101::advanced::SideWindowType> filter_types = {
        ip101::advanced::SideWindowType::BOX,
        ip101::advanced::SideWindowType::MEDIAN
    };
    for (auto filter_type : filter_types) {
        ip101::advanced::side_window_filter(src, dst, 5, filter_type);
        std::string type_name = (filter_type == ip101::advanced::SideWindowType::BOX) ? "box" : "median";
        cv::imwrite("side_window_type_" + type_name + ".jpg", dst);
    }
}

void test_edge_preservation(const cv::Mat& src) {
    cv::Mat dst, opencv_dst;

    // 侧窗滤波
    ip101::advanced::side_window_filter(src, dst, 5, ip101::advanced::SideWindowType::BOX);

    // OpenCV双边滤波对比
    cv::bilateralFilter(src, opencv_dst, 5, 50, 50);

    // 保存对比结果
    cv::imwrite("side_window_result.jpg", dst);
    cv::imwrite("opencv_bilateral_result.jpg", opencv_dst);

    std::cout << "\nEdge Preservation Test:" << std::endl;
    std::cout << "Side Window Filter result saved as: side_window_result.jpg" << std::endl;
    std::cout << "OpenCV Bilateral Filter result saved as: opencv_bilateral_result.jpg" << std::endl;
}

int main(int argc, char** argv) {
    using std::string;

    string image_path = (argc > 1) ? argv[1] : "assets/imori.jpg";
    cv::Mat src = cv::imread(image_path);
    if (src.empty()) {
        std::cerr << "Error: Cannot load image " << image_path << std::endl;
        return -1;
    }

    // Run performance test
    std::cout << "Image size: " << src.cols << "x" << src.rows << std::endl << std::endl;
    test_performance(src);

    // Test parameter effects
    test_parameter_effects(src);

    // Test edge preservation
    test_edge_preservation(src);

    // Visualize results
    visualize_results(src);

    return 0;
}
