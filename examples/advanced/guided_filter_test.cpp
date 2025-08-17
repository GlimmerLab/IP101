#include <advanced/filtering/guided_filter.hpp>
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
    cv::Mat dst, guide;

    // 创建引导图像（高斯模糊版本）
    cv::GaussianBlur(src, guide, cv::Size(15, 15), 0);

    std::cout << "Guided Filter Performance Test:" << std::endl;
    double custom_time = measure_time([&]() {
        ip101::advanced::guided_filter(src, guide, dst, 8, 0.1);
    });
    std::cout << "Guided Filter: " << custom_time << "ms" << std::endl;

    std::cout << "Fast Guided Filter Performance Test:" << std::endl;
    custom_time = measure_time([&]() {
        ip101::advanced::fast_guided_filter(src, guide, dst, 8, 0.1, 4);
    });
    std::cout << "Fast Guided Filter: " << custom_time << "ms" << std::endl;

    std::cout << "Edge-Aware Guided Filter Performance Test:" << std::endl;
    custom_time = measure_time([&]() {
        ip101::advanced::edge_aware_guided_filter(src, guide, dst, 8, 0.1, 10.0);
    });
    std::cout << "Edge-Aware Guided Filter: " << custom_time << "ms" << std::endl;

    std::cout << "Joint Bilateral Filter Performance Test:" << std::endl;
    custom_time = measure_time([&]() {
        ip101::advanced::joint_bilateral_filter(src, guide, dst, 8, 50.0, 50.0);
    });
    std::cout << "Joint Bilateral Filter: " << custom_time << "ms" << std::endl;
}

void visualize_results(const cv::Mat& src) {
    cv::Mat results[5];
    cv::Mat guide;

    // 创建引导图像
    cv::GaussianBlur(src, guide, cv::Size(15, 15), 0);

    // 应用不同的导向滤波
    ip101::advanced::guided_filter(src, guide, results[0], 8, 0.1);
    ip101::advanced::fast_guided_filter(src, guide, results[1], 8, 0.1, 4);
    ip101::advanced::edge_aware_guided_filter(src, guide, results[2], 8, 0.1, 10.0);
    ip101::advanced::joint_bilateral_filter(src, guide, results[3], 8, 50.0, 50.0);

    // 使用原图作为引导图像
    ip101::advanced::guided_filter(src, src, results[4], 8, 0.1);

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
        for (int j = 0; j < cols && idx < 5; j++) {
            roi = display(cv::Rect(j * (src.cols + gap),
                                 i * (src.rows + gap),
                                 src.cols, src.rows));
            results[idx].copyTo(roi);
            idx++;
        }
    }

    // Display results
    cv::imshow("Guided Filter Results", display);
    cv::waitKey(0);
}

void test_parameter_effects(const cv::Mat& src) {
    cv::Mat dst, guide;
    cv::GaussianBlur(src, guide, cv::Size(15, 15), 0);

    std::cout << "\nParameter Effects Test:" << std::endl;

    // 测试不同半径
    std::vector<int> radii = {4, 8, 16, 32};
    for (int radius : radii) {
        auto start = std::chrono::high_resolution_clock::now();
        ip101::advanced::guided_filter(src, guide, dst, radius, 0.1);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Radius " << radius << ": " << duration.count() << "ms" << std::endl;
        cv::imwrite("guided_filter_radius_" + std::to_string(radius) + ".jpg", dst);
    }

    // 测试不同eps值
    std::vector<double> eps_values = {0.01, 0.1, 1.0, 10.0};
    for (double eps : eps_values) {
        ip101::advanced::guided_filter(src, guide, dst, 8, eps);
        cv::imwrite("guided_filter_eps_" + std::to_string(eps) + ".jpg", dst);
    }
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

    // Test parameter effects
    test_parameter_effects(src);

    // Visualize results
    visualize_results(src);

    return 0;
}
