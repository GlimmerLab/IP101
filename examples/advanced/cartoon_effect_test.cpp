#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <tuple>
#include <string>
#include "advanced/effects/cartoon_effect.hpp"

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

void test_basic_cartoon_effect(const cv::Mat& src) {
    cv::Mat results[6];

    // 应用卡通效果
    ip101::advanced::cartoon_effect(src, results[0]);

    // 测试不同参数
    ip101::advanced::CartoonParams params1;
    params1.median_blur_size = 7;
    params1.bilateral_d = 9;
    params1.quantize_levels = 2;
    params1.bilateral_sigma_color = 0.1;
    ip101::advanced::cartoon_effect(src, results[1], params1);

    ip101::advanced::CartoonParams params2;
    params2.median_blur_size = 9;
    params2.bilateral_d = 9;
    params2.quantize_levels = 2;
    params2.bilateral_sigma_color = 0.1;
    ip101::advanced::cartoon_effect(src, results[2], params2);

    ip101::advanced::CartoonParams params3;
    params3.median_blur_size = 7;
    params3.bilateral_d = 11;
    params3.quantize_levels = 2;
    params3.bilateral_sigma_color = 0.1;
    ip101::advanced::cartoon_effect(src, results[3], params3);

    ip101::advanced::CartoonParams params4;
    params4.median_blur_size = 7;
    params4.bilateral_d = 9;
    params4.quantize_levels = 3;
    params4.bilateral_sigma_color = 0.1;
    ip101::advanced::cartoon_effect(src, results[4], params4);

    results[5] = src.clone();

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

    cv::imshow("Cartoon Effect Results", display);
    cv::waitKey(0);
}

void test_parameter_effects(const cv::Mat& src) {
    cv::Mat dst;

    std::cout << "\nParameter Effects Test:" << std::endl;

    // 测试不同模糊核大小
    std::vector<int> blur_sizes = {5, 7, 9, 11, 15};
    for (int blur_size : blur_sizes) {
        auto start = std::chrono::high_resolution_clock::now();
        ip101::advanced::CartoonParams params;
        params.median_blur_size = blur_size;
        params.bilateral_d = 9;
        params.quantize_levels = 2;
        params.bilateral_sigma_color = 0.1;
        ip101::advanced::cartoon_effect(src, dst, params);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Blur size " << blur_size << ": " << duration.count() << "ms" << std::endl;
        cv::imwrite("cartoon_blur_" + std::to_string(blur_size) + ".jpg", dst);
    }

    // 测试不同边缘检测参数
    std::vector<int> edge_sizes = {7, 9, 11, 13, 15};
    for (int edge_size : edge_sizes) {
        ip101::advanced::CartoonParams params;
        params.median_blur_size = 7;
        params.bilateral_d = edge_size;
        params.quantize_levels = 2;
        params.bilateral_sigma_color = 0.1;
        ip101::advanced::cartoon_effect(src, dst, params);
        cv::imwrite("cartoon_edge_" + std::to_string(edge_size) + ".jpg", dst);
    }
}

void test_different_styles(const cv::Mat& src) {
    cv::Mat dst;

    // 创建不同风格的卡通效果
    std::vector<std::tuple<int, int, int, double>> styles = {
        {5, 7, 2, 0.05},   // 轻微卡通
        {7, 9, 2, 0.1},    // 标准卡通
        {9, 11, 3, 0.15},  // 强烈卡通
        {11, 13, 4, 0.2}   // 极强卡通
    };

    std::vector<std::string> style_names = {"light", "standard", "strong", "extreme"};

    for (size_t i = 0; i < styles.size(); ++i) {
        auto [blur, edge, threshold, strength] = styles[i];
        ip101::advanced::CartoonParams params;
        params.median_blur_size = blur;
        params.bilateral_d = edge;
        params.quantize_levels = threshold;
        params.bilateral_sigma_color = strength;
        ip101::advanced::cartoon_effect(src, dst, params);
        cv::imwrite("cartoon_style_" + style_names[i] + ".jpg", dst);
    }

    std::cout << "\nDifferent Cartoon Styles Test:" << std::endl;
    for (const auto& name : style_names) {
        std::cout << name << " style: cartoon_style_" << name << ".jpg" << std::endl;
    }
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
    test_basic_cartoon_effect(src);

    // Test parameter effects
    test_parameter_effects(src);

    // Test different styles
    test_different_styles(src);

    // Visualize results
    test_basic_cartoon_effect(src);

    return 0;
}
