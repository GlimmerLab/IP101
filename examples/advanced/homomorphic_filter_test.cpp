#include <advanced/filtering/homomorphic_filter.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <string>

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

    std::cout << "Homomorphic Filter Performance Test:" << std::endl;
    double custom_time = measure_time([&]() {
        ip101::advanced::homomorphic_filter(src, dst, 0.5, 2.0, 0.1, 0.1);
    });
    std::cout << "Homomorphic Filter: " << custom_time << "ms" << std::endl;

    // 与OpenCV对比（使用对数变换+高斯滤波）
    cv::Mat opencv_dst, log_img, filtered;
    double opencv_time = measure_time([&]() {
        cv::Mat float_img;
        src.convertTo(float_img, CV_32F);
        cv::log(float_img + 1, log_img);
        cv::GaussianBlur(log_img, filtered, cv::Size(15, 15), 0);
        cv::exp(filtered, opencv_dst);
        opencv_dst.convertTo(opencv_dst, src.type());
    });
    std::cout << "OpenCV Log+Gaussian: " << opencv_time << "ms" << std::endl;
    std::cout << "Performance Ratio: " << opencv_time / custom_time << std::endl << std::endl;
}

void visualize_results(const cv::Mat& src) {
    cv::Mat results[6];

    // 应用不同的同态滤波参数
    ip101::advanced::homomorphic_filter(src, results[0], 0.1, 1.5, 0.1, 0.1);  // 低通
    ip101::advanced::homomorphic_filter(src, results[1], 0.3, 2.0, 0.1, 0.1);  // 中通
    ip101::advanced::homomorphic_filter(src, results[2], 0.5, 2.5, 0.1, 0.1);  // 高通
    ip101::advanced::homomorphic_filter(src, results[3], 0.7, 3.0, 0.1, 0.1);  // 强高通
    ip101::advanced::homomorphic_filter(src, results[4], 0.9, 3.5, 0.1, 0.1);  // 极强高通
    ip101::advanced::homomorphic_filter(src, results[5], 0.5, 2.0, 0.05, 0.05); // 小半径

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
    cv::imshow("Homomorphic Filter Results", display);
    cv::waitKey(0);
}

void test_parameter_effects(const cv::Mat& src) {
    cv::Mat dst;

    std::cout << "\nParameter Effects Test:" << std::endl;

    // 测试不同gamma值
    std::vector<double> gamma_values = {0.1, 0.3, 0.5, 0.7, 0.9};
    for (double gamma : gamma_values) {
        auto start = std::chrono::high_resolution_clock::now();
        ip101::advanced::homomorphic_filter(src, dst, gamma, 2.0, 0.1, 0.1);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Gamma " << gamma << ": " << duration.count() << "ms" << std::endl;
        cv::imwrite("homomorphic_gamma_" + std::to_string(gamma) + ".jpg", dst);
    }

    // 测试不同c值
    std::vector<double> c_values = {1.0, 1.5, 2.0, 2.5, 3.0};
    for (double c : c_values) {
        ip101::advanced::homomorphic_filter(src, dst, 0.5, c, 0.1, 0.1);
        cv::imwrite("homomorphic_c_" + std::to_string(c) + ".jpg", dst);
    }

    // 测试不同D0值
    std::vector<double> d0_values = {0.05, 0.1, 0.2, 0.3, 0.5};
    for (double d0 : d0_values) {
        ip101::advanced::homomorphic_filter(src, dst, 0.5, 2.0, d0, d0);
        cv::imwrite("homomorphic_d0_" + std::to_string(d0) + ".jpg", dst);
    }
}

void test_illumination_correction(const cv::Mat& src) {
    cv::Mat dst;

    // 同态滤波用于光照校正
    ip101::advanced::homomorphic_filter(src, dst, 0.5, 2.0, 0.1, 0.1);

    // 保存结果
    cv::imwrite("homomorphic_illumination_corrected.jpg", dst);

    std::cout << "\nIllumination Correction Test:" << std::endl;
    std::cout << "Original image saved as: original.jpg" << std::endl;
    std::cout << "Illumination corrected image saved as: homomorphic_illumination_corrected.jpg" << std::endl;

    cv::imwrite("original.jpg", src);
}

void test_frequency_domain_analysis(const cv::Mat& src) {
    cv::Mat dst;

    // 创建不同频率响应的滤波器
    std::vector<std::pair<double, double>> filter_params = {
        {0.1, 1.5},  // 低通
        {0.5, 2.0},  // 高通
        {0.9, 3.0}   // 强高通
    };

    std::vector<std::string> filter_names = {"low_pass", "high_pass", "strong_high_pass"};

    for (size_t i = 0; i < filter_params.size(); ++i) {
        ip101::advanced::homomorphic_filter(src, dst, filter_params[i].first,
                                          filter_params[i].second, 0.1, 0.1);
        cv::imwrite("homomorphic_" + filter_names[i] + ".jpg", dst);
    }

    std::cout << "\nFrequency Domain Analysis:" << std::endl;
    std::cout << "Low pass filter result: homomorphic_low_pass.jpg" << std::endl;
    std::cout << "High pass filter result: homomorphic_high_pass.jpg" << std::endl;
    std::cout << "Strong high pass filter result: homomorphic_strong_high_pass.jpg" << std::endl;
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

    // Test illumination correction
    test_illumination_correction(src);

    // Test frequency domain analysis
    test_frequency_domain_analysis(src);

    // Visualize results
    visualize_results(src);

    return 0;
}
