#include <advanced/correction/gamma_correction.hpp>
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
    cv::Mat dst;

    std::cout << "Gamma Correction Performance Test:" << std::endl;
    double custom_time = measure_time([&]() {
        ip101::advanced::standard_gamma_correction(src, dst, 2.2);
    });
    std::cout << "Standard Gamma Correction: " << custom_time << "ms" << std::endl;

    // 与OpenCV对比
    cv::Mat opencv_dst;
    double opencv_time = measure_time([&]() {
        cv::pow(src / 255.0, 1.0 / 2.2, opencv_dst);
        opencv_dst = opencv_dst * 255;
        opencv_dst.convertTo(opencv_dst, src.type());
    });
    std::cout << "OpenCV Pow: " << opencv_time << "ms" << std::endl;
    std::cout << "Performance Ratio: " << opencv_time / custom_time << std::endl << std::endl;
}

void visualize_results(const cv::Mat& src) {
    cv::Mat results[6];

    // 应用不同的伽马值
    ip101::advanced::standard_gamma_correction(src, results[0], 0.5);   // 亮化
    ip101::advanced::standard_gamma_correction(src, results[1], 1.0);   // 无变化
    ip101::advanced::standard_gamma_correction(src, results[2], 1.5);   // 轻微暗化
    ip101::advanced::standard_gamma_correction(src, results[3], 2.2);   // 标准伽马
    ip101::advanced::standard_gamma_correction(src, results[4], 3.0);   // 强暗化

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

    cv::imshow("Gamma Correction Results", display);
    cv::waitKey(0);
}

void test_parameter_effects(const cv::Mat& src) {
    cv::Mat dst;

    std::cout << "\nParameter Effects Test:" << std::endl;

    // 测试不同伽马值
    std::vector<double> gamma_values = {0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0};
    for (double gamma : gamma_values) {
        auto start = std::chrono::high_resolution_clock::now();
        ip101::advanced::standard_gamma_correction(src, dst, gamma);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Gamma " << gamma << ": " << duration.count() << "ms" << std::endl;
        cv::imwrite("gamma_" + std::to_string(gamma) + ".jpg", dst);
    }
}

void test_display_correction(const cv::Mat& src) {
    cv::Mat dst;

    // 标准显示伽马校正
    ip101::advanced::standard_gamma_correction(src, dst, 2.2);

    // 保存结果
    cv::imwrite("gamma_display_corrected.jpg", dst);
    cv::imwrite("gamma_original.jpg", src);

    std::cout << "\nDisplay Correction Test:" << std::endl;
    std::cout << "Original image saved as: gamma_original.jpg" << std::endl;
    std::cout << "Display corrected image saved as: gamma_display_corrected.jpg" << std::endl;
}

void test_histogram_analysis(const cv::Mat& src) {
    cv::Mat dst;

    // 应用伽马校正
    ip101::advanced::standard_gamma_correction(src, dst, 2.2);

    // 分析直方图
    cv::Mat gray_src, gray_dst;
    cv::cvtColor(src, gray_src, cv::COLOR_BGR2GRAY);
    cv::cvtColor(dst, gray_dst, cv::COLOR_BGR2GRAY);

    cv::Scalar mean_src = cv::mean(gray_src);
    cv::Scalar mean_dst = cv::mean(gray_dst);
    cv::Scalar stddev_src, stddev_dst;
    cv::meanStdDev(gray_src, cv::noArray(), stddev_src);
    cv::meanStdDev(gray_dst, cv::noArray(), stddev_dst);

    std::cout << "\nHistogram Analysis:" << std::endl;
    std::cout << "Original - Mean: " << mean_src[0] << " StdDev: " << stddev_src[0] << std::endl;
    std::cout << "Corrected - Mean: " << mean_dst[0] << " StdDev: " << stddev_dst[0] << std::endl;

    // 创建直方图对比
    cv::Mat hist_src, hist_dst;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};

    cv::calcHist(&gray_src, 1, 0, cv::Mat(), hist_src, 1, &histSize, &histRange);
    cv::calcHist(&gray_dst, 1, 0, cv::Mat(), hist_dst, 1, &histSize, &histRange);

    cv::normalize(hist_src, hist_src, 0, 255, cv::NORM_MINMAX);
    cv::normalize(hist_dst, hist_dst, 0, 255, cv::NORM_MINMAX);

    cv::Mat histImage(256, 512, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int i = 0; i < 256; i++) {
        cv::line(histImage, cv::Point(i*2, 255), cv::Point(i*2, 255 - hist_src.at<float>(i)), cv::Scalar(255, 255, 255));
        cv::line(histImage, cv::Point(i*2+1, 255), cv::Point(i*2+1, 255 - hist_dst.at<float>(i)), cv::Scalar(0, 255, 0));
    }

    cv::imwrite("gamma_histogram_comparison.jpg", histImage);
    std::cout << "Histogram comparison saved as: gamma_histogram_comparison.jpg" << std::endl;
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

    // Test display correction
    test_display_correction(src);

    // Test histogram analysis
    test_histogram_analysis(src);

    // Visualize results
    visualize_results(src);

    return 0;
}
