#include "edge_detection.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

// 性能测试辅助函数
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

    // 测试Sobel
    std::cout << "Sobel算子性能测试：\n";
    double custom_time = measure_time([&]() {
        ip101::sobel_filter(src, dst, 1, 1);
    });
    double opencv_time = measure_time([&]() {
        cv::Sobel(src, opencv_dst, CV_8U, 1, 1);
    });
    std::cout << "自定义实现：" << custom_time << "ms\n";
    std::cout << "OpenCV实现：" << opencv_time << "ms\n";
    std::cout << "性能比：" << opencv_time / custom_time << "\n\n";

    // 测试Prewitt
    std::cout << "Prewitt算子性能测试：\n";
    custom_time = measure_time([&]() {
        ip101::prewitt_filter(src, dst, 1, 1);
    });
    std::cout << "自定义实现：" << custom_time << "ms\n\n";

    // 测试Laplacian
    std::cout << "Laplacian算子性能测试：\n";
    custom_time = measure_time([&]() {
        ip101::laplacian_filter(src, dst);
    });
    opencv_time = measure_time([&]() {
        cv::Laplacian(src, opencv_dst, CV_8U);
    });
    std::cout << "自定义实现：" << custom_time << "ms\n";
    std::cout << "OpenCV实现：" << opencv_time << "ms\n";
    std::cout << "性能比：" << opencv_time / custom_time << "\n\n";
}

void visualize_results(const cv::Mat& src) {
    cv::Mat results[5];

    // 应用不同的边缘检测方法
    ip101::sobel_filter(src, results[0], 1, 1);
    ip101::prewitt_filter(src, results[1], 1, 1);
    ip101::laplacian_filter(src, results[2]);
    ip101::differential_filter(src, results[3], 1, 1);
    ip101::emboss_effect(src, results[4]);

    // 创建结果展示窗口
    cv::Mat display;
    int gap = 10;
    int cols = 3;
    int rows = 2;
    display.create(src.rows * rows + gap * (rows - 1),
                  src.cols * cols + gap * (cols - 1),
                  CV_8UC1);
    display.setTo(0);

    // 组织显示布局
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

    // 显示结果
    cv::imshow("边缘检测结果对比", display);
    cv::waitKey(0);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "用法: " << argv[0] << " <图像路径>\n";
        return -1;
    }

    // 读取图像
    cv::Mat src = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cout << "无法读取图像: " << argv[1] << "\n";
        return -1;
    }

    // 运行性能测试
    std::cout << "图像尺寸: " << src.cols << "x" << src.rows << "\n\n";
    test_performance(src);

    // 可视化结果
    visualize_results(src);

    return 0;
}