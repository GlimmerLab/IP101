#include "filtering.hpp"
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

    // 测试均值滤波
    std::cout << "均值滤波性能测试：\n";
    double custom_time = measure_time([&]() {
        ip101::mean_filter(src, dst);
    });
    double opencv_time = measure_time([&]() {
        cv::blur(src, opencv_dst, cv::Size(3, 3));
    });
    std::cout << "自定义实现：" << custom_time << "ms\n";
    std::cout << "OpenCV实现：" << opencv_time << "ms\n";
    std::cout << "性能比：" << opencv_time / custom_time << "\n\n";

    // 测试中值滤波
    std::cout << "中值滤波性能测试：\n";
    custom_time = measure_time([&]() {
        ip101::median_filter(src, dst);
    });
    opencv_time = measure_time([&]() {
        cv::medianBlur(src, opencv_dst, 3);
    });
    std::cout << "自定义实现：" << custom_time << "ms\n";
    std::cout << "OpenCV实现：" << opencv_time << "ms\n";
    std::cout << "性能比：" << opencv_time / custom_time << "\n\n";

    // 测试高斯滤波
    std::cout << "高斯滤波性能测试：\n";
    custom_time = measure_time([&]() {
        ip101::gaussian_filter(src, dst);
    });
    opencv_time = measure_time([&]() {
        cv::GaussianBlur(src, opencv_dst, cv::Size(3, 3), 1.0);
    });
    std::cout << "自定义实现：" << custom_time << "ms\n";
    std::cout << "OpenCV实现：" << opencv_time << "ms\n";
    std::cout << "性能比：" << opencv_time / custom_time << "\n\n";

    // 测试均值池化
    std::cout << "均值池化性能测试：\n";
    custom_time = measure_time([&]() {
        ip101::mean_pooling(src, dst);
    });
    opencv_time = measure_time([&]() {
        cv::resize(src, opencv_dst, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
    });
    std::cout << "自定义实现：" << custom_time << "ms\n";
    std::cout << "OpenCV实现：" << opencv_time << "ms\n";
    std::cout << "性能比：" << opencv_time / custom_time << "\n\n";

    // 测试最大池化
    std::cout << "最大池化性能测试：\n";
    custom_time = measure_time([&]() {
        ip101::max_pooling(src, dst);
    });
    std::cout << "自定义实现：" << custom_time << "ms\n";
    // OpenCV没有直接的最大池化实现，所以不进行对比
}

void visualize_results(const cv::Mat& src) {
    cv::Mat results[5];

    // 应用不同的滤波操作
    ip101::mean_filter(src, results[0]);
    ip101::median_filter(src, results[1]);
    ip101::gaussian_filter(src, results[2]);
    ip101::mean_pooling(src, results[3]);
    ip101::max_pooling(src, results[4]);

    // 创建结果展示窗口
    cv::Mat display;
    int gap = 10;
    int cols = 3;
    int rows = 2;
    display.create(src.rows * rows + gap * (rows - 1),
                  src.cols * cols + gap * (cols - 1),
                  src.type());
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
    cv::imshow("滤波操作结果对比", display);
    cv::waitKey(0);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "用法: " << argv[0] << " <图像路径>\n";
        return -1;
    }

    // 读取图像
    cv::Mat src = cv::imread(argv[1]);
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