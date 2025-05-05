#include "color_operations.hpp"
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

    // 测试通道替换
    std::cout << "通道替换性能测试：\n";
    double custom_time = measure_time([&]() {
        ip101::channel_swap(src, dst, 2, 1, 0);  // BGR -> RGB
    });
    double opencv_time = measure_time([&]() {
        cv::cvtColor(src, opencv_dst, cv::COLOR_BGR2RGB);
    });
    std::cout << "自定义实现：" << custom_time << "ms\n";
    std::cout << "OpenCV实现：" << opencv_time << "ms\n";
    std::cout << "性能比：" << opencv_time / custom_time << "\n\n";

    // 测试灰度化
    std::cout << "灰度化性能测试：\n";
    custom_time = measure_time([&]() {
        ip101::to_gray(src, dst, "weighted");
    });
    opencv_time = measure_time([&]() {
        cv::cvtColor(src, opencv_dst, cv::COLOR_BGR2GRAY);
    });
    std::cout << "自定义实现：" << custom_time << "ms\n";
    std::cout << "OpenCV实现：" << opencv_time << "ms\n";
    std::cout << "性能比：" << opencv_time / custom_time << "\n\n";

    // 测试二值化
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    std::cout << "二值化性能测试：\n";
    custom_time = measure_time([&]() {
        ip101::threshold_image(gray, dst, 128);
    });
    opencv_time = measure_time([&]() {
        cv::threshold(gray, opencv_dst, 128, 255, cv::THRESH_BINARY);
    });
    std::cout << "自定义实现：" << custom_time << "ms\n";
    std::cout << "OpenCV实现：" << opencv_time << "ms\n";
    std::cout << "性能比：" << opencv_time / custom_time << "\n\n";

    // 测试大津算法
    std::cout << "大津算法性能测试：\n";
    custom_time = measure_time([&]() {
        ip101::otsu_threshold(gray, dst);
    });
    opencv_time = measure_time([&]() {
        cv::threshold(gray, opencv_dst, 0, 255, cv::THRESH_OTSU);
    });
    std::cout << "自定义实现：" << custom_time << "ms\n";
    std::cout << "OpenCV实现：" << opencv_time << "ms\n";
    std::cout << "性能比：" << opencv_time / custom_time << "\n\n";

    // 测试HSV转换
    std::cout << "HSV转换性能测试：\n";
    custom_time = measure_time([&]() {
        ip101::bgr_to_hsv(src, dst);
    });
    opencv_time = measure_time([&]() {
        cv::cvtColor(src, opencv_dst, cv::COLOR_BGR2HSV);
    });
    std::cout << "自定义实现：" << custom_time << "ms\n";
    std::cout << "OpenCV实现：" << opencv_time << "ms\n";
    std::cout << "性能比：" << opencv_time / custom_time << "\n\n";
}

void visualize_results(const cv::Mat& src) {
    cv::Mat results[6];

    // 应用不同的颜色操作
    ip101::channel_swap(src, results[0], 2, 1, 0);  // BGR -> RGB
    ip101::to_gray(src, results[1], "weighted");
    ip101::threshold_image(results[1], results[2], 128);
    ip101::otsu_threshold(results[1], results[3]);
    ip101::bgr_to_hsv(src, results[4]);

    // HSV调整测试
    cv::Mat hsv;
    ip101::bgr_to_hsv(src, hsv);
    ip101::adjust_hsv(hsv, hsv, 60, 1.2, 1.1);  // 色相偏移60度，饱和度增加20%，亮度增加10%
    ip101::hsv_to_bgr(hsv, results[5]);

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

    // 显示结果
    cv::imshow("颜色操作结果对比", display);
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