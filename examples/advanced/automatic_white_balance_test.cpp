#include <advanced/correction/automatic_white_balance.hpp>
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

    std::cout << "Automatic White Balance Performance Test:" << std::endl;
    double custom_time = measure_time([&]() {
        ip101::advanced::automatic_white_balance(src, dst);
    });
    std::cout << "Automatic White Balance: " << custom_time << "ms" << std::endl;

    // 与OpenCV对比
    cv::Mat opencv_dst;
    double opencv_time = measure_time([&]() {
        // OpenCV的简单白平衡方法
        std::vector<cv::Mat> channels;
        cv::split(src, channels);

        double avg_b = cv::mean(channels[0])[0];
        double avg_g = cv::mean(channels[1])[0];
        double avg_r = cv::mean(channels[2])[0];

        double avg_gray = (avg_b + avg_g + avg_r) / 3.0;

        channels[0] = channels[0] * (avg_gray / avg_b);
        channels[1] = channels[1] * (avg_gray / avg_g);
        channels[2] = channels[2] * (avg_gray / avg_r);

        cv::merge(channels, opencv_dst);
        cv::convertScaleAbs(opencv_dst, opencv_dst);
    });
    std::cout << "OpenCV Simple WB: " << opencv_time << "ms" << std::endl;
    std::cout << "Performance Ratio: " << opencv_time / custom_time << std::endl << std::endl;
}

void visualize_results(const cv::Mat& src) {
    cv::Mat results[4];

    // 应用不同的白平衡方法
    ip101::advanced::automatic_white_balance(src, results[0]);

    // 创建不同色温的图像进行测试
    cv::Mat warm_image = src.clone();
    cv::Mat cool_image = src.clone();

    // 暖色调（增加红色，减少蓝色）
    std::vector<cv::Mat> channels;
    cv::split(warm_image, channels);
    channels[2] = cv::min(channels[2] * 1.2, 255.0);  // 增加红色
    channels[0] = cv::max(channels[0] * 0.8, 0.0);    // 减少蓝色
    cv::merge(channels, warm_image);

    // 冷色调（增加蓝色，减少红色）
    cv::split(cool_image, channels);
    channels[0] = cv::min(channels[0] * 1.2, 255.0);  // 增加蓝色
    channels[2] = cv::max(channels[2] * 0.8, 0.0);    // 减少红色
    cv::merge(channels, cool_image);

    // 对色温偏移的图像应用白平衡
    ip101::advanced::automatic_white_balance(warm_image, results[1]);
    ip101::advanced::automatic_white_balance(cool_image, results[2]);

    // 原图作为对比
    results[3] = src.clone();

    // Create result display window
    cv::Mat display;
    int gap = 10;
    int cols = 2;
    int rows = 2;
    display.create(src.rows * rows + gap * (rows - 1),
                  src.cols * cols + gap * (cols - 1),
                  src.type());
    display.setTo(0);

    // Organize display layout
    cv::Mat roi;
    int idx = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols && idx < 4; j++) {
            roi = display(cv::Rect(j * (src.cols + gap),
                                 i * (src.rows + gap),
                                 src.cols, src.rows));
            results[idx].copyTo(roi);
            idx++;
        }
    }

    // Display results
    cv::imshow("Automatic White Balance Results", display);
    cv::waitKey(0);
}

void test_color_temperature_correction(const cv::Mat& src) {
    cv::Mat dst;

    // 应用自动白平衡
    ip101::advanced::automatic_white_balance(src, dst);

    // 保存结果
    cv::imwrite("awb_corrected.jpg", dst);
    cv::imwrite("awb_original.jpg", src);

    std::cout << "\nColor Temperature Correction Test:" << std::endl;
    std::cout << "Original image saved as: awb_original.jpg" << std::endl;
    std::cout << "White balance corrected image saved as: awb_corrected.jpg" << std::endl;
}

void test_different_lighting_conditions(const cv::Mat& src) {
    cv::Mat dst;

    // 创建不同光照条件下的图像
    std::vector<cv::Mat> lighting_variants;
    std::vector<std::string> lighting_names = {"tungsten", "fluorescent", "daylight", "shade"};

    // 钨丝灯（偏黄）
    cv::Mat tungsten = src.clone();
    std::vector<cv::Mat> channels;
    cv::split(tungsten, channels);
    channels[2] = cv::min(channels[2] * 1.3, 255.0);  // 增加红色
    channels[1] = cv::min(channels[1] * 1.1, 255.0);  // 增加绿色
    channels[0] = cv::max(channels[0] * 0.7, 0.0);    // 减少蓝色
    cv::merge(channels, tungsten);
    lighting_variants.push_back(tungsten);

    // 荧光灯（偏绿）
    cv::Mat fluorescent = src.clone();
    cv::split(fluorescent, channels);
    channels[1] = cv::min(channels[1] * 1.2, 255.0);  // 增加绿色
    channels[2] = cv::max(channels[2] * 0.9, 0.0);    // 减少红色
    channels[0] = cv::max(channels[0] * 0.8, 0.0);    // 减少蓝色
    cv::merge(channels, fluorescent);
    lighting_variants.push_back(fluorescent);

    // 日光（正常）
    lighting_variants.push_back(src.clone());

    // 阴影（偏蓝）
    cv::Mat shade = src.clone();
    cv::split(shade, channels);
    channels[0] = cv::min(channels[0] * 1.2, 255.0);  // 增加蓝色
    channels[2] = cv::max(channels[2] * 0.8, 0.0);    // 减少红色
    cv::merge(channels, shade);
    lighting_variants.push_back(shade);

    // 对每种光照条件应用白平衡
    for (size_t i = 0; i < lighting_variants.size(); ++i) {
        ip101::advanced::automatic_white_balance(lighting_variants[i], dst);
        cv::imwrite("awb_" + lighting_names[i] + "_corrected.jpg", dst);
        cv::imwrite("awb_" + lighting_names[i] + "_original.jpg", lighting_variants[i]);
    }

    std::cout << "\nDifferent Lighting Conditions Test:" << std::endl;
    for (const auto& name : lighting_names) {
        std::cout << name << " lighting corrected: awb_" << name << "_corrected.jpg" << std::endl;
    }
}

void test_histogram_analysis(const cv::Mat& src) {
    cv::Mat dst;
    ip101::advanced::automatic_white_balance(src, dst);

    // 分析RGB通道直方图
    std::vector<cv::Mat> src_channels, dst_channels;
    cv::split(src, src_channels);
    cv::split(dst, dst_channels);

    // 计算每个通道的平均值
    cv::Scalar src_means = cv::mean(src);
    cv::Scalar dst_means = cv::mean(dst);

    std::cout << "\nHistogram Analysis:" << std::endl;
    std::cout << "Original - B:" << src_means[0] << " G:" << src_means[1] << " R:" << src_means[2] << std::endl;
    std::cout << "Corrected - B:" << dst_means[0] << " G:" << dst_means[1] << " R:" << dst_means[2] << std::endl;

    // 保存直方图对比
    cv::Mat hist_src, hist_dst;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};

    cv::calcHist(&src, 1, 0, cv::Mat(), hist_src, 1, &histSize, &histRange);
    cv::calcHist(&dst, 1, 0, cv::Mat(), hist_dst, 1, &histSize, &histRange);

    // 归一化直方图
    cv::normalize(hist_src, hist_src, 0, 255, cv::NORM_MINMAX);
    cv::normalize(hist_dst, hist_dst, 0, 255, cv::NORM_MINMAX);

    // 绘制直方图
    cv::Mat histImage(256, 512, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int i = 0; i < 256; i++) {
        cv::line(histImage, cv::Point(i*2, 255), cv::Point(i*2, 255 - hist_src.at<float>(i)), cv::Scalar(255, 255, 255));
        cv::line(histImage, cv::Point(i*2+1, 255), cv::Point(i*2+1, 255 - hist_dst.at<float>(i)), cv::Scalar(0, 255, 0));
    }

    cv::imwrite("awb_histogram_comparison.jpg", histImage);
    std::cout << "Histogram comparison saved as: awb_histogram_comparison.jpg" << std::endl;
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

    // Test color temperature correction
    test_color_temperature_correction(src);

    // Test different lighting conditions
    test_different_lighting_conditions(src);

    // Test histogram analysis
    test_histogram_analysis(src);

    // Visualize results
    visualize_results(src);

    return 0;
}
