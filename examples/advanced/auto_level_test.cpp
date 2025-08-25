#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <iomanip>

// 包含IP101算法头文件
#include "advanced/correction/auto_level.hpp"

using namespace cv;
using namespace std;

/**
 * @brief 自动色阶调整算法测试
 *
 * 测试功能：
 * 1. 性能测试 - 与OpenCV对比
 * 2. 参数效果测试 - 不同裁剪百分比和通道处理方式
 * 3. 可视化结果 - 显示原图、自动色阶、自动对比度结果
 * 4. 特殊场景测试 - 低对比度、高对比度图像
 */
int main(int argc, char** argv) {
    cout << "=== Auto Level Adjustment Algorithm Test ===" << endl;

    // 加载测试图像
    string image_path = (argc > 1) ? argv[1] : "assets/imori.jpg";
    Mat src = imread(image_path);
    if (src.empty()) {
        cerr << "错误：无法加载图像 " << image_path << endl;
        return -1;
    }

    cout << "Image size: " << src.size() << endl;
    cout << "Image type: " << src.type() << endl;

    // 创建输出目录
    system("mkdir -p output/auto_level");

    // ==================== 性能测试 ====================
    cout << "\n--- Performance Test ---" << endl;

    Mat dst_ip101, dst_opencv;
    int iterations = 10;

    // IP101自动色阶测试
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        ip101::advanced::auto_level(src, dst_ip101, 0.5f, true);
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration_ip101 = chrono::duration_cast<chrono::microseconds>(end - start);

    // OpenCV对比度增强测试（作为对比）
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        Mat lab;
        cvtColor(src, lab, COLOR_BGR2Lab);
        vector<Mat> lab_planes(3);
        split(lab, lab_planes);

        // 对L通道进行直方图均衡化
        equalizeHist(lab_planes[0], lab_planes[0]);

        Mat enhanced_lab;
        merge(lab_planes, enhanced_lab);
        cvtColor(enhanced_lab, dst_opencv, COLOR_Lab2BGR);
    }
    end = chrono::high_resolution_clock::now();
    auto duration_opencv = chrono::duration_cast<chrono::microseconds>(end - start);

    cout << "IP101 Auto Level average time: " << duration_ip101.count() / iterations << " μs" << endl;
    cout << "OpenCV Contrast Enhancement average time: " << duration_opencv.count() / iterations << " μs" << endl;
    cout << "Performance ratio: " << fixed << setprecision(2)
         << (double)duration_opencv.count() / duration_ip101.count() << "x" << endl;

    // ==================== 参数效果测试 ====================
    cout << "\n--- Parameter Effect Test ---" << endl;

    vector<float> clip_percentages = {0.1f, 0.5f, 1.0f, 2.0f, 5.0f};
    vector<bool> separate_channels = {true, false};

    for (bool separate : separate_channels) {
        for (float clip : clip_percentages) {
            Mat result;
            ip101::advanced::auto_level(src, result, clip, separate);

            string filename = "output/auto_level/auto_level_clip" +
                            to_string((int)(clip * 10)) + "_" +
                            (separate ? "separate" : "combined") + ".jpg";
            imwrite(filename, result);

            cout << "Saved: " << filename << " (clip=" << clip << "%, "
                 << (separate ? "separate channels" : "merged channels") << ")" << endl;
        }
    }

    // 自动对比度测试
    for (float clip : clip_percentages) {
        Mat result;
        ip101::advanced::auto_contrast(src, result, clip, false);

        string filename = "output/auto_level/auto_contrast_clip" +
                         to_string((int)(clip * 10)) + ".jpg";
        imwrite(filename, result);

        cout << "Saved: " << filename << " (clip=" << clip << "%)" << endl;
    }

    // ==================== 可视化结果 ====================
    cout << "\n--- Visualization Results ---" << endl;

    Mat auto_level_result, auto_contrast_result;
    ip101::advanced::auto_level(src, auto_level_result, 0.5f, true);
    ip101::advanced::auto_contrast(src, auto_contrast_result, 0.5f, false);

    // 创建对比图
    // Ensure all images have the same size for hconcat
    Mat auto_level_result_resized, auto_contrast_result_resized;
    resize(auto_level_result, auto_level_result_resized, src.size());
    resize(auto_contrast_result, auto_contrast_result_resized, src.size());

    Mat comparison;
    vector<Mat> images = {src, auto_level_result_resized, auto_contrast_result_resized};
    hconcat(images, comparison);

    // 添加标题
    Mat comparison_with_titles;
    comparison.copyTo(comparison_with_titles);
    putText(comparison_with_titles, "Original", Point(10, 30),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
    putText(comparison_with_titles, "Auto Level", Point(src.cols + 10, 30),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
    putText(comparison_with_titles, "Auto Contrast", Point(2 * src.cols + 10, 30),
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);

    imwrite("output/auto_level/comparison.jpg", comparison_with_titles);
    cout << "Comparison image saved: output/auto_level/comparison.jpg" << endl;

    // ==================== 特殊场景测试 ====================
    cout << "\n--- Special Scenario Test ---" << endl;

    // 创建低对比度图像
    Mat low_contrast = src.clone();
    low_contrast.convertTo(low_contrast, -1, 0.5, 50); // 降低对比度，增加亮度

    Mat low_contrast_result;
    ip101::advanced::auto_level(low_contrast, low_contrast_result, 0.5f, true);

    // 创建高对比度图像
    Mat high_contrast = src.clone();
    high_contrast.convertTo(high_contrast, -1, 2.0, -50); // 增加对比度，降低亮度

    Mat high_contrast_result;
    ip101::advanced::auto_level(high_contrast, high_contrast_result, 0.5f, true);

    // 保存特殊场景结果
    imwrite("output/auto_level/low_contrast_original.jpg", low_contrast);
    imwrite("output/auto_level/low_contrast_result.jpg", low_contrast_result);
    imwrite("output/auto_level/high_contrast_original.jpg", high_contrast);
    imwrite("output/auto_level/high_contrast_result.jpg", high_contrast_result);

    cout << "Special scenario test results saved" << endl;

    // ==================== 直方图分析 ====================
    cout << "\n--- Histogram Analysis ---" << endl;

    // 计算并显示直方图
    vector<Mat> histograms;
    vector<string> titles = {"Original", "Auto Level", "Auto Contrast"};
    vector<Mat> test_images = {src, auto_level_result, auto_contrast_result};

    for (size_t i = 0; i < test_images.size(); ++i) {
        Mat hist;
        vector<Mat> bgr_planes;
        split(test_images[i], bgr_planes);

        int histSize = 256;
        float range[] = {0, 256};
        const float* histRange = {range};

        Mat b_hist, g_hist, r_hist;
        calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange);
        calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange);
        calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange);

        // 归一化直方图
        normalize(b_hist, b_hist, 0, 255, NORM_MINMAX);
        normalize(g_hist, g_hist, 0, 255, NORM_MINMAX);
        normalize(r_hist, r_hist, 0, 255, NORM_MINMAX);

        // 创建直方图图像
        int hist_w = 256, hist_h = 200;
        Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

        for (int j = 1; j < histSize; j++) {
            line(histImage, Point(j-1, hist_h - cvRound(b_hist.at<float>(j-1))),
                 Point(j, hist_h - cvRound(b_hist.at<float>(j))), Scalar(255, 0, 0), 2);
            line(histImage, Point(j-1, hist_h - cvRound(g_hist.at<float>(j-1))),
                 Point(j, hist_h - cvRound(g_hist.at<float>(j))), Scalar(0, 255, 0), 2);
            line(histImage, Point(j-1, hist_h - cvRound(r_hist.at<float>(j-1))),
                 Point(j, hist_h - cvRound(r_hist.at<float>(j))), Scalar(0, 0, 255), 2);
        }

        putText(histImage, titles[i], Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
        histograms.push_back(histImage);
    }

    // 合并直方图
    Mat hist_comparison;
    hconcat(histograms, hist_comparison);
    imwrite("output/auto_level/histogram_comparison.jpg", hist_comparison);
    cout << "Histogram comparison saved: output/auto_level/histogram_comparison.jpg" << endl;

    cout << "\n=== Test Completed ===" << endl;
    cout << "All results saved to output/auto_level/ directory" << endl;

    return 0;
}
