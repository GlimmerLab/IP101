#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>

// IP101 头文件
#include "advanced/enhancement/automatic_color_equalization.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    cout << "=== 自动色彩均衡算法测试 ===" << endl;

    // 检查参数
    if (argc < 2) {
        cout << "用法: " << argv[0] << " <输入图像路径>" << endl;
        cout << "示例: " << argv[0] << " assets/imori.jpg" << endl;
        return -1;
    }

    string input_path = argv[1];

    // 读取图像
    Mat src = imread(input_path);
    if (src.empty()) {
        cout << "错误: 无法读取图像 " << input_path << endl;
        return -1;
    }

    cout << "输入图像尺寸: " << src.size() << endl;
    cout << "输入图像类型: " << src.type() << endl;

    // 创建输出目录
    filesystem::create_directories("output/automatic_color_equalization");

    // ==================== 基础功能测试 ====================
    cout << "\n--- 基础功能测试 ---" << endl;

    Mat dst_ace;

    // 测试不同参数组合
    vector<pair<double, double>> param_combinations = {
        {0.5, 0.5},   // 低饱和度，低对比度
        {0.7, 0.7},   // 中等饱和度，中等对比度
        {0.85, 0.85}, // 高饱和度，高对比度
        {1.0, 1.0}    // 极高饱和度，极高对比度
    };

    for (size_t i = 0; i < param_combinations.size(); ++i) {
        double alpha = param_combinations[i].first;
        double beta = param_combinations[i].second;

        cout << "测试参数 - alpha: " << alpha << ", beta: " << beta << endl;

        Mat result;
        ip101::advanced::automatic_color_equalization(src, result, alpha, beta);

        string filename = "output/automatic_color_equalization/result_a" +
                         to_string(int(alpha * 100)) + "_b" + to_string(int(beta * 100)) + ".jpg";
        imwrite(filename, result);

        if (i == 2) { // 保存默认参数的结果
            dst_ace = result.clone();
        }
    }

    // ==================== 性能测试 ====================
    cout << "\n--- 性能测试 ---" << endl;

    // 测试IP101实现性能
    Mat test_result;
    int iterations = 10;

    auto start_time = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        ip101::advanced::automatic_color_equalization(src, test_result, 0.85, 0.85);
    }
    auto end_time = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
    double avg_time = duration.count() / (double)iterations;

    cout << "IP101实现平均处理时间: " << fixed << setprecision(2)
         << avg_time / 1000.0 << " ms" << endl;
    cout << "处理速度: " << fixed << setprecision(2)
         << 1000.0 / avg_time << " FPS" << endl;

    // ==================== 参数效果测试 ====================
    cout << "\n--- 参数效果测试 ---" << endl;

    // 测试alpha参数的影响
    cout << "测试alpha参数影响 (beta=0.85):" << endl;
    for (double alpha = 0.1; alpha <= 1.0; alpha += 0.2) {
        Mat alpha_result;
        ip101::advanced::automatic_color_equalization(src, alpha_result, alpha, 0.85);

        string filename = "output/automatic_color_equalization/alpha_" +
                         to_string(int(alpha * 100)) + ".jpg";
        imwrite(filename, alpha_result);

        cout << "  alpha=" << alpha << " - saved" << endl;
    }

    // Test the effect of beta parameter
    cout << "Testing beta parameter effect (alpha=0.85):" << endl;
    for (double beta = 0.1; beta <= 1.0; beta += 0.2) {
        Mat beta_result;
        ip101::advanced::automatic_color_equalization(src, beta_result, 0.85, beta);

        string filename = "output/automatic_color_equalization/beta_" +
                         to_string(int(beta * 100)) + ".jpg";
        imwrite(filename, beta_result);

        cout << "  beta=" << beta << " - saved" << endl;
    }

    // ==================== Special Scenario Test ====================
    cout << "\n--- Special Scenario Test ---" << endl;

    // Create color cast image
    Mat color_cast = src.clone();
    color_cast.convertTo(color_cast, -1, 1.0, 30);
    imwrite("output/automatic_color_equalization/color_cast.jpg", color_cast);

    // Apply automatic color equalization
    Mat color_cast_result;
    ip101::advanced::automatic_color_equalization(color_cast, color_cast_result, 0.85, 0.85);
    imwrite("output/automatic_color_equalization/color_cast_result.jpg", color_cast_result);

    // Create low saturation image
    Mat low_saturation = src.clone();
    cvtColor(low_saturation, low_saturation, COLOR_BGR2HSV);
    vector<Mat> hsv_planes;
    split(low_saturation, hsv_planes);
    hsv_planes[1] *= 0.3; // Reduce saturation
    merge(hsv_planes, low_saturation);
    cvtColor(low_saturation, low_saturation, COLOR_HSV2BGR);
    imwrite("output/automatic_color_equalization/low_saturation.jpg", low_saturation);

    // Apply automatic color equalization
    Mat low_saturation_result;
    ip101::advanced::automatic_color_equalization(low_saturation, low_saturation_result, 0.85, 0.85);
    imwrite("output/automatic_color_equalization/low_saturation_result.jpg", low_saturation_result);

    // ==================== Quality Assessment ====================
    cout << "\n--- Quality Assessment ---" << endl;

    // Calculate saturation enhancement effect
    Mat hsv_src, hsv_result;
    cvtColor(src, hsv_src, COLOR_BGR2HSV);
    cvtColor(dst_ace, hsv_result, COLOR_BGR2HSV);

    vector<Mat> hsv_planes_src, hsv_planes_result;
    split(hsv_src, hsv_planes_src);
    split(hsv_result, hsv_planes_result);

    Scalar mean_saturation_src, std_saturation_src, mean_saturation_result, std_saturation_result;
    meanStdDev(hsv_planes_src[1], mean_saturation_src, std_saturation_src);
    meanStdDev(hsv_planes_result[1], mean_saturation_result, std_saturation_result);

    cout << "Original - Saturation mean: " << mean_saturation_src[0] << ", Std Dev: " << std_saturation_src[0] << endl;
    cout << "Processed - Saturation mean: " << mean_saturation_result[0] << ", Std Dev: " << std_saturation_result[0] << endl;
    cout << "Saturation enhancement factor: " << mean_saturation_result[0] / mean_saturation_src[0] << endl;

    // Calculate PSNR
    Mat diff;
    absdiff(src, dst_ace, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);

    double mse = mean(diff)[0];
    double psnr = 10.0 * log10((255.0 * 255.0) / mse);
    cout << "PSNR: " << fixed << setprecision(2) << psnr << " dB" << endl;

    // ==================== 直方图分析 ====================
    cout << "\n--- 直方图分析 ---" << endl;

    // 计算原图和增强后图像的直方图
    vector<Mat> bgr_planes;
    split(src, bgr_planes);

    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};

    Mat b_hist, g_hist, r_hist;
    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange);

    // 计算增强后图像的直方图
    vector<Mat> bgr_planes_result;
    split(dst_ace, bgr_planes_result);

    Mat b_hist_result, g_hist_result, r_hist_result;
    calcHist(&bgr_planes_result[0], 1, 0, Mat(), b_hist_result, 1, &histSize, &histRange);
    calcHist(&bgr_planes_result[1], 1, 0, Mat(), g_hist_result, 1, &histSize, &histRange);
    calcHist(&bgr_planes_result[2], 1, 0, Mat(), r_hist_result, 1, &histSize, &histRange);

    // 计算直方图相似度
    double b_similarity = compareHist(b_hist, b_hist_result, HISTCMP_CORREL);
    double g_similarity = compareHist(g_hist, g_hist_result, HISTCMP_CORREL);
    double r_similarity = compareHist(r_hist, r_hist_result, HISTCMP_CORREL);

    cout << "直方图相似度 - B: " << fixed << setprecision(3) << b_similarity
         << ", G: " << g_similarity << ", R: " << r_similarity << endl;

    // ==================== Lab色彩空间分析 ====================
    cout << "\n--- Lab色彩空间分析 ---" << endl;

    // 转换到Lab色彩空间
    Mat lab_src, lab_result;
    cvtColor(src, lab_src, COLOR_BGR2Lab);
    cvtColor(dst_ace, lab_result, COLOR_BGR2Lab);

    vector<Mat> lab_planes_src, lab_planes_result;
    split(lab_src, lab_planes_src);
    split(lab_result, lab_planes_result);

    // 计算a和b通道的均值
    Scalar mean_a_src, mean_b_src, mean_a_result, mean_b_result;
    meanStdDev(lab_planes_src[1], mean_a_src, Scalar());
    meanStdDev(lab_planes_src[2], mean_b_src, Scalar());
    meanStdDev(lab_planes_result[1], mean_a_result, Scalar());
    meanStdDev(lab_planes_result[2], mean_b_result, Scalar());

    cout << "Original - a channel mean: " << mean_a_src[0] << ", b channel mean: " << mean_b_src[0] << endl;
    cout << "Processed - a channel mean: " << mean_a_result[0] << ", b channel mean: " << mean_b_result[0] << endl;

    // ==================== Save Results ====================
    cout << "\n--- Save Results ---" << endl;

    imwrite("output/automatic_color_equalization/original.jpg", src);
    imwrite("output/automatic_color_equalization/result.jpg", dst_ace);

    cout << "All results have been saved to output/automatic_color_equalization/ directory" << endl;
    cout << "Automatic color equalization algorithm test completed!" << endl;

    return 0;
}
