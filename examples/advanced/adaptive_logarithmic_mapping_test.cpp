#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>

// IP101 头文件
#include "advanced/enhancement/adaptive_logarithmic_mapping.hpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
    cout << "=== Adaptive Logarithmic Mapping Algorithm Test ===" << endl;

    // Check parameters
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <input image path>" << endl;
        cout << "Example: " << argv[0] << " assets/imori.jpg" << endl;
        return -1;
    }

    string input_path = argv[1];

    // Read image
    Mat src = imread(input_path);
    if (src.empty()) {
        cout << "Error: Cannot read image " << input_path << endl;
        return -1;
    }

    cout << "Input image size: " << src.size() << endl;
    cout << "Input image type: " << src.type() << endl;

    // Create output directory
    filesystem::create_directories("output/adaptive_logarithmic_mapping");

    // ==================== Basic Functionality Test ====================
    cout << "\n--- Basic Functionality Test ---" << endl;

    Mat dst_adaptive_log;

    // Test different parameter combinations
    vector<pair<double, double>> param_combinations = {
        {0.5, 50.0},   // Low contrast, low brightness
        {0.7, 75.0},   // Medium contrast, medium brightness
        {0.85, 100.0}, // High contrast, high brightness
        {0.9, 120.0}   // Very high contrast, very high brightness
    };

    for (size_t i = 0; i < param_combinations.size(); ++i) {
        double bias = param_combinations[i].first;
        double scale = param_combinations[i].second;

        cout << "Testing parameters - bias: " << bias << ", scale: " << scale << endl;

        Mat result;
        ip101::advanced::adaptive_logarithmic_mapping(src, result, bias, scale);

        string filename = "output/adaptive_logarithmic_mapping/result_b" +
                         to_string(int(bias * 100)) + "_s" + to_string(int(scale)) + ".jpg";
        imwrite(filename, result);

        if (i == 2) { // Save result with default parameters
            dst_adaptive_log = result.clone();
        }
    }

    // ==================== Performance Test ====================
    cout << "\n--- Performance Test ---" << endl;

    // Test IP101 implementation performance
    Mat test_result;
    int iterations = 10;

    auto start_time = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        ip101::advanced::adaptive_logarithmic_mapping(src, test_result, 0.85, 100.0);
    }
    auto end_time = chrono::high_resolution_clock::now();

    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
    double avg_time = duration.count() / (double)iterations;

    cout << "IP101 implementation average processing time: " << fixed << setprecision(2)
         << avg_time / 1000.0 << " ms" << endl;
    cout << "Processing speed: " << fixed << setprecision(2)
         << 1000.0 / avg_time << " FPS" << endl;

    // ==================== Parameter Effect Test ====================
    cout << "\n--- Parameter Effect Test ---" << endl;

    // Test the effect of bias parameter
    cout << "Testing bias parameter effect (scale=100.0):" << endl;
    for (double bias = 0.1; bias <= 0.9; bias += 0.2) {
        Mat bias_result;
        ip101::advanced::adaptive_logarithmic_mapping(src, bias_result, bias, 100.0);

        string filename = "output/adaptive_logarithmic_mapping/bias_" +
                         to_string(int(bias * 100)) + ".jpg";
        imwrite(filename, bias_result);

        cout << "  bias=" << bias << " - saved" << endl;
    }

    // Test the effect of scale parameter
    cout << "Testing scale parameter effect (bias=0.85):" << endl;
    for (double scale = 25.0; scale <= 200.0; scale += 25.0) {
        Mat scale_result;
        ip101::advanced::adaptive_logarithmic_mapping(src, scale_result, 0.85, scale);

        string filename = "output/adaptive_logarithmic_mapping/scale_" +
                         to_string(int(scale)) + ".jpg";
        imwrite(filename, scale_result);

        cout << "  scale=" << scale << " - saved" << endl;
    }

    // ==================== Special Scenario Test ====================
    cout << "\n--- Special Scenario Test ---" << endl;

    // Create high contrast image
    Mat high_contrast = src.clone();
    high_contrast.convertTo(high_contrast, -1, 2.0, 50);
    imwrite("output/adaptive_logarithmic_mapping/high_contrast.jpg", high_contrast);

    // Apply adaptive logarithmic mapping
    Mat high_contrast_result;
    ip101::advanced::adaptive_logarithmic_mapping(high_contrast, high_contrast_result, 0.85, 100.0);
    imwrite("output/adaptive_logarithmic_mapping/high_contrast_result.jpg", high_contrast_result);

    // Create dark image
    Mat dark_image = src.clone();
    dark_image.convertTo(dark_image, -1, 0.3, 0);
    imwrite("output/adaptive_logarithmic_mapping/dark_image.jpg", dark_image);

    // Apply adaptive logarithmic mapping
    Mat dark_result;
    ip101::advanced::adaptive_logarithmic_mapping(dark_image, dark_result, 0.85, 100.0);
    imwrite("output/adaptive_logarithmic_mapping/dark_result.jpg", dark_result);

    // ==================== Quality Assessment ====================
    cout << "\n--- Quality Assessment ---" << endl;

    // Calculate contrast enhancement effect
    Mat gray_src, gray_result;
    cvtColor(src, gray_src, COLOR_BGR2GRAY);
    cvtColor(dst_adaptive_log, gray_result, COLOR_BGR2GRAY);

    Scalar mean_src, std_src, mean_result, std_result;
    meanStdDev(gray_src, mean_src, std_src);
    meanStdDev(gray_result, mean_result, std_result);

    cout << "Original - Mean: " << mean_src[0] << ", Std Dev: " << std_src[0] << endl;
    cout << "Processed - Mean: " << mean_result[0] << ", Std Dev: " << std_result[0] << endl;
    cout << "Contrast enhancement factor: " << std_result[0] / std_src[0] << endl;

    // Calculate PSNR
    Mat diff;
    absdiff(src, dst_adaptive_log, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);

    double mse = mean(diff)[0];
    double psnr = 10.0 * log10((255.0 * 255.0) / mse);
    cout << "PSNR: " << fixed << setprecision(2) << psnr << " dB" << endl;

    // ==================== Histogram Analysis ====================
    cout << "\n--- Histogram Analysis ---" << endl;

    // Calculate histograms for original and enhanced images
    vector<Mat> bgr_planes;
    split(src, bgr_planes);

    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};

    Mat b_hist, g_hist, r_hist;
    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange);

    // Calculate histogram for enhanced image
    vector<Mat> bgr_planes_result;
    split(dst_adaptive_log, bgr_planes_result);

    Mat b_hist_result, g_hist_result, r_hist_result;
    calcHist(&bgr_planes_result[0], 1, 0, Mat(), b_hist_result, 1, &histSize, &histRange);
    calcHist(&bgr_planes_result[1], 1, 0, Mat(), g_hist_result, 1, &histSize, &histRange);
    calcHist(&bgr_planes_result[2], 1, 0, Mat(), r_hist_result, 1, &histSize, &histRange);

    // Calculate histogram similarity
    double b_similarity = compareHist(b_hist, b_hist_result, HISTCMP_CORREL);
    double g_similarity = compareHist(g_hist, g_hist_result, HISTCMP_CORREL);
    double r_similarity = compareHist(r_hist, r_hist_result, HISTCMP_CORREL);

    cout << "Histogram similarity - B: " << fixed << setprecision(3) << b_similarity
         << ", G: " << g_similarity << ", R: " << r_similarity << endl;

    // ==================== Dynamic Range Analysis ====================
    cout << "\n--- Dynamic Range Analysis ---" << endl;

    // Calculate the dynamic range of the original image
    double min_val_src, max_val_src, min_val_result, max_val_result;
    minMaxLoc(gray_src, &min_val_src, &max_val_src);
    minMaxLoc(gray_result, &min_val_result, &max_val_result);

    cout << "Original dynamic range: " << min_val_src << " - " << max_val_src
         << " (range: " << max_val_src - min_val_src << ")" << endl;
    cout << "Processed dynamic range: " << min_val_result << " - " << max_val_result
         << " (range: " << max_val_result - min_val_result << ")" << endl;

    // ==================== Save Results ====================
    cout << "\n--- Save Results ---" << endl;

    imwrite("output/adaptive_logarithmic_mapping/original.jpg", src);
    imwrite("output/adaptive_logarithmic_mapping/result.jpg", dst_adaptive_log);

    cout << "All results have been saved to output/adaptive_logarithmic_mapping/ directory" << endl;
    cout << "Adaptive logarithmic mapping algorithm test completed!" << endl;

    return 0;
}
