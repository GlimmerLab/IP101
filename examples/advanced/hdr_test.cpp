#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>

// Include HDR algorithm header
#include "advanced/enhancement/hdr.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    cout << "=== HDR Algorithm Test ===" << endl;

    // Load test image
    string image_path = (argc > 1) ? argv[1] : "assets/imori.jpg";
    Mat src = imread(image_path);

    if (src.empty()) {
        cerr << "Error: Cannot load image " << image_path << endl;
        return -1;
    }

    cout << "Image size: " << src.size() << endl;

    // Create output directory
    filesystem::create_directories("output/hdr");

    // Create simulated images with different exposures
    vector<Mat> exposure_images;
    vector<float> exposure_times = {0.25f, 0.5f, 1.0f, 2.0f, 4.0f};

    for (float exposure : exposure_times) {
        Mat exposed_image;
        src.convertTo(exposed_image, CV_32F);
        exposed_image = exposed_image * exposure;

        // Add some noise to simulate real shooting
        Mat noise = Mat::zeros(exposed_image.size(), CV_32F);
        randn(noise, 0, 2.0);
        exposed_image = exposed_image + noise;

        // Limit to valid range
        exposed_image = max(0.0f, min(255.0f, exposed_image));
        exposed_image.convertTo(exposed_image, CV_8U);

        exposure_images.push_back(exposed_image);

        string filename = "output/hdr/exposure_" + to_string(exposure).substr(0, 4) + ".jpg";
        imwrite(filename, exposed_image);
    }

    // ==================== Performance Test ====================
    cout << "\n--- Performance Test ---" << endl;

    // Test standard HDR algorithm
    auto start = chrono::high_resolution_clock::now();
    Mat hdr_image = ip101::advanced::create_hdr(exposure_images, exposure_times);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Standard HDR fusion time: " << duration.count() << " microseconds" << endl;

    // Test optimized HDR algorithm
    start = chrono::high_resolution_clock::now();
    Mat hdr_image_optimized = ip101::advanced::create_hdr_optimized(exposure_images, exposure_times);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Optimized HDR fusion time: " << duration.count() << " microseconds" << endl;

    // Test global tone mapping
    start = chrono::high_resolution_clock::now();
    Mat tone_mapped_global = ip101::advanced::tone_mapping_global(hdr_image, 0.18f, 1.0f);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Global tone mapping time: " << duration.count() << " microseconds" << endl;

    // Test local tone mapping
    start = chrono::high_resolution_clock::now();
    Mat tone_mapped_local = ip101::advanced::tone_mapping_local(hdr_image, 2.0f, 4.0f);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Local tone mapping time: " << duration.count() << " microseconds" << endl;

    // Test optimized tone mapping
    start = chrono::high_resolution_clock::now();
    Mat tone_mapped_global_opt = ip101::advanced::tone_mapping_global_optimized(hdr_image, 0.18f, 1.0f);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Optimized global tone mapping time: " << duration.count() << " microseconds" << endl;

    start = chrono::high_resolution_clock::now();
    Mat tone_mapped_local_opt = ip101::advanced::tone_mapping_local_optimized(hdr_image, 2.0f, 4.0f);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Optimized local tone mapping time: " << duration.count() << " microseconds" << endl;

    // ==================== Parameter Effect Test ====================
    cout << "\n--- Parameter Effect Test ---" << endl;

    // Test global tone mapping with different key parameters
    vector<float> key_values = {0.1f, 0.18f, 0.3f, 0.5f};
    vector<float> white_point_values = {0.8f, 1.0f, 1.2f, 1.5f};

    for (float key : key_values) {
        for (float white_point : white_point_values) {
            Mat result;
            result = ip101::advanced::tone_mapping_global(hdr_image, key, white_point);
            string filename = "output/hdr/global_key" + to_string(key).substr(0, 4) +
                            "_wp" + to_string(white_point).substr(0, 4) + ".jpg";
            imwrite(filename, result);
        }
    }

    // Test local tone mapping with different parameters
    vector<float> sigma_values = {1.0f, 2.0f, 3.0f, 4.0f};
    vector<float> contrast_values = {2.0f, 4.0f, 6.0f, 8.0f};

    for (float sigma : sigma_values) {
        for (float contrast : contrast_values) {
            Mat result;
            result = ip101::advanced::tone_mapping_local(hdr_image, sigma, contrast);
            string filename = "output/hdr/local_sigma" + to_string(sigma).substr(0, 3) +
                            "_contrast" + to_string(contrast).substr(0, 3) + ".jpg";
            imwrite(filename, result);
        }
    }

    // ==================== Visualization Results ====================
    cout << "\n--- Visualization Results ---" << endl;

    // Create comparison image
    vector<Mat> images = {src, tone_mapped_global, tone_mapped_local, tone_mapped_global_opt, tone_mapped_local_opt};
    vector<string> titles = {"Original", "Global Tone Mapping", "Local Tone Mapping", "Optimized Global", "Optimized Local"};

    Mat comparison;
    hconcat(images, comparison);

    // Add titles
    int font_face = FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.6;
    int thickness = 2;
    Scalar color(255, 255, 255);

    for (size_t i = 0; i < titles.size(); i++) {
        int x = i * src.cols + 10;
        putText(comparison, titles[i], Point(x, 30), font_face, font_scale, color, thickness);
    }

    imwrite("output/hdr/comparison.jpg", comparison);
    cout << "Comparison image saved to: output/hdr/comparison.jpg" << endl;

    // ==================== Special Scenario Test ====================
    cout << "\n--- Special Scenario Test ---" << endl;

    // Create high contrast scene
    Mat high_contrast = Mat::zeros(src.size(), CV_8UC3);

    // Create bright and dark areas
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (i < src.rows / 2) {
                // Upper half is bright area
                high_contrast.at<Vec3b>(i, j) = Vec3b(200, 200, 200);
            } else {
                // Lower half is dark area
                high_contrast.at<Vec3b>(i, j) = Vec3b(20, 20, 20);
            }
        }
    }

    // Add some details
    rectangle(high_contrast, Point(50, 50), Point(150, 150), Scalar(100, 100, 100), -1);
    rectangle(high_contrast, Point(200, 300), Point(300, 400), Scalar(150, 150, 150), -1);

    imwrite("output/hdr/high_contrast_scene.jpg", high_contrast);

    // Create images with different exposures
    vector<Mat> high_contrast_exposures;
    for (float exposure : exposure_times) {
        Mat exposed_image;
        high_contrast.convertTo(exposed_image, CV_32F);
        exposed_image = exposed_image * exposure;
        exposed_image = max(0.0f, min(255.0f, exposed_image));
        exposed_image.convertTo(exposed_image, CV_8U);
        high_contrast_exposures.push_back(exposed_image);
    }

    // Apply HDR processing
    Mat hdr_high_contrast = ip101::advanced::create_hdr(high_contrast_exposures, exposure_times);
    Mat tone_mapped_high_contrast = ip101::advanced::tone_mapping_local(hdr_high_contrast, 2.0f, 4.0f);

    imwrite("output/hdr/high_contrast_hdr.jpg", tone_mapped_high_contrast);

    // ==================== Quality Assessment ====================
    cout << "\n--- Quality Assessment ---" << endl;

    // Calculate dynamic range
    Mat gray_src, gray_result;
    cvtColor(src, gray_src, COLOR_BGR2GRAY);
    cvtColor(tone_mapped_global, gray_result, COLOR_BGR2GRAY);

    double min_val_src, max_val_src, min_val_result, max_val_result;
    minMaxLoc(gray_src, &min_val_src, &max_val_src);
    minMaxLoc(gray_result, &min_val_result, &max_val_result);

    cout << "Original dynamic range: " << min_val_src << " - " << max_val_src
         << " (range: " << max_val_src - min_val_src << ")" << endl;
    cout << "HDR processed dynamic range: " << min_val_result << " - " << max_val_result
         << " (range: " << max_val_result - min_val_result << ")" << endl;
    cout << "Dynamic range expansion factor: " << (max_val_result - min_val_result) / (max_val_src - min_val_src) << endl;

    // Calculate contrast
    Scalar mean_src, std_src, mean_result, std_result;
    meanStdDev(gray_src, mean_src, std_src);
    meanStdDev(gray_result, mean_result, std_result);

    cout << "Original contrast: " << std_src[0] << endl;
    cout << "HDR processed contrast: " << std_result[0] << endl;
    cout << "Contrast enhancement factor: " << std_result[0] / std_src[0] << endl;

    // Calculate PSNR
    Mat diff;
    absdiff(src, tone_mapped_global, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);

    double mse = mean(diff)[0];
    double psnr = 10.0 * log10((255.0 * 255.0) / mse);
    cout << "PSNR: " << fixed << setprecision(2) << psnr << " dB" << endl;

    // ==================== Histogram Analysis ====================
    cout << "\n--- Histogram Analysis ---" << endl;

    // Calculate histograms of original and HDR processed images
    vector<Mat> bgr_planes;
    split(src, bgr_planes);

    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};

    Mat b_hist, g_hist, r_hist;
    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange);

    // Calculate histogram of HDR processed image
    vector<Mat> bgr_planes_result;
    split(tone_mapped_global, bgr_planes_result);

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

    // ==================== Save Results ====================
    cout << "\n--- Save Results ---" << endl;

    imwrite("output/hdr/original.jpg", src);
    imwrite("output/hdr/hdr_image.jpg", hdr_image);
    imwrite("output/hdr/tone_mapped_global.jpg", tone_mapped_global);
    imwrite("output/hdr/tone_mapped_local.jpg", tone_mapped_local);
    imwrite("output/hdr/tone_mapped_global_optimized.jpg", tone_mapped_global_opt);
    imwrite("output/hdr/tone_mapped_local_optimized.jpg", tone_mapped_local_opt);

    cout << "All results have been saved to output/hdr/ directory" << endl;
    cout << "HDR algorithm test completed!" << endl;

    return 0;
}
