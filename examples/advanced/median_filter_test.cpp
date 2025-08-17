#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <iomanip>
#include <filesystem>
#include "advanced/defogging/median_filter.hpp"
#include "advanced/defogging/dark_channel.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    cout << "=== Median Filter Defogging Algorithm Test ===" << endl;

    // Load test image
    string image_path = (argc > 1) ? argv[1] : "assets/imori.jpg";
    Mat src = imread(image_path);
    if (src.empty()) {
        cerr << "Error: Cannot load image " << image_path << endl;
        return -1;
    }

    cout << "Image size: " << src.size() << endl;
    cout << "Image type: " << src.type() << endl;

    // Create output directory
    filesystem::create_directories("output/median_filter");

    // Declare all Mat and vector variables at the beginning of main function
    Mat hazy_image;
    Mat depthMap;
    Mat dst_median, dst_improved, dst_adaptive;
    Mat median_default, improved_default, adaptive_default;
    Mat comparison;
    Mat comparison_with_titles;
    Mat median_result, improved_result, adaptive_result;
    Mat diff;
    Mat gray_src, gray_result;
    Mat var_src, var_result, covar;
    Mat test_hazy, test_depth, test_float;
    vector<int> kernel_sizes;
    vector<double> omega_values;
    vector<double> t0_values;
    vector<double> lambda_values;
    vector<int> init_sizes;
    vector<int> max_sizes;
    vector<double> haze_strengths;
    vector<Mat> results;
    vector<string> method_names;
    vector<Mat> images;
    vector<Mat> test_channels;

    // Create hazy test image
    cout << "\n--- Creating Hazy Test Image ---" << endl;
    hazy_image = src.clone();
    int rows = hazy_image.rows;
    int cols = hazy_image.cols;

    // Create depth map
    depthMap = Mat::zeros(rows, cols, CV_32F);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double distance = sqrt((i - rows/2) * (i - rows/2) + (j - cols/2) * (j - cols/2));
            double max_distance = sqrt((rows/2) * (rows/2) + (cols/2) * (cols/2));
            depthMap.at<float>(i, j) = distance / max_distance;
        }
    }

    // Apply haze effect
    double beta = 0.8;
    Vec3d A(255, 255, 255);

    Mat hazy_float;
    hazy_image.convertTo(hazy_float, CV_32F);
    vector<Mat> channels(3);
    split(hazy_float, channels);

    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                float transmission = exp(-beta * depthMap.at<float>(i, j));
                channels[c].at<float>(i, j) = channels[c].at<float>(i, j) * transmission + A[c] * (1 - transmission);
            }
        }
    }

    merge(channels, hazy_float);
    hazy_float.convertTo(hazy_image, CV_8U);

    imwrite("output/median_filter/hazy_original.jpg", hazy_image);
    cout << "Created hazy test image: output/median_filter/hazy_original.jpg" << endl;

    // Performance test
    cout << "\n--- Performance Test ---" << endl;
    int iterations = 3;
    chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
    chrono::high_resolution_clock::time_point end = chrono::high_resolution_clock::now();
    chrono::microseconds duration_median = chrono::microseconds::zero();
    chrono::microseconds duration_improved = chrono::microseconds::zero();
    chrono::microseconds duration_adaptive = chrono::microseconds::zero();

    // Median filter defogging test
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        ip101::advanced::median_filter_defogging(hazy_image, dst_median, 31, 0.95, 0.1);
    }
    end = chrono::high_resolution_clock::now();
    duration_median = chrono::duration_cast<chrono::microseconds>(end - start);

    // Improved median filter defogging test
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        ip101::advanced::improved_median_filter_defogging(hazy_image, dst_improved, 31, 0.95, 0.1, 0.001);
    }
    end = chrono::high_resolution_clock::now();
    duration_improved = chrono::duration_cast<chrono::microseconds>(end - start);

    // Adaptive median filter defogging test
    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        ip101::advanced::adaptive_median_filter_defogging(hazy_image, dst_adaptive, 3, 21, 0.95, 0.1);
    }
    end = chrono::high_resolution_clock::now();
    duration_adaptive = chrono::duration_cast<chrono::microseconds>(end - start);

    cout << "Median filter defogging average time: " << duration_median.count() / iterations << " μs" << endl;
    cout << "Improved median filter defogging average time: " << duration_improved.count() / iterations << " μs" << endl;
    cout << "Adaptive median filter defogging average time: " << duration_adaptive.count() / iterations << " μs" << endl;

    // Parameter effect test
    cout << "\n--- Parameter Effect Test ---" << endl;

    // Median filter defogging parameter test
    kernel_sizes.push_back(15);
    kernel_sizes.push_back(31);
    kernel_sizes.push_back(51);
    kernel_sizes.push_back(71);

    omega_values.push_back(0.8);
    omega_values.push_back(0.9);
    omega_values.push_back(0.95);
    omega_values.push_back(0.99);

    t0_values.push_back(0.05);
    t0_values.push_back(0.1);
    t0_values.push_back(0.15);
    t0_values.push_back(0.2);

    for (size_t i = 0; i < kernel_sizes.size(); ++i) {
        int kernel_size = kernel_sizes[i];
        for (size_t j = 0; j < omega_values.size(); ++j) {
            double omega = omega_values[j];
            Mat result;
            ip101::advanced::median_filter_defogging(hazy_image, result, kernel_size, omega, 0.1);

            string filename = "output/median_filter/median_k" + to_string(kernel_size) +
                            "_omega" + to_string((int)(omega * 100)) + ".jpg";
            imwrite(filename, result);
            cout << "Saved: " << filename << " (kernel_size=" << kernel_size << ", omega=" << omega << ")" << endl;
        }
    }

    for (size_t k = 0; k < t0_values.size(); ++k) {
        double t0 = t0_values[k];
        Mat result;
        ip101::advanced::median_filter_defogging(hazy_image, result, 31, 0.95, t0);

        string filename = "output/median_filter/median_t0" + to_string((int)(t0 * 100)) + ".jpg";
        imwrite(filename, result);
        cout << "Saved: " << filename << " (t0=" << t0 << ")" << endl;
    }

    // Improved median filter parameter test
    lambda_values.push_back(0.0001);
    lambda_values.push_back(0.001);
    lambda_values.push_back(0.01);
    lambda_values.push_back(0.1);

    for (size_t l = 0; l < lambda_values.size(); ++l) {
        double lambda = lambda_values[l];
        Mat result;
        ip101::advanced::improved_median_filter_defogging(hazy_image, result, 31, 0.95, 0.1, lambda);

        string filename = "output/median_filter/improved_lambda" + to_string((int)(lambda * 10000)) + ".jpg";
        imwrite(filename, result);
        cout << "Saved: " << filename << " (lambda=" << lambda << ")" << endl;
    }

    // Adaptive median filter parameter test
    init_sizes.push_back(3);
    init_sizes.push_back(5);
    init_sizes.push_back(7);
    init_sizes.push_back(9);

    max_sizes.push_back(15);
    max_sizes.push_back(21);
    max_sizes.push_back(31);
    max_sizes.push_back(51);

    for (size_t m = 0; m < init_sizes.size(); ++m) {
        int init_size = init_sizes[m];
        for (size_t n = 0; n < max_sizes.size(); ++n) {
            int max_size = max_sizes[n];
            Mat result;
            ip101::advanced::adaptive_median_filter_defogging(hazy_image, result, init_size, max_size, 0.95, 0.1);

            string filename = "output/median_filter/adaptive_init" + to_string(init_size) +
                            "_max" + to_string(max_size) + ".jpg";
            imwrite(filename, result);
            cout << "Saved: " << filename << " (init_size=" << init_size << ", max_size=" << max_size << ")" << endl;
        }
    }

    // Visualization results
    cout << "\n--- Visualization Results ---" << endl;

    // Results with default parameters
    ip101::advanced::median_filter_defogging(hazy_image, median_default, 31, 0.95, 0.1);
    ip101::advanced::improved_median_filter_defogging(hazy_image, improved_default, 31, 0.95, 0.1, 0.001);
    ip101::advanced::adaptive_median_filter_defogging(hazy_image, adaptive_default, 3, 21, 0.95, 0.1);

    // Create comparison image
    images.push_back(hazy_image);
    images.push_back(median_default);
    images.push_back(improved_default);
    images.push_back(adaptive_default);

    if (!images.empty()) {
        hconcat(images, comparison);
    } else {
        comparison = hazy_image.clone();
    }

    // Add titles
    comparison.copyTo(comparison_with_titles);
    putText(comparison_with_titles, "Hazy", Point(10, 30),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
    putText(comparison_with_titles, "Median", Point(hazy_image.cols + 10, 30),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
    putText(comparison_with_titles, "Improved", Point(2 * hazy_image.cols + 10, 30),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
    putText(comparison_with_titles, "Adaptive", Point(3 * hazy_image.cols + 10, 30),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);

    imwrite("output/median_filter/comparison.jpg", comparison_with_titles);
    cout << "Saved comparison image: output/median_filter/comparison.jpg" << endl;

    // Special scenario test
    cout << "\n--- Special Scenario Test ---" << endl;

    // Create images with different haze levels
    haze_strengths.push_back(0.3);
    haze_strengths.push_back(0.5);
    haze_strengths.push_back(0.7);
    haze_strengths.push_back(0.9);

    for (size_t p = 0; p < haze_strengths.size(); ++p) {
        double strength = haze_strengths[p];
        // Create haze effect with different strengths
        test_hazy = src.clone();
        test_depth = Mat::zeros(rows, cols, CV_32F);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                double distance = sqrt((i - rows/2) * (i - rows/2) + (j - cols/2) * (j - cols/2));
                double max_distance = sqrt((rows/2) * (rows/2) + (cols/2) * (cols/2));
                test_depth.at<float>(i, j) = distance / max_distance;
            }
        }

        double test_beta = strength * 1.2;

        test_hazy.convertTo(test_float, CV_32F);
        test_channels.clear();
        test_channels.resize(3);
        split(test_float, test_channels);

        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    float transmission = exp(-test_beta * test_depth.at<float>(i, j));
                    test_channels[c].at<float>(i, j) = test_channels[c].at<float>(i, j) * transmission + A[c] * (1 - transmission);
                }
            }
        }

        merge(test_channels, test_float);
        test_float.convertTo(test_hazy, CV_8U);

        // Apply algorithms
        ip101::advanced::median_filter_defogging(test_hazy, median_result, 31, 0.95, 0.1);
        ip101::advanced::improved_median_filter_defogging(test_hazy, improved_result, 31, 0.95, 0.1, 0.001);
        ip101::advanced::adaptive_median_filter_defogging(test_hazy, adaptive_result, 3, 21, 0.95, 0.1);

        // Save results
        string base_filename = "output/median_filter/strength_" + to_string((int)(strength * 100));
        imwrite(base_filename + "_original.jpg", test_hazy);
        imwrite(base_filename + "_median.jpg", median_result);
        imwrite(base_filename + "_improved.jpg", improved_result);
        imwrite(base_filename + "_adaptive.jpg", adaptive_result);

        cout << "Saved results for strength " << strength << endl;
    }

    // Quality assessment
    cout << "\n--- Quality Assessment ---" << endl;

    // Calculate PSNR and SSIM (compared with original image)
    results.push_back(median_default);
    results.push_back(improved_default);
    results.push_back(adaptive_default);
    method_names.push_back("Median");
    method_names.push_back("Improved");
    method_names.push_back("Adaptive");

    for (size_t i = 0; i < results.size(); ++i) {
        // Calculate PSNR
        absdiff(src, results[i], diff);
        diff.convertTo(diff, CV_32F);
        diff = diff.mul(diff);

        double mse = mean(diff)[0];
        double psnr = 10.0 * log10((255.0 * 255.0) / mse);

        // Calculate SSIM (simplified version)
        cvtColor(src, gray_src, COLOR_BGR2GRAY);
        cvtColor(results[i], gray_result, COLOR_BGR2GRAY);

        gray_src.convertTo(gray_src, CV_32F);
        gray_result.convertTo(gray_result, CV_32F);

        Scalar mean_src = mean(gray_src);
        Scalar mean_result = mean(gray_result);

        gray_src = gray_src - mean_src[0];
        gray_result = gray_result - mean_result[0];

        multiply(gray_src, gray_src, var_src);
        multiply(gray_result, gray_result, var_result);
        multiply(gray_src, gray_result, covar);

        double var_src_val = mean(var_src)[0];
        double var_result_val = mean(var_result)[0];
        double covar_val = mean(covar)[0];

        double ssim = (2 * mean_src[0] * mean_result[0] + 0.01) * (2 * covar_val + 0.03) /
                     ((mean_src[0] * mean_src[0] + mean_result[0] * mean_result[0] + 0.01) *
                      (var_src_val + var_result_val + 0.03));

        cout << method_names[i] << " - PSNR: " << fixed << setprecision(2) << psnr
             << " dB, SSIM: " << setprecision(3) << ssim << endl;
    }

    cout << "\n=== Test Completed ===" << endl;
    cout << "All results have been saved to output/median_filter/ directory" << endl;

    return 0;
}