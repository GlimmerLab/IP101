#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <filesystem>
#include <iomanip>
#include <fstream>
#include <algorithm>

#include "advanced/detection/color_cast_detection.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    cout << "=== Color Cast Detection Algorithm Test ===" << endl;

    string image_path = (argc > 1) ? argv[1] : "assets/imori.jpg";
    Mat src = imread(image_path);

    if (src.empty()) {
        cerr << "Error: Cannot load image " << image_path << endl;
        return -1;
    }

    cout << "Image size: " << src.size() << endl;
    filesystem::create_directories("output/color_cast_detection");

    // Performance test
    cout << "\n--- Performance Test ---" << endl;

    ip101::advanced::ColorCastResult result_main, result_hist, result_wb;
    ip101::advanced::ColorCastDetectionParams params;
    params.threshold = 0.15;
    params.use_reference_white = false;
    params.analyze_distribution = true;
    params.auto_white_balance_check = true;

    auto start = chrono::high_resolution_clock::now();
    ip101::advanced::detect_color_cast(src, result_main, params);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Main color cast detection time: " << duration.count() << " microseconds" << endl;

    start = chrono::high_resolution_clock::now();
    ip101::advanced::detect_color_cast_histogram(src, result_hist, params);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Histogram color cast detection time: " << duration.count() << " microseconds" << endl;

    start = chrono::high_resolution_clock::now();
    ip101::advanced::detect_color_cast_white_balance(src, result_wb, params);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "White balance color cast detection time: " << duration.count() << " microseconds" << endl;

    // Parameter effect test
    cout << "\n--- Parameter Effect Test ---" << endl;

    vector<double> thresholds = {0.05, 0.10, 0.15, 0.20, 0.25};
    for (double threshold : thresholds) {
        ip101::advanced::ColorCastResult test_result;
        ip101::advanced::ColorCastDetectionParams test_params = params;
        test_params.threshold = threshold;
        ip101::advanced::detect_color_cast(src, test_result, test_params);

        cout << "Threshold " << threshold << ": color cast degree=" << test_result.color_cast_degree
             << ", dominant color=" << test_result.dominant_color << endl;
    }

    // Test different reference white points
    vector<Vec3f> reference_whites = {
        Vec3f(0.95f, 1.0f, 1.05f),  // D65
        Vec3f(1.0f, 1.0f, 1.0f),    // Standard white
        Vec3f(0.98f, 1.0f, 1.02f),  // Warm white
        Vec3f(1.02f, 1.0f, 0.98f)   // Cool white
    };

    for (size_t i = 0; i < reference_whites.size(); i++) {
        ip101::advanced::ColorCastResult test_result;
        ip101::advanced::ColorCastDetectionParams test_params = params;
        test_params.use_reference_white = true;
        test_params.reference_white = reference_whites[i];
        ip101::advanced::detect_color_cast(src, test_result, test_params);

        cout << "Reference white " << i << ": color cast degree=" << test_result.color_cast_degree
             << ", dominant color=" << test_result.dominant_color << endl;
    }

    // Generate color distribution map
    cout << "\n--- Generate Color Distribution Map ---" << endl;
    Mat distribution_map;
    ip101::advanced::generate_color_distribution_map(src, distribution_map);
    imwrite("output/color_cast_detection/color_distribution.jpg", distribution_map);

    // Visualize results
    cout << "\n--- Visualize Results ---" << endl;

    Mat vis_main, vis_hist, vis_wb;
    ip101::advanced::visualize_color_cast(src, vis_main, result_main);
    ip101::advanced::visualize_color_cast(src, vis_hist, result_hist);
    ip101::advanced::visualize_color_cast(src, vis_wb, result_wb);

    vector<Mat> images = {src, vis_main, vis_hist, vis_wb};
    vector<string> titles = {"Original", "Main Algorithm", "Histogram Detection", "White Balance Detection"};

    Mat comparison;
    hconcat(images, comparison);

    int font_face = FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.5;
    int thickness = 2;
    Scalar color(255, 255, 255);

    for (size_t i = 0; i < titles.size(); i++) {
        int x = i * src.cols + 10;
        putText(comparison, titles[i], Point(x, 30), font_face, font_scale, color, thickness);
    }

    imwrite("output/color_cast_detection/comparison.jpg", comparison);

    // Quality assessment
    cout << "\n--- Quality Assessment ---" << endl;

    cout << "Main algorithm detection results:" << endl;
    cout << "  Has color cast: " << (result_main.has_color_cast ? "Yes" : "No") << endl;
    cout << "  Color cast degree: " << fixed << setprecision(3) << result_main.color_cast_degree << endl;
    cout << "  Dominant color: " << result_main.dominant_color << endl;
    cout << "  Color cast vector: [" << result_main.color_cast_vector[0] << ", "
         << result_main.color_cast_vector[1] << ", " << result_main.color_cast_vector[2] << "]" << endl;

    cout << "\nHistogram detection results:" << endl;
    cout << "  Has color cast: " << (result_hist.has_color_cast ? "Yes" : "No") << endl;
    cout << "  Color cast degree: " << fixed << setprecision(3) << result_hist.color_cast_degree << endl;
    cout << "  Dominant color: " << result_hist.dominant_color << endl;

    cout << "\nWhite balance detection results:" << endl;
    cout << "  Has color cast: " << (result_wb.has_color_cast ? "Yes" : "No") << endl;
    cout << "  Color cast degree: " << fixed << setprecision(3) << result_wb.color_cast_degree << endl;
    cout << "  Dominant color: " << result_wb.dominant_color << endl;

    // Calculate color statistics
    vector<Mat> bgr_planes;
    split(src, bgr_planes);

    Scalar mean_b, mean_g, mean_r, std_b, std_g, std_r;
    meanStdDev(bgr_planes[0], mean_b, std_b);
    meanStdDev(bgr_planes[1], mean_g, std_g);
    meanStdDev(bgr_planes[2], mean_r, std_r);

    cout << "\nColor statistics:" << endl;
    cout << "  Blue channel - mean: " << mean_b[0] << ", std: " << std_b[0] << endl;
    cout << "  Green channel - mean: " << mean_g[0] << ", std: " << std_g[0] << endl;
    cout << "  Red channel - mean: " << mean_r[0] << ", std: " << std_r[0] << endl;

    // Calculate color balance
    double color_balance = min({mean_b[0], mean_g[0], mean_r[0]}) / max({mean_b[0], mean_g[0], mean_r[0]});
    cout << "  Color balance: " << fixed << setprecision(3) << color_balance << endl;

    // Save results
    cout << "\n--- Save Results ---" << endl;

    imwrite("output/color_cast_detection/original.jpg", src);
    imwrite("output/color_cast_detection/visualization_main.jpg", vis_main);
    imwrite("output/color_cast_detection/visualization_hist.jpg", vis_hist);
    imwrite("output/color_cast_detection/visualization_wb.jpg", vis_wb);

    // Save detection results to text file
    ofstream result_file("output/color_cast_detection/detection_results.txt");
    result_file << "Color Cast Detection Results Report" << endl;
    result_file << "===================================" << endl;
    result_file << "Image path: " << image_path << endl;
    result_file << "Image size: " << src.size() << endl;
    result_file << endl;

    result_file << "Main algorithm detection results:" << endl;
    result_file << "  Has color cast: " << (result_main.has_color_cast ? "Yes" : "No") << endl;
    result_file << "  Color cast degree: " << fixed << setprecision(3) << result_main.color_cast_degree << endl;
    result_file << "  Dominant color: " << result_main.dominant_color << endl;
    result_file << "  Color cast vector: [" << result_main.color_cast_vector[0] << ", "
               << result_main.color_cast_vector[1] << ", " << result_main.color_cast_vector[2] << "]" << endl;
    result_file << endl;

    result_file << "Color statistics:" << endl;
    result_file << "  Blue channel - mean: " << mean_b[0] << ", std: " << std_b[0] << endl;
    result_file << "  Green channel - mean: " << mean_g[0] << ", std: " << std_g[0] << endl;
    result_file << "  Red channel - mean: " << mean_r[0] << ", std: " << std_r[0] << endl;
    result_file << "  Color balance: " << fixed << setprecision(3) << color_balance << endl;
    result_file.close();

    cout << "All results have been saved to output/color_cast_detection/ directory" << endl;
    cout << "Color cast detection algorithm test completed!" << endl;

    return 0;
}
