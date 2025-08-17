#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <filesystem>
#include <iomanip>
#include <fstream>

#include "advanced/detection/license_plate_detection.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    cout << "=== License Plate Detection Algorithm Test ===" << endl;

    string image_path = (argc > 1) ? argv[1] : "assets/imori.jpg";
    Mat src = imread(image_path);

    if (src.empty()) {
        cerr << "Error: Cannot load image " << image_path << endl;
        return -1;
    }

    cout << "Image size: " << src.size() << endl;
    filesystem::create_directories("output/license_plate_detection");

    // Performance test
    cout << "\n--- Performance Test ---" << endl;

    vector<ip101::advanced::LicensePlateInfo> plates_main, plates_edge, plates_color;
    ip101::advanced::LicensePlateDetectionParams params;
    params.min_area_ratio = 0.001;
    params.max_area_ratio = 0.05;
    params.min_aspect_ratio = 2.0;
    params.max_aspect_ratio = 6.0;
    params.min_plate_confidence = 0.6;

    auto start = chrono::high_resolution_clock::now();
    ip101::advanced::detect_license_plates(src, plates_main, params);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Main license plate detection time: " << duration.count() << " microseconds" << endl;

    start = chrono::high_resolution_clock::now();
    ip101::advanced::detect_plates_edge_based(src, plates_edge, params);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Edge-based license plate detection time: " << duration.count() << " microseconds" << endl;

    start = chrono::high_resolution_clock::now();
    ip101::advanced::detect_plates_color_based(src, plates_color, params);
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Color-based license plate detection time: " << duration.count() << " microseconds" << endl;

    // Parameter effect test
    cout << "\n--- Parameter Effect Test ---" << endl;

    vector<double> min_area_ratios = {0.0005, 0.001, 0.002, 0.005, 0.01};
    for (double ratio : min_area_ratios) {
        vector<ip101::advanced::LicensePlateInfo> test_plates;
        ip101::advanced::LicensePlateDetectionParams test_params = params;
        test_params.min_area_ratio = ratio;
        ip101::advanced::detect_license_plates(src, test_plates, test_params);

        cout << "Min area ratio " << ratio << ": detected " << test_plates.size() << " plates" << endl;
    }

    vector<double> min_confidences = {0.3, 0.5, 0.6, 0.7, 0.8};
    for (double confidence : min_confidences) {
        vector<ip101::advanced::LicensePlateInfo> test_plates;
        ip101::advanced::LicensePlateDetectionParams test_params = params;
        test_params.min_plate_confidence = confidence;
        ip101::advanced::detect_license_plates(src, test_plates, test_params);

        cout << "Min confidence " << confidence << ": detected " << test_plates.size() << " plates" << endl;
    }

    vector<double> aspect_ratios = {1.5, 2.0, 3.0, 4.0, 5.0};
    for (double ratio : aspect_ratios) {
        vector<ip101::advanced::LicensePlateInfo> test_plates;
        ip101::advanced::LicensePlateDetectionParams test_params = params;
        test_params.min_aspect_ratio = ratio;
        test_params.max_aspect_ratio = ratio + 2.0;
        ip101::advanced::detect_license_plates(src, test_plates, test_params);

        cout << "Aspect ratio range [" << ratio << ", " << (ratio + 2.0) << "]: detected " << test_plates.size() << " plates" << endl;
    }

    // License plate skew correction test
    cout << "\n--- License Plate Skew Correction Test ---" << endl;

    for (size_t i = 0; i < plates_main.size(); i++) {
        Mat corrected_img;
        ip101::advanced::correct_plate_skew(plates_main[i].plate_img, corrected_img);

        string filename = "output/license_plate_detection/corrected_plate_" + to_string(i) + ".jpg";
        imwrite(filename, corrected_img);

        cout << "Plate " << i << " skew correction completed" << endl;
    }

    // Character segmentation test
    cout << "\n--- Character Segmentation Test ---" << endl;

    for (size_t i = 0; i < plates_main.size(); i++) {
        vector<Rect> chars;
        ip101::advanced::segment_plate_chars(plates_main[i].plate_img, chars);

        cout << "Plate " << i << " segmented into " << chars.size() << " character regions" << endl;

        // Save character segmentation results
        Mat plate_with_chars = plates_main[i].plate_img.clone();
        for (size_t j = 0; j < chars.size(); j++) {
            rectangle(plate_with_chars, chars[j], Scalar(0, 255, 0), 2);
            putText(plate_with_chars, to_string(j), Point(chars[j].x, chars[j].y - 5),
                   FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
        }

        string filename = "output/license_plate_detection/plate_chars_" + to_string(i) + ".jpg";
        imwrite(filename, plate_with_chars);
    }

    // Visualize results
    cout << "\n--- Visualize Results ---" << endl;

    Mat vis_main = src.clone();
    Mat vis_edge = src.clone();
    Mat vis_color = src.clone();

    ip101::advanced::draw_license_plates(vis_main, plates_main, true);
    ip101::advanced::draw_license_plates(vis_edge, plates_edge, false);
    ip101::advanced::draw_license_plates(vis_color, plates_color, false);

    vector<Mat> images = {src, vis_main, vis_edge, vis_color};
    vector<string> titles = {"Original", "Main Algorithm", "Edge Detection", "Color Detection"};

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

    imwrite("output/license_plate_detection/comparison.jpg", comparison);

    // Quality assessment
    cout << "\n--- Quality Assessment ---" << endl;

    cout << "Main algorithm detection results:" << endl;
    cout << "  Detected plates: " << plates_main.size() << endl;
    for (size_t i = 0; i < plates_main.size(); i++) {
        cout << "  Plate " << i << ": confidence=" << fixed << setprecision(3) << plates_main[i].confidence
             << ", region=" << plates_main[i].rect << endl;
    }

    cout << "\nEdge detection results:" << endl;
    cout << "  Detected plates: " << plates_edge.size() << endl;
    for (size_t i = 0; i < plates_edge.size(); i++) {
        cout << "  Plate " << i << ": confidence=" << fixed << setprecision(3) << plates_edge[i].confidence
             << ", region=" << plates_edge[i].rect << endl;
    }

    cout << "\nColor detection results:" << endl;
    cout << "  Detected plates: " << plates_color.size() << endl;
    for (size_t i = 0; i < plates_color.size(); i++) {
        cout << "  Plate " << i << ": confidence=" << fixed << setprecision(3) << plates_color[i].confidence
             << ", region=" << plates_color[i].rect << endl;
    }

    // Calculate detection quality metrics
    cout << "\nDetection quality analysis:" << endl;

    // Calculate average confidence
    double avg_confidence_main = 0.0, avg_confidence_edge = 0.0, avg_confidence_color = 0.0;

    if (!plates_main.empty()) {
        for (const auto& plate : plates_main) {
            avg_confidence_main += plate.confidence;
        }
        avg_confidence_main /= plates_main.size();
    }

    if (!plates_edge.empty()) {
        for (const auto& plate : plates_edge) {
            avg_confidence_edge += plate.confidence;
        }
        avg_confidence_edge /= plates_edge.size();
    }

    if (!plates_color.empty()) {
        for (const auto& plate : plates_color) {
            avg_confidence_color += plate.confidence;
        }
        avg_confidence_color /= plates_color.size();
    }

    cout << "  Main algorithm average confidence: " << fixed << setprecision(3) << avg_confidence_main << endl;
    cout << "  Edge detection average confidence: " << fixed << setprecision(3) << avg_confidence_edge << endl;
    cout << "  Color detection average confidence: " << fixed << setprecision(3) << avg_confidence_color << endl;

    // Calculate plate region statistics
    if (!plates_main.empty()) {
        double total_area = 0.0;
        double total_aspect_ratio = 0.0;

        for (const auto& plate : plates_main) {
            total_area += plate.rect.area();
            total_aspect_ratio += (double)plate.rect.width / plate.rect.height;
        }

        double avg_area = total_area / plates_main.size();
        double avg_aspect_ratio = total_aspect_ratio / plates_main.size();

        cout << "  Average plate area: " << avg_area << " pixels" << endl;
        cout << "  Average aspect ratio: " << fixed << setprecision(2) << avg_aspect_ratio << endl;
    }

    // Save results
    cout << "\n--- Save Results ---" << endl;

    imwrite("output/license_plate_detection/original.jpg", src);
    imwrite("output/license_plate_detection/detection_main.jpg", vis_main);
    imwrite("output/license_plate_detection/detection_edge.jpg", vis_edge);
    imwrite("output/license_plate_detection/detection_color.jpg", vis_color);

    // Save detected plate images
    for (size_t i = 0; i < plates_main.size(); i++) {
        string filename = "output/license_plate_detection/plate_" + to_string(i) + ".jpg";
        imwrite(filename, plates_main[i].plate_img);
    }

    // Save detection results to text file
    ofstream result_file("output/license_plate_detection/detection_results.txt");
    result_file << "License Plate Detection Results Report" << endl;
    result_file << "=====================================" << endl;
    result_file << "Image path: " << image_path << endl;
    result_file << "Image size: " << src.size() << endl;
    result_file << endl;

    result_file << "Main algorithm detection results:" << endl;
    result_file << "  Detected plates: " << plates_main.size() << endl;
    result_file << "  Average confidence: " << fixed << setprecision(3) << avg_confidence_main << endl;
    for (size_t i = 0; i < plates_main.size(); i++) {
        result_file << "  Plate " << i << ": confidence=" << fixed << setprecision(3) << plates_main[i].confidence
                   << ", region=" << plates_main[i].rect << endl;
    }
    result_file << endl;

    result_file << "Edge detection results:" << endl;
    result_file << "  Detected plates: " << plates_edge.size() << endl;
    result_file << "  Average confidence: " << fixed << setprecision(3) << avg_confidence_edge << endl;
    result_file << endl;

    result_file << "Color detection results:" << endl;
    result_file << "  Detected plates: " << plates_color.size() << endl;
    result_file << "  Average confidence: " << fixed << setprecision(3) << avg_confidence_color << endl;
    result_file.close();

    cout << "All results have been saved to output/license_plate_detection/ directory" << endl;
    cout << "License plate detection algorithm test completed!" << endl;

    return 0;
}
