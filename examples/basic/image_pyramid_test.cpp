#include <basic/image_pyramid.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;
using namespace ip101;

// Performance testing function
void test_performance(const Mat& src, const string& method_name) {
    auto start = chrono::high_resolution_clock::now();

    if (method_name == "Gaussian Pyramid") {
        vector<Mat> pyramid = build_gaussian_pyramid(src, 4);
        Mat vis = visualize_pyramid(pyramid);
        imshow("Gaussian Pyramid", vis);
    } else if (method_name == "Laplacian Pyramid") {
        vector<Mat> pyramid = build_laplacian_pyramid(src, 4);
        Mat vis = visualize_pyramid(pyramid);
        imshow("Laplacian Pyramid", vis);
    } else if (method_name == "SIFT Scale Space") {
        vector<vector<Mat>> scale_space = build_sift_scale_space(src);
        // Display all scales of each octave
        for (size_t o = 0; o < scale_space.size(); o++) {
            Mat vis = visualize_pyramid(scale_space[o]);
            imshow("SIFT Scale Space - Octave " + to_string(o), vis);
        }
    } else if (method_name == "Saliency") {
        Mat saliency = saliency_detection(src);
        imshow("Saliency Detection Result", saliency);
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << method_name << " processing time: " << duration.count() << "ms" << endl;

    waitKey(0);
}

// Compare with OpenCV implementation
void compare_with_opencv(const Mat& src, const string& method_name) {
    // Test our implementation
    auto start_ours = chrono::high_resolution_clock::now();
    Mat result_ours;
    vector<Mat> pyramid_ours;

    if (method_name == "Gaussian Pyramid") {
        pyramid_ours = build_gaussian_pyramid(src, 4);
    } else if (method_name == "Laplacian Pyramid") {
        pyramid_ours = build_laplacian_pyramid(src, 4);
    } else if (method_name == "Saliency") {
        result_ours = saliency_detection(src);
    }

    auto end_ours = chrono::high_resolution_clock::now();
    auto duration_ours = chrono::duration_cast<chrono::milliseconds>(end_ours - start_ours);

    // Test OpenCV implementation
    auto start_opencv = chrono::high_resolution_clock::now();
    Mat result_opencv;
    vector<Mat> pyramid_opencv;

    if (method_name == "Gaussian Pyramid") {
        Mat current = src.clone();
        pyramid_opencv.push_back(current);
        for (int i = 0; i < 3; i++) {
            Mat next;
            pyrDown(current, next);
            pyramid_opencv.push_back(next);
            current = next;
        }
    } else if (method_name == "Laplacian Pyramid") {
        Mat current = src.clone();
        vector<Mat> gauss_pyr;
        gauss_pyr.push_back(current);
        for (int i = 0; i < 3; i++) {
            Mat next;
            pyrDown(current, next);
            gauss_pyr.push_back(next);
            current = next;
        }

        pyramid_opencv.resize(4);
        for (int i = 0; i < 3; i++) {
            Mat up;
            pyrUp(gauss_pyr[i + 1], up, gauss_pyr[i].size());
            subtract(gauss_pyr[i], up, pyramid_opencv[i]);
        }
        pyramid_opencv[3] = gauss_pyr[3];
    } else if (method_name == "Saliency") {
        // Use a simple implementation since OpenCV saliency module might not be available
        Mat gray;
        if (src.channels() == 3) {
            cvtColor(src, gray, COLOR_BGR2GRAY);
        } else {
            gray = src.clone();
        }

        // Simple spectral residual method
        Mat spectrum;
        gray.convertTo(gray, CV_32F);
        dft(gray, spectrum, DFT_COMPLEX_OUTPUT);

        // Apply log amplitude
        Mat planes[2];
        split(spectrum, planes);
        Mat magnitude, phase;
        cartToPolar(planes[0], planes[1], magnitude, phase);
        magnitude += 1.0f; // Avoid log(0)
        log(magnitude, magnitude);

        // Apply smoothing
        Mat smoothed;
        blur(magnitude, smoothed, Size(3, 3));

        // Calculate residual
        Mat residual = magnitude - smoothed;

        // Back to spatial domain
        polarToCart(residual, phase, planes[0], planes[1]);
        merge(planes, 2, spectrum);
        dft(spectrum, result_opencv, DFT_INVERSE | DFT_SCALE);

        // Get magnitude
        split(result_opencv, planes);
        magnitude = planes[0].mul(planes[0]) + planes[1].mul(planes[1]);
        sqrt(magnitude, magnitude);

        // Normalize
        normalize(magnitude, result_opencv, 0, 255, NORM_MINMAX);
        result_opencv.convertTo(result_opencv, CV_8U);
    }

    auto end_opencv = chrono::high_resolution_clock::now();
    auto duration_opencv = chrono::duration_cast<chrono::milliseconds>(end_opencv - start_opencv);

    cout << method_name << " performance comparison:" << endl;
    cout << "Our implementation: " << duration_ours.count() << "ms" << endl;
    cout << "OpenCV implementation: " << duration_opencv.count() << "ms" << endl;
    cout << "Speed ratio: " << (float)duration_opencv.count() / duration_ours.count() << "x" << endl;

    // Display comparison results
    if (method_name == "Gaussian Pyramid" || method_name == "Laplacian Pyramid") {
        Mat vis_ours = visualize_pyramid(pyramid_ours);
        Mat vis_opencv = visualize_pyramid(pyramid_opencv);
        Mat comparison;
        vconcat(vector<Mat>{vis_ours, vis_opencv}, comparison);
        imshow(method_name + " Comparison (Top: Ours, Bottom: OpenCV)", comparison);
    } else if (method_name == "Saliency") {
        Mat comparison;
        hconcat(vector<Mat>{result_ours, result_opencv}, comparison);
        imshow(method_name + " Comparison (Left: Ours, Right: OpenCV)", comparison);
    }

    waitKey(0);
}

// Test image blending
void test_image_blend(const Mat& src1, const Mat& src2, const Mat& mask) {
    cout << "\nTesting Image Blending:" << endl;

    auto start = chrono::high_resolution_clock::now();
    Mat result = pyramid_blend(src1, src2, mask, 4);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

    cout << "Blending time: " << duration.count() << "ms" << endl;

    // Display results
    Mat vis;
    hconcat(vector<Mat>{src1, src2, result}, vis);
    imshow("Image Blending (Left: Image1, Middle: Image2, Right: Result)", vis);
    waitKey(0);
}

int main() {
    // Read test image
    Mat src = imread("test_images/lena.jpg", IMREAD_GRAYSCALE);
    if (src.empty()) {
        cerr << "Cannot read test image" << endl;
        return -1;
    }

    // Test various methods
    vector<string> methods = {
        "Gaussian Pyramid",
        "Laplacian Pyramid",
        "SIFT Scale Space",
        "Saliency"
    };

    for (const auto& method : methods) {
        cout << "\nTesting " << method << ":" << endl;
        test_performance(src, method);
        if (method != "SIFT Scale Space") {
            compare_with_opencv(src, method);
        }
    }

    // Test image blending
    Mat src1 = imread("test_images/apple.jpg");
    Mat src2 = imread("test_images/orange.jpg");
    Mat mask = Mat::zeros(src1.size(), CV_32F);
    mask(Rect(0, 0, mask.cols/2, mask.rows)) = 1.0;

    if (!src1.empty() && !src2.empty()) {
        test_image_blend(src1, src2, mask);
    }

    return 0;
}

