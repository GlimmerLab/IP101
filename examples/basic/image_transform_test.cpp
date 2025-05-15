#include <basic/image_transform.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <omp.h>

using namespace cv;
using namespace std;
using namespace ip101;

// Performance testing helper function
template<typename Func>
double measure_time(Func&& func, int iterations = 10) {
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = chrono::high_resolution_clock::now();
    return chrono::duration_cast<chrono::milliseconds>(end - start).count() / static_cast<double>(iterations);
}

// Performance testing function
void test_performance(const Mat& src) {
    // Test various transform methods
    cout << "\n--- Image Transform Performance Tests ---\n";
    cout << "Image size: " << src.cols << "x" << src.rows << "\n\n";

    // 1. Test Affine Transform
    cout << "Affine Transform Performance Test:\n";

    // Define test points
    vector<Point2f> src_points = {
        Point2f(0.0f, 0.0f),
        Point2f(static_cast<float>(src.cols - 1), 0.0f),
        Point2f(0.0f, static_cast<float>(src.rows - 1))
    };
    vector<Point2f> dst_points = {
        Point2f(0.0f, 0.0f),
        Point2f(static_cast<float>(src.cols - 1), 0.0f),
        Point2f(50.0f, static_cast<float>(src.rows - 1))
    };

    // Get affine matrix for both our implementation and OpenCV
    Mat M = get_affine_transform(src_points, dst_points);

    // Test original version
    double original_time = measure_time([&]() {
        Mat result = affineTransform(src, src_points, dst_points);
    });

    // Test optimized version
    double optimized_time = measure_time([&]() {
        Mat result = affineTransform_optimized(src, src_points, dst_points);
    });

    // Test affine_transform function
    double modern_time = measure_time([&]() {
        Mat result = affine_transform(src, M, src.size());
    });

    // Test OpenCV version
    double opencv_time = measure_time([&]() {
        Mat result;
        warpAffine(src, result, M, src.size());
    });

    cout << "Original version: " << original_time << "ms\n";
    cout << "Optimized version: " << optimized_time << "ms\n";
    cout << "Modern API version: " << modern_time << "ms\n";
    cout << "OpenCV implementation: " << opencv_time << "ms\n";
    cout << "Optimization ratio (original/optimized): " << original_time / optimized_time << "x\n";
    cout << "OpenCV ratio (OpenCV/modern): " << opencv_time / modern_time << "x\n\n";

    // 2. Test Perspective Transform
    cout << "Perspective Transform Performance Test:\n";

    // Define test points
    vector<Point2f> src_points_persp = {
        Point2f(0.0f, 0.0f),
        Point2f(static_cast<float>(src.cols - 1), 0.0f),
        Point2f(static_cast<float>(src.cols - 1), static_cast<float>(src.rows - 1)),
        Point2f(0.0f, static_cast<float>(src.rows - 1))
    };
    vector<Point2f> dst_points_persp = {
        Point2f(0.0f, 0.0f),
        Point2f(static_cast<float>(src.cols - 1), 0.0f),
        Point2f(static_cast<float>(src.cols - 20), static_cast<float>(src.rows - 1)),
        Point2f(20.0f, static_cast<float>(src.rows - 1))
    };

    // Get perspective matrix
    Mat P = get_perspective_transform(src_points_persp, dst_points_persp);

    // Test original version
    original_time = measure_time([&]() {
        Mat result = perspectiveTransform(src, src_points_persp, dst_points_persp);
    });

    // Test perspective_transform function
    modern_time = measure_time([&]() {
        Mat result = perspective_transform(src, P, src.size());
    });

    // Test OpenCV version
    opencv_time = measure_time([&]() {
        Mat result;
        warpPerspective(src, result, P, src.size());
    });

    cout << "Original version: " << original_time << "ms\n";
    cout << "Modern API version: " << modern_time << "ms\n";
    cout << "OpenCV implementation: " << opencv_time << "ms\n";
    cout << "OpenCV ratio (OpenCV/modern): " << opencv_time / modern_time << "x\n\n";

    // 3. Test Rotation
    cout << "Rotation Performance Test:\n";

    double angle = 45.0;
    Point2f center(static_cast<float>(src.cols/2), static_cast<float>(src.rows/2));

    // Test original version
    original_time = measure_time([&]() {
        Mat result = rotateImage(src, angle, center);
    });

    // Test rotate function
    modern_time = measure_time([&]() {
        Mat result = rotate(src, angle);
    });

    // Test OpenCV version
    opencv_time = measure_time([&]() {
        Mat M = getRotationMatrix2D(center, angle, 1.0);
        Mat result;
        warpAffine(src, result, M, src.size());
    });

    cout << "Original version: " << original_time << "ms\n";
    cout << "Modern API version: " << modern_time << "ms\n";
    cout << "OpenCV implementation: " << opencv_time << "ms\n";
    cout << "OpenCV ratio (OpenCV/modern): " << opencv_time / modern_time << "x\n\n";

    // 4. Test Scaling / Resize
    cout << "Resize Performance Test:\n";

    Size new_size(src.cols/2, src.rows/2);

    // Test original version
    original_time = measure_time([&]() {
        Mat result = scaleImage(src, 0.5, 0.5);
    });

    // Test resize function
    modern_time = measure_time([&]() {
        Mat result = resize(src, new_size);
    });

    // Test OpenCV version
    opencv_time = measure_time([&]() {
        Mat result;
        cv::resize(src, result, new_size);
    });

    cout << "Original version: " << original_time << "ms\n";
    cout << "Modern API version: " << modern_time << "ms\n";
    cout << "OpenCV implementation: " << opencv_time << "ms\n";
    cout << "OpenCV ratio (OpenCV/modern): " << opencv_time / modern_time << "x\n\n";

    // 5. Test Translation
    cout << "Translation Performance Test:\n";

    int tx = 100, ty = 50;

    // Test original version
    original_time = measure_time([&]() {
        Mat result = translateImage(src, tx, ty);
    });

    // Test translate function
    modern_time = measure_time([&]() {
        Mat result = translate(src, static_cast<double>(tx), static_cast<double>(ty));
    });

    // Test OpenCV version
    opencv_time = measure_time([&]() {
        Mat M = (Mat_<float>(2,3) << 1, 0, tx, 0, 1, ty);
        Mat result;
        warpAffine(src, result, M, src.size());
    });

    cout << "Original version: " << original_time << "ms\n";
    cout << "Modern API version: " << modern_time << "ms\n";
    cout << "OpenCV implementation: " << opencv_time << "ms\n";
    cout << "OpenCV ratio (OpenCV/modern): " << opencv_time / modern_time << "x\n\n";

    // 6. Test Mirroring
    cout << "Mirror Performance Test:\n";

    int flip_code = 1; // horizontal flip

    // Test original version
    original_time = measure_time([&]() {
        Mat result = mirrorImage(src, flip_code > 0 ? 0 : 1);
    });

    // Test mirror function
    modern_time = measure_time([&]() {
        Mat result = mirror(src, flip_code);
    });

    // Test OpenCV version
    opencv_time = measure_time([&]() {
        Mat result;
        flip(src, result, flip_code);
    });

    cout << "Original version: " << original_time << "ms\n";
    cout << "Modern API version: " << modern_time << "ms\n";
    cout << "OpenCV implementation: " << opencv_time << "ms\n";
    cout << "OpenCV ratio (OpenCV/modern): " << opencv_time / modern_time << "x\n\n";
}

// Visual results comparison
void visualize_results(const Mat& src) {
    // Define transformations to test
    vector<string> methods = {
        "Affine Transform",
        "Perspective Transform",
        "Rotation",
        "Resize",
        "Translation",
        "Mirror"
    };

    vector<Mat> results;

    // 1. Affine Transform
    vector<Point2f> src_points = {
        Point2f(0.0f, 0.0f),
        Point2f(static_cast<float>(src.cols - 1), 0.0f),
        Point2f(0.0f, static_cast<float>(src.rows - 1))
    };
    vector<Point2f> dst_points = {
        Point2f(0.0f, 0.0f),
        Point2f(static_cast<float>(src.cols - 1), 0.0f),
        Point2f(50.0f, static_cast<float>(src.rows - 1))
    };

    Mat M = get_affine_transform(src_points, dst_points);

    // Original implementation result
    Mat result1 = affineTransform(src, src_points, dst_points);
    results.push_back(result1);
    // Optimized implementation result
    Mat result2 = affineTransform_optimized(src, src_points, dst_points);
    results.push_back(result2);
    // Modern API result
    Mat result3 = affine_transform(src, M, src.size());
    results.push_back(result3);
    // OpenCV result
    Mat opencv_result;
    warpAffine(src, opencv_result, M, src.size());
    results.push_back(opencv_result);

    // 2. Perspective Transform
    vector<Point2f> src_points_persp = {
        Point2f(0.0f, 0.0f),
        Point2f(static_cast<float>(src.cols - 1), 0.0f),
        Point2f(static_cast<float>(src.cols - 1), static_cast<float>(src.rows - 1)),
        Point2f(0.0f, static_cast<float>(src.rows - 1))
    };
    vector<Point2f> dst_points_persp = {
        Point2f(0.0f, 0.0f),
        Point2f(static_cast<float>(src.cols - 1), 0.0f),
        Point2f(static_cast<float>(src.cols - 20), static_cast<float>(src.rows - 1)),
        Point2f(20.0f, static_cast<float>(src.rows - 1))
    };

    Mat P = get_perspective_transform(src_points_persp, dst_points_persp);

    // Original implementation result
    Mat result4 = perspectiveTransform(src, src_points_persp, dst_points_persp);
    results.push_back(result4);
    // Modern API result
    Mat result5 = perspective_transform(src, P, src.size());
    results.push_back(result5);
    // OpenCV result
    warpPerspective(src, opencv_result, P, src.size());
    results.push_back(opencv_result);

    // 3. Rotation
    double angle = 45.0;
    Point2f center(static_cast<float>(src.cols/2), static_cast<float>(src.rows/2));

    // Original implementation result
    Mat result6 = rotateImage(src, angle, center);
    results.push_back(result6);
    // Modern API result
    Mat result7 = rotate(src, angle);
    results.push_back(result7);
    // OpenCV result
    Mat R = getRotationMatrix2D(Point2f(static_cast<float>(src.cols/2), static_cast<float>(src.rows/2)), angle, 1.0);
    warpAffine(src, opencv_result, R, src.size());
    results.push_back(opencv_result);

    // 4. Resize
    Size new_size(src.cols/2, src.rows/2);

    // Original implementation result
    Mat result8 = scaleImage(src, 0.5, 0.5);
    results.push_back(result8);
    // Modern API result
    Mat result9 = resize(src, new_size);
    results.push_back(result9);
    // OpenCV result
    cv::resize(src, opencv_result, new_size);
    results.push_back(opencv_result);

    // 5. Translation
    int tx = 100, ty = 50;

    // Original implementation result
    Mat result10 = translateImage(src, tx, ty);
    results.push_back(result10);
    // Modern API result
    Mat result11 = translate(src, static_cast<double>(tx), static_cast<double>(ty));
    results.push_back(result11);
    // OpenCV result
    Mat T = (Mat_<float>(2,3) << 1, 0, tx, 0, 1, ty);
    warpAffine(src, opencv_result, T, src.size());
    results.push_back(opencv_result);

    // 6. Mirror
    int flip_code = 1; // horizontal flip

    // Original implementation result
    Mat result12 = mirrorImage(src, flip_code > 0 ? 0 : 1);
    results.push_back(result12);
    // Modern API result
    Mat result13 = mirror(src, flip_code);
    results.push_back(result13);
    // OpenCV result
    flip(src, opencv_result, flip_code);
    results.push_back(opencv_result);

    // Display original image
    imshow("Original Image", src);

    // Display results in groups
    for (int i = 0; i < methods.size(); i++) {
        int offset = i * 3; // For affine we have 4 images, rest have 3
        if (i == 0) {
            Mat comparison;
            vector<Mat> row = {results[0], results[1], results[2], results[3]};
            hconcat(row, comparison);
            imshow(methods[i] + " (Original, Optimized, Modern, OpenCV)", comparison);
        } else {
            Mat comparison;
            vector<Mat> row = {results[offset+4], results[offset+5], results[offset+6]};
            hconcat(row, comparison);
            imshow(methods[i] + " (Original, Modern, OpenCV)", comparison);
        }
    }

    waitKey(0);
}

int main(int argc, char** argv) {
    // Set OpenMP thread count
    int num_threads = omp_get_num_procs();
    omp_set_num_threads(num_threads);
    cout << "Using " << num_threads << " threads for parallel computing\n";

    // Read test image
    Mat src;
    if (argc > 1) {
        src = imread(argv[1]);
    } else {
        src = imread("test_images/lena.jpg");
    }

    if (src.empty()) {
        cerr << "Cannot read test image" << endl;
        return -1;
    }

    // Run performance tests
    test_performance(src);

    // Visualize results
    visualize_results(src);

    return 0;
}

