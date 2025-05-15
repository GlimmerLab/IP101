#include <basic/thinning.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;
using namespace ip101;

// Performance test function
void test_performance(const Mat& src, const string& method_name) {
    Mat dst;
    auto start = chrono::high_resolution_clock::now();

    if (method_name == "Basic") {
        basic_thinning(src, dst);
    } else if (method_name == "Hilditch") {
        hilditch_thinning(src, dst);
    } else if (method_name == "Zhang-Suen") {
        zhang_suen_thinning(src, dst);
    } else if (method_name == "Skeleton") {
        skeleton_extraction(src, dst);
    } else if (method_name == "Medial-Axis") {
        Mat dist_transform;
        medial_axis_transform(src, dst, dist_transform);
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << method_name << " processing time: " << duration.count() << "ms" << endl;

    // Display result
    imshow(method_name + " Thinning Result", dst);
    waitKey(0);
}

// Manual implementation of OpenCV thinning for comparison
void manual_thinning_zhangsuen(const Mat& src, Mat& dst) {
    // Zhang-Suen thinning implementation
    src.copyTo(dst);

    // Define neighborhood indices
    int dx[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
    int dy[8] = {0, -1, -1, -1, 0, 1, 1, 1};

    bool has_changed = true;

    while (has_changed) {
        has_changed = false;

        // First sub-iteration
        Mat tmp = dst.clone();
        for (int i = 1; i < dst.rows - 1; i++) {
            for (int j = 1; j < dst.cols - 1; j++) {
                if (tmp.at<uchar>(i, j) == 0) continue;

                // Count non-zero neighbors
                int count = 0;
                for (int k = 0; k < 8; k++) {
                    if (tmp.at<uchar>(i + dy[k], j + dx[k]) > 0) count++;
                }

                // Check conditions
                if (count >= 2 && count <= 6) {
                    // Count transitions from 0 to 1
                    int transitions = 0;
                    for (int k = 0; k < 8; k++) {
                        if (tmp.at<uchar>(i + dy[k], j + dx[k]) == 0 &&
                            tmp.at<uchar>(i + dy[(k+1)%8], j + dx[(k+1)%8]) > 0) {
                            transitions++;
                        }
                    }

                    if (transitions == 1) {
                        // Check conditions for first sub-iteration
                        if (tmp.at<uchar>(i-1, j) * tmp.at<uchar>(i, j+1) * tmp.at<uchar>(i+1, j) == 0 &&
                            tmp.at<uchar>(i, j+1) * tmp.at<uchar>(i+1, j) * tmp.at<uchar>(i, j-1) == 0) {
                            dst.at<uchar>(i, j) = 0;
                            has_changed = true;
                        }
                    }
                }
            }
        }

        if (!has_changed) break;

        // Second sub-iteration
        tmp = dst.clone();
        has_changed = false;

        for (int i = 1; i < dst.rows - 1; i++) {
            for (int j = 1; j < dst.cols - 1; j++) {
                if (tmp.at<uchar>(i, j) == 0) continue;

                // Count non-zero neighbors
                int count = 0;
                for (int k = 0; k < 8; k++) {
                    if (tmp.at<uchar>(i + dy[k], j + dx[k]) > 0) count++;
                }

                // Check conditions
                if (count >= 2 && count <= 6) {
                    // Count transitions from 0 to 1
                    int transitions = 0;
                    for (int k = 0; k < 8; k++) {
                        if (tmp.at<uchar>(i + dy[k], j + dx[k]) == 0 &&
                            tmp.at<uchar>(i + dy[(k+1)%8], j + dx[(k+1)%8]) > 0) {
                            transitions++;
                        }
                    }

                    if (transitions == 1) {
                        // Check conditions for second sub-iteration
                        if (tmp.at<uchar>(i-1, j) * tmp.at<uchar>(i, j+1) * tmp.at<uchar>(i, j-1) == 0 &&
                            tmp.at<uchar>(i-1, j) * tmp.at<uchar>(i+1, j) * tmp.at<uchar>(i, j-1) == 0) {
                            dst.at<uchar>(i, j) = 0;
                            has_changed = true;
                        }
                    }
                }
            }
        }
    }
}

// Manual implementation of skeleton extraction for comparison
void manual_skeleton_extraction(const Mat& src, Mat& dst) {
    // Distance transform based skeleton extraction
    Mat dist;
    distanceTransform(src, dist, DIST_L2, DIST_MASK_PRECISE);

    dst = Mat::zeros(src.size(), CV_8UC1);

    // Find local maxima of distance transform
    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
            if (src.at<uchar>(i, j) == 0) continue;

            float val = dist.at<float>(i, j);
            bool is_max = true;

            // Check 8-neighborhood for local maxima
            for (int di = -1; di <= 1 && is_max; di++) {
                for (int dj = -1; dj <= 1; dj++) {
                    if (di == 0 && dj == 0) continue;

                    if (dist.at<float>(i + di, j + dj) > val) {
                        is_max = false;
                        break;
                    }
                }
            }

            if (is_max) {
                dst.at<uchar>(i, j) = 255;
            }
        }
    }

    // Threshold to create binary skeleton
    double min_val, max_val;
    minMaxLoc(dist, &min_val, &max_val);
    double threshold_val = 0.5 * max_val;

    for (int i = 0; i < dst.rows; i++) {
        for (int j = 0; j < dst.cols; j++) {
            if (dst.at<uchar>(i, j) > 0 && dist.at<float>(i, j) < threshold_val) {
                dst.at<uchar>(i, j) = 0;
            }
        }
    }
}

// Compare with manual implementation
void compare_with_opencv(const Mat& src, const string& method_name) {
    Mat dst_ours, dst_manual;

    // Test our implementation
    auto start_ours = chrono::high_resolution_clock::now();
    if (method_name == "Basic") {
        basic_thinning(src, dst_ours);
    } else if (method_name == "Hilditch") {
        hilditch_thinning(src, dst_ours);
    } else if (method_name == "Zhang-Suen") {
        zhang_suen_thinning(src, dst_ours);
    } else if (method_name == "Skeleton") {
        skeleton_extraction(src, dst_ours);
    } else if (method_name == "Medial-Axis") {
        Mat dist_transform;
        medial_axis_transform(src, dst_ours, dist_transform);
    }
    auto end_ours = chrono::high_resolution_clock::now();
    auto duration_ours = chrono::duration_cast<chrono::milliseconds>(end_ours - start_ours);

    // Test manual implementation
    auto start_manual = chrono::high_resolution_clock::now();
    if (method_name == "Zhang-Suen") {
        manual_thinning_zhangsuen(src, dst_manual);
    } else if (method_name == "Skeleton") {
        manual_skeleton_extraction(src, dst_manual);
    }
    auto end_manual = chrono::high_resolution_clock::now();
    auto duration_manual = chrono::duration_cast<chrono::milliseconds>(end_manual - start_manual);

    if (!dst_manual.empty()) {
        cout << method_name << " performance comparison:" << endl;
        cout << "Our implementation: " << duration_ours.count() << "ms" << endl;
        cout << "Manual implementation: " << duration_manual.count() << "ms" << endl;
        cout << "Speed ratio: " << (float)duration_manual.count() / duration_ours.count() << "x" << endl;

        // Verify result correctness
        Mat diff;
        compare(dst_ours, dst_manual, diff, CMP_NE);
        int error_count = countNonZero(diff);
        cout << "Pixel difference count: " << error_count << endl;

        // Display comparison result
        Mat comparison;
        hconcat(vector<Mat>{dst_ours, dst_manual, diff}, comparison);
        imshow(method_name + " Result Comparison (Our | Manual | Diff)", comparison);
        waitKey(0);
    } else {
        cout << method_name << " No corresponding manual implementation" << endl;
        imshow(method_name + " Our Implementation", dst_ours);
        waitKey(0);
    }
}

// Test thinning effect on different shapes
void test_shapes(const string& method_name) {
    // Create test images
    vector<Mat> test_images;

    // Rectangle
    Mat rect = Mat::zeros(100, 200, CV_8UC1);
    rectangle(rect, Point(50, 25), Point(150, 75), Scalar(255), -1);
    test_images.push_back(rect);

    // Circle
    Mat circle = Mat::zeros(200, 200, CV_8UC1);
    cv::circle(circle, Point(100, 100), 50, Scalar(255), -1);
    test_images.push_back(circle);

    // Text
    Mat text = Mat::zeros(100, 300, CV_8UC1);
    putText(text, "IP101", Point(50, 70), FONT_HERSHEY_SIMPLEX, 2, Scalar(255), 3);
    test_images.push_back(text);

    // Apply thinning to each test image
    for (size_t i = 0; i < test_images.size(); i++) {
        cout << "\nTest shape " << i+1 << ":" << endl;
        test_performance(test_images[i], method_name);
        compare_with_opencv(test_images[i], method_name);
    }
}

int main() {
    // Read test image
    Mat src = imread("test_images/text.png", IMREAD_GRAYSCALE);
    if (src.empty()) {
        cerr << "Cannot read test image" << endl;
        return -1;
    }

    // Binarize
    threshold(src, src, 128, 255, THRESH_BINARY);

    // Test various thinning methods
    vector<string> methods = {"Basic", "Hilditch", "Zhang-Suen", "Skeleton", "Medial-Axis"};
    for (const auto& method : methods) {
        cout << "\nTesting " << method << " thinning algorithm:" << endl;
        test_performance(src, method);
        compare_with_opencv(src, method);

        // Test on different shapes
        cout << "\nTesting " << method << " effect on different shapes" << endl;
        test_shapes(method);
    }

    return 0;
}

