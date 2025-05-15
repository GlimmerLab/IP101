#include <basic/image_matching.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;
using namespace ip101;

// Performance testing function
void test_performance(const Mat& src, const Mat& templ, const string& method_name) {
    Mat result;
    auto start = chrono::high_resolution_clock::now();

    if (method_name == "SSD") {
        ssd_matching(src, templ, result, TM_SQDIFF);
    } else if (method_name == "SAD") {
        sad_matching(src, templ, result);
    } else if (method_name == "NCC") {
        ncc_matching(src, templ, result);
    } else if (method_name == "ZNCC") {
        zncc_matching(src, templ, result);
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << method_name << " matching time: " << duration.count() << "ms" << endl;
}

// Compare with OpenCV implementation
void compare_with_opencv(const Mat& src, const Mat& templ, const string& method_name) {
    Mat result_ours, result_opencv;
    auto start_ours = chrono::high_resolution_clock::now();

    if (method_name == "SSD") {
        ssd_matching(src, templ, result_ours, TM_SQDIFF);
    } else if (method_name == "SAD") {
        sad_matching(src, templ, result_ours);
    } else if (method_name == "NCC") {
        ncc_matching(src, templ, result_ours);
    } else if (method_name == "ZNCC") {
        zncc_matching(src, templ, result_ours);
    }

    auto end_ours = chrono::high_resolution_clock::now();
    auto duration_ours = chrono::duration_cast<chrono::milliseconds>(end_ours - start_ours);

    auto start_opencv = chrono::high_resolution_clock::now();
    matchTemplate(src, templ, result_opencv, TM_SQDIFF);
    auto end_opencv = chrono::high_resolution_clock::now();
    auto duration_opencv = chrono::duration_cast<chrono::milliseconds>(end_opencv - start_opencv);

    cout << method_name << " performance comparison:" << endl;
    cout << "Our implementation: " << duration_ours.count() << "ms" << endl;
    cout << "OpenCV implementation: " << duration_opencv.count() << "ms" << endl;
    cout << "Speed ratio: " << (float)duration_opencv.count() / duration_ours.count() << "x" << endl;

    // Verify result correctness
    double min_val1, max_val1, min_val2, max_val2;
    Point min_loc1, max_loc1, min_loc2, max_loc2;
    minMaxLoc(result_ours, &min_val1, &max_val1, &min_loc1, &max_loc1);
    minMaxLoc(result_opencv, &min_val2, &max_val2, &min_loc2, &max_loc2);

    if (method_name == "SSD" || method_name == "SAD") {
        cout << "Min error location: " << min_loc1 << " vs " << min_loc2 << endl;
        cout << "Min error value: " << min_val1 << " vs " << min_val2 << endl;
    } else {
        cout << "Max similarity location: " << max_loc1 << " vs " << max_loc2 << endl;
        cout << "Max similarity value: " << max_val1 << " vs " << max_val2 << endl;
    }
}

// Feature matching test
void test_feature_matching(const Mat& src1, const Mat& src2) {
    vector<KeyPoint> keypoints1, keypoints2;
    vector<DMatch> matches;

    auto start = chrono::high_resolution_clock::now();
    feature_point_matching(src1, src2, matches, keypoints1, keypoints2);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

    cout << "Feature matching time: " << duration.count() << "ms" << endl;
    cout << "Detected feature points: " << keypoints1.size() << " vs " << keypoints2.size() << endl;
    cout << "Matched feature pairs: " << matches.size() << endl;

    // Visualize matching results
    Mat img_matches;
    drawMatches(src1, keypoints1, src2, keypoints2, matches, img_matches);
    imshow("Feature Matching Results", img_matches);
    waitKey(0);
}

int main() {
    // Read test images
    Mat src = imread("test_images/lena.jpg", IMREAD_GRAYSCALE);
    Mat templ = imread("test_images/lena_eye.jpg", IMREAD_GRAYSCALE);

    if (src.empty() || templ.empty()) {
        cerr << "Cannot read test images" << endl;
        return -1;
    }

    // Test various matching methods
    vector<string> methods = {"SSD", "SAD", "NCC", "ZNCC"};
    for (const auto& method : methods) {
        cout << "\nTesting " << method << " matching method:" << endl;
        test_performance(src, templ, method);
        compare_with_opencv(src, templ, method);

        // Visualize matching results
        Mat result;
        if (method == "SSD") {
            ssd_matching(src, templ, result, TM_SQDIFF);
        } else if (method == "SAD") {
            sad_matching(src, templ, result);
        } else if (method == "NCC") {
            ncc_matching(src, templ, result);
        } else if (method == "ZNCC") {
            zncc_matching(src, templ, result);
        }

        Mat img_result = draw_matching_result(src, templ, result, TM_SQDIFF);
        imshow(method + " Matching Results", img_result);
        waitKey(0);
    }

    // Test feature point matching
    cout << "\nTesting feature point matching" << endl;
    Mat src2 = imread("test_images/lena_rotated.jpg", IMREAD_GRAYSCALE);
    if (src2.empty()) {
        cerr << "Cannot read rotated test image" << endl;
        return -1;
    }
    test_feature_matching(src, src2);

    return 0;
}

