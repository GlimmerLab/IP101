#include "image_matching.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;
using namespace ip101;

// 性能测试函数
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
    cout << method_name << " 匹配耗时: " << duration.count() << "ms" << endl;
}

// 与OpenCV实现对比
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

    cout << method_name << " 性能对比:" << endl;
    cout << "我们的实现: " << duration_ours.count() << "ms" << endl;
    cout << "OpenCV实现: " << duration_opencv.count() << "ms" << endl;
    cout << "加速比: " << (float)duration_opencv.count() / duration_ours.count() << "x" << endl;

    // 验证结果正确性
    double min_val1, max_val1, min_val2, max_val2;
    Point min_loc1, max_loc1, min_loc2, max_loc2;
    minMaxLoc(result_ours, &min_val1, &max_val1, &min_loc1, &max_loc1);
    minMaxLoc(result_opencv, &min_val2, &max_val2, &min_loc2, &max_loc2);

    if (method_name == "SSD" || method_name == "SAD") {
        cout << "最小误差位置: " << min_loc1 << " vs " << min_loc2 << endl;
        cout << "最小误差值: " << min_val1 << " vs " << min_val2 << endl;
    } else {
        cout << "最大相似度位置: " << max_loc1 << " vs " << max_loc2 << endl;
        cout << "最大相似度值: " << max_val1 << " vs " << max_val2 << endl;
    }
}

// 特征点匹配测试
void test_feature_matching(const Mat& src1, const Mat& src2) {
    vector<KeyPoint> keypoints1, keypoints2;
    vector<DMatch> matches;

    auto start = chrono::high_resolution_clock::now();
    feature_point_matching(src1, src2, matches, keypoints1, keypoints2);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

    cout << "特征点匹配耗时: " << duration.count() << "ms" << endl;
    cout << "检测到的特征点数量: " << keypoints1.size() << " vs " << keypoints2.size() << endl;
    cout << "匹配的特征点对数量: " << matches.size() << endl;

    // 可视化匹配结果
    Mat img_matches;
    drawMatches(src1, keypoints1, src2, keypoints2, matches, img_matches);
    imshow("特征点匹配结果", img_matches);
    waitKey(0);
}

int main() {
    // 读取测试图像
    Mat src = imread("test_images/lena.jpg", IMREAD_GRAYSCALE);
    Mat templ = imread("test_images/lena_eye.jpg", IMREAD_GRAYSCALE);

    if (src.empty() || templ.empty()) {
        cerr << "无法读取测试图像" << endl;
        return -1;
    }

    // 测试各种匹配方法
    vector<string> methods = {"SSD", "SAD", "NCC", "ZNCC"};
    for (const auto& method : methods) {
        cout << "\n测试 " << method << " 匹配方法:" << endl;
        test_performance(src, templ, method);
        compare_with_opencv(src, templ, method);

        // 可视化匹配结果
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
        imshow(method + " 匹配结果", img_result);
        waitKey(0);
    }

    // 测试特征点匹配
    cout << "\n测试特征点匹配:" << endl;
    Mat src2 = imread("test_images/lena_rotated.jpg", IMREAD_GRAYSCALE);
    if (src2.empty()) {
        cerr << "无法读取旋转后的测试图像" << endl;
        return -1;
    }
    test_feature_matching(src, src2);

    return 0;
}