#include "thinning.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;
using namespace ip101;

// 性能测试函数
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
    cout << method_name << " 细化耗时: " << duration.count() << "ms" << endl;

    // 显示结果
    imshow(method_name + " 细化结果", dst);
    waitKey(0);
}

// 与OpenCV实现对比
void compare_with_opencv(const Mat& src, const string& method_name) {
    Mat dst_ours, dst_opencv;

    // 测试我们的实现
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

    // 测试OpenCV实现
    auto start_opencv = chrono::high_resolution_clock::now();
    if (method_name == "Zhang-Suen") {
        ximgproc::thinning(src, dst_opencv, ximgproc::THINNING_ZHANGSUEN);
    } else if (method_name == "Skeleton") {
        Mat dist;
        distanceTransform(src, dist, DIST_L2, DIST_MASK_PRECISE);
        threshold(dist, dst_opencv, 0.7 * dist.at<float>(cv::minMaxLoc(dist).maxLoc), 255, THRESH_BINARY);
    }
    auto end_opencv = chrono::high_resolution_clock::now();
    auto duration_opencv = chrono::duration_cast<chrono::milliseconds>(end_opencv - start_opencv);

    if (!dst_opencv.empty()) {
        cout << method_name << " 性能对比:" << endl;
        cout << "我们的实现: " << duration_ours.count() << "ms" << endl;
        cout << "OpenCV实现: " << duration_opencv.count() << "ms" << endl;
        cout << "加速比: " << (float)duration_opencv.count() / duration_ours.count() << "x" << endl;

        // 验证结果正确性
        Mat diff;
        compare(dst_ours, dst_opencv, diff, CMP_NE);
        int error_count = countNonZero(diff);
        cout << "结果差异像素数: " << error_count << endl;

        // 显示对比结果
        Mat comparison;
        hconcat(vector<Mat>{dst_ours, dst_opencv, diff}, comparison);
        imshow(method_name + " 结果对比 (我们的实现 | OpenCV实现 | 差异)", comparison);
        waitKey(0);
    } else {
        cout << method_name << " OpenCV无对应实现" << endl;
        imshow(method_name + " 我们的实现", dst_ours);
        waitKey(0);
    }
}

// 测试不同形状的细化效果
void test_shapes(const string& method_name) {
    // 创建测试图像
    vector<Mat> test_images;

    // 矩形
    Mat rect = Mat::zeros(100, 200, CV_8UC1);
    rectangle(rect, Point(50, 25), Point(150, 75), Scalar(255), -1);
    test_images.push_back(rect);

    // 圆形
    Mat circle = Mat::zeros(200, 200, CV_8UC1);
    cv::circle(circle, Point(100, 100), 50, Scalar(255), -1);
    test_images.push_back(circle);

    // 文字
    Mat text = Mat::zeros(100, 300, CV_8UC1);
    putText(text, "IP101", Point(50, 70), FONT_HERSHEY_SIMPLEX, 2, Scalar(255), 3);
    test_images.push_back(text);

    // 对每个测试图像进行细化
    for (size_t i = 0; i < test_images.size(); i++) {
        cout << "\n测试形状 " << i+1 << ":" << endl;
        test_performance(test_images[i], method_name);
        compare_with_opencv(test_images[i], method_name);
    }
}

int main() {
    // 读取测试图像
    Mat src = imread("test_images/text.png", IMREAD_GRAYSCALE);
    if (src.empty()) {
        cerr << "无法读取测试图像" << endl;
        return -1;
    }

    // 二值化
    threshold(src, src, 128, 255, THRESH_BINARY);

    // 测试各种细化方法
    vector<string> methods = {"Basic", "Hilditch", "Zhang-Suen", "Skeleton", "Medial-Axis"};
    for (const auto& method : methods) {
        cout << "\n测试 " << method << " 细化算法:" << endl;
        test_performance(src, method);
        compare_with_opencv(src, method);

        // 测试不同形状
        cout << "\n测试 " << method << " 在不同形状上的效果:" << endl;
        test_shapes(method);
    }

    return 0;
}