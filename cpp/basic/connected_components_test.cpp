#include "connected_components.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;
using namespace ip101;

// 性能测试函数
void test_performance(const Mat& src, const string& method_name) {
    Mat labels;
    auto start = chrono::high_resolution_clock::now();

    int num_labels;
    if (method_name == "4-Connected") {
        num_labels = label_4connected(src, labels);
    } else {
        num_labels = label_8connected(src, labels);
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << method_name << " 标记耗时: " << duration.count() << "ms" << endl;
    cout << "连通域数量: " << num_labels << endl;
}

// 与OpenCV实现对比
void compare_with_opencv(const Mat& src, const string& method_name) {
    Mat labels_ours, labels_opencv, stats_opencv, centroids_opencv;

    // 测试我们的实现
    auto start_ours = chrono::high_resolution_clock::now();
    int num_labels_ours;
    if (method_name == "4-Connected") {
        num_labels_ours = label_4connected(src, labels_ours);
    } else {
        num_labels_ours = label_8connected(src, labels_ours);
    }
    auto end_ours = chrono::high_resolution_clock::now();
    auto duration_ours = chrono::duration_cast<chrono::milliseconds>(end_ours - start_ours);

    // 测试OpenCV实现
    auto start_opencv = chrono::high_resolution_clock::now();
    int connectivity = (method_name == "4-Connected") ? 4 : 8;
    int num_labels_opencv = connectedComponentsWithStats(src, labels_opencv, stats_opencv,
                                                       centroids_opencv, connectivity);
    auto end_opencv = chrono::high_resolution_clock::now();
    auto duration_opencv = chrono::duration_cast<chrono::milliseconds>(end_opencv - start_opencv);

    cout << method_name << " 性能对比:" << endl;
    cout << "我们的实现: " << duration_ours.count() << "ms" << endl;
    cout << "OpenCV实现: " << duration_opencv.count() << "ms" << endl;
    cout << "加速比: " << (float)duration_opencv.count() / duration_ours.count() << "x" << endl;
    cout << "连通域数量: " << num_labels_ours << " vs " << num_labels_opencv << endl;

    // 验证结果正确性
    Mat diff;
    compare(labels_ours, labels_opencv, diff, CMP_NE);
    int error_count = countNonZero(diff);
    cout << "标记差异像素数: " << error_count << endl;
}

// 测试连通域属性计算
void test_component_analysis(const Mat& src, const Mat& labels, int num_labels) {
    auto start = chrono::high_resolution_clock::now();

    vector<ConnectedComponent> stats = analyze_components(labels, num_labels);

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "属性计算耗时: " << duration.count() << "ms" << endl;

    // 打印每个连通域的属性
    for (const auto& comp : stats) {
        if (comp.area > 100) { // 只显示面积大于100的连通域
            cout << "\n连通域 " << comp.label << " 属性:" << endl;
            cout << "面积: " << comp.area << endl;
            cout << "质心: " << comp.centroid << endl;
            cout << "外接矩形: " << comp.bbox << endl;
            cout << "圆形度: " << comp.circularity << endl;
            cout << "长宽比: " << comp.aspect_ratio << endl;
            cout << "实心度: " << comp.solidity << endl;
        }
    }

    // 可视化结果
    Mat colored = draw_components(src, labels, stats);
    imshow("连通域分析结果", colored);
    waitKey(0);

    // 测试面积过滤
    Mat filtered = filter_components(labels, stats, 100, 10000);
    colored = draw_components(src, filtered, stats);
    imshow("过滤后的连通域", colored);
    waitKey(0);
}

int main() {
    // 读取测试图像
    Mat src = imread("test_images/shapes.png", IMREAD_GRAYSCALE);
    if (src.empty()) {
        cerr << "无法读取测试图像" << endl;
        return -1;
    }

    // 二值化
    threshold(src, src, 128, 255, THRESH_BINARY);

    // 测试4连通和8连通标记
    vector<string> methods = {"4-Connected", "8-Connected"};
    for (const auto& method : methods) {
        cout << "\n测试 " << method << " 标记:" << endl;
        test_performance(src, method);
        compare_with_opencv(src, method);

        // 测试连通域分析
        Mat labels;
        int num_labels;
        if (method == "4-Connected") {
            num_labels = label_4connected(src, labels);
        } else {
            num_labels = label_8connected(src, labels);
        }
        cout << "\n测试连通域分析 (" << method << "):" << endl;
        test_component_analysis(src, labels, num_labels);
    }

    return 0;
}