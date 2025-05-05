#include "texture_analysis.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;
using namespace ip101;

// 性能测试函数
void test_performance(const Mat& src, const string& method_name) {
    auto start = chrono::high_resolution_clock::now();

    if (method_name == "GLCM") {
        Mat glcm = compute_glcm(src);
        vector<double> features = extract_haralick_features(glcm);
        cout << "Haralick特征: ";
        for (double f : features) cout << f << " ";
        cout << endl;
    }
    else if (method_name == "Statistical Features") {
        vector<Mat> features = compute_statistical_features(src);
        for (size_t i = 0; i < features.size(); i++) {
            string feat_name;
            switch(i) {
                case 0: feat_name = "均值"; break;
                case 1: feat_name = "方差"; break;
                case 2: feat_name = "偏度"; break;
                case 3: feat_name = "峰度"; break;
            }
            imshow(feat_name, features[i]);
        }
    }
    else if (method_name == "LBP") {
        Mat lbp = compute_lbp(src);
        imshow("LBP特征", lbp);
    }
    else if (method_name == "Gabor") {
        vector<Mat> filters = generate_gabor_filters();
        vector<Mat> features = extract_gabor_features(src, filters);
        for (size_t i = 0; i < features.size(); i++) {
            imshow("Gabor特征 " + to_string(i), features[i]);
        }
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << method_name << " 处理耗时: " << duration.count() << "ms" << endl;

    waitKey(0);
}

// 与OpenCV实现对比
void compare_with_opencv(const Mat& src, const string& method_name) {
    // 测试我们的实现
    auto start_ours = chrono::high_resolution_clock::now();
    Mat result_ours;
    vector<double> features_ours;

    if (method_name == "GLCM") {
        Mat glcm = compute_glcm(src);
        features_ours = extract_haralick_features(glcm);
    }
    else if (method_name == "LBP") {
        result_ours = compute_lbp(src);
    }

    auto end_ours = chrono::high_resolution_clock::now();
    auto duration_ours = chrono::duration_cast<chrono::milliseconds>(end_ours - start_ours);

    // 测试OpenCV实现
    auto start_opencv = chrono::high_resolution_clock::now();
    Mat result_opencv;
    vector<double> features_opencv;

    if (method_name == "GLCM") {
        Ptr<ml::SVMSGD> svm = ml::SVMSGD::create();
        Mat glcm;

        // OpenCV没有直接的GLCM实现，这里使用替代方案
        int histSize = 256;
        float range[] = {0, 256};
        const float* histRange = {range};
        calcHist(&src, 1, 0, Mat(), glcm, 1, &histSize, &histRange);
        normalize(glcm, glcm, 0, 1, NORM_MINMAX);

        // 提取特征
        features_opencv.push_back(sum(glcm.mul(glcm))[0]);  // 能量
        features_opencv.push_back(sum(abs(glcm))[0]);       // 对比度
    }
    else if (method_name == "LBP") {
        // OpenCV的LBP实现
        Ptr<LBPHFaceRecognizer> lbph = LBPHFaceRecognizer::create();
        Mat hist;
        lbph->compute(src, hist);
        result_opencv = hist.reshape(1, src.rows);
    }

    auto end_opencv = chrono::high_resolution_clock::now();
    auto duration_opencv = chrono::duration_cast<chrono::milliseconds>(end_opencv - start_opencv);

    cout << method_name << " 性能对比:" << endl;
    cout << "我们的实现: " << duration_ours.count() << "ms" << endl;
    cout << "OpenCV实现: " << duration_opencv.count() << "ms" << endl;
    cout << "加速比: " << (float)duration_opencv.count() / duration_ours.count() << "x" << endl;

    // 显示对比结果
    if (method_name == "GLCM") {
        cout << "\n特征值对比:" << endl;
        cout << "我们的实现: ";
        for (double f : features_ours) cout << f << " ";
        cout << "\nOpenCV实现: ";
        for (double f : features_opencv) cout << f << " ";
        cout << endl;
    }
    else if (method_name == "LBP") {
        Mat comparison;
        hconcat(vector<Mat>{result_ours, result_opencv}, comparison);
        imshow(method_name + " 对比 (左: 我们的实现, 右: OpenCV实现)", comparison);
        waitKey(0);
    }
}

int main() {
    // 读取测试图像
    Mat src = imread("test_images/texture.jpg", IMREAD_GRAYSCALE);
    if (src.empty()) {
        cerr << "无法读取测试图像" << endl;
        return -1;
    }

    // 测试各种方法
    vector<string> methods = {
        "GLCM",
        "Statistical Features",
        "LBP",
        "Gabor"
    };

    for (const auto& method : methods) {
        cout << "\n测试 " << method << ":" << endl;
        test_performance(src, method);
        if (method == "GLCM" || method == "LBP") {
            compare_with_opencv(src, method);
        }
    }

    return 0;
}