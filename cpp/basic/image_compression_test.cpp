#include "image_compression.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;
using namespace ip101;

// 性能测试函数
void test_performance(const Mat& src, const string& method_name, int quality = 80) {
    auto start = chrono::high_resolution_clock::now();
    Mat result;
    double compression_ratio = 0.0;

    if (method_name == "RLE") {
        vector<uchar> encoded;
        compression_ratio = rle_encode(src, encoded);
        rle_decode(encoded, result, src.size());
    }
    else if (method_name == "JPEG") {
        compression_ratio = jpeg_compress_manual(src, result, quality);
    }
    else if (method_name == "Fractal") {
        compression_ratio = fractal_compress(src, result);
    }
    else if (method_name == "Wavelet") {
        compression_ratio = wavelet_compress(src, result);
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

    // 计算质量指标
    double psnr = compute_psnr(src, result);

    // 输出结果
    cout << "\n" << method_name << " 压缩测试结果:" << endl;
    cout << "处理时间: " << duration.count() << "ms" << endl;
    cout << "压缩率: " << compression_ratio * 100 << "%" << endl;
    cout << "PSNR: " << psnr << "dB" << endl;

    // 显示结果
    imshow("原图", src);
    imshow(method_name + " 结果", result);
    waitKey(0);
}

// 与OpenCV实现对比
void compare_with_opencv(const Mat& src, int quality = 80) {
    // 测试我们的实现
    auto start_ours = chrono::high_resolution_clock::now();
    Mat result_ours;
    double compression_ratio_ours = jpeg_compress_manual(src, result_ours, quality);
    auto end_ours = chrono::high_resolution_clock::now();
    auto duration_ours = chrono::duration_cast<chrono::milliseconds>(end_ours - start_ours);

    // 测试OpenCV实现
    auto start_opencv = chrono::high_resolution_clock::now();
    vector<uchar> buffer;
    Mat result_opencv;
    vector<int> params = {IMWRITE_JPEG_QUALITY, quality};
    imencode(".jpg", src, buffer, params);
    result_opencv = imdecode(buffer, IMREAD_COLOR);
    auto end_opencv = chrono::high_resolution_clock::now();
    auto duration_opencv = chrono::duration_cast<chrono::milliseconds>(end_opencv - start_opencv);

    // 计算OpenCV的压缩率
    double compression_ratio_opencv = compute_compression_ratio(
        src.total() * src.elemSize(), buffer.size());

    // 计算质量指标
    double psnr_ours = compute_psnr(src, result_ours);
    double psnr_opencv = compute_psnr(src, result_opencv);

    // 输出性能对比
    cout << "\nJPEG压缩性能对比:" << endl;
    cout << "我们的实现:" << endl;
    cout << "  处理时间: " << duration_ours.count() << "ms" << endl;
    cout << "  压缩率: " << compression_ratio_ours * 100 << "%" << endl;
    cout << "  PSNR: " << psnr_ours << "dB" << endl;
    cout << "OpenCV实现:" << endl;
    cout << "  处理时间: " << duration_opencv.count() << "ms" << endl;
    cout << "  压缩率: " << compression_ratio_opencv * 100 << "%" << endl;
    cout << "  PSNR: " << psnr_opencv << "dB" << endl;
    cout << "加速比: " << (float)duration_opencv.count() / duration_ours.count() << "x" << endl;

    // 显示对比结果
    Mat comparison;
    hconcat(vector<Mat>{result_ours, result_opencv}, comparison);
    imshow("JPEG压缩对比 (左: 我们的实现, 右: OpenCV实现)", comparison);
    waitKey(0);
}

// 测试不同质量参数
void test_quality_levels(const Mat& src) {
    vector<int> qualities = {10, 30, 50, 70, 90};

    cout << "\nJPEG不同质量参数测试:" << endl;
    for(int quality : qualities) {
        cout << "\n质量参数: " << quality << endl;
        test_performance(src, "JPEG", quality);
    }
}

// 测试不同图像类型
void test_image_types(const string& method_name) {
    vector<pair<string, string>> image_types = {
        {"自然图像", "test_images/natural.jpg"},
        {"文本图像", "test_images/text.png"},
        {"合成图像", "test_images/synthetic.png"},
        {"医学图像", "test_images/medical.png"}
    };

    cout << "\n" << method_name << " 不同图像类型测试:" << endl;
    for(const auto& [type_name, image_path] : image_types) {
        Mat img = imread(image_path);
        if(img.empty()) {
            cout << "无法读取图像: " << image_path << endl;
            continue;
        }
        cout << "\n图像类型: " << type_name << endl;
        test_performance(img, method_name);
    }
}

int main() {
    // 读取测试图像
    Mat src = imread("test_images/test.jpg");
    if(src.empty()) {
        cerr << "无法读取测试图像" << endl;
        return -1;
    }

    // 测试所有压缩方法
    vector<string> methods = {
        "RLE",
        "JPEG",
        "Fractal",
        "Wavelet"
    };

    // 基本性能测试
    for(const auto& method : methods) {
        test_performance(src, method);
    }

    // 与OpenCV对比测试
    compare_with_opencv(src);

    // JPEG质量参数测试
    test_quality_levels(src);

    // 不同图像类型测试
    for(const auto& method : methods) {
        test_image_types(method);
    }

    return 0;
}
