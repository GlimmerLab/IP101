#include "super_resolution.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;
using namespace ip101;

// 性能测试函数
void test_performance(const Mat& src, float scale_factor, const string& method_name) {
    auto start = chrono::high_resolution_clock::now();
    Mat result;

    if (method_name == "Bicubic") {
        result = bicubic_sr(src, scale_factor);
    }
    else if (method_name == "Sparse") {
        result = sparse_sr(src, scale_factor);
    }
    else if (method_name == "SRCNN") {
        result = srcnn_sr(src, scale_factor);
    }
    else if (method_name == "AdaptiveWeight") {
        result = adaptive_weight_sr(src, scale_factor);
    }
    else if (method_name == "IterativeBackprojection") {
        result = iterative_backprojection_sr(src, scale_factor);
    }
    else if (method_name == "GradientGuided") {
        result = gradient_guided_sr(src, scale_factor);
    }
    else if (method_name == "SelfSimilarity") {
        result = self_similarity_sr(src, scale_factor);
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << method_name << " 处理耗时: " << duration.count() << "ms" << endl;

    // 显示结果
    imshow("原图", src);
    imshow(method_name + " 结果", result);
    waitKey(0);

    // 保存结果
    imwrite(method_name + "_result.png", result);
}

// 与OpenCV实现对比
void compare_with_opencv(const Mat& src, float scale_factor, const string& method_name) {
    // 测试我们的实现
    auto start_ours = chrono::high_resolution_clock::now();
    Mat result_ours;

    if (method_name == "Bicubic") {
        result_ours = bicubic_sr(src, scale_factor);
    }

    auto end_ours = chrono::high_resolution_clock::now();
    auto duration_ours = chrono::duration_cast<chrono::milliseconds>(end_ours - start_ours);

    // 测试OpenCV实现
    auto start_opencv = chrono::high_resolution_clock::now();
    Mat result_opencv;

    if (method_name == "Bicubic") {
        resize(src, result_opencv, Size(), scale_factor, scale_factor, INTER_CUBIC);
    }

    auto end_opencv = chrono::high_resolution_clock::now();
    auto duration_opencv = chrono::duration_cast<chrono::milliseconds>(end_opencv - start_opencv);

    cout << method_name << " 性能对比:" << endl;
    cout << "我们的实现: " << duration_ours.count() << "ms" << endl;
    cout << "OpenCV实现: " << duration_opencv.count() << "ms" << endl;
    cout << "加速比: " << (float)duration_opencv.count() / duration_ours.count() << "x" << endl;

    // 显示对比结果
    Mat comparison;
    hconcat(vector<Mat>{result_ours, result_opencv}, comparison);
    imshow(method_name + " 对比 (左: 我们的实现, 右: OpenCV实现)", comparison);
    waitKey(0);

    // 计算PSNR和SSIM
    double psnr = PSNR(result_ours, result_opencv);
    Scalar ssim = mean(result_ours.mul(result_opencv));
    cout << "PSNR: " << psnr << "dB" << endl;
    cout << "SSIM: " << ssim[0] << endl;

    // 保存对比结果
    imwrite(method_name + "_comparison.png", comparison);
}

// 创建测试图像
void create_test_images(vector<Mat>& test_images) {
    // 读取真实图像
    Mat img = imread("test_images/sr_test.jpg");
    if(!img.empty()) {
        test_images.push_back(img);
    }

    // 创建合成图像
    Mat synthetic(256, 256, CV_8UC3);

    // 渐变图像
    for(int i = 0; i < synthetic.rows; i++) {
        for(int j = 0; j < synthetic.cols; j++) {
            synthetic.at<Vec3b>(i,j) = Vec3b(i,j,(i+j)/2);
        }
    }
    test_images.push_back(synthetic.clone());

    // 棋盘格图案
    for(int i = 0; i < synthetic.rows; i++) {
        for(int j = 0; j < synthetic.cols; j++) {
            if((i/32 + j/32) % 2 == 0) {
                synthetic.at<Vec3b>(i,j) = Vec3b(255,255,255);
            } else {
                synthetic.at<Vec3b>(i,j) = Vec3b(0,0,0);
            }
        }
    }
    test_images.push_back(synthetic.clone());

    // 文本图像
    synthetic = Mat::zeros(256, 256, CV_8UC3);
    putText(synthetic, "OpenCV", Point(20,128),
            FONT_HERSHEY_COMPLEX, 2, Scalar(255,255,255), 2);
    test_images.push_back(synthetic.clone());
}

// 创建多帧测试序列
void create_test_sequence(vector<Mat>& frames, const Mat& reference) {
    frames.push_back(reference.clone());

    // 添加带有小位移的帧
    for(int i = 0; i < 4; i++) {
        Mat frame = reference.clone();
        // 随机平移
        Mat trans = Mat::eye(2,3,CV_32F);
        trans.at<float>(0,2) = rand() % 5 - 2;  // x方向平移
        trans.at<float>(1,2) = rand() % 5 - 2;  // y方向平移
        warpAffine(reference, frame, trans, reference.size());
        frames.push_back(frame);
    }
}

int main() {
    // 创建测试图像
    vector<Mat> test_images;
    create_test_images(test_images);

    vector<string> methods = {
        "Bicubic",
        "Sparse",
        "SRCNN",
        "AdaptiveWeight",
        "IterativeBackprojection",
        "GradientGuided",
        "SelfSimilarity"
    };

    vector<float> scale_factors = {2.0f, 3.0f, 4.0f};

    for(const auto& img : test_images) {
        cout << "\n测试新图像:" << endl;
        for(float scale : scale_factors) {
            cout << "\n放大倍数: " << scale << "x" << endl;
            for(const auto& method : methods) {
                cout << "\n测试 " << method << ":" << endl;
                test_performance(img, scale, method);
                if(method == "Bicubic") {
                    compare_with_opencv(img, scale, method);
                }
            }
        }

        // 测试多帧超分辨率
        vector<Mat> frames;
        create_test_sequence(frames, img);
        cout << "\n测试多帧超分辨率:" << endl;
        for(float scale : scale_factors) {
            cout << "\n放大倍数: " << scale << "x" << endl;
            auto start = chrono::high_resolution_clock::now();
            Mat result = multi_frame_sr(frames, scale);
            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
            cout << "多帧超分辨率处理耗时: " << duration.count() << "ms" << endl;

            imshow("多帧超分辨率结果", result);
            waitKey(0);
            imwrite("multi_frame_sr_result.png", result);
        }
    }

    return 0;
}