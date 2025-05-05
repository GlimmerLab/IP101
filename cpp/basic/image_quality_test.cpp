#include "image_quality.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;
using namespace ip101;

// 性能测试函数
void test_performance(const Mat& src1, const Mat& src2, const string& method_name) {
    auto start = chrono::high_resolution_clock::now();
    double score = 0;

    if (method_name == "PSNR") {
        score = compute_psnr(src1, src2);
    }
    else if (method_name == "SSIM") {
        score = compute_ssim(src1, src2);
    }
    else if (method_name == "MSE") {
        score = compute_mse(src1, src2);
    }
    else if (method_name == "VIF") {
        score = compute_vif(src1, src2);
    }
    else if (method_name == "NIQE") {
        score = compute_niqe(src1);
    }
    else if (method_name == "BRISQUE") {
        score = compute_brisque(src1);
    }
    else if (method_name == "MS-SSIM") {
        score = compute_msssim(src1, src2);
    }
    else if (method_name == "FSIM") {
        score = compute_fsim(src1, src2);
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << method_name << " 分数: " << score << endl;
    cout << method_name << " 处理耗时: " << duration.count() << "ms" << endl;
}

// 与OpenCV实现对比
void compare_with_opencv(const Mat& src1, const Mat& src2, const string& method_name) {
    // 测试我们的实现
    auto start_ours = chrono::high_resolution_clock::now();
    double score_ours = 0;

    if (method_name == "PSNR") {
        score_ours = compute_psnr(src1, src2);
    }
    else if (method_name == "SSIM") {
        score_ours = compute_ssim(src1, src2);
    }
    else if (method_name == "MSE") {
        score_ours = compute_mse(src1, src2);
    }

    auto end_ours = chrono::high_resolution_clock::now();
    auto duration_ours = chrono::duration_cast<chrono::milliseconds>(end_ours - start_ours);

    // 测试OpenCV实现
    auto start_opencv = chrono::high_resolution_clock::now();
    double score_opencv = 0;

    if (method_name == "PSNR") {
        score_opencv = PSNR(src1, src2);
    }
    else if (method_name == "SSIM") {
        Scalar ssim = mean(src1.mul(src2));
        score_opencv = ssim[0];
    }
    else if (method_name == "MSE") {
        Mat diff;
        absdiff(src1, src2, diff);
        diff.convertTo(diff, CV_64F);
        multiply(diff, diff, diff);
        score_opencv = mean(diff)[0];
    }

    auto end_opencv = chrono::high_resolution_clock::now();
    auto duration_opencv = chrono::duration_cast<chrono::milliseconds>(end_opencv - start_opencv);

    cout << method_name << " 性能对比:" << endl;
    cout << "我们的实现: " << score_ours << ", 耗时: " << duration_ours.count() << "ms" << endl;
    cout << "OpenCV实现: " << score_opencv << ", 耗时: " << duration_opencv.count() << "ms" << endl;
    cout << "加速比: " << (float)duration_opencv.count() / duration_ours.count() << "x" << endl;
    cout << "分数差异: " << abs(score_ours - score_opencv) << endl;
}

// 创建测试图像
void create_test_images(Mat& src1, Mat& src2, const string& type) {
    src1 = Mat(512, 512, CV_8UC3);
    src2 = Mat(512, 512, CV_8UC3);

    if (type == "Random") {
        randu(src1, Scalar(0,0,0), Scalar(255,255,255));
        randu(src2, Scalar(0,0,0), Scalar(255,255,255));
    }
    else if (type == "Gradient") {
        for(int i = 0; i < src1.rows; i++) {
            for(int j = 0; j < src1.cols; j++) {
                src1.at<Vec3b>(i,j) = Vec3b(i/2, j/2, (i+j)/4);
                src2.at<Vec3b>(i,j) = Vec3b(i/2+10, j/2+10, (i+j)/4+10);
            }
        }
    }
    else if (type == "Pattern") {
        // 创建棋盘格图案
        for(int i = 0; i < src1.rows; i++) {
            for(int j = 0; j < src1.cols; j++) {
                if((i/64 + j/64) % 2 == 0) {
                    src1.at<Vec3b>(i,j) = Vec3b(255,255,255);
                    src2.at<Vec3b>(i,j) = Vec3b(240,240,240);
                } else {
                    src1.at<Vec3b>(i,j) = Vec3b(0,0,0);
                    src2.at<Vec3b>(i,j) = Vec3b(10,10,10);
                }
            }
        }
    }
}

// 添加不同类型的失真
void add_distortion(const Mat& src, Mat& dst, const string& type) {
    dst = src.clone();

    if (type == "Gaussian") {
        Mat noise(src.size(), src.type());
        randn(noise, 0, 25);
        dst += noise;
    }
    else if (type == "Blur") {
        GaussianBlur(src, dst, Size(5,5), 1.5);
    }
    else if (type == "JPEG") {
        vector<uchar> buffer;
        vector<int> params = {IMWRITE_JPEG_QUALITY, 60};
        imencode(".jpg", src, buffer, params);
        dst = imdecode(buffer, IMREAD_COLOR);
    }
}

int main() {
    // 创建测试图像
    vector<string> image_types = {"Random", "Gradient", "Pattern"};
    vector<string> distortion_types = {"Gaussian", "Blur", "JPEG"};
    vector<string> methods = {
        "PSNR",
        "SSIM",
        "MSE",
        "VIF",
        "NIQE",
        "BRISQUE",
        "MS-SSIM",
        "FSIM"
    };

    for (const auto& image_type : image_types) {
        cout << "\n测试图像类型: " << image_type << endl;
        Mat src1, src2;
        create_test_images(src1, src2, image_type);

        for (const auto& distortion : distortion_types) {
            cout << "\n失真类型: " << distortion << endl;
            Mat distorted;
            add_distortion(src1, distorted, distortion);

            for (const auto& method : methods) {
                cout << "\n测试 " << method << ":" << endl;
                test_performance(src1, distorted, method);
                if (method == "PSNR" || method == "SSIM" || method == "MSE") {
                    compare_with_opencv(src1, distorted, method);
                }
            }
        }
    }

    return 0;
}