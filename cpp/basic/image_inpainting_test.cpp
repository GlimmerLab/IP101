#include "image_inpainting.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;
using namespace ip101;

// 性能测试函数
void test_performance(const Mat& src, const Mat& mask, const string& method_name) {
    auto start = chrono::high_resolution_clock::now();
    Mat result;

    if (method_name == "Diffusion") {
        result = diffusion_inpaint(src, mask);
    }
    else if (method_name == "PatchMatch") {
        result = patch_match_inpaint(src, mask);
    }
    else if (method_name == "FastMarching") {
        result = fast_marching_inpaint(src, mask);
    }
    else if (method_name == "TextureSynthesis") {
        result = texture_synthesis_inpaint(src, mask);
    }
    else if (method_name == "StructurePropagation") {
        result = structure_propagation_inpaint(src, mask);
    }
    else if (method_name == "PatchMatchV2") {
        result = patchmatch_inpaint(src, mask);
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << method_name << " 处理耗时: " << duration.count() << "ms" << endl;

    // 显示结果
    imshow("原图", src);
    imshow("掩码", mask);
    imshow(method_name + " 结果", result);
    waitKey(0);
}

// 与OpenCV实现对比
void compare_with_opencv(const Mat& src, const Mat& mask, const string& method_name) {
    // 测试我们的实现
    auto start_ours = chrono::high_resolution_clock::now();
    Mat result_ours;

    if (method_name == "Diffusion") {
        result_ours = diffusion_inpaint(src, mask);
    }
    else if (method_name == "FastMarching") {
        result_ours = fast_marching_inpaint(src, mask);
    }
    else if (method_name == "PatchMatch") {
        result_ours = patchmatch_inpaint(src, mask);
    }

    auto end_ours = chrono::high_resolution_clock::now();
    auto duration_ours = chrono::duration_cast<chrono::milliseconds>(end_ours - start_ours);

    // 测试OpenCV实现
    auto start_opencv = chrono::high_resolution_clock::now();
    Mat result_opencv;

    if (method_name == "Diffusion") {
        inpaint(src, mask, result_opencv, 3, INPAINT_NS);
    }
    else if (method_name == "FastMarching") {
        inpaint(src, mask, result_opencv, 3, INPAINT_TELEA);
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
}

// 测试视频修复
void test_video_inpaint(const string& video_path, const Mat& mask) {
    // 读取视频帧
    vector<Mat> frames;
    vector<Mat> masks;
    VideoCapture cap(video_path);
    for(int i = 0; i < 5; i++) {  // 读取5帧
        Mat frame;
        if(cap.read(frame)) {
            frames.push_back(frame);
            masks.push_back(mask.clone());
        }
    }
    cap.release();

    if(frames.empty()) {
        cout << "无法读取视频帧" << endl;
        return;
    }

    // 测试视频修复
    auto start = chrono::high_resolution_clock::now();
    vector<Mat> results = video_inpaint(frames, masks);
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "视频修复处理耗时: " << duration.count() << "ms" << endl;

    // 显示结果
    for(size_t i = 0; i < results.size(); i++) {
        imshow("原始帧 " + to_string(i), frames[i]);
        imshow("修复结果 " + to_string(i), results[i]);
    }
    waitKey(0);

    // 保存结果视频
    VideoWriter writer("result_video.avi",
                      VideoWriter::fourcc('M','J','P','G'),
                      30, frames[0].size());
    for(const auto& frame : results) {
        writer.write(frame);
    }
    writer.release();
}

// 创建测试掩码
Mat create_test_mask(const Mat& src, const string& type) {
    Mat mask = Mat::zeros(src.size(), CV_8U);

    if (type == "Rectangle") {
        // 创建矩形掩码
        int x = src.cols/4;
        int y = src.rows/4;
        int width = src.cols/2;
        int height = src.rows/2;
        rectangle(mask, Rect(x,y,width,height), Scalar(255), -1);
    }
    else if (type == "Lines") {
        // 创建随机线条掩码
        RNG rng(12345);
        for(int i = 0; i < 10; i++) {
            Point pt1(rng.uniform(0, src.cols), rng.uniform(0, src.rows));
            Point pt2(rng.uniform(0, src.cols), rng.uniform(0, src.rows));
            line(mask, pt1, pt2, Scalar(255), 5);
        }
    }
    else if (type == "Text") {
        // 创建文字掩码
        putText(mask, "OpenCV", Point(src.cols/4, src.rows/2),
                FONT_HERSHEY_COMPLEX, 2, Scalar(255), 3);
    }

    return mask;
}

int main() {
    // 读取测试图像
    Mat src = imread("test_images/inpaint.jpg");
    if (src.empty()) {
        cerr << "无法读取测试图像" << endl;
        return -1;
    }

    // 创建不同类型的测试掩码
    vector<string> mask_types = {"Rectangle", "Lines", "Text"};
    vector<string> methods = {
        "Diffusion",
        "PatchMatch",
        "FastMarching",
        "TextureSynthesis",
        "StructurePropagation",
        "PatchMatchV2"
    };

    for (const auto& mask_type : mask_types) {
        cout << "\n测试掩码类型: " << mask_type << endl;
        Mat mask = create_test_mask(src, mask_type);

        for (const auto& method : methods) {
            cout << "\n测试 " << method << ":" << endl;
            test_performance(src, mask, method);
            if (method == "Diffusion" || method == "FastMarching" || method == "PatchMatch") {
                compare_with_opencv(src, mask, method);
            }
        }
    }

    // 测试视频修复
    cout << "\n测试视频修复:" << endl;
    test_video_inpaint("test_images/test_video.mp4", create_test_mask(src, "Rectangle"));

    return 0;
}