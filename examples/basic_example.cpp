#include <opencv2/opencv.hpp>
#include "image_processing.h"
#include <iostream>
#include <chrono>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    // 检查命令行参数
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <image_path>" << endl;
        return -1;
    }

    // 读取图像
    Mat image = imread(argv[1]);
    if (image.empty()) {
        cout << "Error: Could not read image " << argv[1] << endl;
        return -1;
    }

    // 创建图像处理器
    ImageProcessor processor;

    // 1. 图像滤波示例
    cout << "1. Image Filtering Example" << endl;

    // 高斯滤波
    auto start = chrono::high_resolution_clock::now();
    Mat blurred = processor.gaussianBlur(image, 5, 1.0);
    auto end = chrono::high_resolution_clock::now();
    cout << "Gaussian blur time: "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;

    // 2. 边缘检测示例
    cout << "\n2. Edge Detection Example" << endl;

    // Sobel边缘检测
    start = chrono::high_resolution_clock::now();
    Mat edges = processor.sobelEdge(blurred);
    end = chrono::high_resolution_clock::now();
    cout << "Sobel edge detection time: "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;

    // 3. 图像变换示例
    cout << "\n3. Image Transformation Example" << endl;

    // 旋转图像
    start = chrono::high_resolution_clock::now();
    Mat rotated = processor.rotateImage(image, 45, Point2f(image.cols/2, image.rows/2));
    end = chrono::high_resolution_clock::now();
    cout << "Image rotation time: "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;

    // 4. 颜色空间转换示例
    cout << "\n4. Color Space Conversion Example" << endl;

    // RGB转灰度
    start = chrono::high_resolution_clock::now();
    Mat gray = processor.rgbToGray(image);
    end = chrono::high_resolution_clock::now();
    cout << "RGB to gray conversion time: "
         << chrono::duration_cast<chrono::milliseconds>(end - start).count()
         << " ms" << endl;

    // 保存结果
    imwrite("blurred.jpg", blurred);
    imwrite("edges.jpg", edges);
    imwrite("rotated.jpg", rotated);
    imwrite("gray.jpg", gray);

    cout << "\nResults saved to: blurred.jpg, edges.jpg, rotated.jpg, gray.jpg" << endl;

    return 0;
}