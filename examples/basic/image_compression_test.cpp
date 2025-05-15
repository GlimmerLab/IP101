#include <basic/image_compression.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;
using namespace ip101;

// Performance testing function
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

    // Calculate quality metrics
    double psnr = compute_compression_psnr(src, result);

    // Output results
    cout << "\n" << method_name << " compression test results:" << endl;
    cout << "Processing time: " << duration.count() << "ms" << endl;
    cout << "Compression ratio: " << compression_ratio * 100 << "%" << endl;
    cout << "PSNR: " << psnr << "dB" << endl;

    // Display results
    imshow("Original", src);
    imshow(method_name + " result", result);
    waitKey(0);
}

// Compare with OpenCV implementation
void compare_with_opencv(const Mat& src, int quality = 80) {
    // Test our implementation
    auto start_ours = chrono::high_resolution_clock::now();
    Mat result_ours;
    double compression_ratio_ours = jpeg_compress_manual(src, result_ours, quality);
    auto end_ours = chrono::high_resolution_clock::now();
    auto duration_ours = chrono::duration_cast<chrono::milliseconds>(end_ours - start_ours);

    // Test OpenCV implementation
    auto start_opencv = chrono::high_resolution_clock::now();
    vector<uchar> buffer;
    Mat result_opencv;
    vector<int> params = {IMWRITE_JPEG_QUALITY, quality};
    imencode(".jpg", src, buffer, params);
    result_opencv = imdecode(buffer, IMREAD_COLOR);
    auto end_opencv = chrono::high_resolution_clock::now();
    auto duration_opencv = chrono::duration_cast<chrono::milliseconds>(end_opencv - start_opencv);

    // Calculate OpenCV compression ratio
    double compression_ratio_opencv = compute_compression_ratio(
        src.total() * src.elemSize(), buffer.size());

    // Calculate quality metrics
    double psnr_ours = compute_compression_psnr(src, result_ours);
    double psnr_opencv = compute_compression_psnr(src, result_opencv);

    // Output performance comparison
    cout << "\nJPEG compression performance comparison:" << endl;
    cout << "Our implementation:" << endl;
    cout << "  Processing time: " << duration_ours.count() << "ms" << endl;
    cout << "  Compression ratio: " << compression_ratio_ours * 100 << "%" << endl;
    cout << "  PSNR: " << psnr_ours << "dB" << endl;
    cout << "OpenCV implementation:" << endl;
    cout << "  Processing time: " << duration_opencv.count() << "ms" << endl;
    cout << "  Compression ratio: " << compression_ratio_opencv * 100 << "%" << endl;
    cout << "  PSNR: " << psnr_opencv << "dB" << endl;
    cout << "Speed ratio: " << (float)duration_opencv.count() / duration_ours.count() << "x" << endl;

    // Show comparison results
    Mat comparison;
    hconcat(vector<Mat>{result_ours, result_opencv}, comparison);
    imshow("JPEG compression comparison (Left: Ours, Right: OpenCV)", comparison);
    waitKey(0);
}

// Test different quality levels
void test_quality_levels(const Mat& src) {
    vector<int> qualities = {10, 30, 50, 70, 90};

    cout << "\nJPEG quality parameter test:" << endl;
    for(int quality : qualities) {
        cout << "\nQuality parameter: " << quality << endl;
        test_performance(src, "JPEG", quality);
    }
}

// Test different image types
void test_image_types(const string& method_name) {
    vector<pair<string, string>> image_types = {
        {"Natural image", "test_images/natural.jpg"},
        {"Text image", "test_images/text.png"},
        {"Synthetic image", "test_images/synthetic.png"},
        {"Medical image", "test_images/medical.png"}
    };

    cout << "\n" << method_name << " test on different image types:" << endl;
    for(const auto& [type_name, image_path] : image_types) {
        Mat img = imread(image_path);
        if(img.empty()) {
            cout << "Cannot read image: " << image_path << endl;
            continue;
        }
        cout << "\nImage type: " << type_name << endl;
        test_performance(img, method_name);
    }
}

int main() {
    // Read test image
    Mat src = imread("test_images/test.jpg");
    if(src.empty()) {
        cerr << "Cannot read test image" << endl;
        return -1;
    }

    // Test all compression methods
    vector<string> methods = {
        "RLE",
        "JPEG",
        "Fractal",
        "Wavelet"
    };

    // Basic performance tests
    for(const auto& method : methods) {
        test_performance(src, method);
    }

    // Comparison with OpenCV
    compare_with_opencv(src);

    // JPEG quality parameter tests
    test_quality_levels(src);

    // Different image types tests
    for(const auto& method : methods) {
        test_image_types(method);
    }

    return 0;
}

