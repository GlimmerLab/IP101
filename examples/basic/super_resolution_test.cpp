#include <basic/super_resolution.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;
using namespace ip101;

// Performance test function
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
    cout << method_name << " processing time: " << duration.count() << "ms" << endl;

    // Show results
    imshow("Original", src);
    imshow(method_name + " Result", result);
    waitKey(0);

    // Save results
    imwrite(method_name + "_result.png", result);
}

// Compare with OpenCV implementation
void compare_with_opencv(const Mat& src, float scale_factor, const string& method_name) {
    // Test our implementation
    auto start_ours = chrono::high_resolution_clock::now();
    Mat result_ours;

    if (method_name == "Bicubic") {
        result_ours = bicubic_sr(src, scale_factor);
    }

    auto end_ours = chrono::high_resolution_clock::now();
    auto duration_ours = chrono::duration_cast<chrono::milliseconds>(end_ours - start_ours);

    // Test OpenCV implementation
    auto start_opencv = chrono::high_resolution_clock::now();
    Mat result_opencv;

    if (method_name == "Bicubic") {
        resize(src, result_opencv, Size(), scale_factor, scale_factor, INTER_CUBIC);
    }

    auto end_opencv = chrono::high_resolution_clock::now();
    auto duration_opencv = chrono::duration_cast<chrono::milliseconds>(end_opencv - start_opencv);

    cout << method_name << " performance comparison:" << endl;
    cout << "Our implementation: " << duration_ours.count() << "ms" << endl;
    cout << "OpenCV implementation: " << duration_opencv.count() << "ms" << endl;
    cout << "Speed ratio: " << (float)duration_opencv.count() / duration_ours.count() << "x" << endl;

    // Show comparison results
    Mat comparison;
    hconcat(vector<Mat>{result_ours, result_opencv}, comparison);
    imshow(method_name + " Comparison (Left: Our | Right: OpenCV)", comparison);
    waitKey(0);

    // Calculate PSNR and SSIM
    double psnr = PSNR(result_ours, result_opencv);
    Scalar ssim = mean(result_ours.mul(result_opencv));
    cout << "PSNR: " << psnr << "dB" << endl;
    cout << "SSIM: " << ssim[0] << endl;

    // Save comparison results
    imwrite(method_name + "_comparison.png", comparison);
}

// Create test images
void create_test_images(vector<Mat>& test_images) {
    // Read real image
    Mat img = imread("test_images/sr_test.jpg");
    if(!img.empty()) {
        test_images.push_back(img);
    }

    // Create synthetic images
    Mat synthetic(256, 256, CV_8UC3);

    // Gradient image
    for(int i = 0; i < synthetic.rows; i++) {
        for(int j = 0; j < synthetic.cols; j++) {
            synthetic.at<Vec3b>(i,j) = Vec3b(i,j,(i+j)/2);
        }
    }
    test_images.push_back(synthetic.clone());

    // Checkerboard image
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

    // Text image
    synthetic = Mat::zeros(256, 256, CV_8UC3);
    putText(synthetic, "OpenCV", Point(20,128),
            FONT_HERSHEY_COMPLEX, 2, Scalar(255,255,255), 2);
    test_images.push_back(synthetic.clone());
}

// Create multi-frame test sequence
void create_test_sequence(vector<Mat>& frames, const Mat& reference) {
    frames.push_back(reference.clone());

    // Add frames with small displacement
    for(int i = 0; i < 4; i++) {
        Mat frame = reference.clone();
        // Random translation
        Mat trans = Mat::eye(2,3,CV_32F);
        trans.at<float>(0,2) = static_cast<float>(rand() % 5 - 2);  // x translation
        trans.at<float>(1,2) = static_cast<float>(rand() % 5 - 2);  // y translation
        warpAffine(reference, frame, trans, reference.size());
        frames.push_back(frame);
    }
}

int main() {
    // Create test images
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
        cout << "\nTesting new image" << endl;
        for(float scale : scale_factors) {
            cout << "\nScale factor: " << scale << "x" << endl;
            for(const auto& method : methods) {
                cout << "\nTesting " << method << ":" << endl;
                test_performance(img, scale, method);
                if(method == "Bicubic") {
                    compare_with_opencv(img, scale, method);
                }
            }
        }

        // Test multi-frame super resolution
        vector<Mat> frames;
        create_test_sequence(frames, img);
        cout << "\nTesting multi-frame super resolution:" << endl;
        for(float scale : scale_factors) {
            cout << "\nScale factor: " << scale << "x" << endl;
            auto start = chrono::high_resolution_clock::now();
            Mat result = multi_frame_sr(frames, scale);
            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
            cout << "Multi-frame super resolution processing time: " << duration.count() << "ms" << endl;

            imshow("Multi-frame super resolution result", result);
            waitKey(0);
            imwrite("multi_frame_sr_result.png", result);
        }
    }

    return 0;
}

