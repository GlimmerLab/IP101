#include <basic/image_features.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;
using namespace ip101;

// Performance testing function
void test_performance(const Mat& src) {
    cout << "===== Feature Extraction Performance Tests =====" << endl;

    // Test HOG features
    {
        vector<float> hog_feat;
        auto start = chrono::high_resolution_clock::now();
        hog_features(src, hog_feat);
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

        cout << "HOG features extraction time: " << duration.count() << "ms" << endl;
        cout << "HOG feature dimension: " << hog_feat.size() << endl;
    }

    // Test LBP features
    {
        Mat lbp_result;
        auto start = chrono::high_resolution_clock::now();
        lbp_features(src, lbp_result);
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

        cout << "LBP features extraction time: " << duration.count() << "ms" << endl;
    }

    // Test Haar features
    {
        vector<float> haar_feat;
        auto start = chrono::high_resolution_clock::now();
        haar_features(src, haar_feat);
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

        cout << "Haar features extraction time: " << duration.count() << "ms" << endl;
        cout << "Haar feature dimension: " << haar_feat.size() << endl;
    }

    // Test Gabor features
    {
        vector<float> gabor_feat;
        auto start = chrono::high_resolution_clock::now();
        gabor_features(src, gabor_feat);
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

        cout << "Gabor features extraction time: " << duration.count() << "ms" << endl;
        cout << "Gabor feature dimension: " << gabor_feat.size() << endl;
    }

    // Test Color Histogram (for color images only)
    if (src.channels() == 3) {
        Mat hist;
        auto start = chrono::high_resolution_clock::now();
        color_histogram(src, hist);
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

        cout << "Color histogram extraction time: " << duration.count() << "ms" << endl;
        cout << "Color histogram dimension: " << hist.total() << endl;
    }
}

// Compare with OpenCV implementation
void compare_with_opencv(const Mat& src) {
    cout << "\n===== Comparison with OpenCV =====" << endl;

    // Compare HOG feature extraction
    {
        // Our implementation
        vector<float> our_features;
        auto start_ours = chrono::high_resolution_clock::now();
        hog_features(src, our_features);
        auto end_ours = chrono::high_resolution_clock::now();
        auto duration_ours = chrono::duration_cast<chrono::milliseconds>(end_ours - start_ours);

        // OpenCV implementation
        HOGDescriptor hog;
        vector<float> opencv_features;
        auto start_opencv = chrono::high_resolution_clock::now();
        hog.compute(src, opencv_features);
        auto end_opencv = chrono::high_resolution_clock::now();
        auto duration_opencv = chrono::duration_cast<chrono::milliseconds>(end_opencv - start_opencv);

        cout << "HOG feature extraction comparison:" << endl;
        cout << "Our implementation: " << duration_ours.count() << "ms" << endl;
        cout << "OpenCV implementation: " << duration_opencv.count() << "ms" << endl;
        cout << "Speed ratio: " << (float)duration_opencv.count() / duration_ours.count() << "x" << endl;
    }

    // Compare LBP feature extraction
    {
        // Our implementation
        Mat our_lbp;
        auto start_ours = chrono::high_resolution_clock::now();
        lbp_features(src, our_lbp);
        auto end_ours = chrono::high_resolution_clock::now();
        auto duration_ours = chrono::duration_cast<chrono::milliseconds>(end_ours - start_ours);

        cout << "LBP feature extraction time: " << duration_ours.count() << "ms" << endl;

        // OpenCV doesn't have a direct LBP implementation in the main modules
    }
}

// Visualize feature extraction results
void visualize_results(const Mat& src) {
    cout << "\n===== Feature Visualization =====" << endl;

    // Visualize LBP
    Mat lbp_result;
    lbp_features(src, lbp_result);

    // Normalize LBP for visualization
    Mat lbp_vis;
    normalize(lbp_result, lbp_vis, 0, 255, NORM_MINMAX, CV_8U);

    // Create Gabor filters and visualize
    vector<Mat> gabor_filters = create_gabor_filters(5, 8);

    // Create a combined visualization of filters
    int filter_size = gabor_filters[0].rows;
    int scales = 5;
    int orientations = 8;

    Mat gabor_vis = Mat::zeros(filter_size * scales, filter_size * orientations, CV_32F);

    for (int s = 0; s < scales; s++) {
        for (int o = 0; o < orientations; o++) {
            Mat filter = gabor_filters[s * orientations + o];
            normalize(filter, filter, 0, 1, NORM_MINMAX);

            Rect roi(o * filter_size, s * filter_size, filter_size, filter_size);
            filter.copyTo(gabor_vis(roi));
        }
    }

    // Normalize for display
    normalize(gabor_vis, gabor_vis, 0, 255, NORM_MINMAX, CV_8U);

    // Display results
    imshow("Original Image", src);
    imshow("LBP Features", lbp_vis);
    imshow("Gabor Filter Bank", gabor_vis);

    waitKey(0);
}

int main(int argc, char** argv) {
    // Read input image
    Mat src;

    if (argc > 1) {
        src = imread(argv[1]);
        if (src.empty()) {
            cerr << "Cannot read image: " << argv[1] << endl;
            return -1;
        }
    } else {
        // Try to load a default test image
        src = imread("test_images/lena.jpg");
        if (src.empty()) {
            cerr << "Cannot read default test image. Please provide an image path." << endl;
            cerr << "Usage: " << argv[0] << " <image_path>" << endl;
            return -1;
        }
    }

    // Resize image if it's too large
    if (src.cols > 640 || src.rows > 480) {
        resize(src, src, Size(640, 480));
    }

    cout << "Image size: " << src.cols << "x" << src.rows << endl;

    // Run performance tests
    test_performance(src);

    // Compare with OpenCV
    compare_with_opencv(src);

    // Visualize results
    visualize_results(src);

    return 0;
}