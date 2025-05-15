#include <basic/texture_analysis.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;
using namespace ip101;

// Manual implementation of simple LBP algorithm for OpenCV comparison
Mat compute_lbp_manual(const Mat& src, int radius = 1, int neighbors = 8) {
    Mat dst = Mat::zeros(src.size(), CV_8U);

    // Pre-compute sampling point coordinates
    vector<int> sample_points_x(neighbors);
    vector<int> sample_points_y(neighbors);

    for(int i = 0; i < neighbors; i++) {
        // Calculate coordinates on the circle
        double angle = 2.0 * CV_PI * i / neighbors;
        sample_points_x[i] = static_cast<int>(radius * cos(angle));
        sample_points_y[i] = static_cast<int>(-radius * sin(angle));
    }

    // Calculate LBP values
    for(int y = radius; y < src.rows - radius; y++) {
        for(int x = radius; x < src.cols - radius; x++) {
            uchar center = src.at<uchar>(y, x);
            uchar lbp_code = 0;

            // Calculate binary pattern
            for(int k = 0; k < neighbors; k++) {
                int sample_y = y + sample_points_y[k];
                int sample_x = x + sample_points_x[k];
                uchar neighbor = src.at<uchar>(sample_y, sample_x);

                // If neighbor pixel is greater than center pixel, set bit to 1
                if(neighbor > center) {
                    lbp_code |= (1 << k);
                }
            }

            dst.at<uchar>(y, x) = lbp_code;
        }
    }

    return dst;
}

// Performance test function
void test_performance(const Mat& src, const string& method_name) {
    auto start = chrono::high_resolution_clock::now();

    if (method_name == "GLCM") {
        Mat glcm = compute_glcm(src);
        vector<double> features = extract_haralick_features(glcm);
        cout << "Haralick features: ";
        for (double f : features) cout << f << " ";
        cout << endl;
    }
    else if (method_name == "Statistical Features") {
        vector<Mat> features = compute_statistical_features(src);
        for (size_t i = 0; i < features.size(); i++) {
            string feat_name;
            switch(i) {
                case 0: feat_name = "Mean"; break;
                case 1: feat_name = "Variance"; break;
                case 2: feat_name = "Skewness"; break;
                case 3: feat_name = "Kurtosis"; break;
            }
            imshow(feat_name, features[i]);
        }
    }
    else if (method_name == "LBP") {
        Mat lbp = compute_lbp(src);
        imshow("LBP Features", lbp);
    }
    else if (method_name == "Gabor") {
        vector<Mat> filters = generate_gabor_filters();
        vector<Mat> features = extract_gabor_features(src, filters);
        for (size_t i = 0; i < features.size(); i++) {
            imshow("Gabor Feature " + to_string(i), features[i]);
        }
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << method_name << " processing time: " << duration.count() << "ms" << endl;

    waitKey(0);
}

// Compare with OpenCV implementation
void compare_with_opencv(const Mat& src, const string& method_name) {
    // Test our implementation
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

    // Test OpenCV implementation
    auto start_opencv = chrono::high_resolution_clock::now();
    Mat result_opencv;
    vector<double> features_opencv;

    if (method_name == "GLCM") {
        Ptr<ml::SVMSGD> svm = ml::SVMSGD::create();
        Mat glcm;

        // OpenCV doesn't have a direct GLCM implementation, using alternative method
        int histSize = 256;
        float range[] = {0, 256};
        const float* histRange = {range};
        calcHist(&src, 1, 0, Mat(), glcm, 1, &histSize, &histRange);
        normalize(glcm, glcm, 0, 1, NORM_MINMAX);

        // Extract features
        features_opencv.push_back(sum(glcm.mul(glcm))[0]);  // Energy
        features_opencv.push_back(sum(abs(glcm))[0]);       // Contrast
    }
    else if (method_name == "LBP") {
        // Use manual implementation of LBP algorithm as "OpenCV" version
        result_opencv = compute_lbp_manual(src);
    }

    auto end_opencv = chrono::high_resolution_clock::now();
    auto duration_opencv = chrono::duration_cast<chrono::milliseconds>(end_opencv - start_opencv);

    cout << method_name << " performance comparison:" << endl;
    cout << "Our implementation: " << duration_ours.count() << "ms" << endl;
    cout << "Manual implementation: " << duration_opencv.count() << "ms" << endl;
    cout << "Speed ratio: " << (float)duration_opencv.count() / duration_ours.count() << "x" << endl;

    // Show comparison results
    if (method_name == "GLCM") {
        cout << "\nFeature value comparison" << endl;
        cout << "Our implementation: ";
        for (double f : features_ours) cout << f << " ";
        cout << "\nOpenCV implementation: ";
        for (double f : features_opencv) cout << f << " ";
        cout << endl;
    }
    else if (method_name == "LBP") {
        Mat comparison;
        hconcat(vector<Mat>{result_ours, result_opencv}, comparison);
        imshow(method_name + " Comparison (Left: Our | Right: Manual)", comparison);
        waitKey(0);
    }
}

int main() {
    // Read test image
    Mat src = imread("test_images/texture.jpg", IMREAD_GRAYSCALE);
    if (src.empty()) {
        cerr << "Cannot read test image" << endl;
        return -1;
    }

    // Test various methods
    vector<string> methods = {
        "GLCM",
        "Statistical Features",
        "LBP",
        "Gabor"
    };

    for (const auto& method : methods) {
        cout << "\nTesting " << method << ":" << endl;
        test_performance(src, method);
        if (method == "GLCM" || method == "LBP") {
            compare_with_opencv(src, method);
        }
    }

    return 0;
}

