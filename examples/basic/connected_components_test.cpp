#include <basic/connected_components.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;
using namespace ip101;

// Performance test function
void test_performance(const Mat& src, const string& method_name) {
    Mat labels;
    auto start = chrono::high_resolution_clock::now();

    int num_labels;
    if (method_name == "4-Connected") {
        num_labels = label_4connected(src, labels);
    } else {
        num_labels = label_8connected(src, labels);
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << method_name << " labeling time: " << duration.count() << "ms" << endl;
    cout << "Number of components: " << num_labels << endl;
}

// Compare with OpenCV implementation
void compare_with_opencv(const Mat& src, const string& method_name) {
    Mat labels_ours, labels_opencv, stats_opencv, centroids_opencv;

    // Test our implementation
    auto start_ours = chrono::high_resolution_clock::now();
    int num_labels_ours;
    if (method_name == "4-Connected") {
        num_labels_ours = label_4connected(src, labels_ours);
    } else {
        num_labels_ours = label_8connected(src, labels_ours);
    }
    auto end_ours = chrono::high_resolution_clock::now();
    auto duration_ours = chrono::duration_cast<chrono::milliseconds>(end_ours - start_ours);

    // Test OpenCV implementation
    auto start_opencv = chrono::high_resolution_clock::now();
    int connectivity = (method_name == "4-Connected") ? 4 : 8;
    int num_labels_opencv = connectedComponentsWithStats(src, labels_opencv, stats_opencv,
                                                       centroids_opencv, connectivity);
    auto end_opencv = chrono::high_resolution_clock::now();
    auto duration_opencv = chrono::duration_cast<chrono::milliseconds>(end_opencv - start_opencv);

    cout << method_name << " performance comparison:" << endl;
    cout << "Our implementation: " << duration_ours.count() << "ms" << endl;
    cout << "OpenCV implementation: " << duration_opencv.count() << "ms" << endl;
    cout << "Speed ratio: " << (float)duration_opencv.count() / duration_ours.count() << "x" << endl;
    cout << "Number of components: " << num_labels_ours << " vs " << num_labels_opencv << endl;

    // Verify result correctness
    Mat diff;
    compare(labels_ours, labels_opencv, diff, CMP_NE);
    int error_count = countNonZero(diff);
    cout << "Pixel difference count: " << error_count << endl;
}

// Test component analysis
void test_component_analysis(const Mat& src, const Mat& labels, int num_labels) {
    auto start = chrono::high_resolution_clock::now();

    vector<ConnectedComponent> stats = analyze_components(labels, num_labels);

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "Analysis time: " << duration.count() << "ms" << endl;

    // Print properties of each component
    for (const auto& comp : stats) {
        if (comp.area > 100) { // Only show components with area > 100
            cout << "\nComponent " << comp.label << " properties:" << endl;
            cout << "Area: " << comp.area << endl;
            cout << "Centroid: " << comp.centroid << endl;
            cout << "Bounding box: " << comp.bbox << endl;
            cout << "Circularity: " << comp.circularity << endl;
            cout << "Aspect ratio: " << comp.aspect_ratio << endl;
            cout << "Solidity: " << comp.solidity << endl;
        }
    }

    // Visualize results
    Mat colored = draw_components(src, labels, stats);
    imshow("Component Analysis Results", colored);
    waitKey(0);

    // Test area filtering
    Mat filtered = filter_components(labels, stats, 100, 10000);
    colored = draw_components(src, filtered, stats);
    imshow("Filtered Components", colored);
    waitKey(0);
}

int main() {
    // Read test image
    Mat src = imread("test_images/shapes.png", IMREAD_GRAYSCALE);
    if (src.empty()) {
        cerr << "Cannot read test image" << endl;
        return -1;
    }

    // Binarize
    threshold(src, src, 128, 255, THRESH_BINARY);

    // Test 4-connected and 8-connected labeling
    vector<string> methods = {"4-Connected", "8-Connected"};
    for (const auto& method : methods) {
        cout << "\nTesting " << method << " labeling:" << endl;
        test_performance(src, method);
        compare_with_opencv(src, method);

        // Test component analysis
        Mat labels;
        int num_labels;
        if (method == "4-Connected") {
            num_labels = label_4connected(src, labels);
        } else {
            num_labels = label_8connected(src, labels);
        }
        cout << "\nTesting component analysis (" << method << "):" << endl;
        test_component_analysis(src, labels, num_labels);
    }

    return 0;
}

