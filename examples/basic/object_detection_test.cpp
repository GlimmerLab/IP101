#include <basic/object_detection.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;
using namespace ip101;

// Performance test function
void test_performance(const Mat& src, const string& method_name) {
    auto start = chrono::high_resolution_clock::now();

    vector<DetectionResult> results;
    if (method_name == "Sliding Window") {
        results = sliding_window_detect(src, Size(64, 128), 8, 0.5);
    } else if (method_name == "HOG+SVM") {
        results = hog_svm_detect(src, 0.5);
    } else if (method_name == "Haar Face") {
        results = haar_face_detect(src, 0.5);
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << method_name << " detection time: " << duration.count() << "ms" << endl;
    cout << "Number of detected objects: " << results.size() << endl;

    // Show results
    Mat img_result = draw_detections(src, results);
    imshow(method_name + " Detection Results", img_result);
    waitKey(0);
}

// Compare with OpenCV implementation
void compare_with_opencv(const Mat& src, const string& method_name) {
    vector<DetectionResult> results_ours;
    vector<Rect> results_opencv;
    vector<double> weights_opencv;

    // Test our implementation
    auto start_ours = chrono::high_resolution_clock::now();
    if (method_name == "HOG+SVM") {
        results_ours = hog_svm_detect(src, 0.5);
    } else if (method_name == "Haar Face") {
        results_ours = haar_face_detect(src, 0.5);
    }
    auto end_ours = chrono::high_resolution_clock::now();
    auto duration_ours = chrono::duration_cast<chrono::milliseconds>(end_ours - start_ours);

    // Test OpenCV implementation
    auto start_opencv = chrono::high_resolution_clock::now();
    if (method_name == "HOG+SVM") {
        HOGDescriptor hog;
        hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
        hog.detectMultiScale(src, results_opencv, weights_opencv);
    } else if (method_name == "Haar Face") {
        CascadeClassifier face_cascade;
        face_cascade.load("haarcascade_frontalface_alt.xml");
        face_cascade.detectMultiScale(src, results_opencv);
    }
    auto end_opencv = chrono::high_resolution_clock::now();
    auto duration_opencv = chrono::duration_cast<chrono::milliseconds>(end_opencv - start_opencv);

    cout << method_name << " performance comparison:" << endl;
    cout << "Our implementation: " << duration_ours.count() << "ms" << endl;
    cout << "OpenCV implementation: " << duration_opencv.count() << "ms" << endl;
    cout << "Speed ratio: " << (float)duration_opencv.count() / duration_ours.count() << "x" << endl;
    cout << "Detection count: " << results_ours.size() << " vs " << results_opencv.size() << endl;

    // Show comparison results
    Mat img_ours = draw_detections(src, results_ours);

    Mat img_opencv = src.clone();
    if (img_opencv.channels() == 1) {
        cvtColor(img_opencv, img_opencv, COLOR_GRAY2BGR);
    }
    for (size_t i = 0; i < results_opencv.size(); i++) {
        rectangle(img_opencv, results_opencv[i], Scalar(0, 255, 0), 2);
        if (!weights_opencv.empty()) {
            string label = format("conf: %.2f", weights_opencv[i]);
            putText(img_opencv, label,
                    Point(results_opencv[i].x, results_opencv[i].y - 5),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
        }
    }

    Mat comparison;
    hconcat(vector<Mat>{img_ours, img_opencv}, comparison);
    imshow(method_name + " Results Comparison (Left: Our | Right: OpenCV)", comparison);
    waitKey(0);
}

// Test NMS
void test_nms(const vector<Rect>& boxes, const vector<float>& scores) {
    cout << "\nTesting Non-Maximum Suppression" << endl;
    cout << "Original detection boxes: " << boxes.size() << endl;

    vector<int> keep = nms(boxes, scores);
    cout << "Boxes kept after NMS: " << keep.size() << endl;
}

// Test object tracking
void test_tracking(VideoCapture& cap) {
    cout << "\nTesting object tracking:" << endl;

    Mat frame, prev_frame;
    vector<DetectionResult> prev_boxes;
    bool first_frame = true;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        if (first_frame) {
            // Detect objects in first frame
            prev_boxes = hog_svm_detect(frame);
            first_frame = false;
        } else {
            // Track objects in subsequent frames
            auto start = chrono::high_resolution_clock::now();
            vector<DetectionResult> curr_boxes = track_objects(frame, prev_frame, prev_boxes);
            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

            cout << "Tracking time: " << duration.count() << "ms" << endl;
            cout << "Tracked objects: " << curr_boxes.size() << endl;

            // Show tracking results
            Mat img_result = draw_detections(frame, curr_boxes);
            imshow("Object Tracking", img_result);

            prev_boxes = curr_boxes;
        }

        frame.copyTo(prev_frame);

        char key = waitKey(30);
        if (key == 27) break;  // ESC to exit
    }
}

int main() {
    // Load test image
    Mat src = imread("test_images/pedestrians.jpg");
    if (src.empty()) {
        cerr << "Cannot read test image" << endl;
        return -1;
    }

    // Test various detection methods
    vector<string> methods = {"Sliding Window", "HOG+SVM", "Haar Face"};
    for (const auto& method : methods) {
        cout << "\nTesting " << method << " detection algorithm" << endl;
        test_performance(src, method);
        if (method != "Sliding Window") {
            compare_with_opencv(src, method);
        }
    }

    // Test NMS
    vector<DetectionResult> detections = hog_svm_detect(src);
    vector<Rect> boxes;
    vector<float> scores;
    for (const auto& det : detections) {
        boxes.push_back(det.bbox);
        scores.push_back(det.confidence);
    }
    test_nms(boxes, scores);

    // Test object tracking
    VideoCapture cap("test_videos/pedestrians.mp4");
    if (!cap.isOpened()) {
        cerr << "Cannot open test video" << endl;
        return -1;
    }
    test_tracking(cap);

    return 0;
}

