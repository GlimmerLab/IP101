#include "object_detection.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;
using namespace ip101;

// 性能测试函数
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
    cout << method_name << " 检测耗时: " << duration.count() << "ms" << endl;
    cout << "检测到的目标数量: " << results.size() << endl;

    // 显示结果
    Mat img_result = draw_detections(src, results);
    imshow(method_name + " 检测结果", img_result);
    waitKey(0);
}

// 与OpenCV实现对比
void compare_with_opencv(const Mat& src, const string& method_name) {
    vector<DetectionResult> results_ours;
    vector<Rect> results_opencv;
    vector<double> weights_opencv;

    // 测试我们的实现
    auto start_ours = chrono::high_resolution_clock::now();
    if (method_name == "HOG+SVM") {
        results_ours = hog_svm_detect(src, 0.5);
    } else if (method_name == "Haar Face") {
        results_ours = haar_face_detect(src, 0.5);
    }
    auto end_ours = chrono::high_resolution_clock::now();
    auto duration_ours = chrono::duration_cast<chrono::milliseconds>(end_ours - start_ours);

    // 测试OpenCV实现
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

    cout << method_name << " 性能对比:" << endl;
    cout << "我们的实现: " << duration_ours.count() << "ms" << endl;
    cout << "OpenCV实现: " << duration_opencv.count() << "ms" << endl;
    cout << "加速比: " << (float)duration_opencv.count() / duration_ours.count() << "x" << endl;
    cout << "检测数量: " << results_ours.size() << " vs " << results_opencv.size() << endl;

    // 显示对比结果
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
    imshow(method_name + " 结果对比 (我们的实现 | OpenCV实现)", comparison);
    waitKey(0);
}

// 测试NMS
void test_nms(const vector<Rect>& boxes, const vector<float>& scores) {
    cout << "\n测试非极大值抑制:" << endl;
    cout << "原始检测框数量: " << boxes.size() << endl;

    vector<int> keep = nms(boxes, scores);
    cout << "NMS后保留的检测框数量: " << keep.size() << endl;
}

// 测试目标跟踪
void test_tracking(const VideoCapture& cap) {
    cout << "\n测试目标跟踪:" << endl;

    Mat frame, prev_frame;
    vector<DetectionResult> prev_boxes;
    bool first_frame = true;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        if (first_frame) {
            // 第一帧进行检测
            prev_boxes = hog_svm_detect(frame);
            first_frame = false;
        } else {
            // 后续帧进行跟踪
            auto start = chrono::high_resolution_clock::now();
            vector<DetectionResult> curr_boxes = track_objects(frame, prev_frame, prev_boxes);
            auto end = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);

            cout << "跟踪耗时: " << duration.count() << "ms" << endl;
            cout << "跟踪目标数量: " << curr_boxes.size() << endl;

            // 显示跟踪结果
            Mat img_result = draw_detections(frame, curr_boxes);
            imshow("目标跟踪", img_result);

            prev_boxes = curr_boxes;
        }

        frame.copyTo(prev_frame);

        char key = waitKey(30);
        if (key == 27) break;  // ESC退出
    }
}

int main() {
    // 读取测试图像
    Mat src = imread("test_images/pedestrians.jpg");
    if (src.empty()) {
        cerr << "无法读取测试图像" << endl;
        return -1;
    }

    // 测试各种检测方法
    vector<string> methods = {"Sliding Window", "HOG+SVM", "Haar Face"};
    for (const auto& method : methods) {
        cout << "\n测试 " << method << " 检测算法:" << endl;
        test_performance(src, method);
        if (method != "Sliding Window") {
            compare_with_opencv(src, method);
        }
    }

    // 测试NMS
    vector<DetectionResult> detections = hog_svm_detect(src);
    vector<Rect> boxes;
    vector<float> scores;
    for (const auto& det : detections) {
        boxes.push_back(det.bbox);
        scores.push_back(det.confidence);
    }
    test_nms(boxes, scores);

    // 测试目标跟踪
    VideoCapture cap("test_videos/pedestrians.mp4");
    if (!cap.isOpened()) {
        cerr << "无法打开测试视频" << endl;
        return -1;
    }
    test_tracking(cap);

    return 0;
}