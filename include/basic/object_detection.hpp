#ifndef OBJECT_DETECTION_HPP
#define OBJECT_DETECTION_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace ip101 {

/**
 * @brief Detection result structure
 */
struct DetectionResult {
    cv::Rect bbox;         // Bounding box
    float confidence;      // Confidence score
    int class_id;         // Class ID
    std::string label;    // Class label
};

/**
 * @brief Sliding window detection
 * @param src Input image
 * @param window_size Window size
 * @param stride Step size
 * @param threshold Detection threshold
 * @return List of detection results
 */
std::vector<DetectionResult> sliding_window_detect(
    const cv::Mat& src,
    const cv::Size& window_size,
    int stride,
    float threshold);

/**
 * @brief HOG+SVM pedestrian detection
 * @param src Input image
 * @param threshold Detection threshold
 * @return List of detection results
 */
std::vector<DetectionResult> hog_svm_detect(
    const cv::Mat& src,
    float threshold = 0.5);

/**
 * @brief Haar+AdaBoost face detection
 * @param src Input image
 * @param threshold Detection threshold
 * @return List of detection results
 */
std::vector<DetectionResult> haar_face_detect(
    const cv::Mat& src,
    float threshold = 0.5);

/**
 * @brief Non-maximum suppression
 * @param boxes List of bounding boxes
 * @param scores List of confidence scores
 * @param iou_threshold NMS threshold
 * @return Indices of kept bounding boxes
 */
std::vector<int> nms(
    const std::vector<cv::Rect>& boxes,
    const std::vector<float>& scores,
    float iou_threshold = 0.5);

/**
 * @brief Object tracking
 * @param src Current frame
 * @param prev Previous frame
 * @param prev_boxes Detection boxes from previous frame
 * @return Tracking results for current frame
 */
std::vector<DetectionResult> track_objects(
    const cv::Mat& src,
    const cv::Mat& prev,
    const std::vector<DetectionResult>& prev_boxes);

/**
 * @brief Draw detection results
 * @param src Input image
 * @param detections List of detection results
 * @return Annotated image
 */
cv::Mat draw_detections(
    const cv::Mat& src,
    const std::vector<DetectionResult>& detections);

} // namespace ip101

#endif // OBJECT_DETECTION_HPP