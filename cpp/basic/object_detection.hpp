#ifndef OBJECT_DETECTION_HPP
#define OBJECT_DETECTION_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace ip101 {

/**
 * @brief 检测结果结构体
 */
struct DetectionResult {
    cv::Rect bbox;         // 边界框
    float confidence;      // 置信度
    int class_id;         // 类别ID
    std::string label;    // 类别标签
};

/**
 * @brief 滑动窗口检测
 * @param src 输入图像
 * @param window_size 窗口大小
 * @param stride 步长
 * @param threshold 检测阈值
 * @return 检测结果列表
 */
std::vector<DetectionResult> sliding_window_detect(
    const cv::Mat& src,
    const cv::Size& window_size,
    int stride,
    float threshold);

/**
 * @brief HOG+SVM行人检测
 * @param src 输入图像
 * @param threshold 检测阈值
 * @return 检测结果列表
 */
std::vector<DetectionResult> hog_svm_detect(
    const cv::Mat& src,
    float threshold = 0.5);

/**
 * @brief Haar+AdaBoost人脸检测
 * @param src 输入图像
 * @param threshold 检测阈值
 * @return 检测结果列表
 */
std::vector<DetectionResult> haar_face_detect(
    const cv::Mat& src,
    float threshold = 0.5);

/**
 * @brief 非极大值抑制
 * @param boxes 边界框列表
 * @param scores 置信度列表
 * @param iou_threshold NMS阈值
 * @return 保留的边界框索引
 */
std::vector<int> nms(
    const std::vector<cv::Rect>& boxes,
    const std::vector<float>& scores,
    float iou_threshold = 0.5);

/**
 * @brief 目标跟踪
 * @param src 当前帧
 * @param prev 前一帧
 * @param prev_boxes 前一帧的检测框
 * @return 当前帧的跟踪结果
 */
std::vector<DetectionResult> track_objects(
    const cv::Mat& src,
    const cv::Mat& prev,
    const std::vector<DetectionResult>& prev_boxes);

/**
 * @brief 绘制检测结果
 * @param src 输入图像
 * @param detections 检测结果列表
 * @return 标注后的图像
 */
cv::Mat draw_detections(
    const cv::Mat& src,
    const std::vector<DetectionResult>& detections);

} // namespace ip101

#endif // OBJECT_DETECTION_HPP