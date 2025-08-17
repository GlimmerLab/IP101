#ifndef LICENSE_PLATE_DETECTION_HPP
#define LICENSE_PLATE_DETECTION_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace ip101 {
namespace advanced {

/**
 * @brief 车牌信息结构体
 */
struct LicensePlateInfo {
    cv::Rect rect;                 // 车牌区域矩形
    cv::Mat plate_img;             // 车牌图像
    double confidence;             // 检测置信度
    std::vector<cv::Rect> chars;   // 字符区域
    std::string plate_number;      // 车牌号码（如果进行了字符识别）
};

/**
 * @brief 车牌检测参数结构体
 */
struct LicensePlateDetectionParams {
    double min_area_ratio;         // 车牌面积与图像面积的最小比例
    double max_area_ratio;         // 车牌面积与图像面积的最大比例
    double min_aspect_ratio;       // 车牌宽高比最小值
    double max_aspect_ratio;       // 车牌宽高比最大值
    double min_plate_confidence;   // 最小车牌检测置信度

    // 默认构造函数
    LicensePlateDetectionParams() :
        min_area_ratio(0.001),
        max_area_ratio(0.05),
        min_aspect_ratio(2.0),
        max_aspect_ratio(6.0),
        min_plate_confidence(0.6)
    {}
};

/**
 * @brief 车牌检测算法
 * @param src 输入图像
 * @param plates 检测到的车牌信息
 * @param params 车牌检测参数
 */
void detect_license_plates(const cv::Mat& src, std::vector<LicensePlateInfo>& plates,
                         const LicensePlateDetectionParams& params = LicensePlateDetectionParams());

/**
 * @brief 基于边缘检测和几何特性的车牌定位
 * @param src 输入图像
 * @param plates 检测到的车牌区域
 * @param params 车牌检测参数
 */
void detect_plates_edge_based(const cv::Mat& src, std::vector<LicensePlateInfo>& plates,
                            const LicensePlateDetectionParams& params);

/**
 * @brief 基于颜色特征的车牌定位
 * @param src 输入图像
 * @param plates 检测到的车牌区域
 * @param params 车牌检测参数
 */
void detect_plates_color_based(const cv::Mat& src, std::vector<LicensePlateInfo>& plates,
                             const LicensePlateDetectionParams& params);

/**
 * @brief 车牌倾斜校正
 * @param plate_img 输入车牌图像
 * @param corrected_img 校正后的图像
 */
void correct_plate_skew(const cv::Mat& plate_img, cv::Mat& corrected_img);

/**
 * @brief 车牌字符分割
 * @param plate_img 车牌图像
 * @param chars 分割后的字符区域
 */
void segment_plate_chars(const cv::Mat& plate_img, std::vector<cv::Rect>& chars);

/**
 * @brief 在图像上绘制检测到的车牌
 * @param img 输入/输出图像
 * @param plates 检测到的车牌信息
 * @param draw_chars 是否绘制字符区域
 */
void draw_license_plates(cv::Mat& img, const std::vector<LicensePlateInfo>& plates, bool draw_chars = false);

} // namespace advanced
} // namespace ip101

#endif // LICENSE_PLATE_DETECTION_HPP