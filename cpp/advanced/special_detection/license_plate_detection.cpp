#include <advanced/detection/license_plate_detection.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace ip101 {
namespace advanced {

void detect_license_plates(const cv::Mat& src, std::vector<LicensePlateInfo>& plates,
                          const LicensePlateDetectionParams& params) {
    CV_Assert(!src.empty());

    // 清空输出向量
    plates.clear();

    // 结合两种方法进行车牌检测
    std::vector<LicensePlateInfo> edge_based_plates;
    detect_plates_edge_based(src, edge_based_plates, params);

    std::vector<LicensePlateInfo> color_based_plates;
    detect_plates_color_based(src, color_based_plates, params);

    // 合并两种方法的结果
    plates.insert(plates.end(), edge_based_plates.begin(), edge_based_plates.end());
    plates.insert(plates.end(), color_based_plates.begin(), color_based_plates.end());

    // 非极大值抑制，去除重叠的检测结果
    if (!plates.empty()) {
        std::vector<LicensePlateInfo> nms_plates;
        std::vector<int> indices(plates.size());
        std::vector<float> confidences(plates.size());
        std::vector<cv::Rect> rects(plates.size());

        for (size_t i = 0; i < plates.size(); i++) {
            indices[i] = static_cast<int>(i);
            confidences[i] = static_cast<float>(plates[i].confidence);
            rects[i] = plates[i].rect;
        }

        // 使用OpenCV的NMS函数
        std::vector<int> keep_indices;
        cv::dnn::NMSBoxes(rects, confidences, static_cast<float>(params.min_plate_confidence), 0.3f, keep_indices);

        // 保留NMS后的车牌信息
        for (int idx : keep_indices) {
            nms_plates.push_back(plates[idx]);
        }

        plates = nms_plates;
    }

    // 处理每个检测到的车牌，提取字符
    for (auto& plate : plates) {
        if (plate.confidence >= params.min_plate_confidence) {
            // 校正车牌倾斜
            cv::Mat corrected_plate;
            correct_plate_skew(plate.plate_img, corrected_plate);
            plate.plate_img = corrected_plate;

            // 分割字符
            segment_plate_chars(plate.plate_img, plate.chars);
        }
    }
}

void detect_plates_edge_based(const cv::Mat& src, std::vector<LicensePlateInfo>& plates,
                             const LicensePlateDetectionParams& params) {
    CV_Assert(!src.empty());

    // 清空输出向量
    plates.clear();

    // 转换为灰度图像
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // 增强对比度
    cv::Mat contrast_enhanced;
    cv::equalizeHist(gray, contrast_enhanced);

    // 高斯模糊去噪
    cv::Mat blurred;
    cv::GaussianBlur(contrast_enhanced, blurred, cv::Size(5, 5), 0);

    // Sobel边缘检测
    cv::Mat grad_x, grad_y, grad;
    cv::Sobel(blurred, grad_x, CV_16S, 1, 0, 3);
    cv::Sobel(blurred, grad_y, CV_16S, 0, 1, 3);

    cv::Mat abs_grad_x, abs_grad_y;
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);

    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    // 二值化处理
    cv::Mat binary;
    cv::threshold(grad, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // 形态学处理，连接边缘
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(17, 3));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, element);

    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 计算图像面积
    double img_area = src.rows * src.cols;
    double min_area = img_area * params.min_area_ratio;
    double max_area = img_area * params.max_area_ratio;

    // 处理轮廓
    for (const auto& contour : contours) {
        // 计算轮廓面积
        double area = cv::contourArea(contour);

        // 面积过滤
        if (area < min_area || area > max_area) {
            continue;
        }

        // 计算最小外接矩形
        cv::RotatedRect rot_rect = cv::minAreaRect(contour);
        cv::Rect rect = rot_rect.boundingRect();

        // 确保矩形在图像内部
        rect &= cv::Rect(0, 0, src.cols, src.rows);
        if (rect.width <= 0 || rect.height <= 0) {
            continue;
        }

        // 计算长宽比
        double aspect_ratio = static_cast<double>(rect.width) / rect.height;

        // 长宽比过滤
        if (aspect_ratio < params.min_aspect_ratio || aspect_ratio > params.max_aspect_ratio) {
            continue;
        }

        // 提取候选车牌区域
        cv::Mat plate_img = src(rect).clone();

        // 计算置信度（这里使用一个简单的启发式方法）
        // 对车牌区域进行分析，计算垂直边缘数量，越多越可能是车牌
        cv::Mat plate_gray;
        cv::cvtColor(plate_img, plate_gray, cv::COLOR_BGR2GRAY);

        cv::Mat plate_sobel_x;
        cv::Sobel(plate_gray, plate_sobel_x, CV_8U, 1, 0);

        cv::Mat plate_thresh;
        cv::threshold(plate_sobel_x, plate_thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        // 统计垂直边缘的数量
        int edge_count = cv::countNonZero(plate_thresh);

        // 计算边缘密度（边缘像素数量除以面积）
        double edge_density = static_cast<double>(edge_count) / (rect.width * rect.height);

        // 边缘密度越高，越可能是车牌
        double confidence = std::min(1.0, edge_density * 100.0);

        // 创建车牌信息结构体
        if (confidence >= params.min_plate_confidence) {
            LicensePlateInfo plate_info;
            plate_info.rect = rect;
            plate_info.plate_img = plate_img;
            plate_info.confidence = confidence;

            plates.push_back(plate_info);
        }
    }
}

void detect_plates_color_based(const cv::Mat& src, std::vector<LicensePlateInfo>& plates,
                              const LicensePlateDetectionParams& params) {
    CV_Assert(!src.empty());
    CV_Assert(src.channels() == 3);  // 颜色检测需要彩色图像

    // 清空输出向量
    plates.clear();

    // 转换到HSV颜色空间，便于颜色分割
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    // 定义蓝色车牌的颜色范围（中国标准蓝牌）
    cv::Scalar lower_blue(100, 70, 70);
    cv::Scalar upper_blue(130, 255, 255);

    // 定义黄色车牌的颜色范围
    cv::Scalar lower_yellow(15, 70, 70);
    cv::Scalar upper_yellow(35, 255, 255);

    // 创建掩码
    cv::Mat blue_mask, yellow_mask, color_mask;
    cv::inRange(hsv, lower_blue, upper_blue, blue_mask);
    cv::inRange(hsv, lower_yellow, upper_yellow, yellow_mask);

    // 合并两种颜色的掩码
    cv::bitwise_or(blue_mask, yellow_mask, color_mask);

    // 形态学处理
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(17, 3));
    cv::morphologyEx(color_mask, color_mask, cv::MORPH_CLOSE, element);
    cv::morphologyEx(color_mask, color_mask, cv::MORPH_OPEN, element);

    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(color_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 计算图像面积
    double img_area = src.rows * src.cols;
    double min_area = img_area * params.min_area_ratio;
    double max_area = img_area * params.max_area_ratio;

    // 处理轮廓
    for (const auto& contour : contours) {
        // 计算轮廓面积
        double area = cv::contourArea(contour);

        // 面积过滤
        if (area < min_area || area > max_area) {
            continue;
        }

        // 计算最小外接矩形
        cv::RotatedRect rot_rect = cv::minAreaRect(contour);
        cv::Rect rect = rot_rect.boundingRect();

        // 确保矩形在图像内部
        rect &= cv::Rect(0, 0, src.cols, src.rows);
        if (rect.width <= 0 || rect.height <= 0) {
            continue;
        }

        // 计算长宽比
        double aspect_ratio = static_cast<double>(rect.width) / rect.height;

        // 长宽比过滤
        if (aspect_ratio < params.min_aspect_ratio || aspect_ratio > params.max_aspect_ratio) {
            continue;
        }

        // 提取候选车牌区域
        cv::Mat plate_img = src(rect).clone();

        // 计算该区域内蓝色或黄色像素的比例
        cv::Mat plate_hsv;
        cv::cvtColor(plate_img, plate_hsv, cv::COLOR_BGR2HSV);

        cv::Mat plate_blue_mask, plate_yellow_mask, plate_color_mask;
        cv::inRange(plate_hsv, lower_blue, upper_blue, plate_blue_mask);
        cv::inRange(plate_hsv, lower_yellow, upper_yellow, plate_yellow_mask);
        cv::bitwise_or(plate_blue_mask, plate_yellow_mask, plate_color_mask);

        int color_pixels = cv::countNonZero(plate_color_mask);
        double color_ratio = static_cast<double>(color_pixels) / (rect.width * rect.height);

        // 颜色比例越高，越可能是车牌
        double confidence = std::min(1.0, color_ratio * 2.0);  // 乘以2是为了归一化到0-1范围

        // 创建车牌信息结构体
        if (confidence >= params.min_plate_confidence) {
            LicensePlateInfo plate_info;
            plate_info.rect = rect;
            plate_info.plate_img = plate_img;
            plate_info.confidence = confidence;

            plates.push_back(plate_info);
        }
    }
}

void correct_plate_skew(const cv::Mat& plate_img, cv::Mat& corrected_img) {
    CV_Assert(!plate_img.empty());

    // 转换为灰度图像
    cv::Mat gray;
    if (plate_img.channels() == 3) {
        cv::cvtColor(plate_img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = plate_img.clone();
    }

    // 二值化处理
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    // 如果没有找到轮廓，直接返回原图
    if (contours.empty()) {
        corrected_img = plate_img.clone();
        return;
    }

    // 按轮廓面积排序
    std::sort(contours.begin(), contours.end(),
             [](const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2) {
                 return cv::contourArea(c1) > cv::contourArea(c2);
             });

    // 取最大的轮廓，可能是车牌边框
    const auto& largest_contour = contours[0];

    // 最小矩形拟合，获取角度
    cv::RotatedRect rotated_rect = cv::minAreaRect(largest_contour);
    float angle = rotated_rect.angle;

    // 确保角度在 -45 到 45 度之间
    if (angle < -45) {
        angle += 90;
    }

    // 如果角度太小，不需要校正
    if (std::abs(angle) < 1.0) {
        corrected_img = plate_img.clone();
        return;
    }

    // 使用仿射变换进行旋转校正
    cv::Point2f center(plate_img.cols / 2.0f, plate_img.rows / 2.0f);
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::warpAffine(plate_img, corrected_img, rotation_matrix, plate_img.size(),
                  cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
}

void segment_plate_chars(const cv::Mat& plate_img, std::vector<cv::Rect>& chars) {
    CV_Assert(!plate_img.empty());

    // 清空输出向量
    chars.clear();

    // 转换为灰度图像
    cv::Mat gray;
    if (plate_img.channels() == 3) {
        cv::cvtColor(plate_img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = plate_img.clone();
    }

    // 自适应阈值二值化
    cv::Mat binary;
    cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2);

    // 去除小噪点
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, element);

    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 过滤并排序字符轮廓
    std::vector<cv::Rect> char_rects;
    for (const auto& contour : contours) {
        // 计算轮廓面积
        double area = cv::contourArea(contour);

        // 面积过滤（去除太小的噪点）
        if (area < 100) {  // 根据实际情况调整阈值
            continue;
        }

        // 计算边界矩形
        cv::Rect rect = cv::boundingRect(contour);

        // 宽高比过滤（字符的宽高比通常在一定范围内）
        float aspect_ratio = static_cast<float>(rect.width) / rect.height;
        if (aspect_ratio < 0.1 || aspect_ratio > 1.2) {  // 根据实际情况调整阈值
            continue;
        }

        // 高度过滤（字符高度通常在一定范围内）
        float height_ratio = static_cast<float>(rect.height) / plate_img.rows;
        if (height_ratio < 0.3 || height_ratio > 0.9) {  // 根据实际情况调整阈值
            continue;
        }

        char_rects.push_back(rect);
    }

    // 按照x坐标从左到右排序
    std::sort(char_rects.begin(), char_rects.end(),
             [](const cv::Rect& r1, const cv::Rect& r2) {
                 return r1.x < r2.x;
             });

    // 返回排序后的字符区域
    chars = char_rects;
}

void draw_license_plates(cv::Mat& img, const std::vector<LicensePlateInfo>& plates, bool draw_chars) {
    for (const auto& plate : plates) {
        // 绘制车牌矩形
        cv::rectangle(img, plate.rect, cv::Scalar(0, 255, 0), 2);

        // 绘制置信度
        std::string conf_text = "Conf: " + std::to_string(int(plate.confidence * 100)) + "%";
        cv::putText(img, conf_text, cv::Point(plate.rect.x, plate.rect.y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

        // 如果需要，绘制字符区域
        if (draw_chars && !plate.chars.empty()) {
            for (const auto& char_rect : plate.chars) {
                // 转换字符区域坐标到原始图像坐标
                cv::Rect global_char_rect(
                    plate.rect.x + char_rect.x,
                    plate.rect.y + char_rect.y,
                    char_rect.width,
                    char_rect.height
                );
                cv::rectangle(img, global_char_rect, cv::Scalar(255, 0, 0), 1);
            }
        }

        // 如果有识别的车牌号码，绘制出来
        if (!plate.plate_number.empty()) {
            cv::putText(img, plate.plate_number,
                       cv::Point(plate.rect.x, plate.rect.y + plate.rect.height + 20),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
        }
    }
}

} // namespace advanced
} // namespace ip101