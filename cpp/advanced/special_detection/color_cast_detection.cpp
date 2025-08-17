#include <advanced/detection/color_cast_detection.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace ip101 {
namespace advanced {

void detect_color_cast(const cv::Mat& src, ColorCastResult& result,
                     const ColorCastDetectionParams& params) {
    CV_Assert(!src.empty());
    CV_Assert(src.channels() == 3);

    // 组合多种偏色检测方法
    ColorCastResult hist_result;
    detect_color_cast_histogram(src, hist_result, params);

    // 如果需要进行白平衡检测
    if (params.auto_white_balance_check) {
        ColorCastResult wb_result;
        detect_color_cast_white_balance(src, wb_result, params);

        // 取两种方法中偏色程度较大的结果
        if (wb_result.color_cast_degree > hist_result.color_cast_degree) {
            result = wb_result;
        } else {
            result = hist_result;
        }
    } else {
        result = hist_result;
    }

    // 如果需要分析颜色分布
    if (params.analyze_distribution && result.has_color_cast) {
        // 生成颜色分布图
        generate_color_distribution_map(src, result.color_distribution_map);
    }
}

void detect_color_cast_histogram(const cv::Mat& src, ColorCastResult& result,
                              const ColorCastDetectionParams& params) {
    CV_Assert(!src.empty());
    CV_Assert(src.channels() == 3);

    // 初始化结果
    result = ColorCastResult();

    // 转换到浮点类型以便精确计算
    cv::Mat float_img;
    src.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    // 分离通道
    std::vector<cv::Mat> channels;
    cv::split(float_img, channels);

    // 计算每个通道的平均值
    cv::Scalar mean_values = cv::mean(float_img);
    cv::Vec3f mean_vec(static_cast<float>(mean_values[0]),
                      static_cast<float>(mean_values[1]),
                      static_cast<float>(mean_values[2]));

    // 计算通道均值的平均
    float avg_mean = (mean_vec[0] + mean_vec[1] + mean_vec[2]) / 3.0f;

    // 计算各通道与均值的偏差，得到偏色向量
    cv::Vec3f color_cast_vec(mean_vec[0] - avg_mean,
                          mean_vec[1] - avg_mean,
                          mean_vec[2] - avg_mean);

    // 如果使用参考白点，考虑参考白点的影响
    if (params.use_reference_white) {
        // 归一化参考白点
        cv::Vec3f normalized_ref = params.reference_white /
                                cv::norm(params.reference_white, cv::NORM_L1);

        // 计算相对于参考白点的偏色
        color_cast_vec = cv::Vec3f(
            mean_vec[0] / normalized_ref[0] - avg_mean,
            mean_vec[1] / normalized_ref[1] - avg_mean,
            mean_vec[2] / normalized_ref[2] - avg_mean
        );
    }

    // 计算偏色程度（使用向量的模长）
    double cast_degree = cv::norm(color_cast_vec);

    // 设置结果
    result.color_cast_vector = color_cast_vec;
    result.color_cast_degree = cast_degree;
    result.has_color_cast = (cast_degree > params.threshold);

    // 如果存在偏色，确定主要方向
    if (result.has_color_cast) {
        result.dominant_color = get_dominant_color_direction(color_cast_vec);
    }

    // 分析颜色直方图
    if (params.analyze_distribution) {
        // 为每个通道计算直方图
        std::vector<cv::Mat> histograms(3);
        int histSize = 256;
        float range[] = { 0, 1 };
        const float* histRange = { range };

        for (int i = 0; i < 3; i++) {
            cv::calcHist(&channels[i], 1, 0, cv::Mat(), histograms[i], 1, &histSize, &histRange);
        }

        // 计算直方图的峰值位置
        std::vector<float> peak_positions(3);
        for (int i = 0; i < 3; i++) {
            float max_val = 0;
            int max_idx = 0;
            for (int j = 0; j < histSize; j++) {
                float val = histograms[i].at<float>(j);
                if (val > max_val) {
                    max_val = val;
                    max_idx = j;
                }
            }
            peak_positions[i] = static_cast<float>(max_idx) / histSize;
        }

        // 峰值位置的偏差也可以指示偏色
        float avg_peak = (peak_positions[0] + peak_positions[1] + peak_positions[2]) / 3.0f;
        cv::Vec3f peak_cast_vec(
            peak_positions[0] - avg_peak,
            peak_positions[1] - avg_peak,
            peak_positions[2] - avg_peak
        );

        // 如果峰值偏差较大，更新偏色程度
        double peak_cast_degree = cv::norm(peak_cast_vec);
        if (peak_cast_degree > result.color_cast_degree) {
            result.color_cast_vector = peak_cast_vec;
            result.color_cast_degree = peak_cast_degree;
            result.has_color_cast = (peak_cast_degree > params.threshold);
            if (result.has_color_cast) {
                result.dominant_color = get_dominant_color_direction(peak_cast_vec);
            }
        }
    }
}

void detect_color_cast_white_balance(const cv::Mat& src, ColorCastResult& result,
                                 const ColorCastDetectionParams& params) {
    CV_Assert(!src.empty());
    CV_Assert(src.channels() == 3);

    // 初始化结果
    result = ColorCastResult();

    // 灰度世界假设：在自然光照下，场景中的平均反射率约为18%灰，对应于RGB三通道的平均值相等
    // 如果三通道平均值不等，可能存在偏色

    // 转换到浮点类型
    cv::Mat float_img;
    src.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    // 计算每个通道的平均值
    cv::Scalar mean_values = cv::mean(float_img);
    cv::Vec3f mean_vec(static_cast<float>(mean_values[0]),
                     static_cast<float>(mean_values[1]),
                     static_cast<float>(mean_values[2]));

    // 使用灰度世界假设计算增益系数
    float gray_avg = (mean_vec[0] + mean_vec[1] + mean_vec[2]) / 3.0f;
    cv::Vec3f gain_vector(gray_avg / mean_vec[0],
                       gray_avg / mean_vec[1],
                       gray_avg / mean_vec[2]);

    // 计算每个通道与理想增益（1.0）的偏差
    cv::Vec3f cast_vector(gain_vector[0] - 1.0f,
                      gain_vector[1] - 1.0f,
                      gain_vector[2] - 1.0f);

    // 使用白点估计方法（假设图像中最亮的点为白色）
    cv::Mat reshaped = float_img.reshape(1, float_img.rows * float_img.cols);
    std::vector<float> max_vals(3, 0);

    // 找出每个通道的最大值（最亮点）
    for (int i = 0; i < reshaped.rows; i++) {
        for (int c = 0; c < 3; c++) {
            float val = reshaped.at<float>(i, c);
            if (val > max_vals[c]) {
                max_vals[c] = val;
            }
        }
    }

    // 计算白点偏差
    float white_avg = (max_vals[0] + max_vals[1] + max_vals[2]) / 3.0f;
    cv::Vec3f white_cast_vec(
        max_vals[0] / white_avg - 1.0f,
        max_vals[1] / white_avg - 1.0f,
        max_vals[2] / white_avg - 1.0f
    );

    // 混合两种方法的结果
    cv::Vec3f final_cast_vec = (cast_vector + white_cast_vec) * 0.5f;

    // 计算偏色程度
    double cast_degree = cv::norm(final_cast_vec);

    // 设置结果
    result.color_cast_vector = final_cast_vec;
    result.color_cast_degree = cast_degree;
    result.has_color_cast = (cast_degree > params.threshold);

    // 如果存在偏色，确定主要方向
    if (result.has_color_cast) {
        result.dominant_color = get_dominant_color_direction(final_cast_vec);
    }
}

void generate_color_distribution_map(const cv::Mat& src, cv::Mat& distribution_map) {
    CV_Assert(!src.empty());
    CV_Assert(src.channels() == 3);

    // 将图像缩小以加快计算
    cv::Mat small_img;
    cv::resize(src, small_img, cv::Size(), 0.1, 0.1, cv::INTER_AREA);

    // 转换到浮点类型
    cv::Mat float_img;
    small_img.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    // 创建颜色分布图
    cv::Mat color_map = cv::Mat::zeros(256, 256, CV_8UC3);

    // 遍历图像的每个像素
    for (int y = 0; y < float_img.rows; y++) {
        for (int x = 0; x < float_img.cols; x++) {
            cv::Vec3f pixel = float_img.at<cv::Vec3f>(y, x);

            // 计算R/G和B/G比率（如果G为0，则设置为1以避免除以0）
            float g = std::max(pixel[1], 0.001f);
            float r_g = pixel[0] / g;
            float b_g = pixel[2] / g;

            // 将比率映射到分布图上
            int map_x = std::min(255, std::max(0, static_cast<int>(r_g * 128)));
            int map_y = 255 - std::min(255, std::max(0, static_cast<int>(b_g * 128)));

            // 增加该点的计数
            color_map.at<cv::Vec3b>(map_y, map_x)[0] = std::min(255, color_map.at<cv::Vec3b>(map_y, map_x)[0] + 1);
            color_map.at<cv::Vec3b>(map_y, map_x)[1] = std::min(255, color_map.at<cv::Vec3b>(map_y, map_x)[1] + 1);
            color_map.at<cv::Vec3b>(map_y, map_x)[2] = std::min(255, color_map.at<cv::Vec3b>(map_y, map_x)[2] + 1);
        }
    }

    // 应用伪彩色映射以便于可视化
    cv::applyColorMap(color_map, distribution_map, cv::COLORMAP_JET);

    // 在分布图上标记中心点
    cv::circle(distribution_map, cv::Point(128, 128), 5, cv::Scalar(0, 0, 0), -1);
    cv::circle(distribution_map, cv::Point(128, 128), 3, cv::Scalar(255, 255, 255), -1);
}

std::string get_dominant_color_direction(const cv::Vec3f& color_vector) {
    // 归一化向量以比较分量大小
    cv::Vec3f normalized = color_vector / cv::norm(color_vector, cv::NORM_L1);
    float r = normalized[0];
    float g = normalized[1];
    float b = normalized[2];

    // 找出最大的正分量和最小的负分量
    float max_positive = -FLT_MAX;
    float min_negative = FLT_MAX;
    int max_pos_idx = -1;
    int min_neg_idx = -1;

    for (int i = 0; i < 3; i++) {
        if (normalized[i] > max_positive) {
            max_positive = normalized[i];
            max_pos_idx = i;
        }
        if (normalized[i] < min_negative) {
            min_negative = normalized[i];
            min_neg_idx = i;
        }
    }

    // 根据主要分量确定偏色方向
    if (max_pos_idx == 0 && min_neg_idx == 2) {
        return "yellow";  // R+ B-
    } else if (max_pos_idx == 0 && min_neg_idx == 1) {
        return "magenta";  // R+ G-
    } else if (max_pos_idx == 1 && min_neg_idx == 0) {
        return "cyan";  // G+ R-
    } else if (max_pos_idx == 1 && min_neg_idx == 2) {
        return "green";  // G+ B-
    } else if (max_pos_idx == 2 && min_neg_idx == 1) {
        return "blue";  // B+ G-
    } else if (max_pos_idx == 2 && min_neg_idx == 0) {
        return "cyan";  // B+ R-
    } else {
        // 如果只看最大的正分量
        if (max_pos_idx == 0) {
            return "red";
        } else if (max_pos_idx == 1) {
            return "green";
        } else {
            return "blue";
        }
    }
}

void visualize_color_cast(const cv::Mat& src, cv::Mat& dst, const ColorCastResult& result) {
    // 创建可视化图像
    dst = src.clone();

    if (!result.has_color_cast) {
        return;  // 如果没有偏色，直接返回原图
    }

    // 在图像上绘制偏色信息
    std::string cast_text = "Color Cast: " + result.dominant_color;
    std::string degree_text = "Degree: " + std::to_string(int(result.color_cast_degree * 100)) + "%";

    // 确定文本颜色（与偏色相反）
    cv::Scalar text_color;
    if (result.dominant_color == "red" || result.dominant_color == "magenta") {
        text_color = cv::Scalar(0, 255, 255);  // 青色
    } else if (result.dominant_color == "green" || result.dominant_color == "yellow") {
        text_color = cv::Scalar(255, 0, 255);  // 紫色
    } else {
        text_color = cv::Scalar(255, 255, 0);  // 黄色
    }

    // 绘制文本
    cv::putText(dst, cast_text, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2);
    cv::putText(dst, degree_text, cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2);

    // 绘制偏色向量指示器
    int center_x = dst.cols / 2;
    int center_y = dst.rows - 50;
    int radius = std::min(dst.cols, dst.rows) / 8;

    // 绘制圆圈
    cv::circle(dst, cv::Point(center_x, center_y), radius, cv::Scalar(255, 255, 255), 2);

    // 计算偏色向量的终点
    float angle = std::atan2(result.color_cast_vector[2], result.color_cast_vector[0]);
    int end_x = center_x + static_cast<int>(radius * std::cos(angle));
    int end_y = center_y - static_cast<int>(radius * std::sin(angle));

    // 绘制偏色向量
    cv::line(dst, cv::Point(center_x, center_y), cv::Point(end_x, end_y), cv::Scalar(0, 0, 255), 2);
    cv::circle(dst, cv::Point(end_x, end_y), 5, cv::Scalar(0, 0, 255), -1);
}

} // namespace advanced
} // namespace ip101