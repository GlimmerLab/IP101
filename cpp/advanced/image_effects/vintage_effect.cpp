#include <advanced/effects/vintage_effect.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <omp.h>

namespace ip101 {
namespace advanced {

void apply_sepia_tone(const cv::Mat& src, cv::Mat& dst, double intensity) {
    CV_Assert(!src.empty());
    CV_Assert(intensity >= 0.0 && intensity <= 1.0);

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 褐色调颜色变换矩阵
    // 标准褐色调系数
    // | R' |   | 0.393  0.769  0.189 | | R |
    // | G' | = | 0.349  0.686  0.168 | | G |
    // | B' |   | 0.272  0.534  0.131 | | B |

    if (src.channels() == 3) {
        // 对于彩色图像，直接应用颜色变换
        #pragma omp parallel for
        for (int y = 0; y < src.rows; y++) {
            for (int x = 0; x < src.cols; x++) {
                cv::Vec3b pixel = src.at<cv::Vec3b>(y, x);

                // 计算褐色调值
                float b = 0.272f * pixel[0] + 0.534f * pixel[1] + 0.131f * pixel[2];
                float g = 0.349f * pixel[0] + 0.686f * pixel[1] + 0.168f * pixel[2];
                float r = 0.393f * pixel[0] + 0.769f * pixel[1] + 0.189f * pixel[2];

                // 根据强度混合原始颜色和褐色调
                dst.at<cv::Vec3b>(y, x)[0] = cv::saturate_cast<uchar>((1.0 - intensity) * pixel[0] + intensity * b);
                dst.at<cv::Vec3b>(y, x)[1] = cv::saturate_cast<uchar>((1.0 - intensity) * pixel[1] + intensity * g);
                dst.at<cv::Vec3b>(y, x)[2] = cv::saturate_cast<uchar>((1.0 - intensity) * pixel[2] + intensity * r);
            }
        }
    } else {
        // 对于灰度图像，创建单通道查找表
        cv::Mat sepia_table(1, 256, CV_8U);
        for (int i = 0; i < 256; i++) {
            // 计算褐色调的灰度值
            float sepia_value = 0.272f * i + 0.534f * i + 0.131f * i;
            sepia_value = std::min(255.0f, sepia_value);

            // 根据强度混合原始颜色和褐色调
            sepia_table.at<uchar>(0, i) = cv::saturate_cast<uchar>((1.0 - intensity) * i + intensity * sepia_value);
        }

        // 应用查找表
        cv::LUT(src, sepia_table, dst);
    }
}

void add_film_grain(const cv::Mat& src, cv::Mat& dst, double noise_level) {
    CV_Assert(!src.empty());
    CV_Assert(noise_level >= 0.0);

    // 创建随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(0.0, noise_level);

    // 创建噪点图像
    cv::Mat noise = cv::Mat::zeros(src.size(), src.type());

    // 生成噪点
    #pragma omp parallel for
    for (int y = 0; y < noise.rows; y++) {
        for (int x = 0; x < noise.cols; x++) {
            if (src.channels() == 1) {
                double noise_value = distribution(gen);
                noise.at<uchar>(y, x) = cv::saturate_cast<uchar>(noise_value);
            } else {
                cv::Vec3b& pixel = noise.at<cv::Vec3b>(y, x);
                for (int c = 0; c < 3; c++) {
                    double noise_value = distribution(gen);
                    pixel[c] = cv::saturate_cast<uchar>(noise_value);
                }
            }
        }
    }

    // 添加噪点到源图像
    cv::addWeighted(src, 1.0, noise, 1.0, 0.0, dst);

    // 应用中值滤波来模拟老照片的颗粒感
    cv::medianBlur(dst, dst, 3);
}

void add_vignette(const cv::Mat& src, cv::Mat& dst, double strength) {
    CV_Assert(!src.empty());
    CV_Assert(strength >= 0.0 && strength <= 1.0);

    // 创建输出图像
    dst = src.clone();

    // 计算图像中心点
    cv::Point center(src.cols / 2, src.rows / 2);

    // 计算最大半径（从中心到图像角落的距离）
    double max_radius = std::sqrt(center.x * center.x + center.y * center.y);

    // 创建暗角遮罩
    cv::Mat mask = cv::Mat::zeros(src.size(), CV_32F);

    // 生成径向渐变暗角
    #pragma omp parallel for
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            // 计算到中心的距离
            double dx = x - center.x;
            double dy = y - center.y;
            double distance = std::sqrt(dx * dx + dy * dy);

            // 归一化距离
            double normalized_distance = distance / max_radius;

            // 计算暗角强度
            double vignette_factor = 1.0 - strength * std::pow(normalized_distance, 2);

            // 存储暗角因子
            mask.at<float>(y, x) = static_cast<float>(vignette_factor);
        }
    }

    // 应用暗角到每个通道
    if (src.channels() == 1) {
        cv::multiply(src, mask, dst, 1.0, src.type());
    } else {
        std::vector<cv::Mat> channels;
        cv::split(dst, channels);

        for (auto& channel : channels) {
            cv::multiply(channel, mask, channel, 1.0, channel.type());
        }

        cv::merge(channels, dst);
    }
}

void add_scratches(const cv::Mat& src, cv::Mat& dst, double count, double intensity) {
    CV_Assert(!src.empty());
    CV_Assert(count >= 0.0 && intensity >= 0.0 && intensity <= 1.0);

    // 创建输出图像
    dst = src.clone();

    // 创建随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist_x(0, src.cols - 1);
    std::uniform_int_distribution<int> dist_y(0, src.rows - 1);
    std::uniform_int_distribution<int> dist_length(src.rows / 10, src.rows / 3);
    std::uniform_int_distribution<int> dist_width(1, 2);
    std::uniform_real_distribution<float> dist_angle(0.0f, static_cast<float>(CV_PI));

    // 将划痕数量取整
    int scratch_count = static_cast<int>(count);

    // 随机生成划痕
    for (int i = 0; i < scratch_count; i++) {
        // 随机划痕起点
        cv::Point pt1(dist_x(gen), dist_y(gen));

        // 随机划痕长度和角度
        int length = dist_length(gen);
        float angle = dist_angle(gen);
        int width = dist_width(gen);

        // 计算划痕终点
        cv::Point pt2;
        pt2.x = pt1.x + static_cast<int>(length * std::cos(angle));
        pt2.y = pt1.y + static_cast<int>(length * std::sin(angle));

        // 设置划痕颜色和透明度
        cv::Scalar color;
        if (gen() % 2 == 0) {
            // 白色划痕
            color = cv::Scalar(255, 255, 255);
        } else {
            // 黑色划痕
            color = cv::Scalar(0, 0, 0);
        }

        // 绘制划痕
        cv::line(dst, pt1, pt2, color, width, cv::LINE_AA);
    }

    // 根据强度参数混合原图和带划痕的图像
    cv::addWeighted(src, 1.0 - intensity, dst, intensity, 0.0, dst);
}

void add_vintage_border(const cv::Mat& src, cv::Mat& dst) {
    CV_Assert(!src.empty());

    // 确定边框宽度
    int border_width = std::min(src.rows, src.cols) / 20;

    // 创建带边框的大图像
    cv::Mat bordered;
    cv::copyMakeBorder(src, bordered, border_width, border_width, border_width, border_width,
                      cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

    // 创建一个稍小一点的内边框
    int inner_width = border_width / 2;
    cv::Mat inner_border = bordered.clone();
    cv::rectangle(inner_border,
                 cv::Point(border_width - inner_width, border_width - inner_width),
                 cv::Point(bordered.cols - border_width + inner_width, bordered.rows - border_width + inner_width),
                 cv::Scalar(0, 0, 0), inner_width);

    // 添加角部装饰
    int corner_size = border_width * 2;

    // 左上角
    cv::circle(inner_border, cv::Point(border_width, border_width), corner_size / 3, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

    // 右上角
    cv::circle(inner_border, cv::Point(bordered.cols - border_width, border_width), corner_size / 3, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

    // 左下角
    cv::circle(inner_border, cv::Point(border_width, bordered.rows - border_width), corner_size / 3, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

    // 右下角
    cv::circle(inner_border, cv::Point(bordered.cols - border_width, bordered.rows - border_width), corner_size / 3, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);

    dst = inner_border;
}

void vintage_effect(const cv::Mat& src, cv::Mat& dst, const VintageParams& params) {
    CV_Assert(!src.empty());

    // 创建一个临时图像用于处理
    cv::Mat temp = src.clone();

    // 1. 应用褐色调特效
    cv::Mat sepia;
    apply_sepia_tone(temp, sepia, params.sepia_intensity);

    // 2. 添加暗角效果
    cv::Mat vignette;
    add_vignette(sepia, vignette, params.vignette_strength);

    // 3. 添加老照片噪点
    cv::Mat grainy;
    add_film_grain(vignette, grainy, params.noise_level);

    // 4. 添加划痕效果
    cv::Mat scratched;
    add_scratches(grainy, scratched, params.scratch_count, params.scratch_intensity);

    // 5. 可选择性地添加边框
    if (params.add_border) {
        add_vintage_border(scratched, dst);
    } else {
        scratched.copyTo(dst);
    }
}

} // namespace advanced
} // namespace ip101