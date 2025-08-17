#include <advanced/filtering/side_window_filter.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace ip101 {
namespace advanced {

void compute_directional_gradients(const cv::Mat& src, std::vector<cv::Mat>& gradients) {
    CV_Assert(!src.empty());

    // 转换为灰度图像
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // 确保图像为浮点型
    cv::Mat gray_float;
    gray.convertTo(gray_float, CV_32F);

    // 定义8个方向的Sobel核
    const int kNumDirections = 8;
    gradients.resize(kNumDirections);

    // N方向 (上)
    cv::Mat kernel_n = (cv::Mat_<float>(3, 3) <<
                       1, 2, 1,
                       0, 0, 0,
                       -1, -2, -1);
    // NE方向 (右上)
    cv::Mat kernel_ne = (cv::Mat_<float>(3, 3) <<
                        0, 1, 2,
                        -1, 0, 1,
                        -2, -1, 0);
    // E方向 (右)
    cv::Mat kernel_e = (cv::Mat_<float>(3, 3) <<
                       -1, 0, 1,
                       -2, 0, 2,
                       -1, 0, 1);
    // SE方向 (右下)
    cv::Mat kernel_se = (cv::Mat_<float>(3, 3) <<
                        -2, -1, 0,
                        -1, 0, 1,
                        0, 1, 2);
    // S方向 (下)
    cv::Mat kernel_s = (cv::Mat_<float>(3, 3) <<
                       -1, -2, -1,
                       0, 0, 0,
                       1, 2, 1);
    // SW方向 (左下)
    cv::Mat kernel_sw = (cv::Mat_<float>(3, 3) <<
                        0, -1, -2,
                        1, 0, -1,
                        2, 1, 0);
    // W方向 (左)
    cv::Mat kernel_w = (cv::Mat_<float>(3, 3) <<
                       1, 0, -1,
                       2, 0, -2,
                       1, 0, -1);
    // NW方向 (左上)
    cv::Mat kernel_nw = (cv::Mat_<float>(3, 3) <<
                        2, 1, 0,
                        1, 0, -1,
                        0, -1, -2);

    // 计算各方向的梯度
    cv::filter2D(gray_float, gradients[static_cast<int>(Direction::N)], CV_32F, kernel_n);
    cv::filter2D(gray_float, gradients[static_cast<int>(Direction::NE)], CV_32F, kernel_ne);
    cv::filter2D(gray_float, gradients[static_cast<int>(Direction::E)], CV_32F, kernel_e);
    cv::filter2D(gray_float, gradients[static_cast<int>(Direction::SE)], CV_32F, kernel_se);
    cv::filter2D(gray_float, gradients[static_cast<int>(Direction::S)], CV_32F, kernel_s);
    cv::filter2D(gray_float, gradients[static_cast<int>(Direction::SW)], CV_32F, kernel_sw);
    cv::filter2D(gray_float, gradients[static_cast<int>(Direction::W)], CV_32F, kernel_w);
    cv::filter2D(gray_float, gradients[static_cast<int>(Direction::NW)], CV_32F, kernel_nw);

    // 取绝对值
    for (auto& grad : gradients) {
        grad = cv::abs(grad);
    }
}

void determine_optimal_window(const std::vector<cv::Mat>& gradients, cv::Mat& optimal_dir) {
    CV_Assert(!gradients.empty());

    const int kNumDirections = 8;
    int rows = gradients[0].rows;
    int cols = gradients[0].cols;

    // 创建最优方向图像
    optimal_dir.create(rows, cols, CV_8UC1);

    // 查找每个像素最小梯度对应的方向
    #pragma omp parallel for
    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            float min_grad = std::numeric_limits<float>::max();
            uint8_t best_dir = 0;

            // 遍历所有方向
            for (int dir = 0; dir < kNumDirections; dir++) {
                float grad = gradients[dir].at<float>(y, x);
                if (grad < min_grad) {
                    min_grad = grad;
                    best_dir = static_cast<uint8_t>(dir);
                }
            }

            // 保存最优方向
            optimal_dir.at<uint8_t>(y, x) = best_dir;
        }
    }
}

// 获取特定方向的窗口掩码
void get_directional_mask(Direction dir, int window_size, cv::Mat& mask) {
    CV_Assert(window_size > 0 && window_size % 2 == 1);

    int radius = window_size / 2;
    mask = cv::Mat::zeros(window_size, window_size, CV_8UC1);

    // 根据方向设置相应的掩码
    switch (dir) {
        case Direction::N: // 北方向 (上)
            for (int i = 0; i <= radius; i++) {
                for (int j = -radius; j <= radius; j++) {
                    int y = radius - i;
                    int x = radius + j;
                    if (y >= 0 && y < window_size && x >= 0 && x < window_size) {
                        mask.at<uint8_t>(y, x) = 1;
                    }
                }
            }
            break;

        case Direction::NE: // 东北方向 (右上)
            for (int i = -radius; i <= radius; i++) {
                for (int j = 0; j <= radius; j++) {
                    if (i + j <= 0) {
                        int y = radius + i;
                        int x = radius + j;
                        if (y >= 0 && y < window_size && x >= 0 && x < window_size) {
                            mask.at<uint8_t>(y, x) = 1;
                        }
                    }
                }
            }
            break;

        case Direction::E: // 东方向 (右)
            for (int i = -radius; i <= radius; i++) {
                for (int j = 0; j <= radius; j++) {
                    int y = radius + i;
                    int x = radius + j;
                    if (y >= 0 && y < window_size && x >= 0 && x < window_size) {
                        mask.at<uint8_t>(y, x) = 1;
                    }
                }
            }
            break;

        case Direction::SE: // 东南方向 (右下)
            for (int i = 0; i <= radius; i++) {
                for (int j = 0; j <= radius; j++) {
                    if (i - j >= 0) {
                        int y = radius + i;
                        int x = radius + j;
                        if (y >= 0 && y < window_size && x >= 0 && x < window_size) {
                            mask.at<uint8_t>(y, x) = 1;
                        }
                    }
                }
            }
            break;

        case Direction::S: // 南方向 (下)
            for (int i = 0; i <= radius; i++) {
                for (int j = -radius; j <= radius; j++) {
                    int y = radius + i;
                    int x = radius + j;
                    if (y >= 0 && y < window_size && x >= 0 && x < window_size) {
                        mask.at<uint8_t>(y, x) = 1;
                    }
                }
            }
            break;

        case Direction::SW: // 西南方向 (左下)
            for (int i = 0; i <= radius; i++) {
                for (int j = -radius; j <= 0; j++) {
                    if (i + j >= 0) {
                        int y = radius + i;
                        int x = radius + j;
                        if (y >= 0 && y < window_size && x >= 0 && x < window_size) {
                            mask.at<uint8_t>(y, x) = 1;
                        }
                    }
                }
            }
            break;

        case Direction::W: // 西方向 (左)
            for (int i = -radius; i <= radius; i++) {
                for (int j = -radius; j <= 0; j++) {
                    int y = radius + i;
                    int x = radius + j;
                    if (y >= 0 && y < window_size && x >= 0 && x < window_size) {
                        mask.at<uint8_t>(y, x) = 1;
                    }
                }
            }
            break;

        case Direction::NW: // 西北方向 (左上)
            for (int i = -radius; i <= 0; i++) {
                for (int j = -radius; j <= 0; j++) {
                    if (i - j <= 0) {
                        int y = radius + i;
                        int x = radius + j;
                        if (y >= 0 && y < window_size && x >= 0 && x < window_size) {
                            mask.at<uint8_t>(y, x) = 1;
                        }
                    }
                }
            }
            break;
    }
}

void apply_optimal_window_filter(const cv::Mat& src, cv::Mat& dst, const cv::Mat& optimal_dir,
                               int window_size, SideWindowType filter_type) {
    CV_Assert(!src.empty() && !optimal_dir.empty());
    CV_Assert(src.size() == optimal_dir.size());
    CV_Assert(window_size > 0 && window_size % 2 == 1);

    int channels = src.channels();
    int radius = window_size / 2;

    // 为每个方向预计算掩码
    std::vector<cv::Mat> directional_masks(8);
    for (int dir = 0; dir < 8; dir++) {
        get_directional_mask(static_cast<Direction>(dir), window_size, directional_masks[dir]);
    }

    // 创建输出图像
    dst.create(src.size(), src.type());

    // 为边界添加镜像边界
    cv::Mat padded;
    cv::copyMakeBorder(src, padded, radius, radius, radius, radius, cv::BORDER_REFLECT);

    // 应用滤波
    #pragma omp parallel for
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            // 获取最优方向
            Direction best_dir = static_cast<Direction>(optimal_dir.at<uint8_t>(y, x));
            cv::Mat& mask = directional_masks[static_cast<int>(best_dir)];

            // 收集窗口内对应方向的像素
            std::vector<std::vector<uint8_t>> window_values(channels);

            for (int i = -radius; i <= radius; i++) {
                for (int j = -radius; j <= radius; j++) {
                    int mask_y = i + radius;
                    int mask_x = j + radius;

                    // 检查是否在掩码内
                    if (mask.at<uint8_t>(mask_y, mask_x) > 0) {
                        // 添加像素值到窗口
                        int py = y + radius + i;
                        int px = x + radius + j;

                        if (channels == 1) {
                            window_values[0].push_back(padded.at<uint8_t>(py, px));
                        } else { // channels == 3
                            cv::Vec3b pixel = padded.at<cv::Vec3b>(py, px);
                            window_values[0].push_back(pixel[0]);
                            window_values[1].push_back(pixel[1]);
                            window_values[2].push_back(pixel[2]);
                        }
                    }
                }
            }

            // 根据滤波类型计算结果
            if (filter_type == SideWindowType::BOX) {
                // Box滤波 (均值)
                if (channels == 1) {
                    int sum = 0;
                    for (auto val : window_values[0]) {
                        sum += val;
                    }
                    dst.at<uint8_t>(y, x) = static_cast<uint8_t>(sum / window_values[0].size());
                } else { // channels == 3
                    cv::Vec3i sum(0, 0, 0);
                    for (int c = 0; c < channels; c++) {
                        for (auto val : window_values[c]) {
                            sum[c] += val;
                        }
                        sum[c] /= window_values[c].size();
                    }
                    dst.at<cv::Vec3b>(y, x) = cv::Vec3b(
                        static_cast<uint8_t>(sum[0]),
                        static_cast<uint8_t>(sum[1]),
                        static_cast<uint8_t>(sum[2])
                    );
                }
            } else { // SideWindowType::MEDIAN
                // Median滤波 (中值)
                if (channels == 1) {
                    std::sort(window_values[0].begin(), window_values[0].end());
                    dst.at<uint8_t>(y, x) = window_values[0][window_values[0].size() / 2];
                } else { // channels == 3
                    cv::Vec3b median_val;
                    for (int c = 0; c < channels; c++) {
                        std::sort(window_values[c].begin(), window_values[c].end());
                        median_val[c] = window_values[c][window_values[c].size() / 2];
                    }
                    dst.at<cv::Vec3b>(y, x) = median_val;
                }
            }
        }
    }
}

void side_window_filter(const cv::Mat& src, cv::Mat& dst, int window_size, SideWindowType filter_type) {
    CV_Assert(!src.empty());
    CV_Assert(window_size > 0 && window_size % 2 == 1);

    // 计算方向梯度
    std::vector<cv::Mat> gradients;
    compute_directional_gradients(src, gradients);

    // 确定每个像素的最优窗口方向
    cv::Mat optimal_dir;
    determine_optimal_window(gradients, optimal_dir);

    // 应用最优窗口滤波
    apply_optimal_window_filter(src, dst, optimal_dir, window_size, filter_type);
}

void box_side_window_filter(const cv::Mat& src, cv::Mat& dst, int window_size) {
    side_window_filter(src, dst, window_size, SideWindowType::BOX);
}

void median_side_window_filter(const cv::Mat& src, cv::Mat& dst, int window_size) {
    side_window_filter(src, dst, window_size, SideWindowType::MEDIAN);
}

} // namespace advanced
} // namespace ip101