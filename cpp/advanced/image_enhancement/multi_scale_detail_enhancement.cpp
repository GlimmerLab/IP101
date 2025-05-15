#include "multi_scale_detail_enhancement.hpp"
#include <cmath>
#include <omp.h>

namespace ip101 {
namespace advanced {

void multi_scale_detail_enhancement(const cv::Mat& src, cv::Mat& dst,
                                  double sigma1, double sigma2,
                                  double alpha, double beta) {
    CV_Assert(!src.empty());

    // 创建高斯金字塔
    std::vector<cv::Mat> pyramid;
    cv::Mat current = src.clone();
    pyramid.push_back(current);

    // 构建高斯金字塔
    while(current.rows > 32 && current.cols > 32) {
        cv::Mat blurred;
        cv::GaussianBlur(current, blurred, cv::Size(0, 0), sigma1);
        cv::resize(blurred, current, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
        pyramid.push_back(current);
    }

    // 从金字塔顶部开始重建
    cv::Mat result = pyramid.back();

    #pragma omp parallel for
    for(int i = pyramid.size() - 2; i >= 0; i--) {
        // 上采样
        cv::Mat upsampled;
        cv::resize(result, upsampled, pyramid[i].size(), 0, 0, cv::INTER_LINEAR);

        // 计算细节层
        cv::Mat detail;
        cv::GaussianBlur(pyramid[i], detail, cv::Size(0, 0), sigma2);
        detail = pyramid[i] - detail;

        // 增强细节
        #pragma omp critical
        {
            result = upsampled + alpha * detail;
        }
    }

    // 对比度增强
    result.convertTo(dst, CV_8UC3, beta);

    // 确保结果在有效范围内
    #pragma omp parallel for
    for(int i = 0; i < dst.rows; i++) {
        for(int j = 0; j < dst.cols; j++) {
            cv::Vec3b& pixel = dst.at<cv::Vec3b>(i, j);
            pixel[0] = cv::saturate_cast<uchar>(pixel[0]);
            pixel[1] = cv::saturate_cast<uchar>(pixel[1]);
            pixel[2] = cv::saturate_cast<uchar>(pixel[2]);
        }
    }
}

} // namespace advanced
} // namespace ip101