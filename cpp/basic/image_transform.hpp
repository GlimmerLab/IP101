#ifndef IMAGE_TRANSFORM_HPP
#define IMAGE_TRANSFORM_HPP

#include <opencv2/opencv.hpp>
#include <immintrin.h>  // 用于SIMD指令(AVX2)
#include <omp.h>        // 用于OpenMP并行计算

namespace ip101 {

/**
 * @brief 仿射变换
 * @param src 输入图像
 * @param M 变换矩阵
 * @param size 输出图像大小
 * @return 变换后的图像
 */
cv::Mat affine_transform(
    const cv::Mat& src,
    const cv::Mat& M,
    const cv::Size& size);

/**
 * @brief 透视变换
 * @param src 输入图像
 * @param M 变换矩阵
 * @param size 输出图像大小
 * @return 变换后的图像
 */
cv::Mat perspective_transform(
    const cv::Mat& src,
    const cv::Mat& M,
    const cv::Size& size);

/**
 * @brief 图像旋转
 * @param src 输入图像
 * @param angle 旋转角度(度)
 * @param center 旋转中心
 * @param scale 缩放比例
 * @return 旋转后的图像
 */
cv::Mat rotate(
    const cv::Mat& src,
    double angle,
    const cv::Point2f& center = cv::Point2f(-1,-1),
    double scale = 1.0);

/**
 * @brief 图像缩放
 * @param src 输入图像
 * @param size 目标大小
 * @param interpolation 插值方法
 * @return 缩放后的图像
 */
cv::Mat resize(
    const cv::Mat& src,
    const cv::Size& size,
    int interpolation = cv::INTER_LINEAR);

/**
 * @brief 图像平移
 * @param src 输入图像
 * @param dx 水平平移量
 * @param dy 垂直平移量
 * @return 平移后的图像
 */
cv::Mat translate(
    const cv::Mat& src,
    double dx,
    double dy);

/**
 * @brief 图像镜像
 * @param src 输入图像
 * @param flip_code 翻转方式(0:垂直, 1:水平, -1:双向)
 * @return 镜像后的图像
 */
cv::Mat mirror(
    const cv::Mat& src,
    int flip_code);

/**
 * @brief 获取仿射变换矩阵
 * @param src_points 源图像中的三个点
 * @param dst_points 目标图像中的对应三个点
 * @return 2x3仿射变换矩阵
 */
cv::Mat get_affine_transform(const std::vector<cv::Point2f>& src_points,
                           const std::vector<cv::Point2f>& dst_points);

/**
 * @brief 获取透视变换矩阵
 * @param src_points 源图像中的四个点
 * @param dst_points 目标图像中的对应四个点
 * @return 3x3透视变换矩阵
 */
cv::Mat get_perspective_transform(const std::vector<cv::Point2f>& src_points,
                                const std::vector<cv::Point2f>& dst_points);

/**
 * @brief 获取旋转变换矩阵
 * @param center 旋转中心点
 * @param angle 旋转角度(度)
 * @param scale 缩放因子
 * @return 2x3旋转变换矩阵
 */
cv::Mat get_rotation_matrix(const cv::Point2f& center,
                          double angle,
                          double scale = 1.0);

} // namespace ip101

#endif // IMAGE_TRANSFORM_HPP