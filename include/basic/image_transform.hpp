#ifndef IMAGE_TRANSFORM_HPP
#define IMAGE_TRANSFORM_HPP

// UTF-8 with BOM marker
#include <opencv2/opencv.hpp>
#include <immintrin.h>  // For SIMD instructions (AVX2)
#include <omp.h>        // For OpenMP parallel computing

namespace ip101 {

// Constants definition
constexpr int CACHE_LINE = 64;  // Cache line size (bytes)
constexpr int BLOCK_SIZE = 16;  // Block processing size

//------------------------------------------------------------------------------
// Original version implementations (for performance comparison and compatibility)
//------------------------------------------------------------------------------

/**
 * @brief Original version of affine transformation
 * @param src Input image
 * @param src_points Three points in source image
 * @param dst_points Three points in target image
 * @return Transformed image
 */
cv::Mat affineTransform(
    const cv::Mat& src,
    const std::vector<cv::Point2f>& src_points,
    const std::vector<cv::Point2f>& dst_points);

/**
 * @brief SIMD optimized affine transformation
 * @param src Input image
 * @param src_points Three points in source image
 * @param dst_points Three points in target image
 * @return Transformed image
 */
cv::Mat affineTransform_optimized(
    const cv::Mat& src,
    const std::vector<cv::Point2f>& src_points,
    const std::vector<cv::Point2f>& dst_points);

/**
 * @brief Original version of perspective transformation
 * @param src Input image
 * @param src_points Four points in source image
 * @param dst_points Four points in target image
 * @return Transformed image
 */
cv::Mat perspectiveTransform(
    const cv::Mat& src,
    const std::vector<cv::Point2f>& src_points,
    const std::vector<cv::Point2f>& dst_points);

/**
 * @brief Original version of image rotation
 * @param src Input image
 * @param angle Rotation angle (degrees)
 * @param center Rotation center point
 * @return Rotated image
 */
cv::Mat rotateImage(
    const cv::Mat& src,
    double angle,
    cv::Point2f center = cv::Point2f(-1,-1));

/**
 * @brief Original version of image scaling
 * @param src Input image
 * @param scale_x X-direction scaling factor
 * @param scale_y Y-direction scaling factor
 * @return Scaled image
 */
cv::Mat scaleImage(
    const cv::Mat& src,
    double scale_x,
    double scale_y);

/**
 * @brief Original version of image translation
 * @param src Input image
 * @param tx X-direction translation amount
 * @param ty Y-direction translation amount
 * @return Translated image
 */
cv::Mat translateImage(
    const cv::Mat& src,
    int tx,
    int ty);

/**
 * @brief Original version of image mirroring
 * @param src Input image
 * @param direction Mirror direction (0: horizontal, 1: vertical)
 * @return Mirrored image
 */
cv::Mat mirrorImage(
    const cv::Mat& src,
    int direction = 0);

//------------------------------------------------------------------------------
// Modern API interfaces (recommended)
//------------------------------------------------------------------------------

/**
 * @brief Affine transformation
 * @param src Input image
 * @param M Transformation matrix
 * @param size Output image size
 * @return Transformed image
 */
cv::Mat affine_transform(
    const cv::Mat& src,
    const cv::Mat& M,
    const cv::Size& size);

/**
 * @brief Perspective transformation
 * @param src Input image
 * @param M Transformation matrix
 * @param size Output image size
 * @return Transformed image
 */
cv::Mat perspective_transform(
    const cv::Mat& src,
    const cv::Mat& M,
    const cv::Size& size);

/**
 * @brief Image rotation
 * @param src Input image
 * @param angle Rotation angle (degrees)
 * @param center Rotation center
 * @param scale Scaling factor
 * @return Rotated image
 */
cv::Mat rotate(
    const cv::Mat& src,
    double angle,
    const cv::Point2f& center = cv::Point2f(-1,-1),
    double scale = 1.0);

/**
 * @brief Image resizing
 * @param src Input image
 * @param size Target size
 * @param interpolation Interpolation method
 * @return Resized image
 */
cv::Mat resize(
    const cv::Mat& src,
    const cv::Size& size,
    int interpolation = cv::INTER_LINEAR);

/**
 * @brief Image translation
 * @param src Input image
 * @param dx Horizontal translation amount
 * @param dy Vertical translation amount
 * @return Translated image
 */
cv::Mat translate(
    const cv::Mat& src,
    double dx,
    double dy);

/**
 * @brief Image mirroring
 * @param src Input image
 * @param flip_code Flip mode (0: vertical, 1: horizontal, -1: both directions)
 * @return Mirrored image
 */
cv::Mat mirror(
    const cv::Mat& src,
    int flip_code);

//------------------------------------------------------------------------------
// Transformation matrix calculation functions
//------------------------------------------------------------------------------

/**
 * @brief Get affine transformation matrix
 * @param src_points Three points in source image
 * @param dst_points Three corresponding points in target image
 * @return 2x3 affine transformation matrix
 */
cv::Mat get_affine_transform(
    const std::vector<cv::Point2f>& src_points,
    const std::vector<cv::Point2f>& dst_points);

/**
 * @brief Get perspective transformation matrix
 * @param src_points Four points in source image
 * @param dst_points Four corresponding points in target image
 * @return 3x3 perspective transformation matrix
 */
cv::Mat get_perspective_transform(
    const std::vector<cv::Point2f>& src_points,
    const std::vector<cv::Point2f>& dst_points);

/**
 * @brief Get rotation matrix
 * @param center Rotation center
 * @param angle Rotation angle (degrees)
 * @param scale Scaling factor
 * @return 2x3 rotation transformation matrix
 */
cv::Mat get_rotation_matrix(
    const cv::Point2f& center,
    double angle,
    double scale = 1.0);

} // namespace ip101

#endif // IMAGE_TRANSFORM_HPP