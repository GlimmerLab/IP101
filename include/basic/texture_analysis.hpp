#ifndef TEXTURE_ANALYSIS_HPP
#define TEXTURE_ANALYSIS_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace ip101 {

/**
 * @brief Calculate Gray Level Co-occurrence Matrix (GLCM)
 * @param src Input grayscale image
 * @param distance Distance between pixel pairs
 * @param angle Angle (0, 45, 90, 135)
 * @return Gray level co-occurrence matrix
 */
cv::Mat compute_glcm(
    const cv::Mat& src,
    int distance = 1,
    int angle = 0);

/**
 * @brief Extract Haralick texture features from GLCM
 * @param glcm Gray level co-occurrence matrix
 * @return Texture feature vector (contrast, correlation, energy, homogeneity, etc.)
 */
std::vector<double> extract_haralick_features(
    const cv::Mat& glcm);

/**
 * @brief Calculate texture statistical features
 * @param src Input grayscale image
 * @param window_size Local window size
 * @return Feature images (mean, variance, skewness, kurtosis)
 */
std::vector<cv::Mat> compute_statistical_features(
    const cv::Mat& src,
    int window_size = 3);

/**
 * @brief Calculate Local Binary Pattern (LBP) features
 * @param src Input grayscale image
 * @param radius LBP operator radius
 * @param neighbors Number of sampling points in the neighborhood
 * @return LBP feature image
 */
cv::Mat compute_lbp(
    const cv::Mat& src,
    int radius = 1,
    int neighbors = 8);

/**
 * @brief Generate Gabor filter bank
 * @param ksize Filter size
 * @param sigma Standard deviation of the Gaussian envelope
 * @param theta Number of orientation angles
 * @param lambda Wavelength of the sinusoidal wave
 * @param gamma Spatial aspect ratio
 * @param psi Phase offset
 * @return Gabor filter bank
 */
std::vector<cv::Mat> generate_gabor_filters(
    int ksize = 31,
    double sigma = 3.0,
    int theta = 8,
    double lambda = 4.0,
    double gamma = 0.5,
    double psi = 0.0);

/**
 * @brief Extract Gabor texture features
 * @param src Input grayscale image
 * @param filters Gabor filter bank
 * @return Gabor feature image set
 */
std::vector<cv::Mat> extract_gabor_features(
    const cv::Mat& src,
    const std::vector<cv::Mat>& filters);

/**
 * @brief Texture classification
 * @param features Texture feature vector
 * @param labels Training sample labels
 * @param train_features Training sample features
 * @return Classification result label
 */
int classify_texture(
    const std::vector<double>& features,
    const std::vector<int>& labels,
    const std::vector<std::vector<double>>& train_features);

} // namespace ip101

#endif // TEXTURE_ANALYSIS_HPP