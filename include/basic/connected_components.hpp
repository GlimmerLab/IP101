#ifndef CONNECTED_COMPONENTS_HPP
#define CONNECTED_COMPONENTS_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace ip101 {

/**
 * @brief Connected component structure
 */
struct ConnectedComponent {
    int label;              // Label value
    int area;              // Area
    cv::Point centroid;    // Centroid
    cv::Rect bbox;         // Bounding box
    double circularity;    // Circularity
    double aspect_ratio;   // Aspect ratio
    double solidity;       // Solidity
};

/**
 * @brief 4-connected component labeling
 * @param src Input binary image
 * @param labels Output labeled image
 * @return Number of connected components
 */
int label_4connected(const cv::Mat& src, cv::Mat& labels);

/**
 * @brief 8-connected component labeling
 * @param src Input binary image
 * @param labels Output labeled image
 * @return Number of connected components
 */
int label_8connected(const cv::Mat& src, cv::Mat& labels);

/**
 * @brief Component statistics
 * @param labels Labeled image
 * @param num_labels Number of components
 * @return List of component properties
 */
std::vector<ConnectedComponent> analyze_components(const cv::Mat& labels, int num_labels);

/**
 * @brief Component filtering
 * @param labels Labeled image
 * @param stats Component properties
 * @param min_area Minimum area
 * @param max_area Maximum area
 * @return Filtered labeled image
 */
cv::Mat filter_components(const cv::Mat& labels,
                         const std::vector<ConnectedComponent>& stats,
                         int min_area = 100,
                         int max_area = 10000);

/**
 * @brief Component property visualization
 * @param src Original image
 * @param labels Labeled image
 * @param stats Component properties
 * @return Colored image with labels
 */
cv::Mat draw_components(const cv::Mat& src,
                       const cv::Mat& labels,
                       const std::vector<ConnectedComponent>& stats);

} // namespace ip101

#endif // CONNECTED_COMPONENTS_HPP