#include <basic/feature_extraction.hpp>
#include <cmath>
#include <vector>
#include <omp.h>

// Check if SURF is available
// Starting from OpenCV 4.5.1, SURF is moved to opencv_contrib
// and needs to be explicitly enabled with OPENCV_ENABLE_NONFREE
#if CV_VERSION_MAJOR < 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR < 5)
#define HAVE_SURF 1
#else
#ifdef OPENCV_ENABLE_NONFREE
#define HAVE_SURF 1
#else
#define HAVE_SURF 0
#endif
#endif

// Include SURF if available
#if HAVE_SURF
#include <opencv2/xfeatures2d.hpp>
#endif

namespace ip101 {

using namespace cv;

namespace {
// Internal constants
constexpr int CACHE_LINE = 64;    // CPU cache line size (bytes)
constexpr int SIMD_WIDTH = 32;    // AVX2 SIMD vector width (bytes)
constexpr int BLOCK_SIZE = 16;    // Block processing size

// Memory alignment helper function
inline uchar* alignPtr(uchar* ptr, size_t align = CACHE_LINE) {
    return (uchar*)(((size_t)ptr + align - 1) & ~(align - 1));
}

// Gaussian kernel generation
void createGaussianKernel(Mat& kernel, int ksize, double sigma) {
    kernel.create(ksize, ksize, CV_64F);
    double sum = 0;
    int center = ksize / 2;

    for (int y = 0; y < ksize; y++) {
        for (int x = 0; x < ksize; x++) {
            double x2 = (x - center) * (x - center);
            double y2 = (y - center) * (y - center);
            double value = exp(-(x2 + y2) / (2 * sigma * sigma));
            kernel.at<double>(y, x) = value;
            sum += value;
        }
    }

    // Normalize
    kernel /= sum;
}

} // anonymous namespace

void compute_harris_manual(const Mat& src, Mat& dst,
                          double k, int window_size,
                          double threshold) {
    CV_Assert(!src.empty() && src.type() == CV_8UC1);

    // Calculate image gradients
    Mat Ix, Iy;
    Sobel(src, Ix, CV_64F, 1, 0, 3);
    Sobel(src, Iy, CV_64F, 0, 1, 3);

    // Calculate gradient products
    Mat Ixx, Ixy, Iyy;
    Ixx = Ix.mul(Ix);
    Ixy = Ix.mul(Iy);
    Iyy = Iy.mul(Iy);

    // Create Gaussian kernel
    Mat gaussian_kernel;
    createGaussianKernel(gaussian_kernel, window_size, 1.0);

    // Apply Gaussian filter to gradient products
    Mat Sxx, Sxy, Syy;
    filter2D(Ixx, Sxx, -1, gaussian_kernel);
    filter2D(Ixy, Sxy, -1, gaussian_kernel);
    filter2D(Iyy, Syy, -1, gaussian_kernel);

    // Calculate Harris response
    Mat det = Sxx.mul(Syy) - Sxy.mul(Sxy);
    Mat trace = Sxx + Syy;
    Mat harris_response = det - k * trace.mul(trace);

    // Threshold processing
    double max_val;
    minMaxLoc(harris_response, nullptr, &max_val);
    threshold *= max_val;

    // Create output image
    dst = Mat::zeros(src.size(), CV_8UC1);
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            if (harris_response.at<double>(y, x) > threshold) {
                dst.at<uchar>(y, x) = 255;
            }
        }
    }
}

void harris_corner_detection(const Mat& src, Mat& dst,
                           int block_size, int ksize,
                           double k, double threshold) {
    CV_Assert(!src.empty());

    // Convert to grayscale
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // Use OpenCV's Harris corner detection
    Mat corners;
    cornerHarris(gray, corners, block_size, ksize, k);

    // Threshold processing
    Mat corners_norm;
    normalize(corners, corners_norm, 0, 255, NORM_MINMAX, CV_8UC1);

    // Mark corners on the original image
    dst = src.clone();
    for (int y = 0; y < corners_norm.rows; y++) {
        for (int x = 0; x < corners_norm.cols; x++) {
            if (corners_norm.at<uchar>(y, x) > threshold) {
                circle(dst, Point(x, y), 5, Scalar(0, 0, 255), 2);
            }
        }
    }
}

void sift_features(const Mat& src, Mat& dst, int nfeatures) {
    CV_Assert(!src.empty());

    // Convert to grayscale
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // Create SIFT object with additional parameters
    Ptr<SIFT> sift = SIFT::create(
        nfeatures,           // Number of features
        4,                   // Number of pyramid layers
        0.04,               // Contrast threshold
        10,                 // Edge response threshold
        1.6                 // Sigma value
    );

    // Use OpenMP parallel computation
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            // Detect keypoints and compute descriptors
            std::vector<KeyPoint> keypoints;
            Mat descriptors;
            sift->detectAndCompute(gray, Mat(), keypoints, descriptors);

            // Draw keypoints on the original image
            drawKeypoints(src, keypoints, dst, Scalar(0, 255, 0),
                         DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        }
    }
}

void surf_features(const Mat& src, Mat& dst, double hessian_threshold) {
    CV_Assert(!src.empty());

    // Convert to grayscale
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

#if HAVE_SURF
    // Create SURF object with additional parameters
    Ptr<xfeatures2d::SURF> surf = xfeatures2d::SURF::create(
        hessian_threshold,    // Hessian threshold
        4,                    // Number of pyramid layers
        2,                    // Descriptor dimensions
        true,                 // Use U-SURF
        false                 // Use extended descriptor
    );

    // Use OpenMP parallel computation
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            // Detect keypoints and compute descriptors
            std::vector<KeyPoint> keypoints;
            Mat descriptors;
            surf->detectAndCompute(gray, Mat(), keypoints, descriptors);

            // Draw keypoints on the original image
            drawKeypoints(src, keypoints, dst, Scalar(0, 255, 0),
                         DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        }
    }
#else
    // SURF not available, use SIFT instead and warn
    std::cout << "Warning: SURF is not available in this OpenCV build. Using SIFT instead." << std::endl;
    sift_features(src, dst, 500);
#endif
}

void orb_features(const Mat& src, Mat& dst, int nfeatures) {
    CV_Assert(!src.empty());

    // Convert to grayscale
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // Create ORB object with additional parameters
    Ptr<ORB> orb = ORB::create(
        nfeatures,           // Number of features
        1.2f,               // Scale factor
        8,                  // Number of pyramid layers
        31,                 // Edge threshold
        0,                  // First level pyramid scale
        2,                  // WTA_K
        ORB::HARRIS_SCORE,  // Score type
        31,                 // Patch size
        20                  // Fast threshold
    );

    // Use OpenMP parallel computation
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            // Detect keypoints and compute descriptors
            std::vector<KeyPoint> keypoints;
            Mat descriptors;
            orb->detectAndCompute(gray, Mat(), keypoints, descriptors);

            // Draw keypoints on the original image
            drawKeypoints(src, keypoints, dst, Scalar(0, 255, 0),
                         DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        }
    }
}

void feature_matching(const Mat& src1, const Mat& src2,
                     Mat& dst, const std::string& method) {
    CV_Assert(!src1.empty() && !src2.empty());

    // Convert to grayscale
    Mat gray1, gray2;
    if (src1.channels() == 3) {
        cvtColor(src1, gray1, COLOR_BGR2GRAY);
    } else {
        gray1 = src1.clone();
    }
    if (src2.channels() == 3) {
        cvtColor(src2, gray2, COLOR_BGR2GRAY);
    } else {
        gray2 = src2.clone();
    }

    // Create feature detector
    Ptr<Feature2D> detector;
    if (method == "sift") {
        detector = SIFT::create(0, 4, 0.04, 10, 1.6);
    }
#if HAVE_SURF
    else if (method == "surf") {
        detector = xfeatures2d::SURF::create(100, 4, 2, true, false);
    }
#endif
    else if (method == "orb") {
        detector = ORB::create(500, 1.2f, 8, 31, 0, 2, ORB::HARRIS_SCORE, 31, 20);
    } else {
        throw std::invalid_argument("Unsupported feature detection method: " + method);
    }

    // Use OpenMP parallel computation
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            detector->detectAndCompute(gray1, Mat(), keypoints1, descriptors1);
        }
        #pragma omp section
        {
            detector->detectAndCompute(gray2, Mat(), keypoints2, descriptors2);
        }
    }

    // Create feature matcher
    Ptr<DescriptorMatcher> matcher;
    if (method == "sift" || method == "surf") {
        matcher = BFMatcher::create(NORM_L2, true);  // With cross-check
    } else {
        matcher = BFMatcher::create(NORM_HAMMING, true);
    }

    // Perform feature matching
    std::vector<DMatch> matches;
    matcher->match(descriptors1, descriptors2, matches);

    // Calculate distances between matching points
    std::vector<double> distances;
    for (const auto& match : matches) {
        distances.push_back(match.distance);
    }

    // Calculate mean and standard deviation of distances
    double mean = 0.0, stddev = 0.0;
    for (double d : distances) {
        mean += d;
    }
    mean /= distances.size();
    for (double d : distances) {
        stddev += (d - mean) * (d - mean);
    }
    stddev = std::sqrt(stddev / distances.size());

    // Filter good matches
    std::vector<DMatch> good_matches;
    for (const auto& match : matches) {
        if (match.distance < mean - stddev) {
            good_matches.push_back(match);
        }
    }

    // Draw matching results
    drawMatches(src1, keypoints1, src2, keypoints2, good_matches, dst,
               Scalar::all(-1), Scalar::all(-1),
               std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
}

} // namespace ip101