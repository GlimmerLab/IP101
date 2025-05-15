#include <basic/texture_analysis.hpp>
#include <omp.h>
#include <cmath>

namespace ip101 {

using namespace cv;
using namespace std;

namespace {
// Internal constants
constexpr int CACHE_LINE = 64;    // CPU cache line size (bytes)
constexpr int BLOCK_SIZE = 16;    // Block processing size
constexpr int GRAY_LEVELS = 256;  // Gray levels

// Calculate statistics
double compute_mean(const Mat& window) {
    Scalar mean = cv::mean(window);
    return mean[0];
}

double compute_variance(const Mat& window, double mean) {
    double variance = 0;
    #pragma omp parallel for reduction(+:variance)
    for (int i = 0; i < window.rows; i++) {
        for (int j = 0; j < window.cols; j++) {
            double diff = static_cast<double>(window.at<uchar>(i,j)) - mean;
            variance += diff * diff;
        }
    }
    return variance / (window.rows * window.cols);
}

double compute_skewness(const Mat& window, double mean, double std_dev) {
    double skewness = 0;
    #pragma omp parallel for reduction(+:skewness)
    for (int i = 0; i < window.rows; i++) {
        for (int j = 0; j < window.cols; j++) {
            double diff = (static_cast<double>(window.at<uchar>(i,j)) - mean) / std_dev;
            skewness += diff * diff * diff;
        }
    }
    return skewness / (window.rows * window.cols);
}

double compute_kurtosis(const Mat& window, double mean, double std_dev) {
    double kurtosis = 0;
    #pragma omp parallel for reduction(+:kurtosis)
    for (int i = 0; i < window.rows; i++) {
        for (int j = 0; j < window.cols; j++) {
            double diff = (static_cast<double>(window.at<uchar>(i,j)) - mean) / std_dev;
            kurtosis += diff * diff * diff * diff;
        }
    }
    return kurtosis / (window.rows * window.cols) - 3.0;
}

} // anonymous namespace

Mat compute_glcm(const Mat& src, int distance, int angle) {
    Mat glcm = Mat::zeros(GRAY_LEVELS, GRAY_LEVELS, CV_32F);

    // Calculate offsets
    int dx = 0, dy = 0;
    switch(angle) {
        case 0:   dx = distance; dy = 0;  break;
        case 45:  dx = distance; dy = -distance; break;
        case 90:  dx = 0; dy = -distance; break;
        case 135: dx = -distance; dy = -distance; break;
        default:  dx = distance; dy = 0;  break;
    }

    // Calculate GLCM
    #pragma omp parallel for
    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {
            int ni = i + dy;
            int nj = j + dx;
            if(ni >= 0 && ni < src.rows && nj >= 0 && nj < src.cols) {
                int val1 = src.at<uchar>(i,j);
                int val2 = src.at<uchar>(ni,nj);
                #pragma omp atomic
                glcm.at<float>(val1,val2)++;
            }
        }
    }

    // Normalize
    glcm /= sum(glcm)[0];

    return glcm;
}

vector<double> extract_haralick_features(const Mat& glcm) {
    vector<double> features;
    features.reserve(4);  // 4 Haralick features

    double contrast = 0, correlation = 0, energy = 0, homogeneity = 0;
    double mean_i = 0, mean_j = 0, std_i = 0, std_j = 0;

    // Calculate mean and standard deviation
    for(int i = 0; i < GRAY_LEVELS; i++) {
        for(int j = 0; j < GRAY_LEVELS; j++) {
            double p_ij = static_cast<double>(glcm.at<float>(i,j));
            mean_i += i * p_ij;
            mean_j += j * p_ij;
        }
    }

    for(int i = 0; i < GRAY_LEVELS; i++) {
        for(int j = 0; j < GRAY_LEVELS; j++) {
            double p_ij = static_cast<double>(glcm.at<float>(i,j));
            std_i += (i - mean_i) * (i - mean_i) * p_ij;
            std_j += (j - mean_j) * (j - mean_j) * p_ij;
        }
    }
    std_i = sqrt(std_i);
    std_j = sqrt(std_j);

    // Calculate Haralick features
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            for(int i = 0; i < GRAY_LEVELS; i++) {
                for(int j = 0; j < GRAY_LEVELS; j++) {
                    double p_ij = static_cast<double>(glcm.at<float>(i,j));
                    contrast += (i-j)*(i-j) * p_ij;
                }
            }
        }

        #pragma omp section
        {
            for(int i = 0; i < GRAY_LEVELS; i++) {
                for(int j = 0; j < GRAY_LEVELS; j++) {
                    double p_ij = static_cast<double>(glcm.at<float>(i,j));
                    correlation += ((i-mean_i)*(j-mean_j)*p_ij)/(std_i*std_j);
                }
            }
        }

        #pragma omp section
        {
            for(int i = 0; i < GRAY_LEVELS; i++) {
                for(int j = 0; j < GRAY_LEVELS; j++) {
                    double p_ij = static_cast<double>(glcm.at<float>(i,j));
                    energy += p_ij * p_ij;
                }
            }
        }

        #pragma omp section
        {
            for(int i = 0; i < GRAY_LEVELS; i++) {
                for(int j = 0; j < GRAY_LEVELS; j++) {
                    double p_ij = static_cast<double>(glcm.at<float>(i,j));
                    homogeneity += p_ij/(1+(i-j)*(i-j));
                }
            }
        }
    }

    features.push_back(contrast);
    features.push_back(correlation);
    features.push_back(energy);
    features.push_back(homogeneity);

    return features;
}

vector<Mat> compute_statistical_features(const Mat& src, int window_size) {
    vector<Mat> features(4);  // mean, variance, skewness, kurtosis
    for(auto& feat : features) {
        feat.create(src.size(), CV_32F);
    }

    int half_size = window_size / 2;

    #pragma omp parallel for
    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {
            // Extract local window
            Rect roi(
                max(0, j-half_size),
                max(0, i-half_size),
                min(window_size, src.cols-max(0,j-half_size)),
                min(window_size, src.rows-max(0,i-half_size))
            );
            Mat window = src(roi);

            // Calculate statistical features
            double mean = compute_mean(window);
            double variance = compute_variance(window, mean);
            double std_dev = sqrt(variance);
            double skewness = compute_skewness(window, mean, std_dev);
            double kurtosis = compute_kurtosis(window, mean, std_dev);

            // Store results
            features[0].at<float>(i,j) = static_cast<float>(mean);
            features[1].at<float>(i,j) = static_cast<float>(variance);
            features[2].at<float>(i,j) = static_cast<float>(skewness);
            features[3].at<float>(i,j) = static_cast<float>(kurtosis);
        }
    }

    return features;
}

Mat compute_lbp(const Mat& src, int radius, int neighbors) {
    Mat dst = Mat::zeros(src.size(), CV_8U);
    vector<int> center_points_x(neighbors);
    vector<int> center_points_y(neighbors);

    // Pre-compute sampling point coordinates
    for(int i = 0; i < neighbors; i++) {
        double angle = 2.0 * CV_PI * i / neighbors;
        center_points_x[i] = static_cast<int>(radius * cos(angle));
        center_points_y[i] = static_cast<int>(-radius * sin(angle));
    }

    #pragma omp parallel for
    for(int i = radius; i < src.rows-radius; i++) {
        for(int j = radius; j < src.cols-radius; j++) {
            uchar center = src.at<uchar>(i,j);
            uchar lbp_code = 0;

            for(int k = 0; k < neighbors; k++) {
                int x = j + center_points_x[k];
                int y = i + center_points_y[k];
                uchar neighbor = src.at<uchar>(y,x);

                lbp_code |= (neighbor > center) << k;
            }

            dst.at<uchar>(i,j) = lbp_code;
        }
    }

    return dst;
}

vector<Mat> generate_gabor_filters(
    int ksize, double sigma, int theta,
    double lambda, double gamma, double psi) {

    vector<Mat> filters;
    filters.reserve(theta);

    double sigma_x = sigma;
    double sigma_y = sigma/gamma;

    int half_size = ksize/2;

    // Generate Gabor filters for different orientations
    for(int t = 0; t < theta; t++) {
        double theta_rad = t * CV_PI / theta;
        Mat kernel(ksize, ksize, CV_32F);

        #pragma omp parallel for
        for(int y = -half_size; y <= half_size; y++) {
            for(int x = -half_size; x <= half_size; x++) {
                // Rotation
                double x_theta = x*cos(theta_rad) + y*sin(theta_rad);
                double y_theta = -x*sin(theta_rad) + y*cos(theta_rad);

                // Gabor function
                double gaussian = exp(-0.5 * (x_theta*x_theta/(sigma_x*sigma_x) +
                                            y_theta*y_theta/(sigma_y*sigma_y)));
                double harmonic = cos(2*CV_PI*x_theta/lambda + psi);

                kernel.at<float>(y+half_size,x+half_size) = static_cast<float>(gaussian * harmonic);
            }
        }

        // Normalize
        kernel = kernel / sum(abs(kernel))[0];
        filters.push_back(kernel);
    }

    return filters;
}

vector<Mat> extract_gabor_features(
    const Mat& src,
    const vector<Mat>& filters) {

    vector<Mat> features;
    features.reserve(filters.size());

    Mat src_float;
    src.convertTo(src_float, CV_32F);

    // Apply convolution with each filter
    #pragma omp parallel for
    for(int i = 0; i < static_cast<int>(filters.size()); i++) {
        Mat response;
        filter2D(src_float, response, CV_32F, filters[i]);

        // Calculate magnitude
        Mat magnitude;
        magnitude = abs(response);

        #pragma omp critical
        features.push_back(magnitude);
    }

    return features;
}

int classify_texture(
    const vector<double>& features,
    const vector<int>& labels,
    const vector<vector<double>>& train_features) {

    // Use K-nearest neighbor classification (K=1)
    int best_label = -1;
    double min_distance = numeric_limits<double>::max();

    // Calculate Euclidean distance to each training sample
    #pragma omp parallel for
    for(int i = 0; i < static_cast<int>(train_features.size()); i++) {
        double distance = 0;
        for(size_t j = 0; j < features.size(); j++) {
            double diff = features[j] - train_features[i][j];
            distance += diff * diff;
        }
        distance = sqrt(distance);

        #pragma omp critical
        {
            if(distance < min_distance) {
                min_distance = distance;
                best_label = labels[i];
            }
        }
    }

    return best_label;
}

} // namespace ip101