#include <basic/image_quality.hpp>
#include <omp.h>
#include <cmath>

namespace ip101 {

using namespace cv;
using namespace std;

namespace {
// Internal constants
constexpr double K1 = 0.01;  // SSIM parameter
constexpr double K2 = 0.03;  // SSIM parameter
constexpr double EPSILON = 1e-10;  // Numerical precision

// Calculate Gaussian kernel
Mat create_gaussian_kernel(int size, double sigma) {
    Mat kernel(size, size, CV_64F);
    double sum = 0.0;
    int half = size / 2;

    for(int i = -half; i <= half; i++) {
        for(int j = -half; j <= half; j++) {
            double g = exp(-(i*i + j*j)/(2*sigma*sigma));
            kernel.at<double>(i+half, j+half) = g;
            sum += g;
        }
    }

    kernel /= sum;
    return kernel;
}

// Calculate local statistics
void compute_local_stats(
    const Mat& src,
    Mat& mean,
    Mat& variance,
    int window_size) {

    Mat kernel = create_gaussian_kernel(window_size, window_size/6.0);

    // Calculate local mean
    filter2D(src, mean, CV_64F, kernel);

    // Calculate local variance
    Mat temp;
    multiply(src, src, temp);
    filter2D(temp, variance, CV_64F, kernel);
    multiply(mean, mean, temp);
    variance -= temp;
    variance = max(variance, 0.0);
}

// Calculate gradient magnitude and direction
void compute_gradient(
    const Mat& src,
    Mat& magnitude,
    Mat& direction) {

    Mat dx, dy;
    Sobel(src, dx, CV_64F, 1, 0);
    Sobel(src, dy, CV_64F, 0, 1);

    magnitude.create(src.size(), CV_64F);
    direction.create(src.size(), CV_64F);

    #pragma omp parallel for
    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {
            double gx = dx.at<double>(i,j);
            double gy = dy.at<double>(i,j);
            magnitude.at<double>(i,j) = sqrt(gx*gx + gy*gy);
            direction.at<double>(i,j) = atan2(gy, gx);
        }
    }
}

} // anonymous namespace

double compute_psnr(const Mat& src1, const Mat& src2) {
    CV_Assert(src1.size() == src2.size());
    CV_Assert(src1.type() == src2.type());

    Mat diff;
    absdiff(src1, src2, diff);
    diff.convertTo(diff, CV_64F);
    multiply(diff, diff, diff);

    double mse = mean(diff)[0];
    if(mse < EPSILON) return INFINITY;

    double max_val = 255.0;  // Assume 8-bit image
    return 20 * log10(max_val) - 10 * log10(mse);
}

double compute_ssim(
    const Mat& src1,
    const Mat& src2,
    int window_size) {

    CV_Assert(src1.size() == src2.size());
    CV_Assert(src1.type() == src2.type());

    // Convert to float
    Mat img1, img2;
    src1.convertTo(img1, CV_64F);
    src2.convertTo(img2, CV_64F);

    // Calculate local statistics
    Mat mu1, mu2, sigma1, sigma2;
    compute_local_stats(img1, mu1, sigma1, window_size);
    compute_local_stats(img2, mu2, sigma2, window_size);

    // Calculate covariance
    Mat mu1_mu2, sigma12;
    multiply(img1, img2, sigma12);
    filter2D(sigma12, sigma12, CV_64F, create_gaussian_kernel(window_size, window_size/6.0));
    multiply(mu1, mu2, mu1_mu2);
    sigma12 -= mu1_mu2;

    // Calculate SSIM
    double C1 = (K1 * 255) * (K1 * 255);
    double C2 = (K2 * 255) * (K2 * 255);

    Mat ssim_map;
    multiply(2*mu1_mu2 + C1, 2*sigma12 + C2, ssim_map);
    Mat denom;
    multiply(mu1.mul(mu1) + mu2.mul(mu2) + C1,
            sigma1 + sigma2 + C2, denom);
    divide(ssim_map, denom, ssim_map);

    return mean(ssim_map)[0];
}

double compute_mse(const Mat& src1, const Mat& src2) {
    CV_Assert(src1.size() == src2.size());
    CV_Assert(src1.type() == src2.type());

    Mat diff;
    absdiff(src1, src2, diff);
    diff.convertTo(diff, CV_64F);
    multiply(diff, diff, diff);

    return mean(diff)[0];
}

double compute_vif(
    const Mat& src1,
    const Mat& src2,
    int num_scales) {

    CV_Assert(src1.size() == src2.size());
    CV_Assert(src1.type() == src2.type());

    // Convert to float
    Mat ref, dist;
    src1.convertTo(ref, CV_64F);
    src2.convertTo(dist, CV_64F);

    double vif = 0.0;
    double total_bits = 0.0;

    // Multi-scale decomposition
    for(int scale = 0; scale < num_scales; scale++) {
        // Calculate local statistics
        Mat mu1, mu2, sigma1, sigma2;
        compute_local_stats(ref, mu1, sigma1, 3);
        compute_local_stats(dist, mu2, sigma2, 3);

        // Calculate mutual information
        Mat g = sigma2 / (sigma1 + EPSILON);
        Mat sigma_n = 0.1 * sigma1;  // Assume noise variance

        Mat bits_ref, bits_dist;
        log(1 + sigma1/(sigma_n + EPSILON), bits_ref);
        log(1 + g.mul(g).mul(sigma1)/(sigma_n + EPSILON), bits_dist);

        vif += sum(bits_dist)[0];
        total_bits += sum(bits_ref)[0];

        // Downsample
        if(scale < num_scales-1) {
            pyrDown(ref, ref);
            pyrDown(dist, dist);
        }
    }

    return vif / total_bits;
}

double compute_niqe(const Mat& src, int patch_size) {
    // Convert to grayscale
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_64F);

    // Extract local features
    vector<double> features;
    int stride = patch_size/2;

    for(int i = 0; i <= gray.rows-patch_size; i += stride) {
        for(int j = 0; j <= gray.cols-patch_size; j += stride) {
            Mat patch = gray(Rect(j,i,patch_size,patch_size));

            // Calculate local statistics
            Scalar mean, stddev;
            meanStdDev(patch, mean, stddev);

            // Calculate skewness and kurtosis
            double m3 = 0, m4 = 0;
            Mat centered = patch - mean[0];
            Mat centered_pow3, centered_pow4;
            pow(centered, 3, centered_pow3);
            m3 = sum(centered_pow3)[0] / (patch_size * patch_size);
            pow(centered, 4, centered_pow4);
            m4 = sum(centered_pow4)[0] / (patch_size * patch_size);

            double skewness = m3 / pow(stddev[0], 3);
            double kurtosis = m4 / pow(stddev[0], 4) - 3;

            features.push_back(mean[0]);
            features.push_back(stddev[0]);
            features.push_back(skewness);
            features.push_back(kurtosis);
        }
    }

    // Calculate feature mean and covariance
    int rows = static_cast<int>(features.size() / 4);
    Mat feat_mat(rows, 4, CV_64F);
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < 4; j++) {
            feat_mat.at<double>(i, j) = features[i*4 + j];
        }
    }

    Mat mean, cov;
    calcCovarMatrix(feat_mat, cov, mean, COVAR_NORMAL | COVAR_ROWS);

    // Calculate distance to MVG model
    Mat diff = feat_mat - repeat(mean, feat_mat.rows, 1);
    Mat dist = diff * cov.inv() * diff.t();

    // Extract diagonal elements from dist matrix for the distance
    Mat diagonal;
    diagonal.create(1, dist.rows, CV_64F);
    for(int i = 0; i < dist.rows; i++) {
        diagonal.at<double>(0, i) = dist.at<double>(i, i);
    }

    // Calculate mean of diagonal elements properly
    double mean_val = cv::mean(diagonal)[0];
    return sqrt(mean_val);
}

double compute_brisque(const Mat& src) {
    // Convert to grayscale
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_64F);

    // MSCN transformation
    Mat mu, sigma;
    compute_local_stats(gray, mu, sigma, 7);
    Mat mscn = (gray - mu) / (sigma + 1);

    // Extract features
    vector<double> features;

    // Calculate MSCN coefficient statistics
    Scalar mean, stddev;
    meanStdDev(mscn, mean, stddev);

    Mat temp;
    pow(mscn - mean[0], 3, temp);
    double skewness = sum(temp)[0] / (mscn.rows * mscn.cols);
    pow(mscn - mean[0], 4, temp);
    double kurtosis = sum(temp)[0] / (mscn.rows * mscn.cols) - 3;

    features.push_back(mean[0]);
    features.push_back(stddev[0]);
    features.push_back(skewness);
    features.push_back(kurtosis);

    // Calculate paired products
    Mat paired_products;
    Mat mscn_shifted_right = mscn(Rect(1, 0, mscn.cols-1, mscn.rows));
    Mat mscn_shifted_left = mscn(Rect(0, 0, mscn.cols-1, mscn.rows));
    multiply(mscn_shifted_right, mscn_shifted_left, paired_products);

    meanStdDev(paired_products, mean, stddev);
    features.push_back(mean[0]);
    features.push_back(stddev[0]);

    // Use SVM to predict quality score
    // Note: This requires pre-trained SVM model
    // Simplified by returning weighted sum of features
    double score = 0;
    for(size_t i = 0; i < features.size(); i++) {
        score += features[i] * (i+1);  // Simple weighting
    }

    return score;
}

double compute_msssim(
    const Mat& src1,
    const Mat& src2,
    int num_scales) {

    CV_Assert(src1.size() == src2.size());
    CV_Assert(src1.type() == src2.type());

    // Weight coefficients
    const double weights[] = {0.0448, 0.2856, 0.3001, 0.2363, 0.1333};

    Mat img1 = src1.clone();
    Mat img2 = src2.clone();
    double msssim = 1.0;

    for(int scale = 0; scale < num_scales; scale++) {
        // Calculate SSIM at current scale
        double ssim = compute_ssim(img1, img2);
        msssim *= pow(ssim, weights[scale]);

        // Downsample
        if(scale < num_scales-1) {
            pyrDown(img1, img1);
            pyrDown(img2, img2);
        }
    }

    return msssim;
}

double compute_fsim(const Mat& src1, const Mat& src2) {
    CV_Assert(src1.size() == src2.size());
    CV_Assert(src1.type() == src2.type());

    // Calculate gradient features
    Mat grad1_mag, grad1_dir, grad2_mag, grad2_dir;
    compute_gradient(src1, grad1_mag, grad1_dir);
    compute_gradient(src2, grad2_mag, grad2_dir);

    // Calculate phase consistency
    Mat phase_diff;
    absdiff(grad1_dir, grad2_dir, phase_diff);

    // Apply cos element-wise to the phase difference
    Mat phase_sim(phase_diff.size(), CV_64F);

    #pragma omp parallel for
    for(int i = 0; i < phase_diff.rows; i++) {
        for(int j = 0; j < phase_diff.cols; j++) {
            double pd = min(phase_diff.at<double>(i, j), 2*CV_PI - phase_diff.at<double>(i, j));
            phase_sim.at<double>(i, j) = cos(pd);
        }
    }

    // Calculate gradient similarity
    Mat grad_sim;
    multiply(2*grad1_mag.mul(grad2_mag) + EPSILON,
            phase_sim + EPSILON,
            grad_sim);
    Mat denom = grad1_mag.mul(grad1_mag) +
                grad2_mag.mul(grad2_mag) + EPSILON;
    divide(grad_sim, denom, grad_sim);

    // Calculate FSIM
    double fsim = sum(grad_sim.mul(grad1_mag + grad2_mag))[0] /
                 sum(grad1_mag + grad2_mag)[0];

    return fsim;
}

} // namespace ip101