#include <basic/super_resolution.hpp>
#include <omp.h>
#include <cmath>

namespace ip101 {

using namespace cv;
using namespace std;

namespace {
// Internal constants
constexpr double EPSILON = 1e-10;  // Numerical precision
constexpr int BLOCK_SIZE = 16;    // Block processing size

// Bicubic interpolation kernel function
double bicubic_kernel(double x) {
    x = abs(x);
    if(x <= 1.0) {
        return 1.5*x*x*x - 2.5*x*x + 1.0;
    }
    else if(x < 2.0) {
        return -0.5*x*x*x + 2.5*x*x - 4.0*x + 2.0;
    }
    return 0.0;
}

// Extract patch features
Mat extract_patch_features(const Mat& patch) {
    Mat features;
    // DCT transform
    Mat dct_patch;
    dct(patch, dct_patch);
    // Use low-frequency coefficients as features
    features = dct_patch(Rect(0,0,8,8)).reshape(1,1);
    return features;
}

// Calculate image gradients
void compute_gradient(const Mat& src, Mat& dx, Mat& dy) {
    // Calculate gradients using Sobel operator
    Sobel(src, dx, CV_32F, 1, 0);
    Sobel(src, dy, CV_32F, 0, 1);
}

// Calculate patch similarity
double compute_patch_similarity(
    const Mat& patch1,
    const Mat& patch2) {
    Mat diff = patch1 - patch2;
    return norm(diff);
}

} // anonymous namespace

Mat bicubic_sr(const Mat& src, float scale_factor) {
    int new_rows = static_cast<int>(round(src.rows * scale_factor));
    int new_cols = static_cast<int>(round(src.cols * scale_factor));
    Mat dst(new_rows, new_cols, src.type());

    // Process each channel separately
    vector<Mat> channels;
    split(src, channels);
    vector<Mat> upscaled_channels;

    #pragma omp parallel for
    for(int c = 0; c < static_cast<int>(channels.size()); c++) {
        Mat upscaled(new_rows, new_cols, CV_32F);

        // Bicubic interpolation
        for(int i = 0; i < new_rows; i++) {
            float y = i / scale_factor;
            int y0 = static_cast<int>(floor(y));

            for(int j = 0; j < new_cols; j++) {
                float x = j / scale_factor;
                int x0 = static_cast<int>(floor(x));

                double sum = 0;
                double weight_sum = 0;

                // 4x4 neighborhood interpolation
                for(int di = -1; di <= 2; di++) {
                    int yi = y0 + di;
                    if(yi < 0 || yi >= src.rows) continue;

                    double wy = bicubic_kernel(y - yi);

                    for(int dj = -1; dj <= 2; dj++) {
                        int xj = x0 + dj;
                        if(xj < 0 || xj >= src.cols) continue;

                        double wx = bicubic_kernel(x - xj);
                        double w = wx * wy;

                        sum += w * channels[c].at<uchar>(yi,xj);
                        weight_sum += w;
                    }
                }

                upscaled.at<float>(i,j) = static_cast<float>(sum / weight_sum);
            }
        }

        upscaled.convertTo(upscaled, CV_8U);
        upscaled_channels.push_back(upscaled);
    }

    merge(upscaled_channels, dst);
    return dst;
}

Mat sparse_sr(
    const Mat& src,
    float scale_factor,
    int dict_size,
    int patch_size) {

    // Use bicubic interpolation as initial estimate
    Mat initial = bicubic_sr(src, scale_factor);
    Mat result = initial.clone();

    // Extract training samples
    vector<Mat> patches;
    for(int i = 0; i <= src.rows-patch_size; i++) {
        for(int j = 0; j <= src.cols-patch_size; j++) {
            Mat patch = src(Rect(j,i,patch_size,patch_size));
            patches.push_back(patch.clone());
        }
    }

    // Train dictionary
    Mat dictionary(dict_size, patch_size*patch_size, CV_32F);
    for(int i = 0; i < dict_size; i++) {
        int idx = rand() % static_cast<int>(patches.size());
        Mat feat = extract_patch_features(patches[idx]);
        feat.copyTo(dictionary.row(i));
    }

    // Sparse reconstruction for each patch
    #pragma omp parallel for
    for(int i = 0; i < result.rows-patch_size; i++) {
        for(int j = 0; j < result.cols-patch_size; j++) {
            Mat patch = result(Rect(j,i,patch_size,patch_size));
            Mat features = extract_patch_features(patch);

            // Find the most similar dictionary atom
            double min_dist = numeric_limits<double>::max();
            Mat best_atom;

            for(int k = 0; k < dict_size; k++) {
                Mat atom = dictionary.row(k);
                double dist = norm(features, atom);
                if(dist < min_dist) {
                    min_dist = dist;
                    best_atom = atom;
                }
            }

            // Reconstruction
            Mat reconstructed;
            idct(best_atom.reshape(1,patch_size), reconstructed);
            reconstructed.copyTo(result(Rect(j,i,patch_size,patch_size)));
        }
    }

    return result;
}

Mat srcnn_sr(const Mat& src, float scale_factor) {
    // Use bicubic interpolation as initial estimate
    Mat initial = bicubic_sr(src, scale_factor);
    Mat result = initial.clone();

    // SRCNN network parameters (simplified version)
    const int conv1_size = 9;
    const int conv2_size = 1;
    const int conv3_size = 5;

    // First convolution layer
    Mat conv1;
    Mat kernel1 = getGaussianKernel(conv1_size, -1);
    kernel1 = kernel1 * kernel1.t();
    filter2D(result, conv1, -1, kernel1);

    // Second convolution layer (1x1 convolution for non-linear mapping)
    Mat conv2;
    Mat kernel2 = Mat::ones(conv2_size, conv2_size, CV_32F) / static_cast<float>(conv2_size*conv2_size);
    filter2D(conv1, conv2, -1, kernel2);

    // Third convolution layer (reconstruction)
    Mat conv3;
    Mat kernel3 = getGaussianKernel(conv3_size, -1);
    kernel3 = kernel3 * kernel3.t();
    filter2D(conv2, conv3, -1, kernel3);

    // Residual learning
    result = conv3 + initial;

    return result;
}

Mat multi_frame_sr(
    const vector<Mat>& frames,
    float scale_factor) {

    if(frames.empty()) return Mat();

    // Select reference frame
    Mat reference = frames[frames.size()/2];
    Size new_size(static_cast<int>(round(reference.cols * scale_factor)),
                  static_cast<int>(round(reference.rows * scale_factor)));

    // Initial estimate
    Mat result = bicubic_sr(reference, scale_factor);

    // Registration and fusion for each frame
    for(const Mat& frame : frames) {
        if(frame.empty()) continue;
        if(frame.size() != reference.size()) continue;

        // Calculate optical flow
        Mat flow;
        calcOpticalFlowFarneback(reference, frame, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

        // Register based on flow
        Mat warped;
        remap(frame, warped, flow, Mat(), INTER_LINEAR);

        // Upscale registered frame
        Mat upscaled = bicubic_sr(warped, scale_factor);

        // Weighted fusion
        double alpha = 0.5;
        addWeighted(result, 1-alpha, upscaled, alpha, 0, result);
    }

    return result;
}

Mat adaptive_weight_sr(
    const Mat& src,
    float scale_factor,
    int patch_size,
    int search_window) {

    // Initial estimate
    Mat initial = bicubic_sr(src, scale_factor);
    Mat result = initial.clone();
    int half_patch = patch_size / 2;
    int half_window = search_window / 2;

    // Adaptive reconstruction for each patch
    #pragma omp parallel for
    for(int i = half_patch; i < result.rows-half_patch; i++) {
        for(int j = half_patch; j < result.cols-half_patch; j++) {
            Mat ref_patch = result(Rect(j-half_patch, i-half_patch,
                                      patch_size, patch_size));

            // Find similar patches in search window
            vector<pair<Mat,double>> similar_patches;

            for(int di = -half_window; di <= half_window; di++) {
                int yi = i + di;
                if(yi < half_patch || yi >= result.rows-half_patch) continue;

                for(int dj = -half_window; dj <= half_window; dj++) {
                    int xj = j + dj;
                    if(xj < half_patch || xj >= result.cols-half_patch) continue;

                    Mat cand_patch = result(Rect(xj-half_patch, yi-half_patch,
                                               patch_size, patch_size));
                    double dist = compute_patch_similarity(ref_patch, cand_patch);
                    similar_patches.push_back({cand_patch, dist});
                }
            }

            // Calculate weights and fuse patches
            Mat weighted_sum = Mat::zeros(patch_size, patch_size, CV_32F);
            double weight_sum = 0;

            for(const auto& p : similar_patches) {
                double weight = exp(-p.second / (2*10*10));  // sigma = 10
                Mat patch_float;
                p.first.convertTo(patch_float, CV_32F);
                weighted_sum += weight * patch_float;
                weight_sum += weight;
            }

            weighted_sum /= weight_sum;
            weighted_sum.convertTo(weighted_sum, result.type());
            weighted_sum.copyTo(result(Rect(j-half_patch, i-half_patch,
                                          patch_size, patch_size)));
        }
    }

    return result;
}

Mat iterative_backprojection_sr(
    const Mat& src,
    float scale_factor,
    int num_iterations) {

    // Initial estimate
    Mat result = bicubic_sr(src, scale_factor);
    Size low_size = src.size();

    // Iterative optimization
    for(int iter = 0; iter < num_iterations; iter++) {
        // Downsample current estimate
        Mat downscaled;
        resize(result, downscaled, low_size, 0, 0, INTER_AREA);

        // Calculate residual
        Mat diff = src - downscaled;

        // Upsample residual
        Mat up_diff;
        resize(diff, up_diff, result.size(), 0, 0, INTER_CUBIC);

        // Update estimate
        result += 0.1 * up_diff;  // Step size 0.1
    }

    return result;
}

Mat gradient_guided_sr(
    const Mat& src,
    float scale_factor,
    float lambda) {

    // Initial estimate
    Mat result = bicubic_sr(src, scale_factor);

    // Calculate gradients of low-resolution image
    Mat dx_low, dy_low;
    compute_gradient(src, dx_low, dy_low);

    // Upsample gradients
    resize(dx_low, dx_low, result.size(), 0, 0, INTER_CUBIC);
    resize(dy_low, dy_low, result.size(), 0, 0, INTER_CUBIC);

    // Calculate gradients of high-resolution image
    Mat dx_high, dy_high;
    compute_gradient(result, dx_high, dy_high);

    // Gradient-guided optimization
    #pragma omp parallel for
    for(int i = 1; i < result.rows-1; i++) {
        for(int j = 1; j < result.cols-1; j++) {
            float gx_low = dx_low.at<float>(i,j);
            float gy_low = dy_low.at<float>(i,j);
            float gx_high = dx_high.at<float>(i,j);
            float gy_high = dy_high.at<float>(i,j);

            // Gradient consistency constraint
            float dx = lambda * (gx_high - gx_low);
            float dy = lambda * (gy_high - gy_low);

            // Update pixel value
            result.at<uchar>(i,j) = saturate_cast<uchar>(
                result.at<uchar>(i,j) - (dx + dy));
        }
    }

    return result;
}

Mat self_similarity_sr(
    const Mat& src,
    float scale_factor,
    int patch_size,
    int num_similar) {

    // Initial estimate
    Mat result = bicubic_sr(src, scale_factor);
    int half_patch = patch_size / 2;

    // Self-similarity reconstruction for each patch
    #pragma omp parallel for
    for(int i = half_patch; i < result.rows-half_patch; i++) {
        for(int j = half_patch; j < result.cols-half_patch; j++) {
            Mat ref_patch = result(Rect(j-half_patch, i-half_patch,
                                      patch_size, patch_size));

            // Search for similar patches across the entire image
            vector<pair<Mat,double>> similar_patches;

            for(int yi = half_patch; yi < result.rows-half_patch; yi++) {
                for(int xj = half_patch; xj < result.cols-half_patch; xj++) {
                    if(abs(yi-i) < patch_size && abs(xj-j) < patch_size) continue;

                    Mat cand_patch = result(Rect(xj-half_patch, yi-half_patch,
                                               patch_size, patch_size));
                    double dist = compute_patch_similarity(ref_patch, cand_patch);

                    if(similar_patches.size() < static_cast<size_t>(num_similar)) {
                        similar_patches.push_back({cand_patch, dist});
                    }
                    else {
                        // Replace patch with largest distance
                        auto max_it = max_element(similar_patches.begin(),
                                                similar_patches.end(),
                                                [](const auto& a, const auto& b) {
                                                    return a.second < b.second;
                                                });
                        if(dist < max_it->second) {
                            *max_it = {cand_patch, dist};
                        }
                    }
                }
            }

            // Reconstruction based on similar patches
            Mat sum = Mat::zeros(patch_size, patch_size, CV_32F);
            double weight_sum = 0;

            for(const auto& p : similar_patches) {
                double weight = exp(-p.second / (2*10*10));  // sigma = 10
                Mat patch_float;
                p.first.convertTo(patch_float, CV_32F);
                sum += weight * patch_float;
                weight_sum += weight;
            }

            sum /= weight_sum;
            sum.convertTo(sum, result.type());
            sum.copyTo(result(Rect(j-half_patch, i-half_patch,
                                 patch_size, patch_size)));
        }
    }

    return result;
}

} // namespace ip101