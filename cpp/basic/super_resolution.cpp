#include "super_resolution.hpp"
#include <omp.h>
#include <cmath>

namespace ip101 {

using namespace cv;
using namespace std;

namespace {
// 内部常量定义
constexpr double EPSILON = 1e-10;  // 数值计算精度
constexpr int BLOCK_SIZE = 16;    // 分块处理大小

// 双三次插值核函数
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

// 计算图像块的特征
Mat extract_patch_features(const Mat& patch) {
    Mat features;
    // DCT变换
    Mat dct_patch;
    dct(patch, dct_patch);
    // 取低频系数作为特征
    features = dct_patch(Rect(0,0,8,8)).reshape(1,1);
    return features;
}

// 计算图像梯度
void compute_gradient(const Mat& src, Mat& dx, Mat& dy) {
    // Sobel算子计算梯度
    Sobel(src, dx, CV_32F, 1, 0);
    Sobel(src, dy, CV_32F, 0, 1);
}

// 计算块相似度
double compute_patch_similarity(
    const Mat& patch1,
    const Mat& patch2) {
    Mat diff = patch1 - patch2;
    return norm(diff);
}

} // anonymous namespace

Mat bicubic_sr(const Mat& src, float scale_factor) {
    int new_rows = round(src.rows * scale_factor);
    int new_cols = round(src.cols * scale_factor);
    Mat dst(new_rows, new_cols, src.type());

    // 对每个通道进行插值
    vector<Mat> channels;
    split(src, channels);
    vector<Mat> upscaled_channels;

    #pragma omp parallel for
    for(int c = 0; c < channels.size(); c++) {
        Mat upscaled(new_rows, new_cols, CV_32F);

        // 双三次插值
        for(int i = 0; i < new_rows; i++) {
            float y = i / scale_factor;
            int y0 = floor(y);

            for(int j = 0; j < new_cols; j++) {
                float x = j / scale_factor;
                int x0 = floor(x);

                double sum = 0;
                double weight_sum = 0;

                // 4x4邻域插值
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

                upscaled.at<float>(i,j) = sum / weight_sum;
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

    // 先使用双三次插值放大
    Mat initial = bicubic_sr(src, scale_factor);
    Mat result = initial.clone();

    // 提取训练样本
    vector<Mat> patches;
    for(int i = 0; i <= src.rows-patch_size; i++) {
        for(int j = 0; j <= src.cols-patch_size; j++) {
            Mat patch = src(Rect(j,i,patch_size,patch_size));
            patches.push_back(patch.clone());
        }
    }

    // 训练字典
    Mat dictionary(dict_size, patch_size*patch_size, CV_32F);
    for(int i = 0; i < dict_size; i++) {
        int idx = rand() % patches.size();
        Mat feat = extract_patch_features(patches[idx]);
        feat.copyTo(dictionary.row(i));
    }

    // 对每个块进行稀疏重建
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < result.rows-patch_size; i++) {
        for(int j = 0; j < result.cols-patch_size; j++) {
            Mat patch = result(Rect(j,i,patch_size,patch_size));
            Mat features = extract_patch_features(patch);

            // 找到最相似的字典原子
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

            // 重建
            Mat reconstructed;
            idct(best_atom.reshape(1,patch_size), reconstructed);
            reconstructed.copyTo(result(Rect(j,i,patch_size,patch_size)));
        }
    }

    return result;
}

Mat srcnn_sr(const Mat& src, float scale_factor) {
    // 先使用双三次插值放大
    Mat initial = bicubic_sr(src, scale_factor);
    Mat result = initial.clone();

    // SRCNN网络参数(简化版本)
    const int conv1_size = 9;
    const int conv2_size = 1;
    const int conv3_size = 5;

    // 第一层卷积
    Mat conv1;
    Mat kernel1 = getGaussianKernel(conv1_size, -1);
    kernel1 = kernel1 * kernel1.t();
    filter2D(result, conv1, -1, kernel1);

    // 第二层卷积(1x1卷积模拟非线性映射)
    Mat conv2;
    Mat kernel2 = Mat::ones(conv2_size, conv2_size, CV_32F) / (float)(conv2_size*conv2_size);
    filter2D(conv1, conv2, -1, kernel2);

    // 第三层卷积(重建)
    Mat conv3;
    Mat kernel3 = getGaussianKernel(conv3_size, -1);
    kernel3 = kernel3 * kernel3.t();
    filter2D(conv2, conv3, -1, kernel3);

    // 残差学习
    result = conv3 + initial;

    return result;
}

Mat multi_frame_sr(
    const vector<Mat>& frames,
    float scale_factor) {

    if(frames.empty()) return Mat();

    // 选择参考帧
    Mat reference = frames[frames.size()/2];
    Size new_size(round(reference.cols * scale_factor),
                  round(reference.rows * scale_factor));

    // 初始估计
    Mat result = bicubic_sr(reference, scale_factor);

    // 对每帧进行配准和融合
    for(const Mat& frame : frames) {
        if(frame.empty()) continue;
        if(frame.size() != reference.size()) continue;

        // 计算光流
        Mat flow;
        calcOpticalFlowFarneback(reference, frame, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

        // 根据光流进行配准
        Mat warped;
        remap(frame, warped, flow, Mat(), INTER_LINEAR);

        // 上采样配准后的帧
        Mat upscaled = bicubic_sr(warped, scale_factor);

        // 加权融合
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

    // 初始估计
    Mat initial = bicubic_sr(src, scale_factor);
    Mat result = initial.clone();
    int half_patch = patch_size / 2;
    int half_window = search_window / 2;

    // 对每个块进行自适应重建
    #pragma omp parallel for collapse(2)
    for(int i = half_patch; i < result.rows-half_patch; i++) {
        for(int j = half_patch; j < result.cols-half_patch; j++) {
            Mat ref_patch = result(Rect(j-half_patch, i-half_patch,
                                      patch_size, patch_size));

            // 在搜索窗口内找相似块
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

            // 根据相似度计算权重并融合
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

    // 初始估计
    Mat result = bicubic_sr(src, scale_factor);
    Size low_size = src.size();

    // 迭代优化
    for(int iter = 0; iter < num_iterations; iter++) {
        // 下采样当前估计
        Mat downscaled;
        resize(result, downscaled, low_size, 0, 0, INTER_AREA);

        // 计算残差
        Mat diff = src - downscaled;

        // 上采样残差
        Mat up_diff;
        resize(diff, up_diff, result.size(), 0, 0, INTER_CUBIC);

        // 更新估计
        result += 0.1 * up_diff;  // 步长0.1
    }

    return result;
}

Mat gradient_guided_sr(
    const Mat& src,
    float scale_factor,
    float lambda) {

    // 初始估计
    Mat result = bicubic_sr(src, scale_factor);

    // 计算低分辨率图像的梯度
    Mat dx_low, dy_low;
    compute_gradient(src, dx_low, dy_low);

    // 上采样梯度
    resize(dx_low, dx_low, result.size(), 0, 0, INTER_CUBIC);
    resize(dy_low, dy_low, result.size(), 0, 0, INTER_CUBIC);

    // 计算高分辨率图像的梯度
    Mat dx_high, dy_high;
    compute_gradient(result, dx_high, dy_high);

    // 梯度引导的优化
    #pragma omp parallel for collapse(2)
    for(int i = 1; i < result.rows-1; i++) {
        for(int j = 1; j < result.cols-1; j++) {
            float gx_low = dx_low.at<float>(i,j);
            float gy_low = dy_low.at<float>(i,j);
            float gx_high = dx_high.at<float>(i,j);
            float gy_high = dy_high.at<float>(i,j);

            // 梯度一致性约束
            float dx = lambda * (gx_high - gx_low);
            float dy = lambda * (gy_high - gy_low);

            // 更新像素值
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

    // 初始估计
    Mat result = bicubic_sr(src, scale_factor);
    int half_patch = patch_size / 2;

    // 对每个块进行自相似性重建
    #pragma omp parallel for collapse(2)
    for(int i = half_patch; i < result.rows-half_patch; i++) {
        for(int j = half_patch; j < result.cols-half_patch; j++) {
            Mat ref_patch = result(Rect(j-half_patch, i-half_patch,
                                      patch_size, patch_size));

            // 在全图范围内搜索相似块
            vector<pair<Mat,double>> similar_patches;

            for(int yi = half_patch; yi < result.rows-half_patch; yi++) {
                for(int xj = half_patch; xj < result.cols-half_patch; xj++) {
                    if(abs(yi-i) < patch_size && abs(xj-j) < patch_size) continue;

                    Mat cand_patch = result(Rect(xj-half_patch, yi-half_patch,
                                               patch_size, patch_size));
                    double dist = compute_patch_similarity(ref_patch, cand_patch);

                    if(similar_patches.size() < num_similar) {
                        similar_patches.push_back({cand_patch, dist});
                    }
                    else {
                        // 替换距离最大的patch
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

            // 基于相似块进行重建
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