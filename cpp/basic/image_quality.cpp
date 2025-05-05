#include "image_quality.hpp"
#include <omp.h>
#include <cmath>

namespace ip101 {

using namespace cv;
using namespace std;

namespace {
// 内部常量定义
constexpr double K1 = 0.01;  // SSIM参数
constexpr double K2 = 0.03;  // SSIM参数
constexpr double EPSILON = 1e-10;  // 数值计算精度

// 计算高斯核
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

// 计算局部统计量
void compute_local_stats(
    const Mat& src,
    Mat& mean,
    Mat& variance,
    int window_size) {

    Mat kernel = create_gaussian_kernel(window_size, window_size/6.0);

    // 计算局部均值
    filter2D(src, mean, CV_64F, kernel);

    // 计算局部方差
    Mat temp;
    multiply(src, src, temp);
    filter2D(temp, variance, CV_64F, kernel);
    multiply(mean, mean, temp);
    variance -= temp;
    variance = max(variance, 0.0);
}

// 计算梯度幅值和方向
void compute_gradient(
    const Mat& src,
    Mat& magnitude,
    Mat& direction) {

    Mat dx, dy;
    Sobel(src, dx, CV_64F, 1, 0);
    Sobel(src, dy, CV_64F, 0, 1);

    magnitude.create(src.size(), CV_64F);
    direction.create(src.size(), CV_64F);

    #pragma omp parallel for collapse(2)
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

    double max_val = 255.0;  // 假设8位图像
    return 20 * log10(max_val) - 10 * log10(mse);
}

double compute_ssim(
    const Mat& src1,
    const Mat& src2,
    int window_size) {

    CV_Assert(src1.size() == src2.size());
    CV_Assert(src1.type() == src2.type());

    // 转换为浮点型
    Mat img1, img2;
    src1.convertTo(img1, CV_64F);
    src2.convertTo(img2, CV_64F);

    // 计算局部统计量
    Mat mu1, mu2, sigma1, sigma2;
    compute_local_stats(img1, mu1, sigma1, window_size);
    compute_local_stats(img2, mu2, sigma2, window_size);

    // 计算协方差
    Mat mu1_mu2, sigma12;
    multiply(img1, img2, sigma12);
    filter2D(sigma12, sigma12, CV_64F, create_gaussian_kernel(window_size, window_size/6.0));
    multiply(mu1, mu2, mu1_mu2);
    sigma12 -= mu1_mu2;

    // 计算SSIM
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

    // 转换为浮点型
    Mat ref, dist;
    src1.convertTo(ref, CV_64F);
    src2.convertTo(dist, CV_64F);

    double vif = 0.0;
    double total_bits = 0.0;

    // 多尺度分解
    for(int scale = 0; scale < num_scales; scale++) {
        // 计算局部统计量
        Mat mu1, mu2, sigma1, sigma2;
        compute_local_stats(ref, mu1, sigma1, 3);
        compute_local_stats(dist, mu2, sigma2, 3);

        // 计算互信息
        Mat g = sigma2 / (sigma1 + EPSILON);
        Mat sigma_n = 0.1 * sigma1;  // 假设噪声方差

        Mat bits_ref, bits_dist;
        log(1 + sigma1/(sigma_n + EPSILON), bits_ref);
        log(1 + g.mul(g).mul(sigma1)/(sigma_n + EPSILON), bits_dist);

        vif += sum(bits_dist)[0];
        total_bits += sum(bits_ref)[0];

        // 降采样
        if(scale < num_scales-1) {
            pyrDown(ref, ref);
            pyrDown(dist, dist);
        }
    }

    return vif / total_bits;
}

double compute_niqe(const Mat& src, int patch_size) {
    // 转换为灰度图
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_64F);

    // 提取局部特征
    vector<double> features;
    int stride = patch_size/2;

    for(int i = 0; i <= gray.rows-patch_size; i += stride) {
        for(int j = 0; j <= gray.cols-patch_size; j += stride) {
            Mat patch = gray(Rect(j,i,patch_size,patch_size));

            // 计算局部统计量
            Scalar mean, stddev;
            meanStdDev(patch, mean, stddev);

            // 计算偏度和峰度
            double m3 = 0, m4 = 0;
            Mat centered = patch - mean[0];
            pow(centered, 3, centered);
            m3 = sum(centered)[0] / (patch_size * patch_size);
            pow(centered, 4, centered);
            m4 = sum(centered)[0] / (patch_size * patch_size);

            double skewness = m3 / pow(stddev[0], 3);
            double kurtosis = m4 / pow(stddev[0], 4) - 3;

            features.push_back(mean[0]);
            features.push_back(stddev[0]);
            features.push_back(skewness);
            features.push_back(kurtosis);
        }
    }

    // 计算特征均值和协方差
    Mat feat_mat(features.size()/4, 4, CV_64F);
    for(size_t i = 0; i < features.size(); i++) {
        feat_mat.at<double>(i/4, i%4) = features[i];
    }

    Mat mean, cov;
    calcCovarMatrix(feat_mat, cov, mean, COVAR_NORMAL | COVAR_ROWS);

    // 计算与MVG模型的距离
    Mat diff = feat_mat - repeat(mean, feat_mat.rows, 1);
    Mat dist = diff * cov.inv() * diff.t();

    return sqrt(mean(dist.diag())[0]);
}

double compute_brisque(const Mat& src) {
    // 转换为灰度图
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_64F);

    // MSCN变换
    Mat mu, sigma;
    compute_local_stats(gray, mu, sigma, 7);
    Mat mscn = (gray - mu) / (sigma + 1);

    // 提取特征
    vector<double> features;

    // 计算MSCN系数统计量
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

    // 计算配对积矩
    Mat paired_products;
    multiply(mscn(Rect(1,0,mscn.cols-1,mscn.rows)),
            mscn(Rect(0,0,mscn.cols-1,mscn.rows)),
            paired_products);

    meanStdDev(paired_products, mean, stddev);
    features.push_back(mean[0]);
    features.push_back(stddev[0]);

    // 使用SVM预测质量分数
    // 注：这里需要预先训练好的SVM模型
    // 简化起见，返回特征的加权和
    double score = 0;
    for(size_t i = 0; i < features.size(); i++) {
        score += features[i] * (i+1);  // 简单的加权
    }

    return score;
}

double compute_msssim(
    const Mat& src1,
    const Mat& src2,
    int num_scales) {

    CV_Assert(src1.size() == src2.size());
    CV_Assert(src1.type() == src2.type());

    // 权重系数
    const double weights[] = {0.0448, 0.2856, 0.3001, 0.2363, 0.1333};

    Mat img1 = src1.clone();
    Mat img2 = src2.clone();
    double msssim = 1.0;

    for(int scale = 0; scale < num_scales; scale++) {
        // 计算当前尺度的SSIM
        double ssim = compute_ssim(img1, img2);
        msssim *= pow(ssim, weights[scale]);

        // 降采样
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

    // 计算梯度特征
    Mat grad1_mag, grad1_dir, grad2_mag, grad2_dir;
    compute_gradient(src1, grad1_mag, grad1_dir);
    compute_gradient(src2, grad2_mag, grad2_dir);

    // 计算相位一致性
    Mat phase_diff = abs(grad1_dir - grad2_dir);
    phase_diff = min(phase_diff, 2*CV_PI - phase_diff);
    Mat phase_sim = cos(phase_diff);

    // 计算梯度相似性
    Mat grad_sim;
    multiply(2*grad1_mag.mul(grad2_mag) + EPSILON,
            phase_sim + EPSILON,
            grad_sim);
    Mat denom = grad1_mag.mul(grad1_mag) +
                grad2_mag.mul(grad2_mag) + EPSILON;
    divide(grad_sim, denom, grad_sim);

    // 计算FSIM
    double fsim = sum(grad_sim.mul(grad1_mag + grad2_mag))[0] /
                 sum(grad1_mag + grad2_mag)[0];

    return fsim;
}

} // namespace ip101