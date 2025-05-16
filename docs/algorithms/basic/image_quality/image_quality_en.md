# Image Quality Assessment 🔍

> Image quality assessment is like a "quality appraiser" in the digital world! Through various "appraisal tools" and "evaluation standards", we can professionally assess image quality, just like an appraiser evaluating artwork. Let's explore this magical image "appraisal studio" together!

## Table of Contents
- [Image Quality Assessment 🔍](#image-quality-assessment-)
  - [Table of Contents](#table-of-contents)
  - [1. Peak Signal-to-Noise Ratio (PSNR)](#1-peak-signal-to-noise-ratio-psnr)
  - [2. Structural Similarity Index (SSIM)](#2-structural-similarity-index-ssim)
  - [3. Mean Squared Error (MSE)](#3-mean-squared-error-mse)
  - [4. Visual Information Fidelity (VIF)](#4-visual-information-fidelity-vif)
  - [5. No-Reference Quality Assessment](#5-no-reference-quality-assessment)
  - [Notes](#notes)
  - [Summary](#summary)
  - [References](#references)

## 1. Peak Signal-to-Noise Ratio (PSNR)

PSNR is like a "signal-to-noise ratio meter" used to measure image distortion. It calculates the mean squared error (MSE) between the original and distorted images, then converts it to decibels (dB).

Mathematical expression:
$$
PSNR = 10 \cdot \log_{10}\left(\frac{MAX_I^2}{MSE}\right)
$$

Where:
- $MAX_I$ is the maximum pixel value (usually 255)
- $MSE$ is the mean squared error

C++ Implementation:
```cpp
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
```

Python Implementation:
```python
def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR)

    Args:
        img1: First image
        img2: Second image

    Returns:
        float: PSNR value in dB
    """
    # Ensure images have same size
    assert img1.shape == img2.shape

    # Calculate MSE
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

    # Avoid division by zero
    if mse == 0:
        return float('inf')

    # Calculate PSNR
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)

    return psnr
```

## 2. Structural Similarity Index (SSIM)

SSIM is like a "structural similarity analyzer" used to evaluate how well image structures are preserved. It considers not only pixel value differences but also structural information.

Mathematical expression:
$$
SSIM(x,y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}
$$

Where:
- $\mu$ is the mean
- $\sigma$ is the standard deviation
- $c_1, c_2$ are constants

C++ Implementation:
```cpp
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
```

Python Implementation:
```python
def compute_ssim(img1: np.ndarray, img2: np.ndarray,
                window_size: int = 11) -> float:
    """Calculate Structural Similarity Index (SSIM)

    Args:
        img1: First image
        img2: Second image
        window_size: Size of the sliding window

    Returns:
        float: SSIM value (between 0 and 1, higher is better)
    """
    # Ensure images have same size
    assert img1.shape == img2.shape

    # Convert to float
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Create Gaussian window
    window = gaussian_filter(np.ones((window_size, window_size)),
                           sigma=1.5)
    window = window / np.sum(window)

    # Calculate means
    mu1 = signal.convolve2d(img1, window, mode='valid')
    mu2 = signal.convolve2d(img2, window, mode='valid')

    # Calculate variances and covariance
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = signal.convolve2d(img1 * img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.convolve2d(img2 * img2, window, mode='valid') - mu2_sq
    sigma12 = signal.convolve2d(img1 * img2, window, mode='valid') - mu1_mu2

    # SSIM parameters
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # Calculate SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(ssim_map)
```

## 3. Mean Squared Error (MSE)

MSE is like an "error measurement ruler" used to calculate pixel differences between two images. It's the foundation of PSNR calculation and the most basic image quality assessment metric.

Mathematical expression:
$$
MSE = \frac{1}{MN}\sum_{i=1}^M\sum_{j=1}^N[I(i,j) - K(i,j)]^2
$$

Where:
- $M,N$ are image dimensions
- $I,K$ are original and distorted images

C++ Implementation:
```cpp
double compute_mse(const Mat& src1, const Mat& src2) {
    CV_Assert(src1.size() == src2.size());
    CV_Assert(src1.type() == src2.type());

    Mat diff;
    absdiff(src1, src2, diff);
    diff.convertTo(diff, CV_64F);
    multiply(diff, diff, diff);

    return mean(diff)[0];
}
```

Python Implementation:
```python
def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Mean Squared Error (MSE)

    Args:
        img1: First image
        img2: Second image

    Returns:
        float: MSE value
    """
    # Ensure images have same size
    assert img1.shape == img2.shape

    # Calculate MSE
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    return mse
```

## 4. Visual Information Fidelity (VIF)

VIF is like a "visual fidelity detector" that evaluates image quality based on natural scene statistics and human visual system characteristics. VIF considers the amount of information in the image and how well this information is preserved during distortion.

Mathematical expression:
$$
VIF = \frac{\sum_{j\in\text{subbands}} I(C_j^N;F_j^N|s_j^N)}{\sum_{j\in\text{subbands}} I(C_j^N;E_j^N|s_j^N)}
$$

Where:
- $I$ is mutual information
- $C_j^N$ is reference image subband
- $F_j^N$ is distorted image subband
- $E_j^N$ is noise image subband

C++ Implementation:
```cpp
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
```

Python Implementation:
```python
def compute_vif(img1: np.ndarray, img2: np.ndarray, num_scales: int = 4) -> float:
    """Calculate Visual Information Fidelity (VIF)

    Args:
        img1: First image
        img2: Second image
        num_scales: Number of scales

    Returns:
        float: VIF value
    """
    # Convert to float
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    vif = 0.0
    total_bits = 0.0

    # Multi-scale decomposition
    for scale in range(num_scales):
        # Calculate local statistics
        mu1 = gaussian_filter(img1, sigma=1.5)
        mu2 = gaussian_filter(img2, sigma=1.5)

        sigma1 = gaussian_filter(img1**2, sigma=1.5) - mu1**2
        sigma2 = gaussian_filter(img2**2, sigma=1.5) - mu2**2
        sigma12 = gaussian_filter(img1*img2, sigma=1.5) - mu1*mu2

        # Calculate mutual information
        g = sigma2 / (sigma1 + 1e-10)
        sigma_n = 0.1 * sigma1  # Assume noise variance

        bits_ref = np.log2(1 + sigma1/(sigma_n + 1e-10))
        bits_dist = np.log2(1 + g**2 * sigma1/(sigma_n + 1e-10))

        vif += np.sum(bits_dist)
        total_bits += np.sum(bits_ref)

        # Downsample
        if scale < num_scales-1:
            img1 = cv2.pyrDown(img1)
            img2 = cv2.pyrDown(img2)

    return vif / total_bits
```

## 5. No-Reference Quality Assessment

No-reference quality assessment is like an "independent appraiser" that can evaluate image quality without a reference image. It assesses quality by analyzing the natural statistical properties of the image.

Main methods:
1. Based on natural scene statistics
2. Based on image feature analysis
3. Based on deep learning

C++ Implementation:
```cpp
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
```

Python Implementation:
```python
def no_reference_quality_assessment(img: np.ndarray) -> float:
    """No-reference quality assessment

    Args:
        img: Input image

    Returns:
        float: Quality score (between 0 and 1, higher is better)
    """
    # Calculate image gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

    # Calculate local contrast
    local_contrast = np.std(gradient_magnitude)

    # Calculate image entropy
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist / np.sum(hist)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))

    # Calculate noise level
    noise = np.std(img - cv2.GaussianBlur(img, (5, 5), 0))

    # Combined score
    score = (local_contrast * 0.4 + entropy * 0.3 + (1 - noise/255) * 0.3)

    return np.clip(score, 0, 1)
```

## Notes

1. Image Preprocessing
   - Ensure proper preprocessing (denoising, alignment, etc.) before quality assessment
   - Convert color images to grayscale for assessment
   - Ensure consistent image sizes, scale if necessary

2. Metric Selection
   - PSNR is suitable for compression quality but insensitive to structural distortion
   - SSIM is better for structural similarity but computationally intensive
   - MSE is simple but poorly correlated with human perception
   - VIF is complex but better matches human perception
   - No-reference assessment is suitable for real-time applications but less accurate

3. Performance Optimization
   - Consider using ROI (Region of Interest) for large images
   - Use multi-threading or GPU acceleration
   - For real-time applications, reduce sampling rate or use fast algorithms

4. Practical Application Suggestions
   - Choose appropriate metrics based on specific application scenarios
   - Consider combining multiple metrics for comprehensive evaluation
   - Regularly validate assessment results
   - Consider computational efficiency of metrics

## Summary

Image quality assessment is like a "quality appraiser" in the digital world! Through objective evaluation metrics, subjective evaluation methods, and no-reference quality assessment, we can professionally evaluate image quality. In practical applications, we need to choose appropriate "appraisal methods" based on specific scenarios, just like appraisers choosing different methods for different artworks.

Remember: Good image quality assessment is like an experienced "appraiser" who can accurately evaluate while considering practical application needs! 🔍

## References

1. Wang Z, et al. Image quality assessment: from error visibility to structural similarity[J]. TIP, 2004
2. Sheikh H R, et al. A statistical evaluation of recent full reference image quality assessment algorithms[J]. TIP, 2006
3. Mittal A, et al. No-reference image quality assessment in the spatial domain[J]. TIP, 2012
4. Zhang L, et al. FSIM: A feature similarity index for image quality assessment[J]. TIP, 2011
5. OpenCV Documentation: https://docs.opencv.org/
6. More Resources: [IP101 Project Homepage](https://github.com/GlimmerLab/IP101)