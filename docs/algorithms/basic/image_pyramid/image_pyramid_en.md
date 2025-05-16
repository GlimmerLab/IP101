# Image Pyramid Exploration Guide 🏛️

> Image pyramids are like master architects! We need to construct images into hierarchical structures at different scales, just like building a pyramid. Let's explore this elegant image processing technique together!

## Table of Contents
- [1. What is an Image Pyramid?](#1-what-is-an-image-pyramid)
- [2. Gaussian Pyramid](#2-gaussian-pyramid)
- [3. Laplacian Pyramid](#3-laplacian-pyramid)
- [4. Image Blending](#4-image-blending)
- [5. SIFT Scale Space](#5-sift-scale-space)
- [6. Saliency Detection](#6-saliency-detection)
- [7. Code Implementation and Optimization](#7-code-implementation-and-optimization)
- [8. Applications and Practice](#8-applications-and-practice)

## 1. What is an Image Pyramid?

Imagine you're an image architect, building a multi-level image structure. Image pyramids help us:

- 🏗️ Build multi-scale representations (constructing different "levels" of the pyramid)
- 📏 Process different resolutions (adapting to different "viewing distances")
- 🎯 Achieve scale invariance (maintaining "structural" stability)
- 🔄 Support multi-resolution analysis (observing details from different "heights")

## 2. Gaussian Pyramid

### 2.1 Basic Principles

The Gaussian pyramid is like observing a building through a Gaussian blur lens, gradually becoming more blurred and smaller from bottom to top.

Construction steps:
1. Gaussian smoothing
2. Downsampling
3. Iterative construction

Mathematical expression:
$$
G_i(x,y) = \sum_{m=-2}^2 \sum_{n=-2}^2 w(m,n)G_{i-1}(2x+m,2y+n)
$$

Where:
- $G_i$ is the i-th level image
- $w(m,n)$ is the Gaussian kernel weight

### 2.2 Implementation Example

```cpp
vector<Mat> build_gaussian_pyramid(const Mat& src, int num_levels) {
    vector<Mat> pyramid;
    pyramid.reserve(num_levels);

    // Convert to float
    Mat current;
    src.convertTo(current, CV_32F, 1.0/255.0);
    pyramid.push_back(current);

    // Build pyramid
    for (int i = 1; i < num_levels; i++) {
        Mat next;
        // Gaussian blur
        gaussian_blur_simd(current, next, 1.0);

        // Downsample
        pyrDown(next, next);
        pyramid.push_back(next);
        current = next;
    }

    return pyramid;
}
```

## 3. Laplacian Pyramid

### 3.1 Algorithm Principles

The Laplacian pyramid is like recording the detail differences of a building, storing the difference information between each level and its reconstructed image.

Construction steps:
1. Build Gaussian pyramid
2. Calculate differences
3. Store residuals

Mathematical expression:
$$
L_i = G_i - up(G_{i+1})
$$

Where:
- $L_i$ is the i-th level Laplacian image
- $up()$ is the upsampling operation

### 3.2 Implementation Example

```cpp
vector<Mat> build_laplacian_pyramid(const Mat& src, int num_levels) {
    vector<Mat> gaussian_pyramid = build_gaussian_pyramid(src, num_levels);
    vector<Mat> laplacian_pyramid(num_levels);

    // Build Laplacian pyramid
    for (int i = 0; i < num_levels - 1; i++) {
        Mat up_level;
        pyrUp(gaussian_pyramid[i + 1], up_level, gaussian_pyramid[i].size());
        subtract(gaussian_pyramid[i], up_level, laplacian_pyramid[i]);
    }

    // Use the top level of Gaussian pyramid as the top level of Laplacian pyramid
    laplacian_pyramid[num_levels - 1] = gaussian_pyramid[num_levels - 1];

    return laplacian_pyramid;
}
```

## 4. Image Blending

### 4.1 Basic Principles

Image blending is like elegantly merging two buildings together, considering:
1. Structure alignment
2. Edge smoothing
3. Detail preservation
4. Gradient transition

### 4.2 Implementation Example

```cpp
Mat pyramid_blend(const Mat& src1, const Mat& src2,
                 const Mat& mask, int num_levels) {
    // Build Laplacian pyramids for both images
    vector<Mat> lap1 = build_laplacian_pyramid(src1, num_levels);
    vector<Mat> lap2 = build_laplacian_pyramid(src2, num_levels);

    // Build Gaussian pyramid for mask
    vector<Mat> gauss_mask = build_gaussian_pyramid(mask, num_levels);

    // Blend at each level
    vector<Mat> blend_pyramid(num_levels);
    #pragma omp parallel for
    for (int i = 0; i < num_levels; i++) {
        blend_pyramid[i] = lap1[i].mul(gauss_mask[i]) +
                          lap2[i].mul(1.0 - gauss_mask[i]);
    }

    // Reconstruct blended image
    Mat result = blend_pyramid[num_levels - 1];
    for (int i = num_levels - 2; i >= 0; i--) {
        pyrUp(result, result, blend_pyramid[i].size());
        result += blend_pyramid[i];
    }

    // Convert back to 8-bit image
    result.convertTo(result, CV_8U, 255.0);
    return result;
}
```

## 5. SIFT Scale Space

### 5.1 Algorithm Principles

SIFT scale space is like observing building features from different heights, detecting keypoints through Gaussian Difference (DoG).

DoG calculation:
$$
D(x,y,\sigma) = L(x,y,k\sigma) - L(x,y,\sigma)
$$

Where:
- $L(x,y,\sigma)$ is the Gaussian blurred image
- $k$ is the scale factor between adjacent scales

### 5.2 Implementation Example

```cpp
vector<vector<Mat>> build_sift_scale_space(
    const Mat& src, int num_octaves, int num_scales, float sigma) {

    vector<vector<Mat>> scale_space(num_octaves);
    for (auto& octave : scale_space) {
        octave.resize(num_scales);
    }

    // Initialize first level of first octave
    Mat base;
    src.convertTo(base, CV_32F, 1.0/255.0);
    gaussian_blur_simd(base, scale_space[0][0], sigma);

    // Build scale space
    float k = pow(2.0f, 1.0f / (num_scales - 3));
    for (int o = 0; o < num_octaves; o++) {
        for (int s = 1; s < num_scales; s++) {
            float sig = sigma * pow(k, s);
            gaussian_blur_simd(scale_space[o][s-1],
                             scale_space[o][s], sig);
        }

        if (o < num_octaves - 1) {
            // Downsample for the base image of next octave
            pyrDown(scale_space[o][num_scales-1],
                   scale_space[o+1][0]);
        }
    }

    return scale_space;
}
```

## 6. Saliency Detection

### 6.1 Basic Principles

Saliency detection is like finding the most eye-catching parts of a building, usually based on multi-scale feature contrast analysis.

Saliency calculation:
$$
S(x,y) = \sum_{l=1}^L w_l |I_l(x,y) - \mu_l|
$$

Where:
- $I_l$ is the l-th level image
- $\mu_l$ is the mean value of the l-th level
- $w_l$ is the weight coefficient

### 6.2 Implementation Example

```cpp
Mat saliency_detection(const Mat& src, int num_levels) {
    // Build Gaussian pyramid
    vector<Mat> pyramid = build_gaussian_pyramid(src, num_levels);

    // Calculate saliency map
    Mat saliency = Mat::zeros(src.size(), CV_32F);

    #pragma omp parallel for
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            float center_value = src.at<float>(y, x);
            float sum_diff = 0.0f;

            // Calculate differences with other scales
            for (int l = 1; l < num_levels; l++) {
                Mat& level = pyramid[l];
                float scale = static_cast<float>(src.rows) / level.rows;
                int py = static_cast<int>(y / scale);
                int px = static_cast<int>(x / scale);

                if (py >= level.rows) py = level.rows - 1;
                if (px >= level.cols) px = level.cols - 1;

                float surround_value = level.at<float>(py, px);
                sum_diff += abs(center_value - surround_value);
            }

            saliency.at<float>(y, x) = sum_diff / (num_levels - 1);
        }
    }

    // Normalize
    normalize(saliency, saliency, 0, 1, NORM_MINMAX);
    saliency.convertTo(saliency, CV_8U, 255);

    return saliency;
}
```

## 7. Code Implementation and Optimization

### 7.1 Performance Optimization Tips

1. Use integral images to accelerate computation
2. Parallel processing of multiple scale levels
3. Memory reuse
4. GPU acceleration

### 7.2 Optimization Example

```cpp
namespace {
// Internal constants
constexpr int CACHE_LINE = 64;    // CPU cache line size (bytes)
constexpr int BLOCK_SIZE = 16;    // Block processing size

// Generate Gaussian kernel
Mat create_gaussian_kernel(float sigma) {
    int kernel_size = static_cast<int>(2 * ceil(3 * sigma) + 1);
    Mat kernel(kernel_size, kernel_size, CV_32F);
    float sum = 0.0f;

    int center = kernel_size / 2;
    float sigma2 = 2 * sigma * sigma;

    #pragma omp parallel for reduction(+:sum)
    for (int y = 0; y < kernel_size; y++) {
        for (int x = 0; x < kernel_size; x++) {
            float value = exp(-((x - center) * (x - center) +
                              (y - center) * (y - center)) / sigma2);
            kernel.at<float>(y, x) = value;
            sum += value;
        }
    }

    // Normalize
    kernel /= sum;
    return kernel;
}

// Optimized Gaussian blur implementation
void gaussian_blur_simd(const Mat& src, Mat& dst, float sigma) {
    Mat kernel = create_gaussian_kernel(sigma);
    int kernel_size = kernel.rows;
    int radius = kernel_size / 2;

    dst.create(src.size(), CV_32F);

    // Horizontal pass
    Mat temp(src.size(), CV_32F);
    #pragma omp parallel for
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            float sum = 0.0f;
            for (int i = -radius; i <= radius; i++) {
                int xx = x + i;
                if (xx < 0) xx = 0;
                if (xx >= src.cols) xx = src.cols - 1;
                sum += src.at<float>(y, xx) * kernel.at<float>(0, i + radius);
            }
            temp.at<float>(y, x) = sum;
        }
    }

    // Vertical pass
    #pragma omp parallel for
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            float sum = 0.0f;
            for (int i = -radius; i <= radius; i++) {
                int yy = y + i;
                if (yy < 0) yy = 0;
                if (yy >= src.rows) yy = src.rows - 1;
                sum += temp.at<float>(yy, x) * kernel.at<float>(i + radius, 0);
            }
            dst.at<float>(y, x) = sum;
        }
    }
}
} // anonymous namespace
```

## 8. Applications and Practice

### 8.1 Typical Applications

- 🔍 Object detection
- 🎯 Feature matching
- 🖼️ Image blending
- 👁️ Saliency detection
- 🎨 Image editing

### 8.2 Practice Suggestions

1. Parameter selection
   - Number of pyramid levels
   - Gaussian kernel size
   - Scale factors

2. Performance optimization
   - Memory management
   - Parallel computing
   - GPU acceleration

3. Quality control
   - Boundary handling
   - Precision balance
   - Noise resistance

## References

1. 📚 Burt, P., & Adelson, E. (1983). The Laplacian pyramid as a compact image code.
2. 📖 Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints.
3. 🔬 Itti, L., et al. (1998). A model of saliency-based visual attention for rapid scene analysis.
4. 📊 Adelson, E. H., et al. (1984). Pyramid methods in image processing.

## Summary

Image pyramids are like architects in computer vision, through different construction methods like Gaussian and Laplacian pyramids, we can achieve multi-scale image representation. Whether used for image blending, feature extraction, or saliency detection, choosing the right pyramid method is key. We hope this tutorial helps you better understand and apply image pyramid techniques! 🏛️

> 💡 Tip: In practical applications, it's recommended to choose appropriate pyramid levels and construction methods based on specific scenarios, and pay attention to the balance between computational efficiency and detail preservation. Also, make good use of advanced techniques like image blending and SIFT scale space to excel in real projects!