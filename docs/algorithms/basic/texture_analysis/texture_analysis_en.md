# Texture Analysis in Detail 🎨

> Texture analysis is like doing "fingerprint recognition" for images! Each texture has its unique "fingerprint", just like the stripes of wood grain, the weaving pattern of fabric, or the random distribution of grass. Let's explore this fascinating and practical field of image processing together!

## Table of Contents
- [1. What is Texture Analysis?](#1-what-is-texture-analysis)
- [2. Gray Level Co-occurrence Matrix (GLCM)](#2-gray-level-co-occurrence-matrix-glcm)
- [3. Statistical Feature Analysis](#3-statistical-feature-analysis)
- [4. Local Binary Pattern (LBP)](#4-local-binary-pattern-lbp)
- [5. Gabor Texture Features](#5-gabor-texture-features)
- [6. Implementation and Optimization](#6-implementation-and-optimization)
- [7. Experimental Results and Analysis](#7-experimental-results-and-analysis)

## 1. What is Texture Analysis?

Imagine you're looking at a photo of a wooden table. Even without seeing its overall shape, you can recognize it's wood just by its grain pattern. That's the magic of texture analysis! It's like studying the "texture fingerprint" of images, helping us understand their detailed features.

Common texture types:
- 🌳 Wood grain: Striped arrangement, like tree rings
- 👕 Fabric: Regular weaving pattern, like knitting patterns
- 🌱 Grass: Random distribution, like scattered sesame seeds
- 🧱 Brick wall: Regular arrangement, like Lego blocks

By analyzing these "fingerprints", we can:
- 🔍 Identify different materials (Is it wood or stone?)
- ✂️ Perform image segmentation (Separate wood from stone)
- 🎯 Implement object detection (Find all wooden objects)
- 📊 Evaluate surface quality (How good is this piece of wood?)

## 2. Gray Level Co-occurrence Matrix (GLCM)

### 2.1 Basic Principles

GLCM is like playing a "pixel pairing" game! It analyzes the gray-level relationships between pixel pairs, just like finding friends in a crowd.

For example:
- If two pixels both have a gray value of 100, they're "best friends"
- If one is 100 and the other is 200, they're "casual friends"
- GLCM counts the frequency of these "friendships"

Mathematical expression:
$$
P(i,j) = \frac{\text{Number of pixel pairs (i,j)}}{\text{Total number of pixel pairs}}
$$

### 2.2 Haralick Features

Using GLCM, we can extract various interesting texture features, like giving the texture a "health check":

1. Contrast: Measures the intensity difference between pixel pairs
   - Like measuring the "height difference between friends"
   - Higher contrast means bigger differences
   $$
   \text{Contrast} = \sum_{i,j} |i-j|^2 P(i,j)
   $$

2. Correlation: Measures the linear relationship between pixel pairs
   - Like measuring the "similarity between friends"
   - Higher correlation means more regular texture
   $$
   \text{Correlation} = \sum_{i,j} \frac{(i-\mu_i)(j-\mu_j)P(i,j)}{\sigma_i \sigma_j}
   $$

3. Energy: Measures the uniformity of texture
   - Like measuring the "stability of friendships"
   - Higher energy means more uniform texture
   $$
   \text{Energy} = \sum_{i,j} P(i,j)^2
   $$

4. Homogeneity: Measures the smoothness of texture
   - Like measuring the "harmony between friends"
   - Higher homogeneity means smoother texture
   $$
   \text{Homogeneity} = \sum_{i,j} \frac{P(i,j)}{1+(i-j)^2}
   $$

### 2.3 Implementation

#### C++ Implementation
```cpp
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
```

#### Python Implementation
```python
def compute_glcm(img: np.ndarray, d: int = 1, theta: int = 0) -> np.ndarray:
    """Compute Gray Level Co-occurrence Matrix (GLCM)

    Args:
        img: Input image
        d: Distance
        theta: Angle (0, 45, 90, 135 degrees)

    Returns:
        np.ndarray: GLCM matrix
    """
    # Ensure grayscale image
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Quantize gray levels
    levels = 8
    img = (img // (256 // levels)).astype(np.uint8)

    # Create GLCM matrix
    glcm = np.zeros((levels, levels), dtype=np.uint32)

    # Determine offsets based on angle
    if theta == 0:
        dx, dy = d, 0
    elif theta == 45:
        dx, dy = d, -d
    elif theta == 90:
        dx, dy = 0, d
    else:  # 135 degrees
        dx, dy = -d, d

    # Calculate GLCM
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            if 0 <= i+dy < h and 0 <= j+dx < w:
                glcm[img[i,j], img[i+dy,j+dx]] += 1

    # Normalize
    glcm = glcm.astype(np.float32)
    if np.sum(glcm) > 0:
        glcm /= np.sum(glcm)

    return glcm

def extract_haralick_features(glcm: np.ndarray) -> List[float]:
    """Extract Haralick features

    Args:
        glcm: Gray Level Co-occurrence Matrix

    Returns:
        List[float]: Haralick features (contrast, correlation, energy, homogeneity)
    """
    # Calculate mean and standard deviation
    rows, cols = glcm.shape
    mean_i = 0
    mean_j = 0

    # Calculate means
    for i in range(rows):
        for j in range(cols):
            mean_i += i * glcm[i, j]
            mean_j += j * glcm[i, j]

    # Calculate standard deviations
    std_i = 0
    std_j = 0
    for i in range(rows):
        for j in range(cols):
            std_i += (i - mean_i)**2 * glcm[i, j]
            std_j += (j - mean_j)**2 * glcm[i, j]

    std_i = np.sqrt(std_i)
    std_j = np.sqrt(std_j)

    # Initialize features
    contrast = 0
    correlation = 0
    energy = 0
    homogeneity = 0

    # Calculate features
    for i in range(rows):
        for j in range(cols):
            contrast += (i - j)**2 * glcm[i, j]
            if std_i > 0 and std_j > 0:  # Prevent division by zero
                correlation += ((i - mean_i) * (j - mean_j) * glcm[i, j]) / (std_i * std_j)
            energy += glcm[i, j]**2
            homogeneity += glcm[i, j] / (1 + (i - j)**2)

    return [contrast, correlation, energy, homogeneity]
```

## 3. Statistical Feature Analysis

### 3.1 First-Order Statistics

These features are like giving the texture a "health report", telling us its basic conditions:

1. Mean: Average gray value of texture
   - Like measuring "average height"
   - Reflects overall brightness
   $$
   \mu = \frac{1}{N} \sum_{i=1}^N x_i
   $$

2. Variance: Degree of gray value variation
   - Like measuring "height differences"
   - Reflects texture contrast
   $$
   \sigma^2 = \frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2
   $$

3. Skewness: Asymmetry of gray value distribution
   - Like checking if "height distribution is symmetric"
   - Reflects texture asymmetry
   $$
   \text{Skewness} = \frac{1}{N\sigma^3} \sum_{i=1}^N (x_i - \mu)^3
   $$

4. Kurtosis: Peakedness of gray value distribution
   - Like checking if "heights are clustered"
   - Reflects texture uniformity
   $$
   \text{Kurtosis} = \frac{1}{N\sigma^4} \sum_{i=1}^N (x_i - \mu)^4 - 3
   $$

### 3.2 Implementation Tips

```cpp
// Using OpenMP to accelerate statistical calculations, like "multi-thread running"
#pragma omp parallel for reduction(+:sum)
for(int i = 0; i < window.rows; i++) {
    for(int j = 0; j < window.cols; j++) {
        sum += window.at<uchar>(i,j);
    }
}
```

## 4. Local Binary Pattern (LBP)

### 4.1 Basic Principles

LBP is like giving each pixel a "binary ID card"! It creates a unique binary code by comparing the center pixel with its neighbors.

Basic steps:
1. Select a center pixel (like choosing a "class monitor")
2. Compare it with neighboring pixels (like comparing heights with classmates)
3. Generate binary code (tall ones get 1, short ones get 0)
4. Calculate decimal value (convert binary to decimal)

Illustration:
```
3  7  4    1  1  1    (128+64+32+
2  6  5 -> 0     1 -> 16+4) = 244
1  9  8    0  1  1
```

### 4.2 Mathematical Expression

For P sampling points in a circular neighborhood of radius R:

$$
LBP_{P,R} = \sum_{p=0}^{P-1} s(g_p - g_c)2^p
$$

where:
- $g_c$ is the gray value of center pixel (the "monitor's height")
- $g_p$ is the gray value of neighboring pixels (the "classmates' heights")
- $s(x)$ is the step function (who's taller?):
$$
s(x) = \begin{cases}
1, & x \geq 0 \\
0, & x < 0
\end{cases}
$$

### 4.3 Implementation

#### C++ Implementation
```cpp
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
```

#### Python Implementation
```python
def compute_lbp(img: np.ndarray, radius: int = 1,
               n_points: int = 8) -> np.ndarray:
    """Compute Local Binary Pattern (LBP)

    Args:
        img: Input image
        radius: Radius
        n_points: Number of sampling points

    Returns:
        np.ndarray: LBP image
    """
    # Ensure grayscale image
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create output image
    h, w = img.shape
    lbp = np.zeros((h, w), dtype=np.uint8)

    # Calculate sampling point coordinates
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    # Calculate LBP for each pixel
    for i in range(radius, h-radius):
        for j in range(radius, w-radius):
            center = img[i, j]
            pattern = 0

            for k in range(n_points):
                # Bilinear interpolation to get sampling point value
                x1 = int(j + x[k])
                y1 = int(i + y[k])
                x2 = x1 + 1
                y2 = y1 + 1

                # Calculate interpolation weights
                wx = j + x[k] - x1
                wy = i + y[k] - y1

                # Bilinear interpolation
                val = (1-wx)*(1-wy)*img[y1,x1] + \
                      wx*(1-wy)*img[y1,x2] + \
                      (1-wx)*wy*img[y2,x1] + \
                      wx*wy*img[y2,x2]

                # Update LBP pattern
                pattern |= (val > center) << k

            lbp[i, j] = pattern

    return lbp
```

## 5. Gabor Texture Features

### 5.1 Gabor Filter

Gabor filter is like a "texture microscope"! It can observe texture features at specific orientations and scales, like using different magnification levels to study cells.

The expression for 2D Gabor filter:

$$
g(x,y) = \frac{1}{2\pi\sigma_x\sigma_y} \exp\left(-\frac{x'^2}{2\sigma_x^2}-\frac{y'^2}{2\sigma_y^2}\right)\cos(2\pi\frac{x'}{\lambda})
$$

where:
- $x' = x\cos\theta + y\sin\theta$ (rotated x-coordinate)
- $y' = -x\sin\theta + y\cos\theta$ (rotated y-coordinate)
- $\theta$ is orientation angle (microscope viewing angle)
- $\lambda$ is wavelength (observation detail level)
- $\sigma_x$ and $\sigma_y$ are standard deviations (observation range)

### 5.2 Feature Extraction

1. Generate Gabor filter bank (prepare different "microscopes")
2. Filter the image (observe with microscopes)
3. Calculate statistical features (record observations)
4. Combine into feature vector (organize observation report)

### 5.3 Implementation

#### C++ Implementation
```cpp
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
```

#### Python Implementation
```python
def compute_gabor_features(img: np.ndarray,
                          num_scales: int = 4,
                          num_orientations: int = 6) -> np.ndarray:
    """Compute Gabor features

    Args:
        img: Input image
        num_scales: Number of scales
        num_orientations: Number of orientations

    Returns:
        np.ndarray: Gabor feature maps
    """
    # Ensure grayscale image
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create Gabor filter bank
    filters = []
    for scale in range(num_scales):
        for orientation in range(num_orientations):
            # Calculate Gabor parameters
            theta = orientation * np.pi / num_orientations
            sigma = 2.0 * (2 ** scale)
            lambda_ = 4.0 * (2 ** scale)

            # Create Gabor filter
            kernel = cv2.getGaborKernel(
                (31, 31), sigma, theta, lambda_, 0.5, 0, ktype=cv2.CV_32F)

            filters.append(kernel)

    # Apply Gabor filters
    features = []
    for kernel in filters:
        filtered = cv2.filter2D(img, cv2.CV_32F, kernel)
        features.append(filtered)

    return np.array(features)
```

## 6. Texture Classification

### 6.1 Basic Principles

Texture classification is like labeling different "fabrics"! We need to:
1. Extract features (measure the "characteristics" of the fabric)
2. Train classifiers (learn the "features" of different fabrics)
3. Predict categories (label new fabrics)

### 6.2 Feature Extraction and Selection

1. GLCM features (fabric's "texture patterns")
2. LBP features (fabric's "local features")
3. Gabor features (fabric's "multi-scale features")
4. Statistical features (fabric's "overall features")

### 6.3 Classification Algorithms

#### 6.3.1 K-Nearest Neighbors (K-NN)

K-NN is like "birds of a feather flock together"! It finds K most similar samples and uses their majority class as the prediction.

Mathematical expression:
$$
\hat{y} = \arg\max_{c} \sum_{i=1}^K I(y_i = c)
$$

where:
- $\hat{y}$ is the predicted class
- $y_i$ is the class of the i-th neighbor
- $I(\cdot)$ is the indicator function
- $c$ is the class label

#### 6.3.2 Support Vector Machine (SVM)

SVM is like "drawing a line"! It tries to find an optimal decision boundary that maximizes the margin between different classes.

Mathematical expression:
$$
\min_{w,b} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^n \xi_i
$$

Constraints:
$$
y_i(w^T x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0
$$

where:
- $w$ is the normal vector
- $b$ is the bias term
- $C$ is the penalty parameter
- $\xi_i$ is the slack variable

### 6.4 Code Implementation

## 7. Code Implementation and Optimization

### 7.1 Parallel Computing

1. Using OpenMP for parallel computation (like "multi-thread running")
2. Setting appropriate thread count (don't "crowd together")
3. Avoiding thread competition (don't "race for the track")

### 7.2 Memory Optimization

1. Using contiguous memory (like "lining up")
2. Avoiding frequent memory allocation (don't "move house often")
3. Using memory pools (like "preparing rooms in advance")

### 7.3 Algorithm Optimization

1. Using lookup tables (like "memorizing answers in advance")
2. Reducing redundant calculations (don't "repeat the same thing")
3. Using SIMD instructions (like "doing multiple things at once")

## 8. Summary

Texture analysis is like performing "fingerprint identification" on images, where each texture has its unique "fingerprint"! Through methods like GLCM, LBP, and Gabor, we can effectively extract and analyze these "fingerprints". In practical applications, we need to choose appropriate methods based on specific scenarios, just like selecting different "microscopes" to observe different samples.

Remember: Good texture analysis is like an experienced "texture detective" who can discover important clues from image details! 🔍

## 9. References

1. Haralick R M. Statistical and structural approaches to texture[J]. Proceedings of the IEEE, 1979
2. Ojala T, et al. Multiresolution gray-scale and rotation invariant texture classification with local binary patterns[J]. IEEE TPAMI, 2002
3. OpenCV Official Documentation: https://docs.opencv.org/
4. More Resources: [IP101 Project Homepage](https://github.com/GlimmerLab/IP101)