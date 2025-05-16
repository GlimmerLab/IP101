# 🎨 Color Operations in Detail

> 🌟 In the world of image processing, color operations are like a magician's basic skills. Today, let's unlock these fascinating and practical "magic tricks"!

## 📚 Table of Contents

1. [Channel Swap - The RGB and BGR "Switching" Game](#channel-swap)
2. [Grayscale - The Art of Color "Fading"](#grayscale)
3. [Thresholding - A Black and White World](#thresholding)
4. [Otsu's Method - The Smart Eye for Finding the Best Threshold](#otsu)
5. [HSV Transform - Exploring a More Natural Color Space](#hsv)

## 🔄 Channel Swap
<a name="channel-swap"></a>

### Theoretical Foundation
In computer vision, we often encounter two color formats: RGB and BGR. They're like the order of names in different cultures - surname first or last. 😄

For a color image $I$, its RGB channels can be represented as:

$$
I_{RGB} = \begin{bmatrix}
R & G & B
\end{bmatrix}
$$

The channel swap operation can be represented as a matrix transformation:

$$
I_{BGR} = I_{RGB} \begin{bmatrix}
0 & 0 & 1 \\
0 & 1 & 0 \\
1 & 0 & 0
\end{bmatrix}
$$

### Implementation

#### C++ Implementation
```cpp
/**
 * @brief Channel swap implementation
 * @param src Input image
 * @param dst Output image
 * @param r_idx Red channel index
 * @param g_idx Green channel index
 * @param b_idx Blue channel index
 */
void channel_swap(const Mat& src, Mat& dst, int r_idx, int g_idx, int b_idx) {
    CV_Assert(!src.empty() && src.type() == CV_8UC3);
    CV_Assert(r_idx >= 0 && r_idx < 3 && g_idx >= 0 && g_idx < 3 && b_idx >= 0 && b_idx < 3);

    dst.create(src.size(), src.type());

    #pragma omp parallel for
    for (int y = 0; y < src.rows; ++y) {
        for (int col = 0; col < src.cols; ++col) {
            const Vec3b& pixel = src.at<Vec3b>(y, col);
            dst.at<Vec3b>(y, col) = Vec3b(pixel[b_idx], pixel[g_idx], pixel[r_idx]);
        }
    }
}
```

#### Python Implementation
```python
def channel_swap(img_path):
    """
    Problem 1: Channel Swap
    Change RGB channel order to BGR

    Parameters:
        img_path: Input image path

    Returns:
        Processed image
    """
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    # Split channels
    b, g, r = cv2.split(img)

    # Recombine channels (BGR -> RGB)
    result = cv2.merge([r, g, b])

    return result
```

## 🌫️ Grayscale
<a name="grayscale"></a>

### Theoretical Foundation
Converting a color image to grayscale is like turning an oil painting into a sketch. We use weighted averaging because human eyes have different sensitivities to different colors.

The standard RGB to grayscale conversion formula:

$$
Y = 0.2126R + 0.7152G + 0.0722B
$$

This formula comes from the ITU-R BT.709 standard, considering human eye sensitivity to different wavelengths. A more general form is:

$$
Y = \sum_{i \in \{R,G,B\}} w_i \cdot C_i
$$

where $w_i$ are weight coefficients and $C_i$ are corresponding color channel values.

### Implementation

#### C++ Implementation
```cpp
/**
 * @brief Grayscale conversion implementation
 * @param src Input image
 * @param dst Output image
 * @param method Grayscale method, options: weighted, average, max, min
 */
void to_gray(const Mat& src, Mat& dst, const std::string& method) {
    CV_Assert(!src.empty() && src.type() == CV_8UC3);

    dst.create(src.size(), CV_8UC1);

    if (method == "weighted") {
        #pragma omp parallel for
        for (int y = 0; y < src.rows; ++y) {
            for (int col = 0; col < src.cols; ++col) {
                const Vec3b& pixel = src.at<Vec3b>(y, col);
                dst.at<uchar>(y, col) = saturate_cast<uchar>(
                    GRAY_WEIGHT_B * pixel[0] +
                    GRAY_WEIGHT_G * pixel[1] +
                    GRAY_WEIGHT_R * pixel[2]
                );
            }
        }
    } else if (method == "average") {
        #pragma omp parallel for
        for (int y = 0; y < src.rows; ++y) {
            for (int col = 0; col < src.cols; ++col) {
                const Vec3b& pixel = src.at<Vec3b>(y, col);
                dst.at<uchar>(y, col) = saturate_cast<uchar>(
                    (pixel[0] + pixel[1] + pixel[2]) / 3.0f
                );
            }
        }
    } else if (method == "max") {
        #pragma omp parallel for
        for (int y = 0; y < src.rows; ++y) {
            for (int col = 0; col < src.cols; ++col) {
                const Vec3b& pixel = src.at<Vec3b>(y, col);
                dst.at<uchar>(y, col) = std::max({pixel[0], pixel[1], pixel[2]});
            }
        }
    } else if (method == "min") {
        #pragma omp parallel for
        for (int y = 0; y < src.rows; ++y) {
            for (int col = 0; col < src.cols; ++col) {
                const Vec3b& pixel = src.at<Vec3b>(y, col);
                dst.at<uchar>(y, col) = std::min({pixel[0], pixel[1], pixel[2]});
            }
        }
    } else {
        throw std::invalid_argument("Unsupported grayscale method: " + method);
    }
}
```

#### Python Implementation
```python
def grayscale(img_path):
    """
    Problem 2: Grayscale
    Convert a color image to grayscale

    Parameters:
        img_path: Input image path

    Returns:
        Grayscale image
    """
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    # Split channels
    b, g, r = cv2.split(img)

    # Calculate grayscale value (Y = 0.2126R + 0.7152G + 0.0722B)
    result = 0.2126 * r + 0.7152 * g + 0.0722 * b
    result = result.astype(np.uint8)

    return result
```

## ⚫⚪ Thresholding
<a name="thresholding"></a>

### Theoretical Foundation
Thresholding is like giving an image an ultimatum: it's either black or white, no middle ground!

Mathematical expression:

$$
g(x,y) = \begin{cases}
255, & \text{if } f(x,y) > T \\
0, & \text{if } f(x,y) \leq T
\end{cases}
$$

where:
- $f(x,y)$ is the input image grayscale value at point $(x,y)$
- $g(x,y)$ is the output image value at point $(x,y)$
- $T$ is the threshold value

### Implementation

#### C++ Implementation
```cpp
/**
 * @brief Thresholding implementation
 * @param src Input image
 * @param dst Output image
 * @param threshold Threshold value
 * @param max_value Maximum value
 * @param method Thresholding method, options: binary, binary_inv, trunc, tozero, tozero_inv
 */
void threshold_image(const Mat& src, Mat& dst, double threshold, double max_value, const std::string& method) {
    CV_Assert(!src.empty() && (src.type() == CV_8UC1 || src.type() == CV_8UC3));

    // Convert to grayscale if input is color image
    Mat gray;
    if (src.type() == CV_8UC3) {
        to_gray(src, gray);
    } else {
        gray = src;
    }

    dst.create(gray.size(), CV_8UC1);

    int thresh_type;
    if (method == "binary") {
        thresh_type = THRESH_BINARY;
    } else if (method == "binary_inv") {
        thresh_type = THRESH_BINARY_INV;
    } else if (method == "trunc") {
        thresh_type = THRESH_TRUNC;
    } else if (method == "tozero") {
        thresh_type = THRESH_TOZERO;
    } else if (method == "tozero_inv") {
        thresh_type = THRESH_TOZERO_INV;
    } else {
        throw std::invalid_argument("Unsupported threshold method: " + method);
    }

    cv::threshold(gray, dst, threshold, max_value, thresh_type);
}
```

#### Python Implementation
```python
def thresholding(img_path, th=128):
    """
    Problem 3: Thresholding
    Convert a grayscale image to binary image

    Parameters:
        img_path: Input image path
        th: Threshold value, default is 128

    Returns:
        Binary image
    """
    # Read image as grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    # Manual thresholding implementation
    result = np.zeros_like(img)
    result[img > th] = 255

    return result
```

## 🎯 Otsu's Method
<a name="otsu"></a>

### Theoretical Foundation
Otsu's method is like a "smart judge" that can automatically find the optimal threshold. It achieves this by maximizing the between-class variance.

The between-class variance formula:

$$
\sigma^2_B(t) = \omega_0(t)\omega_1(t)[\mu_0(t) - \mu_1(t)]^2
$$

where:
- $\omega_0(t)$ is the probability of foreground pixels
- $\omega_1(t)$ is the probability of background pixels
- $\mu_0(t)$ is the mean grayscale value of foreground pixels
- $\mu_1(t)$ is the mean grayscale value of background pixels

The optimal threshold selection:

$$
t^* = \arg\max_{t} \{\sigma^2_B(t)\}
$$

### Implementation

#### C++ Implementation
```cpp
/**
 * @brief Otsu thresholding implementation
 * @param src Input image
 * @param dst Output image
 * @param max_value Maximum value
 * @return Calculated optimal threshold
 */
double otsu_threshold(const Mat& src, Mat& dst, double max_value) {
    CV_Assert(!src.empty() && (src.type() == CV_8UC1 || src.type() == CV_8UC3));

    // Convert to grayscale if input is color image
    Mat gray;
    if (src.type() == CV_8UC3) {
        to_gray(src, gray);
    } else {
        gray = src;
    }

    // Calculate histogram
    int histogram[256] = {0};
    for (int y = 0; y < gray.rows; ++y) {
        for (int col = 0; col < gray.cols; ++col) {
            histogram[gray.at<uchar>(y, col)]++;
        }
    }

    // Calculate total pixels
    int total = gray.rows * gray.cols;

    // Calculate optimal threshold
    double sum = 0;
    for (int i = 0; i < 256; ++i) {
        sum += i * histogram[i];
    }

    double sumB = 0;
    int wB = 0;
    int wF = 0;
    double maxVariance = 0;
    double threshold = 0;

    for (int t = 0; t < 256; ++t) {
        wB += histogram[t];
        if (wB == 0) continue;

        wF = total - wB;
        if (wF == 0) break;

        sumB += t * histogram[t];
        double mB = sumB / wB;
        double mF = (sum - sumB) / wF;

        double variance = wB * wF * (mB - mF) * (mB - mF);
        if (variance > maxVariance) {
            maxVariance = variance;
            threshold = t;
        }
    }

    // Apply threshold
    cv::threshold(gray, dst, threshold, max_value, THRESH_BINARY);

    return threshold;
}
```

#### Python Implementation
```python
def otsu_thresholding(img_path):
    """
    Problem 4: Otsu's Method
    Adaptive thresholding using Otsu's algorithm

    Parameters:
        img_path: Input image path

    Returns:
        Binary image
    """
    # Read image as grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    # Calculate histogram
    hist = np.histogram(img, bins=256, range=(0, 256))[0]

    # Calculate total pixels
    total = img.size

    # Calculate cumulative sum and mean
    sum_total = np.sum(hist * np.arange(256))
    sum_back = 0
    w_back = 0
    w_fore = 0
    max_variance = 0
    threshold = 0

    # Iterate through all possible thresholds
    for t in range(256):
        w_back += hist[t]
        if w_back == 0:
            continue

        w_fore = total - w_back
        if w_fore == 0:
            break

        sum_back += t * hist[t]

        # Calculate means
        mean_back = sum_back / w_back
        mean_fore = (sum_total - sum_back) / w_fore

        # Calculate variance
        variance = w_back * w_fore * (mean_back - mean_fore) ** 2

        if variance > max_variance:
            max_variance = variance
            threshold = t

    # Apply threshold
    result = np.zeros_like(img)
    result[img > threshold] = 255

    return result
```

## 🌈 HSV Transform
<a name="hsv"></a>

### Theoretical Foundation
HSV color space better matches human color perception, like turning RGB from a "tech geek" into an "artist".

RGB to HSV conversion formulas:

$$
V = \max(R,G,B)
$$

$$
S = \begin{cases}
\frac{V-\min(R,G,B)}{V}, & \text{if } V \neq 0 \\
0, & \text{if } V = 0
\end{cases}
$$

$$
H = \begin{cases}
60(G-B)/\Delta, & \text{if } V = R \\
120 + 60(B-R)/\Delta, & \text{if } V = G \\
240 + 60(R-G)/\Delta, & \text{if } V = B
\end{cases}
$$

where $\Delta = V - \min(R,G,B)$

### Implementation

#### C++ Implementation
```cpp
/**
 * @brief RGB to HSV conversion implementation
 * @param src Input image
 * @param dst Output image
 */
void bgr_to_hsv(const Mat& src, Mat& dst) {
    CV_Assert(!src.empty() && src.type() == CV_8UC3);

    dst.create(src.size(), CV_8UC3);

    #pragma omp parallel for
    for (int y = 0; y < src.rows; ++y) {
        for (int col = 0; col < src.cols; ++col) {
            const Vec3b& bgr = src.at<Vec3b>(y, col);
            float b = bgr[0] / 255.0f;
            float g = bgr[1] / 255.0f;
            float r = bgr[2] / 255.0f;

            float max_val = std::max({r, g, b});
            float min_val = std::min({r, g, b});
            float diff = max_val - min_val;

            // Calculate H
            float h = 0;
            if (diff > 0) {
                if (max_val == r) {
                    h = 60.0f * (g - b) / diff;
                } else if (max_val == g) {
                    h = 60.0f * (b - r) / diff + 120.0f;
                } else {
                    h = 60.0f * (r - g) / diff + 240.0f;
                }
            }
            if (h < 0) h += 360.0f;

            // Calculate S
            float s = max_val > 0 ? diff / max_val : 0;

            // Calculate V
            float v = max_val;

            // Convert to OpenCV HSV format
            dst.at<Vec3b>(y, col) = Vec3b(
                saturate_cast<uchar>(h / 2.0f),      // H: 0-180
                saturate_cast<uchar>(s * 255.0f),    // S: 0-255
                saturate_cast<uchar>(v * 255.0f)     // V: 0-255
            );
        }
    }
}
```

#### Python Implementation
```python
def hsv_transform(img_path):
    """
    Problem 5: HSV Transform
    Convert RGB image to HSV color space

    Parameters:
        img_path: Input image path

    Returns:
        HSV image
    """
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    # Normalize to [0,1] range
    img = img.astype(np.float32) / 255.0

    # Split channels
    b, g, r = cv2.split(img)

    # Calculate max and min values
    max_val = np.maximum(np.maximum(r, g), b)
    min_val = np.minimum(np.minimum(r, g), b)

    # Calculate difference
    diff = max_val - min_val

    # Calculate H channel
    h = np.zeros_like(max_val)
    # When max_val equals min_val, h=0
    mask = (diff != 0)
    # When max_val equals r
    mask_r = (max_val == r) & mask
    h[mask_r] = 60 * ((g[mask_r] - b[mask_r]) / diff[mask_r] % 6)
    # When max_val equals g
    mask_g = (max_val == g) & mask
    h[mask_g] = 60 * ((b[mask_g] - r[mask_g]) / diff[mask_g] + 2)
    # When max_val equals b
    mask_b = (max_val == b) & mask
    h[mask_b] = 60 * ((r[mask_b] - g[mask_b]) / diff[mask_b] + 4)
    # Handle negative values
    h[h < 0] += 360

    # Calculate S channel
    s = np.zeros_like(max_val)
    s[max_val != 0] = diff[max_val != 0] / max_val[max_val != 0]

    # Calculate V channel
    v = max_val

    # Merge channels
    h = (h / 2).astype(np.uint8)  # H range in OpenCV is [0,180]
    s = (s * 255).astype(np.uint8)
    v = (v * 255).astype(np.uint8)

    result = cv2.merge([h, s, v])

    return result
```

## 📝 Practical Tips

### 1. Data Type Conversion Notes
- ⚠️ Prevent data overflow
- 🔍 Watch for precision loss
- 💾 Consider memory usage

### 2. Performance Optimization Tips
- 🚀 Use vectorized operations
- 💻 Utilize CPU SIMD instructions
- 🔄 Minimize unnecessary memory copies

### 3. Common Pitfalls
- 🕳️ Handle division by zero
- 🌡️ Check boundary conditions

## 🎓 Quiz

1. Why does green have the highest weight in RGB to grayscale conversion?
2. What's the core idea of Otsu's method?
3. What advantages does HSV color space have over RGB?

<details>
<summary>👉 Click to see answers</summary>

1. Because human eyes are most sensitive to green
2. Maximize between-class variance to best separate foreground and background
3. It better matches human perception of color, making color selection and adjustment more intuitive
</details>

## 🔗 Related Algorithms

- [Image Enhancement](../image_enhancement_en.md)
- [Edge Detection](../edge_detection_en.md)
- [Feature Extraction](../feature_extraction_en.md)

---

> 💡 Remember: Color operations are the foundation of image processing. Master these operations, and you've mastered the magic of the color palette!