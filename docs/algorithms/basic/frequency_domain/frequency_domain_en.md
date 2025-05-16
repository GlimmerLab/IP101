# Frequency Domain Processing in Detail 🎵

> Welcome to the "Frequency Spectrum Concert Hall" of image processing! Here, we'll learn how to "tune" various frequency components of images like a sound engineer. Let's begin this symphony of vision and mathematics! 🎼

## Table of Contents
- [1. Introduction to Frequency Domain Processing](#1-introduction-to-frequency-domain-processing)
- [2. Fourier Transform: Image Spectrum Decomposition](#2-fourier-transform-image-spectrum-decomposition)
- [3. Frequency Domain Filtering: Frequency Adjustment](#3-frequency-domain-filtering-frequency-adjustment)
- [4. Discrete Cosine Transform: Efficient Frequency Compression](#4-discrete-cosine-transform-efficient-frequency-compression)
- [5. Wavelet Transform: Multi-scale Spectrum Analysis](#5-wavelet-transform-multi-scale-spectrum-analysis)
- [6. Practical Applications and Considerations](#6-practical-applications-and-considerations)

## 1. Introduction to Frequency Domain Processing

### 1.1 What is Frequency Domain Processing? 🤔

Frequency domain processing is like performing "spectral analysis" on images:
- 📊 Decompose images into components of different frequencies
- 🎛️ Analyze and adjust these frequency components
- 🔍 Extract specific frequency characteristics
- 🎨 Reconstruct processed images

### 1.2 Why Do We Need Frequency Domain Processing? 💡

- 👀 Some features are easier to observe and process in the frequency domain
- 🚀 Some operations are more efficient in the frequency domain
- 🎯 Can achieve processing tasks that are difficult in the spatial domain
- 📦 Provides theoretical foundation for image compression

## 2. Fourier Transform: Image Spectrum Decomposition

### 2.1 Mathematical Principles

The core idea of Fourier transform is to decompose an image into a superposition of sinusoidal waves of different frequencies:

$$
F(u,v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x,y)e^{-j2\pi(\frac{ux}{M}+\frac{vy}{N})}
$$

Where:
- $f(x,y)$ is the spatial domain image
- $F(u,v)$ is the frequency domain representation
- $M,N$ are image dimensions

### 2.2 Manual Implementation

#### C++ Implementation
```cpp
void fourier_transform_manual(const Mat& src, Mat& dst, int flags) {
    CV_Assert(!src.empty());

    // Convert to grayscale
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // Expand image to optimal DFT size
    Mat padded;
    int m = getOptimalDFTSize(gray.rows);
    int n = getOptimalDFTSize(gray.cols);
    copyMakeBorder(gray, padded, 0, m - gray.rows, 0, n - gray.cols,
                   BORDER_CONSTANT, Scalar::all(0));

    // Create complex matrix
    vector<vector<complex<double>>> complexImg(m, vector<complex<double>>(n));

    // Convert to complex and multiply by (-1)^(x+y) to center the spectrum
    #pragma omp parallel for
    for (int y = 0; y < m; y++) {
        for (int x = 0; x < n; x++) {
            double val = padded.at<uchar>(y, x);
            double sign = ((x + y) % 2 == 0) ? 1.0 : -1.0;
            complexImg[y][x] = sign * complex<double>(val, 0);
        }
    }

    // Row-wise FFT
    #pragma omp parallel for
    for (int y = 0; y < m; y++) {
        fft(complexImg[y], n, flags == DFT_INVERSE);
    }

    // Transpose matrix
    vector<vector<complex<double>>> transposed(n, vector<complex<double>>(m));
    #pragma omp parallel for
    for (int y = 0; y < m; y++) {
        for (int x = 0; x < n; x++) {
            transposed[x][y] = complexImg[y][x];
        }
    }

    // Column-wise FFT
    #pragma omp parallel for
    for (int x = 0; x < n; x++) {
        fft(transposed[x], m, flags == DFT_INVERSE);
    }

    // Transpose back to original orientation
    #pragma omp parallel for
    for (int y = 0; y < m; y++) {
        for (int x = 0; x < n; x++) {
            complexImg[y][x] = transposed[x][y];
        }
    }

    // Create output matrix
    if (flags == DFT_COMPLEX_OUTPUT) {
        vector<Mat> planes = {
            Mat::zeros(m, n, CV_64F),
            Mat::zeros(m, n, CV_64F)
        };

        #pragma omp parallel for
        for (int y = 0; y < m; y++) {
            for (int x = 0; x < n; x++) {
                planes[0].at<double>(y, x) = complexImg[y][x].real();
                planes[1].at<double>(y, x) = complexImg[y][x].imag();
            }
        }

        merge(planes, dst);
    } else {
        dst.create(m, n, CV_64F);
        #pragma omp parallel for
        for (int y = 0; y < m; y++) {
            for (int x = 0; x < n; x++) {
                dst.at<double>(y, x) = magnitude(complexImg[y][x]);
            }
        }
    }
}
```

#### Python Implementation
```python
def fourier_transform_manual(img):
    """Manual implementation of Fourier transform"""
    # Convert to grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert to float type
    img = img.astype(np.float32)

    # Get image dimensions
    rows, cols = img.shape

    # Create frequency domain matrix
    f = np.zeros((rows, cols), dtype=np.complex64)

    # Calculate Fourier transform
    for u in range(rows):
        for v in range(cols):
            sum_complex = 0
            for x in range(rows):
                for y in range(cols):
                    # Calculate exponent of e
                    e_power = -2 * np.pi * 1j * (u*x/rows + v*y/cols)
                    sum_complex += img[x,y] * np.exp(e_power)
            f[u,v] = sum_complex

    # Shift spectrum center
    f_shift = np.fft.fftshift(f)

    return f_shift
```

### 2.3 Optimization Tips 🚀

1. Use Fast Fourier Transform (FFT) algorithm
2. Utilize OpenMP for parallel computing
3. Use SIMD instruction sets for optimization
4. Proper memory alignment
5. Avoid frequent memory allocation

## 3. Frequency Domain Filtering: Frequency Adjustment

### 3.1 Filter Types

1. Low-pass Filter (preserve low frequencies, remove high frequencies):
$$
H(u,v) = \begin{cases}
1, & \text{if } D(u,v) \leq D_0 \\
0, & \text{if } D(u,v) > D_0
\end{cases}
$$

2. High-pass Filter (preserve high frequencies, remove low frequencies):
$$
H(u,v) = 1 - \exp\left(-\frac{D^2(u,v)}{2D_0^2}\right)
$$

### 3.2 Manual Implementation

```cpp
Mat create_frequency_filter(const Size& size,
                          double cutoff_freq,
                          const string& filter_type) {
    Mat filter = Mat::zeros(size, CV_64F);
    Point center(size.width/2, size.height/2);
    double radius2 = cutoff_freq * cutoff_freq;

    #pragma omp parallel for
    for (int y = 0; y < size.height; y++) {
        for (int x = 0; x < size.width; x++) {
            double distance2 = pow(x - center.x, 2) + pow(y - center.y, 2);

            if (filter_type == "lowpass") {
                filter.at<double>(y, x) = exp(-distance2 / (2 * radius2));
            } else if (filter_type == "highpass") {
                filter.at<double>(y, x) = 1.0 - exp(-distance2 / (2 * radius2));
            } else if (filter_type == "bandpass") {
                double r1 = radius2 * 0.5;  // Inner radius
                double r2 = radius2 * 2.0;  // Outer radius
                if (distance2 >= r1 && distance2 <= r2) {
                    filter.at<double>(y, x) = 1.0;
                }
            }
        }
    }

    return filter;
}
```

## 4. Discrete Cosine Transform: Efficient Frequency Compression

### 4.1 Mathematical Principles

The basic formula for DCT transform:

$$
F(u,v) = \frac{2}{\sqrt{MN}}C(u)C(v)\sum_{x=0}^{M-1}\sum_{y=0}^{N-1}f(x,y)\cos\frac{(2x+1)u\pi}{2M}\cos\frac{(2y+1)v\pi}{2N}
$$

Where:
- $C(w) = \frac{1}{\sqrt{2}}$ when $w=0$
- $C(w) = 1$ when $w>0$

### 4.2 Manual Implementation

```cpp
void dct_transform_manual(const Mat& src, Mat& dst, int flags) {
    CV_Assert(!src.empty());

    // Convert to grayscale and normalize
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }
    gray.convertTo(gray, CV_64F);

    int m = gray.rows;
    int n = gray.cols;
    dst.create(m, n, CV_64F);

    if (flags == DCT_FORWARD) {
        #pragma omp parallel for
        for (int u = 0; u < m; u++) {
            for (int v = 0; v < n; v++) {
                double cu = (u == 0) ? 1.0/sqrt(2.0) : 1.0;
                double cv = (v == 0) ? 1.0/sqrt(2.0) : 1.0;
                double sum = 0.0;

                for (int x = 0; x < m; x++) {
                    for (int y = 0; y < n; y++) {
                        double val = gray.at<double>(x, y);
                        double cos1 = cos((2*x + 1) * u * PI / (2*m));
                        double cos2 = cos((2*y + 1) * v * PI / (2*n));
                        sum += val * cos1 * cos2;
                    }
                }

                dst.at<double>(u, v) = cu * cv * sum * 2.0/sqrt(m*n);
            }
        }
    } else {  // DCT_INVERSE
        #pragma omp parallel for
        for (int x = 0; x < m; x++) {
            for (int y = 0; y < n; y++) {
                double sum = 0.0;

                for (int u = 0; u < m; u++) {
                    for (int v = 0; v < n; v++) {
                        double cu = (u == 0) ? 1.0/sqrt(2.0) : 1.0;
                        double cv = (v == 0) ? 1.0/sqrt(2.0) : 1.0;
                        double val = gray.at<double>(u, v);
                        double cos1 = cos((2*x + 1) * u * PI / (2*m));
                        double cos2 = cos((2*y + 1) * v * PI / (2*n));
                        sum += cu * cv * val * cos1 * cos2;
                    }
                }

                dst.at<double>(x, y) = sum * 2.0/sqrt(m*n);
            }
        }
    }
}
```

```python
def dct_transform_manual(img, block_size=8):
    """Manual implementation of DCT transform"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)
    h, w = img.shape
    h = h - h % block_size
    w = w - w % block_size
    img = img[:h, :w]

    result = np.zeros_like(img, dtype=np.float32)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size]

            # Calculate DCT coefficients
            dct_block = np.zeros_like(block)
            for u in range(block_size):
                for v in range(block_size):
                    cu = 1/np.sqrt(2) if u == 0 else 1
                    cv = 1/np.sqrt(2) if v == 0 else 1

                    sum_val = 0
                    for x in range(block_size):
                        for y in range(block_size):
                            cos_x = np.cos((2*x + 1) * u * np.pi / (2*block_size))
                            cos_y = np.cos((2*y + 1) * v * np.pi / (2*block_size))
                            sum_val += block[x,y] * cos_x * cos_y

                    dct_block[u,v] = 2/block_size * cu * cv * sum_val

            result[i:i+block_size, j:j+block_size] = dct_block

    return result
```

## 5. Wavelet Transform: Multi-scale Spectrum Analysis

### 5.1 Mathematical Principles

The basic formula for wavelet transform:

$$
W_\psi f(s,\tau) = \frac{1}{\sqrt{s}}\int_{-\infty}^{\infty}f(t)\psi^*(\frac{t-\tau}{s})dt
$$

Where:
- $\psi$ is the wavelet basis function
- $s$ is the scale parameter
- $\tau$ is the translation parameter

### 5.2 Manual Implementation

```python
def wavelet_transform_manual(img, level=1):
    """Manual implementation of Haar wavelet transform"""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float32)
    h, w = img.shape

    # Ensure image dimensions are powers of 2
    h_pad = 2**int(np.ceil(np.log2(h)))
    w_pad = 2**int(np.ceil(np.log2(w)))
    img_pad = np.pad(img, ((0,h_pad-h), (0,w_pad-w)), 'constant')

    def haar_transform_1d(data):
        n = len(data)
        output = np.zeros(n)

        # Calculate one level of haar transform
        h = n//2
        for i in range(h):
            output[i] = (data[2*i] + data[2*i+1])/np.sqrt(2)  # approximation coefficients
            output[i+h] = (data[2*i] - data[2*i+1])/np.sqrt(2)  # detail coefficients

        return output

    result = img_pad.copy()
    h, w = result.shape

    # Transform for each level
    for l in range(level):
        h_current = h//(2**l)
        w_current = w//(2**l)

        # Row transform
        for i in range(h_current):
            result[i,:w_current] = haar_transform_1d(result[i,:w_current])

        # Column transform
        for j in range(w_current):
            result[:h_current,j] = haar_transform_1d(result[:h_current,j])

    return result
```

## 6. Practical Applications and Considerations

### 6.1 Application Scenarios 🎯

1. Image Enhancement
   - Noise reduction
   - Edge enhancement
   - Detail extraction

2. Image Compression
   - JPEG compression
   - Video coding
   - Data storage

3. Feature Extraction
   - Texture analysis
   - Pattern recognition
   - Object detection

### 6.2 Performance Optimization Tips 💪

1. Algorithm Selection
   - Choose appropriate transform methods based on actual needs
   - Consider computational complexity and memory usage
   - Balance quality and efficiency

2. Implementation Techniques
   - Use parallel computing for acceleration
   - Properly utilize CPU cache
   - Avoid frequent memory allocation and copying

3. Important Considerations
   - Handle boundary effects
   - Consider numerical precision
   - Pay attention to data type conversion

## Summary

Frequency domain processing is like being a "sound engineer" in image processing. Through analysis and adjustment of different frequency components, we can achieve various image processing tasks. Whether using Fourier Transform, DCT Transform, or Wavelet Transform, choosing the right tools and using them correctly is key. We hope this tutorial helps you better understand and apply frequency domain processing techniques! 🎉

> 💡 Tip: In practical applications, it's recommended to start with simple frequency domain processing and gradually deepen your understanding of various transforms and their application scenarios. Also, pay attention to code optimization and efficiency to handle real projects with ease!