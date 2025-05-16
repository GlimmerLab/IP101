# Image Compression Explained 📦

> Welcome to the "Compression Art Gallery" of image processing! Here, we'll learn how to be a "digital magician" who can significantly reduce file sizes while maintaining image quality through clever compression techniques. Let's start this journey of "space folding" in the digital world! 🎨

## Contents
- [1. Introduction to Image Compression](#1-introduction-to-image-compression)
- [2. Lossless Compression: Perfect Preservation](#2-lossless-compression-perfect-preservation)
- [3. JPEG Compression: Smart Compression](#3-jpeg-compression-smart-compression)
- [4. Fractal Compression: Self-Similar Compression](#4-fractal-compression-self-similar-compression)
- [5. Wavelet Compression: Multi-Scale Compression](#5-wavelet-compression-multi-scale-compression)
- [6. Practical Applications and Considerations](#6-practical-applications-and-considerations)
- [7. Performance Evaluation and Comparison](#7-performance-evaluation-and-comparison)
- [8. Summary](#8-summary)

## 1. Introduction to Image Compression

### 1.1 What is Image Compression? 🤔

Image compression is like "space management" in the digital world:
- 📦 Reduce file size (like compressing luggage volume)
- 🎯 Maintain image quality (like protecting fragile items)
- 🚀 Improve transmission efficiency (like fast shipping)
- 💾 Save storage space (like optimizing warehouse space)

### 1.2 Why Do We Need Image Compression? 💡

- 📱 Mobile storage always alerts ("Storage space running out again!")
- 🌐 Network bandwidth is never fast enough ("Why is this image still loading...")
- 💰 Storage costs need to be controlled ("Cloud storage bill exceeded again")
- ⚡ Loading speed needs to be fast enough ("Users can't wait!")

Common compression methods include:
- Lossless compression (like "perfect folding")
- JPEG compression (smart "elastic compression")
- Fractal compression (based on "self-similarity")
- Wavelet compression (multi-level "fine compression")

## 2. Lossless Compression: Perfect Preservation

Lossless compression is like the art of "perfect folding" - ensuring image quality while reducing file size. It's like folding clothes neatly, allowing them to be perfectly restored when needed! 👔

### 2.1 Run-Length Encoding (RLE)

RLE is like the "shorthand master for repetitive elements". For example, writing "5 stars" instead of "🌟🌟🌟🌟🌟" - both clear and space-efficient!

Mathematical expression:
$$
RLE(x_1^{n_1}x_2^{n_2}...x_k^{n_k}) = (x_1,n_1)(x_2,n_2)...(x_k,n_k)
$$

Where:
- $x_i$ is the pixel value
- $n_i$ is the number of consecutive occurrences

#### C++ Implementation
```cpp
double rle_encode(const Mat& src, vector<uchar>& encoded) {
    CV_Assert(!src.empty());

    // Convert to grayscale
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    encoded.clear();
    encoded.reserve(gray.total());

    uchar current = gray.at<uchar>(0, 0);
    int count = 1;

    // RLE encoding
    for (int i = 1; i < gray.total(); i++) {
        uchar pixel = gray.at<uchar>(i / gray.cols, i % gray.cols);

        if (pixel == current && count < 255) {
            count++;
        } else {
            encoded.push_back(current);
            encoded.push_back(count);
            current = pixel;
            count = 1;
        }
    }

    // Handle the last group
    encoded.push_back(current);
    encoded.push_back(count);

    return compute_compression_ratio(gray.total(), encoded.size());
}
```

#### Python Implementation
```python
def rle_compression(img_path):
    """
    Problem 47: Lossless Compression (RLE Encoding)
    Use run-length encoding for lossless compression

    Parameters:
        img_path: Input image path

    Returns:
        Reconstructed image after compression
    """
    # Read image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    # Flatten image
    flat_img = img.flatten()

    # RLE encoding
    encoded = []
    count = 1
    current = flat_img[0]

    for pixel in flat_img[1:]:
        if pixel == current:
            count += 1
        else:
            encoded.extend([current, count])
            current = pixel
            count = 1
    encoded.extend([current, count])

    # RLE decoding
    decoded = []
    for i in range(0, len(encoded), 2):
        decoded.extend([encoded[i]] * encoded[i+1])

    # Rebuild image
    result = np.array(decoded).reshape(img.shape)

    # Convert to color image
    result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    return result
```

### 2.2 Huffman Coding

Huffman coding is like "assigning short codes to commonly used items", with common values represented by short codes. It's like using abbreviations for common words and full names for uncommon ones - both saving space and being easy to understand!

#### C++ Implementation
```cpp
struct HuffmanNode {
    uchar value;
    int frequency;
    HuffmanNode* left;
    HuffmanNode* right;

    HuffmanNode(uchar v, int f) : value(v), frequency(f), left(nullptr), right(nullptr) {}
};

class HuffmanEncoder {
private:
    HuffmanNode* root;
    map<uchar, string> code_table;

    void build_code_table(HuffmanNode* node, string code) {
        if (!node) return;
        if (!node->left && !node->right) {
            code_table[node->value] = code;
            return;
        }
        build_code_table(node->left, code + "0");
        build_code_table(node->right, code + "1");
    }

public:
    void encode(const Mat& src, vector<bool>& encoded) {
        // Count frequencies
        map<uchar, int> frequency;
        for (int i = 0; i < src.total(); i++) {
            frequency[src.at<uchar>(i / src.cols, i % src.cols)]++;
        }

        // Build Huffman tree
        priority_queue<pair<int, HuffmanNode*>, vector<pair<int, HuffmanNode*>>, greater<>> pq;
        for (const auto& pair : frequency) {
            pq.push({pair.second, new HuffmanNode(pair.first, pair.second)});
        }

        while (pq.size() > 1) {
            auto left = pq.top().second; pq.pop();
            auto right = pq.top().second; pq.pop();
            auto parent = new HuffmanNode(0, left->frequency + right->frequency);
            parent->left = left;
            parent->right = right;
            pq.push({parent->frequency, parent});
        }

        root = pq.top().second;
        build_code_table(root, "");

        // Encode
        encoded.clear();
        for (int i = 0; i < src.total(); i++) {
            uchar pixel = src.at<uchar>(i / src.cols, i % src.cols);
            string code = code_table[pixel];
            for (char bit : code) {
                encoded.push_back(bit == '1');
            }
        }
    }
};
```

#### Python Implementation
```python
def huffman_encoding(data):
    """Manual implementation of Huffman encoding"""
    # Count frequencies
    frequency = collections.Counter(data)

    # Build Huffman tree
    heap = [[weight, [symbol, ""]] for symbol, weight in frequency.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    return dict(heap[0][1:])
```

## 3. JPEG Compression: Smart Compression

JPEG compression is like a "smart compression master" - it knows that the human eye is insensitive to certain details, so it can "secretly" discard some information while keeping the image looking beautiful! 🎨

### 3.1 Color Space Conversion

First, we need to convert the image from RGB to YCbCr color space. This is like breaking down the image into luminance (Y) and chrominance (Cb, Cr) components. The human eye is more sensitive to luminance than chrominance, which is the "secret weapon" of JPEG compression!

Mathematical expression:
$$
\begin{bmatrix} Y \\ Cb \\ Cr \end{bmatrix} =
\begin{bmatrix}
0.299 & 0.587 & 0.114 \\
-0.1687 & -0.3313 & 0.5 \\
0.5 & -0.4187 & -0.0813
\end{bmatrix}
\begin{bmatrix} R \\ G \\ B \end{bmatrix}
$$

#### C++ Implementation
```cpp
void rgb_to_ycbcr(const Mat& src, Mat& y, Mat& cb, Mat& cr) {
    // Separate channels
    vector<Mat> channels;
    split(src, channels);
    Mat r = channels[2], g = channels[1], b = channels[0];

    // Convert to YCbCr
    y = 0.299 * r + 0.587 * g + 0.114 * b;
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 128;
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128;
}
```

#### Python Implementation
```python
def rgb_to_ycbcr(img):
    """Manual implementation of RGB to YCbCr conversion"""
    # Separate channels
    b, g, r = cv2.split(img)

    # Convert to YCbCr
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128

    return y, cb, cr
```

### 3.2 DCT Transform

DCT transform is like doing "frequency analysis" on the image, breaking it down into different frequency "notes". Low frequencies are like the "main melody" and high frequencies are like "ornamental notes" - we need to protect the main melody!

Mathematical expression:
$$
F(u,v) = \frac{2}{N}C(u)C(v)\sum_{x=0}^{N-1}\sum_{y=0}^{N-1}f(x,y)\cos\left[\frac{(2x+1)u\pi}{2N}\right]\cos\left[\frac{(2y+1)v\pi}{2N}\right]
$$

Where:
- $C(u) = \frac{1}{\sqrt{2}}$ when $u=0$
- $C(u) = 1$ when $u>0$

#### C++ Implementation
```cpp
void dct_transform(const Mat& src, Mat& dst) {
    const int N = 8;
    dst = Mat::zeros(src.size(), CV_32F);

    for (int u = 0; u < N; u++) {
        for (int v = 0; v < N; v++) {
            float sum = 0;
            float cu = (u == 0) ? 1.0/sqrt(2) : 1.0;
            float cv = (v == 0) ? 1.0/sqrt(2) : 1.0;

            for (int x = 0; x < N; x++) {
                for (int y = 0; y < N; y++) {
                    float cos_u = cos((2*x+1)*u*M_PI/(2*N));
                    float cos_v = cos((2*y+1)*v*M_PI/(2*N));
                    sum += src.at<float>(x,y) * cos_u * cos_v;
                }
            }

            dst.at<float>(u,v) = 2.0/N * cu * cv * sum;
        }
    }
}
```

#### Python Implementation
```python
def dct_transform(block):
    """Manual implementation of DCT transform"""
    N = 8
    result = np.zeros((N, N), dtype=np.float32)

    for u in range(N):
        for v in range(N):
            sum_val = 0
            cu = 1/np.sqrt(2) if u == 0 else 1
            cv = 1/np.sqrt(2) if v == 0 else 1

            for x in range(N):
                for y in range(N):
                    cos_u = np.cos((2*x+1)*u*np.pi/(2*N))
                    cos_v = np.cos((2*y+1)*v*np.pi/(2*N))
                    sum_val += block[x,y] * cos_u * cos_v

            result[u,v] = 2/N * cu * cv * sum_val

    return result
```

### 3.3 Quantization

Quantization is the key step in JPEG compression, like a shrewd "digital accountant" 📊. We use a quantization table for "smart simplification" of DCT coefficients. It's like handling a financial report - important numbers kept to two decimal places, secondary numbers rounded, and least important numbers omitted!

The JPEG standard luminance quantization table (quality factor = 50) is like an "image slimming plan":
```
16  11  10  16  24  40  51  61  ← Preserve important information
12  12  14  19  26  58  60  55
14  13  16  24  40  57  69  56
14  17  22  29  51  87  80  62  ← Progressive compression
18  22  37  56  68 109 103  77
24  35  55  64  81 104 113  92
49  64  78  87 103 121 120 101
72  92  95  98 112 100 103  99  ← Bold compression of details
```

This quantization table design is ingeniously crafted:
- Small values in top-left: Like treating "VIP clients" by carefully preserving low-frequency information (overall structure)
- Large values in bottom-right: Like treating "temporary visitors" by boldly compressing high-frequency information (details)
- Diagonal gradient: Like designing a "smooth transition zone" for natural compression effects

The quantization process looks simple mathematically, but works remarkably well:
$$
F_Q(u,v) = round\left(\frac{F(u,v)}{Q(u,v)}\right)
$$

Each symbol in this formula plays a unique role:
- $F(u,v)$ is the DCT coefficient, like the original "digital assets"
- $Q(u,v)$ is the quantization table value, like the "compression scale"
- $F_Q(u,v)$ is the quantized coefficient, like the "simplified asset ledger"
- The $round()$ function is like a "decisive manager" making final decisions

#### C++ Implementation
```cpp
void quantize(Mat& dct_coeffs, const Mat& quant_table) {
    for (int i = 0; i < dct_coeffs.rows; i++) {
        for (int j = 0; j < dct_coeffs.cols; j++) {
            dct_coeffs.at<float>(i,j) = round(dct_coeffs.at<float>(i,j) / quant_table.at<float>(i,j));
        }
    }
}
```

#### Python Implementation
```python
def quantize(dct_coeffs, quant_table):
    """Manual implementation of quantization"""
    return np.round(dct_coeffs / quant_table)
```

#### Complete JPEG Implementation in Python
```python
def jpeg_compression(img_path, quality=50):
    """
    Problem 48: JPEG Compression
    Use DCT transform and quantization for JPEG compression

    Parameters:
        img_path: Input image path
        quality: Compression quality (1-100), default 50

    Returns:
        Reconstructed image after compression
    """
    # Read image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    # Standard JPEG quantization table
    Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 14, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]])

    # Adjust quantization table based on quality parameter
    if quality < 50:
        S = 5000 / quality
    else:
        S = 200 - 2 * quality
    Q = np.floor((S * Q + 50) / 100)
    Q = np.clip(Q, 1, 255)

    # Block processing
    h, w = img.shape
    h = h - h % 8
    w = w - w % 8
    img = img[:h, :w]
    result = np.zeros_like(img, dtype=np.float32)

    # Apply DCT transform and quantization to each 8x8 block
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = img[i:i+8, j:j+8].astype(np.float32) - 128
            dct_block = fftpack.dct(fftpack.dct(block.T, norm='ortho').T, norm='ortho')
            quantized = np.round(dct_block / Q)
            dequantized = quantized * Q
            idct_block = fftpack.idct(fftpack.idct(dequantized.T, norm='ortho').T, norm='ortho')
            result[i:i+8, j:j+8] = idct_block + 128

    # Clip to valid range
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Convert to color image
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return result
```

## 4. Fractal Compression: Self-Similar Compression

Fractal compression is like "finding self-replication in images" - it uses self-similarity in images to compress data. This is like discovering "Russian nesting dolls" where large patterns contain similar smaller patterns! 🎭

### 4.1 Basic Principles

Fractal compression is based on Iterated Function Systems (IFS), compressing data by finding self-similarities in images. It's like breaking the image into many small blocks and discovering similar relationships between these blocks.

Mathematical expression:
$$
w_i(x,y) = \begin{bmatrix} a_i & b_i \\ c_i & d_i \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} e_i \\ f_i \end{bmatrix}
$$

Where:
- $w_i$ is the affine transformation
- $a_i, b_i, c_i, d_i$ are rotation and scaling parameters
- $e_i, f_i$ are translation parameters

#### C++ Implementation
```cpp
double fractal_compress(const Mat& src, Mat& dst, int block_size) {
    CV_Assert(!src.empty());

    // Convert to grayscale
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // Adjust image size to be multiple of block_size
    int rows = ((gray.rows + block_size - 1) / block_size) * block_size;
    int cols = ((gray.cols + block_size - 1) / block_size) * block_size;
    Mat padded;
    copyMakeBorder(gray, padded, 0, rows - gray.rows, 0, cols - gray.cols, BORDER_REPLICATE);

    vector<FractalBlock> blocks;
    const int domain_step = block_size / 2;  // Domain block step size

    // Use OpenMP to accelerate block matching process
    #pragma omp parallel
    {
        vector<FractalBlock> local_blocks;

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < rows; i += block_size) {
            for (int j = 0; j < cols; j += block_size) {
                Rect range_rect(j, i, block_size, block_size);
                Mat range_block = padded(range_rect);

                double best_error = numeric_limits<double>::max();
                FractalBlock best_match;
                best_match.position = Point(j, i);
                best_match.size = Size(block_size, block_size);

                // Search for best match in domain
                for (int di = 0; di < rows - block_size*2; di += domain_step) {
                    for (int dj = 0; dj < cols - block_size*2; dj += domain_step) {
                        Mat domain_block = padded(Rect(dj, di, block_size*2, block_size*2));
                        Mat domain_small;
                        resize(domain_block, domain_small, Size(block_size, block_size));

                        double domain_mean, range_mean;
                        double domain_var, range_var;
                        compute_block_statistics(domain_small, domain_mean, domain_var);
                        compute_block_statistics(range_block, range_mean, range_var);

                        if (domain_var < 1e-6) continue;  // Skip flat areas

                        // Calculate scaling and offset coefficients
                        double scale = sqrt(range_var / domain_var);
                        double offset = range_mean - scale * domain_mean;

                        // Calculate error
                        Mat predicted = domain_small * scale + offset;
                        Mat diff = predicted - range_block;
                        double error = norm(diff, NORM_L2SQR) / (block_size * block_size);

                        if (error < best_error) {
                            best_error = error;
                            best_match.scale = scale;
                            best_match.offset = offset;
                            best_match.domain_pos = Point(dj, di);
                        }
                    }
                }

                #pragma omp critical
                blocks.push_back(best_match);
            }
        }
    }

    // Reconstruct image
    dst = Mat::zeros(padded.size(), CV_8UC1);
    for (const auto& block : blocks) {
        Mat domain_block = padded(Rect(block.domain_pos.x, block.domain_pos.y,
                                     block_size*2, block_size*2));
        Mat domain_small;
        resize(domain_block, domain_small, block.size);

        Mat range_block = domain_small * block.scale + block.offset;
        range_block.copyTo(dst(Rect(block.position.x, block.position.y,
                               block.size.width, block.size.height)));
    }

    // Crop back to original size
    dst = dst(Rect(0, 0, src.cols, src.rows));

    // Calculate compression ratio (each block stores 5 doubles: position x,y, scale, offset, domain_pos x,y)
    size_t compressed_size = blocks.size() * (sizeof(double) * 5);
    return compute_compression_ratio(src.total(), compressed_size);
}
```

#### Python Implementation
```python
def fractal_compression(img_path, block_size=8):
    """
    Problem 49: Fractal Compression
    Use fractal theory for image compression (simplified version)

    Parameters:
        img_path: Input image path
        block_size: Block size, default 8

    Returns:
        Reconstructed image after compression
    """
    # Read image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    # Ensure image dimensions are multiples of block_size
    h, w = img.shape
    h = h - h % block_size
    w = w - w % block_size
    img = img[:h, :w]
    result = np.zeros_like(img)

    # Process each block
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size]
            # Simplified fractal transform: encode using mean and standard deviation
            mean = np.mean(block)
            std = np.std(block)
            # Reconstruction: rebuild block using statistical features
            result[i:i+block_size, j:j+block_size] = np.clip(
                mean + (block - mean) * (std / (std + 1e-6)), 0, 255)

    # Convert to color image
    result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    return result
```

### 4.2 Decoding Process

The decoding process is like "growing an image from a seed" - repeatedly applying transformations to rebuild the image. It's like copying and transforming a small pattern repeatedly to eventually obtain a complete image!

#### C++ Implementation
```cpp
void FractalCompressor::decompress(Mat& dst, int iterations) {
    // Initialize random image
    dst = Mat::zeros(range_size, range_size, CV_8UC1);
    randn(dst, 128, 50);

    // Iteratively apply transformations
    for (int iter = 0; iter < iterations; iter++) {
        Mat next = Mat::zeros(dst.size(), CV_8UC1);

        for (int i = 0; i < transforms.size(); i++) {
            const auto& transform = transforms[i];
            Mat transformed;
            apply_transform(dst, transformed, transform);

            // Apply contrast and brightness
            transformed = transform.contrast * transformed + transform.brightness;

            // Copy to corresponding position
            int row = (i / (dst.cols/range_size)) * range_size;
            int col = (i % (dst.cols/range_size)) * range_size;
            transformed.copyTo(next(Rect(col, row, range_size, range_size)));
        }

        dst = next;
    }
}
```

#### Python Implementation
```python
def decompress(self, iterations=10):
    """Decompress image"""
    # Initialize random image
    img = np.random.normal(128, 50, (self.range_size, self.range_size))

    # Iteratively apply transformations
    for _ in range(iterations):
        next_img = np.zeros_like(img)

        for i, transform in enumerate(self.transforms):
            # Apply affine transformation
            transformed = self.apply_transform(img, transform['matrix'])

            # Apply contrast and brightness
            transformed = transform['contrast'] * transformed + transform['brightness']

            # Copy to corresponding position
            row = (i // (img.shape[1]//self.range_size)) * self.range_size
            col = (i % (img.shape[1]//self.range_size)) * self.range_size
            next_img[row:row+self.range_size, col:col+self.range_size] = transformed

        img = next_img

    return img
```

## 5. Wavelet Compression: Multi-Scale Compression

Wavelet compression is like "multi-level fine compression" - it uses wavelet transforms to compress data. This is like breaking down the image into wavelets of different frequencies, with high frequencies representing details and low frequencies representing the overall contour.

Mathematical expression:
$$
\psi(t) = \sum_{k=-\infty}^{\infty} h[k] \psi(2t-k)
$$

Where:
- $\psi(t)$ is the wavelet function
- $h[k]$ is the filter coefficient

#### C++ Implementation
```cpp
double wavelet_compress(const Mat& src, Mat& dst, int level, double threshold) {
    CV_Assert(!src.empty());

    // Convert to grayscale and to floating point
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }
    Mat float_img;
    gray.convertTo(float_img, CV_64F);

    // Ensure image dimensions are powers of 2
    int max_dim = max(float_img.rows, float_img.cols);
    int pad_size = 1;
    while (pad_size < max_dim) pad_size *= 2;

    Mat padded;
    copyMakeBorder(float_img, padded, 0, pad_size - float_img.rows,
                   0, pad_size - float_img.cols, BORDER_REFLECT);

    int rows = padded.rows;
    int cols = padded.cols;
    Mat temp = padded.clone();

    // Forward wavelet transform
    for (int l = 0; l < level; l++) {
        // Horizontal transform
        #pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            vector<double> row(cols);
            for (int j = 0; j < cols; j++) {
                row[j] = temp.at<double>(i, j);
            }
            wavelet_transform_1d(row);
            for (int j = 0; j < cols; j++) {
                temp.at<double>(i, j) = row[j];
            }
        }

        // Vertical transform
        #pragma omp parallel for
        for (int j = 0; j < cols; j++) {
            vector<double> col(rows);
            for (int i = 0; i < rows; i++) {
                col[i] = temp.at<double>(i, j);
            }
            wavelet_transform_1d(col);
            for (int i = 0; i < rows; i++) {
                temp.at<double>(i, j) = col[i];
            }
        }

        rows /= 2;
        cols /= 2;
    }

    // Threshold processing
    double max_coef = 0;
    for (int i = 0; i < temp.rows; i++) {
        for (int j = 0; j < temp.cols; j++) {
            max_coef = max(max_coef, abs(temp.at<double>(i, j)));
        }
    }

    double thresh = max_coef * threshold / 100.0;
    int nonzero_count = 0;

    #pragma omp parallel for reduction(+:nonzero_count)
    for (int i = 0; i < temp.rows; i++) {
        for (int j = 0; j < temp.cols; j++) {
            double& val = temp.at<double>(i, j);
            if (abs(val) < thresh) {
                val = 0;
            } else {
                nonzero_count++;
            }
        }
    }

    // Inverse wavelet transform
    rows = temp.rows;
    cols = temp.cols;
    for (int l = level - 1; l >= 0; l--) {
        rows = temp.rows >> l;
        cols = temp.cols >> l;

        // Vertical inverse transform
        #pragma omp parallel for
        for (int j = 0; j < cols; j++) {
            vector<double> col(rows);
            for (int i = 0; i < rows; i++) {
                col[i] = temp.at<double>(i, j);
            }
            wavelet_transform_1d(col, true);
            for (int i = 0; i < rows; i++) {
                temp.at<double>(i, j) = col[i];
            }
        }

        // Horizontal inverse transform
        #pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            vector<double> row(cols);
            for (int j = 0; j < cols; j++) {
                row[j] = temp.at<double>(i, j);
            }
            wavelet_transform_1d(row, true);
            for (int j = 0; j < cols; j++) {
                temp.at<double>(i, j) = row[j];
            }
        }
    }

    // Crop back to original size and convert back to 8-bit image
    Mat result = temp(Rect(0, 0, src.cols, src.rows));
    normalize(result, result, 0, 255, NORM_MINMAX);
    result.convertTo(dst, CV_8UC1);

    // Calculate compression ratio (only store non-zero coefficients)
    size_t compressed_size = nonzero_count * (sizeof(double) + sizeof(int) * 2);  // Value and position
    return compute_compression_ratio(src.total(), compressed_size);
}
```

#### Python Implementation
```python
def wavelet_compression(img_path, threshold=10):
    """
    Problem 50: Wavelet Compression
    Use wavelet transform for image compression

    Parameters:
        img_path: Input image path
        threshold: Coefficient threshold, default 10

    Returns:
        Reconstructed image after compression
    """
    # Read image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    # Perform wavelet transform
    coeffs = pywt.wavedec2(img, 'haar', level=3)

    # Threshold processing
    for i in range(1, len(coeffs)):
        for detail in coeffs[i]:
            detail[np.abs(detail) < threshold] = 0

    # Reconstruct image
    result = pywt.waverec2(coeffs, 'haar')

    # Crop to original dimensions
    result = result[:img.shape[0], :img.shape[1]]

    # Normalize to 0-255
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Convert to color image
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return result
```

## 6. Practical Applications and Considerations

### 6.1 Application Scenarios 🎯

1. Web Image Optimization
   - Speed up loading times
   - Save bandwidth
   - Improve user experience

2. Mobile Application Image Processing
   - Save storage space
   - Optimize memory usage
   - Improve application performance

3. Medical Image Compression
   - Ensure image quality
   - Reduce storage costs
   - Speed up transmission

### 6.2 Performance Optimization Tips 💪

1. Algorithm Selection
   - Choose appropriate compression methods based on needs
   - Balance compression ratio and quality
   - Consider processing speed and compression effect

2. Implementation Techniques
   - Use parallel computing for acceleration
   - Optimize memory usage
   - Avoid redundant calculations

3. Important Notes
   - Control compression quality
   - Consider image type
   - Pay attention to compression parameters

## 7. Performance Evaluation and Comparison

### 7.1 Compression Effect Comparison 📊

| Algorithm | Compression Ratio | Quality Loss | Processing Speed | Use Cases |
|-----------|------------------|--------------|------------------|-----------|
| RLE | 2:1 | None | Fast | Simple images, repeated patterns |
| JPEG | 10:1 | Medium | Fast | Natural images, photos |
| Fractal | 20:1 | High | Slow | Texture images, art pictures |
| Wavelet | 15:1 | Low | Medium | Medical images, high quality needs |

### 7.2 Quality Assessment Metrics 📈

1. Objective Metrics
   - PSNR (Peak Signal-to-Noise Ratio): Measure image quality
   - SSIM (Structural Similarity): Evaluate visual quality
   - Compression ratio: Measure compression efficiency

2. Subjective Assessment
   - Visual effect
   - Detail preservation
   - Edge clarity

### 7.3 Performance Recommendations 💡

1. Image Sharing Applications
   - Recommended: JPEG compression
   - Compression ratio: 8:1 ~ 12:1
   - Quality parameter: 75-85

2. Medical Image Storage
   - Recommended: Lossless or wavelet compression
   - Compression ratio: 2:1 ~ 4:1
   - Ensure diagnostic quality

3. Artistic Image Processing
   - Recommended: Fractal compression
   - Compression ratio: 15:1 ~ 25:1
   - Preserve texture features

## 8. Summary

Image compression is like being a "space management master" in the digital world. Through different compression techniques, we can effectively reduce file sizes while maintaining image quality. From lossless compression's perfect preservation to JPEG's smart compression, and from fractal compression's self-similarity to wavelet compression's multi-scale analysis, each method has its unique advantages and applications. 🎯

### 8.1 Algorithm Comparison

| Algorithm | Advantages | Disadvantages | Use Cases |
|-----------|------------|---------------|-----------|
| RLE | Simple implementation, lossless | Low compression ratio | Simple images, repeated patterns |
| JPEG | High compression ratio, fast | Lossy, blocking artifacts | Photos, web images |
| Fractal | Very high compression ratio | Slow compression, quality loss | Natural images, rich textures |
| Wavelet | Multi-scale analysis, good quality | Computationally complex | Medical images, high quality needs |

> 💡 Tip: In practical applications, choose compression algorithms based on specific needs. For web images, JPEG is a good choice; for medical images, consider lossless or wavelet compression; for artistic images, fractal compression might bring unexpected effects. Remember, there's no best compression algorithm, only the most suitable one!

## References

1. Sayood K. Introduction to data compression[M]. Morgan Kaufmann, 2017
2. Wallace G K. The JPEG still picture compression standard[J]. IEEE transactions on consumer electronics, 1992
3. Barnsley M F, et al. The science of fractal images[M]. Springer, 1988
4. Mallat S G. A theory for multiresolution signal decomposition[J]. TPAMI, 1989
5. OpenCV Documentation: https://docs.opencv.org/
6. More resources: [IP101 Project Homepage](https://github.com/GlimmerLab/IP101)