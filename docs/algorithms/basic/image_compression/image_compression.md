# 图像压缩详解 📦

> 欢迎来到图像处理的"压缩艺术馆"！在这里，我们将学习如何像一位"数字魔术师"一样，通过巧妙的压缩技术，在保持图像品质的同时大幅缩小文件体积。让我们开始这场数字世界的"空间折叠之旅"吧！🎨

## 目录
- [1. 图像压缩简介](#1-图像压缩简介)
- [2. 无损压缩：完美保存](#2-无损压缩完美保存)
- [3. JPEG压缩：智能压缩](#3-jpeg压缩智能压缩)
- [4. 分形压缩：自相似压缩](#4-分形压缩自相似压缩)
- [5. 小波压缩：多尺度压缩](#5-小波压缩多尺度压缩)
- [6. 实际应用与注意事项](#6-实际应用与注意事项)
- [7. 性能评估与对比](#7-性能评估与对比)
- [8. 总结](#8-总结)

## 1. 图像压缩简介

### 1.1 什么是图像压缩？ 🤔

图像压缩就像是数字世界的"空间管理"：
- 📦 减小文件大小（就像压缩行李体积）
- 🎯 保持图像质量（就像保护易碎物品）
- 🚀 提高传输效率（就像快速运输）
- 💾 节省存储空间（就像优化仓储）

### 1.2 为什么需要图像压缩？ 💡

- 📱 手机存储总是告急（"存储空间又不够了！"）
- 🌐 网络带宽永远不嫌快（"这图怎么还在加载..."）
- 💰 存储成本需要控制（"云存储账单又超支了"）
- ⚡ 加载速度要够快（"用户等不及了！"）

常见的压缩方法包括：
- 无损压缩（像是"完美折叠"）
- JPEG压缩（智能"弹性压缩"）
- 分形压缩（基于"自相似性"）
- 小波压缩（多层次"精细压缩"）

## 2. 无损压缩：完美保存

无损压缩就像是"完美折叠"的艺术，保证图像质量的同时减小文件大小。它就像是把衣服叠得整整齐齐，需要时还能完全展开恢复原样！👔

### 2.1 游程编码(RLE)

游程编码就像是"重复元素的简写大师"。比如把"🌟🌟🌟🌟🌟"简写成"5个🌟"，既清晰又节省空间！

数学表达式：
$$
RLE(x_1^{n_1}x_2^{n_2}...x_k^{n_k}) = (x_1,n_1)(x_2,n_2)...(x_k,n_k)
$$

其中：
- $x_i$ 是像素值
- $n_i$ 是连续出现次数

#### C++实现
```cpp
double rle_encode(const Mat& src, vector<uchar>& encoded) {
    CV_Assert(!src.empty());

    // 转换为灰度图
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

    // RLE编码
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

    // 处理最后一组
    encoded.push_back(current);
    encoded.push_back(count);

    return compute_compression_ratio(gray.total(), encoded.size());
}
```

#### Python实现
```python
def rle_compression(img_path):
    """
    问题47：无损压缩（RLE编码）
    使用游程编码进行无损压缩

    参数:
        img_path: 输入图像路径

    返回:
        压缩后重建的图像
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 展平图像
    flat_img = img.flatten()

    # RLE编码
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

    # RLE解码
    decoded = []
    for i in range(0, len(encoded), 2):
        decoded.extend([encoded[i]] * encoded[i+1])

    # 重建图像
    result = np.array(decoded).reshape(img.shape)

    # 转换为彩色图像
    result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    return result
```

### 2.2 霍夫曼编码

霍夫曼编码就像是"给常用物品分配短代号"，常见的值用短编码表示。这就像是我们给常用的词用简写，不常用的词用全称，既节省空间又容易理解！

#### C++实现
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
        // 统计频率
        map<uchar, int> frequency;
        for (int i = 0; i < src.total(); i++) {
            frequency[src.at<uchar>(i / src.cols, i % src.cols)]++;
        }

        // 构建霍夫曼树
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

        // 编码
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

#### Python实现
```python
def huffman_encoding(data):
    """手动实现霍夫曼编码"""
    # 统计频率
    frequency = collections.Counter(data)

    # 构建霍夫曼树
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

## 3. JPEG压缩：智能压缩

JPEG压缩就像是"智能压缩大师"，它知道人眼对某些细节不敏感，所以可以"偷偷"丢掉一些信息，但保持图像看起来依然很美！🎨

### 3.1 色彩空间转换

首先，我们需要把图像从RGB转换到YCbCr色彩空间。这就像是把图像分解成亮度(Y)和色度(Cb, Cr)两个部分。人眼对亮度更敏感，对色度不太敏感，这就是JPEG压缩的"秘密武器"！

数学表达式：
$$
\begin{bmatrix} Y \\ Cb \\ Cr \end{bmatrix} =
\begin{bmatrix}
0.299 & 0.587 & 0.114 \\
-0.1687 & -0.3313 & 0.5 \\
0.5 & -0.4187 & -0.0813
\end{bmatrix}
\begin{bmatrix} R \\ G \\ B \end{bmatrix}
$$

#### C++实现
```cpp
void rgb_to_ycbcr(const Mat& src, Mat& y, Mat& cb, Mat& cr) {
    // 分离通道
    vector<Mat> channels;
    split(src, channels);
    Mat r = channels[2], g = channels[1], b = channels[0];

    // 转换到YCbCr
    y = 0.299 * r + 0.587 * g + 0.114 * b;
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 128;
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128;
}
```

#### Python实现
```python
def rgb_to_ycbcr(img):
    """手动实现RGB到YCbCr的转换"""
    # 分离通道
    b, g, r = cv2.split(img)

    # 转换到YCbCr
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 128
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 128

    return y, cb, cr
```

### 3.2 DCT变换

DCT变换就像是给图像做"频率分析"，把图像分解成不同频率的"音符"。低频就像是"主旋律"，高频就像是"装饰音"，我们要重点保护"主旋律"！

数学表达式：
$$
F(u,v) = \frac{2}{N}C(u)C(v)\sum_{x=0}^{N-1}\sum_{y=0}^{N-1}f(x,y)\cos\left[\frac{(2x+1)u\pi}{2N}\right]\cos\left[\frac{(2y+1)v\pi}{2N}\right]
$$

其中：
- $C(u) = \frac{1}{\sqrt{2}}$ 当 $u=0$
- $C(u) = 1$ 当 $u>0$

#### C++实现
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

#### Python实现
```python
def dct_transform(block):
    """手动实现DCT变换"""
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

### 3.3 量化

量化是JPEG压缩中最关键的一步，就像是一位精明的"数字会计师" 📊。我们用一个量化表来对DCT系数进行"智能化简"，高频部分（细节）会被更大幅度地压缩。这就像是在处理财务报表时，重要的数字保留到小数点后两位，次要的数字直接取整，最不重要的数字可以省略不计！

JPEG标准的亮度量化表（质量因子=50）就像是一张"图像瘦身计划表"：
```
16  11  10  16  24  40  51  61  ← 保留重要信息
12  12  14  19  26  58  60  55
14  13  16  24  40  57  69  56
14  17  22  29  51  87  80  62  ← 渐进压缩
18  22  37  56  68 109 103  77
24  35  55  64  81 104 113  92
49  64  78  87 103 121 120 101
72  92  95  98 112 100 103  99  ← 大胆压缩细节
```

这个量化表的设计可谓是"巧夺天工"：
- 左上角的值较小：像是对待"VIP客户"一样精心保留低频信息（整体结构）
- 右下角的值较大：像是对待"临时访客"一样大胆压缩高频信息（细节）
- 对角线方向渐变：像是设计了一条"平滑过渡带"，让压缩效果自然不突兀

量化过程的数学表达式看起来很简单，但效果却出奇地好：
$$
F_Q(u,v) = round\left(\frac{F(u,v)}{Q(u,v)}\right)
$$

这个公式中的每个符号都像是在演绎不同的角色：
- $F(u,v)$ 是DCT系数，像是原始的"数字资产"
- $Q(u,v)$ 是量化表中的值，像是"压缩比例尺"
- $F_Q(u,v)$ 是量化后的系数，像是"精简后的资产账本"
- $round()$ 函数像是一位"果断的决策者"，负责最终的取舍

#### C++实现
```cpp
void quantize(Mat& dct_coeffs, const Mat& quant_table) {
    for (int i = 0; i < dct_coeffs.rows; i++) {
        for (int j = 0; j < dct_coeffs.cols; j++) {
            dct_coeffs.at<float>(i,j) = round(dct_coeffs.at<float>(i,j) / quant_table.at<float>(i,j));
        }
    }
}
```

#### Python实现
```python
def quantize(dct_coeffs, quant_table):
    """手动实现量化"""
    return np.round(dct_coeffs / quant_table)
```

#### 完整的JPEG实现(Python)
```python
def jpeg_compression(img_path, quality=50):
    """
    问题48：JPEG压缩
    使用DCT变换和量化进行JPEG压缩

    参数:
        img_path: 输入图像路径
        quality: 压缩质量(1-100)，默认50

    返回:
        压缩后重建的图像
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 标准JPEG量化表
    Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                  [12, 12, 14, 19, 26, 58, 60, 55],
                  [14, 13, 16, 24, 40, 57, 69, 56],
                  [14, 17, 22, 29, 51, 87, 80, 62],
                  [18, 22, 37, 56, 68, 109, 103, 77],
                  [24, 35, 55, 64, 81, 104, 113, 92],
                  [49, 64, 78, 87, 103, 121, 120, 101],
                  [72, 92, 95, 98, 112, 100, 103, 99]])

    # 调整量化表根据质量参数
    if quality < 50:
        S = 5000 / quality
    else:
        S = 200 - 2 * quality
    Q = np.floor((S * Q + 50) / 100)
    Q = np.clip(Q, 1, 255)

    # 分块处理
    h, w = img.shape
    h = h - h % 8
    w = w - w % 8
    img = img[:h, :w]
    result = np.zeros_like(img, dtype=np.float32)

    # 对每个8x8块进行DCT变换和量化
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = img[i:i+8, j:j+8].astype(np.float32) - 128
            dct_block = fftpack.dct(fftpack.dct(block.T, norm='ortho').T, norm='ortho')
            quantized = np.round(dct_block / Q)
            dequantized = quantized * Q
            idct_block = fftpack.idct(fftpack.idct(dequantized.T, norm='ortho').T, norm='ortho')
            result[i:i+8, j:j+8] = idct_block + 128

    # 裁剪到有效范围
    result = np.clip(result, 0, 255).astype(np.uint8)

    # 转换为彩色图像
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return result
```

## 4. 分形压缩：自相似压缩

分形压缩就像是"寻找图像中的自我复制"，它利用图像中存在的自相似性来压缩数据。这就像是发现图像中的"俄罗斯套娃"，大图案中藏着相似的小图案！🎭

### 4.1 基本原理

分形压缩基于迭代函数系统(IFS)，通过寻找图像中的自相似性来实现压缩。这就像是把图像分解成许多小块，然后发现这些小块之间存在着相似关系。

数学表达式：
$$
w_i(x,y) = \begin{bmatrix} a_i & b_i \\ c_i & d_i \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} e_i \\ f_i \end{bmatrix}
$$

其中：
- $w_i$ 是仿射变换
- $a_i, b_i, c_i, d_i$ 是旋转和缩放参数
- $e_i, f_i$ 是平移参数

#### C++实现
```cpp
double fractal_compress(const Mat& src, Mat& dst, int block_size) {
    CV_Assert(!src.empty());

    // 转换为灰度图
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }

    // 调整图像大小为block_size的倍数
    int rows = ((gray.rows + block_size - 1) / block_size) * block_size;
    int cols = ((gray.cols + block_size - 1) / block_size) * block_size;
    Mat padded;
    copyMakeBorder(gray, padded, 0, rows - gray.rows, 0, cols - gray.cols, BORDER_REPLICATE);

    vector<FractalBlock> blocks;
    const int domain_step = block_size / 2;  // 定义域块步长

    // 使用OpenMP加速块匹配过程
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

                // 在定义域中搜索最佳匹配
                for (int di = 0; di < rows - block_size*2; di += domain_step) {
                    for (int dj = 0; dj < cols - block_size*2; dj += domain_step) {
                        Mat domain_block = padded(Rect(dj, di, block_size*2, block_size*2));
                        Mat domain_small;
                        resize(domain_block, domain_small, Size(block_size, block_size));

                        double domain_mean, range_mean;
                        double domain_var, range_var;
                        compute_block_statistics(domain_small, domain_mean, domain_var);
                        compute_block_statistics(range_block, range_mean, range_var);

                        if (domain_var < 1e-6) continue;  // 跳过平坦区域

                        // 计算缩放和偏移系数
                        double scale = sqrt(range_var / domain_var);
                        double offset = range_mean - scale * domain_mean;

                        // 计算误差
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

    // 重构图像
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

    // 裁剪回原始大小
    dst = dst(Rect(0, 0, src.cols, src.rows));

    // 计算压缩率（每个块存储5个double：位置x,y，scale，offset，domain_pos x,y）
    size_t compressed_size = blocks.size() * (sizeof(double) * 5);
    return compute_compression_ratio(src.total(), compressed_size);
}
```

#### Python实现
```python
def fractal_compression(img_path, block_size=8):
    """
    问题49：分形压缩
    使用分形理论进行图像压缩（简化版本）

    参数:
        img_path: 输入图像路径
        block_size: 分块大小，默认为8

    返回:
        压缩后重建的图像
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 确保图像尺寸是block_size的整数倍
    h, w = img.shape
    h = h - h % block_size
    w = w - w % block_size
    img = img[:h, :w]
    result = np.zeros_like(img)

    # 对每个块进行处理
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size]
            # 简化的分形变换：使用均值和方差进行编码
            mean = np.mean(block)
            std = np.std(block)
            # 重建：使用统计特征重建块
            result[i:i+block_size, j:j+block_size] = np.clip(
                mean + (block - mean) * (std / (std + 1e-6)), 0, 255)

    # 转换为彩色图像
    result = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    return result
```

### 4.2 解码过程

解码过程就像是"从种子生长出图像"，通过反复应用变换来重建图像。这就像是把一个小图案不断复制、变换，最终得到完整的图像！

#### C++实现
```cpp
void FractalCompressor::decompress(Mat& dst, int iterations) {
    // 初始化随机图像
    dst = Mat::zeros(range_size, range_size, CV_8UC1);
    randn(dst, 128, 50);

    // 迭代应用变换
    for (int iter = 0; iter < iterations; iter++) {
        Mat next = Mat::zeros(dst.size(), CV_8UC1);

        for (int i = 0; i < transforms.size(); i++) {
            const auto& transform = transforms[i];
            Mat transformed;
            apply_transform(dst, transformed, transform);

            // 应用对比度和亮度
            transformed = transform.contrast * transformed + transform.brightness;

            // 复制到对应位置
            int row = (i / (dst.cols/range_size)) * range_size;
            int col = (i % (dst.cols/range_size)) * range_size;
            transformed.copyTo(next(Rect(col, row, range_size, range_size)));
        }

        dst = next;
    }
}
```

#### Python实现
```python
def decompress(self, iterations=10):
    """解压缩图像"""
    # 初始化随机图像
    img = np.random.normal(128, 50, (self.range_size, self.range_size))

    # 迭代应用变换
    for _ in range(iterations):
        next_img = np.zeros_like(img)

        for i, transform in enumerate(self.transforms):
            # 应用仿射变换
            transformed = self.apply_transform(img, transform['matrix'])

            # 应用对比度和亮度
            transformed = transform['contrast'] * transformed + transform['brightness']

            # 复制到对应位置
            row = (i // (img.shape[1]//self.range_size)) * self.range_size
            col = (i % (img.shape[1]//self.range_size)) * self.range_size
            next_img[row:row+self.range_size, col:col+self.range_size] = transformed

        img = next_img

    return img
```

## 5. 小波压缩：多尺度压缩

小波压缩就像是"多层次的精细压缩"，它利用小波变换来压缩数据。这就像是把图像分解成不同频率的小波，高频部分代表细节，低频部分代表整体轮廓。

数学表达式：
$$
\psi(t) = \sum_{k=-\infty}^{\infty} h[k] \psi(2t-k)
$$

其中：
- $\psi(t)$ 是小波函数
- $h[k]$ 是滤波器系数

#### C++实现
```cpp
double wavelet_compress(const Mat& src, Mat& dst, int level, double threshold) {
    CV_Assert(!src.empty());

    // 转换为灰度图并转换为浮点型
    Mat gray;
    if (src.channels() == 3) {
        cvtColor(src, gray, COLOR_BGR2GRAY);
    } else {
        gray = src.clone();
    }
    Mat float_img;
    gray.convertTo(float_img, CV_64F);

    // 确保图像尺寸是2的幂次
    int max_dim = max(float_img.rows, float_img.cols);
    int pad_size = 1;
    while (pad_size < max_dim) pad_size *= 2;

    Mat padded;
    copyMakeBorder(float_img, padded, 0, pad_size - float_img.rows,
                   0, pad_size - float_img.cols, BORDER_REFLECT);

    int rows = padded.rows;
    int cols = padded.cols;
    Mat temp = padded.clone();

    // 前向小波变换
    for (int l = 0; l < level; l++) {
        // 水平方向变换
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

        // 垂直方向变换
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

    // 阈值处理
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

    // 反向小波变换
    rows = temp.rows;
    cols = temp.cols;
    for (int l = level - 1; l >= 0; l--) {
        rows = temp.rows >> l;
        cols = temp.cols >> l;

        // 垂直方向逆变换
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

        // 水平方向逆变换
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

    // 裁剪回原始大小并转换回8位图像
    Mat result = temp(Rect(0, 0, src.cols, src.rows));
    normalize(result, result, 0, 255, NORM_MINMAX);
    result.convertTo(dst, CV_8UC1);

    // 计算压缩率（只存储非零系数）
    size_t compressed_size = nonzero_count * (sizeof(double) + sizeof(int) * 2);  // 值和位置
    return compute_compression_ratio(src.total(), compressed_size);
}
```

#### Python实现
```python
def wavelet_compression(img_path, threshold=10):
    """
    问题50：小波压缩
    使用小波变换进行图像压缩

    参数:
        img_path: 输入图像路径
        threshold: 系数阈值，默认为10

    返回:
        压缩后重建的图像
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 进行小波变换
    coeffs = pywt.wavedec2(img, 'haar', level=3)

    # 阈值处理
    for i in range(1, len(coeffs)):
        for detail in coeffs[i]:
            detail[np.abs(detail) < threshold] = 0

    # 重建图像
    result = pywt.waverec2(coeffs, 'haar')

    # 裁剪到原始尺寸
    result = result[:img.shape[0], :img.shape[1]]

    # 归一化到0-255
    result = np.clip(result, 0, 255).astype(np.uint8)

    # 转换为彩色图像
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return result
```

## 6. 实际应用与注意事项

### 6.1 应用场景 🎯

1. 网页图片优化
   - 加快加载速度
   - 节省带宽
   - 提升用户体验

2. 移动应用图片处理
   - 节省存储空间
   - 优化内存占用
   - 提升应用性能

3. 医学图像压缩
   - 保证图像质量
   - 减少存储成本
   - 加快传输速度

### 6.2 性能优化建议 💪

1. 算法选择
   - 根据实际需求选择合适的压缩方法
   - 考虑压缩率和质量平衡
   - 权衡处理速度和压缩效果

2. 实现技巧
   - 使用并行计算加速处理
   - 优化内存使用
   - 避免重复计算

3. 注意事项
   - 控制压缩质量
   - 考虑图像类型
   - 注意压缩参数设置

## 7. 性能评估与对比

### 7.1 压缩效果对比 📊

| 算法 | 压缩率 | 质量损失 | 处理速度 | 适用场景 |
|------|--------|----------|----------|----------|
| RLE | 2:1 | 无 | 快 | 简单图像，重复图案多 |
| JPEG | 10:1 | 中等 | 快 | 自然图像，照片 |
| 分形 | 20:1 | 较大 | 慢 | 纹理图像，艺术图片 |
| 小波 | 15:1 | 小 | 中等 | 医学图像，高质量需求 |

### 7.2 质量评估指标 📈

1. 客观指标
   - PSNR (峰值信噪比): 衡量图像质量
   - SSIM (结构相似性): 评估视觉质量
   - 压缩比: 衡量压缩效率

2. 主观评估
   - 视觉效果
   - 细节保留
   - 边缘清晰度

### 7.3 性能建议 💡

1. 图片分享类应用
   - 推荐: JPEG压缩
   - 压缩率: 8:1 ~ 12:1
   - 质量参数: 75-85

2. 医学影像存储
   - 推荐: 无损压缩或小波压缩
   - 压缩率: 2:1 ~ 4:1
   - 保证诊断质量

3. 艺术图片处理
   - 推荐: 分形压缩
   - 压缩率: 15:1 ~ 25:1
   - 保留纹理特征

## 8. 总结

图像压缩就像是数字世界的"空间管理大师"，通过不同的压缩技术，我们可以在保持图像质量的同时有效减小文件大小。从无损压缩的完美保真，到JPEG的智能压缩，再到分形压缩的自相似性利用，每种方法都有其独特的优势和应用场景。🎯

### 8.1 算法对比

| 算法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| RLE | 实现简单，无损压缩 | 压缩率低 | 简单图像，重复图案多 |
| JPEG | 压缩率高，处理快 | 有损压缩，块效应 | 照片，网页图片 |
| 分形 | 压缩率极高 | 压缩慢，质量损失大 | 自然图像，纹理丰富 |
| 小波 | 多尺度分析，质量好 | 计算复杂 | 医学图像，需要高质量 |

> 💡 小贴士：在实际应用中，建议根据具体需求选择合适的压缩算法。对于网页图片，JPEG是不错的选择；对于医学图像，可以考虑无损压缩或小波压缩；对于艺术图片，分形压缩可能会带来意想不到的效果。记住，没有最好的压缩算法，只有最适合的算法！

## 参考资料

1. Sayood K. Introduction to data compression[M]. Morgan Kaufmann, 2017
2. Wallace G K. The JPEG still picture compression standard[J]. IEEE transactions on consumer electronics, 1992
3. Barnsley M F, et al. The science of fractal images[M]. Springer, 1988
4. Mallat S G. A theory for multiresolution signal decomposition[J]. TPAMI, 1989
5. OpenCV官方文档: https://docs.opencv.org/
6. 更多资源: [IP101项目主页](https://github.com/GlimmerLab/IP101)