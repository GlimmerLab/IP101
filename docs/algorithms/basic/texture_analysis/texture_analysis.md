# 纹理分析详解 🎨

> 纹理分析就像是给图像做"指纹识别"！每种纹理都有其独特的"指纹"，就像木纹的条纹、布料的编织、草地的随机分布一样。让我们一起来探索这个既有趣又实用的图像处理领域吧！

## 目录
- [1. 什么是纹理分析？](#1-什么是纹理分析)
- [2. 灰度共生矩阵(GLCM)](#2-灰度共生矩阵glcm)
- [3. 统计特征分析](#3-统计特征分析)
- [4. 局部二值模式(LBP)](#4-局部二值模式lbp)
- [5. Gabor纹理特征](#5-gabor纹理特征)
- [6. 纹理分类](#6-纹理分类)
- [7. 代码实现与优化](#7-代码实现与优化)
- [8. 实验结果与分析](#8-实验结果与分析)

## 1. 什么是纹理分析？

想象一下，你正在看一张木桌的照片。即使不看整体形状，你也能通过木纹的条纹认出这是木头。这就是纹理分析的魅力所在！它就像是在研究图像的"肌理"，帮助我们理解图像的细节特征。

常见的纹理类型：
- 🌳 木纹：条状排列，就像树木的年轮
- 👕 布料：规则的编织方式，就像织毛衣的针法
- 🌱 草地：随机分布，就像撒在地上的芝麻
- 🧱 砖墙：规则排列，就像乐高积木

通过分析这些"指纹"，我们可以：
- 🔍 识别不同材质（是木头还是石头？）
- ✂️ 进行图像分割（把木头和石头分开）
- 🎯 实现目标检测（找到所有的木头）
- 📊 评估表面质量（这块木头质量如何？）

## 2. 灰度共生矩阵(GLCM)

### 2.1 基本原理

GLCM就像是给图像做"像素配对"！它统计了图像中像素对的灰度关系，就像是在玩"找朋友"游戏。

举个例子：
- 如果两个像素的灰度值都是100，它们就是"好朋友"
- 如果一个是100，另一个是200，它们就是"普通朋友"
- GLCM就是统计这些"朋友关系"的频率

数学表达式：
$$
P(i,j) = \frac{\text{像素对(i,j)的数量}}{\text{总的像素对数量}}
$$

### 2.2 Haralick特征

基于GLCM，我们可以提取多种有趣的纹理特征，就像是在给纹理做"体检"：

1. 对比度(Contrast)：衡量像素对的差异程度
   - 就像是在看"朋友之间的身高差"
   - 差异越大，对比度越高
   $$
   \text{Contrast} = \sum_{i,j} |i-j|^2 P(i,j)
   $$

2. 相关性(Correlation)：衡量像素对的线性关系
   - 就像是在看"朋友之间的相似度"
   - 相关性越高，说明纹理越规则
   $$
   \text{Correlation} = \sum_{i,j} \frac{(i-\mu_i)(j-\mu_j)P(i,j)}{\sigma_i \sigma_j}
   $$

3. 能量(Energy)：衡量纹理的均匀程度
   - 就像是在看"朋友关系的稳定性"
   - 能量越高，说明纹理越均匀
   $$
   \text{Energy} = \sum_{i,j} P(i,j)^2
   $$

4. 同质性(Homogeneity)：衡量纹理的平滑程度
   - 就像是在看"朋友之间的和谐度"
   - 同质性越高，说明纹理越平滑
   $$
   \text{Homogeneity} = \sum_{i,j} \frac{P(i,j)}{1+(i-j)^2}
   $$

### 2.3 代码实现

#### C++实现
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

#### Python实现
```python
def compute_glcm(img: np.ndarray, d: int = 1, theta: int = 0) -> np.ndarray:
    """计算灰度共生矩阵(GLCM)

    Args:
        img: 输入图像
        d: 距离
        theta: 角度(0,45,90,135度)

    Returns:
        np.ndarray: GLCM矩阵
    """
    # 确保图像是灰度图
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 量化灰度级
    levels = 8
    img = (img // (256 // levels)).astype(np.uint8)

    # 创建GLCM矩阵
    glcm = np.zeros((levels, levels), dtype=np.uint32)

    # 根据角度确定偏移
    if theta == 0:
        dx, dy = d, 0
    elif theta == 45:
        dx, dy = d, -d
    elif theta == 90:
        dx, dy = 0, d
    else:  # 135度
        dx, dy = -d, d

    # 计算GLCM
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            if 0 <= i+dy < h and 0 <= j+dx < w:
                glcm[img[i,j], img[i+dy,j+dx]] += 1

    # 归一化
    glcm = glcm.astype(np.float32)
    if np.sum(glcm) > 0:
        glcm /= np.sum(glcm)

    return glcm

def extract_haralick_features(glcm: np.ndarray) -> List[float]:
    """提取Haralick特征

    Args:
        glcm: 灰度共生矩阵

    Returns:
        List[float]: Haralick特征(对比度、相关性、能量、同质性)
    """
    # 计算均值和标准差
    rows, cols = glcm.shape
    mean_i = 0
    mean_j = 0

    # 计算均值
    for i in range(rows):
        for j in range(cols):
            mean_i += i * glcm[i, j]
            mean_j += j * glcm[i, j]

    # 计算标准差
    std_i = 0
    std_j = 0
    for i in range(rows):
        for j in range(cols):
            std_i += (i - mean_i)**2 * glcm[i, j]
            std_j += (j - mean_j)**2 * glcm[i, j]

    std_i = np.sqrt(std_i)
    std_j = np.sqrt(std_j)

    # 初始化特征
    contrast = 0
    correlation = 0
    energy = 0
    homogeneity = 0

    # 计算特征
    for i in range(rows):
        for j in range(cols):
            contrast += (i - j)**2 * glcm[i, j]
            if std_i > 0 and std_j > 0:  # 防止除零
                correlation += ((i - mean_i) * (j - mean_j) * glcm[i, j]) / (std_i * std_j)
            energy += glcm[i, j]**2
            homogeneity += glcm[i, j] / (1 + (i - j)**2)

    return [contrast, correlation, energy, homogeneity]
```

## 3. 统计特征分析

### 3.1 一阶统计特征

这些特征就像是给纹理做"体检报告"，告诉我们纹理的基本情况：

1. 均值(Mean)：纹理的平均灰度值
   - 就像是在看"平均身高"
   - 反映了纹理的整体亮度
   $$
   \mu = \frac{1}{N} \sum_{i=1}^N x_i
   $$

2. 方差(Variance)：纹理的灰度变化程度
   - 就像是在看"身高差异"
   - 反映了纹理的对比度
   $$
   \sigma^2 = \frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2
   $$

3. 偏度(Skewness)：纹理的灰度分布偏斜程度
   - 就像是在看"身高分布是否对称"
   - 反映了纹理的不对称性
   $$
   \text{Skewness} = \frac{1}{N\sigma^3} \sum_{i=1}^N (x_i - \mu)^3
   $$

4. 峰度(Kurtosis)：纹理的灰度分布尖锐程度
   - 就像是在看"身高分布是否集中"
   - 反映了纹理的均匀性
   $$
   \text{Kurtosis} = \frac{1}{N\sigma^4} \sum_{i=1}^N (x_i - \mu)^4 - 3
   $$

### 3.2 代码实现

```cpp
// 计算统计特征
vector<Mat> compute_statistical_features(const Mat& src, int window_size) {
    vector<Mat> features(4);  // 均值、方差、偏度、峰度
    for(auto& feat : features) {
        feat.create(src.size(), CV_32F);
    }

    int half_size = window_size / 2;

    #pragma omp parallel for collapse(2)
    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {
            // 提取局部窗口
            Rect roi(
                max(0, j-half_size),
                max(0, i-half_size),
                min(window_size, src.cols-max(0,j-half_size)),
                min(window_size, src.rows-max(0,i-half_size))
            );
            Mat window = src(roi);

            // 计算统计特征
            double mean = compute_mean(window);
            double variance = compute_variance(window, mean);
            double std_dev = sqrt(variance);
            double skewness = compute_skewness(window, mean, std_dev);
            double kurtosis = compute_kurtosis(window, mean, std_dev);

            // 存储结果
            features[0].at<float>(i,j) = mean;
            features[1].at<float>(i,j) = variance;
            features[2].at<float>(i,j) = skewness;
            features[3].at<float>(i,j) = kurtosis;
        }
    }

    return features;
}

// 计算均值
double compute_mean(const Mat& window) {
    Scalar mean = cv::mean(window);
    return mean[0];
}

// 计算方差
double compute_variance(const Mat& window, double mean) {
    double variance = 0;
    #pragma omp parallel for reduction(+:variance)
    for (int i = 0; i < window.rows; i++) {
        for (int j = 0; j < window.cols; j++) {
            double diff = window.at<uchar>(i,j) - mean;
            variance += diff * diff;
        }
    }
    return variance / (window.rows * window.cols);
}

// 计算偏度
double compute_skewness(const Mat& window, double mean, double std_dev) {
    double skewness = 0;
    #pragma omp parallel for reduction(+:skewness)
    for (int i = 0; i < window.rows; i++) {
        for (int j = 0; j < window.cols; j++) {
            double diff = (window.at<uchar>(i,j) - mean) / std_dev;
            skewness += diff * diff * diff;
        }
    }
    return skewness / (window.rows * window.cols);
}

// 计算峰度
double compute_kurtosis(const Mat& window, double mean, double std_dev) {
    double kurtosis = 0;
    #pragma omp parallel for reduction(+:kurtosis)
    for (int i = 0; i < window.rows; i++) {
        for (int j = 0; j < window.cols; j++) {
            double diff = (window.at<uchar>(i,j) - mean) / std_dev;
            kurtosis += diff * diff * diff * diff;
        }
    }
    return kurtosis / (window.rows * window.cols) - 3.0;
}
```

## 4. 局部二值模式(LBP)

### 4.1 基本原理

LBP就像是给每个像素点做"二进制编码"！它通过比较中心像素与其邻域像素的大小关系，得到一个独特的"身份证号码"。

基本步骤：
1. 选择一个中心像素（就像选一个"班长"）
2. 将其与邻域像素比较（就像"班长"和"同学们"比身高）
3. 生成二进制编码（高个子记1，矮个子记0）
4. 计算十进制值（把二进制转换成十进制）

示意图：
```
3  7  4    1  1  1    (128+64+32+
2  6  5 -> 0     1 -> 16+4) = 244
1  9  8    0  1  1
```

### 4.2 数学表达式

对于半径为R的圆形邻域中的P个采样点：

$$
LBP_{P,R} = \sum_{p=0}^{P-1} s(g_p - g_c)2^p
$$

其中：
- $g_c$ 是中心像素的灰度值（"班长"的身高）
- $g_p$ 是邻域像素的灰度值（"同学们"的身高）
- $s(x)$ 是阶跃函数（判断谁高谁矮）：
$$
s(x) = \begin{cases}
1, & x \geq 0 \\
0, & x < 0
\end{cases}
$$

### 4.3 代码实现

#### C++实现
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

#### Python实现
```python
def compute_lbp(img: np.ndarray, radius: int = 1,
               n_points: int = 8) -> np.ndarray:
    """计算局部二值模式(LBP)

    Args:
        img: 输入图像
        radius: 半径
        n_points: 采样点数

    Returns:
        np.ndarray: LBP图像
    """
    # 确保图像是灰度图
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 创建输出图像
    h, w = img.shape
    lbp = np.zeros((h, w), dtype=np.uint8)

    # 计算采样点坐标
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    # 对每个像素计算LBP
    for i in range(radius, h-radius):
        for j in range(radius, w-radius):
            center = img[i, j]
            pattern = 0

            for k in range(n_points):
                # 双线性插值获取采样点值
                x1 = int(j + x[k])
                y1 = int(i + y[k])
                x2 = x1 + 1
                y2 = y1 + 1

                # 计算插值权重
                wx = j + x[k] - x1
                wy = i + y[k] - y1

                # 双线性插值
                val = (1-wx)*(1-wy)*img[y1,x1] + \
                      wx*(1-wy)*img[y1,x2] + \
                      (1-wx)*wy*img[y2,x1] + \
                      wx*wy*img[y2,x2]

                # 更新LBP模式
                pattern |= (val > center) << k

            lbp[i, j] = pattern

    return lbp
```

## 5. Gabor纹理特征

### 5.1 Gabor滤波器

Gabor滤波器就像是"纹理显微镜"！它可以在特定方向和尺度上观察纹理特征，就像是在用不同倍数的显微镜观察细胞。

二维Gabor滤波器的表达式：

$$
g(x,y) = \frac{1}{2\pi\sigma_x\sigma_y} \exp\left(-\frac{x'^2}{2\sigma_x^2}-\frac{y'^2}{2\sigma_y^2}\right)\cos(2\pi\frac{x'}{\lambda})
$$

其中：
- $x' = x\cos\theta + y\sin\theta$（旋转后的x坐标）
- $y' = -x\sin\theta + y\cos\theta$（旋转后的y坐标）
- $\theta$ 是方向角（显微镜的观察角度）
- $\lambda$ 是波长（观察的精细程度）
- $\sigma_x$ 和 $\sigma_y$ 是高斯包络的标准差（观察的范围大小）

### 5.2 特征提取

1. 生成Gabor滤波器组（准备不同倍数的"显微镜"）
2. 对图像进行滤波（用"显微镜"观察图像）
3. 计算响应的统计特征（记录观察结果）
4. 组合成特征向量（整理观察报告）

### 5.3 代码实现

#### C++实现
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

#### Python实现
```python
def compute_gabor_features(img: np.ndarray,
                          num_scales: int = 4,
                          num_orientations: int = 6) -> np.ndarray:
    """计算Gabor特征

    Args:
        img: 输入图像
        num_scales: 尺度数
        num_orientations: 方向数

    Returns:
        np.ndarray: Gabor特征图
    """
    # 确保图像是灰度图
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 创建Gabor滤波器组
    filters = []
    for scale in range(num_scales):
        for orientation in range(num_orientations):
            # 计算Gabor参数
            theta = orientation * np.pi / num_orientations
            sigma = 2.0 * (2 ** scale)
            lambda_ = 4.0 * (2 ** scale)

            # 创建Gabor滤波器
            kernel = cv2.getGaborKernel(
                (31, 31), sigma, theta, lambda_, 0.5, 0, ktype=cv2.CV_32F)

            filters.append(kernel)

    # 应用Gabor滤波器
    features = []
    for kernel in filters:
        filtered = cv2.filter2D(img, cv2.CV_32F, kernel)
        features.append(filtered)

    return np.array(features)
```

## 6. 纹理分类

### 6.1 基本原理

纹理分类就像是给不同的"布料"贴标签！我们需要：
1. 提取特征（测量布料的"特征"）
2. 训练分类器（学习不同布料的"特点"）
3. 预测类别（给新布料"贴标签"）

### 6.2 特征提取和选择

1. GLCM特征（布料的"纹理规律"）
2. LBP特征（布料的"局部特征"）
3. Gabor特征（布料的"多尺度特征"）
4. 统计特征（布料的"整体特征"）

### 6.3 分类算法

#### 6.3.1 K近邻(K-NN)

K-NN就像是"物以类聚"！它通过找到K个最相似的样本，用它们的多数类别作为预测结果。

数学表达式：
$$
\hat{y} = \arg\max_{c} \sum_{i=1}^K I(y_i = c)
$$

其中：
- $\hat{y}$ 是预测的类别
- $y_i$ 是第i个近邻的类别
- $I(\cdot)$ 是指示函数
- $c$ 是类别标签

#### 6.3.2 支持向量机(SVM)

SVM就像是"画一条线"！它试图找到一个最优的决策边界，使得不同类别的样本被最大间隔分开。

数学表达式：
$$
\min_{w,b} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^n \xi_i
$$

约束条件：
$$
y_i(w^T x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0
$$

其中：
- $w$ 是法向量
- $b$ 是偏置项
- $C$ 是惩罚参数
- $\xi_i$ 是松弛变量

### 6.4 代码实现

#### C++实现
```cpp
// KNN分类器
class KNNClassifier {
private:
    std::vector<std::vector<float>> train_features;
    std::vector<int> train_labels;
    int k;

public:
    KNNClassifier(int k = 5) : k(k) {}

    void train(const std::vector<std::vector<float>>& features,
              const std::vector<int>& labels) {
        train_features = features;
        train_labels = labels;
    }

    int predict(const std::vector<float>& feature) {
        std::vector<std::pair<float, int>> distances;

        #pragma omp parallel for
        for(size_t i = 0; i < train_features.size(); i++) {
            float dist = 0;
            for(size_t j = 0; j < feature.size(); j++) {
                float diff = feature[j] - train_features[i][j];
                dist += diff * diff;
            }
            distances.push_back({std::sqrt(dist), train_labels[i]});
        }

        std::sort(distances.begin(), distances.end());

        std::vector<int> votes(k);
        for(int i = 0; i < k; i++) {
            votes[distances[i].second]++;
        }

        return std::max_element(votes.begin(), votes.end()) - votes.begin();
    }
};

// SVM分类器
class SVMClassifier {
private:
    std::vector<std::vector<float>> support_vectors;
    std::vector<float> weights;
    float bias;
    float learning_rate;
    int max_iterations;

public:
    SVMClassifier(float learning_rate = 0.001, int max_iterations = 1000)
        : learning_rate(learning_rate), max_iterations(max_iterations) {}

    void train(const std::vector<std::vector<float>>& features,
              const std::vector<int>& labels) {
        int n_samples = features.size();
        int n_features = features[0].size();

        weights.resize(n_features, 0);
        bias = 0;

        for(int iter = 0; iter < max_iterations; iter++) {
            float error = 0;

            #pragma omp parallel for reduction(+:error)
            for(int i = 0; i < n_samples; i++) {
                float prediction = 0;
                for(int j = 0; j < n_features; j++) {
                    prediction += weights[j] * features[i][j];
                }
                prediction += bias;

                float label = labels[i] * 2 - 1;  // 转换为-1和1
                if(label * prediction < 1) {
                    error += 1 - label * prediction;

                    #pragma omp critical
                    {
                        for(int j = 0; j < n_features; j++) {
                            weights[j] += learning_rate * (label * features[i][j] - 0.01 * weights[j]);
                        }
                        bias += learning_rate * label;
                    }
                }
            }

            if(error == 0) break;
        }

        // 保存支持向量
        for(int i = 0; i < n_samples; i++) {
            float prediction = 0;
            for(int j = 0; j < n_features; j++) {
                prediction += weights[j] * features[i][j];
            }
            prediction += bias;

            if(std::abs(prediction) < 1) {
                support_vectors.push_back(features[i]);
            }
        }
    }

    int predict(const std::vector<float>& feature) {
        float prediction = 0;
        for(size_t i = 0; i < feature.size(); i++) {
            prediction += weights[i] * feature[i];
        }
        prediction += bias;

        return prediction > 0 ? 1 : 0;
    }
};
```

#### Python实现
```python
class KNNClassifier:
    """K近邻分类器"""
    def __init__(self, k=5):
        self.k = k
        self.train_features = None
        self.train_labels = None

    def train(self, features, labels):
        """训练模型

        参数:
            features: 训练特征
            labels: 训练标签
        """
        self.train_features = np.array(features)
        self.train_labels = np.array(labels)

    def predict(self, feature):
        """预测单个样本的类别

        参数:
            feature: 输入特征

        返回:
            predicted_label: 预测的类别
        """
        # 计算距离
        distances = np.sqrt(np.sum((self.train_features - feature) ** 2, axis=1))

        # 获取k个最近邻的索引
        k_indices = np.argsort(distances)[:self.k]

        # 获取k个最近邻的标签
        k_nearest_labels = self.train_labels[k_indices]

        # 返回出现次数最多的标签
        return np.bincount(k_nearest_labels).argmax()

class SVMClassifier:
    """支持向量机分类器"""
    def __init__(self, learning_rate=0.001, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.support_vectors = None

    def train(self, features, labels):
        """训练模型

        参数:
            features: 训练特征
            labels: 训练标签
        """
        n_samples, n_features = np.array(features).shape

        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 将标签转换为-1和1
        y = np.array(labels) * 2 - 1

        for _ in range(self.max_iterations):
            error = 0

            for i in range(n_samples):
                prediction = np.dot(self.weights, features[i]) + self.bias

                if y[i] * prediction < 1:
                    error += 1 - y[i] * prediction

                    # 更新权重和偏置
                    self.weights += self.learning_rate * (y[i] * features[i] - 0.01 * self.weights)
                    self.bias += self.learning_rate * y[i]

            if error == 0:
                break

        # 保存支持向量
        self.support_vectors = []
        for i in range(n_samples):
            prediction = np.dot(self.weights, features[i]) + self.bias
            if abs(prediction) < 1:
                self.support_vectors.append(features[i])

    def predict(self, feature):
        """预测单个样本的类别

        参数:
            feature: 输入特征

        返回:
            predicted_label: 预测的类别
        """
        prediction = np.dot(self.weights, feature) + self.bias
        return 1 if prediction > 0 else 0
```

## 7. 性能优化技巧

### 7.1 并行计算

1. 使用OpenMP进行并行计算（就像"多线程跑步"）
2. 合理设置线程数（不要"人太多挤在一起"）
3. 避免线程竞争（不要"抢跑道"）

### 7.2 内存优化

1. 使用连续内存（就像"排好队"）
2. 避免频繁的内存分配（不要"总是搬家"）
3. 使用内存池（就像"提前准备好房间"）

### 7.3 算法优化

1. 使用查找表（就像"提前背好答案"）
2. 减少重复计算（不要"重复做同一件事"）
3. 使用SIMD指令（就像"一次做多件事"）

## 8. 总结

纹理分析就像是在给图像做"指纹识别"，每种纹理都有其独特的"指纹"！通过GLCM、LBP和Gabor等方法，我们可以有效地提取和分析这些"指纹"。在实际应用中，需要根据具体场景选择合适的方法，就像选择不同的"显微镜"来观察不同的样本。

记住：好的纹理分析就像是一个经验丰富的"纹理侦探"，能够从图像的细节中发现重要的线索！🔍

## 9. 参考资料

1. Haralick R M. Statistical and structural approaches to texture[J]. Proceedings of the IEEE, 1979
2. Ojala T, et al. Multiresolution gray-scale and rotation invariant texture classification with local binary patterns[J]. IEEE TPAMI, 2002
3. OpenCV官方文档: https://docs.opencv.org/
4. 更多资源: [IP101项目主页](https://github.com/GlimmerLab/IP101)