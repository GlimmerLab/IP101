# 图像金字塔代码实现指南 🏔️

本文档提供了图像金字塔算法的Python和C++完整实现代码。每个实现都包含了详细的注释说明和参数解释。

## 目录
- [1. Python实现](#1-python实现)
  - [1.1 高斯金字塔](#11-高斯金字塔)
  - [1.2 拉普拉斯金字塔](#12-拉普拉斯金字塔)
  - [1.3 图像融合](#13-图像融合)
  - [1.4 SIFT尺度空间](#14-sift尺度空间)
  - [1.5 显著性检测](#15-显著性检测)
- [2. C++实现](#2-c实现)
  - [2.1 高斯金字塔](#21-高斯金字塔)
  - [2.2 拉普拉斯金字塔](#22-拉普拉斯金字塔)
  - [2.3 图像融合](#23-图像融合)
  - [2.4 SIFT尺度空间](#24-sift尺度空间)
  - [2.5 显著性检测](#25-显著性检测)

## 1. Python实现

### 1.1 高斯金字塔

```python
import numpy as np
import cv2

def gaussian_kernel(size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    生成高斯核

    参数:
        size: int, 核大小，默认5
        sigma: float, 标准差，默认1.0

    返回:
        np.ndarray: 高斯核
    """
    kernel = np.zeros((size, size))
    center = size // 2

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x**2 + y**2)/(2*sigma**2))

    return kernel / kernel.sum()

def manual_conv2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    手动实现2D卷积

    参数:
        img: np.ndarray, 输入图像
        kernel: np.ndarray, 卷积核

    返回:
        np.ndarray: 卷积结果
    """
    h, w = img.shape
    k_h, k_w = kernel.shape
    pad_h = k_h // 2
    pad_w = k_w // 2

    # 填充图像
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros_like(img)

    # 执行卷积
    for i in range(h):
        for j in range(w):
            output[i, j] = np.sum(padded[i:i+k_h, j:j+k_w] * kernel)

    return output

def manual_resize(img: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    手动实现图像缩放

    参数:
        img: np.ndarray, 输入图像
        scale_factor: float, 缩放因子

    返回:
        np.ndarray: 缩放后的图像
    """
    if len(img.shape) == 3:
        h, w, c = img.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        resized = np.zeros((new_h, new_w, c))

        for k in range(c):
            for i in range(new_h):
                for j in range(new_w):
                    src_i = min(int(i / scale_factor), h-1)
                    src_j = min(int(j / scale_factor), w-1)
                    resized[i, j, k] = img[src_i, src_j, k]
    else:
        h, w = img.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        resized = np.zeros((new_h, new_w))

        for i in range(new_h):
            for j in range(new_w):
                src_i = min(int(i / scale_factor), h-1)
                src_j = min(int(j / scale_factor), w-1)
                resized[i, j] = img[src_i, src_j]

    return resized

def gaussian_pyramid(img_path: str, levels: int = 4) -> np.ndarray:
    """
    构建图像的高斯金字塔

    参数:
        img_path: str, 输入图像路径
        levels: int, 金字塔层数，默认4

    返回:
        np.ndarray: 高斯金字塔可视化结果
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 转换为灰度图
    if len(img.shape) == 3:
        gray = np.mean(img, axis=2).astype(np.uint8)
    else:
        gray = img

    # 创建高斯核
    kernel = gaussian_kernel()

    # 构建金字塔
    pyramid = [gray]
    current = gray.copy()

    for _ in range(levels-1):
        # 高斯滤波
        filtered = manual_conv2d(current, kernel)
        # 下采样
        downsampled = manual_resize(filtered, 0.5)
        pyramid.append(downsampled)
        current = downsampled

    # 可视化结果
    result = []
    for level in pyramid:
        # 将图像调整为相同大小以便显示
        resized = manual_resize(level, (pyramid[0].shape[1]/level.shape[1]))
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        result.append(resized)

    return np.hstack(result)
```

### 1.2 拉普拉斯金字塔

```python
def laplacian_pyramid(img_path: str, levels: int = 4) -> np.ndarray:
    """
    构建图像的拉普拉斯金字塔

    参数:
        img_path: str, 输入图像路径
        levels: int, 金字塔层数，默认4

    返回:
        np.ndarray: 拉普拉斯金字塔可视化结果
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 转换为灰度图
    if len(img.shape) == 3:
        gray = np.mean(img, axis=2).astype(np.uint8)
    else:
        gray = img

    # 创建高斯核
    kernel = gaussian_kernel()

    # 构建高斯金字塔
    gaussian_pyr = [gray]
    current = gray.copy()

    for _ in range(levels-1):
        filtered = manual_conv2d(current, kernel)
        downsampled = manual_resize(filtered, 0.5)
        gaussian_pyr.append(downsampled)
        current = downsampled

    # 构建拉普拉斯金字塔
    laplacian_pyr = []
    for i in range(levels-1):
        # 上采样
        upsampled = manual_resize(gaussian_pyr[i+1], 2.0)
        # 调整大小以匹配
        if upsampled.shape[0] > gaussian_pyr[i].shape[0]:
            upsampled = upsampled[:gaussian_pyr[i].shape[0], :]
        if upsampled.shape[1] > gaussian_pyr[i].shape[1]:
            upsampled = upsampled[:, :gaussian_pyr[i].shape[1]]
        # 计算差分
        diff = gaussian_pyr[i] - upsampled
        laplacian_pyr.append(diff)

    # 添加最后一层
    laplacian_pyr.append(gaussian_pyr[-1])

    # 可视化结果
    result = []
    for level in laplacian_pyr:
        # 将图像调整为相同大小以便显示
        resized = manual_resize(level, (laplacian_pyr[0].shape[1]/level.shape[1]))
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        result.append(resized)

    return np.hstack(result)
```

### 1.3 图像融合

```python
def image_blending(img_path1: str, img_path2: str, levels: int = 4) -> np.ndarray:
    """
    使用金字塔进行图像融合

    参数:
        img_path1: str, 第一张输入图像路径
        img_path2: str, 第二张输入图像路径
        levels: int, 金字塔层数，默认4

    返回:
        np.ndarray: 融合结果
    """
    # 读取图像
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    if img1 is None or img2 is None:
        raise ValueError("无法读取图像")

    # 确保两张图像大小相同
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    def build_laplacian_pyramid(img):
        # 构建高斯金字塔
        gaussian_pyr = [img]
        current = img.copy()
        for _ in range(levels-1):
            current = cv2.pyrDown(current)
            gaussian_pyr.append(current)

        # 构建拉普拉斯金字塔
        laplacian_pyr = []
        for i in range(levels-1):
            size = (gaussian_pyr[i].shape[1], gaussian_pyr[i].shape[0])
            upsampled = cv2.pyrUp(gaussian_pyr[i+1], dstsize=size)
            laplacian = cv2.subtract(gaussian_pyr[i], upsampled)
            laplacian_pyr.append(laplacian)
        laplacian_pyr.append(gaussian_pyr[-1])
        return laplacian_pyr

    # 构建两个图像的拉普拉斯金字塔
    lap_pyr1 = build_laplacian_pyramid(img1)
    lap_pyr2 = build_laplacian_pyramid(img2)

    # 融合金字塔
    blended_pyr = []
    for lap1, lap2 in zip(lap_pyr1, lap_pyr2):
        # 在中间位置融合
        rows, cols = lap1.shape[:2]
        mask = np.zeros((rows, cols, 3))
        mask[:, :cols//2] = 1
        blended = lap1 * mask + lap2 * (1 - mask)
        blended_pyr.append(blended)

    # 重建融合图像
    result = blended_pyr[-1]
    for i in range(levels-2, -1, -1):
        size = (blended_pyr[i].shape[1], blended_pyr[i].shape[0])
        result = cv2.pyrUp(result, dstsize=size)
        result = cv2.add(result, blended_pyr[i])

    return result
```

### 1.4 SIFT尺度空间

```python
def sift_scale_space(img_path: str, octaves: int = 4, scales: int = 5, sigma: float = 1.6) -> np.ndarray:
    """
    构建SIFT算法的尺度空间

    参数:
        img_path: str, 输入图像路径
        octaves: int, 组数，默认4
        scales: int, 每组的尺度数，默认5
        sigma: float, 初始sigma值，默认1.6

    返回:
        np.ndarray: 尺度空间可视化结果
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 图像上采样
    img = cv2.resize(img, None, fx=2, fy=2)

    # 初始化尺度空间
    scale_space = []
    k = 2 ** (1/scales)  # 尺度因子

    # 构建尺度空间
    for o in range(octaves):
        octave = []
        if o == 0:
            base = cv2.GaussianBlur(img, (0, 0), sigma)
        else:
            base = cv2.resize(scale_space[-1][-3], None, fx=0.5, fy=0.5)

        current_sigma = sigma
        octave.append(base)

        # 生成每个尺度的图像
        for s in range(scales-1):
            current_sigma *= k
            blurred = cv2.GaussianBlur(base, (0, 0), current_sigma)
            octave.append(blurred)

        scale_space.append(octave)
        base = octave[-3]  # 为下一组准备基础图像

    # 可视化结果
    result = []
    for octave in scale_space:
        # 调整每个尺度的大小以便显示
        resized_octave = []
        for img in octave:
            size = (scale_space[0][0].shape[1], scale_space[0][0].shape[0])
            resized = cv2.resize(img, size)
            resized_octave.append(resized)
        result.append(np.hstack(resized_octave))

    return np.vstack(result)
```

### 1.5 显著性检测

```python
def saliency_detection(img_path: str, levels: int = 4) -> np.ndarray:
    """
    基于金字塔的显著性检测

    参数:
        img_path: str, 输入图像路径
        levels: int, 金字塔层数，默认4

    返回:
        np.ndarray: 显著性检测结果
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 转换到Lab颜色空间
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # 构建高斯金字塔
    pyramids = []
    for i in range(3):  # 对每个通道
        channel = lab[:, :, i]
        pyramid = [channel]
        for _ in range(levels-1):
            channel = cv2.pyrDown(channel)
            pyramid.append(channel)
        pyramids.append(pyramid)

    # 计算显著性图
    saliency = np.zeros_like(img[:, :, 0], dtype=np.float32)
    for i in range(3):  # 对每个通道
        channel_saliency = np.zeros_like(saliency)
        for level in range(levels):
            # 调整当前层的大小
            size = (img.shape[1], img.shape[0])
            resized = cv2.resize(pyramids[i][level], size)
            # 计算与平均值的差异
            mean = np.mean(resized)
            channel_saliency += np.abs(resized - mean)
        saliency += channel_saliency

    # 归一化
    saliency = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX)
    saliency = saliency.astype(np.uint8)

    # 应用颜色映射以便可视化
    saliency_color = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)

    # 将原图和显著性图并排显示
    result = np.hstack([img, saliency_color])

    return result
```

## 2. C++实现

### 2.1 高斯金字塔

```cpp
#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat gaussianPyramid(const std::string& imgPath, int levels = 4) {
    // 读取图像
    cv::Mat img = cv::imread(imgPath);
    if (img.empty()) {
        throw std::runtime_error("无法读取图像");
    }

    // 转换为灰度图
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // 构建金字塔
    std::vector<cv::Mat> pyramid;
    pyramid.push_back(gray);
    cv::Mat current = gray.clone();

    for (int i = 0; i < levels-1; i++) {
        cv::Mat down;
        cv::pyrDown(current, down);
        pyramid.push_back(down);
        current = down;
    }

    // 可视化结果
    std::vector<cv::Mat> resized;
    for (const auto& level : pyramid) {
        cv::Mat temp;
        cv::resize(level, temp, pyramid[0].size());
        cv::Mat colored;
        cv::cvtColor(temp, colored, cv::COLOR_GRAY2BGR);
        resized.push_back(colored);
    }

    // 水平拼接所有层
    cv::Mat result;
    cv::hconcat(resized, result);

    return result;
}
```

### 2.2 拉普拉斯金字塔

```cpp
cv::Mat laplacianPyramid(const std::string& imgPath, int levels = 4) {
    // 读取图像
    cv::Mat img = cv::imread(imgPath);
    if (img.empty()) {
        throw std::runtime_error("无法读取图像");
    }

    // 转换为灰度图
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // 构建高斯金字塔
    std::vector<cv::Mat> gaussianPyr;
    gaussianPyr.push_back(gray);
    cv::Mat current = gray.clone();

    for (int i = 0; i < levels-1; i++) {
        cv::Mat down;
        cv::pyrDown(current, down);
        gaussianPyr.push_back(down);
        current = down;
    }

    // 构建拉普拉斯金字塔
    std::vector<cv::Mat> laplacianPyr;
    for (int i = 0; i < levels-1; i++) {
        cv::Mat up;
        cv::pyrUp(gaussianPyr[i+1], up, gaussianPyr[i].size());
        cv::Mat diff = gaussianPyr[i] - up;
        laplacianPyr.push_back(diff);
    }
    laplacianPyr.push_back(gaussianPyr.back());

    // 可视化结果
    std::vector<cv::Mat> resized;
    for (const auto& level : laplacianPyr) {
        cv::Mat temp;
        cv::resize(level, temp, laplacianPyr[0].size());
        cv::Mat colored;
        cv::cvtColor(temp, colored, cv::COLOR_GRAY2BGR);
        resized.push_back(colored);
    }

    // 水平拼接所有层
    cv::Mat result;
    cv::hconcat(resized, result);

    return result;
}
```

### 2.3 图像融合

```cpp
cv::Mat imageBlending(const std::string& imgPath1, const std::string& imgPath2, int levels = 4) {
    // 读取图像
    cv::Mat img1 = cv::imread(imgPath1);
    cv::Mat img2 = cv::imread(imgPath2);
    if (img1.empty() || img2.empty()) {
        throw std::runtime_error("无法读取图像");
    }

    // 确保两张图像大小相同
    if (img1.size() != img2.size()) {
        cv::resize(img2, img2, img1.size());
    }

    // 构建拉普拉斯金字塔的lambda函数
    auto buildLaplacianPyramid = [](const cv::Mat& img, int levels) {
        std::vector<cv::Mat> gaussianPyr;
        std::vector<cv::Mat> laplacianPyr;

        // 构建高斯金字塔
        gaussianPyr.push_back(img);
        cv::Mat current = img.clone();
        for (int i = 0; i < levels-1; i++) {
            cv::Mat down;
            cv::pyrDown(current, down);
            gaussianPyr.push_back(down);
            current = down;
        }

        // 构建拉普拉斯金字塔
        for (int i = 0; i < levels-1; i++) {
            cv::Mat up;
            cv::pyrUp(gaussianPyr[i+1], up, gaussianPyr[i].size());
            cv::Mat diff = gaussianPyr[i] - up;
            laplacianPyr.push_back(diff);
        }
        laplacianPyr.push_back(gaussianPyr.back());

        return laplacianPyr;
    };

    // 构建两个图像的拉普拉斯金字塔
    auto lapPyr1 = buildLaplacianPyramid(img1, levels);
    auto lapPyr2 = buildLaplacianPyramid(img2, levels);

    // 融合金字塔
    std::vector<cv::Mat> blendedPyr;
    for (int i = 0; i < levels; i++) {
        cv::Mat mask = cv::Mat::zeros(lapPyr1[i].size(), CV_8UC3);
        mask(cv::Rect(0, 0, lapPyr1[i].cols/2, lapPyr1[i].rows)).setTo(cv::Scalar(1,1,1));

        cv::Mat blended;
        cv::multiply(lapPyr1[i], mask, blended);
        cv::multiply(lapPyr2[i], cv::Scalar(1,1,1) - mask, mask);
        blended += mask;

        blendedPyr.push_back(blended);
    }

    // 重建融合图像
    cv::Mat result = blendedPyr.back();
    for (int i = levels-2; i >= 0; i--) {
        cv::Mat up;
        cv::pyrUp(result, up, blendedPyr[i].size());
        result = up + blendedPyr[i];
    }

    return result;
}
```

### 2.4 SIFT尺度空间

```cpp
cv::Mat siftScaleSpace(const std::string& imgPath, int octaves = 4, int scales = 5, double sigma = 1.6) {
    // 读取图像
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        throw std::runtime_error("无法读取图像");
    }

    // 图像上采样
    cv::Mat upsampled;
    cv::resize(img, upsampled, cv::Size(), 2, 2);

    // 初始化尺度空间
    std::vector<std::vector<cv::Mat>> scaleSpace;
    double k = std::pow(2.0, 1.0/scales);  // 尺度因子

    // 构建尺度空间
    for (int o = 0; o < octaves; o++) {
        std::vector<cv::Mat> octave;
        cv::Mat base;

        if (o == 0) {
            cv::GaussianBlur(upsampled, base, cv::Size(), sigma);
        } else {
            cv::resize(scaleSpace.back()[scales-3], base, cv::Size(), 0.5, 0.5);
        }

        double currentSigma = sigma;
        octave.push_back(base);

        // 生成每个尺度的图像
        for (int s = 1; s < scales; s++) {
            currentSigma *= k;
            cv::Mat blurred;
            cv::GaussianBlur(base, blurred, cv::Size(), currentSigma);
            octave.push_back(blurred);
        }

        scaleSpace.push_back(octave);
        base = octave[scales-3];
    }

    // 可视化结果
    std::vector<cv::Mat> rows;
    for (const auto& octave : scaleSpace) {
        std::vector<cv::Mat> resizedOctave;
        for (const auto& img : octave) {
            cv::Mat resized;
            cv::resize(img, resized, scaleSpace[0][0].size());
            cv::Mat colored;
            cv::cvtColor(resized, colored, cv::COLOR_GRAY2BGR);
            resizedOctave.push_back(colored);
        }
        cv::Mat row;
        cv::hconcat(resizedOctave, row);
        rows.push_back(row);
    }

    cv::Mat result;
    cv::vconcat(rows, result);

    return result;
}
```

### 2.5 显著性检测

```cpp
cv::Mat saliencyDetection(const std::string& imgPath, int levels = 4) {
    // 读取图像
    cv::Mat img = cv::imread(imgPath);
    if (img.empty()) {
        throw std::runtime_error("无法读取图像");
    }

    // 转换到Lab颜色空间
    cv::Mat lab;
    cv::cvtColor(img, lab, cv::COLOR_BGR2Lab);

    // 分离通道
    std::vector<cv::Mat> channels;
    cv::split(lab, channels);

    // 构建高斯金字塔
    std::vector<std::vector<cv::Mat>> pyramids(3);
    for (int i = 0; i < 3; i++) {
        pyramids[i].push_back(channels[i]);
        cv::Mat current = channels[i];
        for (int j = 0; j < levels-1; j++) {
            cv::Mat down;
            cv::pyrDown(current, down);
            pyramids[i].push_back(down);
            current = down;
        }
    }

    // 计算显著性图
    cv::Mat saliency = cv::Mat::zeros(img.size(), CV_32F);
    for (int i = 0; i < 3; i++) {
        cv::Mat channelSaliency = cv::Mat::zeros(img.size(), CV_32F);
        for (int level = 0; level < levels; level++) {
            cv::Mat resized;
            cv::resize(pyramids[i][level], resized, img.size());
            cv::Scalar mean = cv::mean(resized);
            cv::Mat diff;
            cv::absdiff(resized, mean, diff);
            channelSaliency += diff;
        }
        saliency += channelSaliency;
    }

    // 归一化
    cv::normalize(saliency, saliency, 0, 255, cv::NORM_MINMAX);
    saliency.convertTo(saliency, CV_8U);

    // 应用颜色映射以便可视化
    cv::Mat saliencyColor;
    cv::applyColorMap(saliency, saliencyColor, cv::COLORMAP_JET);

    // 将原图和显著性图并排显示
    cv::Mat result;
    cv::hconcat(std::vector<cv::Mat>{img, saliencyColor}, result);

    return result;
}
```