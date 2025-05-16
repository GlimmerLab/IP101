# 图像匹配代码实现指南 💻

本文档提供了图像匹配算法的Python和C++完整实现代码。每个实现都包含了详细的注释说明和参数解释。

## 目录
- [1. Python实现](#1-python实现)
  - [1.1 SSD匹配](#11-ssd匹配)
  - [1.2 SAD匹配](#12-sad匹配)
  - [1.3 NCC匹配](#13-ncc匹配)
  - [1.4 ZNCC匹配](#14-zncc匹配)
  - [1.5 特征点匹配](#15-特征点匹配)
- [2. C++实现](#2-c实现)
  - [2.1 SSD匹配](#21-ssd匹配)
  - [2.2 SAD匹配](#22-sad匹配)
  - [2.3 NCC匹配](#23-ncc匹配)
  - [2.4 ZNCC匹配](#24-zncc匹配)
  - [2.5 特征点匹配](#25-特征点匹配)

## 1. Python实现

### 1.1 SSD匹配

```python
import cv2
import numpy as np

def ssd_matching(img_path: str, template_path: str) -> np.ndarray:
    """
    使用平方差和(Sum of Squared Differences)进行模板匹配

    参数:
        img_path: str, 输入图像路径
        template_path: str, 模板图像路径

    返回:
        np.ndarray: 匹配结果可视化图像
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if img is None or template is None:
        raise ValueError("无法读取图像")

    h, w = template.shape
    H, W = img.shape
    result = np.zeros((H-h+1, W-w+1), dtype=np.float32)

    # 计算SSD
    for y in range(H-h+1):
        for x in range(W-w+1):
            diff = img[y:y+h, x:x+w] - template
            result[y, x] = np.sum(diff * diff)

    # 归一化结果
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 找到最佳匹配位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # 在原图上绘制矩形框
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img_color, top_left, bottom_right, (0, 0, 255), 2)

    return img_color
```

### 1.2 SAD匹配

```python
def sad_matching(img_path: str, template_path: str) -> np.ndarray:
    """
    使用绝对差和(Sum of Absolute Differences)进行模板匹配

    参数:
        img_path: str, 输入图像路径
        template_path: str, 模板图像路径

    返回:
        np.ndarray: 匹配结果可视化图像
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if img is None or template is None:
        raise ValueError("无法读取图像")

    h, w = template.shape
    H, W = img.shape
    result = np.zeros((H-h+1, W-w+1), dtype=np.float32)

    # 计算SAD
    for y in range(H-h+1):
        for x in range(W-w+1):
            diff = np.abs(img[y:y+h, x:x+w] - template)
            result[y, x] = np.sum(diff)

    # 归一化结果
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 找到最佳匹配位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # 在原图上绘制矩形框
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img_color, top_left, bottom_right, (0, 0, 255), 2)

    return img_color
```

### 1.3 NCC匹配

```python
def ncc_matching(img_path: str, template_path: str) -> np.ndarray:
    """
    使用归一化互相关(Normalized Cross Correlation)进行模板匹配

    参数:
        img_path: str, 输入图像路径
        template_path: str, 模板图像路径

    返回:
        np.ndarray: 匹配结果可视化图像
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if img is None or template is None:
        raise ValueError("无法读取图像")

    h, w = template.shape
    H, W = img.shape
    result = np.zeros((H-h+1, W-w+1), dtype=np.float32)

    # 计算模板的范数
    template_norm = np.sqrt(np.sum(template * template))

    # 计算NCC
    for y in range(H-h+1):
        for x in range(W-w+1):
            window = img[y:y+h, x:x+w]
            window_norm = np.sqrt(np.sum(window * window))
            if window_norm > 0 and template_norm > 0:
                result[y, x] = np.sum(window * template) / (window_norm * template_norm)

    # 归一化结果
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 找到最佳匹配位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc  # NCC使用最大值
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # 在原图上绘制矩形框
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img_color, top_left, bottom_right, (0, 0, 255), 2)

    return img_color
```

### 1.4 ZNCC匹配

```python
def zncc_matching(img_path: str, template_path: str) -> np.ndarray:
    """
    使用零均值归一化互相关(Zero-mean Normalized Cross Correlation)进行模板匹配

    参数:
        img_path: str, 输入图像路径
        template_path: str, 模板图像路径

    返回:
        np.ndarray: 匹配结果可视化图像
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if img is None or template is None:
        raise ValueError("无法读取图像")

    h, w = template.shape
    H, W = img.shape
    result = np.zeros((H-h+1, W-w+1), dtype=np.float32)

    # 计算模板的均值和标准差
    template_mean = np.mean(template)
    template_std = np.std(template)

    # 计算ZNCC
    for y in range(H-h+1):
        for x in range(W-w+1):
            window = img[y:y+h, x:x+w]
            window_mean = np.mean(window)
            window_std = np.std(window)
            if window_std > 0 and template_std > 0:
                zncc = np.sum((window - window_mean) * (template - template_mean)) / (window_std * template_std * h * w)
                result[y, x] = zncc

    # 归一化结果
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 找到最佳匹配位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc  # ZNCC使用最大值
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # 在原图上绘制矩形框
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img_color, top_left, bottom_right, (0, 0, 255), 2)

    return img_color
```

### 1.5 特征点匹配

```python
def feature_point_matching(img_path1: str, img_path2: str) -> np.ndarray:
    """
    使用特征点进行图像匹配

    参数:
        img_path1: str, 第一张图像路径
        img_path2: str, 第二张图像路径

    返回:
        np.ndarray: 匹配结果可视化图像
    """
    # 读取图像
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        raise ValueError("无法读取图像")

    # 创建SIFT检测器
    sift = cv2.SIFT_create()

    # 检测关键点和计算描述子
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 创建FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 进行匹配
    matches = flann.knnMatch(des1, des2, k=2)

    # 应用比率测试
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 绘制匹配结果
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return img_matches
```

## 2. C++实现

### 2.1 SSD匹配

```cpp
#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat ssdMatching(const std::string& imgPath, const std::string& templatePath) {
    // 读取图像
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    cv::Mat templ = cv::imread(templatePath, cv::IMREAD_GRAYSCALE);

    if (img.empty() || templ.empty()) {
        throw std::runtime_error("无法读取图像");
    }

    cv::Mat result;
    int result_cols = img.cols - templ.cols + 1;
    int result_rows = img.rows - templ.rows + 1;
    result.create(result_rows, result_cols, CV_32FC1);

    // 使用OpenCV的matchTemplate函数进行SSD匹配
    cv::matchTemplate(img, templ, result, cv::TM_SQDIFF);

    // 归一化结果
    cv::normalize(result, result, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    // 找到最佳匹配位置
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    // 绘制结果
    cv::Mat imgColor;
    cv::cvtColor(img, imgColor, cv::COLOR_GRAY2BGR);
    cv::rectangle(imgColor, minLoc,
                 cv::Point(minLoc.x + templ.cols, minLoc.y + templ.rows),
                 cv::Scalar(0, 0, 255), 2);

    return imgColor;
}
```

### 2.2 SAD匹配

```cpp
cv::Mat sadMatching(const std::string& imgPath, const std::string& templatePath) {
    // 读取图像
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    cv::Mat templ = cv::imread(templatePath, cv::IMREAD_GRAYSCALE);

    if (img.empty() || templ.empty()) {
        throw std::runtime_error("无法读取图像");
    }

    cv::Mat result;
    int result_cols = img.cols - templ.cols + 1;
    int result_rows = img.rows - templ.rows + 1;
    result.create(result_rows, result_cols, CV_32FC1);

    // 使用OpenCV的matchTemplate函数进行SAD匹配
    cv::matchTemplate(img, templ, result, cv::TM_SQDIFF_NORMED);

    // 归一化结果
    cv::normalize(result, result, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    // 找到最佳匹配位置
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    // 绘制结果
    cv::Mat imgColor;
    cv::cvtColor(img, imgColor, cv::COLOR_GRAY2BGR);
    cv::rectangle(imgColor, minLoc,
                 cv::Point(minLoc.x + templ.cols, minLoc.y + templ.rows),
                 cv::Scalar(0, 0, 255), 2);

    return imgColor;
}
```

### 2.3 NCC匹配

```cpp
cv::Mat nccMatching(const std::string& imgPath, const std::string& templatePath) {
    // 读取图像
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    cv::Mat templ = cv::imread(templatePath, cv::IMREAD_GRAYSCALE);

    if (img.empty() || templ.empty()) {
        throw std::runtime_error("无法读取图像");
    }

    cv::Mat result;
    int result_cols = img.cols - templ.cols + 1;
    int result_rows = img.rows - templ.rows + 1;
    result.create(result_rows, result_cols, CV_32FC1);

    // 使用OpenCV的matchTemplate函数进行NCC匹配
    cv::matchTemplate(img, templ, result, cv::TM_CCORR_NORMED);

    // 归一化结果
    cv::normalize(result, result, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    // 找到最佳匹配位置
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    // 绘制结果
    cv::Mat imgColor;
    cv::cvtColor(img, imgColor, cv::COLOR_GRAY2BGR);
    cv::rectangle(imgColor, maxLoc,
                 cv::Point(maxLoc.x + templ.cols, maxLoc.y + templ.rows),
                 cv::Scalar(0, 0, 255), 2);

    return imgColor;
}
```

### 2.4 ZNCC匹配

```cpp
cv::Mat znccMatching(const std::string& imgPath, const std::string& templatePath) {
    // 读取图像
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    cv::Mat templ = cv::imread(templatePath, cv::IMREAD_GRAYSCALE);

    if (img.empty() || templ.empty()) {
        throw std::runtime_error("无法读取图像");
    }

    cv::Mat result;
    int result_cols = img.cols - templ.cols + 1;
    int result_rows = img.rows - templ.rows + 1;
    result.create(result_rows, result_cols, CV_32FC1);

    // 使用OpenCV的matchTemplate函数进行ZNCC匹配
    cv::matchTemplate(img, templ, result, cv::TM_CCOEFF_NORMED);

    // 归一化结果
    cv::normalize(result, result, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    // 找到最佳匹配位置
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    // 绘制结果
    cv::Mat imgColor;
    cv::cvtColor(img, imgColor, cv::COLOR_GRAY2BGR);
    cv::rectangle(imgColor, maxLoc,
                 cv::Point(maxLoc.x + templ.cols, maxLoc.y + templ.rows),
                 cv::Scalar(0, 0, 255), 2);

    return imgColor;
}
```

### 2.5 特征点匹配

```cpp
cv::Mat featurePointMatching(const std::string& imgPath1, const std::string& imgPath2) {
    // 读取图像
    cv::Mat img1 = cv::imread(imgPath1, cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(imgPath2, cv::IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        throw std::runtime_error("无法读取图像");
    }

    // 创建SIFT检测器
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

    // 检测关键点和计算描述子
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    sift->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);

    // 创建FLANN匹配器
    cv::FlannBasedMatcher matcher;
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);

    // 应用比率测试
    std::vector<cv::DMatch> good_matches;
    const float ratio_thresh = 0.7f;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    // 绘制匹配结果
    cv::Mat img_matches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches,
                   cv::Scalar::all(-1), cv::Scalar::all(-1),
                   std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    return img_matches;
}
```