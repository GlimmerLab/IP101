# 图像细化代码实现指南 🎨

本文档提供了图像细化算法的Python和C++完整实现代码。每个实现都包含了详细的注释说明和参数解释。

## 目录
- [1. Python实现](#1-python实现)
  - [1.1 基本细化算法](#11-基本细化算法)
  - [1.2 Hilditch细化](#12-hilditch细化)
  - [1.3 Zhang-Suen细化](#13-zhang-suen细化)
  - [1.4 骨架提取](#14-骨架提取)
  - [1.5 中轴变换](#15-中轴变换)
- [2. C++实现](#2-c实现)
  - [2.1 基本细化算法](#21-基本细化算法)
  - [2.2 Hilditch细化](#22-hilditch细化)
  - [2.3 Zhang-Suen细化](#23-zhang-suen细化)
  - [2.4 骨架提取](#24-骨架提取)
  - [2.5 中轴变换](#25-中轴变换)

## 1. Python实现

### 1.1 基本细化算法

```python
import cv2
import numpy as np

def basic_thinning(img_path: str) -> np.ndarray:
    """
    使用基本的细化算法进行图像细化

    参数:
        img_path: str, 输入图像路径

    返回:
        np.ndarray: 细化结果图像
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 二值化
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 转换为0和1的格式
    skeleton = binary.copy() // 255
    changing = True

    # 定义结构元素
    B1 = np.array([[0, 0, 0],
                   [0, 1, 0],
                   [1, 1, 1]], dtype=np.uint8)
    B2 = np.array([[0, 0, 0],
                   [1, 1, 0],
                   [1, 1, 0]], dtype=np.uint8)

    while changing:
        changing = False
        temp = skeleton.copy()

        # 应用细化操作
        for i in range(4):
            # 旋转结构元素
            b1 = np.rot90(B1, i)
            b2 = np.rot90(B2, i)

            # 应用结构元素
            eroded1 = cv2.erode(temp, b1)
            dilated1 = cv2.dilate(eroded1, b1)
            eroded2 = cv2.erode(temp, b2)
            dilated2 = cv2.dilate(eroded2, b2)

            # 更新骨架
            skeleton = skeleton & ~(temp - dilated1)
            skeleton = skeleton & ~(temp - dilated2)

            if np.any(temp != skeleton):
                changing = True

    # 转换回0-255格式
    result = skeleton.astype(np.uint8) * 255

    # 转换为彩色图像
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return result
```

### 1.2 Hilditch细化

```python
def hilditch_thinning(img_path: str) -> np.ndarray:
    """
    使用Hilditch算法进行图像细化

    参数:
        img_path: str, 输入图像路径

    返回:
        np.ndarray: 细化结果图像
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 二值化
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 转换为0和1的格式
    skeleton = binary.copy() // 255
    changing = True

    def hilditch_condition(p):
        # 获取8邻域
        p2,p3,p4,p5,p6,p7,p8,p9 = p[0:2], p[1:3], p[2:4], p[3:5], p[4:6], p[5:7], p[6:8], p[7:9]

        # Hilditch条件
        c1 = sum(p[1:]) >= 2 and sum(p[1:]) <= 6  # 连通性条件
        c2 = sum([p2[1], p4[1], p6[1], p8[1]]) >= 1  # 端点条件
        c3 = p2[1] * p4[1] * p6[1] == 0  # 连续性条件1
        c4 = p4[1] * p6[1] * p8[1] == 0  # 连续性条件2

        return c1 and c2 and c3 and c4

    while changing:
        changing = False
        temp = skeleton.copy()

        # 遍历图像
        for i in range(1, skeleton.shape[0]-1):
            for j in range(1, skeleton.shape[1]-1):
                if temp[i,j] == 1:
                    # 获取3x3邻域
                    neighborhood = []
                    for x in range(i-1, i+2):
                        for y in range(j-1, j+2):
                            neighborhood.append(temp[x,y])

                    # 应用Hilditch条件
                    if hilditch_condition(neighborhood):
                        skeleton[i,j] = 0
                        changing = True

    # 转换回0-255格式
    result = skeleton.astype(np.uint8) * 255

    # 转换为彩色图像
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return result
```

### 1.3 Zhang-Suen细化

```python
def zhang_suen_thinning(img_path: str) -> np.ndarray:
    """
    使用Zhang-Suen算法进行图像细化

    参数:
        img_path: str, 输入图像路径

    返回:
        np.ndarray: 细化结果图像
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 二值化
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 转换为0和1的格式
    skeleton = binary.copy() // 255

    def zhang_suen_iteration(img, iter_type):
        changing = False
        rows, cols = img.shape

        # 创建标记数组
        markers = np.zeros_like(img)

        for i in range(1, rows-1):
            for j in range(1, cols-1):
                if img[i,j] == 1:
                    # 获取8邻域
                    p2,p3,p4,p5,p6,p7,p8,p9 = (img[i-1,j], img[i-1,j+1], img[i,j+1],
                                              img[i+1,j+1], img[i+1,j], img[i+1,j-1],
                                              img[i,j-1], img[i-1,j-1])

                    # 计算条件
                    A = 0
                    for k in range(len([p2,p3,p4,p5,p6,p7,p8,p9])-1):
                        if [p2,p3,p4,p5,p6,p7,p8,p9][k] == 0 and [p2,p3,p4,p5,p6,p7,p8,p9][k+1] == 1:
                            A += 1
                    B = sum([p2,p3,p4,p5,p6,p7,p8,p9])

                    m1 = p2 * p4 * p6 if iter_type == 0 else p2 * p4 * p8
                    m2 = p4 * p6 * p8 if iter_type == 0 else p2 * p6 * p8

                    if (A == 1 and B >= 2 and B <= 6 and m1 == 0 and m2 == 0):
                        markers[i,j] = 1
                        changing = True

        img[markers == 1] = 0
        return img, changing

    # 迭代进行细化
    changing = True
    while changing:
        skeleton, changing1 = zhang_suen_iteration(skeleton, 0)
        skeleton, changing2 = zhang_suen_iteration(skeleton, 1)
        changing = changing1 or changing2

    # 转换回0-255格式
    result = skeleton.astype(np.uint8) * 255

    # 转换为彩色图像
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return result
```

### 1.4 骨架提取

```python
def skeleton_extraction(img_path: str) -> np.ndarray:
    """
    使用形态学操作提取图像骨架

    参数:
        img_path: str, 输入图像路径

    返回:
        np.ndarray: 骨架提取结果图像
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 二值化
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    # 初始化骨架
    skeleton = np.zeros_like(binary)
    eroded = binary.copy()

    while True:
        # 腐蚀
        eroded = cv2.erode(eroded, kernel)
        # 膨胀
        dilated = cv2.dilate(eroded, kernel)
        # 得到差值图像
        temp = cv2.subtract(binary, dilated)
        # 加入骨架
        skeleton = cv2.bitwise_or(skeleton, temp)
        # 判断是否继续迭代
        if cv2.countNonZero(eroded) == 0:
            break

    # 转换为彩色图像
    result = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)

    return result
```

### 1.5 中轴变换

```python
def medial_axis_transform(img_path: str) -> np.ndarray:
    """
    计算图像的中轴变换

    参数:
        img_path: str, 输入图像路径

    返回:
        np.ndarray: 中轴变换结果图像
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 二值化
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 计算距离变换
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # 归一化距离变换结果
    cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)

    # 提取局部最大值
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(dist_transform, kernel)
    local_max = (dist_transform == dilated).astype(np.uint8) * 255

    # 转换为彩色图像
    result = cv2.cvtColor(local_max, cv2.COLOR_GRAY2BGR)

    return result
```

## 2. C++实现

### 2.1 基本细化算法

```cpp
#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat basicThinning(const std::string& imgPath) {
    // 读取图像
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        throw std::runtime_error("无法读取图像");
    }

    // 二值化
    cv::Mat binary;
    cv::threshold(img, binary, 127, 255, cv::THRESH_BINARY);

    // 转换为0和1的格式
    cv::Mat skeleton = binary / 255;
    bool changing = true;

    // 定义结构元素
    cv::Mat B1 = (cv::Mat_<uchar>(3,3) << 0,0,0, 0,1,0, 1,1,1);
    cv::Mat B2 = (cv::Mat_<uchar>(3,3) << 0,0,0, 1,1,0, 1,1,0);

    while (changing) {
        changing = false;
        cv::Mat temp = skeleton.clone();

        // 应用细化操作
        for (int i = 0; i < 4; i++) {
            // 旋转结构元素
            cv::Mat b1, b2;
            cv::rotate(B1, b1, i);
            cv::rotate(B2, b2, i);

            // 应用结构元素
            cv::Mat eroded1, dilated1, eroded2, dilated2;
            cv::erode(temp, eroded1, b1);
            cv::dilate(eroded1, dilated1, b1);
            cv::erode(temp, eroded2, b2);
            cv::dilate(eroded2, dilated2, b2);

            // 更新骨架
            cv::Mat diff1 = temp - dilated1;
            cv::Mat diff2 = temp - dilated2;
            skeleton = skeleton & ~diff1;
            skeleton = skeleton & ~diff2;

            cv::Mat diff;
            cv::compare(temp, skeleton, diff, cv::CMP_NE);
            if (cv::countNonZero(diff) > 0) {
                changing = true;
            }
        }
    }

    // 转换回0-255格式
    cv::Mat result;
    skeleton.convertTo(result, CV_8UC1, 255);

    // 转换为彩色图像
    cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);

    return result;
}
```

### 2.2 Hilditch细化

```cpp
cv::Mat hilditchThinning(const std::string& imgPath) {
    // 读取图像
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        throw std::runtime_error("无法读取图像");
    }

    // 二值化
    cv::Mat binary;
    cv::threshold(img, binary, 127, 255, cv::THRESH_BINARY);

    // 转换为0和1的格式
    cv::Mat skeleton = binary / 255;
    bool changing = true;

    auto hilditchCondition = [](const std::vector<uchar>& p) {
        // 获取8邻域
        std::vector<uchar> p2{p[0], p[1]}, p3{p[1], p[2]}, p4{p[2], p[3]},
                          p5{p[3], p[4]}, p6{p[4], p[5]}, p7{p[5], p[6]},
                          p8{p[6], p[7]}, p9{p[7], p[8]};

        // Hilditch条件
        int sum = std::accumulate(p.begin()+1, p.end(), 0);
        int sum_corners = p2[1] + p4[1] + p6[1] + p8[1];
        bool c1 = sum >= 2 && sum <= 6;
        bool c2 = sum_corners >= 1;
        bool c3 = p2[1] * p4[1] * p6[1] == 0;
        bool c4 = p4[1] * p6[1] * p8[1] == 0;

        return c1 && c2 && c3 && c4;
    };

    while (changing) {
        changing = false;
        cv::Mat temp = skeleton.clone();

        for (int i = 1; i < skeleton.rows-1; i++) {
            for (int j = 1; j < skeleton.cols-1; j++) {
                if (temp.at<uchar>(i,j) == 1) {
                    // 获取3x3邻域
                    std::vector<uchar> neighborhood;
                    for (int x = i-1; x <= i+1; x++) {
                        for (int y = j-1; y <= j+1; y++) {
                            neighborhood.push_back(temp.at<uchar>(x,y));
                        }
                    }

                    // 应用Hilditch条件
                    if (hilditchCondition(neighborhood)) {
                        skeleton.at<uchar>(i,j) = 0;
                        changing = true;
                    }
                }
            }
        }
    }

    // 转换回0-255格式
    cv::Mat result;
    skeleton.convertTo(result, CV_8UC1, 255);

    // 转换为彩色图像
    cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);

    return result;
}
```

### 2.3 Zhang-Suen细化

```cpp
cv::Mat zhangSuenThinning(const std::string& imgPath) {
    // 读取图像
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        throw std::runtime_error("无法读取图像");
    }

    // 二值化
    cv::Mat binary;
    cv::threshold(img, binary, 127, 255, cv::THRESH_BINARY);

    // 转换为0和1的格式
    cv::Mat skeleton = binary / 255;

    auto zhangSuenIteration = [](cv::Mat& img, int iter_type) {
        bool changing = false;
        cv::Mat markers = cv::Mat::zeros(img.size(), CV_8UC1);

        for (int i = 1; i < img.rows-1; i++) {
            for (int j = 1; j < img.cols-1; j++) {
                if (img.at<uchar>(i,j) == 1) {
                    // 获取8邻域
                    uchar p2 = img.at<uchar>(i-1,j);
                    uchar p3 = img.at<uchar>(i-1,j+1);
                    uchar p4 = img.at<uchar>(i,j+1);
                    uchar p5 = img.at<uchar>(i+1,j+1);
                    uchar p6 = img.at<uchar>(i+1,j);
                    uchar p7 = img.at<uchar>(i+1,j-1);
                    uchar p8 = img.at<uchar>(i,j-1);
                    uchar p9 = img.at<uchar>(i-1,j-1);

                    // 计算条件
                    std::vector<uchar> neighbors{p2,p3,p4,p5,p6,p7,p8,p9};
                    int A = 0;
                    for (size_t k = 0; k < neighbors.size()-1; k++) {
                        if (neighbors[k] == 0 && neighbors[k+1] == 1) {
                            A++;
                        }
                    }
                    if (neighbors.back() == 0 && neighbors.front() == 1) {
                        A++;
                    }

                    int B = std::accumulate(neighbors.begin(), neighbors.end(), 0);
                    int m1 = iter_type == 0 ? p2 * p4 * p6 : p2 * p4 * p8;
                    int m2 = iter_type == 0 ? p4 * p6 * p8 : p2 * p6 * p8;

                    if (A == 1 && B >= 2 && B <= 6 && m1 == 0 && m2 == 0) {
                        markers.at<uchar>(i,j) = 1;
                        changing = true;
                    }
                }
            }
        }

        img.setTo(0, markers);
        return changing;
    };

    // 迭代进行细化
    bool changing = true;
    while (changing) {
        bool changing1 = zhangSuenIteration(skeleton, 0);
        bool changing2 = zhangSuenIteration(skeleton, 1);
        changing = changing1 || changing2;
    }

    // 转换回0-255格式
    cv::Mat result;
    skeleton.convertTo(result, CV_8UC1, 255);

    // 转换为彩色图像
    cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);

    return result;
}
```

### 2.4 骨架提取

```cpp
cv::Mat skeletonExtraction(const std::string& imgPath) {
    // 读取图像
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        throw std::runtime_error("无法读取图像");
    }

    // 二值化
    cv::Mat binary;
    cv::threshold(img, binary, 127, 255, cv::THRESH_BINARY);

    // 定义结构元素
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3,3));

    // 初始化骨架
    cv::Mat skeleton = cv::Mat::zeros(binary.size(), CV_8UC1);
    cv::Mat eroded = binary.clone();

    while (true) {
        // 腐蚀
        cv::erode(eroded, eroded, kernel);
        // 膨胀
        cv::Mat dilated;
        cv::dilate(eroded, dilated, kernel);
        // 得到差值图像
        cv::Mat temp;
        cv::subtract(binary, dilated, temp);
        // 加入骨架
        cv::bitwise_or(skeleton, temp, skeleton);
        // 判断是否继续迭代
        if (cv::countNonZero(eroded) == 0) {
            break;
        }
    }

    // 转换为彩色图像
    cv::Mat result;
    cv::cvtColor(skeleton, result, cv::COLOR_GRAY2BGR);

    return result;
}
```

### 2.5 中轴变换

```cpp
cv::Mat medialAxisTransform(const std::string& imgPath) {
    // 读取图像
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        throw std::runtime_error("无法读取图像");
    }

    // 二值化
    cv::Mat binary;
    cv::threshold(img, binary, 127, 255, cv::THRESH_BINARY);

    // 计算距离变换
    cv::Mat dist_transform;
    cv::distanceTransform(binary, dist_transform, cv::DIST_L2, 5);

    // 归一化距离变换结果
    cv::normalize(dist_transform, dist_transform, 0, 255, cv::NORM_MINMAX);

    // 提取局部最大值
    cv::Mat kernel = cv::Mat::ones(3, 3, CV_8UC1);
    cv::Mat dilated;
    cv::dilate(dist_transform, dilated, kernel);
    cv::Mat local_max = (dist_transform == dilated);
    local_max.convertTo(local_max, CV_8UC1, 255);

    // 转换为彩色图像
    cv::Mat result;
    cv::cvtColor(local_max, result, cv::COLOR_GRAY2BGR);

    return result;
}
```