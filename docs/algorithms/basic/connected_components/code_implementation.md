# 连通域分析代码实现指南 🏝️

本文档提供了连通域分析算法的Python和C++完整实现代码。每个实现都包含了详细的注释说明和参数解释。

## 目录
- [连通域分析代码实现指南 🏝️](#连通域分析代码实现指南-️)
  - [目录](#目录)
  - [1. Python实现](#1-python实现)
    - [1.1 四连通域标记](#11-四连通域标记)
    - [1.2 八连通域标记](#12-八连通域标记)
    - [1.3 连通域统计](#13-连通域统计)
    - [1.4 连通域过滤](#14-连通域过滤)
    - [1.5 连通域属性计算](#15-连通域属性计算)
  - [2. C++实现](#2-c实现)
    - [2.1 四连通域标记](#21-四连通域标记)
    - [2.2 八连通域标记](#22-八连通域标记)
    - [2.3 连通域统计](#23-连通域统计)
    - [2.4 连通域过滤](#24-连通域过滤)
    - [2.5 连通域属性计算](#25-连通域属性计算)

## 1. Python实现

### 1.1 四连通域标记

```python
import cv2
import numpy as np

def four_connected_labeling(img_path: str) -> np.ndarray:
    """
    使用4连通性进行区域标记

    参数:
        img_path: str, 输入图像路径

    返回:
        np.ndarray: 标记结果可视化图像
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 二值化
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 使用OpenCV的连通域标记函数
    num_labels, labels = cv2.connectedComponents(binary, connectivity=4)

    # 为标记结果分配不同的颜色
    colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # 背景为黑色

    # 创建彩色标记图像
    result = colors[labels]

    return result
```

### 1.2 八连通域标记

```python
def eight_connected_labeling(img_path: str) -> np.ndarray:
    """
    使用8连通性进行区域标记

    参数:
        img_path: str, 输入图像路径

    返回:
        np.ndarray: 标记结果可视化图像
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 二值化
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 使用OpenCV的连通域标记函数
    num_labels, labels = cv2.connectedComponents(binary, connectivity=8)

    # 为标记结果分配不同的颜色
    colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # 背景为黑色

    # 创建彩色标记图像
    result = colors[labels]

    return result
```

### 1.3 连通域统计

```python
def connected_components_stats(img_path: str) -> np.ndarray:
    """
    统计连通域的数量和大小

    参数:
        img_path: str, 输入图像路径

    返回:
        np.ndarray: 统计结果可视化图像
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 二值化
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 使用OpenCV的连通域分析函数
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    # 创建彩色结果图像
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 绘制连通域信息
    for i in range(1, num_labels):  # 跳过背景
        x, y, w, h, area = stats[i]
        center = tuple(map(int, centroids[i]))

        # 绘制边界框
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # 绘制中心点
        cv2.circle(result, center, 4, (0, 0, 255), -1)
        # 显示面积
        cv2.putText(result, f"Area: {area}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return result
```

### 1.4 连通域过滤

```python
def connected_components_filtering(img_path: str, min_area: int = 100) -> np.ndarray:
    """
    根据面积过滤连通域

    参数:
        img_path: str, 输入图像路径
        min_area: int, 最小面积阈值，默认为100

    返回:
        np.ndarray: 过滤结果可视化图像
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 二值化
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 使用OpenCV的连通域分析函数
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    # 创建掩码
    mask = np.zeros_like(labels, dtype=np.uint8)

    # 根据面积过滤连通域
    for i in range(1, num_labels):  # 跳过背景
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            mask[labels == i] = 255

    # 转换为彩色图像
    result = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    return result
```

### 1.5 连通域属性计算

```python
def connected_components_properties(img_path: str) -> np.ndarray:
    """
    计算连通域的各种属性

    参数:
        img_path: str, 输入图像路径

    返回:
        np.ndarray: 属性可视化结果图像
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 二值化
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 使用OpenCV的连通域分析函数
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    # 创建彩色结果图像
    result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 计算和绘制每个连通域的属性
    for i in range(1, num_labels):  # 跳过背景
        # 获取基本属性
        x, y, w, h, area = stats[i]
        center = tuple(map(int, centroids[i]))

        # 计算轮廓
        mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # 计算周长
            perimeter = cv2.arcLength(contours[0], True)
            # 计算圆形度
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            # 计算矩形度
            extent = area / (w * h) if w * h > 0 else 0

            # 绘制轮廓
            cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
            # 绘制中心点
            cv2.circle(result, center, 4, (0, 0, 255), -1)
            # 显示属性
            cv2.putText(result, f"Area: {area}", (x, y-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(result, f"Circularity: {circularity:.2f}", (x, y-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(result, f"Extent: {extent:.2f}", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return result
```

## 2. C++实现

### 2.1 四连通域标记

```cpp
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <queue>

// 并查集数据结构
class DisjointSet {
private:
    std::vector<int> parent;
    std::vector<int> rank;

public:
    DisjointSet(int size) : parent(size), rank(size, 0) {
        for (int i = 0; i < size; i++) {
            parent[i] = i;
        }
    }

    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]); // 路径压缩
        }
        return parent[x];
    }

    void unite(int x, int y) {
        x = find(x);
        y = find(y);
        if (x == y) return;

        // 按秩合并
        if (rank[x] < rank[y]) {
            parent[x] = y;
        } else {
            parent[y] = x;
            if (rank[x] == rank[y]) {
                rank[x]++;
            }
        }
    }
};

/**
 * @brief 4连通域标记算法
 * @param src 输入图像（二值图）
 * @param labels 输出标记图像
 * @return 连通域数量
 */
int label_4connected(const cv::Mat& src, cv::Mat& labels) {
    CV_Assert(!src.empty() && src.type() == CV_8UC1);

    int height = src.rows;
    int width = src.cols;

    // 初始化标记图像
    labels = cv::Mat::zeros(height, width, CV_32S);
    int current_label = 1;
    DisjointSet ds(height * width / 4); // 估计标记数量

    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // 跳过背景像素
            if (src.at<uchar>(y, x) == 0) continue;

            std::vector<int> neighbor_labels;
            // 检查上方和左侧像素
            if (y > 0 && labels.at<int>(y-1, x) > 0)
                neighbor_labels.push_back(labels.at<int>(y-1, x));
            if (x > 0 && labels.at<int>(y, x-1) > 0)
                neighbor_labels.push_back(labels.at<int>(y, x-1));

            if (neighbor_labels.empty()) {
                // 新连通域
                labels.at<int>(y, x) = current_label++;
            } else {
                // 取最小标记
                int min_label = *std::min_element(neighbor_labels.begin(), neighbor_labels.end());
                labels.at<int>(y, x) = min_label;
                // 合并等价标记
                for (int label : neighbor_labels) {
                    ds.unite(min_label-1, label-1);
                }
            }
        }
    }

    // 第二次遍历：解决标记等价性
    std::vector<int> label_map(current_label);
    int num_labels = 0;
    for (int i = 0; i < current_label; i++) {
        if (ds.find(i) == i) {
            label_map[i] = ++num_labels;
        }
    }
    for (int i = 0; i < current_label; i++) {
        label_map[i] = label_map[ds.find(i)];
    }

    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (labels.at<int>(y, x) > 0) {
                labels.at<int>(y, x) = label_map[labels.at<int>(y, x)-1];
            }
        }
    }

    return num_labels;
}
```

### 2.2 八连通域标记

```cpp
/**
 * @brief 8连通域标记算法
 * @param src 输入图像（二值图）
 * @param labels 输出标记图像
 * @return 连通域数量
 */
int label_8connected(const cv::Mat& src, cv::Mat& labels) {
    CV_Assert(!src.empty() && src.type() == CV_8UC1);

    int height = src.rows;
    int width = src.cols;

    // 初始化标记图像
    labels = cv::Mat::zeros(height, width, CV_32S);
    int current_label = 1;
    DisjointSet ds(height * width / 4); // 估计标记数量

    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // 跳过背景像素
            if (src.at<uchar>(y, x) == 0) continue;

            std::vector<int> neighbor_labels;
            // 检查8邻域像素
            for (int dy = -1; dy <= 0; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dy == 0 && dx >= 0) break;
                    int ny = y + dy;
                    int nx = x + dx;
                    if (ny >= 0 && nx >= 0 && nx < width) {
                        if (labels.at<int>(ny, nx) > 0) {
                            neighbor_labels.push_back(labels.at<int>(ny, nx));
                        }
                    }
                }
            }

            if (neighbor_labels.empty()) {
                // 新连通域
                labels.at<int>(y, x) = current_label++;
            } else {
                // 取最小标记
                int min_label = *std::min_element(neighbor_labels.begin(), neighbor_labels.end());
                labels.at<int>(y, x) = min_label;
                // 合并等价标记
                for (int label : neighbor_labels) {
                    ds.unite(min_label-1, label-1);
                }
            }
        }
    }

    // 第二次遍历：解决标记等价性
    std::vector<int> label_map(current_label);
    int num_labels = 0;
    for (int i = 0; i < current_label; i++) {
        if (ds.find(i) == i) {
            label_map[i] = ++num_labels;
        }
    }
    for (int i = 0; i < current_label; i++) {
        label_map[i] = label_map[ds.find(i)];
    }

    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (labels.at<int>(y, x) > 0) {
                labels.at<int>(y, x) = label_map[labels.at<int>(y, x)-1];
            }
        }
    }

    return num_labels;
}
```

### 2.3 连通域统计

```cpp
// 连通域结构体
struct ConnectedComponent {
    int label;
    int area;
    cv::Point centroid;
    cv::Rect bbox;
    double circularity;
};

/**
 * @brief 分析连通域
 * @param labels 标记图像
 * @param num_labels 连通域数量
 * @return 连通域统计信息
 */
std::vector<ConnectedComponent> analyze_components(const cv::Mat& labels, int num_labels) {
    std::vector<ConnectedComponent> stats(num_labels);

    // 初始化统计信息
    for (int i = 0; i < num_labels; i++) {
        stats[i].label = i + 1;
        stats[i].area = 0;
        stats[i].bbox = cv::Rect(labels.cols, labels.rows, 0, 0);
        stats[i].centroid = cv::Point(0, 0);
    }

    // 计算基本属性
    #pragma omp parallel for
    for (int y = 0; y < labels.rows; y++) {
        for (int x = 0; x < labels.cols; x++) {
            int label = labels.at<int>(y, x);
            if (label == 0) continue;

            ConnectedComponent& comp = stats[label-1];
            #pragma omp atomic
            comp.area++;

            #pragma omp critical
            {
                comp.bbox.x = std::min(comp.bbox.x, x);
                comp.bbox.y = std::min(comp.bbox.y, y);
                comp.bbox.width = std::max(comp.bbox.width, x - comp.bbox.x + 1);
                comp.bbox.height = std::max(comp.bbox.height, y - comp.bbox.y + 1);
                comp.centroid.x += x;
                comp.centroid.y += y;
            }
        }
    }

    // 计算高级属性
    for (auto& comp : stats) {
        if (comp.area > 0) {
            comp.centroid.x /= comp.area;
            comp.centroid.y /= comp.area;

            // 计算圆形度
            double perimeter = 0;
            for (int y = comp.bbox.y; y < comp.bbox.y + comp.bbox.height; y++) {
                for (int x = comp.bbox.x; x < comp.bbox.x + comp.bbox.width; x++) {
                    if (labels.at<int>(y, x) == comp.label) {
                        // 检查边界点
                        bool is_boundary = false;
                        for (int dy = -1; dy <= 1; dy++) {
                            for (int dx = -1; dx <= 1; dx++) {
                                int ny = y + dy;
                                int nx = x + dx;
                                if (ny >= 0 && ny < labels.rows && nx >= 0 && nx < labels.cols) {
                                    if (labels.at<int>(ny, nx) != comp.label) {
                                        is_boundary = true;
                                        break;
                                    }
                                }
                            }
                            if (is_boundary) break;
                        }
                        if (is_boundary) perimeter++;
                    }
                }
            }
            comp.circularity = (perimeter > 0) ? 4 * CV_PI * comp.area / (perimeter * perimeter) : 0;
        }
    }

    return stats;
}
```

### 2.4 连通域过滤

```cpp
/**
 * @brief 过滤连通域
 * @param labels 标记图像
 * @param stats 连通域统计信息
 * @param min_area 最小面积
 * @param max_area 最大面积
 * @return 过滤后的图像
 */
cv::Mat filter_components(const cv::Mat& labels,
                     const std::vector<ConnectedComponent>& stats,
                     int min_area,
                     int max_area) {
    cv::Mat filtered = cv::Mat::zeros(labels.size(), CV_8UC1);

    #pragma omp parallel for
    for (int y = 0; y < labels.rows; y++) {
        for (int x = 0; x < labels.cols; x++) {
            int label = labels.at<int>(y, x);
            if (label > 0) {
                const auto& comp = stats[label-1];
                if (comp.area >= min_area && comp.area <= max_area) {
                    filtered.at<uchar>(y, x) = 255;
                }
            }
        }
    }

    return filtered;
}
```

### 2.5 连通域属性计算

```cpp
/**
 * @brief 绘制连通域
 * @param src 原始图像
 * @param labels 标记图像
 * @param stats 连通域统计信息
 * @return 可视化结果
 */
cv::Mat draw_components(const cv::Mat& src,
                   const cv::Mat& labels,
                   const std::vector<ConnectedComponent>& stats) {
    cv::Mat result;
    cv::cvtColor(src, result, cv::COLOR_GRAY2BGR);

    // 为每个连通域分配不同的颜色
    cv::RNG rng(12345);
    for (const auto& comp : stats) {
        cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

        // 绘制连通域
        for (int y = 0; y < labels.rows; y++) {
            for (int x = 0; x < labels.cols; x++) {
                if (labels.at<int>(y, x) == comp.label) {
                    result.at<cv::Vec3b>(y, x) = cv::Vec3b(color[0], color[1], color[2]);
                }
            }
        }

        // 绘制边界框
        cv::rectangle(result, comp.bbox, cv::Scalar(0, 255, 0), 2);

        // 绘制中心点
        cv::circle(result, comp.centroid, 4, cv::Scalar(0, 0, 255), -1);

        // 显示属性
        std::string info = "Label: " + std::to_string(comp.label) +
                     " Area: " + std::to_string(comp.area);
        cv::putText(result, info, cv::Point(comp.bbox.x, comp.bbox.y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }

    return result;
}
```