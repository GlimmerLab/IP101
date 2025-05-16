# 目标检测代码实现指南 🎯

本文档提供了目标检测算法的Python和C++完整实现代码。每个实现都包含了详细的注释说明和参数解释。

## 目录
- [1. Python实现](#1-python实现)
  - [1.1 滑动窗口检测](#11-滑动窗口检测)
  - [1.2 HOG+SVM检测](#12-hogsvm检测)
  - [1.3 Haar+AdaBoost检测](#13-haaradaboost检测)
  - [1.4 非极大值抑制](#14-非极大值抑制)
  - [1.5 目标跟踪](#15-目标跟踪)
- [2. C++实现](#2-c实现)
  - [2.1 滑动窗口检测](#21-滑动窗口检测)
  - [2.2 HOG+SVM检测](#22-hogsvm检测)
  - [2.3 Haar+AdaBoost检测](#23-haaradaboost检测)
  - [2.4 非极大值抑制](#24-非极大值抑制)
  - [2.5 目标跟踪](#25-目标跟踪)

## 1. Python实现

### 1.1 滑动窗口检测

```python
import cv2
import numpy as np

def sliding_window_detection(img_path: str, window_size: tuple = (64, 64), stride: int = 32) -> np.ndarray:
    """
    使用滑动窗口进行目标检测

    参数:
        img_path: str, 输入图像路径
        window_size: tuple, 窗口大小，默认(64, 64)
        stride: int, 步长，默认32

    返回:
        np.ndarray: 检测结果可视化图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 复制原图用于绘制结果
    result = img.copy()

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 定义简单的目标检测函数（这里使用边缘检测作为示例）
    def detect_object(window):
        edges = cv2.Canny(window, 100, 200)
        return np.sum(edges) > 1000  # 简单的阈值判断

    # 滑动窗口检测
    for y in range(0, img.shape[0] - window_size[1], stride):
        for x in range(0, img.shape[1] - window_size[0], stride):
            # 提取窗口
            window = gray[y:y+window_size[1], x:x+window_size[0]]

            # 检测目标
            if detect_object(window):
                # 绘制检测框
                cv2.rectangle(result, (x, y),
                            (x + window_size[0], y + window_size[1]),
                            (0, 255, 0), 2)

    return result
```

### 1.2 HOG+SVM检测

```python
from sklearn import svm
from sklearn.preprocessing import StandardScaler

def hog_svm_detection(img_path: str, window_size: tuple = (64, 64), stride: int = 32) -> np.ndarray:
    """
    使用HOG特征和SVM进行目标检测

    参数:
        img_path: str, 输入图像路径
        window_size: tuple, 窗口大小，默认(64, 64)
        stride: int, 步长，默认32

    返回:
        np.ndarray: 检测结果可视化图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 复制原图用于绘制结果
    result = img.copy()

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 创建HOG描述子
    hog = cv2.HOGDescriptor()

    # 创建和训练SVM（这里使用简单的示例数据）
    # 在实际应用中，需要使用大量的正负样本进行训练
    svm_classifier = svm.LinearSVC()
    scaler = StandardScaler()

    # 生成一些示例数据进行训练
    n_samples = 100
    n_features = hog.compute(cv2.resize(gray, window_size)).shape[0]
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)

    # 训练SVM
    X_scaled = scaler.fit_transform(X)
    svm_classifier.fit(X_scaled, y)

    # 滑动窗口检测
    for y in range(0, img.shape[0] - window_size[1], stride):
        for x in range(0, img.shape[1] - window_size[0], stride):
            # 提取窗口
            window = gray[y:y+window_size[1], x:x+window_size[0]]
            window = cv2.resize(window, window_size)

            # 计算HOG特征
            features = hog.compute(window)

            # 特征标准化
            features_scaled = scaler.transform(features.reshape(1, -1))

            # SVM预测
            if svm_classifier.predict(features_scaled)[0] == 1:
                # 绘制检测框
                cv2.rectangle(result, (x, y),
                            (x + window_size[0], y + window_size[1]),
                            (0, 255, 0), 2)

    return result
```

### 1.3 Haar+AdaBoost检测

```python
def haar_adaboost_detection(img_path: str) -> np.ndarray:
    """
    使用Haar特征和AdaBoost进行目标检测

    参数:
        img_path: str, 输入图像路径

    返回:
        np.ndarray: 检测结果可视化图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 复制原图用于绘制结果
    result = img.copy()

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 加载预训练的人脸检测器（作为示例）
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 进行人脸检测
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # 绘制检测结果
    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return result
```

### 1.4 非极大值抑制

```python
def compute_nms_manual(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> list:
    """
    手动实现非极大值抑制算法

    参数:
        boxes: np.ndarray, 边界框坐标列表，每个框为[x1, y1, x2, y2]格式
        scores: np.ndarray, 每个边界框对应的置信度分数
        iou_threshold: float, IoU阈值，默认0.5

    返回:
        list: 保留的边界框索引列表
    """
    # 确保输入数据格式正确
    boxes = np.array(boxes)
    scores = np.array(scores)

    # 计算所有框的面积
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    # 根据分数排序
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        # 保留分数最高的框
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        # 计算其他框与当前框的IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        # 计算IoU
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 保留IoU小于阈值的框
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep
```

### 1.5 目标跟踪

```python
def object_tracking(img_path: str, video_path: str = None) -> np.ndarray:
    """
    实现简单的目标跟踪算法

    参数:
        img_path: str, 目标模板图像路径
        video_path: str, 视频路径，默认为None（使用摄像头）

    返回:
        np.ndarray: 跟踪结果可视化图像
    """
    # 读取目标模板
    template = cv2.imread(img_path)
    if template is None:
        raise ValueError(f"无法读取模板图像: {img_path}")

    # 转换为灰度图
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # 初始化视频捕获
    if video_path is None:
        cap = cv2.VideoCapture(0)  # 使用摄像头
    else:
        cap = cv2.VideoCapture(video_path)

    # 读取第一帧
    ret, frame = cap.read()
    if not ret:
        raise ValueError("无法读取视频帧")

    # 选择初始跟踪区域（这里使用模板大小）
    h, w = template_gray.shape
    x, y = frame.shape[1]//2 - w//2, frame.shape[0]//2 - h//2
    track_window = (x, y, w, h)

    # 设置终止条件
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 使用模板匹配进行跟踪
        result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # 更新跟踪窗口
        track_window = (max_loc[0], max_loc[1], w, h)

        # 绘制跟踪框
        x, y, w, h = track_window
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 显示结果
        cv2.imshow('Tracking', frame)

        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return frame
```

## 2. C++实现

### 2.1 滑动窗口检测

```cpp
#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat slidingWindowDetection(const std::string& imgPath,
                             const cv::Size& windowSize = cv::Size(64, 64),
                             int stride = 32) {
    // 读取图像
    cv::Mat img = cv::imread(imgPath);
    if (img.empty()) {
        throw std::runtime_error("无法读取图像");
    }

    // 复制原图用于绘制结果
    cv::Mat result = img.clone();

    // 转换为灰度图
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // 定义简单的目标检测函数
    auto detectObject = [](const cv::Mat& window) {
        cv::Mat edges;
        cv::Canny(window, edges, 100, 200);
        return cv::sum(edges)[0] > 1000;  // 简单的阈值判断
    };

    // 滑动窗口检测
    for (int y = 0; y < img.rows - windowSize.height; y += stride) {
        for (int x = 0; x < img.cols - windowSize.width; x += stride) {
            // 提取窗口
            cv::Mat window = gray(cv::Rect(x, y, windowSize.width, windowSize.height));

            // 检测目标
            if (detectObject(window)) {
                // 绘制检测框
                cv::rectangle(result,
                            cv::Point(x, y),
                            cv::Point(x + windowSize.width, y + windowSize.height),
                            cv::Scalar(0, 255, 0), 2);
            }
        }
    }

    return result;
}
```

### 2.2 HOG+SVM检测

```cpp
cv::Mat hogSvmDetection(const std::string& imgPath,
                       const cv::Size& windowSize = cv::Size(64, 64),
                       int stride = 32) {
    // 读取图像
    cv::Mat img = cv::imread(imgPath);
    if (img.empty()) {
        throw std::runtime_error("无法读取图像");
    }

    // 复制原图用于绘制结果
    cv::Mat result = img.clone();

    // 转换为灰度图
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // 创建HOG描述子
    cv::HOGDescriptor hog;
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

    // 检测目标
    std::vector<cv::Rect> detections;
    std::vector<double> weights;
    hog.detectMultiScale(img, detections, weights);

    // 绘制检测结果
    for (const auto& rect : detections) {
        cv::rectangle(result, rect, cv::Scalar(0, 255, 0), 2);
    }

    return result;
}
```

### 2.3 Haar+AdaBoost检测

```cpp
cv::Mat haarAdaboostDetection(const std::string& imgPath) {
    // 读取图像
    cv::Mat img = cv::imread(imgPath);
    if (img.empty()) {
        throw std::runtime_error("无法读取图像");
    }

    // 复制原图用于绘制结果
    cv::Mat result = img.clone();

    // 转换为灰度图
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // 加载预训练的人脸检测器
    cv::CascadeClassifier faceCascade;
    faceCascade.load(cv::samples::findFile("haarcascade_frontalface_default.xml"));

    // 检测人脸
    std::vector<cv::Rect> faces;
    faceCascade.detectMultiScale(gray, faces, 1.1, 4);

    // 绘制检测结果
    for (const auto& rect : faces) {
        cv::rectangle(result, rect, cv::Scalar(0, 255, 0), 2);
    }

    return result;
}
```

### 2.4 非极大值抑制

```cpp
std::vector<int> computeNmsManual(const std::vector<cv::Rect>& boxes,
                                 const std::vector<float>& scores,
                                 float iouThreshold = 0.5) {
    std::vector<int> keep;
    if (boxes.empty()) return keep;

    // 计算所有框的面积
    std::vector<float> areas;
    for (const auto& box : boxes) {
        areas.push_back(box.width * box.height);
    }

    // 创建索引数组并根据分数排序
    std::vector<size_t> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&scores](size_t i1, size_t i2) { return scores[i1] > scores[i2]; });

    while (!order.empty()) {
        int i = order[0];
        keep.push_back(i);

        if (order.size() == 1) break;

        std::vector<size_t> new_order;
        for (size_t j = 1; j < order.size(); j++) {
            int n = order[j];

            // 计算IoU
            int xx1 = std::max(boxes[i].x, boxes[n].x);
            int yy1 = std::max(boxes[i].y, boxes[n].y);
            int xx2 = std::min(boxes[i].x + boxes[i].width,
                             boxes[n].x + boxes[n].width);
            int yy2 = std::min(boxes[i].y + boxes[i].height,
                             boxes[n].y + boxes[n].height);

            int w = std::max(0, xx2 - xx1);
            int h = std::max(0, yy2 - yy1);
            float inter = w * h;
            float ovr = inter / (areas[i] + areas[n] - inter);

            if (ovr <= iouThreshold) {
                new_order.push_back(n);
            }
        }

        order = new_order;
    }

    return keep;
}
```

### 2.5 目标跟踪

```cpp
cv::Mat objectTracking(const std::string& imgPath, const std::string& videoPath = "") {
    // 读取目标模板
    cv::Mat templ = cv::imread(imgPath);
    if (templ.empty()) {
        throw std::runtime_error("无法读取模板图像");
    }

    // 转换为灰度图
    cv::Mat templGray;
    cv::cvtColor(templ, templGray, cv::COLOR_BGR2GRAY);

    // 初始化视频捕获
    cv::VideoCapture cap;
    if (videoPath.empty()) {
        cap.open(0);  // 使用摄像头
    } else {
        cap.open(videoPath);
    }

    if (!cap.isOpened()) {
        throw std::runtime_error("无法打开视频源");
    }

    cv::Mat frame, result;
    cap >> frame;
    if (frame.empty()) {
        throw std::runtime_error("无法读取视频帧");
    }

    // 选择初始跟踪区域
    int w = templGray.cols;
    int h = templGray.rows;
    int x = frame.cols/2 - w/2;
    int y = frame.rows/2 - h/2;
    cv::Rect trackWindow(x, y, w, h);

    // 设置终止条件
    cv::TermCriteria termCrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10, 1);

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        result = frame.clone();

        // 转换为灰度图
        cv::Mat frameGray;
        cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);

        // 使用模板匹配进行跟踪
        cv::Mat matchResult;
        cv::matchTemplate(frameGray, templGray, matchResult, cv::TM_CCOEFF_NORMED);

        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc(matchResult, &minVal, &maxVal, &minLoc, &maxLoc);

        // 更新跟踪窗口
        trackWindow = cv::Rect(maxLoc.x, maxLoc.y, w, h);

        // 绘制跟踪框
        cv::rectangle(result, trackWindow, cv::Scalar(0, 255, 0), 2);

        // 显示结果
        cv::imshow("Tracking", result);

        // 按'q'退出
        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();

    return result;
}
```