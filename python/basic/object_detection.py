"""
目标检测相关问题：
71. 滑动窗口检测 - 使用滑动窗口进行目标检测
72. HOG+SVM检测 - 使用HOG特征和SVM进行目标检测
73. Haar+AdaBoost检测 - 使用Haar特征和AdaBoost进行目标检测
74. 非极大值抑制 - 实现非极大值抑制算法
75. 目标跟踪 - 实现简单的目标跟踪算法
"""

import cv2
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler

def sliding_window_detection(img_path, window_size=(64, 64), stride=32):
    """
    问题71：滑动窗口检测
    使用滑动窗口进行目标检测

    参数:
        img_path: 输入图像路径
        window_size: 窗口大小，默认(64, 64)
        stride: 步长，默认32

    返回:
        检测结果可视化
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

class HOGExtractor:
    """HOG特征提取器"""
    def __init__(self, win_size=(64, 128), block_size=(16, 16),
                 block_stride=(8, 8), cell_size=(8, 8), nbins=9):
        self.win_size = win_size
        self.block_size = block_size
        self.block_stride = block_stride
        self.cell_size = cell_size
        self.nbins = nbins

        # 计算特征维度
        n_cells = (win_size[0] // cell_size[0], win_size[1] // cell_size[1])
        n_blocks = ((n_cells[0] - block_size[0] // cell_size[0]) // (block_stride[0] // cell_size[0]) + 1,
                   (n_cells[1] - block_size[1] // cell_size[1]) // (block_stride[1] // cell_size[1]) + 1)
        self.feature_dim = n_blocks[0] * n_blocks[1] * block_size[0] * block_size[1] * nbins // (cell_size[0] * cell_size[1])

    def compute_gradients(self, img):
        """计算图像梯度"""
        # 确保图像是灰度图
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 计算x和y方向的梯度
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

        # 计算梯度幅值和方向
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx) * 180 / np.pi
        angle[angle < 0] += 180

        return magnitude, angle

    def compute_cell_histogram(self, magnitude, angle):
        """计算cell直方图"""
        n_cells = (self.win_size[0] // self.cell_size[0],
                  self.win_size[1] // self.cell_size[1])
        cell_hists = np.zeros((n_cells[1], n_cells[0], self.nbins))

        # 计算每个像素的贡献
        for y in range(magnitude.shape[0]):
            for x in range(magnitude.shape[1]):
                mag = magnitude[y, x]
                ang = angle[y, x]

                # 计算bin索引
                bin_width = 180.0 / self.nbins
                bin_idx = int(ang / bin_width)
                next_bin = (bin_idx + 1) % self.nbins
                alpha = (ang - bin_idx * bin_width) / bin_width

                # 计算cell索引
                cell_x = x // self.cell_size[0]
                cell_y = y // self.cell_size[1]

                if cell_x < n_cells[0] and cell_y < n_cells[1]:
                    cell_hists[cell_y, cell_x, bin_idx] += mag * (1 - alpha)
                    cell_hists[cell_y, cell_x, next_bin] += mag * alpha

        return cell_hists

    def compute_block_features(self, cell_hists):
        """计算block特征并归一化"""
        n_cells = (self.win_size[0] // self.cell_size[0],
                  self.win_size[1] // self.cell_size[1])
        n_blocks = ((n_cells[0] - self.block_size[0] // self.cell_size[0]) //
                   (self.block_stride[0] // self.cell_size[0]) + 1,
                   (n_cells[1] - self.block_size[1] // self.cell_size[1]) //
                   (self.block_stride[1] // self.cell_size[1]) + 1)

        features = []
        for by in range(0, n_cells[1] - self.block_size[1] // self.cell_size[1] + 1,
                       self.block_stride[1] // self.cell_size[1]):
            for bx in range(0, n_cells[0] - self.block_size[0] // self.cell_size[0] + 1,
                          self.block_stride[0] // self.cell_size[0]):
                # 提取block中的cell直方图
                block_features = cell_hists[by:by + self.block_size[1] // self.cell_size[1],
                                         bx:bx + self.block_size[0] // self.cell_size[0]].flatten()

                # L2-Norm归一化
                norm = np.sqrt(np.sum(block_features**2) + 1e-5)
                block_features = block_features / norm

                features.extend(block_features)

        return np.array(features)

    def compute(self, img):
        """计算HOG特征"""
        # 调整图像大小
        img = cv2.resize(img, self.win_size)

        # 计算梯度
        magnitude, angle = self.compute_gradients(img)

        # 计算cell直方图
        cell_hists = self.compute_cell_histogram(magnitude, angle)

        # 计算block特征并归一化
        features = self.compute_block_features(cell_hists)

        return features

def hog_svm_detection(img_path, window_size=(64, 64), stride=32):
    """
    问题72：HOG+SVM检测
    使用HOG特征和SVM进行目标检测

    参数:
        img_path: 输入图像路径
        window_size: 窗口大小，默认(64, 64)
        stride: 步长，默认32

    返回:
        检测结果可视化
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 复制原图用于绘制结果
    result = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 创建HOG特征提取器
    hog = HOGExtractor(win_size=window_size)

    # 创建和训练SVM（这里使用简单的示例数据）
    svm_classifier = svm.LinearSVC()
    scaler = StandardScaler()

    # 生成一些示例数据进行训练
    n_samples = 100
    n_features = hog.feature_dim
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

def haar_adaboost_detection(img_path):
    """
    问题73：Haar+AdaBoost检测
    使用Haar特征和AdaBoost进行目标检测

    参数:
        img_path: 输入图像路径

    返回:
        检测结果可视化
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

def compute_nms_manual(boxes, scores, iou_threshold=0.5):
    """手动实现非极大值抑制算法

    参数:
        boxes: 边界框坐标列表，每个框为[x1, y1, x2, y2]格式
        scores: 每个边界框对应的置信度分数
        iou_threshold: IoU阈值，默认0.5

    返回:
        keep: 保留的边界框索引列表
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

def object_tracking(img_path, video_path=None):
    """
    问题75：目标跟踪
    实现简单的目标跟踪算法

    参数:
        img_path: 输入图像路径
        video_path: 视频路径（可选）

    返回:
        跟踪结果可视化
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 复制原图用于绘制结果
    result = img.copy()

    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义目标颜色范围（以红色为例）
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])

    # 创建掩码
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # 寻找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 找到最大的轮廓
        c = max(contours, key=cv2.contourArea)

        # 计算边界框
        x, y, w, h = cv2.boundingRect(c)

        # 绘制跟踪框
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 绘制目标中心
        center = (x + w//2, y + h//2)
        cv2.circle(result, center, 5, (0, 0, 255), -1)

    return result

def main(problem_id=71):
    """
    主函数，通过problem_id选择要运行的问题

    参数:
        problem_id: 问题编号，默认为71
    """
    input_path = "../images/imori.jpg"

    if problem_id == 71:
        result = sliding_window_detection(input_path)
        output_path = "../images/answer_71.jpg"
        title = "滑动窗口检测"
    elif problem_id == 72:
        result = hog_svm_detection(input_path)
        output_path = "../images/answer_72.jpg"
        title = "HOG+SVM检测"
    elif problem_id == 73:
        result = haar_adaboost_detection(input_path)
        output_path = "../images/answer_73.jpg"
        title = "Haar+AdaBoost检测"
    elif problem_id == 74:
        # 示例边界框和分数
        boxes = [[100, 100, 200, 200], [150, 150, 250, 250], [120, 120, 220, 220]]
        scores = [0.9, 0.8, 0.7]
        keep = compute_nms_manual(boxes, scores)

        # 在原图上绘制保留的边界框
        result = cv2.imread(input_path)
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

        output_path = "../images/answer_74.jpg"
        title = "非极大值抑制"
    elif problem_id == 75:
        result = object_tracking(input_path)
        output_path = "../images/answer_75.jpg"
        title = "目标跟踪"
    else:
        print(f"问题 {problem_id} 不存在")
        return

    # 保存结果
    cv2.imwrite(output_path, result)

    # 显示结果
    cv2.imshow(title, result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    problem_id = int(sys.argv[1]) if len(sys.argv) > 1 else 71
    main(problem_id)