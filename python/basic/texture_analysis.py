import numpy as np
import cv2
from typing import List, Tuple, Dict
from scipy import signal
from scipy.stats import entropy

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

def texture_classification(img: np.ndarray,
                         num_classes: int = 4) -> np.ndarray:
    """纹理分类

    Args:
        img: 输入图像
        num_classes: 类别数

    Returns:
        np.ndarray: 分类结果
    """
    # 计算LBP特征
    lbp = compute_lbp(img)

    # 计算LBP直方图
    hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
    hist = hist.flatten() / np.sum(hist)

    # 计算Gabor特征
    gabor_features = compute_gabor_features(img)

    # 计算Gabor特征统计量
    gabor_stats = []
    for feature in gabor_features:
        gabor_stats.extend([
            np.mean(feature),
            np.std(feature),
            entropy(feature.flatten())
        ])

    # 组合特征
    features = np.concatenate([hist, gabor_stats])

    # 使用K-means聚类
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    _, labels, centers = cv2.kmeans(
        features.reshape(1, -1).astype(np.float32),
        num_classes, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    return labels.reshape(img.shape[:2])

def texture_segmentation(img: np.ndarray,
                        num_classes: int = 4) -> np.ndarray:
    """纹理分割

    Args:
        img: 输入图像
        num_classes: 类别数

    Returns:
        np.ndarray: 分割结果
    """
    # 计算多尺度特征
    scales = [1.0, 0.5, 0.25]
    features = []

    for scale in scales:
        # 缩放图像
        h, w = img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        img_scaled = cv2.resize(img, (new_w, new_h))

        # 计算LBP特征
        lbp = compute_lbp(img_scaled)
        hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        hist = hist.flatten() / np.sum(hist)

        # 计算Gabor特征
        gabor = compute_gabor_features(img_scaled)
        gabor_stats = []
        for feature in gabor:
            gabor_stats.extend([
                np.mean(feature),
                np.std(feature),
                entropy(feature.flatten())
            ])

        # 组合特征
        feature = np.concatenate([hist, gabor_stats])
        features.append(feature)

    # 合并多尺度特征
    combined_features = np.concatenate(features)

    # 使用K-means聚类
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    _, labels, centers = cv2.kmeans(
        combined_features.reshape(1, -1).astype(np.float32),
        num_classes, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 重构分割结果
    segmentation = labels.reshape(img.shape[:2])

    # 后处理
    kernel = np.ones((5, 5), np.uint8)
    segmentation = cv2.morphologyEx(segmentation, cv2.MORPH_CLOSE, kernel)
    segmentation = cv2.morphologyEx(segmentation, cv2.MORPH_OPEN, kernel)

    return segmentation

def main():
    """主函数,用于测试各种纹理分析算法"""
    # 读取测试图像
    img = cv2.imread('test_texture.jpg')

    # 计算GLCM
    glcm = compute_glcm(img)
    print('GLCM shape:', glcm.shape)

    # 计算LBP
    lbp = compute_lbp(img)
    cv2.imwrite('result_lbp.jpg', lbp)

    # 计算Gabor特征
    gabor_features = compute_gabor_features(img)
    print('Gabor features shape:', gabor_features.shape)

    # 纹理分类
    classification = texture_classification(img)
    cv2.imwrite('result_classification.jpg', classification * 50)

    # 纹理分割
    segmentation = texture_segmentation(img)
    cv2.imwrite('result_segmentation.jpg', segmentation * 50)

if __name__ == '__main__':
    main()
