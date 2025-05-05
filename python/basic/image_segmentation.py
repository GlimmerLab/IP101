"""
图像分割相关问题：
33. 阈值分割 - 使用多种阈值方法进行图像分割
34. K均值分割 - 使用K均值聚类进行图像分割
35. 区域生长 - 使用区域生长方法进行图像分割
36. 分水岭分割 - 使用分水岭算法进行图像分割
37. 图割分割 - 使用图割算法进行图像分割
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans

def threshold_segmentation(img_path, method='otsu'):
    """
    问题33：阈值分割
    使用多种阈值方法进行图像分割

    参数:
        img_path: 输入图像路径
        method: 阈值方法，可选'otsu', 'adaptive', 'triangle'

    返回:
        分割结果图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if method == 'otsu':
        # Otsu阈值分割
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'adaptive':
        # 自适应阈值分割
        result = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    elif method == 'triangle':
        # 三角形阈值分割
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    else:
        raise ValueError(f"不支持的阈值方法: {method}")

    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

def kmeans_segmentation(img_path, k=3):
    """
    问题34：K均值分割
    使用K均值聚类进行图像分割

    参数:
        img_path: 输入图像路径
        k: 聚类数量，默认为3

    返回:
        分割结果图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 将图像转换为特征向量
    pixels = img.reshape((-1, 3))
    pixels = np.float32(pixels)

    # 定义K均值的终止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # 应用K均值聚类
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 将聚类中心转换为uint8类型
    centers = np.uint8(centers)

    # 重建图像
    result = centers[labels.flatten()]
    result = result.reshape(img.shape)

    return result

def region_growing(img_path, seed_point=None, threshold=30):
    """
    问题35：区域生长
    使用区域生长方法进行图像分割

    参数:
        img_path: 输入图像路径
        seed_point: 种子点坐标(x,y)，默认为图像中心
        threshold: 生长阈值，默认为30

    返回:
        分割结果图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 如果未指定种子点，使用图像中心
    if seed_point is None:
        h, w = img.shape[:2]
        seed_point = (w//2, h//2)

    # 创建标记图像
    mask = np.zeros(img.shape[:2], np.uint8)

    # 获取种子点的颜色
    seed_color = img[seed_point[1], seed_point[0]]

    # 定义8邻域
    neighbors = [(0,1), (1,0), (0,-1), (-1,0),
                (1,1), (-1,-1), (-1,1), (1,-1)]

    # 创建待处理点队列
    stack = [seed_point]
    mask[seed_point[1], seed_point[0]] = 255

    while stack:
        x, y = stack.pop()
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if (0 <= nx < img.shape[1] and 0 <= ny < img.shape[0] and
                mask[ny, nx] == 0 and
                np.all(np.abs(img[ny, nx] - seed_color) < threshold)):
                mask[ny, nx] = 255
                stack.append((nx, ny))

    # 应用掩码
    result = img.copy()
    result[mask == 0] = 0

    return result

def watershed_segmentation(img_path):
    """
    问题36：分水岭分割
    使用分水岭算法进行图像分割

    参数:
        img_path: 输入图像路径

    返回:
        分割结果图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用Otsu算法进行二值化
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 噪声去除
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 确定背景区域
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 确定前景区域
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 找到未知区域
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 标记
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 应用分水岭算法
    markers = cv2.watershed(img, markers)

    # 标记边界
    result = img.copy()
    result[markers == -1] = [0, 0, 255]  # 红色标记边界

    return result

def graph_cut_segmentation(img_path):
    """
    问题37：图割分割
    使用图割算法进行图像分割

    参数:
        img_path: 输入图像路径

    返回:
        分割结果图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 创建掩码
    mask = np.zeros(img.shape[:2], np.uint8)

    # 定义矩形区域
    rect = (50, 50, img.shape[1]-100, img.shape[0]-100)

    # 初始化背景和前景模型
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    # 应用GrabCut算法
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # 修改掩码
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')

    # 应用掩码到图像
    result = img * mask2[:,:,np.newaxis]

    return result

def compute_kmeans_manual(image, k=3, max_iters=100):
    """手动实现K均值分割

    参数:
        image: 输入图像
        k: 聚类数量，默认3
        max_iters: 最大迭代次数，默认100

    返回:
        segmented: 分割后的图像
    """
    if len(image.shape) != 3:
        raise ValueError("输入图像必须是RGB图像")

    # 将图像转换为特征向量
    height, width = image.shape[:2]
    pixels = image.reshape((-1, 3)).astype(np.float32)

    # 随机初始化聚类中心
    centers = pixels[np.random.choice(pixels.shape[0], k, replace=False)]

    # 迭代优化
    for _ in range(max_iters):
        old_centers = centers.copy()

        # 计算每个像素到各个中心的距离
        distances = np.sqrt(((pixels[:, np.newaxis] - centers) ** 2).sum(axis=2))

        # 分配标签
        labels = np.argmin(distances, axis=1)

        # 更新聚类中心
        for i in range(k):
            mask = (labels == i)
            if np.any(mask):
                centers[i] = pixels[mask].mean(axis=0)

        # 检查收敛
        if np.allclose(old_centers, centers):
            break

    # 重建图像
    segmented = centers[labels].reshape((height, width, 3)).astype(np.uint8)

    return segmented

def compute_region_growing_manual(image, seed_point=None, threshold=30):
    """手动实现区域生长算法

    参数:
        image: 输入图像
        seed_point: 种子点坐标(x,y)，默认为图像中心
        threshold: 生长阈值，默认30

    返回:
        segmented: 分割后的图像
    """
    if len(image.shape) != 3:
        raise ValueError("输入图像必须是RGB图像")

    height, width = image.shape[:2]

    # 如果未指定种子点，使用图像中心
    if seed_point is None:
        seed_point = (width//2, height//2)

    # 创建标记图像
    mask = np.zeros((height, width), dtype=np.uint8)

    # 获取种子点的颜色
    seed_color = image[seed_point[1], seed_point[0]]

    # 定义8邻域
    neighbors = [(0,1), (1,0), (0,-1), (-1,0),
                (1,1), (-1,-1), (-1,1), (1,-1)]

    # 创建待处理点队列
    stack = [seed_point]
    mask[seed_point[1], seed_point[0]] = 1

    # 区域生长
    while stack:
        x, y = stack.pop()
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if (0 <= nx < width and 0 <= ny < height and
                mask[ny, nx] == 0 and
                np.all(np.abs(image[ny, nx] - seed_color) < threshold)):
                mask[ny, nx] = 1
                stack.append((nx, ny))

    # 应用掩码
    segmented = image.copy()
    segmented[mask == 0] = 0

    return segmented

def compute_watershed_manual(image):
    """手动实现分水岭分割

    参数:
        image: 输入图像

    返回:
        segmented: 分割后的图像
    """
    if len(image.shape) != 3:
        raise ValueError("输入图像必须是RGB图像")

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用Sobel算子计算梯度
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient = np.uint8(gradient * 255 / gradient.max())

    # 二值化梯度图像
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学操作
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # 计算距离变换
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, markers = cv2.connectedComponents(np.uint8(dist_transform > 0.5 * dist_transform.max()))

    # 标记边界
    segmented = image.copy()
    boundaries = np.zeros_like(gray)
    for i in range(1, markers.max() + 1):
        mask = (markers == i)
        dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        boundary = dilated - mask.astype(np.uint8)
        boundaries[boundary > 0] = 255

    # 在原图上标记边界
    segmented[boundaries > 0] = [0, 0, 255]  # 红色标记边界

    return segmented

def main(problem_id=33):
    """
    主函数，通过problem_id选择要运行的问题

    参数:
        problem_id: 问题编号，默认为33
    """
    input_path = "../images/imori.jpg"

    if problem_id == 33:
        result = threshold_segmentation(input_path)
        output_path = "../images/answer_33.jpg"
        title = "阈值分割"
    elif problem_id == 34:
        result = kmeans_segmentation(input_path)
        output_path = "../images/answer_34.jpg"
        title = "K均值分割"
    elif problem_id == 35:
        result = region_growing(input_path)
        output_path = "../images/answer_35.jpg"
        title = "区域生长"
    elif problem_id == 36:
        result = watershed_segmentation(input_path)
        output_path = "../images/answer_36.jpg"
        title = "分水岭分割"
    elif problem_id == 37:
        result = graph_cut_segmentation(input_path)
        output_path = "../images/answer_37.jpg"
        title = "图割分割"
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
    problem_id = int(sys.argv[1]) if len(sys.argv) > 1 else 33
    main(problem_id)