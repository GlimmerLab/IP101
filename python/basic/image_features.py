"""
图像特征相关问题：
51. HOG特征提取 - 使用方向梯度直方图提取特征
52. LBP特征提取 - 使用局部二值模式提取特征
53. Haar特征提取 - 使用Haar-like特征提取特征
54. Gabor特征提取 - 使用Gabor滤波器提取特征
55. 颜色直方图 - 计算图像的颜色直方图特征
"""

import numpy as np
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt
import math

def hog_features(img_path, cell_size=8, block_size=2, bins=9):
    """
    问题51：HOG特征提取
    使用方向梯度直方图提取特征

    参数:
        img_path: 输入图像路径
        cell_size: 单元格大小，默认为8
        block_size: 块大小（以cell为单位），默认为2
        bins: 方向直方图的bin数量，默认为9

    返回:
        HOG特征可视化结果
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 计算梯度
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * 180 / np.pi % 180

    # 计算cell直方图
    h, w = img.shape
    h_cells = h // cell_size
    w_cells = w // cell_size
    histogram = np.zeros((h_cells, w_cells, bins))

    for i in range(h_cells):
        for j in range(w_cells):
            cell_mag = magnitude[i*cell_size:(i+1)*cell_size,
                               j*cell_size:(j+1)*cell_size]
            cell_ori = orientation[i*cell_size:(i+1)*cell_size,
                                 j*cell_size:(j+1)*cell_size]

            # 计算直方图
            for m in range(cell_size):
                for n in range(cell_size):
                    ori = cell_ori[m, n]
                    mag = cell_mag[m, n]
                    bin_index = int(ori / 180 * bins)
                    histogram[i, j, bin_index] += mag

    # 可视化HOG特征
    result = np.zeros_like(img)
    for i in range(h_cells):
        for j in range(w_cells):
            for k in range(bins):
                angle = k * 180 / bins
                rho = histogram[i, j, k]
                x = int((j + 0.5) * cell_size)
                y = int((i + 0.5) * cell_size)
                dx = int(rho * np.cos(angle * np.pi / 180))
                dy = int(rho * np.sin(angle * np.pi / 180))
                cv2.line(result, (x, y), (x + dx, y + dy), 255, 1)

    # 转换为彩色图像
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return result

def lbp_features(img_path):
    """
    问题52：LBP特征提取
    使用局部二值模式提取特征

    参数:
        img_path: 输入图像路径

    返回:
        LBP特征图像
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    h, w = img.shape
    result = np.zeros((h-2, w-2), dtype=np.uint8)

    # 计算LBP特征
    for i in range(1, h-1):
        for j in range(1, w-1):
            center = img[i, j]
            code = 0
            code |= (img[i-1, j-1] > center) << 7
            code |= (img[i-1, j] > center) << 6
            code |= (img[i-1, j+1] > center) << 5
            code |= (img[i, j+1] > center) << 4
            code |= (img[i+1, j+1] > center) << 3
            code |= (img[i+1, j] > center) << 2
            code |= (img[i+1, j-1] > center) << 1
            code |= (img[i, j-1] > center) << 0
            result[i-1, j-1] = code

    # 转换为彩色图像
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return result

def haar_features(img_path):
    """
    问题53：Haar特征提取
    使用Haar-like特征提取特征

    参数:
        img_path: 输入图像路径

    返回:
        Haar特征可视化结果
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 计算积分图
    integral = cv2.integral(img)

    # 定义Haar特征类型
    def compute_haar_feature(x, y, w, h, feature_type):
        if feature_type == 'edge_horizontal':
            # 水平边缘特征
            top = integral[y+h//2, x+w] - integral[y+h//2, x] - integral[y, x+w] + integral[y, x]
            bottom = integral[y+h, x+w] - integral[y+h, x] - integral[y+h//2, x+w] + integral[y+h//2, x]
            return abs(top - bottom)
        elif feature_type == 'edge_vertical':
            # 垂直边缘特征
            left = integral[y+h, x+w//2] - integral[y+h, x] - integral[y, x+w//2] + integral[y, x]
            right = integral[y+h, x+w] - integral[y+h, x+w//2] - integral[y, x+w] + integral[y, x+w//2]
            return abs(left - right)
        return 0

    # 计算特征响应
    result = np.zeros_like(img)
    window_size = 24
    step = 4

    for y in range(0, img.shape[0]-window_size, step):
        for x in range(0, img.shape[1]-window_size, step):
            # 计算水平和垂直边缘特征
            h_response = compute_haar_feature(x, y, window_size, window_size, 'edge_horizontal')
            v_response = compute_haar_feature(x, y, window_size, window_size, 'edge_vertical')
            response = (h_response + v_response) / 2
            result[y:y+window_size, x:x+window_size] = response

    # 归一化结果
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 转换为彩色图像
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return result

def gabor_features(img_path, ksize=31, sigma=4.0, theta=0, lambd=10.0, gamma=0.5):
    """
    问题54：Gabor特征提取
    使用Gabor滤波器提取特征

    参数:
        img_path: 输入图像路径
        ksize: 核大小
        sigma: 标准差
        theta: 方向
        lambd: 波长
        gamma: 空间纵横比

    返回:
        Gabor特征图像
    """
    # 读取图像
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 创建Gabor滤波器组
    filters = []
    num_theta = 8  # 方向数量
    for theta in np.arange(0, np.pi, np.pi/num_theta):
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
        filters.append(kernel)

    # 应用滤波器
    responses = []
    for kernel in filters:
        response = cv2.filter2D(img, cv2.CV_32F, kernel)
        responses.append(response)

    # 合并响应
    result = np.max(responses, axis=0)
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 转换为彩色图像
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return result

def color_histogram(img_path, bins=32):
    """
    问题55：颜色直方图
    计算图像的颜色直方图特征

    参数:
        img_path: 输入图像路径
        bins: 直方图bin数量，默认为32

    返回:
        颜色直方图可视化结果
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 转换到HSV色彩空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 计算直方图
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256])

    # 归一化直方图
    cv2.normalize(hist_h, hist_h, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(hist_s, hist_s, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(hist_v, hist_v, 0, 255, cv2.NORM_MINMAX)

    # 创建直方图可视化
    height = 400
    width = bins * 3
    result = np.zeros((height, width, 3), dtype=np.uint8)

    # 绘制直方图
    for i in range(bins):
        # H通道 - 红色
        cv2.line(result, (i*3, height), (i*3, height-int(hist_h[i])), (0,0,255), 1)
        # S通道 - 绿色
        cv2.line(result, (i*3+1, height), (i*3+1, height-int(hist_s[i])), (0,255,0), 1)
        # V通道 - 蓝色
        cv2.line(result, (i*3+2, height), (i*3+2, height-int(hist_v[i])), (255,0,0), 1)

    return result

def compute_hog_manual(image, cell_size=8, block_size=2, bins=9):
    """
    手动实现HOG特征提取

    参数:
        image: 输入图像(灰度图)
        cell_size: 每个cell的大小
        block_size: 每个block包含的cell数量
        bins: 方向梯度直方图的bin数量

    返回:
        hog_features: HOG特征向量
    """
    # 1. 计算图像梯度
    dx = ndimage.sobel(image, axis=1)
    dy = ndimage.sobel(image, axis=0)

    # 2. 计算梯度幅值和方向
    magnitude = np.sqrt(dx**2 + dy**2)
    orientation = np.arctan2(dy, dx) * 180 / np.pi % 180

    # 3. 计算cell的梯度直方图
    cell_rows = image.shape[0] // cell_size
    cell_cols = image.shape[1] // cell_size
    histogram = np.zeros((cell_rows, cell_cols, bins))

    for i in range(cell_rows):
        for j in range(cell_cols):
            # 获取当前cell的梯度和方向
            cell_mag = magnitude[i*cell_size:(i+1)*cell_size,
                               j*cell_size:(j+1)*cell_size]
            cell_ori = orientation[i*cell_size:(i+1)*cell_size,
                                 j*cell_size:(j+1)*cell_size]

            # 计算投票权重
            for m in range(cell_size):
                for n in range(cell_size):
                    ori = cell_ori[m, n]
                    mag = cell_mag[m, n]

                    # 双线性插值投票
                    bin_index = int(ori / 180 * bins)
                    bin_index_next = (bin_index + 1) % bins
                    weight_next = (ori - bin_index * 180 / bins) / (180 / bins)
                    weight = 1 - weight_next

                    histogram[i, j, bin_index] += mag * weight
                    histogram[i, j, bin_index_next] += mag * weight_next

    # 4. Block归一化
    blocks_rows = cell_rows - block_size + 1
    blocks_cols = cell_cols - block_size + 1
    normalized_blocks = np.zeros((blocks_rows, blocks_cols,
                                block_size * block_size * bins))

    for i in range(blocks_rows):
        for j in range(blocks_cols):
            block = histogram[i:i+block_size, j:j+block_size, :].ravel()
            normalized_blocks[i, j, :] = block / np.sqrt(np.sum(block**2) + 1e-6)

    return normalized_blocks.ravel()

def compute_lbp_manual(image, points=8, radius=1):
    """
    手动实现LBP特征提取

    参数:
        image: 输入图像(灰度图)
        points: 采样点数量
        radius: 采样半径

    返回:
        lbp_image: LBP特征图
    """
    rows = image.shape[0]
    cols = image.shape[1]
    lbp_image = np.zeros_like(image)

    for i in range(radius, rows-radius):
        for j in range(radius, cols-radius):
            center = image[i, j]
            binary = ''

            # 计算采样点的坐标和值
            for p in range(points):
                angle = 2 * np.pi * p / points
                x = i + radius * np.cos(angle)
                y = j - radius * np.sin(angle)

                # 双线性插值
                x1, y1 = int(x), int(y)
                x2, y2 = math.ceil(x), math.ceil(y)
                fx, fy = x - x1, y - y1

                value = (1-fx)*(1-fy)*image[x1,y1] + \
                        fx*(1-fy)*image[x2,y1] + \
                        (1-fx)*fy*image[x1,y2] + \
                        fx*fy*image[x2,y2]

                binary += '1' if value >= center else '0'

            lbp_image[i,j] = int(binary, 2)

    return lbp_image

def compute_haar_manual(image, feature_type='edge'):
    """手动实现Haar特征提取

    参数:
        image: 输入的灰度图像
        feature_type: Haar特征类型，可选值为'edge'(边缘),'line'(线),'center'(中心)

    返回:
        feature_map: Haar特征图
    """
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = image.shape
    feature_map = np.zeros((height, width))

    # 根据特征类型选择不同的计算方法
    if feature_type == 'edge':
        # 计算水平边缘特征
        for y in range(height-1):
            for x in range(width):
                feature_map[y,x] = float(image[y+1,x]) - float(image[y,x])

    elif feature_type == 'line':
        # 计算垂直线特征
        for y in range(height):
            for x in range(width-2):
                feature_map[y,x] = float(image[y,x]) + float(image[y,x+2]) - 2*float(image[y,x+1])

    elif feature_type == 'center':
        # 计算中心特征
        for y in range(1, height-1):
            for x in range(1, width-1):
                center = float(image[y,x])
                surroundings = (float(image[y-1,x]) + float(image[y+1,x]) +
                              float(image[y,x-1]) + float(image[y,x+1])) / 4
                feature_map[y,x] = center - surroundings

    return feature_map

def compute_gabor_manual(image, ksize=31, sigma=4.0, theta=0, lambda_=10.0, gamma=0.5, psi=0):
    """手动实现Gabor滤波器特征提取

    参数:
        image: 输入的灰度图像
        ksize: 滤波器核的大小
        sigma: 高斯包络的标准差
        theta: 滤波器的方向(弧度)
        lambda_: 正弦波的波长
        gamma: 空间纵横比
        psi: 相位偏移

    返回:
        feature_map: Gabor特征图
    """
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 确保ksize是奇数
    ksize = int(ksize)
    if ksize % 2 == 0:
        ksize += 1

    # 创建网格坐标
    k = (ksize-1) // 2
    y, x = np.mgrid[-k:k+1, -k:k+1]

    # 旋转坐标
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    # 计算Gabor滤波器核
    gb = np.exp(-.5 * (x_theta**2 + gamma**2 * y_theta**2) / sigma**2)
    gb *= np.cos(2*np.pi*x_theta/lambda_ + psi)
    gb /= 2*np.pi*sigma**2

    # 应用滤波器
    feature_map = ndimage.convolve(image.astype(float), gb)

    return feature_map

def compute_color_histogram(image, bins=32):
    """计算颜色直方图特征

    参数:
        image: 输入的RGB图像
        bins: 每个颜色通道的直方图柱数

    返回:
        features: 归一化后的颜色直方图特征向量
    """
    if len(image.shape) < 3:
        raise ValueError("输入图像必须是RGB图像")

    features = []

    # 对每个颜色通道计算直方图
    for i in range(3):
        hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        features.extend(hist)

    return np.array(features)

def compute_harris_manual(image, k=0.04, window_size=3, threshold=0.01):
    """手动实现Harris角点检测

    参数:
        image: 输入的灰度图像
        k: Harris响应函数参数，默认0.04
        window_size: 局部窗口大小，默认3
        threshold: 角点检测阈值，默认0.01

    返回:
        corners: 角点检测结果图像
    """
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算x和y方向的梯度
    dx = ndimage.sobel(image, axis=1)
    dy = ndimage.sobel(image, axis=0)

    # 计算梯度乘积
    Ixx = dx * dx
    Ixy = dx * dy
    Iyy = dy * dy

    # 使用高斯窗口进行平滑
    window = np.ones((window_size, window_size)) / (window_size * window_size)
    Sxx = ndimage.convolve(Ixx, window)
    Sxy = ndimage.convolve(Ixy, window)
    Syy = ndimage.convolve(Iyy, window)

    # 计算Harris响应
    det = Sxx * Syy - Sxy * Sxy
    trace = Sxx + Syy
    harris_response = det - k * (trace * trace)

    # 阈值处理
    corners = np.zeros_like(image)
    corners[harris_response > threshold * harris_response.max()] = 255

    return corners

def main(problem_id=51):
    """
    主函数，通过problem_id选择要运行的问题

    参数:
        problem_id: 问题编号，默认为51
    """
    input_path = "../images/imori.jpg"

    if problem_id == 51:
        result = hog_features(input_path)
        output_path = "../images/answer_51.jpg"
        title = "HOG特征"
    elif problem_id == 52:
        result = lbp_features(input_path)
        output_path = "../images/answer_52.jpg"
        title = "LBP特征"
    elif problem_id == 53:
        result = haar_features(input_path)
        output_path = "../images/answer_53.jpg"
        title = "Haar特征"
    elif problem_id == 54:
        result = gabor_features(input_path)
        output_path = "../images/answer_54.jpg"
        title = "Gabor特征"
    elif problem_id == 55:
        result = color_histogram(input_path)
        output_path = "../images/answer_55.jpg"
        title = "颜色直方图"
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
    problem_id = int(sys.argv[1]) if len(sys.argv) > 1 else 51
    main(problem_id)