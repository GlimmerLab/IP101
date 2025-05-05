"""
边缘检测相关问题：
11. 微分滤波 - 使用3x3微分滤波器进行边缘检测
12. Sobel滤波 - 使用Sobel算子进行边缘检测
13. Prewitt滤波 - 使用Prewitt算子进行边缘检测
14. Laplacian滤波 - 使用Laplacian算子进行边缘检测
15. 浮雕效果 - 使用差分和偏移实现浮雕效果
16. 边缘检测 - 综合多种边缘检测方法
"""

import cv2
import numpy as np

def differential_filter(img_path, kernel_size=3):
    """
    问题11：微分滤波
    使用3x3微分滤波器进行边缘检测

    参数:
        img_path: 输入图像路径
        kernel_size: 滤波器大小，默认为3

    返回:
        边缘检测结果
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 获取图像尺寸
    h, w = gray.shape

    # 创建输出图像
    result = np.zeros_like(gray)

    # 计算填充大小
    pad = kernel_size // 2

    # 对图像进行填充
    padded = np.pad(gray, ((pad, pad), (pad, pad)), mode='edge')

    # 手动实现微分滤波
    for y in range(h):
        for x in range(w):
            # 提取当前窗口
            window = padded[y:y+kernel_size, x:x+kernel_size]

            # 计算x方向和y方向的差分
            dx = window[1, 2] - window[1, 0]
            dy = window[2, 1] - window[0, 1]

            # 计算梯度幅值
            result[y, x] = np.sqrt(dx*dx + dy*dy)

    # 归一化到0-255
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result

def sobel_filter(img_path, kernel_size=3):
    """
    问题12：Sobel滤波
    使用Sobel算子进行边缘检测

    参数:
        img_path: 输入图像路径
        kernel_size: 滤波器大小，默认为3

    返回:
        边缘检测结果
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 获取图像尺寸
    h, w = gray.shape

    # 创建输出图像
    result = np.zeros_like(gray)

    # 计算填充大小
    pad = kernel_size // 2

    # 对图像进行填充
    padded = np.pad(gray, ((pad, pad), (pad, pad)), mode='edge')

    # 定义Sobel算子
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # 手动实现Sobel滤波
    for y in range(h):
        for x in range(w):
            # 提取当前窗口
            window = padded[y:y+kernel_size, x:x+kernel_size]

            # 计算x方向和y方向的卷积
            gx = np.sum(window * sobel_x)
            gy = np.sum(window * sobel_y)

            # 计算梯度幅值
            result[y, x] = np.sqrt(gx*gx + gy*gy)

    # 归一化到0-255
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result

def prewitt_filter(img_path, kernel_size=3):
    """
    问题13：Prewitt滤波
    使用Prewitt算子进行边缘检测

    参数:
        img_path: 输入图像路径
        kernel_size: 滤波器大小，默认为3

    返回:
        边缘检测结果
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 获取图像尺寸
    h, w = gray.shape

    # 创建输出图像
    result = np.zeros_like(gray)

    # 计算填充大小
    pad = kernel_size // 2

    # 对图像进行填充
    padded = np.pad(gray, ((pad, pad), (pad, pad)), mode='edge')

    # 定义Prewitt算子
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    # 手动实现Prewitt滤波
    for y in range(h):
        for x in range(w):
            # 提取当前窗口
            window = padded[y:y+kernel_size, x:x+kernel_size]

            # 计算x方向和y方向的卷积
            gx = np.sum(window * prewitt_x)
            gy = np.sum(window * prewitt_y)

            # 计算梯度幅值
            result[y, x] = np.sqrt(gx*gx + gy*gy)

    # 归一化到0-255
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result

def laplacian_filter(img_path, kernel_size=3):
    """
    问题14：Laplacian滤波
    使用Laplacian算子进行边缘检测

    参数:
        img_path: 输入图像路径
        kernel_size: 滤波器大小，默认为3

    返回:
        边缘检测结果
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 获取图像尺寸
    h, w = gray.shape

    # 创建输出图像
    result = np.zeros_like(gray)

    # 计算填充大小
    pad = kernel_size // 2

    # 对图像进行填充
    padded = np.pad(gray, ((pad, pad), (pad, pad)), mode='edge')

    # 定义Laplacian算子
    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    # 手动实现Laplacian滤波
    for y in range(h):
        for x in range(w):
            # 提取当前窗口
            window = padded[y:y+kernel_size, x:x+kernel_size]

            # 计算Laplacian卷积
            result[y, x] = np.sum(window * laplacian)

    # 取绝对值并归一化到0-255
    result = np.abs(result)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result

def emboss_effect(img_path, kernel_size=3, offset=128):
    """
    问题15：浮雕效果
    使用差分和偏移实现浮雕效果

    参数:
        img_path: 输入图像路径
        kernel_size: 滤波器大小，默认为3
        offset: 偏移值，默认为128

    返回:
        浮雕效果图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 获取图像尺寸
    h, w = gray.shape

    # 创建输出图像
    result = np.zeros_like(gray)

    # 计算填充大小
    pad = kernel_size // 2

    # 对图像进行填充
    padded = np.pad(gray, ((pad, pad), (pad, pad)), mode='edge')

    # 定义浮雕算子
    emboss = np.array([[2, 0, 0], [0, -1, 0], [0, 0, -1]])

    # 手动实现浮雕效果
    for y in range(h):
        for x in range(w):
            # 提取当前窗口
            window = padded[y:y+kernel_size, x:x+kernel_size]

            # 计算浮雕卷积
            result[y, x] = np.sum(window * emboss) + offset

    # 归一化到0-255
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result

def edge_detection(img_path, method='sobel', threshold=100):
    """
    问题16：边缘检测
    综合多种边缘检测方法

    参数:
        img_path: 输入图像路径
        method: 边缘检测方法，可选 'sobel', 'prewitt', 'laplacian'
        threshold: 阈值，默认为100

    返回:
        边缘检测结果
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 根据选择的方法进行边缘检测
    if method == 'sobel':
        # 使用Sobel算子
        result = sobel_filter(img_path)
    elif method == 'prewitt':
        # 使用Prewitt算子
        result = prewitt_filter(img_path)
    elif method == 'laplacian':
        # 使用Laplacian算子
        result = laplacian_filter(img_path)
    else:
        raise ValueError(f"不支持的方法: {method}")

    # 二值化处理
    _, binary = cv2.threshold(result, threshold, 255, cv2.THRESH_BINARY)

    return binary

def main(problem_id=11):
    """
    主函数，通过problem_id选择要运行的问题

    参数:
        problem_id: 问题编号，默认为11
    """
    input_path = "../images/imori.jpg"

    if problem_id == 11:
        result = differential_filter(input_path)
        output_path = "../images/answer_11.jpg"
        title = "微分滤波"
    elif problem_id == 12:
        result = sobel_filter(input_path)
        output_path = "../images/answer_12.jpg"
        title = "Sobel滤波"
    elif problem_id == 13:
        result = prewitt_filter(input_path)
        output_path = "../images/answer_13.jpg"
        title = "Prewitt滤波"
    elif problem_id == 14:
        result = laplacian_filter(input_path)
        output_path = "../images/answer_14.jpg"
        title = "Laplacian滤波"
    elif problem_id == 15:
        result = emboss_effect(input_path)
        output_path = "../images/answer_15.jpg"
        title = "浮雕效果"
    elif problem_id == 16:
        result = edge_detection(input_path)
        output_path = "../images/answer_16.jpg"
        title = "边缘检测"
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
    problem_id = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    main(problem_id)