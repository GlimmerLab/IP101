"""
图像滤波相关问题：
6. 均值滤波 - 使用3x3均值滤波器进行图像平滑
7. 中值滤波 - 使用3x3中值滤波器进行图像平滑
8. 高斯滤波 - 使用3x3高斯滤波器进行图像平滑
9. 均值池化 - 将图像按照固定大小进行分块，对每个块进行均值操作
10. Max池化 - 将图像按照固定大小进行分块，对每个块取最大值
"""

import cv2
import numpy as np

def mean_filter(img_path, kernel_size=3):
    """
    问题6：均值滤波
    使用3x3均值滤波器进行图像平滑

    参数:
        img_path: 输入图像路径
        kernel_size: 核大小，默认为3

    返回:
        平滑后的图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 获取图像尺寸
    height, width = img.shape[:2]

    # 创建输出图像
    result = np.zeros_like(img)

    # 计算填充大小
    pad = kernel_size // 2

    # 对图像进行填充
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    # 手动实现均值滤波
    for y in range(height):
        for x in range(width):
            for c in range(3):  # 对每个通道进行处理
                window = padded[y:y+kernel_size, x:x+kernel_size, c]
                result[y, x, c] = np.mean(window)

    return result.astype(np.uint8)

def median_filter(img_path, kernel_size=3):
    """
    问题7：中值滤波
    使用3x3中值滤波器进行图像平滑

    参数:
        img_path: 输入图像路径
        kernel_size: 核大小，默认为3

    返回:
        平滑后的图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 获取图像尺寸
    height, width = img.shape[:2]

    # 创建输出图像
    result = np.zeros_like(img)

    # 计算填充大小
    pad = kernel_size // 2

    # 对图像进行填充
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    # 手动实现中值滤波
    for y in range(height):
        for x in range(width):
            for c in range(3):  # 对每个通道进行处理
                window = padded[y:y+kernel_size, x:x+kernel_size, c]
                result[y, x, c] = np.median(window)

    return result.astype(np.uint8)

def gaussian_filter(img_path, kernel_size=3, sigma=1.0):
    """
    问题8：高斯滤波
    使用3x3高斯滤波器进行图像平滑

    参数:
        img_path: 输入图像路径
        kernel_size: 核大小，默认为3
        sigma: 标准差，默认为1.0

    返回:
        平滑后的图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 获取图像尺寸
    height, width = img.shape[:2]

    # 创建输出图像
    result = np.zeros_like(img)

    # 计算填充大小
    pad = kernel_size // 2

    # 生成高斯核
    x = np.arange(-pad, pad + 1)
    y = np.arange(-pad, pad + 1)
    X, Y = np.meshgrid(x, y)
    kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    # 对图像进行填充
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    # 手动实现高斯滤波
    for y in range(height):
        for x in range(width):
            for c in range(3):  # 对每个通道进行处理
                window = padded[y:y+kernel_size, x:x+kernel_size, c]
                result[y, x, c] = np.sum(window * kernel)

    return result.astype(np.uint8)

def mean_pooling(img_path, pool_size=8):
    """
    问题9：均值池化
    将图像按照固定大小进行分块，对每个块进行均值操作

    参数:
        img_path: 输入图像路径
        pool_size: 池化大小，默认为8

    返回:
        池化后的图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 获取图像尺寸
    height, width = img.shape[:2]

    # 计算输出尺寸
    out_height = height // pool_size
    out_width = width // pool_size

    # 创建输出图像
    result = np.zeros((out_height, out_width, 3), dtype=np.uint8)

    # 手动实现均值池化
    for y in range(out_height):
        for x in range(out_width):
            for c in range(3):  # 对每个通道进行处理
                block = img[y*pool_size:(y+1)*pool_size,
                          x*pool_size:(x+1)*pool_size, c]
                result[y, x, c] = np.mean(block)

    return result

def max_pooling(img_path, pool_size=8):
    """
    问题10：最大池化
    将图像按照固定大小进行分块，对每个块进行最大值操作

    参数:
        img_path: 输入图像路径
        pool_size: 池化大小，默认为8

    返回:
        池化后的图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 获取图像尺寸
    height, width = img.shape[:2]

    # 计算输出尺寸
    out_height = height // pool_size
    out_width = width // pool_size

    # 创建输出图像
    result = np.zeros((out_height, out_width, 3), dtype=np.uint8)

    # 手动实现最大池化
    for y in range(out_height):
        for x in range(out_width):
            for c in range(3):  # 对每个通道进行处理
                block = img[y*pool_size:(y+1)*pool_size,
                          x*pool_size:(x+1)*pool_size, c]
                result[y, x, c] = np.max(block)

    return result

def main(problem_id=6):
    """
    主函数，通过problem_id选择要运行的问题

    参数:
        problem_id: 问题编号，默认为6
    """
    input_path = "../images/imori.jpg"

    if problem_id == 6:
        result = mean_filter(input_path)
        output_path = "../images/answer_6.jpg"
        title = "均值滤波"
    elif problem_id == 7:
        result = median_filter(input_path)
        output_path = "../images/answer_7.jpg"
        title = "中值滤波"
    elif problem_id == 8:
        result = gaussian_filter(input_path)
        output_path = "../images/answer_8.jpg"
        title = "高斯滤波"
    elif problem_id == 9:
        result = mean_pooling(input_path)
        output_path = "../images/answer_9.jpg"
        title = "均值池化"
    elif problem_id == 10:
        result = max_pooling(input_path)
        output_path = "../images/answer_10.jpg"
        title = "最大池化"
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
    problem_id = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    main(problem_id)