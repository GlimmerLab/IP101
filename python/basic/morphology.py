"""
形态学处理相关问题：
38. 膨胀操作 - 使用结构元素对图像进行膨胀
39. 腐蚀操作 - 使用结构元素对图像进行腐蚀
40. 开运算 - 先腐蚀后膨胀
41. 闭运算 - 先膨胀后腐蚀
42. 形态学梯度 - 膨胀图像与腐蚀图像之差
"""

import cv2
import numpy as np

def dilation(img_path, kernel_size=3):
    """
    问题38：膨胀操作
    使用结构元素对图像进行膨胀

    参数:
        img_path: 输入图像路径
        kernel_size: 结构元素大小，默认为3

    返回:
        膨胀后的图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 创建结构元素
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 应用膨胀操作
    result = cv2.dilate(img, kernel, iterations=1)

    return result

def erosion(img_path, kernel_size=3):
    """
    问题39：腐蚀操作
    使用结构元素对图像进行腐蚀

    参数:
        img_path: 输入图像路径
        kernel_size: 结构元素大小，默认为3

    返回:
        腐蚀后的图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 创建结构元素
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 应用腐蚀操作
    result = cv2.erode(img, kernel, iterations=1)

    return result

def opening(img_path, kernel_size=3):
    """
    问题40：开运算
    先腐蚀后膨胀

    参数:
        img_path: 输入图像路径
        kernel_size: 结构元素大小，默认为3

    返回:
        开运算结果图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 创建结构元素
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 应用开运算
    result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    return result

def closing(img_path, kernel_size=3):
    """
    问题41：闭运算
    先膨胀后腐蚀

    参数:
        img_path: 输入图像路径
        kernel_size: 结构元素大小，默认为3

    返回:
        闭运算结果图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 创建结构元素
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 应用闭运算
    result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    return result

def morphological_gradient(img_path, kernel_size=3):
    """
    问题42：形态学梯度
    膨胀图像与腐蚀图像之差

    参数:
        img_path: 输入图像路径
        kernel_size: 结构元素大小，默认为3

    返回:
        形态学梯度结果图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 创建结构元素
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 应用形态学梯度
    result = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    return result

def compute_dilation_manual(image, kernel_size=3):
    """手动实现膨胀操作

    参数:
        image: 输入图像
        kernel_size: 结构元素大小，默认3

    返回:
        dilated: 膨胀后的图像
    """
    if len(image.shape) == 3:
        height, width, channels = image.shape
    else:
        height, width = image.shape
        channels = 1
        image = image[..., np.newaxis]

    # 创建输出图像
    dilated = np.zeros_like(image)

    # 计算填充大小
    pad = kernel_size // 2

    # 对图像进行填充
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant')

    # 执行膨胀操作
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                # 提取当前窗口
                window = padded[y:y+kernel_size, x:x+kernel_size, c]
                # 取窗口中的最大值
                dilated[y, x, c] = np.max(window)

    if channels == 1:
        dilated = dilated.squeeze()

    return dilated

def compute_erosion_manual(image, kernel_size=3):
    """手动实现腐蚀操作

    参数:
        image: 输入图像
        kernel_size: 结构元素大小，默认3

    返回:
        eroded: 腐蚀后的图像
    """
    if len(image.shape) == 3:
        height, width, channels = image.shape
    else:
        height, width = image.shape
        channels = 1
        image = image[..., np.newaxis]

    # 创建输出图像
    eroded = np.zeros_like(image)

    # 计算填充大小
    pad = kernel_size // 2

    # 对图像进行填充
    padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='constant')

    # 执行腐蚀操作
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                # 提取当前窗口
                window = padded[y:y+kernel_size, x:x+kernel_size, c]
                # 取窗口中的最小值
                eroded[y, x, c] = np.min(window)

    if channels == 1:
        eroded = eroded.squeeze()

    return eroded

def compute_opening_manual(image, kernel_size=3):
    """手动实现开运算

    参数:
        image: 输入图像
        kernel_size: 结构元素大小，默认3

    返回:
        opened: 开运算结果图像
    """
    # 先腐蚀后膨胀
    eroded = compute_erosion_manual(image, kernel_size)
    opened = compute_dilation_manual(eroded, kernel_size)
    return opened

def compute_closing_manual(image, kernel_size=3):
    """手动实现闭运算

    参数:
        image: 输入图像
        kernel_size: 结构元素大小，默认3

    返回:
        closed: 闭运算结果图像
    """
    # 先膨胀后腐蚀
    dilated = compute_dilation_manual(image, kernel_size)
    closed = compute_erosion_manual(dilated, kernel_size)
    return closed

def compute_morphological_gradient_manual(image, kernel_size=3):
    """手动实现形态学梯度

    参数:
        image: 输入图像
        kernel_size: 结构元素大小，默认3

    返回:
        gradient: 形态学梯度结果图像
    """
    # 计算膨胀和腐蚀结果
    dilated = compute_dilation_manual(image, kernel_size)
    eroded = compute_erosion_manual(image, kernel_size)
    # 计算梯度（膨胀-腐蚀）
    gradient = dilated.astype(np.float32) - eroded.astype(np.float32)
    gradient = np.clip(gradient, 0, 255).astype(np.uint8)
    return gradient

def main(problem_id=38):
    """
    主函数，通过problem_id选择要运行的问题

    参数:
        problem_id: 问题编号，默认为38
    """
    input_path = "../images/imori.jpg"

    if problem_id == 38:
        result = dilation(input_path)
        output_path = "../images/answer_38.jpg"
        title = "膨胀操作"
    elif problem_id == 39:
        result = erosion(input_path)
        output_path = "../images/answer_39.jpg"
        title = "腐蚀操作"
    elif problem_id == 40:
        result = opening(input_path)
        output_path = "../images/answer_40.jpg"
        title = "开运算"
    elif problem_id == 41:
        result = closing(input_path)
        output_path = "../images/answer_41.jpg"
        title = "闭运算"
    elif problem_id == 42:
        result = morphological_gradient(input_path)
        output_path = "../images/answer_42.jpg"
        title = "形态学梯度"
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
    problem_id = int(sys.argv[1]) if len(sys.argv) > 1 else 38
    main(problem_id)