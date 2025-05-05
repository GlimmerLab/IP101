"""
图像增强相关问题：
23. 直方图均衡化 - 使用直方图均衡化增强图像对比度
24. 伽马变换 - 使用伽马变换调整图像亮度
25. 对比度拉伸 - 使用对比度拉伸增强图像对比度
26. 亮度调整 - 调整图像亮度
27. 饱和度调整 - 调整图像饱和度
"""

import cv2
import numpy as np

def histogram_equalization(img_path):
    """
    问题23：直方图均衡化
    使用直方图均衡化增强图像对比度

    参数:
        img_path: 输入图像路径

    返回:
        增强后的图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 获取图像尺寸
    h, w = gray.shape

    # 计算直方图
    hist = np.zeros(256, dtype=np.int32)
    for y in range(h):
        for x in range(w):
            hist[gray[y, x]] += 1

    # 计算累积直方图
    cum_hist = np.zeros(256, dtype=np.int32)
    cum_hist[0] = hist[0]
    for i in range(1, 256):
        cum_hist[i] = cum_hist[i-1] + hist[i]

    # 归一化累积直方图
    norm_cum_hist = np.zeros(256, dtype=np.float32)
    for i in range(256):
        norm_cum_hist[i] = cum_hist[i] / (h * w) * 255

    # 应用直方图均衡化
    result = np.zeros_like(gray)
    for y in range(h):
        for x in range(w):
            result[y, x] = norm_cum_hist[gray[y, x]]

    # 转换回彩色图像
    result_color = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return result_color

def gamma_transform(img_path, gamma=2.2):
    """
    问题24：伽马变换
    使用伽马变换调整图像亮度

    参数:
        img_path: 输入图像路径
        gamma: 伽马值，默认为2.2

    返回:
        调整后的图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 归一化到0-1
    img_norm = img / 255.0

    # 应用伽马变换
    result_norm = np.power(img_norm, gamma)

    # 转换回0-255
    result = (result_norm * 255).astype(np.uint8)

    return result

def contrast_stretching(img_path, min_val=0, max_val=255):
    """
    问题25：对比度拉伸
    使用对比度拉伸增强图像对比度

    参数:
        img_path: 输入图像路径
        min_val: 最小像素值，默认为0
        max_val: 最大像素值，默认为255

    返回:
        增强后的图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 获取图像尺寸和通道数
    h, w, c = img.shape

    # 创建输出图像
    result = np.zeros_like(img)

    # 对每个通道进行对比度拉伸
    for i in range(c):
        # 获取当前通道
        channel = img[:, :, i]

        # 计算当前通道的最小值和最大值
        min_channel = np.min(channel)
        max_channel = np.max(channel)

        # 应用对比度拉伸
        if max_channel > min_channel:
            result[:, :, i] = (channel - min_channel) / (max_channel - min_channel) * (max_val - min_val) + min_val
        else:
            result[:, :, i] = channel

    return result.astype(np.uint8)

def brightness_adjustment(img_path, factor=1.5):
    """
    问题26：亮度调整
    调整图像亮度

    参数:
        img_path: 输入图像路径
        factor: 亮度调整因子，默认为1.5

    返回:
        调整后的图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 调整亮度通道
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)

    # 转换回BGR颜色空间
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return result

def saturation_adjustment(img_path, factor=1.5):
    """
    问题27：饱和度调整
    调整图像饱和度

    参数:
        img_path: 输入图像路径
        factor: 饱和度调整因子，默认为1.5

    返回:
        调整后的图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 调整饱和度通道
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)

    # 转换回BGR颜色空间
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return result

def compute_gamma_manual(image, gamma=1.0):
    """手动实现伽马变换

    参数:
        image: 输入图像
        gamma: 伽马值，默认1.0

    返回:
        gamma_corrected: 伽马校正后的图像
    """
    # 归一化到[0,1]范围
    image_normalized = image.astype(float) / 255.0

    # 应用伽马变换
    gamma_corrected = np.power(image_normalized, gamma)

    # 转回[0,255]范围
    gamma_corrected = (gamma_corrected * 255).astype(np.uint8)

    return gamma_corrected

def compute_contrast_stretch_manual(image, low_percentile=1, high_percentile=99):
    """手动实现对比度拉伸

    参数:
        image: 输入图像
        low_percentile: 低百分位数，默认1
        high_percentile: 高百分位数，默认99

    返回:
        stretched: 对比度拉伸后的图像
    """
    # 计算百分位数
    low = np.percentile(image, low_percentile)
    high = np.percentile(image, high_percentile)

    # 线性拉伸
    stretched = np.clip((image - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)

    return stretched

def compute_brightness_manual(image, beta=0):
    """手动实现亮度调整

    参数:
        image: 输入图像
        beta: 亮度调整值，正值增加亮度，负值降低亮度，默认0

    返回:
        adjusted: 亮度调整后的图像
    """
    # 直接加减亮度值
    adjusted = np.clip(image.astype(float) + beta, 0, 255).astype(np.uint8)

    return adjusted

def compute_saturation_manual(image, alpha=1.0):
    """手动实现饱和度调整

    参数:
        image: 输入的RGB图像
        alpha: 饱和度调整系数，>1增加饱和度，<1降低饱和度

    返回:
        adjusted: 饱和度调整后的图像
    """
    if len(image.shape) != 3:
        raise ValueError("输入图像必须是RGB图像")

    # 转换为HSV色彩空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    # 调整饱和度通道
    hsv[..., 1] = np.clip(hsv[..., 1] * alpha, 0, 255)

    # 转换回BGR色彩空间
    adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return adjusted

def main(problem_id=23):
    """
    主函数，通过problem_id选择要运行的问题

    参数:
        problem_id: 问题编号，默认为23
    """
    input_path = "../images/imori.jpg"

    if problem_id == 23:
        result = histogram_equalization(input_path)
        output_path = "../images/answer_23.jpg"
        title = "直方图均衡化"
    elif problem_id == 24:
        result = gamma_transform(input_path)
        output_path = "../images/answer_24.jpg"
        title = "伽马变换"
    elif problem_id == 25:
        result = contrast_stretching(input_path)
        output_path = "../images/answer_25.jpg"
        title = "对比度拉伸"
    elif problem_id == 26:
        result = brightness_adjustment(input_path)
        output_path = "../images/answer_26.jpg"
        title = "亮度调整"
    elif problem_id == 27:
        result = saturation_adjustment(input_path)
        output_path = "../images/answer_27.jpg"
        title = "饱和度调整"
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
    problem_id = int(sys.argv[1]) if len(sys.argv) > 1 else 23
    main(problem_id)