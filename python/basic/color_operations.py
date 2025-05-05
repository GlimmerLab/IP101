"""
颜色操作相关问题：
1. 通道替换 - 将RGB通道顺序改为BGR
2. 灰度化 - 将彩色图像转换为灰度图像
3. 二值化 - 将灰度图像转换为二值图像
4. 大津算法 - 自适应二值化
5. HSV变换 - 将RGB图像转换为HSV色彩空间
"""

import cv2
import numpy as np

def channel_swap(img_path):
    """
    问题1：通道替换
    将RGB通道顺序改为BGR

    参数:
        img_path: 输入图像路径

    返回:
        处理后的图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 分离通道
    b, g, r = cv2.split(img)

    # 重新组合通道 (BGR -> RGB)
    result = cv2.merge([r, g, b])

    return result

def grayscale(img_path):
    """
    问题2：灰度化
    将彩色图像转换为灰度图像

    参数:
        img_path: 输入图像路径

    返回:
        灰度图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 分离通道
    b, g, r = cv2.split(img)

    # 计算灰度值 (Y = 0.2126R + 0.7152G + 0.0722B)
    result = 0.2126 * r + 0.7152 * g + 0.0722 * b
    result = result.astype(np.uint8)

    return result

def thresholding(img_path, th=128):
    """
    问题3：二值化
    将灰度图像转换为二值图像

    参数:
        img_path: 输入图像路径
        th: 阈值，默认为128

    返回:
        二值化图像
    """
    # 读取图像并转换为灰度图
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 手动实现二值化
    result = np.zeros_like(img)
    result[img > th] = 255

    return result

def otsu_thresholding(img_path):
    """
    问题4：大津算法
    使用大津算法进行自适应二值化

    参数:
        img_path: 输入图像路径

    返回:
        二值化图像
    """
    # 读取图像并转换为灰度图
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 计算直方图
    hist = np.histogram(img, bins=256, range=(0, 256))[0]

    # 计算总像素数
    total = img.size

    # 计算累积和和累积均值
    sum_total = np.sum(hist * np.arange(256))
    sum_back = 0
    w_back = 0
    w_fore = 0
    max_variance = 0
    threshold = 0

    # 遍历所有可能的阈值
    for t in range(256):
        w_back += hist[t]
        if w_back == 0:
            continue

        w_fore = total - w_back
        if w_fore == 0:
            break

        sum_back += t * hist[t]

        # 计算均值
        mean_back = sum_back / w_back
        mean_fore = (sum_total - sum_back) / w_fore

        # 计算方差
        variance = w_back * w_fore * (mean_back - mean_fore) ** 2

        if variance > max_variance:
            max_variance = variance
            threshold = t

    # 应用阈值
    result = np.zeros_like(img)
    result[img > threshold] = 255

    return result

def hsv_transform(img_path):
    """
    问题5：HSV变换
    将RGB图像转换为HSV色彩空间

    参数:
        img_path: 输入图像路径

    返回:
        HSV图像
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 归一化到[0,1]范围
    img = img.astype(np.float32) / 255.0

    # 分离通道
    b, g, r = cv2.split(img)

    # 计算最大值和最小值
    max_val = np.maximum(np.maximum(r, g), b)
    min_val = np.minimum(np.minimum(r, g), b)

    # 计算差值
    diff = max_val - min_val

    # 计算H通道
    h = np.zeros_like(max_val)
    # 当max_val等于min_val时，h=0
    mask = (diff != 0)
    # 当max_val等于r时
    mask_r = (max_val == r) & mask
    h[mask_r] = 60 * ((g[mask_r] - b[mask_r]) / diff[mask_r] % 6)
    # 当max_val等于g时
    mask_g = (max_val == g) & mask
    h[mask_g] = 60 * ((b[mask_g] - r[mask_g]) / diff[mask_g] + 2)
    # 当max_val等于b时
    mask_b = (max_val == b) & mask
    h[mask_b] = 60 * ((r[mask_b] - g[mask_b]) / diff[mask_b] + 4)
    # 处理负值
    h[h < 0] += 360

    # 计算S通道
    s = np.zeros_like(max_val)
    s[max_val != 0] = diff[max_val != 0] / max_val[max_val != 0]

    # 计算V通道
    v = max_val

    # 合并通道
    h = (h / 2).astype(np.uint8)  # OpenCV中H的范围是[0,180]
    s = (s * 255).astype(np.uint8)
    v = (v * 255).astype(np.uint8)

    result = cv2.merge([h, s, v])

    return result

def compute_rgb_to_hsv_manual(image):
    """手动实现RGB到HSV的颜色空间转换

    参数:
        image: 输入的RGB图像

    返回:
        hsv: HSV颜色空间的图像
    """
    if len(image.shape) != 3:
        raise ValueError("输入图像必须是RGB图像")

    # 将图像归一化到[0,1]范围
    rgb = image.astype(np.float32) / 255.0

    # 分离RGB通道
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]

    # 计算最大值、最小值和差值
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    diff = cmax - cmin

    # 初始化HSV数组
    h = np.zeros_like(r)
    s = np.zeros_like(r)
    v = cmax

    # 计算色相H
    # 当cmax和cmin相等时，h保持为0
    mask = (diff != 0)

    # R是最大值的情况
    mask_r = (cmax == r) & mask
    h[mask_r] = 60 * ((g[mask_r] - b[mask_r]) / diff[mask_r] % 6)

    # G是最大值的情况
    mask_g = (cmax == g) & mask
    h[mask_g] = 60 * ((b[mask_g] - r[mask_g]) / diff[mask_g] + 2)

    # B是最大值的情况
    mask_b = (cmax == b) & mask
    h[mask_b] = 60 * ((r[mask_b] - g[mask_b]) / diff[mask_b] + 4)

    # 处理负值
    h[h < 0] += 360

    # 计算饱和度S
    s[cmax != 0] = diff[cmax != 0] / cmax[cmax != 0]

    # 转换到OpenCV的HSV范围
    h = h / 2  # OpenCV中H的范围是[0,180]
    s = s * 255  # OpenCV中S的范围是[0,255]
    v = v * 255  # OpenCV中V的范围是[0,255]

    # 合并通道
    hsv = np.stack([h, s, v], axis=2).astype(np.uint8)

    return hsv

def main(problem_id=1):
    """
    主函数，通过problem_id选择要运行的问题

    参数:
        problem_id: 问题编号，默认为1
    """
    input_path = "../images/imori.jpg"

    if problem_id == 1:
        result = channel_swap(input_path)
        output_path = "../images/answer_1.jpg"
        title = "通道替换 (RGB -> BGR)"
    elif problem_id == 2:
        result = grayscale(input_path)
        output_path = "../images/answer_2.jpg"
        title = "灰度化"
    elif problem_id == 3:
        result = thresholding(input_path)
        output_path = "../images/answer_3.jpg"
        title = "二值化"
    elif problem_id == 4:
        result = otsu_thresholding(input_path)
        output_path = "../images/answer_4.jpg"
        title = "大津算法"
    elif problem_id == 5:
        result = hsv_transform(input_path)
        output_path = "../images/answer_5.jpg"
        title = "HSV变换"
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
    problem_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(problem_id)