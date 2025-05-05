"""
图像金字塔相关问题：
76. 高斯金字塔 - 构建图像的高斯金字塔
77. 拉普拉斯金字塔 - 构建图像的拉普拉斯金字塔
78. 图像融合 - 使用金字塔进行图像融合
79. SIFT尺度空间 - 构建SIFT算法的尺度空间
80. 显著性检测 - 基于金字塔的显著性检测
"""

import numpy as np
import cv2

def gaussian_kernel(size=5, sigma=1.0):
    """
    生成高斯核
    """
    kernel = np.zeros((size, size))
    center = size // 2

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x**2 + y**2)/(2*sigma**2))

    return kernel / kernel.sum()

def manual_conv2d(img, kernel):
    """
    手动实现2D卷积
    """
    h, w = img.shape
    k_h, k_w = kernel.shape
    pad_h = k_h // 2
    pad_w = k_w // 2

    # 填充图像
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros_like(img)

    # 执行卷积
    for i in range(h):
        for j in range(w):
            output[i, j] = np.sum(padded[i:i+k_h, j:j+k_w] * kernel)

    return output

def manual_resize(img, scale_factor):
    """
    手动实现图像缩放
    """
    if len(img.shape) == 3:
        h, w, c = img.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        resized = np.zeros((new_h, new_w, c))

        for k in range(c):
            for i in range(new_h):
                for j in range(new_w):
                    src_i = min(int(i / scale_factor), h-1)
                    src_j = min(int(j / scale_factor), w-1)
                    resized[i, j, k] = img[src_i, src_j, k]
    else:
        h, w = img.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        resized = np.zeros((new_h, new_w))

        for i in range(new_h):
            for j in range(new_w):
                src_i = min(int(i / scale_factor), h-1)
                src_j = min(int(j / scale_factor), w-1)
                resized[i, j] = img[src_i, src_j]

    return resized

def gaussian_pyramid(img_path, levels=4):
    """
    问题76：高斯金字塔
    构建图像的高斯金字塔

    参数:
        img_path: 输入图像路径
        levels: 金字塔层数

    返回:
        高斯金字塔列表
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 转换为灰度图
    if len(img.shape) == 3:
        gray = np.mean(img, axis=2).astype(np.uint8)
    else:
        gray = img

    # 创建高斯核
    kernel = gaussian_kernel()

    # 构建金字塔
    pyramid = [gray]
    current = gray.copy()

    for _ in range(levels-1):
        # 高斯滤波
        filtered = manual_conv2d(current, kernel)
        # 下采样
        downsampled = manual_resize(filtered, 0.5)
        pyramid.append(downsampled)
        current = downsampled

    # 可视化结果
    result = []
    for level in pyramid:
        # 将图像调整为相同大小以便显示
        resized = manual_resize(level, (pyramid[0].shape[1]/level.shape[1]))
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        result.append(resized)

    return np.hstack(result)

def laplacian_pyramid(img_path, levels=4):
    """
    问题77：拉普拉斯金字塔
    构建图像的拉普拉斯金字塔

    参数:
        img_path: 输入图像路径
        levels: 金字塔层数

    返回:
        拉普拉斯金字塔列表
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 转换为灰度图
    if len(img.shape) == 3:
        gray = np.mean(img, axis=2).astype(np.uint8)
    else:
        gray = img

    # 创建高斯核
    kernel = gaussian_kernel()

    # 构建高斯金字塔
    gaussian_pyr = [gray]
    current = gray.copy()

    for _ in range(levels-1):
        filtered = manual_conv2d(current, kernel)
        downsampled = manual_resize(filtered, 0.5)
        gaussian_pyr.append(downsampled)
        current = downsampled

    # 构建拉普拉斯金字塔
    laplacian_pyr = []
    for i in range(levels-1):
        # 上采样
        upsampled = manual_resize(gaussian_pyr[i+1], 2.0)
        # 调整大小以匹配
        if upsampled.shape[0] > gaussian_pyr[i].shape[0]:
            upsampled = upsampled[:gaussian_pyr[i].shape[0], :]
        if upsampled.shape[1] > gaussian_pyr[i].shape[1]:
            upsampled = upsampled[:, :gaussian_pyr[i].shape[1]]
        # 计算差分
        diff = gaussian_pyr[i] - upsampled
        laplacian_pyr.append(diff)

    # 添加最后一层
    laplacian_pyr.append(gaussian_pyr[-1])

    # 可视化结果
    result = []
    for level in laplacian_pyr:
        # 将图像调整为相同大小以便显示
        resized = manual_resize(level, (laplacian_pyr[0].shape[1]/level.shape[1]))
        if len(resized.shape) == 2:
            resized = cv2.cvtColor(resized.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        result.append(resized)

    return np.hstack(result)

def image_blending(img_path1, img_path2, levels=4):
    """
    问题78：图像融合
    使用金字塔进行图像融合

    参数:
        img_path1: 第一张输入图像路径
        img_path2: 第二张输入图像路径
        levels: 金字塔层数

    返回:
        融合结果
    """
    # 读取图像
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    if img1 is None or img2 is None:
        raise ValueError("无法读取图像")

    # 确保图像大小相同
    if img1.shape != img2.shape:
        img2 = manual_resize(img2, (img1.shape[1]/img2.shape[1]))

    # 构建两个图像的拉普拉斯金字塔
    kernel = gaussian_kernel()

    def build_laplacian_pyramid(img):
        gaussian_pyr = [img]
        current = img.copy()

        for _ in range(levels-1):
            filtered = np.stack([manual_conv2d(current[..., i], kernel) for i in range(3)], axis=-1)
            downsampled = manual_resize(filtered, 0.5)
            gaussian_pyr.append(downsampled)
            current = downsampled

        laplacian_pyr = []
        for i in range(levels-1):
            upsampled = manual_resize(gaussian_pyr[i+1], 2.0)
            if upsampled.shape[0] > gaussian_pyr[i].shape[0]:
                upsampled = upsampled[:gaussian_pyr[i].shape[0], :]
            if upsampled.shape[1] > gaussian_pyr[i].shape[1]:
                upsampled = upsampled[:, :gaussian_pyr[i].shape[1]]
            diff = gaussian_pyr[i] - upsampled
            laplacian_pyr.append(diff)

        laplacian_pyr.append(gaussian_pyr[-1])
        return laplacian_pyr

    lap_pyr1 = build_laplacian_pyramid(img1)
    lap_pyr2 = build_laplacian_pyramid(img2)

    # 融合金字塔
    blended_pyr = []
    for la1, la2 in zip(lap_pyr1, lap_pyr2):
        blended_pyr.append((la1 + la2) * 0.5)

    # 重建图像
    reconstructed = blended_pyr[-1]
    for i in range(levels-2, -1, -1):
        upsampled = manual_resize(reconstructed, 2.0)
        if upsampled.shape[0] > blended_pyr[i].shape[0]:
            upsampled = upsampled[:blended_pyr[i].shape[0], :]
        if upsampled.shape[1] > blended_pyr[i].shape[1]:
            upsampled = upsampled[:, :blended_pyr[i].shape[1]]
        reconstructed = upsampled + blended_pyr[i]

    return reconstructed.astype(np.uint8)

def sift_scale_space(img_path, octaves=4, scales=5, sigma=1.6):
    """
    问题79：SIFT尺度空间
    构建SIFT算法的尺度空间

    参数:
        img_path: 输入图像路径
        octaves: 组数
        scales: 每组的尺度数
        sigma: 初始sigma值

    返回:
        尺度空间可视化结果
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 转换为灰度图
    if len(img.shape) == 3:
        gray = np.mean(img, axis=2).astype(np.uint8)
    else:
        gray = img

    # 上采样
    initial_image = manual_resize(gray, 2.0)

    # 构建尺度空间
    scale_space = []
    current = initial_image.copy()
    k = 2 ** (1/scales)

    for o in range(octaves):
        octave = []
        for s in range(scales):
            # 计算当前的sigma
            current_sigma = sigma * (k ** s)
            # 创建高斯核
            kernel_size = int(6 * current_sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = gaussian_kernel(kernel_size, current_sigma)
            # 高斯滤波
            blurred = manual_conv2d(current, kernel)
            octave.append(blurred)

        scale_space.append(octave)
        # 为下一个octave准备图像
        current = manual_resize(octave[-1], 0.5)

    # 可视化结果
    result = []
    for octave in scale_space:
        # 将每个octave的图像调整为相同大小
        octave_images = []
        for img in octave:
            resized = manual_resize(img, (scale_space[0][0].shape[1]/img.shape[1]))
            if len(resized.shape) == 2:
                resized = cv2.cvtColor(resized.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            octave_images.append(resized)
        result.append(np.hstack(octave_images))

    return np.vstack(result)

def saliency_detection(img_path, levels=4):
    """
    问题80：显著性检测
    基于金字塔的显著性检测

    参数:
        img_path: 输入图像路径
        levels: 金字塔层数

    返回:
        显著性图
    """
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图像: {img_path}")

    # 转换为Lab颜色空间
    # 这里我们简单地使用RGB通道的均值作为亮度
    l = np.mean(img, axis=2)

    # 构建高斯金字塔
    kernel = gaussian_kernel()
    pyramid = [l]
    current = l.copy()

    for _ in range(levels-1):
        filtered = manual_conv2d(current, kernel)
        downsampled = manual_resize(filtered, 0.5)
        pyramid.append(downsampled)
        current = downsampled

    # 计算显著性
    saliency = np.zeros_like(l)
    weights = [1/(i+1) for i in range(levels)]  # 不同层级的权重

    for i, level in enumerate(pyramid):
        # 将level调整到原始大小
        resized = manual_resize(level, (pyramid[0].shape[1]/level.shape[1]))
        if resized.shape[0] > saliency.shape[0]:
            resized = resized[:saliency.shape[0], :]
        if resized.shape[1] > saliency.shape[1]:
            resized = resized[:, :saliency.shape[1]]
        # 计算与平均值的差异
        mean_val = np.mean(resized)
        diff = np.abs(resized - mean_val)
        saliency += weights[i] * diff

    # 归一化
    saliency = (saliency - np.min(saliency)) / (np.max(saliency) - np.min(saliency)) * 255

    # 转换为彩色图像
    result = cv2.cvtColor(saliency.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    return result

def main(problem_id=76):
    """
    主函数，通过problem_id选择要运行的问题

    参数:
        problem_id: 问题编号，默认为76
    """
    input_path = "../images/imori.jpg"
    input_path2 = "../images/imori_noise.jpg"  # 用于图像融合

    if problem_id == 76:
        result = gaussian_pyramid(input_path)
        output_path = "../images/answer_76.jpg"
        title = "高斯金字塔"
    elif problem_id == 77:
        result = laplacian_pyramid(input_path)
        output_path = "../images/answer_77.jpg"
        title = "拉普拉斯金字塔"
    elif problem_id == 78:
        result = image_blending(input_path, input_path2)
        output_path = "../images/answer_78.jpg"
        title = "图像融合"
    elif problem_id == 79:
        result = sift_scale_space(input_path)
        output_path = "../images/answer_79.jpg"
        title = "SIFT尺度空间"
    elif problem_id == 80:
        result = saliency_detection(input_path)
        output_path = "../images/answer_80.jpg"
        title = "显著性检测"
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
    problem_id = int(sys.argv[1]) if len(sys.argv) > 1 else 76
    main(problem_id)