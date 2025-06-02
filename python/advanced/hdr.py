#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
高动态范围(HDR)图像生成算法的Python实现
"""

import numpy as np
import cv2
import random
from typing import List, Tuple, Optional

def weight_function(pixel_value: int) -> float:
    """
    权重函数，根据像素值计算其在融合中的权重

    Args:
        pixel_value: 像素值(0-255)

    Returns:
        权重值(0-1)
    """
    if pixel_value <= 0 or pixel_value >= 255:
        return 0.0
    else:
        # 使用三角形权重函数，中间值权重最高
        return 1.0 - abs(pixel_value - 128.0) / 128.0

def calculate_camera_response(images: List[np.ndarray],
                             exposure_times: List[float],
                             lambda_smooth: float = 10.0,
                             samples: int = 100) -> np.ndarray:
    """
    计算相机响应曲线

    Args:
        images: 输入的多张不同曝光的图像
        exposure_times: 对应的曝光时间数组
        lambda_smooth: 平滑度参数
        samples: 采样点数量

    Returns:
        相机响应曲线
    """
    if len(images) != len(exposure_times):
        raise ValueError("图像数量必须与曝光时间数量匹配")

    num_images = len(images)
    height, width = images[0].shape[:2]
    channels = images[0].shape[2] if len(images[0].shape) > 2 else 1

    # 随机选择样本点
    sample_points = []
    valid_samples = 0

    # 尝试找到在所有图像中都不是过曝或欠曝的点
    max_attempts = samples * 10
    attempts = 0

    while valid_samples < samples and attempts < max_attempts:
        # 随机选择一个点
        y = random.randint(0, height - 1)
        x = random.randint(0, width - 1)
        valid = True

        # 检查该点在所有图像中是否都不是极值
        for img in images:
            pixel = img[y, x]
            # 检查所有通道
            if np.any(pixel <= 5) or np.any(pixel >= 250):
                valid = False
                break

        if valid:
            sample_points.append((y, x))
            valid_samples += 1

        attempts += 1

    # 确保我们有足够的样本点
    if valid_samples < samples / 2:
        print(f"警告: 只找到 {valid_samples} 个有效样本点")

    # 为每个通道计算响应曲线
    response_curve = np.zeros((256, channels), dtype=np.float32)

    for c in range(channels):
        # 构建线性方程组
        # 方程数量 = 样本点数 * 图像数 + 254 (平滑约束)
        num_equations = len(sample_points) * num_images + 254
        A = np.zeros((num_equations, 256), dtype=np.float32)
        b = np.zeros((num_equations, 1), dtype=np.float32)

        # 设置样本点的方程
        eq_idx = 0
        for i, (y, x) in enumerate(sample_points):
            for j, (img, exposure) in enumerate(zip(images, exposure_times)):
                z = int(img[y, x, c] if channels > 1 else img[y, x])
                w = weight_function(z)

                A[eq_idx, z] = w
                A[eq_idx, 128] = -w  # 中间值作为参考
                b[eq_idx, 0] = w * np.log(exposure)
                eq_idx += 1

        # 添加平滑约束
        for i in range(254):
            w = weight_function(i + 1)
            A[eq_idx, i] = lambda_smooth * w
            A[eq_idx, i+1] = -2 * lambda_smooth * w
            A[eq_idx, i+2] = lambda_smooth * w
            eq_idx += 1

        # 固定中间值为0，以避免平凡解
        A[eq_idx, 128] = 1.0

        # 求解方程组
        x, res, rank, s = np.linalg.lstsq(A, b, rcond=None)

        # 保存响应曲线
        response_curve[:, c] = x.flatten()

    return response_curve

def create_hdr(images: List[np.ndarray],
               exposure_times: List[float],
               response_curve: Optional[np.ndarray] = None) -> np.ndarray:
    """
    创建HDR图像，融合多张不同曝光的图像

    Args:
        images: 输入的多张不同曝光的图像
        exposure_times: 对应的曝光时间数组
        response_curve: 可选的相机响应曲线（若为空则自动计算）

    Returns:
        HDR图像
    """
    if len(images) != len(exposure_times):
        raise ValueError("图像数量必须与曝光时间数量匹配")

    # 如果没有提供响应曲线，则计算它
    if response_curve is None:
        response_curve = calculate_camera_response(images, exposure_times)

    # 创建HDR图像
    height, width = images[0].shape[:2]
    channels = images[0].shape[2] if len(images[0].shape) > 2 else 1
    hdr_image = np.zeros((height, width, channels), dtype=np.float32)

    # 对每个像素进行加权合并
    for y in range(height):
        for x in range(width):
            sum_weights = np.zeros(channels)
            pixel_values = np.zeros(channels)

            # 遍历所有曝光图像
            for i, (img, exposure) in enumerate(zip(images, exposure_times)):
                pixel = img[y, x]

                for c in range(channels):
                    z = int(pixel[c] if channels > 1 else pixel)
                    w = weight_function(z)
                    # 使用响应曲线映射像素值到辐射度
                    radiance = response_curve[z, c] - np.log(exposure)

                    # 加权累加
                    pixel_values[c] += w * radiance
                    sum_weights[c] += w

            # 计算加权平均值
            for c in range(channels):
                if sum_weights[c] > 0:
                    hdr_image[y, x, c] = np.exp(pixel_values[c] / sum_weights[c])

    return hdr_image

def tone_mapping_global(hdr_image: np.ndarray,
                        key: float = 0.18,
                        white_point: float = 1.0) -> np.ndarray:
    """
    全局色调映射算法（Reinhard）

    Args:
        hdr_image: HDR图像
        key: 图像亮度参数
        white_point: 白点参数

    Returns:
        色调映射后的LDR图像
    """
    # 确保输入是3通道图像
    if len(hdr_image.shape) == 2:
        hdr_image = np.stack((hdr_image,) * 3, axis=-1)

    # 创建输出LDR图像
    ldr_image = np.zeros_like(hdr_image, dtype=np.uint8)

    # 计算亮度通道
    luminance = 0.2126 * hdr_image[:, :, 2] + 0.7152 * hdr_image[:, :, 1] + 0.0722 * hdr_image[:, :, 0]

    # 计算对数平均亮度
    epsilon = 1e-6
    valid_pixels = luminance > epsilon
    log_avg_luminance = np.exp(np.mean(np.log(luminance[valid_pixels] + epsilon)))

    # 缩放因子
    scale_factor = key / log_avg_luminance
    Lwhite2 = white_point * white_point

    # 应用Reinhard全局色调映射
    scaled_luminance = scale_factor * luminance
    mapped_luminance = (scaled_luminance * (1.0 + scaled_luminance / Lwhite2)) / (1.0 + scaled_luminance)

    # 保持色彩，修改亮度
    for c in range(3):
        color_ratio = np.ones_like(luminance)
        # 避免除以零
        color_ratio[valid_pixels] = hdr_image[:, :, c][valid_pixels] / luminance[valid_pixels]
        mapped_value = 255.0 * color_ratio * mapped_luminance
        ldr_image[:, :, c] = np.clip(mapped_value, 0, 255).astype(np.uint8)

    return ldr_image

def tone_mapping_local(hdr_image: np.ndarray,
                      sigma: float = 2.0,
                      contrast: float = 4.0) -> np.ndarray:
    """
    局部色调映射算法（Durand）

    Args:
        hdr_image: HDR图像
        sigma: 高斯滤波的标准差
        contrast: 对比度参数

    Returns:
        色调映射后的LDR图像
    """
    # 确保输入是3通道图像
    if len(hdr_image.shape) == 2:
        hdr_image = np.stack((hdr_image,) * 3, axis=-1)

    # 创建输出LDR图像
    ldr_image = np.zeros_like(hdr_image, dtype=np.uint8)

    # 计算亮度通道
    luminance = 0.2126 * hdr_image[:, :, 2] + 0.7152 * hdr_image[:, :, 1] + 0.0722 * hdr_image[:, :, 0]

    # 对亮度取对数
    epsilon = 1e-6
    log_luminance = np.log(luminance + epsilon)

    # 使用双边滤波分离基础层和细节层
    base_layer = cv2.bilateralFilter(log_luminance.astype(np.float32), 0, sigma * 3, sigma)

    # 计算细节层
    detail_layer = log_luminance - base_layer

    # 压缩基础层的动态范围
    log_min = np.min(base_layer)
    log_max = np.max(base_layer)
    compressed_base = (base_layer - log_max) * contrast / (log_max - log_min)

    # 重建压缩后的亮度（对数域）
    log_output = compressed_base + detail_layer

    # 从对数域返回线性域
    output_luminance = np.exp(log_output)

    # 保持色彩，修改亮度
    valid_pixels = luminance > epsilon
    for c in range(3):
        color_ratio = np.ones_like(luminance)
        color_ratio[valid_pixels] = hdr_image[:, :, c][valid_pixels] / luminance[valid_pixels]
        mapped_value = 255.0 * color_ratio * output_luminance
        ldr_image[:, :, c] = np.clip(mapped_value, 0, 255).astype(np.uint8)

    return ldr_image

def demo():
    """演示HDR算法的使用"""
    try:
        # 加载多曝光图像
        # 这里假设有3张不同曝光的图像: low.jpg, mid.jpg, high.jpg
        image_paths = ["low.jpg", "mid.jpg", "high.jpg"]
        exposure_times = [1/30.0, 1/8.0, 1.0]  # 对应的曝光时间

        images = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                print(f"无法加载图像: {path}")
                return
            images.append(img)

        # 创建HDR图像
        hdr_image = create_hdr(images, exposure_times)

        # 应用全局色调映射
        global_result = tone_mapping_global(hdr_image)

        # 应用局部色调映射
        local_result = tone_mapping_local(hdr_image)

        # 保存结果
        cv2.imwrite("hdr_global.jpg", global_result)
        cv2.imwrite("hdr_local.jpg", local_result)

        print("HDR处理完成，结果已保存为 hdr_global.jpg 和 hdr_local.jpg")

    except Exception as e:
        print(f"处理过程中出错: {str(e)}")

if __name__ == "__main__":
    demo()