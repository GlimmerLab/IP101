#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
导向滤波去雾算法 - Python实现
Guided Filter Dehazing Algorithm

基于导向滤波优化的暗通道去雾技术，
通过导向滤波细化透射率图，实现高质量的图像去雾。

Author: GlimmerLab
Date: 2024
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import time

class DehazingMethod(Enum):
    """去雾方法枚举"""
    GUIDED_FILTER_DEFOGGING = "guided_filter_defogging"    # 导向滤波去雾
    KAIMING_HE_GUIDED = "kaiming_he_guided"               # 何凯明导向滤波去雾
    FAST_GUIDED_DEFOGGING = "fast_guided_defogging"       # 快速导向滤波去雾

@dataclass
class GuidedDefoggingParams:
    """导向滤波去雾参数配置"""
    radius: int = 8               # 导向滤波半径
    eps: float = 0.01             # 导向滤波正则化参数
    omega: float = 0.95           # 雾霾保留系数 (0.8-1.0)
    t0: float = 0.1              # 最小透射率阈值 (0.05-0.3)
    patch_size: int = 15          # 暗通道计算窗口大小
    atmospheric_ratio: float = 0.001  # 大气光估计像素比例
    subsample: int = 1            # 下采样比例（快速模式）

    def __post_init__(self):
        """参数验证"""
        if not 5 <= self.patch_size <= 51 or self.patch_size % 2 == 0:
            raise ValueError("窗口大小必须是5-51的奇数")
        if not 0.8 <= self.omega <= 1.0:
            raise ValueError("雾霾保留系数必须在0.8-1.0范围内")
        if not 0.05 <= self.t0 <= 0.3:
            raise ValueError("最小透射率必须在0.05-0.3范围内")
        if self.radius <= 0:
            raise ValueError("导向滤波半径必须为正数")

class GuidedFilterDehazer:
    """导向滤波去雾处理类"""

    def __init__(self):
        """初始化去雾器"""
        pass

    def dehaze(self, image: np.ndarray,
              method: DehazingMethod = DehazingMethod.GUIDED_FILTER_DEFOGGING,
              params: Optional[GuidedDefoggingParams] = None) -> np.ndarray:
        """
        图像去雾主函数

        Args:
            image: 输入有雾图像 (BGR格式)
            method: 去雾方法
            params: 去雾参数

        Returns:
            去雾后的图像
        """
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")

        if params is None:
            params = GuidedDefoggingParams()

        if method == DehazingMethod.GUIDED_FILTER_DEFOGGING:
            return self._guided_filter_defogging(image, params)
        elif method == DehazingMethod.KAIMING_HE_GUIDED:
            return self._kaiming_he_guided_defogging(image, params)
        elif method == DehazingMethod.FAST_GUIDED_DEFOGGING:
            return self._fast_guided_defogging(image, params)
        else:
            raise ValueError(f"不支持的去雾方法: {method}")

    def _compute_dark_channel(self, image: np.ndarray, patch_size: int) -> np.ndarray:
        """
        计算暗通道

        Args:
            image: 输入图像 (H, W, 3)
            patch_size: 邻域窗口大小

        Returns:
            暗通道图像 (H, W)
        """
        # 计算每个像素的RGB最小值
        min_channel = np.min(image, axis=2)

        # 使用形态学腐蚀操作实现邻域最小值
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
        dark_channel = cv2.erode(min_channel, kernel)

        return dark_channel

    def _estimate_atmospheric_light(self, image: np.ndarray,
                                  dark_channel: np.ndarray,
                                  ratio: float = 0.001) -> np.ndarray:
        """
        估算大气光值

        Args:
            image: 原始图像
            dark_channel: 暗通道图像
            ratio: 选取像素比例

        Returns:
            大气光值 (3,) BGR格式
        """
        h, w = dark_channel.shape
        num_pixels = h * w
        num_brightest = max(1, int(num_pixels * ratio))

        # 将图像展平便于处理
        dark_vec = dark_channel.reshape(num_pixels)
        image_vec = image.reshape(num_pixels, 3)

        # 获取暗通道中最亮像素的索引
        indices = np.argpartition(dark_vec, -num_brightest)[-num_brightest:]

        # 在候选位置中找到原图最亮的像素
        brightest_pixels = image_vec[indices]
        brightest_intensities = np.sum(brightest_pixels, axis=1)
        brightest_idx = indices[np.argmax(brightest_intensities)]

        atmospheric_light = image_vec[brightest_idx].astype(np.float64)

        return atmospheric_light

    def _estimate_transmission(self, image: np.ndarray,
                             atmospheric_light: np.ndarray,
                             omega: float,
                             patch_size: int) -> np.ndarray:
        """
        计算透射率

        Args:
            image: 输入图像
            atmospheric_light: 大气光值
            omega: 雾霾保留系数
            patch_size: 邻域窗口大小

        Returns:
            透射率图 (H, W)
        """
        # 归一化图像和大气光
        norm_image = image.astype(np.float64) / 255.0
        norm_A = atmospheric_light / 255.0

        # 防止除零错误
        norm_A = np.maximum(norm_A, 1e-6)

        # 计算 I(x)/A
        ratio_image = norm_image / norm_A

        # 将比值图像转换为8位用于暗通道计算
        ratio_uint8 = np.clip(ratio_image * 255, 0, 255).astype(np.uint8)

        # 计算比值图像的暗通道
        dark_channel = self._compute_dark_channel(ratio_uint8, patch_size)
        dark_channel = dark_channel.astype(np.float64) / 255.0

        # 计算透射率: t(x) = 1 - ω * dark_channel(I(x)/A)
        transmission = 1.0 - omega * dark_channel

        return transmission

    def _guided_filter(self, guide: np.ndarray, src: np.ndarray,
                      radius: int, epsilon: float) -> np.ndarray:
        """
        导向滤波实现

        Args:
            guide: 导向图像
            src: 输入图像
            radius: 滤波半径
            epsilon: 正则化参数

        Returns:
            滤波后图像
        """
        # 确保输入为单通道
        if len(guide.shape) == 3:
            guide = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)
        if len(src.shape) == 3:
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        # 转换为浮点型
        guide = guide.astype(np.float64) / 255.0
        src = src.astype(np.float64) / 255.0

        # 计算均值
        mean_I = cv2.boxFilter(guide, cv2.CV_64F, (radius, radius))
        mean_p = cv2.boxFilter(src, cv2.CV_64F, (radius, radius))
        corr_Ip = cv2.boxFilter(guide * src, cv2.CV_64F, (radius, radius))

        # 计算协方差和方差
        cov_Ip = corr_Ip - mean_I * mean_p
        mean_II = cv2.boxFilter(guide * guide, cv2.CV_64F, (radius, radius))
        var_I = mean_II - mean_I * mean_I

        # 计算线性系数
        a = cov_Ip / (var_I + epsilon)
        b = mean_p - a * mean_I

        # 对系数进行平滑
        mean_a = cv2.boxFilter(a, cv2.CV_64F, (radius, radius))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (radius, radius))

        # 输出结果
        result = mean_a * guide + mean_b

        return result

    def _recover_scene(self, image: np.ndarray,
                     transmission: np.ndarray,
                     atmospheric_light: np.ndarray,
                      t0: float) -> np.ndarray:
        """
        场景复原

        Args:
            image: 有雾图像
            transmission: 透射率图
            atmospheric_light: 大气光值
            t0: 最小透射率阈值

        Returns:
            去雾后的图像
        """
        # 确保透射率不会过小
        transmission = np.maximum(transmission, t0)

        # 场景复原公式: J(x) = (I(x) - A) / t(x) + A
        image_float = image.astype(np.float64)
        atmospheric_light = atmospheric_light.reshape(1, 1, 3)
        transmission = transmission[:, :, np.newaxis]

        recovered = (image_float - atmospheric_light) / transmission + atmospheric_light

        # 限制像素值范围并转换为8位
        recovered = np.clip(recovered, 0, 255).astype(np.uint8)

        return recovered

    def _guided_filter_defogging(self, image: np.ndarray,
                               params: GuidedDefoggingParams) -> np.ndarray:
        """
        标准导向滤波去雾

        Args:
            image: 输入图像
            params: 去雾参数

        Returns:
            去雾后的图像
        """
        # 步骤1: 计算暗通道
        dark_channel = self._compute_dark_channel(image, params.patch_size)

        # 步骤2: 估算大气光值
        atmospheric_light = self._estimate_atmospheric_light(
            image, dark_channel, params.atmospheric_ratio)

        # 步骤3: 计算粗透射率
        transmission = self._estimate_transmission(
            image, atmospheric_light, params.omega, params.patch_size)

        # 步骤4: 使用导向滤波细化透射率
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        refined_transmission = self._guided_filter(
            gray, transmission, params.radius, params.eps)

        # 步骤5: 场景复原
        result = self._recover_scene(image, refined_transmission, atmospheric_light, params.t0)

        return result

    def _kaiming_he_guided_defogging(self, image: np.ndarray,
                                   params: GuidedDefoggingParams) -> np.ndarray:
        """
        何凯明导向滤波去雾（改进版本）

        Args:
            image: 输入图像
            params: 去雾参数

        Returns:
            去雾后的图像
        """
        # 步骤1-3: 与标准方法相同
        dark_channel = self._compute_dark_channel(image, params.patch_size)
        atmospheric_light = self._estimate_atmospheric_light(
            image, dark_channel, params.atmospheric_ratio)
        transmission = self._estimate_transmission(
            image, atmospheric_light, params.omega, params.patch_size)

        # 步骤4: 使用图像本身作为导向图像进行滤波
        guidance = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 使用导向滤波细化透射率
        refined_transmission = self._guided_filter(
            guidance, transmission, params.radius, params.eps)

        # 步骤5: 场景复原
        result = self._recover_scene(image, refined_transmission, atmospheric_light, params.t0)

        return result

    def _fast_guided_defogging(self, image: np.ndarray,
                             params: GuidedDefoggingParams) -> np.ndarray:
        """
        快速导向滤波去雾（降采样优化）

        Args:
            image: 输入图像
            params: 去雾参数

        Returns:
            去雾后的图像
        """
        # 降采样加速处理
        scale_factor = 1.0 / params.subsample
        small_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

        # 在低分辨率图像上执行去雾
        small_params = GuidedDefoggingParams(
            patch_size=max(3, params.patch_size // 2),
            omega=params.omega,
            t0=params.t0,
            radius=max(4, params.radius // 2),
            eps=params.eps
        )

        # 在低分辨率上计算暗通道和大气光
        dark_channel = self._compute_dark_channel(small_image, small_params.patch_size)
        atmospheric_light = self._estimate_atmospheric_light(
            small_image, dark_channel, params.atmospheric_ratio)

        # 在原始分辨率上计算透射率
        transmission = self._estimate_transmission(
            image, atmospheric_light, params.omega, params.patch_size)

        # 导向滤波优化
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        refined_transmission = self._guided_filter(
            gray, transmission, small_params.radius, small_params.eps)

        # 场景复原
        result = self._recover_scene(image, refined_transmission, atmospheric_light, params.t0)

        return result

    def process_with_intermediate_results(self, image: np.ndarray,
                                        method: DehazingMethod = DehazingMethod.GUIDED_FILTER_DEFOGGING,
                                        params: Optional[GuidedDefoggingParams] = None) -> Dict[str, np.ndarray]:
        """
        处理图像并返回中间结果

        Args:
            image: 输入图像
            method: 去雾方法
            params: 去雾参数

        Returns:
            包含所有中间结果的字典
        """
        if params is None:
            params = GuidedDefoggingParams()

        # 计算暗通道
        dark_channel = self._compute_dark_channel(image, params.patch_size)

        # 估算大气光值
        atmospheric_light = self._estimate_atmospheric_light(
            image, dark_channel, params.atmospheric_ratio)

        # 计算透射率
        transmission = self._estimate_transmission(
            image, atmospheric_light, params.omega, params.patch_size)

        # 导向滤波优化透射率
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        refined_transmission = self._guided_filter(
            gray, transmission, params.radius, params.eps)

        # 场景复原
        dehazed = self._recover_scene(image, refined_transmission, atmospheric_light, params.t0)

        return {
            'original': image,
            'dark_channel': dark_channel,
            'transmission': transmission,
            'refined_transmission': refined_transmission,
            'atmospheric_light': atmospheric_light,
            'dehazed': dehazed
        }

    def analyze_image_quality(self, original: np.ndarray, dehazed: np.ndarray) -> Dict[str, float]:
        """
        分析去雾效果质量指标

        Args:
            original: 原始有雾图像
            dehazed: 去雾后图像

        Returns:
            质量指标字典
        """
        # 转换为灰度图进行分析
        if len(original.shape) == 3:
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            dehazed_gray = cv2.cvtColor(dehazed, cv2.COLOR_BGR2GRAY)
        else:
            orig_gray = original
            dehazed_gray = dehazed

        # 对比度增强比
        orig_contrast = np.std(orig_gray)
        dehazed_contrast = np.std(dehazed_gray)
        contrast_enhancement = dehazed_contrast / (orig_contrast + 1e-6)

        # 平均梯度增强
        def calculate_average_gradient(img):
            grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            gradient = np.sqrt(grad_x**2 + grad_y**2)
            return np.mean(gradient)

        orig_gradient = calculate_average_gradient(orig_gray)
        dehazed_gradient = calculate_average_gradient(dehazed_gray)
        gradient_enhancement = dehazed_gradient / (orig_gradient + 1e-6)

        # 信息熵
        def calculate_entropy(img):
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-6)
            hist_nonzero = hist[hist > 0]
            return -np.sum(hist_nonzero * np.log2(hist_nonzero))

        orig_entropy = calculate_entropy(orig_gray)
        dehazed_entropy = calculate_entropy(dehazed_gray)
        entropy_ratio = dehazed_entropy / (orig_entropy + 1e-6)

        # 可见距离增强（基于边缘密度）
        def calculate_edge_density(img):
            edges = cv2.Canny(img, 50, 150)
            return np.sum(edges > 0) / edges.size

        orig_edge_density = calculate_edge_density(orig_gray)
        dehazed_edge_density = calculate_edge_density(dehazed_gray)
        visibility_enhancement = dehazed_edge_density / (orig_edge_density + 1e-6)

        return {
            'contrast_enhancement': contrast_enhancement,
            'gradient_enhancement': gradient_enhancement,
            'entropy_ratio': entropy_ratio,
            'visibility_enhancement': visibility_enhancement
        }

    def compare_methods(self, image: np.ndarray,
                       params: Optional[GuidedDefoggingParams] = None) -> Dict[str, Dict[str, Any]]:
        """
        比较不同去雾方法的效果

        Args:
            image: 输入图像
            params: 去雾参数

        Returns:
            各方法的处理结果和质量指标
        """
        if params is None:
            params = GuidedDefoggingParams()

        methods = [
            DehazingMethod.GUIDED_FILTER_DEFOGGING,
            DehazingMethod.KAIMING_HE_GUIDED,
            DehazingMethod.FAST_GUIDED_DEFOGGING
        ]

        results = {}

        for method in methods:
            try:
                start_time = time.time()
                dehazed = self.dehaze(image, method, params)
                processing_time = time.time() - start_time

                # 计算质量指标
                quality_metrics = self.analyze_image_quality(image, dehazed)

                results[method.value] = {
                    'image': dehazed,
                    'time': processing_time,
                    'quality': quality_metrics
                }

            except Exception as e:
                print(f"{method.value} 方法处理失败: {e}")

        return results


def demo_guided_filter_defogging():
    """演示导向滤波去雾功能"""
    print("=== 导向滤波去雾演示 ===")

    # 创建去雾器
    dehazer = GuidedFilterDehazer()

    # 创建测试图像
    test_image = create_hazy_test_image()

    print(f"测试图像尺寸: {test_image.shape}")

    # 测试不同方法
    methods = [
        DehazingMethod.GUIDED_FILTER_DEFOGGING,
        DehazingMethod.KAIMING_HE_GUIDED,
        DehazingMethod.FAST_GUIDED_DEFOGGING
    ]

    for method in methods:
        print(f"\n测试 {method.value} 方法:")
        start_time = time.time()
        result = dehazer.dehaze(test_image, method)
        processing_time = time.time() - start_time

        print(f"  处理时间: {processing_time:.3f}s")
        print(f"  输出图像形状: {result.shape}")

    print("导向滤波去雾演示完成")


def create_hazy_test_image(size: Tuple[int, int] = (300, 400)) -> np.ndarray:
    """
    创建有雾测试图像

    Args:
        size: 图像尺寸 (高度, 宽度)

    Returns:
        有雾测试图像
    """
    height, width = size

    # 创建清晰的基础图像
    clear_image = np.zeros((height, width, 3), dtype=np.uint8)

    # 添加一些几何图形
    cv2.rectangle(clear_image, (50, 100), (150, 200), (80, 120, 160), -1)
    cv2.circle(clear_image, (300, 80), 30, (60, 100, 140), -1)
    cv2.rectangle(clear_image, (250, 200), (350, 280), (100, 160, 200), -1)

    # 添加渐变背景
    for y in range(height):
        for x in range(width):
            brightness = int(100 + 100 * (x / width))
            clear_image[y, x] = np.maximum(clear_image[y, x],
                                         [brightness-20, brightness, brightness+20])

    # 模拟雾霾效果
    atmospheric_light = np.array([200, 210, 220])  # 大气光

    # 创建距离相关的透射率
    center_x, center_y = width // 2, height // 2
    max_distance = np.sqrt((width/2)**2 + (height/2)**2)

    hazy_image = np.zeros_like(clear_image, dtype=np.float64)

    for y in range(height):
        for x in range(width):
            # 计算到中心的距离
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)

            # 透射率随距离衰减
            transmission = 0.6 + 0.35 * (1 - distance / max_distance)
            transmission = np.clip(transmission, 0.2, 0.95)

            # 应用雾霾模型
            clear_pixel = clear_image[y, x].astype(np.float64)
            hazy_pixel = clear_pixel * transmission + atmospheric_light * (1 - transmission)
            hazy_image[y, x] = hazy_pixel

    return np.clip(hazy_image, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    demo_guided_filter_defogging()