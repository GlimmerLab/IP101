#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
暗通道去雾算法 - Python实现
Dark Channel Prior Dehazing Algorithm

基于何凯明等人2009年论文的单幅图像去雾算法，
通过暗通道先验约束实现雾霾环境下的场景复原。

Author: GlimmerLab
Date: 2024
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict, Any, Union
from dataclasses import dataclass
import time
import argparse
from pathlib import Path
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')


class DehazingMethod(Enum):
    """去雾方法枚举"""
    DARK_CHANNEL = "dark_channel"        # 标准暗通道
    REFINED_DARK_CHANNEL = "refined"     # 导向滤波优化版本
    FAST_DARK_CHANNEL = "fast"           # 快速近似版本
    ADAPTIVE_DARK_CHANNEL = "adaptive"   # 自适应参数版本


@dataclass
class DehazingParams:
    """去雾参数配置"""
    patch_size: int = 15                 # 暗通道计算窗口大小
    omega: float = 0.95                  # 雾霾保留系数 (0.8-1.0)
    t0: float = 0.1                     # 最小透射率阈值 (0.05-0.3)
    guided_radius: int = 60              # 导向滤波半径
    guided_epsilon: float = 1e-3         # 导向滤波正则化参数
    atmospheric_ratio: float = 0.001     # 大气光估计像素比例
    parallel_workers: int = 4            # 并行工作线程数

    def __post_init__(self):
        """参数验证"""
        if not 5 <= self.patch_size <= 51 or self.patch_size % 2 == 0:
            raise ValueError("窗口大小必须是5-51的奇数")
        if not 0.8 <= self.omega <= 1.0:
            raise ValueError("雾霾保留系数必须在0.8-1.0范围内")
        if not 0.05 <= self.t0 <= 0.3:
            raise ValueError("最小透射率必须在0.05-0.3范围内")
        if self.guided_radius <= 0:
            raise ValueError("导向滤波半径必须为正数")


class DarkChannelDehazer:
    """暗通道去雾处理类"""

    def __init__(self):
        """初始化去雾器"""
        print("暗通道去雾器初始化完成")

    def dehaze(self, image: np.ndarray,
              method: DehazingMethod = DehazingMethod.REFINED_DARK_CHANNEL,
              params: Optional[DehazingParams] = None) -> np.ndarray:
        """
        图像去雾主函数

        Args:
            image: 输入有雾图像 (BGR格式)
            method: 去雾方法
            params: 去雾参数

        Returns:
            去雾后的图像

        Raises:
            ValueError: 当输入图像为空时
        """
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")

        if params is None:
            params = DehazingParams()

        print(f"开始{method.value}去雾处理")

        if method == DehazingMethod.DARK_CHANNEL:
            return self._standard_dark_channel_dehaze(image, params)
        elif method == DehazingMethod.REFINED_DARK_CHANNEL:
            return self._refined_dark_channel_dehaze(image, params)
        elif method == DehazingMethod.FAST_DARK_CHANNEL:
            return self._fast_dark_channel_dehaze(image, params)
        elif method == DehazingMethod.ADAPTIVE_DARK_CHANNEL:
            return self._adaptive_dark_channel_dehaze(image, params)
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

    def _compute_transmission(self, image: np.ndarray,
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

    def _standard_dark_channel_dehaze(self, image: np.ndarray,
                                    params: DehazingParams) -> np.ndarray:
        """
        标准暗通道去雾

        Args:
            image: 输入图像
            params: 去雾参数

        Returns:
            去雾后的图像
        """
        print("执行标准暗通道去雾")

        # 步骤1: 计算暗通道
        dark_channel = self._compute_dark_channel(image, params.patch_size)

        # 步骤2: 估算大气光值
        atmospheric_light = self._estimate_atmospheric_light(
            image, dark_channel, params.atmospheric_ratio)

        # 步骤3: 计算透射率
        transmission = self._compute_transmission(
            image, atmospheric_light, params.omega, params.patch_size)

        # 步骤4: 场景复原
        result = self._recover_scene(image, transmission, atmospheric_light, params.t0)

        return result

    def _refined_dark_channel_dehaze(self, image: np.ndarray,
                                   params: DehazingParams) -> np.ndarray:
        """
        导向滤波优化的暗通道去雾

        Args:
            image: 输入图像
            params: 去雾参数

        Returns:
            去雾后的图像
        """
        print("执行导向滤波优化暗通道去雾")

        # 前三步与标准方法相同
        dark_channel = self._compute_dark_channel(image, params.patch_size)
        atmospheric_light = self._estimate_atmospheric_light(
            image, dark_channel, params.atmospheric_ratio)
        transmission = self._compute_transmission(
            image, atmospheric_light, params.omega, params.patch_size)

        # 步骤4: 使用导向滤波优化透射率
        refined_transmission = self._guided_filter(
            image, transmission, params.guided_radius, params.guided_epsilon)

        # 步骤5: 场景复原
        result = self._recover_scene(image, refined_transmission, atmospheric_light, params.t0)

        return result

    def _fast_dark_channel_dehaze(self, image: np.ndarray,
                                params: DehazingParams) -> np.ndarray:
        """
        快速暗通道去雾（降采样优化）

        Args:
            image: 输入图像
            params: 去雾参数

        Returns:
            去雾后的图像
        """
        print("执行快速暗通道去雾")

        # 降采样加速处理
        scale_factor = 0.5
        small_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

        # 在低分辨率图像上执行去雾
        small_params = DehazingParams(
            patch_size=max(3, params.patch_size // 2),
            omega=params.omega,
            t0=params.t0,
            guided_radius=max(15, params.guided_radius // 2),
            guided_epsilon=params.guided_epsilon
        )

        dark_channel = self._compute_dark_channel(small_image, small_params.patch_size)
        atmospheric_light = self._estimate_atmospheric_light(
            small_image, dark_channel, params.atmospheric_ratio)

        # 在原始分辨率上计算透射率
        transmission = self._compute_transmission(
            image, atmospheric_light, params.omega, params.patch_size)

        # 导向滤波优化
        refined_transmission = self._guided_filter(
            image, transmission, small_params.guided_radius, small_params.guided_epsilon)

        # 场景复原
        result = self._recover_scene(image, refined_transmission, atmospheric_light, params.t0)

        return result

    def _adaptive_dark_channel_dehaze(self, image: np.ndarray,
                                    params: DehazingParams) -> np.ndarray:
        """
        自适应参数暗通道去雾

        Args:
            image: 输入图像
            params: 去雾参数

        Returns:
            去雾后的图像
        """
        print("执行自适应参数暗通道去雾")

        # 分析图像特征调整参数
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 基于图像统计调整参数
        mean_brightness = np.mean(gray) / 255.0
        contrast = np.std(gray) / 255.0

        # 自适应调整omega参数
        adaptive_omega = params.omega
        if mean_brightness < 0.3:  # 暗图像
            adaptive_omega = min(0.98, params.omega + 0.02)
        elif mean_brightness > 0.7:  # 亮图像
            adaptive_omega = max(0.90, params.omega - 0.02)

        # 自适应调整t0参数
        adaptive_t0 = params.t0
        if contrast < 0.1:  # 低对比度
            adaptive_t0 = max(0.05, params.t0 - 0.02)

        print(f"自适应参数: omega={adaptive_omega:.3f}, t0={adaptive_t0:.3f}")

        # 使用自适应参数执行去雾
        adaptive_params = DehazingParams(
            patch_size=params.patch_size,
            omega=adaptive_omega,
            t0=adaptive_t0,
            guided_radius=params.guided_radius,
            guided_epsilon=params.guided_epsilon
        )

        return self._refined_dark_channel_dehaze(image, adaptive_params)

    def process_with_intermediate_results(self, image: np.ndarray,
                                        method: DehazingMethod = DehazingMethod.REFINED_DARK_CHANNEL,
                                        params: Optional[DehazingParams] = None) -> Dict[str, np.ndarray]:
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
            params = DehazingParams()

        print("开始处理并保存中间结果")

        # 计算暗通道
        dark_channel = self._compute_dark_channel(image, params.patch_size)

        # 估算大气光值
        atmospheric_light = self._estimate_atmospheric_light(
            image, dark_channel, params.atmospheric_ratio)

        # 计算透射率
        transmission = self._compute_transmission(
            image, atmospheric_light, params.omega, params.patch_size)

        # 根据方法选择是否进行导向滤波
        if method in [DehazingMethod.REFINED_DARK_CHANNEL, DehazingMethod.FAST_DARK_CHANNEL]:
            refined_transmission = self._guided_filter(
                image, transmission, params.guided_radius, params.guided_epsilon)
        else:
            refined_transmission = transmission

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
            'visibility_enhancement': visibility_enhancement,
            'atmospheric_light_estimate': None  # 在调用时设置
        }

    def compare_methods(self, image: np.ndarray,
                       params: Optional[DehazingParams] = None,
                       save_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        比较不同去雾方法的效果

        Args:
            image: 输入图像
            params: 去雾参数
            save_path: 保存路径

        Returns:
            各方法的处理结果和质量指标
        """
        if params is None:
            params = DehazingParams()

        methods = [
            DehazingMethod.DARK_CHANNEL,
            DehazingMethod.REFINED_DARK_CHANNEL,
            DehazingMethod.FAST_DARK_CHANNEL,
            DehazingMethod.ADAPTIVE_DARK_CHANNEL
        ]

        results = {}

        print("开始方法比较测试")

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

                print(f"{method.value} 方法:")
                print(f"  处理时间: {processing_time:.3f}s")
                print(f"  对比度增强: {quality_metrics['contrast_enhancement']:.2f}x")
                print(f"  梯度增强: {quality_metrics['gradient_enhancement']:.2f}x")
                print(f"  可见度提升: {quality_metrics['visibility_enhancement']:.2f}x")

            except Exception as e:
                print(f"{method.value} 方法处理失败: {e}")

        # 可视化结果
        if results:
            self._visualize_comparison(image, results, save_path)

        return results

    def _visualize_comparison(self, original: np.ndarray,
                            results: Dict[str, Dict[str, Any]],
                            save_path: Optional[str] = None) -> None:
        """
        可视化比较结果

        Args:
            original: 原始图像
            results: 处理结果
            save_path: 保存路径
        """
        num_images = len(results) + 1
        cols = min(3, num_images)
        rows = (num_images + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))

        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()

        # 显示原图
        if len(original.shape) == 3:
            axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        else:
            axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')

        # 显示处理结果
        for i, (method, data) in enumerate(results.items(), 1):
            if i < len(axes):
                if len(data['image'].shape) == 3:
                    axes[i].imshow(cv2.cvtColor(data['image'], cv2.COLOR_BGR2RGB))
                else:
                    axes[i].imshow(data['image'], cmap='gray')

                title = f"{method}\nTime: {data['time']:.3f}s"
                if 'quality' in data:
                    title += f"\nContrast: {data['quality']['contrast_enhancement']:.2f}x"

                axes[i].set_title(title, fontsize=10)
                axes[i].axis('off')

        # 隐藏多余的子图
        for i in range(num_images, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"比较结果已保存至: {save_path}")

        plt.show()

    def performance_test(self, image: np.ndarray,
                        params: Optional[DehazingParams] = None,
                        iterations: int = 5) -> Dict[str, float]:
        """
        性能基准测试

    Args:
            image: 输入图像
            params: 去雾参数
            iterations: 测试迭代次数

        Returns:
            性能统计结果
        """
        if params is None:
            params = DehazingParams()

        print(f"开始性能测试 - 图像尺寸: {image.shape}")

        methods = [
            DehazingMethod.DARK_CHANNEL,
            DehazingMethod.REFINED_DARK_CHANNEL,
            DehazingMethod.FAST_DARK_CHANNEL
        ]

        performance_results = {}

        for method in methods:
            times = []

            for _ in range(iterations):
                start_time = time.time()
                try:
                    self.dehaze(image, method, params)
                    elapsed_time = time.time() - start_time
                    times.append(elapsed_time)
                except Exception as e:
                    print(f"{method.value} 方法测试失败: {e}")
                    break

            if times:
                avg_time = np.mean(times)
                std_time = np.std(times)
                performance_results[method.value] = {
                    'avg_time': avg_time,
                    'std_time': std_time,
                    'fps': 1.0 / avg_time if avg_time > 0 else 0.0
                }

                print(f"{method.value}: {avg_time*1000:.1f}ms (±{std_time*1000:.1f}ms), {1.0/avg_time:.1f} FPS")

        return performance_results


def demo_basic_dehazing():
    """演示基础去雾功能"""
    print("=== 基础去雾演示 ===")

    # 创建去雾器
    dehazer = DarkChannelDehazer()

    # 创建测试图像
    test_image = create_hazy_test_image()

    print(f"测试图像尺寸: {test_image.shape}")

    # 测试不同方法
    methods = [
        DehazingMethod.DARK_CHANNEL,
        DehazingMethod.REFINED_DARK_CHANNEL,
        DehazingMethod.FAST_DARK_CHANNEL
    ]

    for method in methods:
        print(f"\n测试 {method.value} 方法:")
        start_time = time.time()
        result = dehazer.dehaze(test_image, method)
        processing_time = time.time() - start_time

        print(f"  处理时间: {processing_time:.3f}s")
        print(f"  输出图像形状: {result.shape}")

    print("基础演示完成")


def demo_intermediate_results():
    """演示中间结果可视化"""
    print("=== 中间结果演示 ===")

    dehazer = DarkChannelDehazer()
    test_image = create_hazy_test_image()

    # 获取中间结果
    results = dehazer.process_with_intermediate_results(test_image)

    # 可视化中间结果
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    # 显示各个步骤的结果
    images = [
        ('Original', results['original']),
        ('Dark Channel', results['dark_channel']),
        ('Transmission', results['transmission']),
        ('Refined Transmission', results['refined_transmission']),
        ('Dehazed', results['dehazed'])
    ]

    for i, (title, img) in enumerate(images):
        if i < len(axes):
            if len(img.shape) == 3:
                axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                axes[i].imshow(img, cmap='gray')
            axes[i].set_title(title)
            axes[i].axis('off')

    # 隐藏最后一个子图
    axes[-1].axis('off')

    plt.tight_layout()
    plt.show()

    print(f"大气光值: {results['atmospheric_light']}")
    print("中间结果演示完成")


def demo_method_comparison():
    """演示方法比较功能"""
    print("=== 方法比较演示 ===")

    dehazer = DarkChannelDehazer()

    # 创建不同类型的测试图像
    test_images = {
        'dense_haze': create_dense_hazy_image(),
        'light_haze': create_light_hazy_image(),
        'normal_haze': create_hazy_test_image()
    }

    for img_type, image in test_images.items():
        print(f"\n测试 {img_type} 图像:")
        results = dehazer.compare_methods(image)

        # 找出最佳方法
        if results:
            best_method = max(results.keys(),
                            key=lambda k: results[k]['quality']['contrast_enhancement'])
            print(f"最佳方法: {best_method}")

    print("方法比较演示完成")


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
    # 简单的雾霾模型: I = J*t + A*(1-t)
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


def create_dense_hazy_image() -> np.ndarray:
    """创建浓雾图像"""
    base_image = create_hazy_test_image()

    # 增加雾霾浓度
    atmospheric_light = np.array([220, 225, 230])
    transmission = 0.3  # 低透射率

    dense_hazy = base_image.astype(np.float64) * transmission + atmospheric_light * (1 - transmission)
    return np.clip(dense_hazy, 0, 255).astype(np.uint8)


def create_light_hazy_image() -> np.ndarray:
    """创建轻雾图像"""
    base_image = create_hazy_test_image()

    # 减少雾霾浓度
    atmospheric_light = np.array([180, 185, 190])
    transmission = 0.8  # 高透射率

    light_hazy = base_image.astype(np.float64) * transmission + atmospheric_light * (1 - transmission)
    return np.clip(light_hazy, 0, 255).astype(np.uint8)


def performance_benchmark():
    """性能基准测试"""
    print("=== 性能基准测试 ===")

    # 测试不同图像尺寸
    test_sizes = [(200, 300), (400, 600), (600, 800)]

    dehazer = DarkChannelDehazer()

    for size in test_sizes:
        print(f"\n测试图像尺寸: {size[0]}x{size[1]}")

        # 创建指定尺寸的测试图像
        test_image = create_hazy_test_image(size)

        dehazer.performance_test(test_image)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='暗通道去雾演示程序')
    parser.add_argument('--mode', choices=['demo', 'intermediate', 'comparison', 'performance'],
                       default='demo', help='运行模式')
    parser.add_argument('--image', type=str, help='输入图像路径')
    parser.add_argument('--method', choices=['dark_channel', 'refined', 'fast', 'adaptive'],
                       default='refined', help='去雾方法')
    parser.add_argument('--omega', type=float, default=0.95, help='雾霾保留系数')
    parser.add_argument('--t0', type=float, default=0.1, help='最小透射率')
    parser.add_argument('--output', type=str, help='输出路径')

    args = parser.parse_args()

    if args.mode == 'demo':
        demo_basic_dehazing()
    elif args.mode == 'intermediate':
        demo_intermediate_results()
    elif args.mode == 'comparison':
        demo_method_comparison()
    elif args.mode == 'performance':
        performance_benchmark()

    # 处理单个图像
    if args.image and Path(args.image).exists():
        print(f"\n处理图像: {args.image}")
        dehazer = DarkChannelDehazer()
        image = cv2.imread(args.image)

        if image is not None:
            method_map = {
                'dark_channel': DehazingMethod.DARK_CHANNEL,
                'refined': DehazingMethod.REFINED_DARK_CHANNEL,
                'fast': DehazingMethod.FAST_DARK_CHANNEL,
                'adaptive': DehazingMethod.ADAPTIVE_DARK_CHANNEL
            }

            method = method_map[args.method]
            params = DehazingParams(omega=args.omega, t0=args.t0)

            result = dehazer.dehaze(image, method, params)

            if args.output:
                cv2.imwrite(args.output, result)
                print(f"结果已保存至: {args.output}")
            else:
                # 显示比较结果
                dehazer.compare_methods(image, params)
        else:
            print("无法读取图像文件")


if __name__ == "__main__":
    main()