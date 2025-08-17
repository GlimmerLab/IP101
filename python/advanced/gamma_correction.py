#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
二维伽马校正算法 - Python实现
2D Gamma Correction Algorithm

基于自适应亮度映射的数学建模技术，
支持全局、自适应、局部和直方图等多种校正方法。

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


class GammaCorrectionMethod(Enum):
    """伽马校正方法枚举"""
    GLOBAL = "global"                    # 全局校正
    ADAPTIVE = "adaptive"                # 自适应校正
    LOCAL = "local"                      # 局部校正
    HISTOGRAM_BASED = "histogram"        # 基于直方图的校正


@dataclass
class GammaCorrectionParams:
    """伽马校正参数配置"""
    base_gamma: float = 1.2              # 基础伽马值 (0.1-3.0)
    adaptation_strength: float = 0.5     # 自适应强度 (0.0-1.0)
    window_size: int = 51                # 局部分析窗口大小
    block_size: int = 64                 # 分块处理块大小
    min_gamma: float = 0.3               # 最小伽马值
    max_gamma: float = 3.0               # 最大伽马值
    parallel_workers: int = 4            # 并行工作线程数

    def __post_init__(self):
        """参数验证"""
        if not 0.1 <= self.base_gamma <= 3.0:
            raise ValueError("基础伽马值必须在0.1-3.0范围内")
        if not 0.0 <= self.adaptation_strength <= 1.0:
            raise ValueError("自适应强度必须在0.0-1.0范围内")
        if self.window_size <= 0 or self.window_size % 2 == 0:
            raise ValueError("窗口大小必须是正奇数")
        if self.min_gamma >= self.max_gamma:
            raise ValueError("最小伽马值必须小于最大伽马值")


class GammaCorrector:
    """二维伽马校正处理类"""

    def __init__(self):
        """初始化伽马校正器"""
        print("二维伽马校正器初始化完成")

    def correct_gamma(self, image: np.ndarray,
                     method: GammaCorrectionMethod = GammaCorrectionMethod.ADAPTIVE,
                     params: Optional[GammaCorrectionParams] = None) -> np.ndarray:
        """
        多策略伽马校正主函数

        Args:
            image: 输入图像 (BGR格式或灰度图)
            method: 校正方法
            params: 校正参数

        Returns:
            校正后的图像

        Raises:
            ValueError: 当输入图像为空时
        """
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")

        if params is None:
            params = GammaCorrectionParams()

        print(f"开始{method.value}伽马校正")

        if method == GammaCorrectionMethod.GLOBAL:
            return self._global_gamma_correction(image, params)
        elif method == GammaCorrectionMethod.ADAPTIVE:
            return self._adaptive_gamma_correction(image, params)
        elif method == GammaCorrectionMethod.LOCAL:
            return self._local_2d_gamma_correction(image, params)
        elif method == GammaCorrectionMethod.HISTOGRAM_BASED:
            return self._histogram_based_gamma_correction(image, params)
        else:
            raise ValueError(f"不支持的校正方法: {method}")

    def _create_gamma_lut(self, gamma: float) -> np.ndarray:
        """
        创建伽马查找表

        Args:
            gamma: 伽马值

        Returns:
            查找表数组
        """
        lut = np.zeros(256, dtype=np.uint8)

        for i in range(256):
            # 伽马变换公式
            normalized = i / 255.0
            corrected = np.power(normalized, 1.0 / gamma)
            lut[i] = np.clip(corrected * 255.0, 0, 255).astype(np.uint8)

        return lut

    def _compute_local_gamma(self, image: np.ndarray, x: int, y: int,
                           params: GammaCorrectionParams) -> float:
        """
        计算局部自适应伽马值

        Args:
            image: 输入图像
            x, y: 像素坐标
            params: 校正参数

        Returns:
            局部最优伽马值
        """
        # 定义局部分析窗口
        radius = params.window_size // 2
        h, w = image.shape[:2]

        x_min = max(0, x - radius)
        y_min = max(0, y - radius)
        x_max = min(w, x + radius + 1)
        y_max = min(h, y + radius + 1)

        # 提取局部区域
        if len(image.shape) == 3:
            local_region = cv2.cvtColor(image[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2GRAY)
        else:
            local_region = image[y_min:y_max, x_min:x_max]

        # 计算局部统计特征
        local_brightness = np.mean(local_region) / 255.0
        local_contrast = np.std(local_region) / 255.0

        # 自适应因子计算
        adaptive_factor = 1.0

        if local_brightness < 0.3:
            # 暗部区域提亮
            adaptive_factor = 1.0 + params.adaptation_strength * (0.3 - local_brightness)
        elif local_brightness > 0.7:
            # 亮部区域压制
            adaptive_factor = 1.0 - params.adaptation_strength * (local_brightness - 0.7)

        # 对比度影响
        contrast_factor = 1.0 + (0.1 - local_contrast) * 0.5

        local_gamma = params.base_gamma * adaptive_factor * contrast_factor

        # 限制伽马值范围
        return np.clip(local_gamma, params.min_gamma, params.max_gamma)

    def _global_gamma_correction(self, image: np.ndarray,
                               params: GammaCorrectionParams) -> np.ndarray:
        """
        全局伽马校正

        Args:
            image: 输入图像
            params: 校正参数

        Returns:
            校正后的图像
        """
        print("执行全局伽马校正")

        # 构建伽马查找表
        lut = self._create_gamma_lut(params.base_gamma)

        # 应用查找表变换
        return cv2.LUT(image, lut)

    def _adaptive_gamma_correction(self, image: np.ndarray,
                                 params: GammaCorrectionParams) -> np.ndarray:
        """
        自适应伽马校正

        Args:
            image: 输入图像
            params: 校正参数

        Returns:
            校正后的图像
        """
        print("执行自适应伽马校正")

        # 转换为灰度图分析
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 计算图像统计特征
        avg_brightness = np.mean(gray) / 255.0
        contrast = np.std(gray) / 255.0

        # 自适应伽马计算
        adaptive_gamma = params.base_gamma

        if avg_brightness < 0.3:
            # 偏暗图像处理
            adaptive_gamma = params.base_gamma * (1.0 + params.adaptation_strength * (0.3 - avg_brightness))
        elif avg_brightness > 0.7:
            # 偏亮图像处理
            adaptive_gamma = params.base_gamma * (1.0 - params.adaptation_strength * (avg_brightness - 0.7))

        # 考虑对比度影响
        if contrast < 0.1:
            adaptive_gamma *= (1.0 + (0.1 - contrast) * 2.0)

        # 限制伽马值范围
        adaptive_gamma = np.clip(adaptive_gamma, params.min_gamma, params.max_gamma)

        print(f"计算得到自适应伽马值: {adaptive_gamma:.3f}")

        # 应用自适应伽马校正
        lut = self._create_gamma_lut(adaptive_gamma)
        return cv2.LUT(image, lut)

    def _local_2d_gamma_correction(self, image: np.ndarray,
                                 params: GammaCorrectionParams) -> np.ndarray:
        """
        二维局部伽马校正

        Args:
            image: 输入图像
            params: 校正参数

        Returns:
            校正后的图像
        """
        print("执行二维局部伽马校正")

        # 转换为浮点型处理
        image_float = image.astype(np.float32) / 255.0
        result = np.zeros_like(image_float)

        h, w = image.shape[:2]

        # 为每个像素计算局部伽马值
        for y in range(h):
            for x in range(w):
                # 计算局部伽马值
                local_gamma = self._compute_local_gamma(image, x, y, params)

                if len(image.shape) == 2:
                    # 灰度图像处理
                    pixel_value = image_float[y, x]
                    corrected_value = np.power(pixel_value, 1.0 / local_gamma)
                    result[y, x] = corrected_value
                else:
                    # 彩色图像处理
                    for c in range(3):
                        pixel_value = image_float[y, x, c]
                        corrected_value = np.power(pixel_value, 1.0 / local_gamma)
                        result[y, x, c] = corrected_value

        # 转换回8位图像
        return np.clip(result * 255.0, 0, 255).astype(np.uint8)

    def _histogram_based_gamma_correction(self, image: np.ndarray,
                                        params: GammaCorrectionParams) -> np.ndarray:
        """
        基于直方图的伽马校正

        Args:
            image: 输入图像
            params: 校正参数

        Returns:
            校正后的图像
        """
        print("执行基于直方图的伽马校正")

        # 转换为灰度图分析
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 计算直方图
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()

        # 计算累积分布函数
        cdf = np.cumsum(hist)
        cdf = cdf / cdf[-1]  # 归一化

        # 构建自适应伽马查找表
        lut = np.zeros(256, dtype=np.uint8)

        for i in range(256):
            # 基于CDF的动态伽马调整
            normalized = i / 255.0
            cdf_value = cdf[i]

            # 根据累积分布调整伽马值
            dynamic_gamma = params.base_gamma

            if cdf_value < 0.3:
                # 暗部区域密集 - 提亮处理
                dynamic_gamma *= (1.0 + params.adaptation_strength)
            elif cdf_value > 0.7:
                # 亮部区域密集 - 压制处理
                dynamic_gamma *= (1.0 - params.adaptation_strength * 0.5)

            corrected = np.power(normalized, 1.0 / dynamic_gamma)
            lut[i] = np.clip(corrected * 255.0, 0, 255).astype(np.uint8)

        # 应用动态查找表
        return cv2.LUT(image, lut)

    def optimized_block_gamma_correction(self, image: np.ndarray,
                                       params: Optional[GammaCorrectionParams] = None) -> np.ndarray:
        """
        优化版分块伽马校正

        Args:
            image: 输入图像
            params: 校正参数

        Returns:
            校正后的图像
        """
        if params is None:
            params = GammaCorrectionParams()

        print("执行优化版分块伽马校正")

        h, w = image.shape[:2]
        result = np.zeros_like(image, dtype=np.float32)

        # 分块处理
        num_blocks_x = (w + params.block_size - 1) // params.block_size
        num_blocks_y = (h + params.block_size - 1) // params.block_size

        # 计算每个块的伽马值
        def compute_block_gamma(block_info):
            bx, by = block_info
            x_start = bx * params.block_size
            y_start = by * params.block_size
            x_end = min(x_start + params.block_size, w)
            y_end = min(y_start + params.block_size, h)

            # 提取块区域
            if len(image.shape) == 3:
                block = cv2.cvtColor(image[y_start:y_end, x_start:x_end], cv2.COLOR_BGR2GRAY)
            else:
                block = image[y_start:y_end, x_start:x_end]

            # 计算块统计特征
            block_brightness = np.mean(block) / 255.0
            block_gamma = params.base_gamma

            if block_brightness < 0.3:
                block_gamma *= (1.0 + params.adaptation_strength * (0.3 - block_brightness))
            elif block_brightness > 0.7:
                block_gamma *= (1.0 - params.adaptation_strength * (block_brightness - 0.7))

            return (bx, by, np.clip(block_gamma, params.min_gamma, params.max_gamma))

        # 并行计算块伽马值
        block_coords = [(bx, by) for by in range(num_blocks_y) for bx in range(num_blocks_x)]

        with ThreadPoolExecutor(max_workers=params.parallel_workers) as executor:
            block_results = list(executor.map(compute_block_gamma, block_coords))

        # 构建伽马值矩阵
        gamma_matrix = np.zeros((num_blocks_y, num_blocks_x))
        for bx, by, gamma_val in block_results:
            gamma_matrix[by, bx] = gamma_val

        # 应用伽马校正
        image_float = image.astype(np.float32) / 255.0

        for y in range(h):
            for x in range(w):
                # 确定当前像素所属的块
                bx = min(x // params.block_size, num_blocks_x - 1)
                by = min(y // params.block_size, num_blocks_y - 1)

                # 应用对应的伽马值
                gamma = gamma_matrix[by, bx]

                if len(image.shape) == 2:
                    pixel_value = image_float[y, x]
                    corrected = np.power(pixel_value, 1.0 / gamma)
                    result[y, x] = corrected
                else:
                    for c in range(3):
                        pixel_value = image_float[y, x, c]
                        corrected = np.power(pixel_value, 1.0 / gamma)
                        result[y, x, c] = corrected

        return np.clip(result * 255.0, 0, 255).astype(np.uint8)

    def compare_methods(self, image: np.ndarray,
                       params: Optional[GammaCorrectionParams] = None,
                       save_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        比较不同校正方法的效果

        Args:
            image: 输入图像
            params: 校正参数
            save_path: 保存路径

        Returns:
            各方法的处理结果和统计信息
        """
        if params is None:
            params = GammaCorrectionParams()

        methods = [
            GammaCorrectionMethod.GLOBAL,
            GammaCorrectionMethod.ADAPTIVE,
            GammaCorrectionMethod.HISTOGRAM_BASED
        ]

        results = {}

        print("开始方法比较测试")

        for method in methods:
            try:
                start_time = time.time()
                corrected = self.correct_gamma(image, method, params)
                processing_time = time.time() - start_time

                # 计算质量指标
                quality_metrics = self._analyze_image_quality(image, corrected)

                results[method.value] = {
                    'image': corrected,
                    'time': processing_time,
                    'quality': quality_metrics
                }

                print(f"{method.value} 方法:")
                print(f"  处理时间: {processing_time:.3f}s")
                print(f"  对比度改善: {quality_metrics['contrast_improvement']:.2f}x")
                print(f"  信息熵改善: {quality_metrics['entropy_improvement']:.2f}x")

            except Exception as e:
                print(f"{method.value} 方法处理失败: {e}")

        # 可视化结果
        if results:
            self._visualize_comparison(image, results, save_path)

        return results

    def analyze_gamma_response(self, image: np.ndarray,
                             gamma_range: Tuple[float, float] = (0.3, 3.0),
                             num_samples: int = 10) -> None:
        """
        分析不同伽马值的响应效果

        Args:
            image: 输入图像
            gamma_range: 伽马值范围
            num_samples: 采样数量
        """
        print("开始伽马响应分析")

        gamma_values = np.linspace(gamma_range[0], gamma_range[1], num_samples)

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()

        for i, gamma in enumerate(gamma_values[:10]):
            # 创建临时参数
            temp_params = GammaCorrectionParams(base_gamma=gamma)
            corrected = self._global_gamma_correction(image, temp_params)

            # 显示结果
            if len(corrected.shape) == 3:
                axes[i].imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
            else:
                axes[i].imshow(corrected, cmap='gray')

            axes[i].set_title(f'γ = {gamma:.1f}')
            axes[i].axis('off')

        plt.tight_layout()
        plt.suptitle('伽马值响应分析', y=1.02)
        plt.show()

        # 分析统计信息
        print("\n伽马值响应统计:")
        for gamma in gamma_values:
            temp_params = GammaCorrectionParams(base_gamma=gamma)
            corrected = self._global_gamma_correction(image, temp_params)

            # 计算图像统计
            if len(corrected.shape) == 3:
                gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
            else:
                gray = corrected

            mean_brightness = np.mean(gray) / 255.0
            contrast = np.std(gray) / 255.0

            print(f"  γ={gamma:.1f}: 平均亮度={mean_brightness:.3f}, 对比度={contrast:.3f}")

    def _analyze_image_quality(self, original: np.ndarray,
                            corrected: np.ndarray) -> Dict[str, float]:
        """
        分析图像质量指标

        Args:
            original: 原始图像
            corrected: 校正后图像

        Returns:
            质量指标字典
        """
        # 转换为灰度图进行分析
        if len(original.shape) == 3:
            orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            corr_gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
        else:
            orig_gray = original.copy()
            corr_gray = corrected.copy()

        # 对比度改善
        orig_contrast = np.std(orig_gray)
        corr_contrast = np.std(corr_gray)
        contrast_improvement = corr_contrast / (orig_contrast + 1e-6)

        # 计算信息熵
        def calculate_entropy(img):
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-6)
            hist_nonzero = hist[hist > 0]
            return -np.sum(hist_nonzero * np.log2(hist_nonzero))

        orig_entropy = calculate_entropy(orig_gray)
        corr_entropy = calculate_entropy(corr_gray)
        entropy_improvement = corr_entropy / (orig_entropy + 1e-6)

        # 估计伽马参数
        def estimate_gamma(img1, img2):
            # 简单的伽马估计方法
            mean1 = np.mean(img1) / 255.0
            mean2 = np.mean(img2) / 255.0

            if mean1 > 0 and mean2 > 0:
                return np.log(mean2) / np.log(mean1)
            return 1.0

        estimated_gamma = estimate_gamma(orig_gray, corr_gray)

        # 动态范围扩展
        orig_range = np.max(orig_gray) - np.min(orig_gray)
        corr_range = np.max(corr_gray) - np.min(corr_gray)
        range_expansion = corr_range / (orig_range + 1e-6)

        return {
            'contrast_improvement': contrast_improvement,
            'entropy_improvement': entropy_improvement,
            'estimated_gamma': estimated_gamma,
            'range_expansion': range_expansion
        }

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
        cols = min(4, num_images)
        rows = (num_images + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

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
                    title += f"\nContrast: {data['quality']['contrast_improvement']:.2f}x"

                axes[i].set_title(title)
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
                        params: Optional[GammaCorrectionParams] = None,
                        iterations: int = 5) -> Dict[str, float]:
        """
        性能基准测试

        Args:
            image: 输入图像
            params: 校正参数
            iterations: 测试迭代次数

        Returns:
            性能统计结果
        """
        if params is None:
            params = GammaCorrectionParams()

        print(f"开始性能测试 - 图像尺寸: {image.shape}")

        methods = [
            GammaCorrectionMethod.GLOBAL,
            GammaCorrectionMethod.ADAPTIVE,
            GammaCorrectionMethod.HISTOGRAM_BASED
        ]

        performance_results = {}

        for method in methods:
            times = []

            for _ in range(iterations):
                start_time = time.time()
                try:
                    self.correct_gamma(image, method, params)
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


def demo_basic_correction():
    """演示基础伽马校正功能"""
    print("=== 基础伽马校正演示 ===")

    # 创建校正器
    corrector = GammaCorrector()

    # 创建测试图像
    test_image = create_test_image()

    print(f"测试图像尺寸: {test_image.shape}")

    # 测试不同方法
    methods = [
        GammaCorrectionMethod.GLOBAL,
        GammaCorrectionMethod.ADAPTIVE,
        GammaCorrectionMethod.HISTOGRAM_BASED
    ]

    for method in methods:
        print(f"\n测试 {method.value} 方法:")
        start_time = time.time()
        result = corrector.correct_gamma(test_image, method)
        processing_time = time.time() - start_time

        print(f"  处理时间: {processing_time:.3f}s")
        print(f"  输出图像形状: {result.shape}")

    print("基础演示完成")


def demo_method_comparison():
    """演示方法比较功能"""
    print("=== 方法比较演示 ===")

    corrector = GammaCorrector()

    # 创建不同类型的测试图像
    test_images = {
        'dark_image': create_dark_image(),
        'bright_image': create_bright_image(),
        'normal_image': create_test_image()
    }

    for img_type, image in test_images.items():
        print(f"\n测试 {img_type} 图像:")
        results = corrector.compare_methods(image)

        # 找出最佳方法
        if results:
            best_method = max(results.keys(),
                            key=lambda k: results[k]['quality']['contrast_improvement'])
            print(f"最佳方法: {best_method}")

    print("方法比较演示完成")


def create_test_image(size: Tuple[int, int] = (300, 400)) -> np.ndarray:
    """
    创建测试图像

    Args:
        size: 图像尺寸 (高度, 宽度)

    Returns:
        测试图像
    """
    height, width = size
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # 创建渐变背景
    for y in range(height):
        for x in range(width):
            # 从左到右的亮度渐变
            brightness = int(50 + 150 * (x / width))

            r = min(255, brightness + 20)
            g = min(255, brightness)
            b = min(255, brightness - 10)
            image[y, x] = [b, g, r]

    # 添加一些几何形状
    cv2.rectangle(image, (50, 100), (150, 200), (80, 90, 100), -1)
    cv2.circle(image, (300, 80), 30, (60, 70, 80), -1)
    cv2.rectangle(image, (250, 200), (350, 280), (180, 190, 200), -1)

    return image


def create_dark_image() -> np.ndarray:
    """创建偏暗图像"""
    image = create_test_image()
    return cv2.convertScaleAbs(image, alpha=0.4, beta=-20)


def create_bright_image() -> np.ndarray:
    """创建偏亮图像"""
    image = create_test_image()
    return cv2.convertScaleAbs(image, alpha=1.3, beta=30)


def performance_benchmark():
    """性能基准测试"""
    print("=== 性能基准测试 ===")

    # 测试不同图像尺寸
    test_sizes = [(200, 300), (400, 600), (800, 1200)]

    corrector = GammaCorrector()

    for size in test_sizes:
        print(f"\n测试图像尺寸: {size[0]}x{size[1]}")
        test_image = np.random.randint(0, 256, (size[0], size[1], 3), dtype=np.uint8)

        corrector.performance_test(test_image)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='二维伽马校正演示程序')
    parser.add_argument('--mode', choices=['demo', 'comparison', 'performance'],
                       default='demo', help='运行模式')
    parser.add_argument('--image', type=str, help='输入图像路径')
    parser.add_argument('--method', choices=['global', 'adaptive', 'local', 'histogram'],
                       default='adaptive', help='校正方法')
    parser.add_argument('--gamma', type=float, default=1.2, help='基础伽马值')
    parser.add_argument('--output', type=str, help='输出路径')

    args = parser.parse_args()

    if args.mode == 'demo':
        demo_basic_correction()
    elif args.mode == 'comparison':
        demo_method_comparison()
    elif args.mode == 'performance':
        performance_benchmark()

    # 处理单个图像
    if args.image and Path(args.image).exists():
        print(f"\n处理图像: {args.image}")
        corrector = GammaCorrector()
        image = cv2.imread(args.image)

        if image is not None:
            method_map = {
                'global': GammaCorrectionMethod.GLOBAL,
                'adaptive': GammaCorrectionMethod.ADAPTIVE,
                'local': GammaCorrectionMethod.LOCAL,
                'histogram': GammaCorrectionMethod.HISTOGRAM_BASED
            }

            method = method_map[args.method]
            params = GammaCorrectionParams(base_gamma=args.gamma)
            result = corrector.correct_gamma(image, method, params)

            if args.output:
                cv2.imwrite(args.output, result)
                print(f"结果已保存至: {args.output}")
            else:
                # 显示比较结果
                corrector.compare_methods(image, params)
        else:
            print("无法读取图像文件")


if __name__ == "__main__":
    main()