#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自动色阶调整算法 - Python实现
Auto Level Adjustment Algorithm

基于直方图分析的图像动态范围优化技术，
支持多种色阶调整策略和自适应处理。

Author: GlimmerLab
Date: 2024
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
import time
import argparse
from pathlib import Path
from enum import Enum


class LevelAdjustmentMethod(Enum):
    """色阶调整方法枚举"""
    GLOBAL = "global"                    # 全局调整
    ADAPTIVE = "adaptive"                # 自适应调整
    LOCAL = "local"                      # 局部调整
    CONTRAST_ENHANCE = "contrast"        # 对比度增强


@dataclass
class LevelAdjustmentParams:
    """色阶调整参数配置"""
    clip_percent: float = 2.0            # 裁剪百分比 (0.0-5.0)
    separate_channels: bool = True       # 是否独立处理每个通道
    window_size: int = 51                # 局部调整窗口大小
    target_std: float = 50.0             # 对比度增强目标标准差
    enhancement_factor_range: Tuple[float, float] = (0.5, 2.0)  # 增强因子范围

    def __post_init__(self):
        """参数验证"""
        if not 0.0 <= self.clip_percent <= 5.0:
            raise ValueError("裁剪百分比必须在0.0-5.0范围内")
        if self.window_size <= 0 or self.window_size % 2 == 0:
            raise ValueError("窗口大小必须是正奇数")
        if self.target_std <= 0:
            raise ValueError("目标标准差必须大于0")


class AutoLevelAdjuster:
    """自动色阶调整处理类"""

    def __init__(self):
        """初始化色阶调整器"""
        print("自动色阶调整器初始化完成")

    def adjust_levels(self, image: np.ndarray,
                     method: LevelAdjustmentMethod = LevelAdjustmentMethod.GLOBAL,
                     params: LevelAdjustmentParams = None) -> np.ndarray:
        """
        多策略色阶调整主函数

        Args:
            image: 输入图像 (BGR格式或灰度图)
            method: 调整方法
            params: 调整参数

        Returns:
            调整后的图像

        Raises:
            ValueError: 当输入图像为空时
        """
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")

        if params is None:
            params = LevelAdjustmentParams()

        print(f"开始{method.value}色阶调整")

        if method == LevelAdjustmentMethod.GLOBAL:
            return self._global_level_adjustment(image, params)
        elif method == LevelAdjustmentMethod.ADAPTIVE:
            return self._adaptive_level_adjustment(image, params)
        elif method == LevelAdjustmentMethod.LOCAL:
            return self._local_level_adjustment(image, params)
        elif method == LevelAdjustmentMethod.CONTRAST_ENHANCE:
            return self._contrast_enhancement(image, params)
        else:
            raise ValueError(f"不支持的调整方法: {method}")

    def _compute_clipping_thresholds(self, channel: np.ndarray,
                                   clip_percent: float) -> Tuple[int, int]:
        """
        计算裁剪阈值

        Args:
            channel: 单通道图像
            clip_percent: 裁剪百分比

        Returns:
            (低阈值, 高阈值)元组
        """
        # 计算直方图
        histogram, _ = np.histogram(channel.flatten(), bins=256, range=(0, 256))

        # 计算累积分布
        cumulative_hist = np.cumsum(histogram)
        total_pixels = channel.size
        clip_pixels = int(total_pixels * clip_percent / 100.0)

        # 查找低阈值
        low_threshold = 0
        for i in range(256):
            if cumulative_hist[i] > clip_pixels:
                low_threshold = i
                break

        # 查找高阈值
        high_threshold = 255
        for i in range(255, -1, -1):
            remaining_pixels = total_pixels - cumulative_hist[i]
            if remaining_pixels > clip_pixels:
                high_threshold = i
                break

        return low_threshold, high_threshold

    def _create_lookup_table(self, low_thresh: int, high_thresh: int) -> np.ndarray:
        """
        创建查找表

        Args:
            low_thresh: 低阈值
            high_thresh: 高阈值

        Returns:
            查找表数组
        """
        lut = np.zeros(256, dtype=np.uint8)

        if high_thresh > low_thresh:
            # 线性变换
            scale = 255.0 / (high_thresh - low_thresh)

            for i in range(256):
                if i <= low_thresh:
                    lut[i] = 0
                elif i >= high_thresh:
                    lut[i] = 255
                else:
                    normalized = (i - low_thresh) * scale
                    lut[i] = np.clip(normalized, 0, 255).astype(np.uint8)
        else:
            # 特殊情况：统一亮度
            lut.fill(128)

        return lut

    def _global_level_adjustment(self, image: np.ndarray,
                               params: LevelAdjustmentParams) -> np.ndarray:
        """
        全局色阶调整

        Args:
            image: 输入图像
            params: 调整参数

        Returns:
            调整后的图像
        """
        print("执行全局色阶调整")

        if len(image.shape) == 2:
            # 灰度图像处理
            low_thresh, high_thresh = self._compute_clipping_thresholds(image, params.clip_percent)
            lut = self._create_lookup_table(low_thresh, high_thresh)
            return cv2.LUT(image, lut)

        elif len(image.shape) == 3:
            # 彩色图像处理
            if params.separate_channels:
                # 独立处理每个通道
                result_channels = []
                for c in range(image.shape[2]):
                    channel = image[:, :, c]
                    low_thresh, high_thresh = self._compute_clipping_thresholds(channel, params.clip_percent)
                    lut = self._create_lookup_table(low_thresh, high_thresh)
                    adjusted_channel = cv2.LUT(channel, lut)
                    result_channels.append(adjusted_channel)

                return np.stack(result_channels, axis=2)
            else:
                # 基于亮度的全局处理
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                low_thresh, high_thresh = self._compute_clipping_thresholds(gray, params.clip_percent)
                lut = self._create_lookup_table(low_thresh, high_thresh)

                result_channels = []
                for c in range(image.shape[2]):
                    adjusted_channel = cv2.LUT(image[:, :, c], lut)
                    result_channels.append(adjusted_channel)

                return np.stack(result_channels, axis=2)

    def _adaptive_level_adjustment(self, image: np.ndarray,
                                 params: LevelAdjustmentParams) -> np.ndarray:
        """
        自适应色阶调整

        Args:
            image: 输入图像
            params: 调整参数

        Returns:
            调整后的图像
        """
        print("执行自适应色阶调整")

        # 计算图像统计特征
        mean_val = np.mean(image)
        std_val = np.std(image)

        # 根据图像特征调整裁剪百分比
        if std_val < 20:  # 低对比度图像
            adaptive_clip = params.clip_percent * 2.0
        elif std_val > 60:  # 高对比度图像
            adaptive_clip = params.clip_percent * 0.5
        else:
            adaptive_clip = params.clip_percent

        # 调整参数范围
        adaptive_clip = np.clip(adaptive_clip, 0.1, 5.0)

        # 创建自适应参数
        adaptive_params = LevelAdjustmentParams(
            clip_percent=adaptive_clip,
            separate_channels=params.separate_channels
        )

        return self._global_level_adjustment(image, adaptive_params)

    def _local_level_adjustment(self, image: np.ndarray,
                              params: LevelAdjustmentParams) -> np.ndarray:
        """
        局部色阶调整

        Args:
            image: 输入图像
            params: 调整参数

        Returns:
            调整后的图像
        """
        print("执行局部色阶调整")

        if len(image.shape) == 2:
            # 灰度图像局部处理
            return self._local_adjust_single_channel(image, params)
        else:
            # 彩色图像局部处理
            if params.separate_channels:
                result_channels = []
                for c in range(image.shape[2]):
                    channel = image[:, :, c]
                    adjusted = self._local_adjust_single_channel(channel, params)
                    result_channels.append(adjusted)
                return np.stack(result_channels, axis=2)
            else:
                # 基于亮度的局部处理
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # 使用全局色阶调整作为简化版本
                return self._global_level_adjustment(image, params)

    def _local_adjust_single_channel(self, channel: np.ndarray,
                                   params: LevelAdjustmentParams) -> np.ndarray:
        """
        单通道局部调整

        Args:
            channel: 输入通道
            params: 调整参数

        Returns:
            调整后的通道
        """
        height, width = channel.shape
        result = np.zeros_like(channel)
        radius = params.window_size // 2

        # 为每个像素计算局部调整
        for y in range(height):
            for x in range(width):
                # 定义局部窗口
                y_min = max(0, y - radius)
                y_max = min(height, y + radius + 1)
                x_min = max(0, x - radius)
                x_max = min(width, x + radius + 1)

                # 提取局部区域
                local_region = channel[y_min:y_max, x_min:x_max]

                # 计算局部阈值
                low_thresh, high_thresh = self._compute_clipping_thresholds(
                    local_region, params.clip_percent)

                # 应用局部变换
                pixel_value = channel[y, x]
                if pixel_value <= low_thresh:
                    result[y, x] = 0
                elif pixel_value >= high_thresh:
                    result[y, x] = 255
                else:
                    normalized = (pixel_value - low_thresh) / (high_thresh - low_thresh)
                    result[y, x] = np.clip(normalized * 255, 0, 255).astype(np.uint8)

        return result

    def _contrast_enhancement(self, image: np.ndarray,
                            params: LevelAdjustmentParams) -> np.ndarray:
        """
        对比度增强

        Args:
            image: 输入图像
            params: 调整参数

        Returns:
            增强后的图像
        """
        print("执行对比度增强")

        # 先进行色阶调整
        level_adjusted = self._global_level_adjustment(image, params)

        # 计算当前标准差
        current_std = np.std(level_adjusted)

        # 计算增强因子
        if current_std > 0:
            enhancement_factor = params.target_std / current_std
        else:
            enhancement_factor = 1.0

        # 限制增强因子范围
        enhancement_factor = np.clip(enhancement_factor,
                                   params.enhancement_factor_range[0],
                                   params.enhancement_factor_range[1])

        # 应用对比度增强
        mean_val = np.mean(level_adjusted)
        enhanced = level_adjusted.astype(np.float32)
        enhanced = (enhanced - mean_val) * enhancement_factor + mean_val

        return np.clip(enhanced, 0, 255).astype(np.uint8)

    def compare_methods(self, image: np.ndarray,
                       params: LevelAdjustmentParams = None,
                       save_path: Optional[str] = None) -> None:
        """
        比较不同方法的效果

        Args:
            image: 输入图像
            params: 调整参数
            save_path: 保存路径
        """
        if params is None:
            params = LevelAdjustmentParams()

        methods = [
            LevelAdjustmentMethod.GLOBAL,
            LevelAdjustmentMethod.ADAPTIVE,
            LevelAdjustmentMethod.CONTRAST_ENHANCE
        ]

        results = {}
        for method in methods:
            try:
                start_time = time.time()
                result = self.adjust_levels(image, method, params)
                processing_time = time.time() - start_time

                results[method.value] = {
                    'image': result,
                    'time': processing_time
                }

                print(f"{method.value} 方法处理时间: {processing_time:.3f}s")
            except Exception as e:
                print(f"{method.value} 方法处理失败: {e}")

        # 可视化结果
        self._visualize_comparison(image, results, save_path)

    def _visualize_comparison(self, original: np.ndarray,
                            results: Dict[str, Dict],
                            save_path: Optional[str] = None) -> None:
        """
        可视化比较结果

        Args:
            original: 原始图像
            results: 处理结果
            save_path: 保存路径
        """
        num_images = len(results) + 1
        fig, axes = plt.subplots(2, (num_images + 1) // 2, figsize=(15, 8))

        if num_images <= 2:
            axes = axes.reshape(1, -1)

        # 显示原图
        if len(original.shape) == 3:
            axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        else:
            axes[0, 0].imshow(original, cmap='gray')
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')

        # 显示处理结果
        for i, (method, data) in enumerate(results.items(), 1):
            row = i // ((num_images + 1) // 2)
            col = i % ((num_images + 1) // 2)

            if len(data['image'].shape) == 3:
                axes[row, col].imshow(cv2.cvtColor(data['image'], cv2.COLOR_BGR2RGB))
            else:
                axes[row, col].imshow(data['image'], cmap='gray')

            axes[row, col].set_title(f"{method}\nTime: {data['time']:.3f}s")
            axes[row, col].axis('off')

        # 隐藏多余的子图
        for i in range(num_images, axes.size):
            row = i // ((num_images + 1) // 2)
            col = i % ((num_images + 1) // 2)
            if row < axes.shape[0] and col < axes.shape[1]:
                axes[row, col].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"比较结果已保存至: {save_path}")

        plt.show()

    def analyze_image_quality(self, original: np.ndarray,
                            adjusted: np.ndarray) -> Dict[str, float]:
        """
        分析图像质量

        Args:
            original: 原始图像
            adjusted: 调整后图像

        Returns:
            质量指标字典
        """
        # 计算对比度
        contrast_original = np.std(original)
        contrast_adjusted = np.std(adjusted)

        # 计算动态范围
        range_original = np.max(original) - np.min(original)
        range_adjusted = np.max(adjusted) - np.min(adjusted)

        # 计算熵
        def calculate_entropy(img):
            histogram, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
            histogram = histogram + 1e-10  # 避免log(0)
            probabilities = histogram / np.sum(histogram)
            return -np.sum(probabilities * np.log2(probabilities))

        entropy_original = calculate_entropy(original)
        entropy_adjusted = calculate_entropy(adjusted)

        return {
            'contrast_improvement': contrast_adjusted / (contrast_original + 1e-10),
            'range_improvement': range_adjusted / (range_original + 1e-10),
            'entropy_improvement': entropy_adjusted / (entropy_original + 1e-10),
            'contrast_original': contrast_original,
            'contrast_adjusted': contrast_adjusted,
            'entropy_original': entropy_original,
            'entropy_adjusted': entropy_adjusted
        }

    def performance_test(self, image: np.ndarray,
                        params: LevelAdjustmentParams = None,
                        iterations: int = 5) -> Dict[str, float]:
        """
        性能测试

        Args:
            image: 测试图像
            params: 参数配置
            iterations: 测试迭代次数

        Returns:
            性能统计结果
        """
        if params is None:
            params = LevelAdjustmentParams()

        methods = [
            LevelAdjustmentMethod.GLOBAL,
            LevelAdjustmentMethod.ADAPTIVE,
            LevelAdjustmentMethod.CONTRAST_ENHANCE
        ]

        performance_stats = {}

        for method in methods:
            times = []
            for _ in range(iterations):
                start_time = time.time()
                self.adjust_levels(image, method, params)
                elapsed_time = time.time() - start_time
                times.append(elapsed_time)

            performance_stats[method.value] = {
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'min_time': np.min(times),
                'max_time': np.max(times)
            }

        return performance_stats


def demo_basic_level_adjustment():
    """演示基础色阶调整功能"""
    print("=== 基础色阶调整演示 ===")

    # 创建调整器
    adjuster = AutoLevelAdjuster()

    # 创建测试图像
    test_image = create_test_image()

    # 调整参数
    params = LevelAdjustmentParams(clip_percent=2.0, separate_channels=True)

    # 测试不同方法
    methods = [
        LevelAdjustmentMethod.GLOBAL,
        LevelAdjustmentMethod.ADAPTIVE,
        LevelAdjustmentMethod.CONTRAST_ENHANCE
    ]

    for method in methods:
        print(f"\n测试 {method.value} 方法:")
        start_time = time.time()
        result = adjuster.adjust_levels(test_image, method, params)
        processing_time = time.time() - start_time

        print(f"  处理时间: {processing_time:.3f}s")
        print(f"  输出图像形状: {result.shape}")

        # 质量分析
        quality_metrics = adjuster.analyze_image_quality(test_image, result)
        print(f"  对比度提升: {quality_metrics['contrast_improvement']:.2f}x")
        print(f"  动态范围提升: {quality_metrics['range_improvement']:.2f}x")

    print("基础演示完成！")


def demo_performance_comparison():
    """演示性能比较"""
    print("=== 性能比较演示 ===")

    adjuster = AutoLevelAdjuster()

    # 测试不同图像尺寸
    test_sizes = [(200, 300), (400, 600), (800, 1200)]

    for size in test_sizes:
        print(f"\n测试图像尺寸: {size[0]}x{size[1]}")
        test_image = np.random.randint(0, 256, (size[0], size[1], 3), dtype=np.uint8)

        performance_stats = adjuster.performance_test(test_image, iterations=3)

        for method, stats in performance_stats.items():
            print(f"  {method}: 平均时间 {stats['mean_time']*1000:.1f}ms "
                  f"(±{stats['std_time']*1000:.1f}ms)")

    print("性能比较完成！")


def create_test_image() -> np.ndarray:
    """
    创建测试图像

    Returns:
        测试图像
    """
    height, width = 300, 400
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # 创建渐变背景
    for y in range(height):
        for x in range(width):
            r = int(100 + 50 * (x / width))
            g = int(120 + 60 * (y / height))
            b = int(80 + 40 * ((x + y) / (width + height)))
            image[y, x] = [b, g, r]

    # 添加几何形状
    cv2.rectangle(image, (50, 50), (150, 150), (200, 200, 200), -1)
    cv2.circle(image, (250, 100), 40, (100, 100, 100), -1)
    cv2.rectangle(image, (100, 200), (300, 250), (50, 50, 50), -1)

    return image


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='自动色阶调整演示程序')
    parser.add_argument('--mode', choices=['demo', 'performance'], default='demo',
                       help='运行模式')
    parser.add_argument('--image', type=str, help='输入图像路径')

    args = parser.parse_args()

    if args.mode == 'demo':
        demo_basic_level_adjustment()
    elif args.mode == 'performance':
        demo_performance_comparison()

    if args.image and Path(args.image).exists():
        print(f"\n处理图像: {args.image}")
        adjuster = AutoLevelAdjuster()
        image = cv2.imread(args.image)

        if image is not None:
            params = LevelAdjustmentParams()
            adjuster.compare_methods(image, params)
        else:
            print("无法读取图像文件")


if __name__ == "__main__":
    main()