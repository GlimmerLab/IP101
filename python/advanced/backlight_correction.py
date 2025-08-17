#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
逆光图像恢复算法 - Python实现
Backlight Image Correction Algorithm

基于多尺度增强的图像细节恢复技术，
支持INRBL、自适应CLAHE、多重曝光融合等方法。

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


class BacklightCorrectionMethod(Enum):
    """逆光校正方法枚举"""
    INRBL = "inrbl"                      # 强度反比增强
    ADAPTIVE_CLAHE = "adaptive_clahe"    # 自适应CLAHE
    EXPOSURE_FUSION = "exposure_fusion"  # 多重曝光融合
    AUTO = "auto"                        # 自动选择


@dataclass
class BacklightCorrectionParams:
    """逆光校正参数配置"""
    # INRBL参数
    gamma: float = 0.6                    # 伽马校正参数 (0.1-2.0)
    lambda_factor: float = 0.8            # 增强系数 (0.0-1.0)

    # CLAHE参数
    clip_limit: float = 3.0               # 裁剪限制 (1.0-8.0)
    grid_size: Tuple[int, int] = (8, 8)   # 网格大小

    # 曝光融合参数
    exposure_levels: List[float] = None   # 曝光级别
    saturation_enhancement: float = 1.2   # 饱和度增强因子

    # 自动选择参数
    dark_threshold: float = 0.4           # 暗部比例阈值
    contrast_threshold: float = 30.0      # 对比度阈值

    def __post_init__(self):
        """参数验证和默认值设置"""
        if self.exposure_levels is None:
            self.exposure_levels = [0.5, 1.0, 2.0]

        # 参数范围验证
        if not 0.1 <= self.gamma <= 2.0:
            raise ValueError("伽马值必须在0.1-2.0范围内")
        if not 0.0 <= self.lambda_factor <= 1.0:
            raise ValueError("增强系数必须在0.0-1.0范围内")
        if not 1.0 <= self.clip_limit <= 8.0:
            raise ValueError("裁剪限制必须在1.0-8.0范围内")


class BacklightCorrector:
    """逆光图像校正处理类"""

    def __init__(self, params: Optional[BacklightCorrectionParams] = None):
        """
        初始化逆光校正器

        Args:
            params: 校正参数配置
        """
        self.params = params or BacklightCorrectionParams()
        print("逆光图像校正器初始化完成")

    def correct_backlight(self, image: np.ndarray,
                         method: BacklightCorrectionMethod = BacklightCorrectionMethod.AUTO) -> np.ndarray:
        """
        多策略逆光校正主函数

        Args:
            image: 输入图像 (BGR格式或灰度图)
            method: 校正方法

        Returns:
            校正后的图像

        Raises:
            ValueError: 当输入图像为空时
        """
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")

        print(f"开始{method.value}逆光校正")

        if method == BacklightCorrectionMethod.INRBL:
            return self._inrbl_correction(image)
        elif method == BacklightCorrectionMethod.ADAPTIVE_CLAHE:
            return self._adaptive_clahe_correction(image)
        elif method == BacklightCorrectionMethod.EXPOSURE_FUSION:
            return self._exposure_fusion_correction(image)
        elif method == BacklightCorrectionMethod.AUTO:
            return self._auto_correction(image)
        else:
            raise ValueError(f"不支持的校正方法: {method}")

    def _inrbl_correction(self, image: np.ndarray) -> np.ndarray:
        """
        INRBL强度反比增强校正

        Args:
            image: 输入图像

        Returns:
            校正后的图像
        """
        print("执行INRBL强度反比增强")

        if len(image.shape) == 2:
            # 灰度图像处理
            return self._inrbl_single_channel(image)
        else:
            # 彩色图像在YUV空间处理
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            channels = cv2.split(yuv)

            # 只对亮度通道进行INRBL处理
            channels[0] = self._inrbl_single_channel(channels[0])

            # 重构图像
            yuv_enhanced = cv2.merge(channels)
            return cv2.cvtColor(yuv_enhanced, cv2.COLOR_YUV2BGR)

    def _inrbl_single_channel(self, channel: np.ndarray) -> np.ndarray:
        """
        单通道INRBL处理

        Args:
            channel: 输入通道

        Returns:
            处理后的通道
        """
        # 转换为浮点格式
        Y = channel.astype(np.float32)

        # 计算强度反比
        inv_Y = 255.0 - Y

        # 伽马校正
        gamma_Y = 255.0 * np.power(Y / 255.0, self.params.gamma)

        # 计算自适应增强系数
        alpha = (self.params.lambda_factor * (inv_Y / 255.0) +
                (1.0 - self.params.lambda_factor) * 0.5)

        # 智能融合增强
        enhanced_Y = Y * (1.0 - alpha) + gamma_Y * alpha

        return np.clip(enhanced_Y, 0, 255).astype(np.uint8)

    def _adaptive_clahe_correction(self, image: np.ndarray) -> np.ndarray:
        """
        自适应CLAHE校正

        Args:
            image: 输入图像

        Returns:
            校正后的图像
        """
        print("执行自适应CLAHE校正")

        if len(image.shape) == 2:
            # 灰度图像直接处理
            clahe = cv2.createCLAHE(
                clipLimit=self.params.clip_limit,
                tileGridSize=self.params.grid_size
            )
            return clahe.apply(image)
        else:
            # 彩色图像在Lab空间处理
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
            l_channel, a_channel, b_channel = cv2.split(lab)

            # 仅对亮度通道应用CLAHE
            clahe = cv2.createCLAHE(
                clipLimit=self.params.clip_limit,
                tileGridSize=self.params.grid_size
            )
            l_enhanced = clahe.apply(l_channel)

            # 重构Lab图像
            lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
            bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_Lab2BGR)

            # 适度增强饱和度
            hsv = cv2.cvtColor(bgr_enhanced, cv2.COLOR_BGR2HSV)
            h_channel, s_channel, v_channel = cv2.split(hsv)

            # 增强饱和度
            s_enhanced = cv2.multiply(s_channel, self.params.saturation_enhancement)
            s_enhanced = np.clip(s_enhanced, 0, 255).astype(np.uint8)

            hsv_enhanced = cv2.merge([h_channel, s_enhanced, v_channel])
            return cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    def _exposure_fusion_correction(self, image: np.ndarray) -> np.ndarray:
        """
        多重曝光融合校正

        Args:
            image: 输入图像

        Returns:
            校正后的图像
        """
        print("执行多重曝光融合校正")

        # 生成不同曝光级别的图像
        exposures = []
        for level in self.params.exposure_levels:
            exposure = cv2.convertScaleAbs(image, alpha=level, beta=0)
            exposures.append(exposure)

        # 计算融合权重
        weights = []
        for exposure in exposures:
            weight = self._calculate_fusion_weights(exposure)
            weights.append(weight)

        # 归一化权重
        weight_sum = np.sum(weights, axis=0)
        weight_sum[weight_sum == 0] = 1e-6  # 避免除零

        for i in range(len(weights)):
            weights[i] = weights[i] / weight_sum

        # 加权融合
        result = np.zeros_like(image, dtype=np.float32)
        for exposure, weight in zip(exposures, weights):
            if len(image.shape) == 3:
                # 彩色图像，扩展权重到三个通道
                weight_3d = np.repeat(weight[:, :, np.newaxis], 3, axis=2)
                result += exposure.astype(np.float32) * weight_3d
            else:
                # 灰度图像
                result += exposure.astype(np.float32) * weight

        return np.clip(result, 0, 255).astype(np.uint8)

    def _calculate_fusion_weights(self, image: np.ndarray) -> np.ndarray:
        """
        计算曝光融合权重

        Args:
            image: 输入图像

        Returns:
            权重矩阵
        """
        # 转换为灰度图进行权重计算
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gray_float = gray.astype(np.float32) / 255.0

        # 对比度权重（基于拉普拉斯算子）
        laplacian = cv2.Laplacian(gray_float, cv2.CV_32F)
        contrast_weight = np.abs(laplacian)

        # 适当曝光权重（高斯型）
        exposure_weight = np.exp(-12.5 * np.power(gray_float - 0.5, 2))

        # 饱和度权重（仅对彩色图像）
        if len(image.shape) == 3:
            img_float = image.astype(np.float32) / 255.0
            mean_intensity = np.mean(img_float, axis=2)
            saturation_weight = np.sqrt(np.sum(np.power(img_float - mean_intensity[:, :, np.newaxis], 2), axis=2) / 3)
        else:
            saturation_weight = np.ones_like(gray_float)

        # 组合权重
        weight = contrast_weight * exposure_weight * saturation_weight
        return weight

    def _auto_correction(self, image: np.ndarray) -> np.ndarray:
        """
        自动选择最佳校正方法

        Args:
            image: 输入图像

        Returns:
            校正后的图像
        """
        print("执行自动校正方法选择")

        # 分析图像特征
        features = self._analyze_image_features(image)

        # 根据特征选择方法
        if features['dark_ratio'] > 0.5 and features['contrast'] < 25:
            # 严重逆光，使用INRBL
            print("检测到严重逆光，使用INRBL方法")
            return self._inrbl_correction(image)
        elif features['dark_ratio'] > 0.3:
            # 中等逆光，使用自适应CLAHE
            print("检测到中等逆光，使用自适应CLAHE方法")
            return self._adaptive_clahe_correction(image)
        else:
            # 轻微逆光，使用曝光融合
            print("检测到轻微逆光，使用曝光融合方法")
            return self._exposure_fusion_correction(image)

    def _analyze_image_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        分析图像特征

        Args:
            image: 输入图像

        Returns:
            特征字典
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 计算直方图
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # 归一化

        # 计算暗部比例
        dark_ratio = np.sum(hist[:64])

        # 计算对比度（标准差）
        contrast = np.std(gray)

        # 计算平均亮度
        mean_brightness = np.mean(gray)

        # 计算信息熵
        hist_nonzero = hist[hist > 0]
        entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero))

        return {
            'dark_ratio': dark_ratio,
            'contrast': contrast,
            'mean_brightness': mean_brightness,
            'entropy': entropy
        }

    def compare_methods(self, image: np.ndarray,
                       save_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        比较不同校正方法的效果

        Args:
            image: 输入图像
            save_path: 保存路径

        Returns:
            各方法的处理结果和统计信息
        """
        methods = [
            BacklightCorrectionMethod.INRBL,
            BacklightCorrectionMethod.ADAPTIVE_CLAHE,
            BacklightCorrectionMethod.EXPOSURE_FUSION
        ]

        results = {}
        for method in methods:
            try:
                start_time = time.time()
                corrected = self.correct_backlight(image, method)
                processing_time = time.time() - start_time

                # 计算质量指标
                quality_metrics = self._evaluate_correction_quality(image, corrected)

                results[method.value] = {
                    'image': corrected,
                    'time': processing_time,
                    'quality': quality_metrics
                }

                print(f"{method.value} 方法:")
                print(f"  处理时间: {processing_time:.3f}s")
                print(f"  对比度改善: {quality_metrics['contrast_improvement']:.2f}x")
                print(f"  细节增强: {quality_metrics['detail_enhancement']:.2f}x")

            except Exception as e:
                print(f"{method.value} 方法处理失败: {e}")

        # 可视化结果
        if results:
            self._visualize_comparison(image, results, save_path)

        return results

    def _evaluate_correction_quality(self, original: np.ndarray,
                                   corrected: np.ndarray) -> Dict[str, float]:
        """
        评估校正质量

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
        corr_contrast = np.std(corrected)
        contrast_improvement = corr_contrast / (orig_contrast + 1e-6)

        # 暗部细节增强
        orig_dark_detail = self._calculate_dark_detail(orig_gray)
        corr_dark_detail = self._calculate_dark_detail(corr_gray)
        detail_enhancement = corr_dark_detail / (orig_dark_detail + 1e-6)

        # 动态范围扩展
        orig_range = np.max(orig_gray) - np.min(orig_gray)
        corr_range = np.max(corr_gray) - np.min(corr_gray)
        range_expansion = corr_range / (orig_range + 1e-6)

        # 信息熵变化
        orig_entropy = self._calculate_entropy(orig_gray)
        corr_entropy = self._calculate_entropy(corr_gray)
        entropy_improvement = corr_entropy / (orig_entropy + 1e-6)

        return {
            'contrast_improvement': contrast_improvement,
            'detail_enhancement': detail_enhancement,
            'range_expansion': range_expansion,
            'entropy_improvement': entropy_improvement
        }

    def _calculate_dark_detail(self, image: np.ndarray) -> float:
        """
        计算暗部细节量

        Args:
            image: 输入图像

        Returns:
            暗部细节评分
        """
        # 提取暗部区域
        dark_mask = image < 80
        if not np.any(dark_mask):
            return 0.0

        dark_region = image[dark_mask]

        # 计算暗部区域的标准差作为细节量
        return np.std(dark_region)

    def _calculate_entropy(self, image: np.ndarray) -> float:
        """
        计算图像信息熵

        Args:
            image: 输入图像

        Returns:
            信息熵值
        """
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-6)

        # 计算熵
        hist_nonzero = hist[hist > 0]
        return -np.sum(hist_nonzero * np.log2(hist_nonzero))

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
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
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

    def batch_process(self, image_paths: List[str],
                     output_dir: str,
                     method: BacklightCorrectionMethod = BacklightCorrectionMethod.AUTO) -> None:
        """
        批量处理图像

        Args:
            image_paths: 输入图像路径列表
            output_dir: 输出目录
            method: 校正方法
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"开始批量处理 {len(image_paths)} 张图像...")

        for i, img_path in enumerate(image_paths):
            try:
                print(f"处理图像 {i+1}/{len(image_paths)}: {img_path}")

                # 读取图像
                image = cv2.imread(img_path)
                if image is None:
                    print(f"无法读取图像: {img_path}")
                    continue

                # 校正处理
                corrected = self.correct_backlight(image, method)

                # 保存结果
                filename = Path(img_path).stem
                output_file = output_path / f"{filename}_corrected.jpg"
                cv2.imwrite(str(output_file), corrected)

                print(f"已保存至: {output_file}")

            except Exception as e:
                print(f"处理图像 {img_path} 失败: {e}")

        print("批量处理完成！")


def demo_basic_correction():
    """演示基础逆光校正功能"""
    print("=== 基础逆光校正演示 ===")

    # 创建校正器
    corrector = BacklightCorrector()

    # 创建测试图像（模拟逆光场景）
    test_image = create_backlight_test_image()

    print(f"测试图像尺寸: {test_image.shape}")

    # 分析图像特征
    features = corrector._analyze_image_features(test_image)
    print("图像特征分析:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")

    # 测试不同方法
    methods = [
        BacklightCorrectionMethod.INRBL,
        BacklightCorrectionMethod.ADAPTIVE_CLAHE,
        BacklightCorrectionMethod.EXPOSURE_FUSION,
        BacklightCorrectionMethod.AUTO
    ]

    for method in methods:
        print(f"\n测试 {method.value} 方法:")
        start_time = time.time()
        result = corrector.correct_backlight(test_image, method)
        processing_time = time.time() - start_time

        print(f"  处理时间: {processing_time:.3f}s")
        print(f"  输出图像形状: {result.shape}")

        # 质量评估
        if method != BacklightCorrectionMethod.AUTO:
            quality = corrector._evaluate_correction_quality(test_image, result)
            print(f"  对比度改善: {quality['contrast_improvement']:.2f}x")
            print(f"  细节增强: {quality['detail_enhancement']:.2f}x")

    print("基础演示完成！")


def demo_method_comparison():
    """演示方法比较功能"""
    print("=== 方法比较演示 ===")

    corrector = BacklightCorrector()

    # 创建不同类型的逆光测试图像
    test_images = {
        'severe_backlight': create_severe_backlight_image(),
        'moderate_backlight': create_moderate_backlight_image(),
        'mild_backlight': create_mild_backlight_image()
    }

    for img_type, image in test_images.items():
        print(f"\n测试 {img_type} 图像:")
        results = corrector.compare_methods(image)

        # 找出最佳方法
        if results:
            best_method = max(results.keys(),
                            key=lambda k: results[k]['quality']['contrast_improvement'])
            print(f"最佳方法: {best_method}")

    print("方法比较演示完成！")


def create_backlight_test_image(size: Tuple[int, int] = (300, 400)) -> np.ndarray:
    """
    创建逆光测试图像

    Args:
        size: 图像尺寸 (高度, 宽度)

    Returns:
        测试图像
    """
    height, width = size
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # 创建渐变背景（模拟逆光效果）
    for y in range(height):
        for x in range(width):
            # 从左上到右下的亮度渐变
            brightness = int(50 + 150 * ((x + y) / (width + height)))

            # 模拟逆光：右侧更亮
            brightness += int(50 * (x / width))

            r = min(255, brightness + 20)
            g = min(255, brightness)
            b = min(255, brightness - 10)
            image[y, x] = [b, g, r]

    # 添加一些暗部物体
    cv2.rectangle(image, (50, 100), (150, 200), (30, 30, 30), -1)  # 暗色矩形
    cv2.circle(image, (300, 80), 30, (20, 25, 35), -1)             # 暗色圆

    # 添加一些亮部区域
    cv2.rectangle(image, (250, 200), (350, 280), (240, 245, 250), -1)  # 亮色矩形

    return image


def create_severe_backlight_image() -> np.ndarray:
    """创建严重逆光图像"""
    image = create_backlight_test_image()
    # 进一步增强对比度，模拟严重逆光
    image[:, 200:] = cv2.convertScaleAbs(image[:, 200:], alpha=1.5, beta=50)
    image[:, :200] = cv2.convertScaleAbs(image[:, :200], alpha=0.3, beta=-30)
    return np.clip(image, 0, 255).astype(np.uint8)


def create_moderate_backlight_image() -> np.ndarray:
    """创建中等逆光图像"""
    image = create_backlight_test_image()
    # 中等对比度调整
    image[:, 200:] = cv2.convertScaleAbs(image[:, 200:], alpha=1.2, beta=20)
    image[:, :200] = cv2.convertScaleAbs(image[:, :200], alpha=0.7, beta=-10)
    return np.clip(image, 0, 255).astype(np.uint8)


def create_mild_backlight_image() -> np.ndarray:
    """创建轻微逆光图像"""
    image = create_backlight_test_image()
    # 轻微对比度调整
    image[:, 200:] = cv2.convertScaleAbs(image[:, 200:], alpha=1.1, beta=10)
    image[:, :200] = cv2.convertScaleAbs(image[:, :200], alpha=0.9, beta=-5)
    return np.clip(image, 0, 255).astype(np.uint8)


def performance_benchmark():
    """性能基准测试"""
    print("=== 性能基准测试 ===")

    # 测试不同图像尺寸
    test_sizes = [(200, 300), (400, 600), (800, 1200)]
    methods = [
        BacklightCorrectionMethod.INRBL,
        BacklightCorrectionMethod.ADAPTIVE_CLAHE,
        BacklightCorrectionMethod.EXPOSURE_FUSION
    ]

    corrector = BacklightCorrector()

    for size in test_sizes:
        print(f"\n测试图像尺寸: {size[0]}x{size[1]}")
        test_image = np.random.randint(0, 256, (size[0], size[1], 3), dtype=np.uint8)

        for method in methods:
            times = []
            for _ in range(3):  # 运行3次取平均
                start_time = time.time()
                corrector.correct_backlight(test_image, method)
                elapsed_time = time.time() - start_time
                times.append(elapsed_time)

            avg_time = np.mean(times)
            std_time = np.std(times)

            print(f"  {method.value}: {avg_time*1000:.1f}ms (±{std_time*1000:.1f}ms)")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='逆光图像校正演示程序')
    parser.add_argument('--mode', choices=['demo', 'comparison', 'performance'],
                       default='demo', help='运行模式')
    parser.add_argument('--image', type=str, help='输入图像路径')
    parser.add_argument('--method', choices=['inrbl', 'adaptive_clahe', 'exposure_fusion', 'auto'],
                       default='auto', help='校正方法')
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
        corrector = BacklightCorrector()
        image = cv2.imread(args.image)

        if image is not None:
            method_map = {
                'inrbl': BacklightCorrectionMethod.INRBL,
                'adaptive_clahe': BacklightCorrectionMethod.ADAPTIVE_CLAHE,
                'exposure_fusion': BacklightCorrectionMethod.EXPOSURE_FUSION,
                'auto': BacklightCorrectionMethod.AUTO
            }

            method = method_map[args.method]
            result = corrector.correct_backlight(image, method)

            if args.output:
                cv2.imwrite(args.output, result)
                print(f"结果已保存至: {args.output}")
            else:
                # 显示比较结果
                corrector.compare_methods(image)
        else:
            print("无法读取图像文件")


if __name__ == "__main__":
    main()