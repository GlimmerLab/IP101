#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多尺度细节增强算法实现
Multi-Scale Detail Enhancement Algorithm Implementation

基于拉普拉斯金字塔的图像细节增强算法，能够在不同尺度上
增强图像细节，同时保持图像的自然性。

Author: GlimmerLab
Date: 2024
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Union
from dataclasses import dataclass
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MultiScaleParams:
    """多尺度细节增强参数配置"""
    sigma1: float = 1.0      # 金字塔构建标准差
    sigma2: float = 2.0      # 细节提取标准差
    alpha: float = 1.5       # 细节增强强度
    beta: float = 1.2        # 对比度增强因子
    max_levels: int = 6      # 最大金字塔层数

    def __post_init__(self):
        """参数验证"""
        if self.sigma1 <= 0 or self.sigma2 <= 0:
            raise ValueError("标准差参数必须大于0")
        if self.alpha < 0 or self.beta <= 0:
            raise ValueError("增强参数必须为正数")
        if self.max_levels < 1:
            raise ValueError("金字塔层数必须至少为1")


class MultiScaleEnhancer:
    """多尺度细节增强处理器"""

    def __init__(self, params: Optional[MultiScaleParams] = None):
        """
        初始化多尺度增强器

        Args:
            params: 增强参数配置
        """
        self.params = params or MultiScaleParams()
        logger.info(f"初始化多尺度增强器，参数: {self.params}")

    def enhance_image(self, image: np.ndarray,
                     params: Optional[MultiScaleParams] = None) -> np.ndarray:
        """
        执行多尺度细节增强

        Args:
            image: 输入图像 (H, W, C) 或 (H, W)
            params: 增强参数（可选）

        Returns:
            增强后的图像

        Raises:
            ValueError: 输入图像无效时
        """
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")

        if len(image.shape) not in [2, 3]:
            raise ValueError("输入图像必须是2D或3D数组")

        p = params or self.params
        original_dtype = image.dtype

        # 转换为浮点型进行处理
        float_img = image.astype(np.float32)

        # 对于彩色图像，分别处理每个通道
        if len(image.shape) == 3:
            channels = cv2.split(float_img)
            enhanced_channels = []

            for channel in channels:
                enhanced_channel = self._enhance_single_channel(channel, p)
                enhanced_channels.append(enhanced_channel)

            result = cv2.merge(enhanced_channels)
        else:
            result = self._enhance_single_channel(float_img, p)

        # 转换回原始数据类型
        if original_dtype == np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        else:
            result = result.astype(original_dtype)

        return result

    def _enhance_single_channel(self, channel: np.ndarray,
                               params: MultiScaleParams) -> np.ndarray:
        """
        对单个通道执行多尺度增强

        Args:
            channel: 单通道图像
            params: 增强参数

        Returns:
            增强后的单通道图像
        """
        # 构建高斯金字塔
        pyramid = self._build_pyramid(channel, params)

        # 从顶层开始重建
        result = pyramid[-1].copy()

        for i in range(len(pyramid) - 2, -1, -1):
            # 上采样到当前尺度
            upsampled = cv2.resize(
                result,
                (pyramid[i].shape[1], pyramid[i].shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

            # 提取细节层
            detail = self._extract_detail(pyramid[i], params.sigma2)

            # 增强细节并融合
            result = upsampled + params.alpha * detail

        # 对比度调整
        enhanced = result * params.beta

        return enhanced

    def _build_pyramid(self, image: np.ndarray,
                      params: MultiScaleParams) -> List[np.ndarray]:
        """
        构建高斯金字塔

        Args:
            image: 输入图像
            params: 金字塔参数

        Returns:
            金字塔层级列表
        """
        pyramid = []
        current = image.copy()

        for level in range(params.max_levels):
            pyramid.append(current.copy())

            # 检查尺寸是否过小
            if current.shape[0] <= 32 or current.shape[1] <= 32:
                break

            # 模糊并下采样
            blurred = cv2.GaussianBlur(current, (0, 0), params.sigma1)
            current = cv2.resize(
                blurred, None, fx=0.5, fy=0.5,
                interpolation=cv2.INTER_LINEAR
            )

        logger.debug(f"构建了{len(pyramid)}层金字塔")
        return pyramid

    def _extract_detail(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """
        提取细节层

        Args:
            image: 输入图像
            sigma: 高斯模糊标准差

        Returns:
            细节层图像
        """
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        return image - blurred

    def adaptive_enhance(self, image: np.ndarray) -> np.ndarray:
        """
        自适应多尺度增强

        根据图像特征自动调整增强参数

        Args:
            image: 输入图像

        Returns:
            自适应增强后的图像
        """
        # 分析图像特征
        quality_metrics = self.analyze_image_quality(image)
        detail_measure = quality_metrics['detail_richness']

        # 自适应参数调节
        params = MultiScaleParams()

        if detail_measure < 50.0:
            # 低细节图像 - 需要更强的增强
            params.alpha = 2.0
            params.beta = 1.3
            logger.info("检测到低细节图像，使用强增强参数")
        elif detail_measure > 150.0:
            # 高细节图像 - 保守增强避免过度处理
            params.alpha = 1.2
            params.sigma2 = 2.5
            logger.info("检测到高细节图像，使用保守增强参数")
        else:
            # 中等细节图像 - 使用标准参数
            logger.info("使用标准增强参数")

        return self.enhance_image(image, params)

    def analyze_image_quality(self, image: np.ndarray) -> dict:
        """
        分析图像质量特征

        Args:
            image: 输入图像

        Returns:
            图像质量分析结果字典
        """
        # 转换为灰度图像进行分析
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gray = gray.astype(np.float32)

        # 计算细节丰富度（拉普拉斯算子方差）
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        detail_richness = np.var(laplacian)

        # 计算边缘强度
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        edge_strength = np.mean(gradient_magnitude)

        # 计算对比度和亮度
        contrast_level = np.std(gray)
        brightness_level = np.mean(gray)

        # 计算噪声估计
        noise_estimate = self._estimate_noise(gray)

        metrics = {
            'detail_richness': detail_richness,
            'edge_strength': edge_strength,
            'contrast_level': contrast_level,
            'brightness_level': brightness_level,
            'noise_estimate': noise_estimate
        }

        logger.debug(f"图像质量分析: {metrics}")
        return metrics

    def _estimate_noise(self, image: np.ndarray) -> float:
        """
        估计图像噪声水平

        Args:
            image: 灰度图像

        Returns:
            噪声估计值
        """
        # 使用高通滤波器估计噪声
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]]) / 8.0

        filtered = cv2.filter2D(image, -1, kernel)
        noise_estimate = np.std(filtered)

        return noise_estimate

    def batch_enhance(self, images: List[np.ndarray],
                     use_adaptive: bool = False) -> List[np.ndarray]:
        """
        批量处理图像

        Args:
            images: 图像列表
            use_adaptive: 是否使用自适应增强

        Returns:
            增强后的图像列表
        """
        enhanced_images = []

        for i, image in enumerate(images):
            logger.info(f"处理第 {i+1}/{len(images)} 张图像")

            if use_adaptive:
                enhanced = self.adaptive_enhance(image)
            else:
                enhanced = self.enhance_image(image)

            enhanced_images.append(enhanced)

        return enhanced_images

    def save_comparison(self, original: np.ndarray, enhanced: np.ndarray,
                       save_path: str) -> None:
        """
        保存对比图像

        Args:
            original: 原始图像
            enhanced: 增强后的图像
            save_path: 保存路径
        """
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        if len(original.shape) == 3:
            plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(original, cmap='gray')
        plt.title('原始图像')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        if len(enhanced.shape) == 3:
            plt.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(enhanced, cmap='gray')
        plt.title('增强后图像')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"对比图像已保存到: {save_path}")


def demo_multi_scale_enhancement():
    """演示多尺度细节增强功能"""

    # 创建测试图像
    def create_test_image() -> np.ndarray:
        """创建用于测试的合成图像"""
        image = np.zeros((256, 256), dtype=np.uint8)

        # 添加大尺度结构
        cv2.rectangle(image, (50, 50), (200, 200), 100, -1)

        # 添加中等尺度纹理
        for i in range(0, 256, 10):
            cv2.line(image, (i, 0), (i, 255), 150, 1)

        # 添加细节
        noise = np.random.normal(0, 10, (256, 256))
        image = image.astype(np.float32) + noise
        image = np.clip(image, 0, 255).astype(np.uint8)

        return image

    print("=== 多尺度细节增强演示 ===")

    # 创建增强器
    enhancer = MultiScaleEnhancer()

    # 创建测试图像
    test_image = create_test_image()
    print(f"创建测试图像，尺寸: {test_image.shape}")

    # 标准增强
    print("\n执行标准多尺度增强...")
    enhanced_std = enhancer.enhance_image(test_image)

    # 自适应增强
    print("执行自适应多尺度增强...")
    enhanced_adaptive = enhancer.adaptive_enhance(test_image)

    # 分析图像质量
    print("\n图像质量分析:")
    metrics = enhancer.analyze_image_quality(test_image)
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")

    # 测试不同参数
    print("\n测试不同增强参数:")
    params_list = [
        MultiScaleParams(alpha=1.0, beta=1.0),  # 轻度增强
        MultiScaleParams(alpha=2.0, beta=1.3),  # 强增强
        MultiScaleParams(alpha=1.5, beta=1.2, sigma2=3.0)  # 保守增强
    ]

    for i, params in enumerate(params_list):
        enhanced = enhancer.enhance_image(test_image, params)
        print(f"  参数组合 {i+1}: alpha={params.alpha}, beta={params.beta}, sigma2={params.sigma2}")

    print("\n演示完成！")


def test_real_image(image_path: str) -> None:
    """
    测试真实图像的多尺度增强效果

    Args:
        image_path: 图像文件路径
    """
    if not Path(image_path).exists():
        print(f"图像文件不存在: {image_path}")
        return

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    print(f"处理图像: {image_path}")
    print(f"图像尺寸: {image.shape}")

    # 创建增强器
    enhancer = MultiScaleEnhancer()

    # 分析原始图像
    print("\n原始图像质量分析:")
    metrics = enhancer.analyze_image_quality(image)
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")

    # 执行增强
    print("\n执行自适应增强...")
    enhanced = enhancer.adaptive_enhance(image)

    # 保存结果
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # 保存增强后的图像
    output_path = output_dir / f"enhanced_{Path(image_path).name}"
    cv2.imwrite(str(output_path), enhanced)
    print(f"增强图像已保存: {output_path}")

    # 保存对比图
    comparison_path = output_dir / f"comparison_{Path(image_path).stem}.png"
    enhancer.save_comparison(image, enhanced, str(comparison_path))

    print("处理完成！")


if __name__ == "__main__":
    # 运行演示
    demo_multi_scale_enhancement()

    # 如果有真实图像，取消注释以下行进行测试
    # test_real_image("path/to/your/image.jpg")