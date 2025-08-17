#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于中值滤波的图像去雾算法实现

该模块实现了一种改进的图像去雾算法，通过中值滤波优化传输图估计，
提升去雾效果的细节保持能力和处理速度。

算法特点：
1. 中值滤波优化：使用中值滤波替代引导滤波优化传输图
2. 快速处理：相比引导滤波具有更快的处理速度
3. 边缘保持：中值滤波能较好地保持边缘信息
4. 自适应参数：根据图像内容自动调整去雾参数

作者: GlimmerLab
版本: 1.0.0
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class MedianDehazingParams:
    """中值滤波去雾算法参数配置"""

    # 暗通道滤波器窗口大小
    dark_channel_size: int = 15

    # 中值滤波核大小
    median_filter_size: int = 51

    # 最小传输图阈值
    min_transmission: float = 0.1

    # 大气光估计参数
    airlight_percentage: float = 0.001  # 选择最亮像素的百分比

    # 去雾强度因子
    omega: float = 0.95

    # 传输图优化迭代次数
    refinement_iterations: int = 3

    # 边缘保持参数
    edge_preserve_threshold: float = 50.0

    # 后处理参数
    gamma_correction: float = 1.0

    # 是否启用颜色校正
    enable_color_correction: bool = True

    def __post_init__(self):
        """参数验证"""
        if self.dark_channel_size % 2 == 0:
            self.dark_channel_size += 1
        if self.median_filter_size % 2 == 0:
            self.median_filter_size += 1

        if not (0 < self.min_transmission <= 1):
            raise ValueError("最小传输图阈值必须在(0,1]范围内")
        if not (0 < self.omega <= 1):
            raise ValueError("去雾强度因子必须在(0,1]范围内")
        if not (0 < self.airlight_percentage <= 1):
            raise ValueError("大气光估计百分比必须在(0,1]范围内")


class BaseDehazingProcessor(ABC):
    """去雾处理器基类"""

    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """处理图像的抽象方法"""
        pass


class MedianFilterDehazingProcessor(BaseDehazingProcessor):
    """基于中值滤波的去雾处理器"""

    def __init__(self, params: Optional[MedianDehazingParams] = None):
        """
        初始化中值滤波去雾处理器

        Args:
            params: 算法参数配置，如果为None则使用默认参数
        """
        self.params = params or MedianDehazingParams()
        self.logger = logging.getLogger(__name__)

    def _compute_dark_channel(self, image: np.ndarray) -> np.ndarray:
        """
        计算图像的暗通道先验

        Args:
            image: 输入图像 (H, W, 3)

        Returns:
            np.ndarray: 暗通道图 (H, W)
        """
        # 计算RGB三通道的最小值
        min_channel = np.min(image, axis=2)

        # 使用最小值滤波得到暗通道
        kernel_size = self.params.dark_channel_size
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # 形态学腐蚀操作等效于最小值滤波
        dark_channel = cv2.erode(min_channel, kernel)

        return dark_channel

    def _estimate_atmospheric_light(self, image: np.ndarray,
                                  dark_channel: np.ndarray) -> np.ndarray:
        """
        估计大气光值

        Args:
            image: 输入图像
            dark_channel: 暗通道图

        Returns:
            np.ndarray: 大气光值 (3,)
        """
        h, w = dark_channel.shape
        flat_image = image.reshape(-1, 3)
        flat_dark = dark_channel.reshape(-1)

        # 选择暗通道值最大的像素点
        num_pixels = int(h * w * self.params.airlight_percentage)
        indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]

        # 在这些像素中选择亮度最大的点作为大气光
        brightest_pixels = flat_image[indices]
        brightest_intensity = np.sum(brightest_pixels, axis=1)
        brightest_idx = indices[np.argmax(brightest_intensity)]

        atmospheric_light = flat_image[brightest_idx].astype(np.float64)

        return atmospheric_light

    def _estimate_transmission_raw(self, image: np.ndarray,
                                 atmospheric_light: np.ndarray) -> np.ndarray:
        """
        估计原始传输图

        Args:
            image: 输入图像
            atmospheric_light: 大气光值

        Returns:
            np.ndarray: 原始传输图
        """
        # 归一化图像
        normalized_image = image.astype(np.float64) / atmospheric_light

        # 计算归一化图像的暗通道
        dark_channel_norm = self._compute_dark_channel(
            (normalized_image * 255).astype(np.uint8)
        )

        # 计算传输图
        transmission = 1.0 - self.params.omega * (dark_channel_norm / 255.0)

        return transmission

    def _refine_transmission_median(self, transmission: np.ndarray,
                                  image: np.ndarray) -> np.ndarray:
        """
        使用中值滤波优化传输图

        Args:
            transmission: 原始传输图
            image: 输入图像（用于边缘检测）

        Returns:
            np.ndarray: 优化后的传输图
        """
        # 转换为合适的数据类型
        transmission_8bit = (transmission * 255).astype(np.uint8)

        # 边缘检测
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # 多次中值滤波迭代优化
        refined_transmission = transmission_8bit.copy()

        for i in range(self.params.refinement_iterations):
            # 应用中值滤波
            kernel_size = max(3, self.params.median_filter_size - i * 10)
            if kernel_size % 2 == 0:
                kernel_size += 1

            filtered = cv2.medianBlur(refined_transmission, kernel_size)

            # 在边缘处保持原始传输图
            edge_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8))
            mask = edge_dilated > self.params.edge_preserve_threshold

            refined_transmission = np.where(mask, refined_transmission, filtered)

        # 转换回浮点数
        refined_transmission = refined_transmission.astype(np.float64) / 255.0

        return refined_transmission

    def _adaptive_median_filter(self, transmission: np.ndarray,
                              window_size: int) -> np.ndarray:
        """
        自适应中值滤波

        Args:
            transmission: 输入传输图
            window_size: 滤波窗口大小

        Returns:
            np.ndarray: 滤波后的传输图
        """
        h, w = transmission.shape
        filtered = transmission.copy()
        pad_size = window_size // 2

        # 边界填充
        padded = np.pad(transmission, pad_size, mode='reflect')

        for i in range(h):
            for j in range(w):
                # 提取窗口
                window = padded[i:i+window_size, j:j+window_size]

                # 计算中值
                median_val = np.median(window)

                # 计算窗口内的方差
                variance = np.var(window)

                # 自适应权重：方差大的区域（可能是边缘）保持原值
                if variance > 0.01:  # 方差阈值
                    alpha = 0.3  # 较小的滤波权重
                else:
                    alpha = 0.8  # 较大的滤波权重

                filtered[i, j] = alpha * median_val + (1 - alpha) * transmission[i, j]

        return filtered

    def _recover_scene_radiance(self, image: np.ndarray,
                              transmission: np.ndarray,
                              atmospheric_light: np.ndarray) -> np.ndarray:
        """
        恢复场景辐射

        Args:
            image: 输入图像
            transmission: 传输图
            atmospheric_light: 大气光值

        Returns:
            np.ndarray: 去雾后的图像
        """
        # 限制传输图的最小值
        transmission = np.maximum(transmission, self.params.min_transmission)

        # 场景辐射恢复公式
        # J(x) = (I(x) - A) / max(t(x), t0) + A
        image_float = image.astype(np.float64)

        recovered = np.zeros_like(image_float)

        for c in range(3):
            numerator = image_float[:, :, c] - atmospheric_light[c]
            recovered[:, :, c] = numerator / transmission + atmospheric_light[c]

        # 限制到有效范围
        recovered = np.clip(recovered, 0, 255)

        return recovered.astype(np.uint8)

    def _color_correction(self, image: np.ndarray) -> np.ndarray:
        """
        颜色校正后处理

        Args:
            image: 输入图像

        Returns:
            np.ndarray: 颜色校正后的图像
        """
        if not self.params.enable_color_correction:
            return image

        # 简单的白平衡校正
        image_float = image.astype(np.float64)

        # 计算各通道的平均值
        mean_values = np.mean(image_float.reshape(-1, 3), axis=0)
        global_mean = np.mean(mean_values)

        # 色彩校正
        corrected = np.zeros_like(image_float)
        for c in range(3):
            if mean_values[c] > 0:
                corrected[:, :, c] = image_float[:, :, c] * (global_mean / mean_values[c])

        # Gamma校正
        if self.params.gamma_correction != 1.0:
            corrected = corrected / 255.0
            corrected = np.power(corrected, 1.0 / self.params.gamma_correction)
            corrected = corrected * 255.0

        return np.clip(corrected, 0, 255).astype(np.uint8)

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        对比度增强

        Args:
            image: 输入图像

        Returns:
            np.ndarray: 增强后的图像
        """
        # 直方图均衡化
        if len(image.shape) == 3:
            # 彩色图像：在YUV空间进行
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            # 灰度图像
            enhanced = cv2.equalizeHist(image)

        return enhanced

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        执行中值滤波去雾处理

        Args:
            image: 输入的有雾图像

        Returns:
            Tuple[np.ndarray, dict]: (去雾后图像, 中间结果字典)

        Raises:
            ValueError: 输入图像格式错误
        """
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")

        if len(image.shape) != 3:
            raise ValueError("输入必须是3通道彩色图像")

        # 确保图像为uint8格式
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        try:
            self.logger.info("开始中值滤波去雾处理")

            # 1. 计算暗通道
            self.logger.debug("计算暗通道先验")
            dark_channel = self._compute_dark_channel(image)

            # 2. 估计大气光
            self.logger.debug("估计大气光值")
            atmospheric_light = self._estimate_atmospheric_light(image, dark_channel)

            # 3. 估计原始传输图
            self.logger.debug("估计传输图")
            transmission_raw = self._estimate_transmission_raw(image, atmospheric_light)

            # 4. 中值滤波优化传输图
            self.logger.debug("使用中值滤波优化传输图")
            transmission_refined = self._refine_transmission_median(
                transmission_raw, image
            )

            # 5. 恢复场景辐射
            self.logger.debug("恢复场景辐射")
            recovered = self._recover_scene_radiance(
                image, transmission_refined, atmospheric_light
            )

            # 6. 后处理
            self.logger.debug("执行后处理")
            color_corrected = self._color_correction(recovered)
            final_result = self._enhance_contrast(color_corrected)

            # 构建中间结果字典
            intermediate_results = {
                'dark_channel': dark_channel,
                'atmospheric_light': atmospheric_light,
                'transmission_raw': transmission_raw,
                'transmission_refined': transmission_refined,
                'recovered': recovered,
                'color_corrected': color_corrected
            }

            self.logger.info("中值滤波去雾处理完成")

            return final_result, intermediate_results

        except Exception as e:
            self.logger.error(f"去雾处理过程中发生错误: {str(e)}")
            raise

    def process_simple(self, image: np.ndarray) -> np.ndarray:
        """
        简化的去雾处理接口（仅返回最终结果）

        Args:
            image: 输入图像

        Returns:
            np.ndarray: 去雾后的图像
        """
        result, _ = self.process(image)
        return result


def fast_median_dehazing(image: np.ndarray,
                        strength: str = "medium") -> np.ndarray:
    """
    快速中值滤波去雾的便捷函数

    Args:
        image: 输入的有雾图像
        strength: 去雾强度，可选 "light", "medium", "strong"

    Returns:
        np.ndarray: 去雾后的图像
    """
    # 预设参数
    if strength == "light":
        params = MedianDehazingParams(
            omega=0.85,
            median_filter_size=31,
            min_transmission=0.2,
            refinement_iterations=2
        )
    elif strength == "medium":
        params = MedianDehazingParams(
            omega=0.95,
            median_filter_size=51,
            min_transmission=0.1,
            refinement_iterations=3
        )
    elif strength == "strong":
        params = MedianDehazingParams(
            omega=0.98,
            median_filter_size=71,
            min_transmission=0.05,
            refinement_iterations=4
        )
    else:
        raise ValueError("strength必须是 'light', 'medium', 或 'strong'")

    processor = MedianFilterDehazingProcessor(params)
    return processor.process_simple(image)


def compare_dehazing_methods(image: np.ndarray) -> dict:
    """
    比较不同强度的去雾效果

    Args:
        image: 输入的有雾图像

    Returns:
        dict: 包含不同方法结果的字典
    """
    results = {}

    for strength in ["light", "medium", "strong"]:
        try:
            results[strength] = fast_median_dehazing(image, strength)
        except Exception as e:
            print(f"处理强度 {strength} 时发生错误: {str(e)}")
            results[strength] = None

    return results


def batch_process_hazy_images(input_dir: str, output_dir: str,
                            params: Optional[MedianDehazingParams] = None) -> None:
    """
    批量处理有雾图像

    Args:
        input_dir: 输入图像目录
        output_dir: 输出目录
        params: 处理参数
    """
    import os
    import glob

    processor = MedianFilterDehazingProcessor(params)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 支持的图像格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    for i, image_path in enumerate(image_paths):
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像: {image_path}")
                continue

            # 去雾处理
            dehazed = processor.process_simple(image)

            # 保存结果
            filename = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{filename}_dehazed.jpg")
            cv2.imwrite(output_path, dehazed)

            print(f"处理完成 ({i+1}/{len(image_paths)}): {output_path}")

        except Exception as e:
            print(f"处理图像 {image_path} 时发生错误: {str(e)}")


def main():
    """主函数：演示中值滤波去雾算法的使用"""

    # 配置日志
    logging.basicConfig(level=logging.INFO)

    # 示例：处理有雾图像
    image_path = "hazy_image.jpg"

    try:
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像文件: {image_path}")
            print("请提供有效的有雾图像文件")
            return

        print(f"输入图像尺寸: {image.shape}")

        # 方法1：使用默认参数
        print("使用默认参数进行去雾...")
        processor = MedianFilterDehazingProcessor()
        result1, intermediate = processor.process(image)

        # 方法2：使用自定义参数
        print("使用自定义参数进行去雾...")
        custom_params = MedianDehazingParams(
            omega=0.9,
            median_filter_size=61,
            min_transmission=0.15,
            refinement_iterations=4,
            enable_color_correction=True,
            gamma_correction=1.2
        )
        custom_processor = MedianFilterDehazingProcessor(custom_params)
        result2 = custom_processor.process_simple(image)

        # 方法3：比较不同强度效果
        print("比较不同去雾强度...")
        comparison_results = compare_dehazing_methods(image)

        # 显示结果
        cv2.imshow('Original Hazy Image', image)
        cv2.imshow('Default Dehazing', result1)
        cv2.imshow('Custom Dehazing', result2)

        # 显示中间结果
        cv2.imshow('Dark Channel', intermediate['dark_channel'])
        cv2.imshow('Transmission Map',
                  (intermediate['transmission_refined'] * 255).astype(np.uint8))

        # 显示比较结果
        for strength, result in comparison_results.items():
            if result is not None:
                cv2.imshow(f'Dehazing - {strength}', result)

        print("按任意键退出...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 保存结果
        cv2.imwrite('median_dehazing_default.jpg', result1)
        cv2.imwrite('median_dehazing_custom.jpg', result2)

        # 保存中间结果
        cv2.imwrite('dark_channel.jpg', intermediate['dark_channel'])
        cv2.imwrite('transmission_map.jpg',
                   (intermediate['transmission_refined'] * 255).astype(np.uint8))

        # 保存比较结果
        for strength, result in comparison_results.items():
            if result is not None:
                cv2.imwrite(f'median_dehazing_{strength}.jpg', result)

        print("结果已保存到文件")

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")


if __name__ == "__main__":
    main()