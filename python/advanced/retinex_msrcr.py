#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Retinex MSRCR (Multi-Scale Retinex with Color Restoration) 算法实现

该模块实现了多尺度视网膜色彩常性增强算法，用于提升图像的动态范围和色彩饱和度，
特别适用于低照度图像增强和颜色恒常性处理。

算法特点：
1. 多尺度处理：使用不同尺度的高斯核进行处理
2. 色彩恢复：通过色彩恢复因子恢复颜色信息
3. 动态范围压缩：将高动态范围映射到可显示范围
4. 自适应增强：根据图像内容自动调整增强参数

作者: GlimmerLab
版本: 1.0.0
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class RetinexMSRCRParams:
    """Retinex MSRCR算法参数配置"""

    # 多尺度高斯核标准差
    sigma_list: List[float] = None

    # 色彩恢复因子
    color_restoration: float = 125.0

    # 增益系数
    gain: float = 1.0

    # 偏移量
    offset: float = 0.0

    # 动态范围拉伸参数
    dynamic_range: Tuple[float, float] = (0.01, 0.99)

    # 是否启用色彩平衡
    enable_color_balance: bool = True

    # 处理精度（float32或float64）
    precision: str = "float32"

    def __post_init__(self):
        """参数后处理和验证"""
        if self.sigma_list is None:
            # 默认多尺度参数：小、中、大三个尺度
            self.sigma_list = [15.0, 80.0, 250.0]

        # 参数验证
        if not all(sigma > 0 for sigma in self.sigma_list):
            raise ValueError("所有sigma值必须大于0")

        if self.color_restoration < 0:
            raise ValueError("色彩恢复因子必须非负")

        if not (0 <= self.dynamic_range[0] < self.dynamic_range[1] <= 1):
            raise ValueError("动态范围参数无效")


class BaseRetinexProcessor(ABC):
    """Retinex处理器基类"""

    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """处理图像的抽象方法"""
        pass


class RetinexMSRCRProcessor(BaseRetinexProcessor):
    """Retinex MSRCR处理器主类"""

    def __init__(self, params: Optional[RetinexMSRCRParams] = None):
        """
        初始化Retinex MSRCR处理器

        Args:
            params: 算法参数配置，如果为None则使用默认参数
        """
        self.params = params or RetinexMSRCRParams()
        self.logger = logging.getLogger(__name__)

        # 预计算高斯核
        self._gaussian_kernels = self._prepare_gaussian_kernels()

    def _prepare_gaussian_kernels(self) -> List[Tuple[int, int]]:
        """
        预计算高斯核大小

        Returns:
            List[Tuple[int, int]]: 高斯核大小列表
        """
        kernels = []
        for sigma in self.params.sigma_list:
            # 根据3sigma原则计算核大小
            kernel_size = int(2 * np.ceil(3 * sigma) + 1)
            # 确保核大小为奇数
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernels.append((kernel_size, kernel_size))

        return kernels

    def _single_scale_retinex(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """
        单尺度Retinex处理

        Args:
            image: 输入图像（对数域）
            sigma: 高斯核标准差

        Returns:
            np.ndarray: 单尺度Retinex结果
        """
        # 高斯模糊
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

        # Retinex = log(I) - log(I * G)
        retinex = image - blurred

        return retinex

    def _multi_scale_retinex(self, image: np.ndarray) -> np.ndarray:
        """
        多尺度Retinex处理

        Args:
            image: 输入图像（对数域）

        Returns:
            np.ndarray: 多尺度Retinex结果
        """
        # 获取图像通道数
        if len(image.shape) == 3:
            height, width, channels = image.shape
            msr = np.zeros_like(image)

            # 分通道处理
            for c in range(channels):
                channel_msr = np.zeros((height, width), dtype=image.dtype)

                # 多尺度累加
                for sigma in self.params.sigma_list:
                    ssr = self._single_scale_retinex(image[:, :, c], sigma)
                    channel_msr += ssr

                # 平均
                msr[:, :, c] = channel_msr / len(self.params.sigma_list)
        else:
            # 灰度图像
            msr = np.zeros_like(image)
            for sigma in self.params.sigma_list:
                ssr = self._single_scale_retinex(image, sigma)
                msr += ssr
            msr /= len(self.params.sigma_list)

        return msr

    def _color_restoration(self, image: np.ndarray, msr: np.ndarray) -> np.ndarray:
        """
        色彩恢复处理

        Args:
            image: 原始图像（对数域）
            msr: 多尺度Retinex结果

        Returns:
            np.ndarray: 色彩恢复后的结果
        """
        if len(image.shape) != 3:
            # 灰度图像不需要色彩恢复
            return msr

        # 计算色彩恢复因子
        # C = β * [log(αI) - log(ΣI)]
        alpha = self.params.color_restoration

        # 计算各通道强度和
        sum_channels = np.sum(image, axis=2, keepdims=True)

        # 避免除零
        sum_channels = np.maximum(sum_channels, 1e-6)

        # 色彩恢复
        color_restored = np.zeros_like(msr)
        for c in range(image.shape[2]):
            # 计算色彩恢复因子
            restoration_factor = np.log(alpha * image[:, :, c] + 1e-6) - np.log(sum_channels[:, :, 0] + 1e-6)

            # 应用色彩恢复
            color_restored[:, :, c] = restoration_factor * msr[:, :, c]

        return color_restored

    def _dynamic_range_compression(self, image: np.ndarray) -> np.ndarray:
        """
        动态范围压缩

        Args:
            image: 输入图像

        Returns:
            np.ndarray: 压缩后的图像
        """
        # 计算百分位数进行动态范围拉伸
        low_percentile = self.params.dynamic_range[0] * 100
        high_percentile = self.params.dynamic_range[1] * 100

        if len(image.shape) == 3:
            # 彩色图像：分通道处理
            compressed = np.zeros_like(image)
            for c in range(image.shape[2]):
                channel = image[:, :, c]
                low_val = np.percentile(channel, low_percentile)
                high_val = np.percentile(channel, high_percentile)

                # 线性拉伸到[0, 255]范围
                if high_val > low_val:
                    compressed[:, :, c] = np.clip(
                        255 * (channel - low_val) / (high_val - low_val), 0, 255
                    )
                else:
                    compressed[:, :, c] = channel
        else:
            # 灰度图像
            low_val = np.percentile(image, low_percentile)
            high_val = np.percentile(image, high_percentile)

            if high_val > low_val:
                compressed = np.clip(
                    255 * (image - low_val) / (high_val - low_val), 0, 255
                )
            else:
                compressed = image

        return compressed

    def _color_balance(self, image: np.ndarray) -> np.ndarray:
        """
        色彩平衡处理

        Args:
            image: 输入图像

        Returns:
            np.ndarray: 色彩平衡后的图像
        """
        if len(image.shape) != 3 or not self.params.enable_color_balance:
            return image

        # 简单的灰度世界假设色彩平衡
        balanced = image.copy()

        for c in range(image.shape[2]):
            channel = image[:, :, c]
            mean_val = np.mean(channel)

            if mean_val > 0:
                # 调整通道使其均值接近全局均值
                global_mean = np.mean(image)
                balanced[:, :, c] = channel * (global_mean / mean_val)

        return np.clip(balanced, 0, 255)

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        执行Retinex MSRCR处理

        Args:
            image: 输入图像，支持BGR/RGB彩色图像或灰度图像

        Returns:
            np.ndarray: 增强后的图像

        Raises:
            ValueError: 输入图像格式错误
        """
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")

        # 确保图像为合适的数据类型
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # 转换为浮点数并归一化到[1, 256]范围（避免log(0)）
        if self.params.precision == "float64":
            normalized = image.astype(np.float64) + 1.0
        else:
            normalized = image.astype(np.float32) + 1.0

        # 转换到对数域
        log_image = np.log(normalized)

        try:
            # 1. 多尺度Retinex处理
            self.logger.debug("执行多尺度Retinex处理")
            msr = self._multi_scale_retinex(log_image)

            # 2. 色彩恢复（仅对彩色图像）
            if len(image.shape) == 3:
                self.logger.debug("执行色彩恢复")
                msrcr = self._color_restoration(log_image, msr)
            else:
                msrcr = msr

            # 3. 应用增益和偏移
            msrcr = self.params.gain * msrcr + self.params.offset

            # 4. 从对数域转换回线性域
            linear_result = np.exp(msrcr) - 1.0

            # 5. 动态范围压缩
            self.logger.debug("执行动态范围压缩")
            compressed = self._dynamic_range_compression(linear_result)

            # 6. 色彩平衡（可选）
            if self.params.enable_color_balance:
                self.logger.debug("执行色彩平衡")
                balanced = self._color_balance(compressed)
            else:
                balanced = compressed

            # 7. 转换为uint8格式
            result = np.clip(balanced, 0, 255).astype(np.uint8)

            return result

        except Exception as e:
            self.logger.error(f"Retinex MSRCR处理过程中发生错误: {str(e)}")
            raise


class RetinexVariations:
    """Retinex算法变体实现"""

    @staticmethod
    def single_scale_retinex(image: np.ndarray, sigma: float = 80.0) -> np.ndarray:
        """
        单尺度Retinex处理

        Args:
            image: 输入图像
            sigma: 高斯核标准差

        Returns:
            np.ndarray: 处理后的图像
        """
        params = RetinexMSRCRParams(sigma_list=[sigma], color_restoration=0.0)
        processor = RetinexMSRCRProcessor(params)
        return processor.process(image)

    @staticmethod
    def multi_scale_retinex(image: np.ndarray,
                          sigma_list: List[float] = None) -> np.ndarray:
        """
        多尺度Retinex处理（不含色彩恢复）

        Args:
            image: 输入图像
            sigma_list: 多尺度参数列表

        Returns:
            np.ndarray: 处理后的图像
        """
        if sigma_list is None:
            sigma_list = [15.0, 80.0, 250.0]

        params = RetinexMSRCRParams(sigma_list=sigma_list, color_restoration=0.0)
        processor = RetinexMSRCRProcessor(params)
        return processor.process(image)


def enhance_low_light_image(image: np.ndarray,
                          enhancement_level: str = "medium") -> np.ndarray:
    """
    低照度图像增强的便捷函数

    Args:
        image: 输入的低照度图像
        enhancement_level: 增强级别，可选 "light", "medium", "strong"

    Returns:
        np.ndarray: 增强后的图像
    """
    # 预设参数
    if enhancement_level == "light":
        params = RetinexMSRCRParams(
            sigma_list=[25.0, 120.0, 300.0],
            color_restoration=80.0,
            gain=1.2,
            dynamic_range=(0.02, 0.98)
        )
    elif enhancement_level == "medium":
        params = RetinexMSRCRParams(
            sigma_list=[15.0, 80.0, 250.0],
            color_restoration=125.0,
            gain=1.5,
            dynamic_range=(0.01, 0.99)
        )
    elif enhancement_level == "strong":
        params = RetinexMSRCRParams(
            sigma_list=[10.0, 60.0, 200.0],
            color_restoration=150.0,
            gain=2.0,
            dynamic_range=(0.005, 0.995)
        )
    else:
        raise ValueError("enhancement_level必须是 'light', 'medium', 或 'strong'")

    processor = RetinexMSRCRProcessor(params)
    return processor.process(image)


def batch_process_images(image_paths: List[str],
                        output_dir: str,
                        params: Optional[RetinexMSRCRParams] = None) -> None:
    """
    批量处理图像

    Args:
        image_paths: 输入图像路径列表
        output_dir: 输出目录
        params: 处理参数
    """
    import os

    processor = RetinexMSRCRProcessor(params)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, image_path in enumerate(image_paths):
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像: {image_path}")
                continue

            # 处理图像
            enhanced = processor.process(image)

            # 保存结果
            filename = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{filename}_retinex.jpg")
            cv2.imwrite(output_path, enhanced)

            print(f"处理完成 ({i+1}/{len(image_paths)}): {output_path}")

        except Exception as e:
            print(f"处理图像 {image_path} 时发生错误: {str(e)}")


def main():
    """主函数：演示Retinex MSRCR算法的使用"""

    # 配置日志
    logging.basicConfig(level=logging.INFO)

    # 示例：处理低照度图像
    image_path = "low_light_image.jpg"

    try:
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像文件: {image_path}")
            print("请提供有效的图像文件路径")
            return

        print(f"输入图像尺寸: {image.shape}")

        # 方法1：使用默认参数
        print("使用默认参数进行处理...")
        processor = RetinexMSRCRProcessor()
        result1 = processor.process(image)

        # 方法2：使用自定义参数
        print("使用自定义参数进行处理...")
        custom_params = RetinexMSRCRParams(
            sigma_list=[20.0, 100.0, 300.0],
            color_restoration=150.0,
            gain=1.8,
            dynamic_range=(0.01, 0.99),
            enable_color_balance=True
        )
        custom_processor = RetinexMSRCRProcessor(custom_params)
        result2 = custom_processor.process(image)

        # 方法3：使用便捷函数
        print("使用便捷函数进行处理...")
        result3 = enhance_low_light_image(image, "strong")

        # 显示结果
        cv2.imshow('Original', image)
        cv2.imshow('Default Retinex', result1)
        cv2.imshow('Custom Retinex', result2)
        cv2.imshow('Enhanced Low Light', result3)

        print("按任意键退出...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 保存结果
        cv2.imwrite('retinex_default.jpg', result1)
        cv2.imwrite('retinex_custom.jpg', result2)
        cv2.imwrite('retinex_enhanced.jpg', result3)
        print("结果已保存到文件")

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")


if __name__ == "__main__":
    main()