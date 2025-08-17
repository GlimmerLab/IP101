#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
快速单图去雾算法实现

该模块实现了一种高效的单图像去雾算法，通过优化的暗通道先验计算和快速传输图估计，
在保证去雾质量的同时显著提升处理速度。

算法特点：
1. 快速暗通道计算：使用形态学操作加速暗通道计算
2. 分块处理：对大图像进行分块处理以提升效率
3. 快速大气光估计：基于直方图的快速大气光值估计
4. 优化传输图：使用快速滤波技术优化传输图

作者: GlimmerLab
版本: 1.0.0
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, List, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time


@dataclass
class FastDefoggingParams:
    """快速去雾算法参数配置"""

    # 暗通道计算参数
    dark_channel_size: int = 15

    # 最小传输图阈值
    min_transmission: float = 0.1

    # 去雾强度因子
    omega: float = 0.95

    # 大气光估计参数
    airlight_percentage: float = 0.001

    # 分块处理参数
    patch_size: int = 256
    patch_overlap: int = 32

    # 快速滤波参数
    fast_filter_radius: int = 10
    fast_filter_eps: float = 0.01

    # 后处理参数
    enhance_contrast: bool = True
    gamma_correction: float = 1.0

    # 是否使用多线程
    use_multithread: bool = True

    # 内存优化模式
    memory_efficient: bool = False

    def __post_init__(self):
        """参数验证和调整"""
        if self.dark_channel_size % 2 == 0:
            self.dark_channel_size += 1

        if not (0 < self.min_transmission <= 1):
            raise ValueError("最小传输图阈值必须在(0,1]范围内")
        if not (0 < self.omega <= 1):
            raise ValueError("去雾强度因子必须在(0,1]范围内")
        if not (0 < self.airlight_percentage <= 1):
            raise ValueError("大气光估计百分比必须在(0,1]范围内")


class BaseFastProcessor(ABC):
    """快速处理器基类"""

    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """处理图像的抽象方法"""
        pass


class FastDefoggingProcessor(BaseFastProcessor):
    """快速去雾处理器主类"""

    def __init__(self, params: Optional[FastDefoggingParams] = None):
        """
        初始化快速去雾处理器

        Args:
            params: 算法参数配置，如果为None则使用默认参数
        """
        self.params = params or FastDefoggingParams()
        self.logger = logging.getLogger(__name__)

        # 性能统计
        self.timing_stats = {}

    def _fast_dark_channel(self, image: np.ndarray) -> np.ndarray:
        """
        快速计算暗通道先验

        Args:
            image: 输入图像 (H, W, 3)

        Returns:
            np.ndarray: 暗通道图 (H, W)
        """
        start_time = time.time()

        # 方法1：使用OpenCV的快速最小值滤波
        min_channel = np.min(image, axis=2)

        # 使用形态学腐蚀进行最小值滤波
        kernel_size = self.params.dark_channel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dark_channel = cv2.erode(min_channel, kernel)

        self.timing_stats['dark_channel'] = time.time() - start_time
        return dark_channel

    def _fast_atmospheric_light(self, image: np.ndarray,
                              dark_channel: np.ndarray) -> np.ndarray:
        """
        快速估计大气光值

        Args:
            image: 输入图像
            dark_channel: 暗通道图

        Returns:
            np.ndarray: 大气光值 (3,)
        """
        start_time = time.time()

        # 使用直方图快速找到最亮的像素
        h, w = dark_channel.shape

        # 将暗通道和图像展平
        flat_dark = dark_channel.reshape(-1)
        flat_image = image.reshape(-1, 3)

        # 选择暗通道值最大的像素
        num_pixels = max(1, int(h * w * self.params.airlight_percentage))

        # 使用argpartition快速选择最大值
        if num_pixels < len(flat_dark):
            indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
        else:
            indices = np.arange(len(flat_dark))

        # 在选中的像素中找到最亮的
        candidate_pixels = flat_image[indices]
        intensities = np.sum(candidate_pixels, axis=1)
        brightest_idx = indices[np.argmax(intensities)]

        atmospheric_light = flat_image[brightest_idx].astype(np.float64)

        self.timing_stats['atmospheric_light'] = time.time() - start_time
        return atmospheric_light

    def _fast_transmission_estimation(self, image: np.ndarray,
                                    atmospheric_light: np.ndarray) -> np.ndarray:
        """
        快速传输图估计

        Args:
            image: 输入图像
            atmospheric_light: 大气光值

        Returns:
            np.ndarray: 传输图
        """
        start_time = time.time()

        # 避免除零
        atmospheric_light = np.maximum(atmospheric_light, 1.0)

        # 归一化图像
        normalized = image.astype(np.float64) / atmospheric_light

        # 快速计算暗通道
        normalized_uint8 = np.clip(normalized * 255, 0, 255).astype(np.uint8)
        dark_norm = self._fast_dark_channel(normalized_uint8)

        # 计算传输图
        transmission = 1.0 - self.params.omega * (dark_norm.astype(np.float64) / 255.0)

        self.timing_stats['transmission'] = time.time() - start_time
        return transmission

    def _fast_guided_filter_approximation(self, transmission: np.ndarray,
                                        guide_image: np.ndarray) -> np.ndarray:
        """
        快速引导滤波近似

        Args:
            transmission: 输入传输图
            guide_image: 引导图像

        Returns:
            np.ndarray: 滤波后的传输图
        """
        start_time = time.time()

        # 转换引导图像为灰度
        if len(guide_image.shape) == 3:
            guide_gray = cv2.cvtColor(guide_image, cv2.COLOR_BGR2GRAY)
        else:
            guide_gray = guide_image

        # 使用双边滤波作为引导滤波的快速近似
        guide_gray_8bit = guide_gray.astype(np.uint8)
        transmission_8bit = (transmission * 255).astype(np.uint8)

        radius = self.params.fast_filter_radius
        sigma_color = 50
        sigma_space = 50

        filtered = cv2.bilateralFilter(transmission_8bit, radius, sigma_color, sigma_space)

        # 转换回浮点数
        filtered_transmission = filtered.astype(np.float64) / 255.0

        self.timing_stats['filtering'] = time.time() - start_time
        return filtered_transmission

    def _fast_box_filter(self, image: np.ndarray, radius: int) -> np.ndarray:
        """
        快速盒式滤波实现

        Args:
            image: 输入图像
            radius: 滤波半径

        Returns:
            np.ndarray: 滤波后的图像
        """
        # 使用积分图像实现快速盒式滤波
        if len(image.shape) == 2:
            # 灰度图像
            integral = cv2.integral(image.astype(np.float32))
            h, w = image.shape

            filtered = np.zeros_like(image, dtype=np.float32)

            for i in range(h):
                for j in range(w):
                    # 确定窗口边界
                    y1 = max(0, i - radius)
                    y2 = min(h - 1, i + radius)
                    x1 = max(0, j - radius)
                    x2 = min(w - 1, j + radius)

                    # 使用积分图像计算区域和
                    area = (y2 - y1 + 1) * (x2 - x1 + 1)
                    if area > 0:
                        sum_val = (integral[y2 + 1, x2 + 1] -
                                 integral[y1, x2 + 1] -
                                 integral[y2 + 1, x1] +
                                 integral[y1, x1])
                        filtered[i, j] = sum_val / area

            return filtered
        else:
            # 彩色图像：分通道处理
            filtered = np.zeros_like(image, dtype=np.float32)
            for c in range(image.shape[2]):
                filtered[:, :, c] = self._fast_box_filter(image[:, :, c], radius)
            return filtered

    def _patch_based_processing(self, image: np.ndarray) -> np.ndarray:
        """
        基于分块的图像处理

        Args:
            image: 输入图像

        Returns:
            np.ndarray: 处理后的图像
        """
        if not self.params.memory_efficient:
            # 不使用分块处理
            return self._process_single_patch(image)

        h, w = image.shape[:2]
        patch_size = self.params.patch_size
        overlap = self.params.patch_overlap

        # 计算分块数量
        num_patches_h = (h + patch_size - 1) // patch_size
        num_patches_w = (w + patch_size - 1) // patch_size

        result = np.zeros_like(image, dtype=np.float64)
        weight_map = np.zeros((h, w), dtype=np.float64)

        for i in range(num_patches_h):
            for j in range(num_patches_w):
                # 计算分块边界
                y1 = i * patch_size
                y2 = min(h, y1 + patch_size + overlap)
                x1 = j * patch_size
                x2 = min(w, x1 + patch_size + overlap)

                # 提取分块
                patch = image[y1:y2, x1:x2]

                # 处理分块
                processed_patch = self._process_single_patch(patch)

                # 计算权重（中心权重更高）
                patch_h, patch_w = processed_patch.shape[:2]
                weight_patch = np.ones((patch_h, patch_w))

                # 边缘衰减权重
                fade_size = overlap // 2
                if fade_size > 0:
                    for y in range(patch_h):
                        for x in range(patch_w):
                            dist_to_edge = min(y, x, patch_h - 1 - y, patch_w - 1 - x)
                            if dist_to_edge < fade_size:
                                weight_patch[y, x] = dist_to_edge / fade_size

                # 累加结果
                if len(image.shape) == 3:
                    for c in range(3):
                        result[y1:y2, x1:x2, c] += processed_patch[:, :, c] * weight_patch
                else:
                    result[y1:y2, x1:x2] += processed_patch * weight_patch

                weight_map[y1:y2, x1:x2] += weight_patch

        # 归一化
        weight_map = np.maximum(weight_map, 1e-6)
        if len(image.shape) == 3:
            for c in range(3):
                result[:, :, c] /= weight_map
        else:
            result /= weight_map

        return result.astype(np.uint8)

    def _process_single_patch(self, image: np.ndarray) -> np.ndarray:
        """
        处理单个图像块

        Args:
            image: 输入图像块

        Returns:
            np.ndarray: 处理后的图像块
        """
        # 1. 计算暗通道
        dark_channel = self._fast_dark_channel(image)

        # 2. 估计大气光
        atmospheric_light = self._fast_atmospheric_light(image, dark_channel)

        # 3. 估计传输图
        transmission = self._fast_transmission_estimation(image, atmospheric_light)

        # 4. 快速滤波优化传输图
        transmission_refined = self._fast_guided_filter_approximation(
            transmission, image
        )

        # 5. 恢复场景辐射
        result = self._recover_scene_radiance(
            image, transmission_refined, atmospheric_light
        )

        return result

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
        start_time = time.time()

        # 限制传输图的最小值
        transmission = np.maximum(transmission, self.params.min_transmission)

        # 场景辐射恢复
        image_float = image.astype(np.float64)
        recovered = np.zeros_like(image_float)

        for c in range(3):
            numerator = image_float[:, :, c] - atmospheric_light[c]
            recovered[:, :, c] = numerator / transmission + atmospheric_light[c]

        # 限制到有效范围
        recovered = np.clip(recovered, 0, 255)

        self.timing_stats['recovery'] = time.time() - start_time
        return recovered.astype(np.uint8)

    def _post_processing(self, image: np.ndarray) -> np.ndarray:
        """
        后处理增强

        Args:
            image: 输入图像

        Returns:
            np.ndarray: 增强后的图像
        """
        start_time = time.time()

        result = image.copy()

        # 对比度增强
        if self.params.enhance_contrast:
            # 使用CLAHE进行局部对比度增强
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Gamma校正
        if self.params.gamma_correction != 1.0:
            result = result.astype(np.float64) / 255.0
            result = np.power(result, 1.0 / self.params.gamma_correction)
            result = (result * 255.0).astype(np.uint8)

        self.timing_stats['post_processing'] = time.time() - start_time
        return result

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        执行快速去雾处理

        Args:
            image: 输入的有雾图像

        Returns:
            Tuple[np.ndarray, dict]: (去雾后图像, 统计信息)

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

        total_start_time = time.time()
        self.timing_stats.clear()

        try:
            self.logger.info("开始快速去雾处理")

            # 根据图像大小选择处理策略
            h, w = image.shape[:2]
            if h * w > 1000000 and self.params.memory_efficient:
                # 大图像使用分块处理
                self.logger.debug("使用分块处理大图像")
                result = self._patch_based_processing(image)
            else:
                # 小图像直接处理
                self.logger.debug("直接处理图像")
                result = self._process_single_patch(image)

            # 后处理
            self.logger.debug("执行后处理")
            final_result = self._post_processing(result)

            # 统计总时间
            total_time = time.time() - total_start_time
            self.timing_stats['total'] = total_time

            # 构建统计信息
            stats = {
                'timing': self.timing_stats.copy(),
                'image_size': f"{w}x{h}",
                'processing_mode': 'patch_based' if h * w > 1000000 and self.params.memory_efficient else 'direct',
                'fps': 1.0 / total_time if total_time > 0 else 0,
                'params': self.params
            }

            self.logger.info(f"快速去雾处理完成，耗时: {total_time:.3f}秒")

            return final_result, stats

        except Exception as e:
            self.logger.error(f"快速去雾处理过程中发生错误: {str(e)}")
            raise

    def process_simple(self, image: np.ndarray) -> np.ndarray:
        """
        简化的去雾处理接口

        Args:
            image: 输入图像

        Returns:
            np.ndarray: 去雾后的图像
        """
        result, _ = self.process(image)
        return result


def quick_defog(image: np.ndarray, quality: str = "balanced") -> np.ndarray:
    """
    快速去雾的便捷函数

    Args:
        image: 输入的有雾图像
        quality: 质量级别，可选 "fast", "balanced", "quality"

    Returns:
        np.ndarray: 去雾后的图像
    """
    # 预设参数
    if quality == "fast":
        params = FastDefoggingParams(
            dark_channel_size=7,
            fast_filter_radius=5,
            patch_size=128,
            memory_efficient=True,
            enhance_contrast=False
        )
    elif quality == "balanced":
        params = FastDefoggingParams(
            dark_channel_size=15,
            fast_filter_radius=10,
            patch_size=256,
            memory_efficient=False,
            enhance_contrast=True
        )
    elif quality == "quality":
        params = FastDefoggingParams(
            dark_channel_size=21,
            fast_filter_radius=15,
            patch_size=512,
            memory_efficient=False,
            enhance_contrast=True,
            gamma_correction=1.1
        )
    else:
        raise ValueError("quality必须是 'fast', 'balanced', 或 'quality'")

    processor = FastDefoggingProcessor(params)
    return processor.process_simple(image)


def benchmark_defogging_speed(image: np.ndarray, num_runs: int = 5) -> dict:
    """
    基准测试去雾速度

    Args:
        image: 测试图像
        num_runs: 运行次数

    Returns:
        dict: 基准测试结果
    """
    results = {}

    for quality in ["fast", "balanced", "quality"]:
        times = []

        for _ in range(num_runs):
            start_time = time.time()
            try:
                _ = quick_defog(image, quality)
                elapsed = time.time() - start_time
                times.append(elapsed)
            except Exception as e:
                print(f"质量级别 {quality} 测试失败: {str(e)}")
                times.append(float('inf'))

        if times:
            avg_time = np.mean(times)
            std_time = np.std(times)
            fps = 1.0 / avg_time if avg_time > 0 else 0

            results[quality] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'fps': fps,
                'times': times
            }

    return results


def batch_fast_defogging(input_dir: str, output_dir: str,
                        quality: str = "balanced") -> None:
    """
    批量快速去雾处理

    Args:
        input_dir: 输入图像目录
        output_dir: 输出目录
        quality: 处理质量级别
    """
    import os
    import glob

    processor = FastDefoggingProcessor()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 支持的图像格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))

    total_start_time = time.time()

    for i, image_path in enumerate(image_paths):
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像: {image_path}")
                continue

            # 快速去雾处理
            start_time = time.time()
            dehazed = quick_defog(image, quality)
            process_time = time.time() - start_time

            # 保存结果
            filename = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{filename}_fast_dehazed.jpg")
            cv2.imwrite(output_path, dehazed)

            print(f"处理完成 ({i+1}/{len(image_paths)}): {output_path}, "
                  f"耗时: {process_time:.3f}秒")

        except Exception as e:
            print(f"处理图像 {image_path} 时发生错误: {str(e)}")

    total_time = time.time() - total_start_time
    print(f"批量处理完成，总耗时: {total_time:.3f}秒")


def main():
    """主函数：演示快速去雾算法的使用"""

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

        # 基准测试
        print("开始基准测试...")
        benchmark_results = benchmark_defogging_speed(image, num_runs=3)

        print("\n基准测试结果:")
        for quality, result in benchmark_results.items():
            print(f"{quality:>10}: {result['avg_time']:.3f}±{result['std_time']:.3f}秒, "
                  f"FPS: {result['fps']:.1f}")

        # 不同质量级别处理
        results = {}
        for quality in ["fast", "balanced", "quality"]:
            print(f"\n使用{quality}模式进行处理...")
            processor = FastDefoggingProcessor()
            start_time = time.time()
            results[quality], stats = processor.process(image)
            process_time = time.time() - start_time

            print(f"处理时间: {process_time:.3f}秒")
            print(f"详细统计: {stats['timing']}")

        # 显示结果
        cv2.imshow('Original Hazy Image', image)

        for quality, result in results.items():
            cv2.imshow(f'Fast Defogging - {quality}', result)

        print("\n按任意键退出...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 保存结果
        for quality, result in results.items():
            cv2.imwrite(f'fast_defogging_{quality}.jpg', result)

        print("结果已保存到文件")

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")


if __name__ == "__main__":
    main()