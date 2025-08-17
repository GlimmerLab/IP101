"""
自动白平衡算法 - Python实现
Automatic White Balance Algorithm

基于色彩统计的图像色温校正技术，
支持多种白平衡算法策略和自适应融合。

Author: GlimmerLab
Date: 2024
"""

import cv2
import numpy as np
import logging
import time
from typing import Tuple, Optional, List, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path
import math

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AWBParams:
    """自动白平衡参数配置"""
    percentile: float = 99.0           # 白色补丁百分位数
    gray_world_weight: float = 0.4     # 灰度世界算法权重
    white_patch_weight: float = 0.3    # 白色补丁算法权重
    perfect_ref_weight: float = 0.3    # 完美反射面算法权重
    brightness_threshold: float = 0.8  # 亮度阈值
    balance_threshold: float = 20.0    # 色彩平衡阈值
    color_temp_sensitivity: float = 0.1  # 色温敏感度


@dataclass
class AWBResult:
    """白平衡处理结果数据结构"""
    corrected_image: np.ndarray      # 校正后的图像
    gain_factors: np.ndarray         # 增益系数
    color_temperature: float         # 估计的色温
    balance_quality: float           # 平衡质量评分
    processing_time: float           # 处理时间


class AutomaticWhiteBalancer:
    """自动白平衡处理类"""

    def __init__(self, params: Optional[AWBParams] = None):
        """
        初始化自动白平衡器

        Args:
            params: 白平衡参数配置
        """
        self.params = params or AWBParams()
        logger.info(f"初始化自动白平衡器，百分位数: {self.params.percentile}")

    def gray_world_balance(self, image: np.ndarray) -> AWBResult:
        """
        灰度世界白平衡算法

        Args:
            image: 输入图像

        Returns:
            白平衡处理结果
        """
        start_time = time.time()

        if image is None or image.size == 0:
            raise ValueError("输入图像为空")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("输入图像必须是三通道彩色图像")

        # 转换为浮点格式
        img_float = image.astype(np.float32) / 255.0

        # 计算各通道均值
        mean_b = np.mean(img_float[:, :, 0])
        mean_g = np.mean(img_float[:, :, 1])
        mean_r = np.mean(img_float[:, :, 2])

        # 计算全局灰度基准
        gray_value = (mean_b + mean_g + mean_r) / 3.0

        # 计算增益系数
        gain_b = gray_value / (mean_b + 1e-6)
        gain_g = gray_value / (mean_g + 1e-6)
        gain_r = gray_value / (mean_r + 1e-6)

        gain_factors = np.array([gain_b, gain_g, gain_r])

        # 应用颜色校正
        corrected = img_float.copy()
        corrected[:, :, 0] *= gain_b
        corrected[:, :, 1] *= gain_g
        corrected[:, :, 2] *= gain_r

        # 限制到有效范围并转换回uint8
        corrected = np.clip(corrected, 0, 1)
        corrected_image = (corrected * 255).astype(np.uint8)

        # 计算色温和质量评分
        color_temp = self._estimate_color_temperature(gain_factors)
        balance_quality = self._evaluate_balance_quality(corrected_image)

        processing_time = time.time() - start_time

        return AWBResult(
            corrected_image=corrected_image,
            gain_factors=gain_factors,
            color_temperature=color_temp,
            balance_quality=balance_quality,
            processing_time=processing_time
        )

    def white_patch_balance(self, image: np.ndarray) -> AWBResult:
        """
        白色补丁白平衡算法

        Args:
            image: 输入图像

        Returns:
            白平衡处理结果
        """
        start_time = time.time()

        if image is None or image.size == 0:
            raise ValueError("输入图像为空")

        # 转换为浮点格式
        img_float = image.astype(np.float32) / 255.0

        # 计算各通道的最大值（使用百分位数避免噪声）
        max_b = np.percentile(img_float[:, :, 0], self.params.percentile)
        max_g = np.percentile(img_float[:, :, 1], self.params.percentile)
        max_r = np.percentile(img_float[:, :, 2], self.params.percentile)

        # 找到全局最大值作为白点参考
        white_point = max(max_b, max_g, max_r)

        # 计算增益系数
        gain_b = white_point / (max_b + 1e-6)
        gain_g = white_point / (max_g + 1e-6)
        gain_r = white_point / (max_r + 1e-6)

        gain_factors = np.array([gain_b, gain_g, gain_r])

        # 应用校正
        corrected = img_float.copy()
        corrected[:, :, 0] *= gain_b
        corrected[:, :, 1] *= gain_g
        corrected[:, :, 2] *= gain_r

        # 限制到有效范围
        corrected = np.clip(corrected, 0, 1)
        corrected_image = (corrected * 255).astype(np.uint8)

        # 计算质量指标
        color_temp = self._estimate_color_temperature(gain_factors)
        balance_quality = self._evaluate_balance_quality(corrected_image)

        processing_time = time.time() - start_time

        return AWBResult(
            corrected_image=corrected_image,
            gain_factors=gain_factors,
            color_temperature=color_temp,
            balance_quality=balance_quality,
            processing_time=processing_time
        )

    def perfect_reflector_balance(self, image: np.ndarray) -> AWBResult:
        """
        完美反射面白平衡算法

        Args:
            image: 输入图像

        Returns:
            白平衡处理结果
        """
        start_time = time.time()

        if image is None or image.size == 0:
            raise ValueError("输入图像为空")

        # 转换为浮点格式
        img_float = image.astype(np.float32) / 255.0

        # 计算每个像素的亮度和色彩平衡度
        brightness = np.mean(img_float, axis=2)
        color_std = np.std(img_float, axis=2)

        # 寻找亮度高且色彩平衡的像素
        brightness_threshold = np.percentile(brightness, 90)
        balance_threshold = np.percentile(color_std, 30)

        # 创建白色候选掩码
        white_mask = (brightness > brightness_threshold) & (color_std < balance_threshold)

        if np.sum(white_mask) > 0:
            # 计算白色区域的平均颜色
            white_pixels = img_float[white_mask]
            mean_white_b = np.mean(white_pixels[:, 0])
            mean_white_g = np.mean(white_pixels[:, 1])
            mean_white_r = np.mean(white_pixels[:, 2])

            # 以理想白色为目标
            target_white = 1.0

            # 计算增益
            gain_b = target_white / (mean_white_b + 1e-6)
            gain_g = target_white / (mean_white_g + 1e-6)
            gain_r = target_white / (mean_white_r + 1e-6)

            # 应用校正
            corrected = img_float.copy()
            corrected[:, :, 0] *= gain_b
            corrected[:, :, 1] *= gain_g
            corrected[:, :, 2] *= gain_r

            gain_factors = np.array([gain_b, gain_g, gain_r])
        else:
            # 找不到理想白色区域，退回到灰度世界方法
            logger.warning("未找到合适的白色区域，使用灰度世界方法")
            result = self.gray_world_balance(image)
            return result

        # 限制到有效范围
        corrected = np.clip(corrected, 0, 1)
        corrected_image = (corrected * 255).astype(np.uint8)

        # 计算质量指标
        color_temp = self._estimate_color_temperature(gain_factors)
        balance_quality = self._evaluate_balance_quality(corrected_image)

        processing_time = time.time() - start_time

        return AWBResult(
            corrected_image=corrected_image,
            gain_factors=gain_factors,
            color_temperature=color_temp,
            balance_quality=balance_quality,
            processing_time=processing_time
        )

    def adaptive_white_balance(self, image: np.ndarray) -> AWBResult:
        """
        自适应白平衡算法（多算法融合）

        Args:
            image: 输入图像

        Returns:
            白平衡处理结果
        """
        start_time = time.time()

        # 分别使用三种方法
        gray_result = self.gray_world_balance(image)
        patch_result = self.white_patch_balance(image)
        reflector_result = self.perfect_reflector_balance(image)

        # 评估各算法的可信度
        gray_confidence = gray_result.balance_quality
        patch_confidence = patch_result.balance_quality
        reflector_confidence = reflector_result.balance_quality

        # 归一化权重
        total_confidence = gray_confidence + patch_confidence + reflector_confidence
        if total_confidence > 0:
            gray_weight = gray_confidence / total_confidence
            patch_weight = patch_confidence / total_confidence
            reflector_weight = reflector_confidence / total_confidence
        else:
            # 使用默认权重
            gray_weight = self.params.gray_world_weight
            patch_weight = self.params.white_patch_weight
            reflector_weight = self.params.perfect_ref_weight

        # 加权融合结果
        gray_img = gray_result.corrected_image.astype(np.float32)
        patch_img = patch_result.corrected_image.astype(np.float32)
        reflector_img = reflector_result.corrected_image.astype(np.float32)

        fused_image = (gray_weight * gray_img +
                      patch_weight * patch_img +
                      reflector_weight * reflector_img)

        corrected_image = np.clip(fused_image, 0, 255).astype(np.uint8)

        # 计算融合后的增益系数
        fused_gains = (gray_weight * gray_result.gain_factors +
                      patch_weight * patch_result.gain_factors +
                      reflector_weight * reflector_result.gain_factors)

        # 计算质量指标
        color_temp = self._estimate_color_temperature(fused_gains)
        balance_quality = self._evaluate_balance_quality(corrected_image)

        processing_time = time.time() - start_time

        return AWBResult(
            corrected_image=corrected_image,
            gain_factors=fused_gains,
            color_temperature=color_temp,
            balance_quality=balance_quality,
            processing_time=processing_time
        )

    def _estimate_color_temperature(self, gain_factors: np.ndarray) -> float:
        """
        估计色温

        Args:
            gain_factors: 增益系数

        Returns:
            估计的色温值（单位：K）
        """
        # 基于增益系数的简化色温估计
        gain_b, gain_g, gain_r = gain_factors

        # 计算蓝红比值
        blue_red_ratio = gain_b / (gain_r + 1e-6)

        # 简化的色温映射
        if blue_red_ratio > 1.5:
            color_temp = 7000  # 冷色调
        elif blue_red_ratio > 1.2:
            color_temp = 6500  # 日光
        elif blue_red_ratio > 0.8:
            color_temp = 5000  # 平衡
        elif blue_red_ratio > 0.6:
            color_temp = 3500  # 荧光灯
        else:
            color_temp = 2700  # 白炽灯

        return color_temp

    def _evaluate_balance_quality(self, image: np.ndarray) -> float:
        """
        评估色彩平衡质量

        Args:
            image: 输入图像

        Returns:
            平衡质量评分 [0, 1]
        """
        # 转换到Lab颜色空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

        # 分离通道
        l_channel, a_channel, b_channel = cv2.split(lab)

        # 计算a*和b*通道的标准差
        a_std = np.std(a_channel.astype(np.float32))
        b_std = np.std(b_channel.astype(np.float32))

        # 平衡度评分：标准差越小，平衡度越高
        balance_score = 1.0 / (1.0 + (a_std + b_std) / 100.0)

        return balance_score

    def analyze_image_statistics(self, image: np.ndarray) -> Dict[str, Any]:
        """
        分析图像的色彩统计特征

        Args:
            image: 输入图像

        Returns:
            统计分析结果
        """
        # 转换为浮点格式
        img_float = image.astype(np.float32) / 255.0

        # 计算各通道统计量
        channels = cv2.split(img_float)
        stats = {}

        for i, (channel, name) in enumerate(zip(channels, ['Blue', 'Green', 'Red'])):
            stats[f'{name}_mean'] = np.mean(channel)
            stats[f'{name}_std'] = np.std(channel)
            stats[f'{name}_min'] = np.min(channel)
            stats[f'{name}_max'] = np.max(channel)

        # 计算色彩比值
        mean_b, mean_g, mean_r = [np.mean(ch) for ch in channels]
        stats['RG_ratio'] = mean_r / (mean_g + 1e-6)
        stats['BG_ratio'] = mean_b / (mean_g + 1e-6)
        stats['BR_ratio'] = mean_b / (mean_r + 1e-6)

        # 计算亮度统计
        brightness = np.mean(img_float, axis=2)
        stats['brightness_mean'] = np.mean(brightness)
        stats['brightness_std'] = np.std(brightness)

        return stats

    def compare_methods(self, image: np.ndarray) -> Dict[str, AWBResult]:
        """
        比较不同白平衡方法的效果

        Args:
            image: 输入图像

        Returns:
            各方法的处理结果
        """
        results = {}

        methods = {
            'gray_world': self.gray_world_balance,
            'white_patch': self.white_patch_balance,
            'perfect_reflector': self.perfect_reflector_balance,
            'adaptive': self.adaptive_white_balance
        }

        for name, method in methods.items():
            try:
                result = method(image)
                results[name] = result
                logger.info(f"{name} 方法 - 质量评分: {result.balance_quality:.3f}, "
                          f"处理时间: {result.processing_time*1000:.1f}ms")
            except Exception as e:
                logger.error(f"{name} 方法处理失败: {e}")

        return results

    def visualize_results(self, image: np.ndarray, results: Dict[str, AWBResult],
                         save_path: Optional[str] = None) -> None:
        """
        可视化不同方法的处理结果

        Args:
            image: 原始图像
            results: 处理结果字典
            save_path: 保存路径
        """
        import matplotlib.pyplot as plt

        num_results = len(results) + 1  # +1 for original
        fig, axes = plt.subplots(2, (num_results + 1) // 2, figsize=(15, 8))
        if num_results <= 2:
            axes = axes.reshape(1, -1)

        # 显示原图
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')

        # 显示处理结果
        for i, (name, result) in enumerate(results.items(), 1):
            row = i // ((num_results + 1) // 2)
            col = i % ((num_results + 1) // 2)

            axes[row, col].imshow(cv2.cvtColor(result.corrected_image, cv2.COLOR_BGR2RGB))
            axes[row, col].set_title(f'{name}\nQuality: {result.balance_quality:.3f}')
            axes[row, col].axis('off')

        # 隐藏多余的子图
        for i in range(num_results, axes.size):
            row = i // ((num_results + 1) // 2)
            col = i % ((num_results + 1) // 2)
            if row < axes.shape[0] and col < axes.shape[1]:
                axes[row, col].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"结果图像已保存至: {save_path}")

        plt.show()


def demo_basic_awb():
    """演示基础自动白平衡功能"""
    print("=== 自动白平衡基础演示 ===")

    # 创建白平衡器
    awb = AutomaticWhiteBalancer()

    # 创建测试图像
    test_image = create_test_image()

    print(f"测试图像尺寸: {test_image.shape}")

    # 分析图像统计
    stats = awb.analyze_image_statistics(test_image)
    print("图像统计分析:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")

    # 测试不同方法
    methods = ['gray_world', 'white_patch', 'perfect_reflector', 'adaptive']

    for method_name in methods:
        print(f"\n测试 {method_name} 方法:")
        method = getattr(awb, f"{method_name}_balance")
        result = method(test_image)

        print(f"  质量评分: {result.balance_quality:.4f}")
        print(f"  估计色温: {result.color_temperature:.0f}K")
        print(f"  处理时间: {result.processing_time*1000:.2f}ms")
        print(f"  增益系数: B={result.gain_factors[0]:.3f}, "
              f"G={result.gain_factors[1]:.3f}, R={result.gain_factors[2]:.3f}")

    print("基础演示完成！")


def demo_method_comparison():
    """演示方法比较功能"""
    print("=== 白平衡方法比较演示 ===")

    awb = AutomaticWhiteBalancer()

    # 创建不同偏色的测试图像
    test_images = create_color_cast_images()

    for cast_type, image in test_images.items():
        print(f"\n测试 {cast_type} 图像:")

        # 比较不同方法
        results = awb.compare_methods(image)

        # 显示最佳方法
        best_method = max(results.keys(), key=lambda k: results[k].balance_quality)
        print(f"最佳方法: {best_method} (质量评分: {results[best_method].balance_quality:.4f})")

    print("方法比较演示完成！")


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
            r = int(100 + 50 * (x / width))
            g = int(120 + 60 * (y / height))
            b = int(80 + 40 * ((x + y) / (width + height)))
            image[y, x] = [b, g, r]

    # 添加一些几何形状
    cv2.rectangle(image, (50, 50), (150, 150), (200, 200, 200), -1)  # 白色矩形
    cv2.circle(image, (250, 100), 40, (100, 100, 100), -1)          # 灰色圆
    cv2.rectangle(image, (100, 200), (300, 250), (180, 180, 180), -1)  # 浅灰矩形

    return image


def create_color_cast_images() -> Dict[str, np.ndarray]:
    """
    创建具有不同偏色的测试图像

    Returns:
        偏色图像字典
    """
    base_image = create_test_image()
    images = {}

    # 原图
    images['normal'] = base_image

    # 偏蓝图像（高色温）
    blue_cast = base_image.copy()
    blue_cast[:, :, 0] = np.clip(blue_cast[:, :, 0] * 1.3, 0, 255)  # 增强蓝色
    blue_cast[:, :, 2] = np.clip(blue_cast[:, :, 2] * 0.8, 0, 255)  # 减弱红色
    images['blue_cast'] = blue_cast

    # 偏红图像（低色温）
    red_cast = base_image.copy()
    red_cast[:, :, 2] = np.clip(red_cast[:, :, 2] * 1.3, 0, 255)   # 增强红色
    red_cast[:, :, 0] = np.clip(red_cast[:, :, 0] * 0.8, 0, 255)   # 减弱蓝色
    images['red_cast'] = red_cast

    # 偏绿图像
    green_cast = base_image.copy()
    green_cast[:, :, 1] = np.clip(green_cast[:, :, 1] * 1.2, 0, 255)  # 增强绿色
    images['green_cast'] = green_cast

    return images


def performance_benchmark():
    """性能基准测试"""
    print("=== 性能基准测试 ===")

    # 测试不同图像尺寸
    test_sizes = [(200, 300), (400, 600), (800, 1200)]
    methods = ['gray_world_balance', 'white_patch_balance', 'perfect_reflector_balance', 'adaptive_white_balance']

    awb = AutomaticWhiteBalancer()

    for size in test_sizes:
        print(f"\n测试图像尺寸: {size[0]}x{size[1]}")
        test_image = np.random.randint(0, 256, (size[0], size[1], 3), dtype=np.uint8)

        for method_name in methods:
            method = getattr(awb, method_name)

            # 预热
            _ = method(test_image)

            # 计时测试
            start_time = time.time()
            num_iterations = 5

            for _ in range(num_iterations):
                result = method(test_image)

            elapsed_time = time.time() - start_time
            avg_time = elapsed_time / num_iterations

            print(f"  {method_name}: {avg_time*1000:.2f}ms, "
                 f"质量评分: {result.balance_quality:.3f}")


if __name__ == "__main__":
    # 运行演示
    print("启动自动白平衡演示程序\n")

    # 基础功能演示
    demo_basic_awb()
    print()

    # 方法比较演示
    demo_method_comparison()
    print()

    # 性能基准测试
    performance_benchmark()

    print("\n所有演示完成！")