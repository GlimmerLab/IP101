"""
偏色检测算法 - Python实现
Color Cast Detection Algorithm

基于色彩统计分析的图像质量评估技术，
支持灰度世界假设和白点检测两种检测策略。

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
class ColorStatistics:
    """色彩统计信息数据结构"""
    mean_rgb: np.ndarray             # RGB均值
    std_rgb: np.ndarray              # RGB标准差
    max_rgb: np.ndarray              # RGB最大值
    min_rgb: np.ndarray              # RGB最小值
    brightness_avg: float            # 平均亮度
    contrast_measure: float          # 对比度度量


@dataclass
class ColorCastResult:
    """偏色检测结果数据结构"""
    has_color_cast: bool             # 是否存在偏色
    cast_vector: np.ndarray          # 偏色向量
    cast_magnitude: float            # 偏色程度
    dominant_color: str              # 主要偏色方向
    confidence: float                # 检测置信度
    correction_vector: np.ndarray    # 建议校正向量


@dataclass
class ColorCastParams:
    """偏色检测参数配置"""
    gray_world_threshold: float = 0.05      # 灰度世界偏色阈值
    white_point_threshold: float = 0.8      # 白点亮度阈值
    cast_score_threshold: float = 0.08      # 综合偏色评分阈值
    correction_strength: float = 0.5        # 校正强度
    min_pixels_for_analysis: int = 1000     # 分析所需最少像素数


class ColorCastDetector:
    """偏色检测处理类"""

    def __init__(self, params: Optional[ColorCastParams] = None):
        """
        初始化偏色检测器

        Args:
            params: 检测参数配置
        """
        self.params = params or ColorCastParams()
        logger.info(f"初始化偏色检测器，灰度世界阈值: {self.params.gray_world_threshold}")

    def compute_color_statistics(self, image: np.ndarray) -> ColorStatistics:
        """
        计算图像的色彩统计特征

        Args:
            image: 输入图像

        Returns:
            色彩统计信息
        """
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")

        if len(image.shape) == 3 and image.shape[2] == 3:
            # 转换为浮点数格式
            float_image = image.astype(np.float64) / 255.0
        else:
            raise ValueError("输入图像必须是三通道彩色图像")

        # 分离RGB通道
        channels = cv2.split(float_image)

        # 计算各通道统计量
        mean_rgb = np.array([np.mean(ch) for ch in channels])
        std_rgb = np.array([np.std(ch) for ch in channels])
        max_rgb = np.array([np.max(ch) for ch in channels])
        min_rgb = np.array([np.min(ch) for ch in channels])

        # 计算整体亮度（使用感知加权）
        gray = cv2.cvtColor(float_image.astype(np.float32), cv2.COLOR_BGR2GRAY)
        brightness_avg = np.mean(gray)

        # 计算对比度度量
        contrast_measure = np.std(gray) / (brightness_avg + 1e-6)

        return ColorStatistics(
            mean_rgb=mean_rgb,
            std_rgb=std_rgb,
            max_rgb=max_rgb,
            min_rgb=min_rgb,
            brightness_avg=brightness_avg,
            contrast_measure=contrast_measure
        )

    def detect_color_cast_gray_world(self, image: np.ndarray) -> ColorCastResult:
        """
        使用灰度世界假设检测偏色

        Args:
            image: 输入图像

        Returns:
            偏色检测结果
        """
        # 计算色彩统计
        stats = self.compute_color_statistics(image)

        # 计算全局均值作为基准
        global_mean = np.mean(stats.mean_rgb)

        # 计算偏色向量
        cast_vector = stats.mean_rgb - global_mean

        # 计算偏色程度
        cast_magnitude = np.linalg.norm(cast_vector)

        # 判断是否存在显著偏色
        has_color_cast = cast_magnitude > self.params.gray_world_threshold

        # 确定主要偏色方向
        dominant_color = "色彩平衡"
        if has_color_cast:
            max_channel = np.argmax(np.abs(cast_vector))
            color_names = ["蓝色", "绿色", "红色"]
            direction = "偏" if cast_vector[max_channel] > 0 else "缺"
            dominant_color = direction + color_names[max_channel]

        # 计算检测置信度
        confidence = min(1.0, cast_magnitude / 0.2)  # 归一化到[0,1]

        # 计算校正向量
        correction_vector = -cast_vector * self.params.correction_strength

        return ColorCastResult(
            has_color_cast=has_color_cast,
            cast_vector=cast_vector,
            cast_magnitude=cast_magnitude,
            dominant_color=dominant_color,
            confidence=confidence,
            correction_vector=correction_vector
        )

    def detect_color_cast_white_point(self, image: np.ndarray) -> ColorCastResult:
        """
        使用白点检测法检测偏色

        Args:
            image: 输入图像

        Returns:
            偏色检测结果
        """
        # 转换为浮点数格式
        float_image = image.astype(np.float64) / 255.0

        # 计算亮度
        gray = cv2.cvtColor(float_image.astype(np.float32), cv2.COLOR_BGR2GRAY)

        # 寻找最亮像素
        max_brightness = np.max(gray)
        max_locations = np.where(gray == max_brightness)

        # 检查白点的可靠性
        reliable_white_point = max_brightness >= self.params.white_point_threshold

        if reliable_white_point and len(max_locations[0]) > 0:
            # 获取白点的色彩值（取第一个最亮点）
            y, x = max_locations[0][0], max_locations[1][0]
            white_point_color = float_image[y, x]

            # 理想白点应该是(1.0, 1.0, 1.0)
            ideal_white = np.array([1.0, 1.0, 1.0])

            # 计算偏色向量
            cast_vector = white_point_color - ideal_white

            # 计算偏色程度
            cast_magnitude = np.linalg.norm(cast_vector)

            # 归一化偏色程度
            white_brightness = np.linalg.norm(white_point_color)
            if white_brightness > 0:
                cast_magnitude /= white_brightness

            # 判断是否存在偏色
            has_color_cast = cast_magnitude > 0.1

            # 确定主要偏色方向
            dominant_color = "色彩平衡"
            if has_color_cast:
                max_channel = np.argmax(np.abs(cast_vector))
                color_names = ["蓝色", "绿色", "红色"]
                direction = "偏" if cast_vector[max_channel] > 0 else "缺"
                dominant_color = direction + color_names[max_channel]

            confidence = 0.8 if reliable_white_point else 0.3

        else:
            # 没有可靠白点
            cast_vector = np.array([0.0, 0.0, 0.0])
            cast_magnitude = 0.0
            has_color_cast = False
            dominant_color = "无法确定"
            confidence = 0.0

        # 计算校正向量
        correction_vector = -cast_vector * self.params.correction_strength

        return ColorCastResult(
            has_color_cast=has_color_cast,
            cast_vector=cast_vector,
            cast_magnitude=cast_magnitude,
            dominant_color=dominant_color,
            confidence=confidence,
            correction_vector=correction_vector
        )

    def detect_color_cast_comprehensive(self, image: np.ndarray) -> Dict[str, Any]:
        """
        综合多种方法的偏色检测

        Args:
            image: 输入图像

        Returns:
            综合检测结果字典
        """
        # 分别使用两种方法
        gray_world_result = self.detect_color_cast_gray_world(image)
        white_point_result = self.detect_color_cast_white_point(image)

        # 融合两种方法的结果
        gray_world_weight = 0.6
        white_point_weight = 0.4

        # 如果白点不可靠，增加灰度世界的权重
        if white_point_result.confidence < 0.5:
            gray_world_weight = 0.8
            white_point_weight = 0.2

        # 计算综合偏色程度
        final_cast_score = (gray_world_weight * gray_world_result.cast_magnitude +
                           white_point_weight * white_point_result.cast_magnitude)

        # 生成综合评估
        if final_cast_score < 0.03:
            cast_assessment = "色彩平衡良好"
        elif final_cast_score < 0.08:
            cast_assessment = "轻微偏色"
        elif final_cast_score < 0.15:
            cast_assessment = "中等偏色"
        else:
            cast_assessment = "严重偏色"

        # 确定主要偏色方向（优先考虑置信度高的结果）
        if gray_world_result.confidence > white_point_result.confidence:
            dominant_color = gray_world_result.dominant_color
        else:
            dominant_color = white_point_result.dominant_color

        # 计算综合校正向量
        correction_vector = (gray_world_weight * gray_world_result.correction_vector +
                           white_point_weight * white_point_result.correction_vector)

        return {
            'gray_world_result': gray_world_result,
            'white_point_result': white_point_result,
            'final_cast_score': final_cast_score,
            'cast_assessment': cast_assessment,
            'dominant_color': dominant_color,
            'has_color_cast': final_cast_score > self.params.cast_score_threshold,
            'correction_vector': correction_vector,
            'confidence': max(gray_world_result.confidence, white_point_result.confidence)
        }

    def apply_color_correction(self, image: np.ndarray,
                             correction_vector: np.ndarray) -> np.ndarray:
        """
        应用色彩校正

        Args:
            image: 输入图像
            correction_vector: 校正向量

        Returns:
            校正后的图像
        """
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")

        # 转换为浮点数格式
        float_image = image.astype(np.float32) / 255.0

        # 分离通道
        channels = cv2.split(float_image)

        # 应用校正
        for i, channel in enumerate(channels):
            channels[i] = channel + correction_vector[i]
            # 限制到有效范围[0, 1]
            channels[i] = np.clip(channels[i], 0.0, 1.0)

        # 合并通道并转换回8位
        corrected_image = cv2.merge(channels)
        return (corrected_image * 255).astype(np.uint8)

    def analyze_image_regions(self, image: np.ndarray,
                            region_size: int = 64) -> Dict[str, Any]:
        """
        分析图像的区域偏色情况

        Args:
            image: 输入图像
            region_size: 分析区域尺寸

        Returns:
            区域分析结果
        """
        h, w = image.shape[:2]
        regions = []

        # 分块分析
        for y in range(0, h - region_size, region_size):
            for x in range(0, w - region_size, region_size):
                # 提取区域
                region = image[y:y+region_size, x:x+region_size]

                # 检测该区域的偏色
                result = self.detect_color_cast_gray_world(region)

                regions.append({
                    'position': (x, y),
                    'cast_magnitude': result.cast_magnitude,
                    'dominant_color': result.dominant_color,
                    'has_cast': result.has_color_cast
                })

        # 统计区域偏色情况
        cast_regions = [r for r in regions if r['has_cast']]
        cast_percentage = len(cast_regions) / len(regions) * 100

        # 计算平均偏色程度
        avg_cast_magnitude = np.mean([r['cast_magnitude'] for r in regions])

        return {
            'total_regions': len(regions),
            'cast_regions': len(cast_regions),
            'cast_percentage': cast_percentage,
            'avg_cast_magnitude': avg_cast_magnitude,
            'region_details': regions
        }

    def generate_cast_heatmap(self, image: np.ndarray,
                            region_size: int = 32) -> np.ndarray:
        """
        生成偏色热力图

        Args:
            image: 输入图像
            region_size: 分析区域尺寸

        Returns:
            偏色热力图
        """
        h, w = image.shape[:2]
        heatmap = np.zeros((h // region_size, w // region_size), dtype=np.float32)

        # 逐块分析
        for i, y in enumerate(range(0, h - region_size, region_size)):
            for j, x in enumerate(range(0, w - region_size, region_size)):
                region = image[y:y+region_size, x:x+region_size]
                result = self.detect_color_cast_gray_world(region)
                heatmap[i, j] = result.cast_magnitude

        # 调整热力图尺寸到原图尺寸
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # 应用颜色映射
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        return heatmap_colored

    def visualize_results(self, image: np.ndarray,
                         result: Dict[str, Any]) -> np.ndarray:
        """
        可视化偏色检测结果

        Args:
            image: 原始图像
            result: 检测结果

        Returns:
            可视化图像
        """
        # 创建结果图像
        result_image = image.copy()
        h, w = result_image.shape[:2]

        # 在图像上添加文本信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 255, 0) if not result['has_color_cast'] else (0, 0, 255)
        thickness = 2

        # 显示检测结果
        cv2.putText(result_image, f"Assessment: {result['cast_assessment']}",
                   (10, 30), font, font_scale, color, thickness)

        cv2.putText(result_image, f"Cast Score: {result['final_cast_score']:.3f}",
                   (10, 60), font, font_scale, color, thickness)

        cv2.putText(result_image, f"Dominant: {result['dominant_color']}",
                   (10, 90), font, font_scale, color, thickness)

        # 显示校正向量信息
        correction = result['correction_vector']
        cv2.putText(result_image, f"Correction: [{correction[0]:.2f}, {correction[1]:.2f}, {correction[2]:.2f}]",
                   (10, 120), font, font_scale-0.1, (255, 255, 0), 1)

        return result_image


def demo_basic_detection():
    """演示基础偏色检测功能"""
    print("=== 偏色检测基础演示 ===")

    # 创建检测器
    detector = ColorCastDetector()

    # 创建测试图像
    test_images = create_test_images()

    for name, image in test_images.items():
        print(f"\n分析测试图像: {name}")
        print(f"图像尺寸: {image.shape}")

        # 灰度世界检测
        gray_result = detector.detect_color_cast_gray_world(image)
        print(f"灰度世界检测: {'偏色' if gray_result.has_color_cast else '平衡'}")
        print(f"  偏色程度: {gray_result.cast_magnitude:.4f}")
        print(f"  主要方向: {gray_result.dominant_color}")

        # 白点检测
        white_result = detector.detect_color_cast_white_point(image)
        print(f"白点检测: {'偏色' if white_result.has_color_cast else '平衡'}")
        print(f"  偏色程度: {white_result.cast_magnitude:.4f}")
        print(f"  置信度: {white_result.confidence:.4f}")

        # 综合检测
        comprehensive_result = detector.detect_color_cast_comprehensive(image)
        print(f"综合评估: {comprehensive_result['cast_assessment']}")
        print(f"  综合评分: {comprehensive_result['final_cast_score']:.4f}")

    print("基础演示完成！")


def demo_color_correction():
    """演示色彩校正功能"""
    print("=== 色彩校正演示 ===")

    detector = ColorCastDetector()
    test_images = create_test_images()

    for name, image in test_images.items():
        if "cast" not in name.lower():  # 跳过平衡图像
            continue

        print(f"\n校正测试图像: {name}")

        # 检测偏色
        result = detector.detect_color_cast_comprehensive(image)

        if result['has_color_cast']:
            print(f"检测到偏色: {result['cast_assessment']}")
            print(f"校正向量: {result['correction_vector']}")

            # 应用校正
            corrected_image = detector.apply_color_correction(
                image, result['correction_vector']
            )

            # 重新检测校正后的图像
            corrected_result = detector.detect_color_cast_comprehensive(corrected_image)
            print(f"校正后评估: {corrected_result['cast_assessment']}")
            print(f"校正前评分: {result['final_cast_score']:.4f}")
            print(f"校正后评分: {corrected_result['final_cast_score']:.4f}")

            improvement = result['final_cast_score'] - corrected_result['final_cast_score']
            print(f"改善程度: {improvement:.4f}")
        else:
            print("图像色彩平衡，无需校正")

    print("色彩校正演示完成！")


def demo_parameter_optimization():
    """演示参数优化功能"""
    print("=== 参数优化演示 ===")

    # 不同参数配置
    configs = [
        ColorCastParams(gray_world_threshold=0.03, cast_score_threshold=0.05),
        ColorCastParams(gray_world_threshold=0.05, cast_score_threshold=0.08),
        ColorCastParams(gray_world_threshold=0.08, cast_score_threshold=0.12),
    ]

    test_images = create_test_images()
    test_image = test_images['blue_cast']

    results = {}
    for i, params in enumerate(configs):
        print(f"\n配置 {i+1}:")
        print(f"  灰度世界阈值: {params.gray_world_threshold}")
        print(f"  综合评分阈值: {params.cast_score_threshold}")

        detector = ColorCastDetector(params)
        result = detector.detect_color_cast_comprehensive(test_image)

        print(f"  检测结果: {result['cast_assessment']}")
        print(f"  综合评分: {result['final_cast_score']:.4f}")
        print(f"  是否偏色: {'是' if result['has_color_cast'] else '否'}")

        results[f'config_{i+1}'] = result

    print("参数优化演示完成！")
    return results


def create_test_images() -> Dict[str, np.ndarray]:
    """
    创建测试图像集

    Returns:
        测试图像字典
    """
    images = {}

    # 创建平衡图像
    balanced_image = np.ones((300, 400, 3), dtype=np.uint8) * 128
    # 添加一些随机变化
    noise = np.random.normal(0, 20, balanced_image.shape).astype(np.int16)
    balanced_image = np.clip(balanced_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    images['balanced'] = balanced_image

    # 创建偏蓝图像
    blue_cast_image = balanced_image.copy()
    blue_cast_image[:, :, 0] = np.clip(blue_cast_image[:, :, 0] + 30, 0, 255)  # 增加蓝色
    images['blue_cast'] = blue_cast_image

    # 创建偏红图像
    red_cast_image = balanced_image.copy()
    red_cast_image[:, :, 2] = np.clip(red_cast_image[:, :, 2] + 25, 0, 255)  # 增加红色
    images['red_cast'] = red_cast_image

    # 创建偏绿图像
    green_cast_image = balanced_image.copy()
    green_cast_image[:, :, 1] = np.clip(green_cast_image[:, :, 1] + 35, 0, 255)  # 增加绿色
    images['green_cast'] = green_cast_image

    # 创建复杂场景图像
    complex_image = np.zeros((300, 400, 3), dtype=np.uint8)

    # 添加不同色彩的区域
    complex_image[50:150, 50:150] = [200, 100, 100]  # 红色区域
    complex_image[150:250, 150:250] = [100, 200, 100]  # 绿色区域
    complex_image[100:200, 250:350] = [100, 100, 200]  # 蓝色区域
    complex_image[200:280, 50:130] = [200, 200, 200]  # 白色区域
    complex_image[20:80, 300:380] = [50, 50, 50]    # 深色区域

    # 添加噪声
    noise = np.random.normal(0, 10, complex_image.shape).astype(np.int16)
    complex_image = np.clip(complex_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    images['complex_scene'] = complex_image

    return images


def performance_benchmark():
    """性能基准测试"""
    print("=== 性能基准测试 ===")

    # 测试不同图像尺寸
    test_sizes = [(200, 300), (400, 600), (800, 1200)]
    methods = ['gray_world', 'white_point', 'comprehensive']

    detector = ColorCastDetector()

    for size in test_sizes:
        print(f"\n测试图像尺寸: {size[0]}x{size[1]}")
        test_image = np.random.randint(0, 256, (size[0], size[1], 3), dtype=np.uint8)

        for method in methods:
            # 预热
            if method == 'gray_world':
                _ = detector.detect_color_cast_gray_world(test_image)
            elif method == 'white_point':
                _ = detector.detect_color_cast_white_point(test_image)
            else:
                _ = detector.detect_color_cast_comprehensive(test_image)

            # 计时测试
            start_time = time.time()
            num_iterations = 10

            for _ in range(num_iterations):
                if method == 'gray_world':
                    _ = detector.detect_color_cast_gray_world(test_image)
                elif method == 'white_point':
                    _ = detector.detect_color_cast_white_point(test_image)
                else:
                    _ = detector.detect_color_cast_comprehensive(test_image)

            elapsed_time = time.time() - start_time
            avg_time = elapsed_time / num_iterations

            print(f"  {method} 方法: {avg_time*1000:.2f}ms")


if __name__ == "__main__":
    # 运行演示
    print("启动偏色检测演示程序\n")

    # 基础功能演示
    demo_basic_detection()
    print()

    # 色彩校正演示
    demo_color_correction()
    print()

    # 参数优化演示
    param_results = demo_parameter_optimization()
    print()

    # 性能基准测试
    performance_benchmark()

    print("\n所有演示完成！")