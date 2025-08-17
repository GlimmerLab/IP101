"""
侧窗口滤波算法 - Python实现
Side Window Filter Algorithm

基于自适应窗口选择的边缘保持滤波技术，
通过动态选择最优局部窗口实现精确的图像滤波。

Author: GlimmerLab
Date: 2024
"""

import cv2
import numpy as np
import logging
import time
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FilterType(Enum):
    """滤波类型枚举"""
    BOX = "box"           # 盒式滤波
    MEDIAN = "median"     # 中值滤波


class WindowDirection(Enum):
    """窗口方向枚举"""
    LEFT = "left"         # 左侧窗口
    RIGHT = "right"       # 右侧窗口
    TOP = "top"           # 上侧窗口
    BOTTOM = "bottom"     # 下侧窗口
    CENTER = "center"     # 中心窗口


@dataclass
class SideWindowParams:
    """侧窗口滤波参数配置"""
    radius: int = 3                      # 窗口半径
    filter_type: FilterType = FilterType.BOX  # 滤波类型
    edge_threshold: float = 10.0         # 边缘检测阈值
    enable_adaptive: bool = True         # 是否启用自适应权重


@dataclass
class SideWindow:
    """侧窗口数据结构"""
    positions: List[Tuple[int, int]]     # 窗口内像素位置
    direction: WindowDirection           # 窗口方向
    quality: float = 0.0                 # 窗口质量（方差）
    weight: float = 1.0                  # 窗口权重


class SideWindowFilter:
    """侧窗口滤波处理类"""

    def __init__(self, params: Optional[SideWindowParams] = None):
        """
        初始化侧窗口滤波器

        Args:
            params: 滤波参数配置
        """
        self.params = params or SideWindowParams()
        logger.info(f"初始化侧窗口滤波器，半径: {self.params.radius}, "
                   f"滤波类型: {self.params.filter_type.value}")

    def side_window_filter(self, image: np.ndarray,
                          params: Optional[SideWindowParams] = None) -> np.ndarray:
        """
        执行侧窗口滤波

        Args:
            image: 输入图像
            params: 可选的参数覆盖

        Returns:
            滤波后的图像

        Raises:
            ValueError: 输入图像无效时抛出
        """
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")

        p = params or self.params

        if len(image.shape) == 2:
            return self._process_grayscale(image, p)
        elif len(image.shape) == 3:
            return self._process_color(image, p)
        else:
            raise ValueError("不支持的图像格式，仅支持灰度或彩色图像")

    def _process_grayscale(self, image: np.ndarray, params: SideWindowParams) -> np.ndarray:
        """
        处理灰度图像

        Args:
            image: 输入灰度图像
            params: 滤波参数

        Returns:
            滤波后的图像
        """
        h, w = image.shape
        result = np.zeros_like(image, dtype=np.float32)

        # 遍历每个像素
        for y in range(h):
            for x in range(w):
                center = (x, y)

                # 选择最优窗口
                optimal_window = self._select_optimal_window(image, center, params)

                # 计算滤波结果
                if params.filter_type == FilterType.BOX:
                    filtered_value = self._compute_box_filter(image, optimal_window, center, params)
                else:
                    filtered_value = self._compute_median_filter(image, optimal_window, center, params)

                result[y, x] = filtered_value

        return np.clip(result, 0, 255).astype(np.uint8)

    def _process_color(self, image: np.ndarray, params: SideWindowParams) -> np.ndarray:
        """
        处理彩色图像

        Args:
            image: 输入彩色图像
            params: 滤波参数

        Returns:
            滤波后的图像
        """
        h, w, c = image.shape
        result = np.zeros_like(image, dtype=np.float32)

        # 分别处理每个通道
        for ch in range(c):
            channel_result = self._process_grayscale(image[:, :, ch], params)
            result[:, :, ch] = channel_result

        return np.clip(result, 0, 255).astype(np.uint8)

    def _create_candidate_windows(self, center: Tuple[int, int],
                                 radius: int) -> List[SideWindow]:
        """
        创建候选侧窗口

        Args:
            center: 中心点坐标 (x, y)
            radius: 窗口半径

        Returns:
            候选窗口列表
        """
        x, y = center
        r = radius
        windows = []

        # 左侧窗口
        left_positions = []
        for dy in range(-r, r + 1):
            for dx in range(-r, 1):
                left_positions.append((x + dx, y + dy))
        windows.append(SideWindow(left_positions, WindowDirection.LEFT))

        # 右侧窗口
        right_positions = []
        for dy in range(-r, r + 1):
            for dx in range(0, r + 1):
                right_positions.append((x + dx, y + dy))
        windows.append(SideWindow(right_positions, WindowDirection.RIGHT))

        # 上侧窗口
        top_positions = []
        for dy in range(-r, 1):
            for dx in range(-r, r + 1):
                top_positions.append((x + dx, y + dy))
        windows.append(SideWindow(top_positions, WindowDirection.TOP))

        # 下侧窗口
        bottom_positions = []
        for dy in range(0, r + 1):
            for dx in range(-r, r + 1):
                bottom_positions.append((x + dx, y + dy))
        windows.append(SideWindow(bottom_positions, WindowDirection.BOTTOM))

        # 中心窗口
        center_positions = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                center_positions.append((x + dx, y + dy))
        windows.append(SideWindow(center_positions, WindowDirection.CENTER))

        return windows

    def _evaluate_window_quality(self, image: np.ndarray, window: SideWindow) -> float:
        """
        评估窗口质量

        Args:
            image: 输入图像
            window: 待评估的窗口

        Returns:
            窗口质量分数（方差，越小越好）
        """
        if not window.positions:
            return float('inf')

        values = []
        h, w = image.shape[:2]

        # 收集窗口内的有效像素值
        for px, py in window.positions:
            if 0 <= px < w and 0 <= py < h:
                if len(image.shape) == 2:
                    values.append(float(image[py, px]))
                else:
                    values.append(float(np.mean(image[py, px])))

        if not values:
            return float('inf')

        # 计算方差
        variance = np.var(values)
        return variance

    def _select_optimal_window(self, image: np.ndarray, center: Tuple[int, int],
                              params: SideWindowParams) -> SideWindow:
        """
        选择最优窗口

        Args:
            image: 输入图像
            center: 中心点坐标
            params: 滤波参数

        Returns:
            最优的侧窗口
        """
        candidates = self._create_candidate_windows(center, params.radius)

        best_quality = float('inf')
        optimal_window = candidates[0]  # 默认选择第一个

        # 评估每个候选窗口
        for window in candidates:
            quality = self._evaluate_window_quality(image, window)
            window.quality = quality

            # 记录质量最高（方差最小）的窗口
            if quality < best_quality:
                best_quality = quality
                optimal_window = window

        # 计算权重
        optimal_window.weight = 1.0 / (1.0 + best_quality)

        return optimal_window

    def _compute_adaptive_weights(self, image: np.ndarray, window: SideWindow,
                                 center: Tuple[int, int]) -> np.ndarray:
        """
        计算自适应权重

        Args:
            image: 输入图像
            window: 选定的窗口
            center: 中心点坐标

        Returns:
            权重数组
        """
        if not window.positions:
            return np.array([])

        cx, cy = center
        h, w = image.shape[:2]

        # 获取中心像素值作为参考
        if len(image.shape) == 2:
            center_value = float(image[cy, cx])
        else:
            center_value = float(np.mean(image[cy, cx]))

        weights = []

        for px, py in window.positions:
            if 0 <= px < w and 0 <= py < h:
                # 计算空间距离
                spatial_dist = np.sqrt((px - cx)**2 + (py - cy)**2)

                # 计算强度差异
                if len(image.shape) == 2:
                    pixel_value = float(image[py, px])
                else:
                    pixel_value = float(np.mean(image[py, px]))

                intensity_diff = abs(pixel_value - center_value)

                # 综合空间和强度权重
                spatial_weight = np.exp(-spatial_dist**2 / (2.0 * 1.0**2))
                intensity_weight = np.exp(-intensity_diff**2 / (2.0 * 15.0**2))

                final_weight = spatial_weight * intensity_weight
                weights.append(final_weight)
            else:
                weights.append(0.0)

        return np.array(weights)

    def _compute_box_filter(self, image: np.ndarray, window: SideWindow,
                           center: Tuple[int, int], params: SideWindowParams) -> float:
        """
        计算盒式滤波结果

        Args:
            image: 输入图像
            window: 选定的窗口
            center: 中心点坐标
            params: 滤波参数

        Returns:
            滤波后的像素值
        """
        if not window.positions:
            return 0.0

        values = []
        h, w = image.shape[:2]

        # 收集有效的像素值
        for px, py in window.positions:
            if 0 <= px < w and 0 <= py < h:
                if len(image.shape) == 2:
                    values.append(float(image[py, px]))
                else:
                    values.append(float(np.mean(image[py, px])))

        if not values:
            return 0.0

        if params.enable_adaptive:
            # 使用自适应权重
            weights = self._compute_adaptive_weights(image, window, center)
            if len(weights) > 0 and np.sum(weights) > 0:
                values = np.array(values)
                valid_weights = weights[:len(values)]
                return np.sum(values * valid_weights) / np.sum(valid_weights)

        # 简单平均
        return np.mean(values)

    def _compute_median_filter(self, image: np.ndarray, window: SideWindow,
                              center: Tuple[int, int], params: SideWindowParams) -> float:
        """
        计算中值滤波结果

        Args:
            image: 输入图像
            window: 选定的窗口
            center: 中心点坐标
            params: 滤波参数

        Returns:
            滤波后的像素值
        """
        if not window.positions:
            return 0.0

        values = []
        h, w = image.shape[:2]

        # 收集有效的像素值
        for px, py in window.positions:
            if 0 <= px < w and 0 <= py < h:
                if len(image.shape) == 2:
                    values.append(float(image[py, px]))
                else:
                    values.append(float(np.mean(image[py, px])))

        if not values:
            return 0.0

        if params.enable_adaptive:
            # 加权中值（近似）
            weights = self._compute_adaptive_weights(image, window, center)
            if len(weights) > 0:
                # 根据权重重复采样，然后取中值
                weighted_samples = []
                for val, weight in zip(values, weights[:len(values)]):
                    count = max(1, int(weight * 100))
                    weighted_samples.extend([val] * count)
                return np.median(weighted_samples)

        # 简单中值
        return np.median(values)

    def analyze_window_selection(self, image: np.ndarray,
                                sample_points: Optional[List[Tuple[int, int]]] = None) -> Dict:
        """
        分析窗口选择情况

        Args:
            image: 输入图像
            sample_points: 采样点列表，如果为None则随机选择

        Returns:
            分析结果字典
        """
        if sample_points is None:
            h, w = image.shape[:2]
            # 选择一些有代表性的采样点
            sample_points = [
                (w//4, h//4), (3*w//4, h//4),
                (w//4, 3*h//4), (3*w//4, 3*h//4),
                (w//2, h//2)
            ]

        analysis = {
            'window_choices': {},
            'quality_scores': {},
            'direction_distribution': {dir.value: 0 for dir in WindowDirection}
        }

        for i, (x, y) in enumerate(sample_points):
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                optimal_window = self._select_optimal_window(image, (x, y), self.params)

                analysis['window_choices'][f'point_{i}'] = {
                    'position': (x, y),
                    'direction': optimal_window.direction.value,
                    'quality': optimal_window.quality,
                    'weight': optimal_window.weight
                }

                analysis['direction_distribution'][optimal_window.direction.value] += 1

        return analysis

    def fast_side_window_filter(self, image: np.ndarray,
                               subsample_ratio: int = 2) -> np.ndarray:
        """
        快速侧窗口滤波（通过下采样加速）

        Args:
            image: 输入图像
            subsample_ratio: 下采样比例

        Returns:
            滤波后的图像
        """
        h, w = image.shape[:2]

        # 下采样
        small_image = cv2.resize(image, (w//subsample_ratio, h//subsample_ratio))

        # 调整参数
        fast_params = SideWindowParams(
            radius=max(1, self.params.radius // subsample_ratio),
            filter_type=self.params.filter_type,
            edge_threshold=self.params.edge_threshold,
            enable_adaptive=False  # 快速模式禁用自适应权重
        )

        # 在小尺寸图像上处理
        small_result = self.side_window_filter(small_image, fast_params)

        # 上采样回原始尺寸
        result = cv2.resize(small_result, (w, h), interpolation=cv2.INTER_LINEAR)

        return result


def demo_basic_filtering():
    """演示基础侧窗口滤波功能"""
    print("=== 侧窗口滤波基础演示 ===")

    # 创建滤波器
    swf = SideWindowFilter()

    # 创建测试图像
    test_image = create_test_image()

    print(f"原始图像形状: {test_image.shape}")

    # 盒式滤波
    print("执行盒式侧窗口滤波...")
    box_params = SideWindowParams(radius=3, filter_type=FilterType.BOX)
    box_filtered = swf.side_window_filter(test_image, box_params)

    # 中值滤波
    print("执行中值侧窗口滤波...")
    median_params = SideWindowParams(radius=3, filter_type=FilterType.MEDIAN)
    median_filtered = swf.side_window_filter(test_image, median_params)

    # 快速滤波
    print("执行快速侧窗口滤波...")
    fast_filtered = swf.fast_side_window_filter(test_image, subsample_ratio=2)

    print("演示完成！")

    return {
        'original': test_image,
        'box_filtered': box_filtered,
        'median_filtered': median_filtered,
        'fast_filtered': fast_filtered
    }


def demo_window_analysis():
    """演示窗口选择分析功能"""
    print("=== 窗口选择分析演示 ===")

    swf = SideWindowFilter()

    # 创建包含边缘的测试图像
    edge_image = create_edge_test_image()

    print("分析窗口选择情况...")
    analysis = swf.analyze_window_selection(edge_image)

    print("窗口选择分析结果:")
    for point_id, info in analysis['window_choices'].items():
        print(f"  {point_id}: 位置{info['position']}, "
              f"方向={info['direction']}, 质量={info['quality']:.2f}")

    print("\n方向分布统计:")
    for direction, count in analysis['direction_distribution'].items():
        print(f"  {direction}: {count} 次")

    print("分析演示完成！")

    return analysis


def create_test_image() -> np.ndarray:
    """创建测试图像"""
    height, width = 300, 400
    image = np.zeros((height, width), dtype=np.uint8)

    # 创建背景
    image.fill(128)

    # 添加几何形状
    cv2.rectangle(image, (50, 50), (150, 150), 255, -1)
    cv2.circle(image, (300, 100), 40, 0, -1)
    cv2.ellipse(image, (200, 220), (60, 30), 45, 0, 360, 200, -1)

    # 添加噪声
    noise = np.random.normal(0, 15, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return image


def create_edge_test_image() -> np.ndarray:
    """创建包含明显边缘的测试图像"""
    height, width = 200, 300
    image = np.zeros((height, width), dtype=np.uint8)

    # 创建垂直边缘
    image[:, :width//3] = 50
    image[:, width//3:2*width//3] = 150
    image[:, 2*width//3:] = 200

    # 创建水平边缘
    image[:height//3, :] += 30
    image[2*height//3:, :] -= 30

    # 添加轻微噪声
    noise = np.random.normal(0, 5, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return image


def create_color_test_image() -> np.ndarray:
    """创建彩色测试图像"""
    height, width = 300, 400
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # 创建彩色区域
    image[:, :width//3] = [100, 50, 200]    # 紫色
    image[:, width//3:2*width//3] = [50, 200, 100]  # 绿色
    image[:, 2*width//3:] = [200, 100, 50]  # 橙色

    # 添加一些几何形状
    cv2.circle(image, (100, 150), 50, (255, 255, 255), -1)
    cv2.rectangle(image, (250, 100), (350, 200), (0, 0, 0), -1)

    # 添加噪声
    for c in range(3):
        noise = np.random.normal(0, 10, image.shape[:2]).astype(np.int16)
        image[:, :, c] = np.clip(image[:, :, c].astype(np.int16) + noise, 0, 255)

    return image


def performance_benchmark():
    """性能基准测试"""
    print("=== 性能基准测试 ===")

    # 测试不同参数配置
    configs = [
        SideWindowParams(radius=2, filter_type=FilterType.BOX, enable_adaptive=False),
        SideWindowParams(radius=3, filter_type=FilterType.BOX, enable_adaptive=False),
        SideWindowParams(radius=3, filter_type=FilterType.BOX, enable_adaptive=True),
        SideWindowParams(radius=3, filter_type=FilterType.MEDIAN, enable_adaptive=False),
    ]

    # 创建测试图像
    test_sizes = [(100, 150), (200, 300), (300, 400)]

    for size in test_sizes:
        print(f"\n测试图像大小: {size[0]}x{size[1]}")
        test_image = np.random.randint(0, 256, size, dtype=np.uint8)

        for i, config in enumerate(configs):
            swf = SideWindowFilter(config)

            # 预热
            _ = swf.side_window_filter(test_image)

            # 计时测试
            start_time = time.time()
            num_iterations = 3

            for _ in range(num_iterations):
                result = swf.side_window_filter(test_image)

            elapsed_time = time.time() - start_time
            avg_time = elapsed_time / num_iterations

            adaptive_str = "自适应" if config.enable_adaptive else "普通"
            print(f"  配置{i+1} ({config.filter_type.value}, "
                 f"半径={config.radius}, {adaptive_str}): {avg_time*1000:.2f}ms")


if __name__ == "__main__":
    # 运行演示
    print("启动侧窗口滤波演示程序\n")

    # 基础功能演示
    basic_results = demo_basic_filtering()
    print()

    # 窗口分析演示
    analysis_results = demo_window_analysis()
    print()

    # 性能基准测试
    performance_benchmark()

    print("\n所有演示完成！")