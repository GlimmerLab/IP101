"""
矩形检测算法 - Python实现
Rectangle Detection Algorithm

基于几何特征分析的矩形检测技术，
支持轮廓分析和霍夫变换两种检测策略。

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
class RectangleInfo:
    """矩形检测结果数据结构"""
    center: Tuple[float, float]          # 中心坐标
    size: Tuple[float, float]            # 宽度和高度
    angle: float                         # 旋转角度
    confidence: float                    # 置信度分数
    corners: List[Tuple[int, int]]       # 四个角点坐标


@dataclass
class RectDetectionParams:
    """矩形检测参数配置"""
    min_area: float = 500.0              # 最小面积阈值
    max_area: float = 50000.0            # 最大面积阈值
    min_aspect_ratio: float = 0.2        # 最小长宽比
    max_aspect_ratio: float = 10.0       # 最大长宽比
    confidence_threshold: float = 0.6    # 置信度阈值
    edge_threshold_low: float = 50.0     # Canny边缘检测低阈值
    edge_threshold_high: float = 150.0   # Canny边缘检测高阈值
    hough_threshold: int = 50            # 霍夫变换阈值
    min_line_length: float = 30.0        # 最小直线长度
    max_line_gap: float = 10.0           # 最大直线间隙


class RectangleDetector:
    """矩形检测处理类"""

    def __init__(self, params: Optional[RectDetectionParams] = None):
        """
        初始化矩形检测器

        Args:
            params: 检测参数配置
        """
        self.params = params or RectDetectionParams()
        logger.info(f"初始化矩形检测器，面积范围: [{self.params.min_area}, {self.params.max_area}]")

    def detect_rectangles(self, image: np.ndarray,
                         method: str = 'hybrid') -> List[RectangleInfo]:
        """
        执行矩形检测

        Args:
            image: 输入图像
            method: 检测方法 ('contour', 'hough', 'hybrid')

        Returns:
            检测到的矩形列表

        Raises:
            ValueError: 输入参数无效时抛出
        """
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")

        if method == 'contour':
            return self._contour_based_detection(image)
        elif method == 'hough':
            return self._hough_based_detection(image)
        elif method == 'hybrid':
            return self._hybrid_detection(image)
        else:
            raise ValueError(f"不支持的检测方法: {method}")

    def _contour_based_detection(self, image: np.ndarray) -> List[RectangleInfo]:
        """
        基于轮廓的矩形检测

        Args:
            image: 输入图像

        Returns:
            检测到的矩形列表
        """
        # 图像预处理
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # 自适应二值化
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)

        # 形态学操作去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 轮廓检测
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rectangles = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # 面积过滤
            if area < self.params.min_area or area > self.params.max_area:
                continue

            # 轮廓近似
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # 检查是否为四边形
            if len(approx) == 4:
                # 直接使用近似的四边形
                corners = approx.reshape(4, 2)
                rect_info = self._create_rectangle_from_corners(corners, area)
                if rect_info and rect_info.confidence >= self.params.confidence_threshold:
                    rectangles.append(rect_info)
            else:
                # 使用最小外接矩形
                rotated_rect = cv2.minAreaRect(contour)
                width, height = rotated_rect[1]

                if width == 0 or height == 0:
                    continue

                # 长宽比检查
                aspect_ratio = max(width, height) / min(width, height)
                if (aspect_ratio < self.params.min_aspect_ratio or
                    aspect_ratio > self.params.max_aspect_ratio):
                    continue

                # 计算置信度
                corners = cv2.boxPoints(rotated_rect)
                corners = np.int0(corners)
                confidence = self._calculate_contour_confidence(contour, corners)

                if confidence < self.params.confidence_threshold:
                    continue

                rect_info = RectangleInfo(
                    center=(rotated_rect[0][0], rotated_rect[0][1]),
                    size=(width, height),
                    angle=rotated_rect[2],
                    confidence=confidence,
                    corners=[(int(corner[0]), int(corner[1])) for corner in corners]
                )

                rectangles.append(rect_info)

        return rectangles

    def _hough_based_detection(self, image: np.ndarray) -> List[RectangleInfo]:
        """
        基于霍夫变换的矩形检测

        Args:
            image: 输入图像

        Returns:
            检测到的矩形列表
        """
        # 预处理
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # 边缘检测
        edges = cv2.Canny(gray, self.params.edge_threshold_low, self.params.edge_threshold_high)

        # 霍夫直线检测
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, self.params.hough_threshold,
                               minLineLength=self.params.min_line_length,
                               maxLineGap=self.params.max_line_gap)

        if lines is None or len(lines) < 4:
            return []

        # 直线分类和矩形构建
        horizontal_lines, vertical_lines = self._classify_lines_by_direction(lines)
        rectangles = self._combine_lines_to_rectangles(horizontal_lines, vertical_lines, image.shape)

        return rectangles

    def _hybrid_detection(self, image: np.ndarray) -> List[RectangleInfo]:
        """
        混合检测方法（轮廓+霍夫变换）

        Args:
            image: 输入图像

        Returns:
            检测到的矩形列表
        """
        # 分别使用两种方法
        contour_rects = self._contour_based_detection(image)
        hough_rects = self._hough_based_detection(image)

        # 合并结果
        all_rectangles = contour_rects + hough_rects

        if not all_rectangles:
            return []

        # 非极大值抑制
        final_rectangles = self._apply_non_maximum_suppression(all_rectangles, iou_threshold=0.3)

        # 按置信度排序
        final_rectangles.sort(key=lambda x: x.confidence, reverse=True)

        return final_rectangles

    def _create_rectangle_from_corners(self, corners: np.ndarray, area: float) -> Optional[RectangleInfo]:
        """
        从四个角点创建矩形信息

        Args:
            corners: 四个角点坐标
            area: 轮廓面积

        Returns:
            矩形信息对象，如果无效则返回None
        """
        # 计算中心点
        center_x = np.mean(corners[:, 0])
        center_y = np.mean(corners[:, 1])

        # 计算边长
        edges = []
        for i in range(4):
            edge_len = np.linalg.norm(corners[i] - corners[(i + 1) % 4])
            edges.append(edge_len)

        # 对边应该相等
        if len(set(np.round(edges[::2], 1))) > 1 or len(set(np.round(edges[1::2], 1))) > 1:
            # 边长不匹配，置信度降低
            confidence = 0.5
        else:
            confidence = 0.8

        # 计算尺寸
        width = max(edges[0], edges[2])
        height = max(edges[1], edges[3])

        # 长宽比检查
        aspect_ratio = max(width, height) / min(width, height)
        if (aspect_ratio < self.params.min_aspect_ratio or
            aspect_ratio > self.params.max_aspect_ratio):
            return None

        # 计算角度（简化处理）
        angle = 0.0

        return RectangleInfo(
            center=(center_x, center_y),
            size=(width, height),
            angle=angle,
            confidence=confidence,
            corners=[(int(c[0]), int(c[1])) for c in corners]
        )

    def _calculate_contour_confidence(self, contour: np.ndarray, corners: np.ndarray) -> float:
        """
        计算轮廓与矩形的匹配置信度

        Args:
            contour: 原始轮廓
            corners: 矩形角点

        Returns:
            置信度分数 [0, 1]
        """
        if len(corners) != 4:
            return 0.0

        # 计算轮廓点到矩形边的平均距离
        total_distance = 0.0
        for point in contour:
            min_distance = float('inf')
            for i in range(4):
                distance = self._point_to_line_distance(
                    point[0], corners[i], corners[(i + 1) % 4]
                )
                min_distance = min(min_distance, distance)
            total_distance += min_distance

        # 计算矩形周长
        perimeter = 0.0
        for i in range(4):
            perimeter += np.linalg.norm(corners[i] - corners[(i + 1) % 4])

        # 归一化距离
        avg_distance = total_distance / len(contour)
        normalized_distance = avg_distance / (perimeter / 16.0)

        # 转换为置信度
        confidence = max(0.0, 1.0 - min(1.0, normalized_distance))

        return confidence

    def _point_to_line_distance(self, point: np.ndarray,
                               line_start: np.ndarray, line_end: np.ndarray) -> float:
        """
        计算点到直线段的最短距离

        Args:
            point: 查询点
            line_start: 直线起点
            line_end: 直线终点

        Returns:
            最短距离
        """
        if np.array_equal(line_start, line_end):
            return np.linalg.norm(point - line_start)

        line_vec = line_end - line_start
        point_vec = point - line_start

        line_len_squared = np.dot(line_vec, line_vec)
        if line_len_squared == 0:
            return np.linalg.norm(point - line_start)

        # 计算投影参数
        projection = np.dot(point_vec, line_vec) / line_len_squared

        # 限制在线段范围内
        if projection < 0:
            return np.linalg.norm(point - line_start)
        elif projection > 1:
            return np.linalg.norm(point - line_end)

        # 计算投影点
        projection_point = line_start + projection * line_vec
        return np.linalg.norm(point - projection_point)

    def _classify_lines_by_direction(self, lines: np.ndarray) -> Tuple[List, List]:
        """
        按方向分类直线

        Args:
            lines: 检测到的直线数组

        Returns:
            (水平线列表, 垂直线列表)
        """
        horizontal_lines = []
        vertical_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # 计算直线角度
            if x2 == x1:
                angle = 90  # 垂直线
            else:
                angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))

            # 角度分类（考虑45度分界）
            if angle < 45 or angle > 135:
                horizontal_lines.append(line[0])
            else:
                vertical_lines.append(line[0])

        return horizontal_lines, vertical_lines

    def _combine_lines_to_rectangles(self, horizontal_lines: List,
                                   vertical_lines: List,
                                   image_shape: Tuple) -> List[RectangleInfo]:
        """
        组合直线构建矩形

        Args:
            horizontal_lines: 水平线列表
            vertical_lines: 垂直线列表
            image_shape: 图像尺寸

        Returns:
            检测到的矩形列表
        """
        rectangles = []
        h, w = image_shape[:2]

        # 尝试所有可能的直线组合
        for i, h1 in enumerate(horizontal_lines):
            for j, h2 in enumerate(horizontal_lines[i + 1:], i + 1):
                for k, v1 in enumerate(vertical_lines):
                    for l, v2 in enumerate(vertical_lines[k + 1:], k + 1):

                        # 计算两条水平线的距离
                        h_dist = abs((h1[1] + h1[3]) / 2 - (h2[1] + h2[3]) / 2)

                        # 计算两条垂直线的距离
                        v_dist = abs((v1[0] + v1[2]) / 2 - (v2[0] + v2[2]) / 2)

                        # 距离合理性检查
                        if h_dist < 20 or h_dist > min(h, w) * 0.8:
                            continue
                        if v_dist < 20 or v_dist > min(h, w) * 0.8:
                            continue

                        # 面积检查
                        area = h_dist * v_dist
                        if area < self.params.min_area or area > self.params.max_area:
                            continue

                        # 构建矩形
                        rect_info = self._construct_rectangle_from_lines(h1, h2, v1, v2, image_shape)
                        if rect_info and rect_info.confidence >= self.params.confidence_threshold:
                            rectangles.append(rect_info)

        return rectangles

    def _construct_rectangle_from_lines(self, h1, h2, v1, v2, image_shape) -> Optional[RectangleInfo]:
        """
        从四条直线构建矩形

        Args:
            h1, h2: 两条水平线
            v1, v2: 两条垂直线
            image_shape: 图像尺寸

        Returns:
            矩形信息对象，如果无效则返回None
        """
        try:
            # 计算交点（简化处理）
            h_y1, h_y2 = (h1[1] + h1[3]) / 2, (h2[1] + h2[3]) / 2
            v_x1, v_x2 = (v1[0] + v1[2]) / 2, (v2[0] + v2[2]) / 2

            corners = [
                (v_x1, h_y1), (v_x2, h_y1),
                (v_x2, h_y2), (v_x1, h_y2)
            ]

            # 检查角点是否在图像范围内
            h, w = image_shape[:2]
            for corner in corners:
                if corner[0] < 0 or corner[0] >= w or corner[1] < 0 or corner[1] >= h:
                    return None

            # 计算矩形属性
            center_x = (v_x1 + v_x2) / 2
            center_y = (h_y1 + h_y2) / 2
            width = abs(v_x2 - v_x1)
            height = abs(h_y2 - h_y1)

            # 长宽比检查
            aspect_ratio = max(width, height) / min(width, height)
            if (aspect_ratio < self.params.min_aspect_ratio or
                aspect_ratio > self.params.max_aspect_ratio):
                return None

            # 计算置信度（基于线段质量）
            confidence = 0.7  # 霍夫变换的基础置信度

            return RectangleInfo(
                center=(center_x, center_y),
                size=(width, height),
                angle=0,  # 轴对齐矩形
                confidence=confidence,
                corners=[(int(c[0]), int(c[1])) for c in corners]
            )

        except Exception as e:
            logger.warning(f"构建矩形失败: {e}")
            return None

    def _apply_non_maximum_suppression(self, rectangles: List[RectangleInfo],
                                     iou_threshold: float = 0.3) -> List[RectangleInfo]:
        """
        非极大值抑制去除重叠检测

        Args:
            rectangles: 候选矩形列表
            iou_threshold: IoU阈值

        Returns:
            过滤后的矩形列表
        """
        if not rectangles:
            return []

        # 按置信度排序
        sorted_rects = sorted(rectangles, key=lambda x: x.confidence, reverse=True)

        keep = []
        while sorted_rects:
            # 保留置信度最高的矩形
            best = sorted_rects.pop(0)
            keep.append(best)

            # 移除与当前最佳矩形重叠度过高的其他矩形
            filtered_rects = []
            for rect in sorted_rects:
                iou = self._calculate_iou(best, rect)
                if iou < iou_threshold:
                    filtered_rects.append(rect)

            sorted_rects = filtered_rects

        return keep

    def _calculate_iou(self, rect1: RectangleInfo, rect2: RectangleInfo) -> float:
        """
        计算两个矩形的交并比(IoU)

        Args:
            rect1, rect2: 两个矩形

        Returns:
            IoU值 [0, 1]
        """
        # 计算边界框
        x1_min = rect1.center[0] - rect1.size[0] / 2
        y1_min = rect1.center[1] - rect1.size[1] / 2
        x1_max = rect1.center[0] + rect1.size[0] / 2
        y1_max = rect1.center[1] + rect1.size[1] / 2

        x2_min = rect2.center[0] - rect2.size[0] / 2
        y2_min = rect2.center[1] - rect2.size[1] / 2
        x2_max = rect2.center[0] + rect2.size[0] / 2
        y2_max = rect2.center[1] + rect2.size[1] / 2

        # 计算交集
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # 计算并集
        area1 = rect1.size[0] * rect1.size[1]
        area2 = rect2.size[0] * rect2.size[1]
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def visualize_results(self, image: np.ndarray, rectangles: List[RectangleInfo],
                         show_confidence: bool = True) -> np.ndarray:
        """
        可视化检测结果

        Args:
            image: 原始图像
            rectangles: 检测到的矩形
            show_confidence: 是否显示置信度

        Returns:
            标注后的图像
        """
        result_image = image.copy()

        for i, rect in enumerate(rectangles):
            # 绘制矩形轮廓
            cv2.drawContours(result_image, [np.array(rect.corners)], -1, (0, 255, 0), 2)

            # 绘制中心点
            center = (int(rect.center[0]), int(rect.center[1]))
            cv2.circle(result_image, center, 3, (0, 0, 255), -1)

            # 标注置信度和编号
            if show_confidence:
                label = f"{i}: {rect.confidence:.2f}"
                cv2.putText(result_image, label,
                          (center[0] - 20, center[1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        return result_image

    def analyze_detection_quality(self, rectangles: List[RectangleInfo]) -> Dict[str, Any]:
        """
        分析检测质量

        Args:
            rectangles: 检测结果

        Returns:
            质量分析报告
        """
        if not rectangles:
            return {'count': 0, 'avg_confidence': 0.0, 'confidence_std': 0.0}

        confidences = [rect.confidence for rect in rectangles]
        areas = [rect.size[0] * rect.size[1] for rect in rectangles]
        aspect_ratios = [max(rect.size) / min(rect.size) for rect in rectangles]

        analysis = {
            'count': len(rectangles),
            'avg_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'avg_area': np.mean(areas),
            'area_std': np.std(areas),
            'avg_aspect_ratio': np.mean(aspect_ratios),
            'aspect_ratio_std': np.std(aspect_ratios)
        }

        return analysis


def demo_basic_detection():
    """演示基础矩形检测功能"""
    print("=== 矩形检测基础演示 ===")

    # 创建检测器
    detector = RectangleDetector()

    # 创建测试图像
    test_image = create_test_image()

    print(f"测试图像尺寸: {test_image.shape}")

    # 轮廓检测
    print("执行轮廓检测...")
    contour_rects = detector.detect_rectangles(test_image, method='contour')
    print(f"轮廓检测结果: {len(contour_rects)} 个矩形")

    # 霍夫变换检测
    print("执行霍夫变换检测...")
    hough_rects = detector.detect_rectangles(test_image, method='hough')
    print(f"霍夫变换结果: {len(hough_rects)} 个矩形")

    # 混合检测
    print("执行混合检测...")
    hybrid_rects = detector.detect_rectangles(test_image, method='hybrid')
    print(f"混合检测结果: {len(hybrid_rects)} 个矩形")

    # 质量分析
    analysis = detector.analyze_detection_quality(hybrid_rects)
    print("\n检测质量分析:")
    for key, value in analysis.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")

    print("演示完成！")

    return {
        'test_image': test_image,
        'contour_rects': contour_rects,
        'hough_rects': hough_rects,
        'hybrid_rects': hybrid_rects
    }


def demo_parameter_tuning():
    """演示参数调优功能"""
    print("=== 参数调优演示 ===")

    # 不同参数配置
    configs = [
        RectDetectionParams(min_area=100.0, max_area=10000.0, confidence_threshold=0.5),
        RectDetectionParams(min_area=500.0, max_area=20000.0, confidence_threshold=0.7),
        RectDetectionParams(min_area=1000.0, max_area=50000.0, confidence_threshold=0.8),
    ]

    test_image = create_test_image()

    results = {}
    for i, params in enumerate(configs):
        print(f"\n配置 {i+1}:")
        print(f"  面积范围: [{params.min_area}, {params.max_area}]")
        print(f"  置信度阈值: {params.confidence_threshold}")

        detector = RectangleDetector(params)
        rectangles = detector.detect_rectangles(test_image, method='hybrid')

        analysis = detector.analyze_detection_quality(rectangles)
        print(f"  检测结果: {analysis['count']} 个矩形")
        print(f"  平均置信度: {analysis['avg_confidence']:.3f}")

        results[f'config_{i+1}'] = {
            'params': params,
            'rectangles': rectangles,
            'analysis': analysis
        }

    print("参数调优演示完成！")
    return results


def create_test_image(size: Tuple[int, int] = (400, 600)) -> np.ndarray:
    """
    创建包含矩形的测试图像

    Args:
        size: 图像尺寸 (高度, 宽度)

    Returns:
        测试图像
    """
    height, width = size
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # 背景渐变
    for y in range(height):
        for x in range(width):
            intensity = int(50 + 100 * (x / width))
            image[y, x] = [intensity, intensity, intensity]

    # 添加一些矩形
    rectangles = [
        [(50, 50), (200, 150)],   # 大矩形
        [(250, 80), (350, 180)],  # 中等矩形
        [(400, 200), (500, 270)], # 长方形
        [(100, 250), (180, 330)], # 正方形
        [(350, 300), (480, 380)]  # 斜矩形区域
    ]

    for (x1, y1), (x2, y2) in rectangles:
        # 绘制实心矩形
        cv2.rectangle(image, (x1, y1), (x2, y2), (200, 150, 100), -1)
        # 绘制边框
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)

    # 添加一些干扰形状
    cv2.circle(image, (300, 150), 30, (100, 200, 100), -1)
    cv2.ellipse(image, (450, 100), (40, 20), 30, 0, 360, (150, 100, 200), -1)

    # 添加噪声
    noise = np.random.normal(0, 10, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return image


def create_document_image() -> np.ndarray:
    """创建文档样式的测试图像"""
    image = np.ones((400, 300, 3), dtype=np.uint8) * 250  # 白色背景

    # 模拟文档页面边界
    cv2.rectangle(image, (20, 20), (280, 380), (0, 0, 0), 2)

    # 添加文本框模拟
    text_boxes = [
        (30, 40, 270, 80),   # 标题区域
        (30, 100, 270, 140), # 段落1
        (30, 160, 270, 200), # 段落2
        (30, 220, 130, 300), # 左列
        (150, 220, 270, 300) # 右列
    ]

    for x1, y1, x2, y2 in text_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (50, 50, 50), 1)

    return image


def performance_benchmark():
    """性能基准测试"""
    print("=== 性能基准测试 ===")

    # 测试不同图像尺寸
    test_sizes = [(200, 300), (400, 600), (800, 1200)]
    methods = ['contour', 'hough', 'hybrid']

    detector = RectangleDetector()

    for size in test_sizes:
        print(f"\n测试图像尺寸: {size[0]}x{size[1]}")
        test_image = create_test_image(size)

        for method in methods:
            # 预热
            _ = detector.detect_rectangles(test_image, method=method)

            # 计时测试
            start_time = time.time()
            num_iterations = 5

            for _ in range(num_iterations):
                rectangles = detector.detect_rectangles(test_image, method=method)

            elapsed_time = time.time() - start_time
            avg_time = elapsed_time / num_iterations

            print(f"  {method} 方法: {avg_time*1000:.2f}ms, "
                 f"检测到 {len(rectangles)} 个矩形")


if __name__ == "__main__":
    # 运行演示
    print("启动矩形检测演示程序\n")

    # 基础功能演示
    basic_results = demo_basic_detection()
    print()

    # 参数调优演示
    tuning_results = demo_parameter_tuning()
    print()

    # 性能基准测试
    performance_benchmark()

    print("\n所有演示完成！")