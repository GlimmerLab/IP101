#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🚗 数字世界的交通警察：车牌检测算法的智慧探索

一个能在复杂视觉环境中精确定位车牌的智能检测系统
通过边缘分析、色彩特征和几何约束的融合，实现车牌的自动识别与定位

作者: 智能交通探索者
版本: 1.0.0
日期: 2024年
项目: IP101/GlimmerLab - 让技术服务于美好生活 ✨
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import math
import time
import argparse
from pathlib import Path


@dataclass
class LicensePlateInfo:
    """🚗 车牌信息的数据结构"""
    rect: Tuple[int, int, int, int]  # 车牌区域矩形 (x, y, w, h)
    plate_img: np.ndarray           # 车牌图像
    confidence: float               # 检测置信度
    chars: List[Tuple[int, int, int, int]] = None  # 字符区域列表
    plate_number: str = ""          # 车牌号码

    def __post_init__(self):
        """初始化字符区域列表"""
        if self.chars is None:
            self.chars = []


@dataclass
class LicensePlateDetectionParams:
    """🎯 车牌检测的参数配置"""
    min_area_ratio: float = 0.001      # 车牌面积与图像面积的最小比例
    max_area_ratio: float = 0.05       # 车牌面积与图像面积的最大比例
    min_aspect_ratio: float = 2.0      # 车牌宽高比最小值
    max_aspect_ratio: float = 6.0      # 车牌宽高比最大值
    min_plate_confidence: float = 0.6  # 最小车牌检测置信度

    def __post_init__(self):
        """🔧 参数验证 - 确保检测的合理性"""
        if self.min_area_ratio <= 0 or self.max_area_ratio <= 0:
            raise ValueError("🚫 面积比例必须大于0！")
        if self.min_area_ratio >= self.max_area_ratio:
            raise ValueError("🚫 最小面积比例不能大于等于最大面积比例！")
        if self.min_aspect_ratio <= 0 or self.max_aspect_ratio <= 0:
            raise ValueError("🚫 宽高比必须大于0！")
        if self.min_aspect_ratio >= self.max_aspect_ratio:
            raise ValueError("🚫 最小宽高比不能大于等于最大宽高比！")
        if not 0 <= self.min_plate_confidence <= 1:
            raise ValueError("🚫 置信度必须在0-1范围内！")


class LicensePlateDetector:
    """🔍 智能车牌检测器：数字世界的交通警察"""

    def __init__(self):
        """🌟 初始化车牌检测器"""
        print("🚗 智能车牌检测器已启动，准备在数字世界中执勤！")

    def detect_license_plates(self, image: np.ndarray,
                            params: LicensePlateDetectionParams) -> List[LicensePlateInfo]:
        """
        🎯 主检测函数：综合多种方法的智能车牌检测

        通过边缘检测和色彩分析的结合，在复杂场景中精确定位车牌。
        这个过程就像训练有素的交警，能从车流中快速识别目标。

        Args:
            image: 输入图像 (BGR格式)
            params: 检测参数配置

        Returns:
            检测到的车牌信息列表

        Raises:
            ValueError: 当输入图像为空时
        """
        if image is None or image.size == 0:
            raise ValueError("🚫 输入图像为空，检测需要图像数据！")

        print("🔍 开始车牌检测，多重方法并行分析...")

        # 🔍 边缘检测方法 - 理性的几何分析
        edge_plates = self._detect_plates_edge_based(image, params)
        print(f"📐 边缘检测发现 {len(edge_plates)} 个候选车牌")

        # 🎨 色彩检测方法 - 感性的视觉感知
        color_plates = self._detect_plates_color_based(image, params)
        print(f"🌈 色彩检测发现 {len(color_plates)} 个候选车牌")

        # 🤝 融合两种检测结果
        all_plates = edge_plates + color_plates

        if not all_plates:
            print("🤔 未检测到车牌，请检查图像质量或调整参数")
            return []

        # 🎯 非极大值抑制 - 去除重叠检测
        final_plates = self._apply_nms(all_plates, params)
        print(f"✅ 最终确认 {len(final_plates)} 个有效车牌")

        # 🔧 后处理：校正和字符分割
        for plate in final_plates:
            if plate.confidence >= params.min_plate_confidence:
                # 📐 校正车牌倾斜
                plate.plate_img = self._correct_plate_skew(plate.plate_img)

                # ✂️ 字符分割
                plate.chars = self._segment_plate_chars(plate.plate_img)

        return final_plates

    def _detect_plates_edge_based(self, image: np.ndarray,
                                 params: LicensePlateDetectionParams) -> List[LicensePlateInfo]:
        """
        🔍 基于边缘检测的车牌定位：寻找结构的艺术

        通过Sobel算子和形态学操作，在复杂背景中发现车牌的几何特征。

        Args:
            image: 输入图像
            params: 检测参数

        Returns:
            基于边缘特征检测到的车牌列表
        """
        print("🔍 启动边缘检测模式...")

        # 🎨 转换为灰度世界 - 简化复杂，突出本质
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # ✨ 增强对比度 - 让隐藏的结构显现
        enhanced = cv2.equalizeHist(gray)

        # 🌫️ 高斯模糊去噪 - 在清晰与模糊间寻找平衡
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

        # 🔍 Sobel边缘检测 - 数学与艺术的完美结合
        grad_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=3)

        # 🎭 梯度的艺术融合
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        # ⚫⚪ 二值化 - 将灰度世界简化为黑白对立
        _, binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # 🔗 形态学处理 - 连接断裂的边缘，构建完整形状
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, element)

        # 🔍 轮廓检测 - 寻找边缘构成的封闭区域
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return self._analyze_contours(contours, image, params, "edge")

    def _detect_plates_color_based(self, image: np.ndarray,
                                  params: LicensePlateDetectionParams) -> List[LicensePlateInfo]:
        """
        🌈 基于色彩特征的车牌定位：文化符号的视觉密码

        在HSV色彩空间中寻找符合车牌颜色特征的区域。

        Args:
            image: 输入图像
            params: 检测参数

        Returns:
            基于色彩特征检测到的车牌列表
        """
        if len(image.shape) != 3:
            print("🎨 色彩检测需要彩色图像，跳过此方法")
            return []

        print("🌈 启动色彩检测模式...")

        # 🌈 转换到HSV色彩空间 - 更接近人类的色彩感知
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 🔵 中国蓝牌的色彩范围 - 庄重而稳定的蓝色
        lower_blue = np.array([100, 70, 70])
        upper_blue = np.array([130, 255, 255])

        # 🟡 黄色车牌的色彩范围 - 醒目而活跃的黄色
        lower_yellow = np.array([15, 70, 70])
        upper_yellow = np.array([35, 255, 255])

        # 🎭 创建色彩掩码 - 在色彩海洋中寻找特定岛屿
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # ✨ 融合不同颜色的检测结果
        combined_mask = cv2.bitwise_or(blue_mask, yellow_mask)

        # 🔍 形态学优化 - 清理噪声，保留主要结构
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, element)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, element)

        # 🔍 轮廓检测
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return self._analyze_contours(contours, image, params, "color")

    def _analyze_contours(self, contours: List, image: np.ndarray,
                         params: LicensePlateDetectionParams,
                         method: str) -> List[LicensePlateInfo]:
        """
        📊 轮廓分析：从形状中提取车牌候选

        对检测到的轮廓进行几何特征分析，筛选出符合车牌特征的区域。

        Args:
            contours: 检测到的轮廓列表
            image: 原始图像
            params: 检测参数
            method: 检测方法标识

        Returns:
            分析后的车牌候选列表
        """
        plates = []
        img_area = image.shape[0] * image.shape[1]
        min_area = img_area * params.min_area_ratio
        max_area = img_area * params.max_area_ratio

        print(f"📊 分析 {len(contours)} 个轮廓，方法: {method}")

        for i, contour in enumerate(contours):
            # 📐 计算轮廓面积
            area = cv2.contourArea(contour)

            # 🔍 面积过滤
            if area < min_area or area > max_area:
                continue

            # 📦 计算边界矩形
            x, y, w, h = cv2.boundingRect(contour)

            # 🔍 确保矩形在图像范围内
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)

            if w <= 0 or h <= 0:
                continue

            # 📏 计算长宽比
            aspect_ratio = w / h

            # 🎯 长宽比过滤
            if aspect_ratio < params.min_aspect_ratio or aspect_ratio > params.max_aspect_ratio:
                continue

            # ✂️ 提取候选车牌区域
            plate_img = image[y:y+h, x:x+w].copy()

            # 🎯 计算置信度
            confidence = self._calculate_confidence(plate_img, method)

            # 📝 创建车牌信息
            if confidence >= params.min_plate_confidence:
                plate_info = LicensePlateInfo(
                    rect=(x, y, w, h),
                    plate_img=plate_img,
                    confidence=confidence
                )
                plates.append(plate_info)

        print(f"✅ {method}检测找到 {len(plates)} 个有效候选")
        return plates

    def _calculate_confidence(self, plate_img: np.ndarray, method: str) -> float:
        """
        🎯 置信度计算：评估车牌候选的可信程度

        通过分析图像的纹理特征来计算检测置信度。

        Args:
            plate_img: 车牌候选图像
            method: 检测方法

        Returns:
            置信度分数 (0-1)
        """
        if plate_img.size == 0:
            return 0.0

        # 🎨 转换为灰度图
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()

        # 🔍 Sobel垂直边缘检测 - 字符检测的关键特征
        sobel_x = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)

        # ⚫⚪ 二值化处理
        _, thresh = cv2.threshold(sobel_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # 📊 计算边缘密度
        edge_count = cv2.countNonZero(thresh)
        total_pixels = gray.shape[0] * gray.shape[1]
        edge_density = edge_count / total_pixels

        # 🎯 基于方法调整置信度
        if method == "edge":
            # 边缘检测方法更依赖边缘密度
            confidence = min(1.0, edge_density * 50.0)
        else:  # color method
            # 色彩检测方法适当降低边缘密度要求
            confidence = min(1.0, edge_density * 30.0 + 0.3)

        return confidence

    def _apply_nms(self, plates: List[LicensePlateInfo],
                   params: LicensePlateDetectionParams) -> List[LicensePlateInfo]:
        """
        🎯 非极大值抑制：在众多候选中选择最优秀的

        通过IoU计算去除重叠度过高的检测框，保留最佳结果。

        Args:
            plates: 车牌候选列表
            params: 检测参数

        Returns:
            NMS后的车牌列表
        """
        if not plates:
            return []

        # 📊 准备NMS数据
        boxes = []
        scores = []

        for plate in plates:
            x, y, w, h = plate.rect
            boxes.append([x, y, x + w, y + h])
            scores.append(plate.confidence)

        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)

        # 🏆 应用OpenCV的NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            params.min_plate_confidence,
            0.3  # IoU阈值
        )

        # ✅ 提取保留的车牌
        if len(indices) > 0:
            indices = indices.flatten()
            return [plates[i] for i in indices]
        else:
            return []

    def _correct_plate_skew(self, plate_img: np.ndarray) -> np.ndarray:
        """
        📐 车牌倾斜校正：还原最佳视角

        通过霍夫直线检测找到车牌的主要方向，然后进行旋转校正。

        Args:
            plate_img: 输入车牌图像

        Returns:
            校正后的车牌图像
        """
        if plate_img.size == 0:
            return plate_img

        # 🎨 转换为灰度
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()

        # 🔍 边缘检测
        edges = cv2.Canny(gray, 50, 150)

        # 📏 霍夫直线检测 - 寻找车牌的主要方向
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=30, maxLineGap=10)

        if lines is not None and len(lines) > 0:
            # 🎯 计算主要倾斜角度
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi

                # 只考虑接近水平的线条
                if abs(angle) < 45:
                    angles.append(angle)

            if angles:
                avg_angle = np.mean(angles)

                # 🔄 应用旋转校正
                center = (plate_img.shape[1] // 2, plate_img.shape[0] // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
                corrected = cv2.warpAffine(plate_img, rotation_matrix,
                                         (plate_img.shape[1], plate_img.shape[0]))
                return corrected

        return plate_img

    def _segment_plate_chars(self, plate_img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        ✂️ 车牌字符分割：解析文本的智慧

        通过垂直投影分析，将车牌图像中的字符逐个分离出来。

        Args:
            plate_img: 车牌图像

        Returns:
            字符区域列表 [(x, y, w, h), ...]
        """
        if plate_img.size == 0:
            return []

        # 🎨 转换为灰度并二值化
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # 📊 垂直投影分析 - 寻找字符间的分界
        height, width = binary.shape
        vertical_projection = np.sum(binary, axis=0) // 255

        # 🔍 寻找字符边界
        char_bounds = []
        in_char = False
        char_start = 0

        for x in range(width):
            if not in_char and vertical_projection[x] > 0:
                # 字符开始
                char_start = x
                in_char = True
            elif in_char and vertical_projection[x] == 0:
                # 字符结束
                if x - char_start > 5:  # 过滤太小的区域
                    char_bounds.append((char_start, 0, x - char_start, height))
                in_char = False

        # 处理最后一个字符
        if in_char and width - char_start > 5:
            char_bounds.append((char_start, 0, width - char_start, height))

        return char_bounds

    def visualize_detection(self, image: np.ndarray,
                          plates: List[LicensePlateInfo],
                          save_path: Optional[str] = None) -> None:
        """
        🎨 检测结果可视化：展示算法的智慧成果

        在原图上绘制检测到的车牌区域和字符分割结果。

        Args:
            image: 原始图像
            plates: 检测到的车牌列表
            save_path: 保存路径 (可选)
        """
        print(f"🎨 可视化 {len(plates)} 个检测结果...")

        # 🖼️ 创建结果图像
        result_img = image.copy()

        # 🎯 绘制车牌区域
        for i, plate in enumerate(plates):
            x, y, w, h = plate.rect

            # 🔲 绘制车牌边框
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 📝 添加置信度标签
            label = f"Plate {i+1}: {plate.confidence:.2f}"
            cv2.putText(result_img, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ✂️ 绘制字符分割线
            for char_x, char_y, char_w, char_h in plate.chars:
                abs_x = x + char_x
                abs_y = y + char_y
                cv2.rectangle(result_img, (abs_x, abs_y),
                            (abs_x + char_w, abs_y + char_h), (255, 0, 0), 1)

        # 🖼️ 显示结果
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 原图
        if len(image.shape) == 3:
            axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            axes[0].imshow(image, cmap='gray')
        axes[0].set_title('📷 原始图像', fontsize=12)
        axes[0].axis('off')

        # 检测结果
        if len(result_img.shape) == 3:
            axes[1].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        else:
            axes[1].imshow(result_img, cmap='gray')
        axes[1].set_title(f'🎯 检测结果 (发现{len(plates)}个车牌)', fontsize=12)
        axes[1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 检测结果已保存至: {save_path}")

        plt.show()

    def performance_test(self, image: np.ndarray,
                        params: LicensePlateDetectionParams,
                        iterations: int = 3) -> Dict[str, float]:
        """
        ⚡ 性能测试：评估检测算法的效率

        测试车牌检测算法在不同条件下的执行时间。

        Args:
            image: 测试图像
            params: 检测参数
            iterations: 测试迭代次数

        Returns:
            性能测试结果字典
        """
        print(f"⚡ 开始性能测试，图像尺寸: {image.shape}, 迭代次数: {iterations}")

        results = {}

        # 测试完整检测流程
        start_time = time.time()
        for _ in range(iterations):
            self.detect_license_plates(image, params)
        results['complete_detection'] = (time.time() - start_time) / iterations

        # 测试边缘检测方法
        start_time = time.time()
        for _ in range(iterations):
            self._detect_plates_edge_based(image, params)
        results['edge_detection'] = (time.time() - start_time) / iterations

        # 测试色彩检测方法
        start_time = time.time()
        for _ in range(iterations):
            self._detect_plates_color_based(image, params)
        results['color_detection'] = (time.time() - start_time) / iterations

        print("\n📊 性能测试结果:")
        for method, time_cost in results.items():
            print(f"   {method}: {time_cost:.4f}s")

        return results


def main():
    """
    🚀 主函数：命令行界面和使用示例

    提供完整的命令行接口，支持车牌检测的各种功能。
    """
    parser = argparse.ArgumentParser(
        description="🚗 智能车牌检测器 - 数字世界的交通警察",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python license_plate_detection.py input.jpg --output result.jpg
  python license_plate_detection.py input.jpg --visualize --save-vis detection_result.png
  python license_plate_detection.py input.jpg --performance
        """
    )

    parser.add_argument('input', help='📁 输入图像路径')
    parser.add_argument('--output', '-o', help='💾 输出图像路径')
    parser.add_argument('--visualize', action='store_true', help='🎨 可视化检测结果')
    parser.add_argument('--save-vis', help='💾 保存可视化结果的路径')
    parser.add_argument('--performance', action='store_true', help='⚡ 性能测试模式')

    # 参数配置
    parser.add_argument('--min-area-ratio', type=float, default=0.001,
                       help='📐 最小面积比例 (默认: 0.001)')
    parser.add_argument('--max-area-ratio', type=float, default=0.05,
                       help='📐 最大面积比例 (默认: 0.05)')
    parser.add_argument('--min-aspect-ratio', type=float, default=2.0,
                       help='📏 最小宽高比 (默认: 2.0)')
    parser.add_argument('--max-aspect-ratio', type=float, default=6.0,
                       help='📏 最大宽高比 (默认: 6.0)')
    parser.add_argument('--min-confidence', type=float, default=0.6,
                       help='🎯 最小置信度 (默认: 0.6)')

    args = parser.parse_args()

    # 🔍 初始化检测器
    detector = LicensePlateDetector()

    # 📷 加载图像
    image_path = Path(args.input)
    if not image_path.exists():
        print(f"🚫 图像文件不存在: {args.input}")
        return

    print(f"📷 正在加载图像: {args.input}")
    image = cv2.imread(str(image_path))

    if image is None:
        print("🚫 无法加载图像，请检查文件格式！")
        return

    print(f"✅ 图像加载成功，尺寸: {image.shape}")

    # 📝 创建参数对象
    params = LicensePlateDetectionParams(
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        min_aspect_ratio=args.min_aspect_ratio,
        max_aspect_ratio=args.max_aspect_ratio,
        min_plate_confidence=args.min_confidence
    )

    try:
        if args.performance:
            # ⚡ 性能测试模式
            print("⚡ 启动性能测试模式...")
            detector.performance_test(image, params)
        else:
            # 🔍 常规检测模式
            print("🔍 开始车牌检测...")
            plates = detector.detect_license_plates(image, params)

            if plates:
                print(f"🎉 成功检测到 {len(plates)} 个车牌！")

                # 📊 显示检测详情
                for i, plate in enumerate(plates):
                    x, y, w, h = plate.rect
                    print(f"   车牌 {i+1}: 位置({x}, {y}), 尺寸({w}×{h}), 置信度{plate.confidence:.3f}")
                    print(f"           字符数量: {len(plate.chars)}")

                # 🎨 可视化结果
                if args.visualize:
                    detector.visualize_detection(image, plates, args.save_vis)

                # 💾 保存结果
                if args.output:
                    result_img = image.copy()
                    for plate in plates:
                        x, y, w, h = plate.rect
                        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    success = cv2.imwrite(args.output, result_img)
                    if success:
                        print(f"💾 检测结果已保存至: {args.output}")
                    else:
                        print("❌ 保存失败，请检查输出路径！")
            else:
                print("🤔 未检测到车牌，建议:")
                print("   1. 调整置信度阈值 (--min-confidence)")
                print("   2. 检查图像质量和车牌清晰度")
                print("   3. 确认车牌尺寸在合理范围内")

    except Exception as e:
        print(f"❌ 检测过程中发生错误: {e}")
        print("💡 请检查输入参数和图像文件！")


if __name__ == "__main__":
    print("🚗" + "="*60)
    print("    智能车牌检测器 - 数字世界的交通警察")
    print("    IP101/GlimmerLab - 让技术服务于美好生活")
    print("="*64)
    main()