#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌐 球面化效果算法：数字世界的空间魔法师
=========================================

🎨 将平面世界转化为三维空间感知的视觉魔法

作者: GlimmerLab 视觉算法实验室
项目: IP101 - 图像处理算法集
描述: 球面化效果的艺术实现
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time
from typing import Tuple, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import math

class CurveType(Enum):
    """🎨 变形曲线类型"""
    LINEAR = "linear"
    SMOOTH = "smooth"
    S_CURVE = "s_curve"
    EXPONENTIAL = "exponential"

@dataclass
class SpherizeParams:
    """🌟 球面化效果参数"""
    strength: float = 0.5                    # 变形强度 [0.0, 1.0]
    radius: float = 0.8                      # 影响半径比例 [0.1, 1.0]
    center: Optional[Tuple[float, float]] = None  # 变形中心
    invert: bool = False                     # 是否反向（凹陷）
    curve_type: CurveType = CurveType.SMOOTH # 变形曲线类型
    curve_power: float = 2.0                 # 指数曲线的幂次

class SpherizeArtist:
    """🎨 球面化效果艺术家：空间变形的创造者"""

    def __init__(self):
        """🌟 初始化空间变形大师"""
        print("🌐 球面化效果艺术家已准备就绪，开始创造空间变形的魔法！")

    def spherize(self, image: np.ndarray, params: SpherizeParams) -> np.ndarray:
        """
        🌐 主球面化函数：空间变形的核心魔法

        Args:
            image: 输入图像
            params: 球面化参数

        Returns:
            变形后的图像
        """
        if image is None or image.size == 0:
            raise ValueError("🚫 输入图像为空")

        h, w = image.shape[:2]

        # 🎯 确定变形中心
        if params.center is None:
            center = (w / 2, h / 2)
        else:
            center = params.center

        # 📏 计算最大半径
        max_radius = params.radius * min(
            center[0], center[1],
            w - center[0], h - center[1]
        )

        # 🎨 创建坐标网格
        y_coords, x_coords = np.ogrid[:h, :w]

        # 📐 计算到中心的距离
        dx = x_coords - center[0]
        dy = y_coords - center[1]
        distance = np.sqrt(dx*dx + dy*dy)

        # 🌊 创建变形掩码
        mask = distance < max_radius
        valid_mask = (distance > 0) & mask

        # ✨ 计算变形后的坐标
        normalized_dist = np.zeros_like(distance)
        normalized_dist[valid_mask] = distance[valid_mask] / max_radius

        # 🎭 应用变形曲线
        strength_factor = self._apply_curve(normalized_dist, params.curve_type, params.curve_power)

        # 🌟 计算变形系数
        if params.invert:
            # 凹陷效果：向内收缩
            factor = 1.0 + params.strength * strength_factor * (1.0 - normalized_dist)
        else:
            # 凸出效果：向外扩张
            factor = 1.0 - params.strength * strength_factor * (1.0 - normalized_dist)

        # 📍 计算新坐标
        new_distance = np.zeros_like(distance)
        new_distance[valid_mask] = distance[valid_mask] * factor[valid_mask]

        scale = np.ones_like(distance)
        scale[valid_mask] = new_distance[valid_mask] / distance[valid_mask]

        src_x = center[0] + dx * scale
        src_y = center[1] + dy * scale

        # 🎨 双线性插值
        result = self._bilinear_interpolate(image, src_x, src_y)

        # 📋 保留未变形区域
        result[~mask] = image[~mask]

        return result

    def _apply_curve(self, t: np.ndarray, curve_type: CurveType, power: float = 2.0) -> np.ndarray:
        """
        🎨 应用变形曲线：不同的艺术表达方式

        Args:
            t: 归一化距离 [0, 1]
            curve_type: 曲线类型
            power: 指数曲线的幂次

        Returns:
            变形后的强度因子
        """
        if curve_type == CurveType.LINEAR:
            return t
        elif curve_type == CurveType.SMOOTH:
            # Hermite插值：平滑过渡
            return t * t * (3.0 - 2.0 * t)
        elif curve_type == CurveType.S_CURVE:
            # S型曲线：生动的变化
            return 0.5 * (1.0 + np.sin((t - 0.5) * np.pi))
        elif curve_type == CurveType.EXPONENTIAL:
            # 指数曲线：渐进式变化
            return np.power(t, power)
        else:
            return t

    def _bilinear_interpolate(self, image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        🔍 双线性插值：高质量的像素重采样

        Args:
            image: 源图像
            x: 目标x坐标
            y: 目标y坐标

        Returns:
            插值后的像素值
        """
        h, w = image.shape[:2]

        # 📐 计算插值坐标
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1

        # ⚖️ 边界检查
        x0 = np.clip(x0, 0, w - 1)
        y0 = np.clip(y0, 0, h - 1)
        x1 = np.clip(x1, 0, w - 1)
        y1 = np.clip(y1, 0, h - 1)

        # 🎯 计算权重
        wx = x - x0
        wy = y - y0

        # 🎨 执行插值
        if len(image.shape) == 3:
            # 彩色图像
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:, :, c] = (
                    (1 - wx) * (1 - wy) * image[y0, x0, c] +
                    wx * (1 - wy) * image[y0, x1, c] +
                    (1 - wx) * wy * image[y1, x0, c] +
                    wx * wy * image[y1, x1, c]
                )
        else:
            # 灰度图像
            result = (
                (1 - wx) * (1 - wy) * image[y0, x0] +
                wx * (1 - wy) * image[y0, x1] +
                (1 - wx) * wy * image[y1, x0] +
                wx * wy * image[y1, x1]
            )

        return result.astype(image.dtype)

    def bulge_effect(self, image: np.ndarray, strength: float = 0.5,
                    center: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        🌟 凸出效果：向外的张力美学

        Args:
            image: 输入图像
            strength: 效果强度
            center: 效果中心

        Returns:
            凸出效果后的图像
        """
        params = SpherizeParams(
            strength=strength,
            center=center,
            invert=False,
            curve_type=CurveType.SMOOTH
        )
        return self.spherize(image, params)

    def pinch_effect(self, image: np.ndarray, strength: float = 0.5,
                    center: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        🌙 收缩效果：向内的聚合美学

        Args:
            image: 输入图像
            strength: 效果强度
            center: 效果中心

        Returns:
            收缩效果后的图像
        """
        params = SpherizeParams(
            strength=strength,
            center=center,
            invert=True,
            curve_type=CurveType.SMOOTH
        )
        return self.spherize(image, params)

    def fisheye_effect(self, image: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """
        🐟 鱼眼效果：广角视野的艺术表达

        Args:
            image: 输入图像
            strength: 效果强度

        Returns:
            鱼眼效果后的图像
        """
        h, w = image.shape[:2]
        center = (w / 2, h / 2)

        # 鱼眼效果使用更大的变形范围
        params = SpherizeParams(
            strength=strength,
            radius=1.0,
            center=center,
            invert=False,
            curve_type=CurveType.EXPONENTIAL,
            curve_power=1.5
        )
        return self.spherize(image, params)

    def create_artistic_showcase(self, image: np.ndarray, save_path: Optional[str] = None) -> None:
        """
        🎭 创建艺术效果展示：球面化的视觉交响乐

        Args:
            image: 输入图像
            save_path: 保存路径
        """
        print("🎨 开始创作球面化效果艺术作品...")

        effects = {
            "📷 原始图像": image,
            "🌟 轻微凸出": self.bulge_effect(image, 0.3),
            "🌙 轻微凹陷": self.pinch_effect(image, 0.3),
            "✨ 强烈凸出": self.bulge_effect(image, 0.7),
            "🔮 强烈凹陷": self.pinch_effect(image, 0.7),
            "🌊 平滑曲线": self.spherize(image, SpherizeParams(
                strength=0.5, curve_type=CurveType.SMOOTH)),
            "📈 S型曲线": self.spherize(image, SpherizeParams(
                strength=0.5, curve_type=CurveType.S_CURVE)),
            "🐟 鱼眼效果": self.fisheye_effect(image, 0.6)
        }

        # 🖼️ 创造视觉艺术馆
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('🌐 球面化效果艺术馆：空间变形的魔法展示',
                    fontsize=16, fontweight='bold', y=0.98)

        for i, (title, effect_image) in enumerate(effects.items()):
            row, col = i // 4, i % 4

            if len(effect_image.shape) == 3:
                # BGR转RGB显示
                display_image = cv2.cvtColor(effect_image, cv2.COLOR_BGR2RGB)
                axes[row, col].imshow(display_image)
            else:
                axes[row, col].imshow(effect_image, cmap='gray')

            axes[row, col].set_title(title, fontsize=11, pad=10)
            axes[row, col].axis('off')

            # 添加美化边框
            for spine in axes[row, col].spines.values():
                spine.set_linewidth(2)
                spine.set_color('lightgray')

        plt.tight_layout()
        plt.subplots_adjust(top=0.93, hspace=0.15, wspace=0.1)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"💾 艺术展示已保存至: {save_path}")

        plt.show()
        print("🎨 艺术展示完成，感谢欣赏空间变形的魔法！")

    def interactive_demo(self, image: np.ndarray) -> None:
        """
        🎮 交互式演示：实时体验球面化参数效果

        Args:
            image: 演示图像
        """
        print("🎮 启动交互式演示模式")
        print("💡 提示：观察不同参数组合的空间变形效果")

        # 创建参数组合
        parameter_sets = [
            ("轻微凸出", SpherizeParams(strength=0.2, invert=False)),
            ("中等凸出", SpherizeParams(strength=0.5, invert=False)),
            ("强烈凸出", SpherizeParams(strength=0.8, invert=False)),
            ("轻微凹陷", SpherizeParams(strength=0.2, invert=True)),
            ("中等凹陷", SpherizeParams(strength=0.5, invert=True)),
            ("强烈凹陷", SpherizeParams(strength=0.8, invert=True))
        ]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('🎮 交互式球面化演示：探索空间变形的艺术',
                    fontsize=14, fontweight='bold')

        for i, (name, params) in enumerate(parameter_sets):
            row, col = i // 3, i % 3

            result = self.spherize(image, params)

            if len(result.shape) == 3:
                display_image = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                axes[row, col].imshow(display_image)
            else:
                axes[row, col].imshow(result, cmap='gray')

            axes[row, col].set_title(f'{name}\n强度={params.strength:.1f}', fontsize=10)
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.show()

        print("🎮 交互式演示完成！你可以修改参数来探索更多效果")

    def performance_test(self, image: np.ndarray, iterations: int = 5) -> Dict[str, float]:
        """
        ⚡ 性能测试：评估算法的运行效率

        Args:
            image: 测试图像
            iterations: 测试迭代次数

        Returns:
            各种方法的平均执行时间
        """
        print(f"⚡ 开始性能测试，图像尺寸: {image.shape}, 迭代次数: {iterations}")

        results = {}

        # 测试基础球面化
        params = SpherizeParams()
        start_time = time.time()
        for _ in range(iterations):
            self.spherize(image, params)
        results['basic_spherize'] = (time.time() - start_time) / iterations

        # 测试凸出效果
        start_time = time.time()
        for _ in range(iterations):
            self.bulge_effect(image)
        results['bulge_effect'] = (time.time() - start_time) / iterations

        # 测试收缩效果
        start_time = time.time()
        for _ in range(iterations):
            self.pinch_effect(image)
        results['pinch_effect'] = (time.time() - start_time) / iterations

        # 测试鱼眼效果
        start_time = time.time()
        for _ in range(iterations):
            self.fisheye_effect(image)
        results['fisheye_effect'] = (time.time() - start_time) / iterations

        print("\n📊 性能测试结果:")
        for method, time_cost in results.items():
            print(f"   {method}: {time_cost:.4f}s")

        return results

def create_test_image(size: Tuple[int, int] = (400, 400)) -> np.ndarray:
    """
    🎨 创建测试图像：球面化效果的完美画布

    Args:
        size: 图像尺寸

    Returns:
        测试图像
    """
    h, w = size
    image = np.zeros((h, w, 3), dtype=np.uint8)

    # 创建渐变背景
    for y in range(h):
        for x in range(w):
            # 径向渐变
            center_x, center_y = w // 2, h // 2
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)

            intensity = int(255 * (1 - distance / max_distance))
            image[y, x] = [intensity // 3, intensity // 2, intensity]

    # 添加网格线，便于观察变形效果
    grid_size = 40
    for i in range(0, w, grid_size):
        cv2.line(image, (i, 0), (i, h), (255, 255, 255), 1)
    for i in range(0, h, grid_size):
        cv2.line(image, (0, i), (w, i), (255, 255, 255), 1)

    # 添加中心标记
    center_x, center_y = w // 2, h // 2
    cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), -1)
    cv2.circle(image, (center_x, center_y), 15, (255, 0, 0), 2)

    return image

def main():
    """🎯 主函数"""
    parser = argparse.ArgumentParser(
        description="🌐 球面化效果算法 - 数字世界的空间魔法师",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--input', '-i', type=str, help='输入图像路径')
    parser.add_argument('--output', '-o', type=str, help='输出图像路径')
    parser.add_argument('--strength', type=float, default=0.5, help='变形强度 (0.0-1.0)')
    parser.add_argument('--radius', type=float, default=0.8, help='影响半径比例 (0.1-1.0)')
    parser.add_argument('--invert', action='store_true', help='反向效果（凹陷）')
    parser.add_argument('--curve', type=str, default='smooth',
                       choices=['linear', 'smooth', 's_curve', 'exponential'],
                       help='变形曲线类型')
    parser.add_argument('--showcase', action='store_true', help='显示艺术效果展示')
    parser.add_argument('--performance', action='store_true', help='运行性能测试')
    parser.add_argument('--interactive', action='store_true', help='启动交互式演示')
    parser.add_argument('--demo', action='store_true', help='运行完整演示')

    args = parser.parse_args()

    print("🌐 球面化效果算法启动")

    # 创建艺术家
    artist = SpherizeArtist()

    # 创建参数
    curve_map = {
        'linear': CurveType.LINEAR,
        'smooth': CurveType.SMOOTH,
        's_curve': CurveType.S_CURVE,
        'exponential': CurveType.EXPONENTIAL
    }

    params = SpherizeParams(
        strength=args.strength,
        radius=args.radius,
        invert=args.invert,
        curve_type=curve_map[args.curve]
    )

    if args.demo:
        # 演示模式
        print("🎭 启动演示模式，使用内置测试图像...")
        test_image = create_test_image()

        print(f"📊 图像信息: 尺寸={test_image.shape}")

        # 艺术展示
        if args.showcase or True:
            artist.create_artistic_showcase(test_image)

        # 性能测试
        if args.performance or True:
            artist.performance_test(test_image)

        # 交互式演示
        if args.interactive:
            artist.interactive_demo(test_image)

    elif args.input:
        # 文件处理模式
        if not os.path.exists(args.input):
            print(f"🚫 文件不存在: {args.input}")
            return

        image = cv2.imread(args.input)
        if image is None:
            print(f"🚫 无法读取图像: {args.input}")
            return

        print(f"📖 读取图像: {args.input}, 尺寸: {image.shape}")

        # 应用球面化效果
        result = artist.spherize(image, params)
        print("✨ 球面化效果处理完成")

        # 保存结果
        if args.output:
            cv2.imwrite(args.output, result)
            print(f"💾 结果已保存至: {args.output}")

        # 显示结果
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('原始图像')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title(f'球面化效果 (强度={args.strength})')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        # 其他功能
        if args.showcase:
            artist.create_artistic_showcase(image)

        if args.performance:
            artist.performance_test(image)

        if args.interactive:
            artist.interactive_demo(image)

    else:
        print("🤔 请提供输入图像路径或使用 --demo 运行演示")
        print("💡 使用 --help 查看详细帮助信息")

if __name__ == "__main__":
    print("🌐 欢迎使用球面化效果算法 - 数字世界的空间魔法师")
    print("✨ 让平面的世界拥有立体的灵魂...")

    try:
        main()
    except KeyboardInterrupt:
        print("\n🎭 用户中断，感谢体验球面化效果的艺术之旅！")
    except Exception as e:
        print(f"\n🚫 程序异常: {e}")
        print("💡 请检查输入参数和图像文件")

    print("🌟 探索永无止境，创造无限可能！")