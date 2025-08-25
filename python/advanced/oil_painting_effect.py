#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎨 数字画布上的艺术革命：油画效果算法的创意风暴

一个让像素学会用画家的眼光观察世界的魔法实现
通过邻域分析、色彩量化和笔触模拟，将数字图像转化为充满艺术气息的油画作品

作者: 数字艺术探索者
版本: 1.0.0
日期: 2024年
项目: IP101/GlimmerLab - 让技术与艺术完美融合 ✨
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
import random
import time
import argparse
from pathlib import Path


@dataclass
class OilPaintingParams:
    """🎨 油画效果的艺术参数集合"""
    radius: int = 3              # 邻域半径：画家观察的范围
    levels: int = 10             # 色彩强度级别：调色盘的丰富程度
    dynamic_ratio: int = 15      # 动态范围比例：色彩的表现力

    def __post_init__(self):
        """🔧 参数验证 - 确保艺术创作的合理性"""
        if self.radius < 1:
            raise ValueError("🚫 邻域半径必须大于0，艺术需要观察的范围！")
        if self.levels < 2:
            raise ValueError("🚫 色彩级别必须大于1，单色无法表达丰富情感！")
        if self.dynamic_ratio < 1:
            raise ValueError("🚫 动态比例必须大于0，艺术需要变化！")


class OilPaintingArtist:
    """🎭 数字油画艺术家：让像素学会绘画的魔法师"""

    def __init__(self):
        """🌟 初始化我们的数字艺术家"""
        print("🎨 数字油画艺术家已就位，准备将像素转化为艺术！")

    def basic_oil_painting(self, image: np.ndarray,
                          params: OilPaintingParams) -> np.ndarray:
        """
        🌅 基础油画效果：数字艺术的启蒙之作

        如同学习绘画的第一堂课，简单却蕴含着艺术的本质。
        通过邻域分析和色彩量化，模拟画家用笔刷重新诠释世界的过程。

        Args:
            image: 输入图像 (BGR格式或灰度图)
            params: 油画效果参数

        Returns:
            油画效果图像

        Raises:
            ValueError: 当输入图像为空时
        """
        if image is None or image.size == 0:
            raise ValueError("🚫 输入图像为空，艺术需要素材！")

        # 🎯 获取图像尺寸信息
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1

        # 🎨 创建艺术创作的画布
        result = np.zeros_like(image)

        print("🖼️ 开始艺术创作，这可能需要一些时间...")
        print(f"📐 画布尺寸: {width}x{height}, 色彩通道: {channels}")

        # 🖌️ 遍历每一个"笔触点"
        for y in range(height):
            if y % max(1, height // 10) == 0:
                print(f"🎨 创作进度: {y/height*100:.1f}%")

            for x in range(width):
                # 🔍 定义画家的观察范围 - 艺术的视野边界
                y_min = max(0, y - params.radius)
                y_max = min(height, y + params.radius + 1)
                x_min = max(0, x - params.radius)
                x_max = min(width, x + params.radius + 1)

                # 🎭 为每个强度级别准备"调色盘"
                intensity_counts = [np.zeros(channels, dtype=np.int32)
                                  for _ in range(params.levels)]
                intensity_nums = [0] * params.levels

                # 🌈 在邻域内采集色彩信息 - 如同画家观察自然
                for ny in range(y_min, y_max):
                    for nx in range(x_min, x_max):
                        pixel = image[ny, nx]

                        # 💡 计算色彩的"情感强度" - 不只是亮度，更是感受
                        if channels == 1:
                            intensity = float(pixel)
                        else:
                            intensity = float(np.mean(pixel))

                        # 🎯 量化到指定级别 - 艺术的简化智慧
                        level = min(params.levels - 1,
                                  int(intensity * params.levels / 255))

                        # ✨ 收集色彩"投票" - 民主的艺术决策
                        if channels == 1:
                            intensity_counts[level][0] += pixel
                        else:
                            intensity_counts[level] += pixel
                        intensity_nums[level] += 1

                # 🏆 选择获得最多"支持"的色彩层次
                max_count = max(intensity_nums) if intensity_nums else 0
                if max_count > 0:
                    max_index = intensity_nums.index(max_count)

                    # 🎨 计算该级别的平均色彩 - 调和的艺术
                    if channels == 1:
                        result[y, x] = intensity_counts[max_index][0] // max_count
                    else:
                        result[y, x] = intensity_counts[max_index] // max_count

        print("✅ 艺术创作完成！")
        return result

    def generate_brush_texture(self, size: Tuple[int, int],
                             brush_size: int = 10,
                             brush_density: int = 500,
                             angle: float = 45.0) -> np.ndarray:
        """
        🖌️ 生成笔刷纹理：为数字油画注入真实笔触的灵魂

        如同为画家准备专属的画笔，每一道纹理都承载着独特的表达力。
        通过随机生成的笔触线条，创造出富有质感的纹理效果。

        Args:
            size: 纹理大小 (height, width)
            brush_size: 笔刷大小
            brush_density: 笔刷密度 (笔触数量)
            angle: 笔刷角度 (度)

        Returns:
            笔刷纹理图像 (单通道灰度图)
        """
        height, width = size

        # 🎨 创建空白纹理画布 - 等待艺术的降临
        texture = np.zeros((height, width), dtype=np.uint8)

        # 🎲 艺术创作中的随机性 - 正如生活中的不可预知
        random.seed(42)  # 为了可重复的艺术效果

        # 🌀 角度转换 - 从度数到弧度的数学诗意
        radian = np.radians(angle)

        print(f"🖌️ 正在生成笔刷纹理，密度: {brush_density}, 角度: {angle}°")

        # 🖌️ 生成每一笔触 - 如同画家挥洒的激情
        for i in range(brush_density):
            if i % max(1, brush_density // 5) == 0:
                print(f"🎨 纹理生成进度: {i/brush_density*100:.1f}%")

            # 🎯 随机起点 - 艺术灵感的源泉
            start_x = random.randint(0, width - 1)
            start_y = random.randint(0, height - 1)

            # 📏 笔触的形态参数 - 每一笔都有其独特性格
            length = random.uniform(brush_size * 0.5, brush_size * 1.5)
            alpha = random.randint(30, 60)  # 透明度的艺术选择

            # 🎨 计算笔触的终点 - 方向决定命运
            end_x = int(start_x + length * np.cos(radian))
            end_y = int(start_y + length * np.sin(radian))

            # 🖌️ 确保终点在画布范围内 - 艺术的边界意识
            end_x = np.clip(end_x, 0, width - 1)
            end_y = np.clip(end_y, 0, height - 1)

            # ✨ 在画布上留下笔触 - 艺术创作的瞬间
            thickness = random.randint(1, max(1, int(brush_size * 0.3)))
            cv2.line(texture, (start_x, start_y), (end_x, end_y),
                    alpha, thickness, cv2.LINE_AA)

        # 🌫️ 高斯模糊 - 让笔触更加柔和自然
        texture = cv2.GaussianBlur(texture, (5, 5), 0)

        print("✅ 笔刷纹理生成完成！")
        return texture

    def enhanced_oil_painting(self, image: np.ndarray,
                            params: OilPaintingParams,
                            texture_strength: float = 0.5) -> np.ndarray:
        """
        ✨ 增强型油画效果：技术与艺术的完美融合

        在基础油画效果的基础上，加入笔刷纹理的质感，
        就像在已完成的画作上再次挥洒画笔，增添层次感和艺术表现力。

        Args:
            image: 输入图像
            params: 油画参数
            texture_strength: 纹理强度 (0.0-1.0)

        Returns:
            增强型油画效果图像

        Raises:
            ValueError: 当纹理强度超出有效范围时
        """
        if not 0.0 <= texture_strength <= 1.0:
            raise ValueError("🚫 纹理强度必须在0.0-1.0范围内！")

        print(f"✨ 开始创作增强型油画，纹理强度: {texture_strength:.1f}")

        # 🎨 首先应用基础油画效果 - 奠定艺术基调
        oil_painted = self.basic_oil_painting(image, params)

        if texture_strength <= 0.0:
            # 🏃‍♂️ 如果不需要纹理，直接返回基础效果
            return oil_painted

        # 🖌️ 生成独特的笔刷纹理 - 艺术个性的体现
        texture = self.generate_brush_texture(
            image.shape[:2], params.radius * 3, 500, 45.0
        )

        # 🔄 数值类型的艺术转换
        texture_float = texture.astype(np.float32) / 255.0
        oil_float = oil_painted.astype(np.float32) / 255.0

        print("🎭 正在应用笔刷纹理效果...")

        # 🎭 应用纹理效果 - 色彩与质感的艺术融合
        if len(image.shape) == 3:
            # 彩色图像处理 - 为每个色彩通道独立施加纹理魔法
            for c in range(3):
                # ✨ 纹理与色彩的平衡艺术
                # 这个公式体现了平衡的智慧：既保持原色彩，又增加纹理感
                oil_float[:, :, c] = oil_float[:, :, c] * (
                    1.0 - texture_strength + texture_strength * texture_float
                )
        else:
            # 灰度图像处理 - 单色的纹理诗意
            oil_float = oil_float * (
                1.0 - texture_strength + texture_strength * texture_float
            )

        # 🖼️ 转换回8位图像 - 从艺术理想回归现实展示
        result = np.clip(oil_float * 255, 0, 255).astype(np.uint8)

        print("✅ 增强型油画创作完成！")
        return result

    def fast_oil_painting(self, image: np.ndarray,
                         params: OilPaintingParams) -> np.ndarray:
        """
        ⚡ 快速油画效果：优化的数字艺术创作

        使用NumPy向量化操作和OpenCV优化，大幅提升处理速度。
        在保持艺术效果的同时，实现接近C++版本的性能。

        Args:
            image: 输入图像 (BGR格式或灰度图)
            params: 油画效果参数

        Returns:
            油画效果图像
        """
        if image is None or image.size == 0:
            raise ValueError("🚫 输入图像为空，艺术需要素材！")

        # 转换为灰度图进行强度计算
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        height, width = gray.shape
        result = np.zeros_like(image)

        print("⚡ 开始快速艺术创作...")
        print(f"📐 画布尺寸: {width}x{height}")

        # 使用滑动窗口方法优化
        radius = params.radius
        levels = params.levels

        # 预计算强度量化查找表
        intensity_lut = np.clip((np.arange(256) * levels) // 255, 0, levels - 1)

        # 使用积分图像优化邻域统计
        for y in range(height):
            if y % max(1, height // 10) == 0:
                print(f"⚡ 优化进度: {y/height*100:.1f}%")

            for x in range(width):
                # 定义邻域边界
                y_min = max(0, y - radius)
                y_max = min(height, y + radius + 1)
                x_min = max(0, x - radius)
                x_max = min(width, x + radius + 1)

                # 提取邻域
                neighborhood = gray[y_min:y_max, x_min:x_max]

                # 计算强度级别
                intensities = intensity_lut[neighborhood]

                # 使用bincount快速统计
                level_counts = np.bincount(intensities.flatten(), minlength=levels)

                # 找到最频繁的级别
                if np.sum(level_counts) > 0:
                    max_level = np.argmax(level_counts)

                    # 计算该级别的平均颜色
                    mask = (intensities == max_level)
                    if len(image.shape) == 3:
                        # 彩色图像
                        for c in range(3):
                            channel_neighborhood = image[y_min:y_max, x_min:x_max, c]
                            result[y, x, c] = np.mean(channel_neighborhood[mask])
                    else:
                        # 灰度图像
                        result[y, x] = np.mean(neighborhood[mask])

        print("✅ 快速艺术创作完成！")
        return result.astype(np.uint8)

    def optimized_oil_painting(self, image: np.ndarray,
                             params: OilPaintingParams) -> np.ndarray:
        """
        🚀 超快速油画效果：使用OpenCV优化的终极版本

        结合OpenCV的优化算法和NumPy的向量化操作，
        实现接近实时处理的油画效果。

        Args:
            image: 输入图像 (BGR格式或灰度图)
            params: 油画效果参数

        Returns:
            油画效果图像
        """
        if image is None or image.size == 0:
            raise ValueError("🚫 输入图像为空，艺术需要素材！")

        print("🚀 开始超快速艺术创作...")

        # 缩小图像进行处理
        scale_factor = 0.5
        small_size = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
        small_image = cv2.resize(image, small_size, interpolation=cv2.INTER_LINEAR)

        # 在小图像上应用油画效果
        small_result = self.fast_oil_painting(small_image, params)

        # 放大回原始尺寸
        result = cv2.resize(small_result, (image.shape[1], image.shape[0]),
                           interpolation=cv2.INTER_LINEAR)

        # 应用锐化滤波器增强细节
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        result = cv2.filter2D(result, -1, kernel)

        print("✅ 超快速艺术创作完成！")
        return result

    def realtime_oil_painting(self, image: np.ndarray,
                            params: OilPaintingParams) -> np.ndarray:
        """
        ⚡ 实时油画效果：速度与质量的平衡艺术

        通过降采样和后处理优化，在保持艺术效果的同时提升处理速度，
        适用于实时应用场景。

        Args:
            image: 输入图像
            params: 油画参数

        Returns:
            优化后的油画效果图像
        """
        print("⚡ 开始实时油画处理...")

        # 🎯 通过降采样提高处理速度 - 效率的艺术
        scale_factor = 0.5
        small_size = (int(image.shape[1] * scale_factor),
                     int(image.shape[0] * scale_factor))

        # 📐 缩小图像
        small_image = cv2.resize(image, small_size, interpolation=cv2.INTER_LINEAR)

        # 🎨 在小图上应用油画效果
        small_result = self.basic_oil_painting(small_image, params)

        # 📐 放大回原始尺寸
        result = cv2.resize(small_result, (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_LINEAR)

        # ✨ 锐化滤波器增强细节 - 弥补缩放带来的细节损失
        kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]], dtype=np.float32)
        result = cv2.filter2D(result, -1, kernel)

        print("✅ 实时油画处理完成！")
        return result

    def artistic_showcase(self, image: np.ndarray,
                         save_path: Optional[str] = None) -> None:
        """
        🎭 艺术效果展示：展现油画算法的无限魅力

        创建一个数字艺术画廊，展示不同参数下的艺术效果，
        让观者感受算法的创意潜力。

        Args:
            image: 输入图像
            save_path: 保存路径 (可选)
        """
        print("🎨 开始创作油画艺术作品集...")

        # 🎨 创建不同风格的艺术作品
        effects = {
            "📷 原始图像": image,
            "🌅 基础油画": self.basic_oil_painting(
                image, OilPaintingParams(radius=3, levels=8)
            ),
            "🎨 经典油画": self.basic_oil_painting(
                image, OilPaintingParams(radius=5, levels=12)
            ),
            "🖌️ 粗犷笔触": self.basic_oil_painting(
                image, OilPaintingParams(radius=8, levels=6)
            ),
            "✨ 增强纹理": self.enhanced_oil_painting(
                image, OilPaintingParams(radius=4, levels=10), 0.6
            ),
            "🌟 艺术大师": self.enhanced_oil_painting(
                image, OilPaintingParams(radius=6, levels=15), 0.8
            ),
            "🎭 现代风格": self.basic_oil_painting(
                image, OilPaintingParams(radius=4, levels=20)
            ),
            "🔥 浓烈笔触": self.enhanced_oil_painting(
                image, OilPaintingParams(radius=7, levels=8), 1.0
            )
        }

        # 🖼️ 创造艺术画廊
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('🎨 数字油画艺术画廊：像素与笔触的诗意对话',
                    fontsize=16, fontweight='bold')

        for i, (title, effect_image) in enumerate(effects.items()):
            row, col = i // 4, i % 4

            # 显示图像
            if len(effect_image.shape) == 3:
                # BGR to RGB for matplotlib
                display_image = cv2.cvtColor(effect_image, cv2.COLOR_BGR2RGB)
                axes[row, col].imshow(display_image)
            else:
                axes[row, col].imshow(effect_image, cmap='gray')

            axes[row, col].set_title(title, fontsize=11, pad=10)
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.93)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 艺术画廊已保存至: {save_path}")

        plt.show()
        print("🎨 艺术展示完成，感谢欣赏数字油画的魅力！")

    def performance_test(self, image: np.ndarray,
                        iterations: int = 3) -> Dict[str, float]:
        """
        ⚡ 性能测试：比较不同算法的处理速度

        测试基础版本、快速版本和优化版本的性能差异，
        帮助用户选择最适合的算法。

        Args:
            image: 测试图像
            iterations: 测试迭代次数

        Returns:
            性能测试结果字典
        """
        print("⚡ 开始性能测试...")

        # 创建参数对象
        params = OilPaintingParams(radius=3, levels=10)

        results = {}

        # 测试基础版本
        print("🌅 测试基础版本...")
        start_time = time.time()
        for _ in range(iterations):
            _ = self.basic_oil_painting(image, params)
        basic_time = (time.time() - start_time) / iterations
        results['basic'] = basic_time

        # 测试快速版本
        print("⚡ 测试快速版本...")
        start_time = time.time()
        for _ in range(iterations):
            _ = self.fast_oil_painting(image, params)
        fast_time = (time.time() - start_time) / iterations
        results['fast'] = fast_time

        # 测试优化版本
        print("🚀 测试优化版本...")
        start_time = time.time()
        for _ in range(iterations):
            _ = self.optimized_oil_painting(image, params)
        optimized_time = (time.time() - start_time) / iterations
        results['optimized'] = optimized_time

        # 测试实时版本
        print("⚡ 测试实时版本...")
        start_time = time.time()
        for _ in range(iterations):
            _ = self.realtime_oil_painting(image, params)
        realtime_time = (time.time() - start_time) / iterations
        results['realtime'] = realtime_time

        # 打印性能对比
        print("\n📊 性能测试结果:")
        print(f"🌅 基础版本: {basic_time:.3f}s")
        print(f"⚡ 快速版本: {fast_time:.3f}s (加速比: {basic_time/fast_time:.1f}x)")
        print(f"🚀 优化版本: {optimized_time:.3f}s (加速比: {basic_time/optimized_time:.1f}x)")
        print(f"⚡ 实时版本: {realtime_time:.3f}s (加速比: {basic_time/realtime_time:.1f}x)")

        return results
        """
        ⚡ 性能测试：评估艺术创作的效率

        测试不同算法版本的执行时间，为实际应用提供性能参考。

        Args:
            image: 测试图像
            iterations: 测试迭代次数

        Returns:
            各种方法的平均执行时间字典
        """
        print(f"⚡ 开始性能测试，图像尺寸: {image.shape}, 迭代次数: {iterations}")

        results = {}
        params = OilPaintingParams(radius=3, levels=8)

        # 测试基础油画效果
        print("🎨 测试基础油画效果...")
        start_time = time.time()
        for _ in range(iterations):
            self.basic_oil_painting(image, params)
        results['basic_oil_painting'] = (time.time() - start_time) / iterations

        # 测试增强油画效果
        print("✨ 测试增强油画效果...")
        start_time = time.time()
        for _ in range(iterations):
            self.enhanced_oil_painting(image, params, 0.5)
        results['enhanced_oil_painting'] = (time.time() - start_time) / iterations

        # 测试实时油画效果
        print("⚡ 测试实时油画效果...")
        start_time = time.time()
        for _ in range(iterations):
            self.realtime_oil_painting(image, params)
        results['realtime_oil_painting'] = (time.time() - start_time) / iterations

        print("\n📊 性能测试结果:")
        for method, time_cost in results.items():
            print(f"   {method}: {time_cost:.4f}s")

        return results

    def interactive_demo(self, image: np.ndarray) -> None:
        """
        🎮 交互式演示：探索参数对艺术效果的影响

        通过交互式界面，让用户直观地理解不同参数对油画效果的影响。

        Args:
            image: 输入图像
        """
        print("🎮 欢迎来到油画效果交互式演示！")
        print("💡 提示：输入不同的参数值来观察艺术效果的变化")

        while True:
            try:
                print("\n" + "="*50)
                print("🎨 油画参数设置")
                print("="*50)

                # 获取用户输入
                radius = int(input("🔍 邻域半径 (1-15, 推荐3-8): ") or "3")
                levels = int(input("🌈 色彩级别 (2-30, 推荐8-15): ") or "8")
                texture_strength = float(input("🖌️ 纹理强度 (0.0-1.0, 推荐0.5): ") or "0.5")

                # 创建参数对象
                params = OilPaintingParams(radius=radius, levels=levels)

                # 生成效果
                print("\n🎨 正在创作艺术作品...")
                if texture_strength > 0:
                    result = self.enhanced_oil_painting(image, params, texture_strength)
                else:
                    result = self.basic_oil_painting(image, params)

                # 显示结果
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # 原图
                if len(image.shape) == 3:
                    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                else:
                    ax1.imshow(image, cmap='gray')
                ax1.set_title('📷 原始图像', fontsize=12)
                ax1.axis('off')

                # 效果图
                if len(result.shape) == 3:
                    ax2.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                else:
                    ax2.imshow(result, cmap='gray')
                ax2.set_title(f'🎨 油画效果 (半径:{radius}, 级别:{levels}, 纹理:{texture_strength})',
                            fontsize=12)
                ax2.axis('off')

                plt.tight_layout()
                plt.show()

                # 询问是否继续
                continue_demo = input("\n🤔 是否继续尝试其他参数？(y/n): ").lower()
                if continue_demo != 'y':
                    break

            except KeyboardInterrupt:
                print("\n👋 感谢体验油画效果演示！")
                break
            except Exception as e:
                print(f"⚠️ 参数错误: {e}")
                print("请输入有效的参数值！")

        print("🎨 演示结束，期待您创作出更多精彩的数字艺术作品！")


def main():
    """
    🚀 主函数：命令行界面和使用示例

    提供完整的命令行接口，支持多种功能模式。
    """
    parser = argparse.ArgumentParser(
        description="🎨 数字油画艺术家 - 让像素学会绘画的魔法",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基础版本 (较慢但效果最好)
  python oil_painting_effect.py input.jpg --output output.jpg --radius 5 --levels 12

  # 快速版本 (推荐日常使用)
  python oil_painting_effect.py input.jpg --mode fast --output output.jpg

  # 优化版本 (最快，适合实时应用)
  python oil_painting_effect.py input.jpg --mode optimized --output output.jpg

  # 性能测试
  python oil_painting_effect.py input.jpg --mode performance

  # 交互式演示
  python oil_painting_effect.py input.jpg --mode interactive
        """
    )

    parser.add_argument('input', help='📁 输入图像路径')
    parser.add_argument('--output', '-o', help='💾 输出图像路径')
    parser.add_argument('--mode', choices=['basic', 'fast', 'optimized', 'enhanced', 'realtime', 'showcase', 'interactive', 'performance'],
                       default='basic', help='🎭 处理模式')
    parser.add_argument('--radius', type=int, default=3, help='🔍 邻域半径 (默认: 3)')
    parser.add_argument('--levels', type=int, default=10, help='🌈 色彩级别 (默认: 10)')
    parser.add_argument('--texture-strength', type=float, default=0.5,
                       help='🖌️ 纹理强度 (默认: 0.5)')
    parser.add_argument('--showcase-output', help='🖼️ 艺术展示输出路径')

    args = parser.parse_args()

    # 🎨 初始化数字艺术家
    artist = OilPaintingArtist()

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

    # 创建参数对象
    params = OilPaintingParams(radius=args.radius, levels=args.levels)

    try:
        # 🎭 根据模式执行不同功能
        if args.mode == 'basic':
            print("🌅 执行基础油画效果...")
            result = artist.basic_oil_painting(image, params)

        elif args.mode == 'fast':
            print("⚡ 执行快速油画效果...")
            result = artist.fast_oil_painting(image, params)

        elif args.mode == 'optimized':
            print("🚀 执行优化油画效果...")
            result = artist.optimized_oil_painting(image, params)

        elif args.mode == 'enhanced':
            print("✨ 执行增强油画效果...")
            result = artist.enhanced_oil_painting(image, params, args.texture_strength)

        elif args.mode == 'realtime':
            print("⚡ 执行实时油画效果...")
            result = artist.realtime_oil_painting(image, params)

        elif args.mode == 'showcase':
            print("🎭 启动艺术展示模式...")
            artist.artistic_showcase(image, args.showcase_output)
            return

        elif args.mode == 'interactive':
            print("🎮 启动交互式演示模式...")
            artist.interactive_demo(image)
            return

        elif args.mode == 'performance':
            print("⚡ 启动性能测试模式...")
            artist.performance_test(image)
            return

        # 💾 保存结果
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            success = cv2.imwrite(str(output_path), result)
            if success:
                print(f"💾 艺术作品已保存至: {args.output}")
            else:
                print("❌ 保存失败，请检查输出路径！")
        else:
            # 显示结果
            cv2.imshow('🎨 油画效果', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except Exception as e:
        print(f"❌ 处理过程中发生错误: {e}")
        print("💡 请检查输入参数和图像文件！")


if __name__ == "__main__":
    print("🎨" + "="*60)
    print("    数字油画艺术家 - 让像素学会绘画的魔法")
    print("    GlimmerLab-IP101 - 技术与艺术的完美融合")
    print("="*64)
    main()