"""
🎭 现实的卡通化：将真实世界变成动画的魔法算法

这个模块实现了完整的卡通效果算法，包含：
- 基础卡通化：简洁的动画风格转换
- 增强卡通化：带纹理细节的高级效果
- 多层次卡通化：景深效果的艺术表达
- 参数分析：理解每个参数的艺术影响
- 交互式演示：实时体验卡通化魔法

作者：GlimmerLab
创建时间：2024年
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
import math
import argparse
import os
from pathlib import Path
import time

@dataclass
class CartoonParams:
    """🎨 卡通效果的艺术配置参数"""
    edge_size: int = 1                  # 边缘宽度
    median_blur_size: int = 7           # 中值滤波核大小
    bilateral_d: int = 9                # 双边滤波d参数
    bilateral_sigma_color: float = 75.0 # 双边滤波颜色标准差
    bilateral_sigma_space: float = 75.0 # 双边滤波空间标准差
    quantize_levels: int = 8            # 颜色量化级别

    def __post_init__(self):
        """参数有效性检查"""
        assert self.edge_size >= 1, "边缘大小必须 >= 1"
        assert self.median_blur_size >= 3 and self.median_blur_size % 2 == 1, "中值滤波核必须是大于等于3的奇数"
        assert self.bilateral_d > 0, "双边滤波d必须 > 0"
        assert self.bilateral_sigma_color > 0, "双边滤波颜色标准差必须 > 0"
        assert self.bilateral_sigma_space > 0, "双边滤波空间标准差必须 > 0"
        assert self.quantize_levels >= 2, "量化级别必须 >= 2"

class CartoonArtist:
    """🎭 卡通艺术家：用算法的画笔创造二次元世界"""

    def __init__(self, params: Optional[CartoonParams] = None):
        """
        🌟 初始化我们的卡通艺术家
        每个参数都是创作动画世界的魔法咒语
        """
        self.params = params or CartoonParams()

    def detect_edges(self, image: np.ndarray, edge_size: int = 1) -> np.ndarray:
        """
        🖌️ 边缘觉醒：用线条勾勒世界的灵魂轮廓

        就像动画师用铅笔勾勒角色的第一笔，边缘是所有美好的开始

        Args:
            image: 等待转换的现实世界
            edge_size: 线条的粗细，决定边缘的性格

        Returns:
            世界的轮廓图，黑白分明的真理
        """
        # 🎨 将彩色的复杂世界转换为纯净的灰度诗篇
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 💭 用中值滤波去除生活中的噪音和杂念
        blurred = cv2.medianBlur(gray, 5)

        # ⚡ 自适应阈值：让每个区域都能表达自己的个性
        edges = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 9, 2
        )

        # 🖼️ 扩张边缘，让每条线都更加清晰有力
        if edge_size > 1:
            kernel = np.ones((edge_size, edge_size), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

        return edges

    def color_quantization(self, image: np.ndarray, levels: int = 8) -> np.ndarray:
        """
        🌈 色彩的诗意简化：将无穷的颜色归纳为有限的美

        就像画家调色板上的几种主色，能创造出整个世界的情感

        Args:
            image: 输入图像
            levels: 量化级别，决定色彩的丰富程度

        Returns:
            简化后的色彩世界，每种颜色都有自己的故事
        """
        # 🎭 计算量化的"情感强度"
        factor = 255.0 / levels

        # 🌟 为每个像素找到它的"精神归属"
        quantized = np.round(image / factor) * factor + factor / 2
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)

        return quantized

    def bilateral_smooth(self, image: np.ndarray,
                        d: int = 9,
                        sigma_color: float = 75.0,
                        sigma_space: float = 75.0) -> np.ndarray:
        """
        ✨ 双边滤波：智慧的平滑艺术

        在保持重要特征的同时，抚平表面的粗糙，如同岁月给予的智慧

        Args:
            image: 输入图像
            d: 滤波直径
            sigma_color: 颜色标准差
            sigma_space: 空间标准差

        Returns:
            平滑但保留本质的图像
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    def basic_cartoon_effect(self, image: np.ndarray,
                            params: Optional[CartoonParams] = None) -> np.ndarray:
        """
        🎪 基础卡通魔法：将现实世界变成动画片

        这是算法的核心魔法，每一步都是从现实到梦想的跨越

        Args:
            image: 待转换的现实图像
            params: 卡通化参数配置

        Returns:
            充满想象力的卡通世界
        """
        p = params or self.params

        if len(image.shape) != 3:
            raise ValueError("🚫 请提供彩色图像，就像生活需要色彩一样")

        # 1. 🔍 发现世界的轮廓——每个重要的边界
        edges = self.detect_edges(image, p.edge_size)

        # 2. 💫 用中值滤波抚平生活的粗糙纹理
        smoothed = cv2.medianBlur(image, p.median_blur_size)

        # 3. 🌸 双边滤波：在保持个性的同时创造和谐
        bilateral = self.bilateral_smooth(
            smoothed, p.bilateral_d,
            p.bilateral_sigma_color, p.bilateral_sigma_space
        )

        # 4. 🎨 颜色的诗意简化
        quantized = self.color_quantization(bilateral, p.quantize_levels)

        # 5. 🖼️ 将线条与色彩完美融合——艺术的最高境界
        cartoon = quantized.copy()
        cartoon[edges > 0] = [0, 0, 0]  # 黑色边缘，如人生的重要时刻

        return cartoon

    def enhanced_cartoon_effect(self, image: np.ndarray,
                               params: Optional[CartoonParams] = None,
                               texture_strength: float = 0.5) -> np.ndarray:
        """
        🎵 高级卡通魔法：为平面世界增添生命的质感

        在基础卡通效果上增加纹理细节，就像为动画注入灵魂

        Args:
            image: 输入图像
            params: 卡通化参数
            texture_strength: 纹理增强强度 [0, 1]

        Returns:
            有生命力的卡通图像
        """
        # 1. 🎭 先创造基础的卡通世界
        basic_cartoon = self.basic_cartoon_effect(image, params)

        if texture_strength <= 0.0:
            return basic_cartoon

        # 2. 🔍 提取生命的纹理细节
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 🌀 使用高斯差分捕捉不同尺度的"生命力"
        blur1 = cv2.GaussianBlur(gray, (3, 3), 1.0)
        blur2 = cv2.GaussianBlur(gray, (9, 9), 3.0)
        dog = blur1.astype(np.float32) - blur2.astype(np.float32)

        # 3. 🎨 将纹理与卡通世界温柔融合
        # 归一化纹理
        texture = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
        texture = texture.astype(np.float32) * texture_strength

        # 转换为三通道
        texture_rgb = np.stack([texture] * 3, axis=-1)

        # 4. ✨ 创造有生命力的卡通世界
        cartoon_float = basic_cartoon.astype(np.float32)
        enhanced = cartoon_float * (1.0 + texture_rgb / 255.0)

        # 保持边缘的纯黑色
        p = params or self.params
        edges = self.detect_edges(image, p.edge_size)
        enhanced[edges > 0] = [0, 0, 0]

        return np.clip(enhanced, 0, 255).astype(np.uint8)

    def multi_level_cartoon(self, image: np.ndarray,
                           levels: List[int] = [4, 8, 16],
                           weights: Optional[List[float]] = None) -> np.ndarray:
        """
        🌀 多层次卡通化：如同动画中的景深效果

        在不同的量化级别上创造卡通效果，然后智慧地融合

        Args:
            image: 输入图像
            levels: 不同的量化级别列表
            weights: 各级别的权重，如果为None则使用均匀权重

        Returns:
            多层次融合的卡通图像
        """
        if not levels:
            return self.basic_cartoon_effect(image)

        if weights is None:
            weights = [1.0 / len(levels)] * len(levels)
        elif len(weights) != len(levels):
            raise ValueError("权重数量必须与级别数量相匹配")

        # 归一化权重
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        results = []
        for level in levels:
            params = CartoonParams(quantize_levels=level)
            cartoon = self.basic_cartoon_effect(image, params)
            results.append(cartoon.astype(np.float32))

        # 🎭 智慧的权重融合
        final_result = np.zeros_like(results[0])

        for result, weight in zip(results, weights):
            final_result += weight * result

        return np.clip(final_result, 0, 255).astype(np.uint8)

    def stylized_cartoon_effect(self, image: np.ndarray,
                               style: str = "anime",
                               intensity: float = 1.0) -> np.ndarray:
        """
        🎨 风格化卡通效果：不同动画风格的艺术表达

        Args:
            image: 输入图像
            style: 卡通风格 ("anime", "western", "vintage", "minimal")
            intensity: 效果强度 [0, 2]

        Returns:
            特定风格的卡通图像
        """
        intensity = np.clip(intensity, 0.0, 2.0)

        if style == "anime":
            # 日式动画风格：高对比度，清晰边缘
            params = CartoonParams(
                quantize_levels=int(6 * intensity + 2),
                edge_size=max(1, int(2 * intensity)),
                bilateral_sigma_color=50.0 * intensity,
                bilateral_sigma_space=50.0 * intensity
            )
            return self.enhanced_cartoon_effect(image, params, 0.3 * intensity)

        elif style == "western":
            # 西式卡通风格：柔和边缘，丰富色彩
            params = CartoonParams(
                quantize_levels=int(12 * intensity + 4),
                edge_size=max(1, int(3 * intensity)),
                bilateral_sigma_color=100.0 * intensity,
                bilateral_sigma_space=100.0 * intensity
            )
            return self.basic_cartoon_effect(image, params)

        elif style == "vintage":
            # 复古卡通风格：温暖色调，适度纹理
            params = CartoonParams(
                quantize_levels=int(8 * intensity + 2),
                edge_size=max(1, int(1 * intensity)),
                bilateral_sigma_color=80.0 * intensity,
                bilateral_sigma_space=80.0 * intensity
            )
            result = self.enhanced_cartoon_effect(image, params, 0.5 * intensity)

            # 添加温暖滤镜
            warm_filter = np.array([[[0.95, 1.0, 1.05]]], dtype=np.float32)
            result = result.astype(np.float32) * warm_filter
            return np.clip(result, 0, 255).astype(np.uint8)

        elif style == "minimal":
            # 极简风格：少量颜色，粗边缘
            params = CartoonParams(
                quantize_levels=max(2, int(4 * intensity)),
                edge_size=max(1, int(4 * intensity)),
                bilateral_sigma_color=30.0 * intensity,
                bilateral_sigma_space=30.0 * intensity
            )
            return self.basic_cartoon_effect(image, params)

        else:
            raise ValueError(f"未知的卡通风格: {style}")

    def artistic_showcase(self, image: np.ndarray, save_path: Optional[str] = None) -> None:
        """
        🎭 卡通艺术展示：展现算法的多种创作风格

        如同动画工作室的概念稿展览，展示卡通化的无限可能
        """
        print("🎨 开始创作卡通艺术作品...")

        # 🎨 创建不同风格的卡通作品
        effects = {
            "📷 现实世界": image,
            "🎭 基础卡通": self.basic_cartoon_effect(image),
            "✨ 纹理增强": self.enhanced_cartoon_effect(image, texture_strength=0.3),
            "🌟 高纹理": self.enhanced_cartoon_effect(image, texture_strength=0.7),
            "🎨 简洁风格": self.basic_cartoon_effect(image, CartoonParams(quantize_levels=4)),
            "🌈 丰富色彩": self.basic_cartoon_effect(image, CartoonParams(quantize_levels=16)),
            "🖌️ 粗线条": self.basic_cartoon_effect(image, CartoonParams(edge_size=3)),
            "🌸 多层次": self.multi_level_cartoon(image)
        }

        # 🖼️ 创建动画艺术画廊
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('🎭 卡通化艺术馆：现实与梦想的桥梁', fontsize=16, fontweight='bold')

        for i, (title, effect_image) in enumerate(effects.items()):
            row, col = i // 4, i % 4
            axes[row, col].imshow(cv2.cvtColor(effect_image, cv2.COLOR_BGR2RGB))
            axes[row, col].set_title(title, fontsize=11)
            axes[row, col].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 卡通艺术展示已保存至: {save_path}")

        plt.show()

    def style_comparison(self, image: np.ndarray) -> None:
        """
        🎪 风格对比展示：不同动画风格的艺术表达
        """
        print("🎪 展示不同的卡通风格...")

        styles = {
            "📷 原图": image,
            "🌸 日式动画": self.stylized_cartoon_effect(image, "anime", 1.0),
            "🎭 西式卡通": self.stylized_cartoon_effect(image, "western", 1.0),
            "📼 复古风格": self.stylized_cartoon_effect(image, "vintage", 1.0),
            "⚪ 极简风格": self.stylized_cartoon_effect(image, "minimal", 1.0),
            "🔥 强化日式": self.stylized_cartoon_effect(image, "anime", 1.5),
            "💫 柔和西式": self.stylized_cartoon_effect(image, "western", 0.7),
            "🌟 复古增强": self.stylized_cartoon_effect(image, "vintage", 1.3)
        }

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('🎪 卡通风格对比：多样化的艺术表达', fontsize=16, fontweight='bold')

        for i, (title, style_image) in enumerate(styles.items()):
            row, col = i // 4, i % 4
            axes[row, col].imshow(cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB))
            axes[row, col].set_title(title, fontsize=11)
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.show()

    def parameter_analysis(self, image: np.ndarray) -> None:
        """
        🔍 参数影响分析：理解每个参数如何改变艺术表达
        """
        print("🔍 分析参数对卡通效果的影响...")

        # 测试不同的量化级别
        levels = [3, 6, 12, 20]
        level_results = []

        for level in levels:
            params = CartoonParams(quantize_levels=level)
            result = self.basic_cartoon_effect(image, params)
            level_results.append((f"量化级别: {level}", result))

        # 测试不同的边缘大小
        edge_sizes = [1, 2, 4, 6]
        edge_results = []

        for edge_size in edge_sizes:
            params = CartoonParams(edge_size=edge_size)
            result = self.basic_cartoon_effect(image, params)
            edge_results.append((f"边缘大小: {edge_size}", result))

        # 可视化对比
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('🔍 参数影响分析', fontsize=14, fontweight='bold')

        # 显示量化级别影响
        for i, (title, img) in enumerate(level_results):
            axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[0, i].set_title(title, fontsize=10)
            axes[0, i].axis('off')

        # 显示边缘大小影响
        for i, (title, img) in enumerate(edge_results):
            axes[1, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[1, i].set_title(title, fontsize=10)
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.show()

    def interactive_cartoon_effect(self, image: np.ndarray) -> None:
        """
        🎮 交互式卡通效果：实时调整参数体验卡通魔法

        Args:
            image: 输入图像
        """
        try:
            from matplotlib.widgets import Slider, RadioButtons
        except ImportError:
            print("❌ 需要matplotlib.widgets模块进行交互式演示")
            return

        fig = plt.figure(figsize=(16, 10))

        # 创建子图布局
        ax_original = plt.subplot2grid((4, 4), (0, 0), rowspan=2, colspan=2)
        ax_result = plt.subplot2grid((4, 4), (0, 2), rowspan=2, colspan=2)
        ax_style = plt.subplot2grid((4, 4), (2, 0), rowspan=1, colspan=1)

        # 显示原图
        ax_original.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax_original.set_title('📷 原始图像', fontsize=12)
        ax_original.axis('off')

        # 初始效果
        initial_result = self.basic_cartoon_effect(image)
        im_result = ax_result.imshow(cv2.cvtColor(initial_result, cv2.COLOR_BGR2RGB))
        ax_result.set_title('🎭 卡通效果', fontsize=12)
        ax_result.axis('off')

        # 风格选择
        styles = ['基础', '日式', '西式', '复古', '极简']
        radio = RadioButtons(ax_style, styles, active=0)
        ax_style.set_title('🎨 风格选择')

        # 创建滑块
        ax_quantize = plt.axes([0.15, 0.25, 0.25, 0.03])
        ax_edge = plt.axes([0.55, 0.25, 0.25, 0.03])
        ax_texture = plt.axes([0.15, 0.2, 0.25, 0.03])
        ax_intensity = plt.axes([0.55, 0.2, 0.25, 0.03])

        slider_quantize = Slider(ax_quantize, '量化级别', 2, 20, valinit=8, valfmt='%d')
        slider_edge = Slider(ax_edge, '边缘大小', 1, 8, valinit=1, valfmt='%d')
        slider_texture = Slider(ax_texture, '纹理强度', 0.0, 1.0, valinit=0.0)
        slider_intensity = Slider(ax_intensity, '风格强度', 0.1, 2.0, valinit=1.0)

        def update(_):
            """更新卡通效果"""
            style = radio.value_selected
            quantize_level = int(slider_quantize.val)
            edge_size = int(slider_edge.val)
            texture_strength = slider_texture.val
            intensity = slider_intensity.val

            if style == '基础':
                params = CartoonParams(quantize_levels=quantize_level, edge_size=edge_size)
                if texture_strength > 0:
                    result = self.enhanced_cartoon_effect(image, params, texture_strength)
                else:
                    result = self.basic_cartoon_effect(image, params)
            elif style == '日式':
                result = self.stylized_cartoon_effect(image, "anime", intensity)
            elif style == '西式':
                result = self.stylized_cartoon_effect(image, "western", intensity)
            elif style == '复古':
                result = self.stylized_cartoon_effect(image, "vintage", intensity)
            elif style == '极简':
                result = self.stylized_cartoon_effect(image, "minimal", intensity)

            im_result.set_data(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            ax_result.set_title(f'🎭 {style}卡通效果')
            fig.canvas.draw()

        # 绑定事件
        slider_quantize.on_changed(update)
        slider_edge.on_changed(update)
        slider_texture.on_changed(update)
        slider_intensity.on_changed(update)
        radio.on_clicked(update)

        plt.tight_layout()
        plt.show()

    def performance_test(self, image_sizes: List[Tuple[int, int]] = None) -> Dict[str, float]:
        """
        ⚡ 性能测试：评估不同卡通化方法的处理速度

        Args:
            image_sizes: 测试的图像尺寸列表

        Returns:
            性能测试结果字典
        """
        if image_sizes is None:
            image_sizes = [(256, 256), (512, 512), (1024, 1024)]

        results = {}

        print("🚀 开始卡通化性能测试...")
        print("=" * 60)

        for width, height in image_sizes:
            # 创建测试图像
            test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

            # 测试不同方法
            methods = {
                '基础卡通': lambda img: self.basic_cartoon_effect(img),
                '增强卡通': lambda img: self.enhanced_cartoon_effect(img, texture_strength=0.5),
                '多层次卡通': lambda img: self.multi_level_cartoon(img),
                '日式风格': lambda img: self.stylized_cartoon_effect(img, "anime")
            }

            print(f"📊 图像尺寸: {width}x{height}")

            for method_name, method_func in methods.items():
                start_time = time.time()
                _ = method_func(test_image)
                processing_time = time.time() - start_time

                key = f"{method_name}_{width}x{height}"
                results[key] = processing_time

                print(f"  🎭 {method_name}: {processing_time:.3f}秒")

            print("-" * 40)

        print("✅ 性能测试完成")
        return results

def create_cartoon_effect_demo():
    """🎯 创建卡通效果演示程序"""

    def process_image_interactive():
        """交互式图像处理"""
        while True:
            print("\n" + "="*60)
            print("🎭 卡通艺术家 - 交互式演示")
            print("="*60)

            # 获取图像路径
            image_path = input("📷 请输入图像路径 (或输入 'q' 退出): ").strip()
            if image_path.lower() == 'q':
                break

            if not os.path.exists(image_path):
                print("❌ 文件不存在，请检查路径")
                continue

            # 加载图像
            image = cv2.imread(image_path)
            if image is None:
                print("❌ 无法读取图像文件")
                continue

            print(f"✅ 成功加载图像: {image.shape}")

            # 创建卡通艺术家
            artist = CartoonArtist()

            print("\n🎨 请选择卡通效果:")
            print("1. 🎭 基础卡通效果")
            print("2. ✨ 增强卡通效果")
            print("3. 🌀 多层次卡通")
            print("4. 🌸 日式动画风格")
            print("5. 🎪 西式卡通风格")
            print("6. 📼 复古卡通风格")
            print("7. ⚪ 极简卡通风格")
            print("8. 🎨 艺术展示")
            print("9. 🎪 风格对比")
            print("10. 🔍 参数分析")
            print("11. 🎮 交互式调节")

            choice = input("请选择 (1-11): ").strip()

            try:
                if choice == '1':
                    levels = int(input("🎭 量化级别 [2-20, 默认8]: ") or "8")
                    edge_size = int(input("🎭 边缘大小 [1-8, 默认1]: ") or "1")
                    params = CartoonParams(quantize_levels=levels, edge_size=edge_size)
                    result = artist.basic_cartoon_effect(image, params)
                elif choice == '2':
                    texture = float(input("✨ 纹理强度 [0-1, 默认0.5]: ") or "0.5")
                    result = artist.enhanced_cartoon_effect(image, texture_strength=texture)
                elif choice == '3':
                    result = artist.multi_level_cartoon(image)
                elif choice == '4':
                    intensity = float(input("🌸 风格强度 [0.1-2.0, 默认1.0]: ") or "1.0")
                    result = artist.stylized_cartoon_effect(image, "anime", intensity)
                elif choice == '5':
                    intensity = float(input("🎪 风格强度 [0.1-2.0, 默认1.0]: ") or "1.0")
                    result = artist.stylized_cartoon_effect(image, "western", intensity)
                elif choice == '6':
                    intensity = float(input("📼 风格强度 [0.1-2.0, 默认1.0]: ") or "1.0")
                    result = artist.stylized_cartoon_effect(image, "vintage", intensity)
                elif choice == '7':
                    intensity = float(input("⚪ 风格强度 [0.1-2.0, 默认1.0]: ") or "1.0")
                    result = artist.stylized_cartoon_effect(image, "minimal", intensity)
                elif choice == '8':
                    artist.artistic_showcase(image)
                    continue
                elif choice == '9':
                    artist.style_comparison(image)
                    continue
                elif choice == '10':
                    artist.parameter_analysis(image)
                    continue
                elif choice == '11':
                    artist.interactive_cartoon_effect(image)
                    continue
                else:
                    print("❌ 无效选择")
                    continue

                # 显示结果
                comparison = np.hstack([image, result])
                cv2.imshow("Cartoon Effect (Original | Cartoon)", comparison)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # 询问是否保存
                save_choice = input("\n💾 是否保存结果? (y/n): ").strip().lower()
                if save_choice == 'y':
                    output_path = input("📁 输入保存路径 (默认: cartoon_result.jpg): ").strip() or "cartoon_result.jpg"
                    cv2.imwrite(output_path, result)
                    print(f"✅ 结果已保存至: {output_path}")

            except ValueError:
                print("❌ 参数格式错误")
            except Exception as e:
                print(f"❌ 处理出错: {e}")

    def batch_process_demo():
        """批量处理演示"""
        print("\n" + "="*60)
        print("🚀 批量卡通化处理演示")
        print("="*60)

        input_dir = input("📁 输入图像目录路径: ").strip()
        if not os.path.exists(input_dir):
            print("❌ 目录不存在")
            return

        output_dir = input("📁 输出目录路径: ").strip() or "cartoon_results"
        os.makedirs(output_dir, exist_ok=True)

        # 选择风格
        print("\n🎨 选择卡通风格:")
        print("1. 基础卡通")
        print("2. 日式动画")
        print("3. 西式卡通")
        print("4. 复古风格")
        print("5. 极简风格")

        style_choice = input("请选择 (1-5): ").strip()
        style_map = {
            '1': ('basic', None),
            '2': ('anime', 'anime'),
            '3': ('western', 'western'),
            '4': ('vintage', 'vintage'),
            '5': ('minimal', 'minimal')
        }

        if style_choice not in style_map:
            print("❌ 无效选择")
            return

        method_name, style_name = style_map[style_choice]

        # 获取图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = [f for f in os.listdir(input_dir)
                      if Path(f).suffix.lower() in image_extensions]

        if not image_files:
            print("❌ 未找到图像文件")
            return

        print(f"📸 找到 {len(image_files)} 张图像")

        # 创建卡通艺术家
        artist = CartoonArtist()

        # 批量处理
        for i, filename in enumerate(image_files):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"cartoon_{filename}")

            print(f"🎨 处理 ({i+1}/{len(image_files)}): {filename}")

            image = cv2.imread(input_path)
            if image is not None:
                if method_name == 'basic':
                    result = artist.basic_cartoon_effect(image)
                else:
                    result = artist.stylized_cartoon_effect(image, style_name, 1.0)

                cv2.imwrite(output_path, result)
                print(f"✅ 已保存: {output_path}")
            else:
                print(f"❌ 无法读取: {filename}")

        print(f"\n🎉 批量处理完成！结果保存在: {output_dir}")

    # 主菜单
    while True:
        print("\n" + "="*70)
        print("🎭 卡通艺术家 - 现实与梦想的桥梁")
        print("="*70)
        print("1. 📷 交互式单图处理")
        print("2. 🚀 批量图像处理")
        print("3. 🎨 艺术效果展示")
        print("4. 🎪 风格对比展示")
        print("5. 🎮 交互式参数调节")
        print("6. 📊 性能测试")
        print("7. 🔍 参数影响分析")
        print("0. 👋 退出程序")
        print("="*70)

        choice = input("请选择功能 (0-7): ").strip()

        if choice == '0':
            print("👋 感谢体验卡通艺术家！")
            print("愿你的世界如动画般充满想象力！ ✨")
            break
        elif choice == '1':
            process_image_interactive()
        elif choice == '2':
            batch_process_demo()
        elif choice == '3':
            image_path = input("📷 请输入测试图像路径: ").strip()
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                if image is not None:
                    artist = CartoonArtist()
                    artist.artistic_showcase(image)
                else:
                    print("❌ 无法读取图像")
            else:
                print("❌ 文件不存在")
        elif choice == '4':
            image_path = input("📷 请输入图像路径: ").strip()
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                if image is not None:
                    artist = CartoonArtist()
                    artist.style_comparison(image)
                else:
                    print("❌ 无法读取图像")
            else:
                print("❌ 文件不存在")
        elif choice == '5':
            image_path = input("📷 请输入图像路径: ").strip()
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                if image is not None:
                    artist = CartoonArtist()
                    artist.interactive_cartoon_effect(image)
                else:
                    print("❌ 无法读取图像")
            else:
                print("❌ 文件不存在")
        elif choice == '6':
            artist = CartoonArtist()
            artist.performance_test()
        elif choice == '7':
            image_path = input("📷 请输入图像路径: ").strip()
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                if image is not None:
                    artist = CartoonArtist()
                    artist.parameter_analysis(image)
                else:
                    print("❌ 无法读取图像")
            else:
                print("❌ 文件不存在")
        else:
            print("❌ 无效选择，请重新输入")

def main():
    """🌟 主函数：展示卡通化的魔法魅力"""
    parser = argparse.ArgumentParser(description="🎭 卡通效果 - 现实与梦想的桥梁")
    parser.add_argument("--input", "-i", type=str, help="输入图像路径")
    parser.add_argument("--output", "-o", type=str, help="输出图像路径")
    parser.add_argument("--style", "-s", type=str, default="basic",
                       choices=["basic", "anime", "western", "vintage", "minimal"],
                       help="卡通风格")
    parser.add_argument("--quantize", "-q", type=int, default=8, help="量化级别 (2-20)")
    parser.add_argument("--edge-size", "-e", type=int, default=1, help="边缘大小 (1-8)")
    parser.add_argument("--texture", "-t", type=float, default=0.0, help="纹理强度 (0-1)")
    parser.add_argument("--intensity", type=float, default=1.0, help="风格强度 (0.1-2.0)")
    parser.add_argument("--demo", action="store_true", help="启动演示模式")
    parser.add_argument("--showcase", action="store_true", help="显示艺术展示")
    parser.add_argument("--styles", action="store_true", help="显示风格对比")
    parser.add_argument("--interactive", action="store_true", help="交互式参数调节")
    parser.add_argument("--analysis", action="store_true", help="参数影响分析")
    parser.add_argument("--performance", action="store_true", help="运行性能测试")

    args = parser.parse_args()

    if args.demo:
        create_cartoon_effect_demo()
        return

    if not args.input:
        print("🚫 请提供输入图像路径，或使用 --demo 启动演示模式")
        print("💡 使用示例: python cartoon_effect.py -i image.jpg -o result.jpg")
        print("💡 演示模式: python cartoon_effect.py --demo")
        return

    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        return

    # 加载图像
    image = cv2.imread(args.input)
    if image is None:
        print(f"❌ 无法读取图像: {args.input}")
        return

    print(f"✅ 成功加载图像: {image.shape}")

    # 创建卡通艺术家
    artist = CartoonArtist()

    if args.performance:
        # 性能测试
        artist.performance_test()
        return

    if args.showcase:
        # 艺术展示
        save_path = args.output.replace('.jpg', '_showcase.png') if args.output else None
        artist.artistic_showcase(image, save_path)
        return

    if args.styles:
        # 风格对比
        artist.style_comparison(image)
        return

    if args.interactive:
        # 交互式调节
        artist.interactive_cartoon_effect(image)
        return

    if args.analysis:
        # 参数分析
        artist.parameter_analysis(image)
        return

    # 应用指定的卡通效果
    print(f"🎨 应用{args.style}风格卡通效果...")

    if args.style == "basic":
        params = CartoonParams(quantize_levels=args.quantize, edge_size=args.edge_size)
        if args.texture > 0:
            result = artist.enhanced_cartoon_effect(image, params, args.texture)
        else:
            result = artist.basic_cartoon_effect(image, params)
    else:
        result = artist.stylized_cartoon_effect(image, args.style, args.intensity)

    if args.output:
        cv2.imwrite(args.output, result)
        print(f"✅ 卡通艺术作品已保存至: {args.output}")
    else:
        # 显示对比
        comparison = np.hstack([image, result])
        cv2.imshow("Cartoon Effect (Original | Cartoon)", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()