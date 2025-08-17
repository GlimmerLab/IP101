#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌟 自适应对数映射算法：数字世界的智慧眼镜
=====================================

🎨 这是一个将数学之美与视觉艺术完美融合的算法实现
每一行代码都承载着对光影平衡的深度理解

作者: GlimmerLab 视觉算法实验室
项目: IP101 - 图像处理算法集
描述: 自适应对数映射的诗意实现
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import time
from typing import Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from matplotlib.patches import Rectangle
import seaborn as sns

# 设置中文字体和美观的样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

@dataclass
class AdaptiveLogParams:
    """
    🌟 自适应对数映射的艺术参数集合

    每个参数都如音符般精心调校，共同谱写视觉的交响乐
    """
    bias: float = 0.85                 # 偏置参数：生活的基调，决定整体亮度倾向
    max_scale: float = 100.0           # 最大缩放因子：人生的格局，控制对比度幅度
    local_adaptation: bool = False     # 局部自适应：因地制宜的智慧开关
    window_ratio: float = 0.125        # 窗口比例：观察世界的视野大小
    tone_curve_strength: float = 0.3   # 色调曲线强度：S型曲线的优雅程度

    def __post_init__(self):
        """参数验证：确保每个参数都在合理范围内"""
        assert 0.0 <= self.bias <= 2.0, "🚫 偏置参数应在[0, 2]范围内"
        assert 1.0 <= self.max_scale <= 500.0, "🚫 最大缩放因子应在[1, 500]范围内"
        assert 0.01 <= self.window_ratio <= 0.5, "🚫 窗口比例应在[0.01, 0.5]范围内"
        assert 0.0 <= self.tone_curve_strength <= 1.0, "🚫 色调曲线强度应在[0, 1]范围内"

class AdaptiveLogArtist:
    """
    🎨 自适应对数映射艺术家

    如同经验丰富的调色师，根据每张画布的特点精心调制色彩
    在数学的精确与艺术的感性之间寻找完美的平衡点
    """

    def __init__(self, params: Optional[AdaptiveLogParams] = None):
        """
        🌟 初始化我们的视觉调色师

        Args:
            params: 自适应对数映射参数，如果不提供则使用默认的艺术配置
        """
        self.params = params or AdaptiveLogParams()
        print("🎨 自适应对数映射艺术家已准备就绪，开始创作数学与美学的交响诗！")

    def basic_log_mapping(self, image: np.ndarray) -> np.ndarray:
        """
        🌅 基础对数映射：发现对数美学的第一步

        如同学习音乐的第一个音符，简单却蕴含深意
        这是理解对数变换最朴素而纯真的方式

        Args:
            image: 输入图像，支持灰度和彩色

        Returns:
            对数映射后的图像，展现基础的动态范围压缩之美

        Raises:
            ValueError: 当输入图像为空或格式不正确时
        """
        if image is None or image.size == 0:
            raise ValueError("🚫 输入图像为空，请提供有效的图像数据")

        # 🌟 转换为浮点型，开启精确计算的大门
        # 浮点运算如同艺术创作，需要精确的控制
        float_img = image.astype(np.float32)

        # 📊 找到图像的峰值：了解世界的上限
        # 每张图像都有其独特的动态范围特征
        max_val = np.max(float_img)

        if max_val <= 0:
            print("⚠️ 检测到图像最大值为0，返回原图")
            return image

        # ✨ 应用对数变换：数学的魔法时刻
        # 对数函数的非线性特性模拟人眼的视觉响应
        log_img = np.log(float_img + 1.0)  # 加1避免log(0)的数学困扰

        # 🎨 归一化到可见范围：让美回归人眼的世界
        # 将无限的数学空间映射到有限的显示空间
        normalized = log_img * 255.0 / np.log(max_val + 1.0)

        return np.clip(normalized, 0, 255).astype(np.uint8)

    def adaptive_log_mapping(self, image: np.ndarray,
                           params: Optional[AdaptiveLogParams] = None) -> np.ndarray:
        """
        🌈 自适应对数映射：智慧选择的数学诗篇

        如同经验丰富的调色师，根据每张画布的特点调整色彩
        这是对数映射的进阶形态，体现因材施教的算法智慧

        Args:
            image: 输入图像，支持灰度和彩色格式
            params: 映射参数，如不提供则使用实例默认参数

        Returns:
            自适应对数映射后的图像，展现个性化的动态范围调整

        Raises:
            ValueError: 当输入图像格式不支持时
        """
        if image is None or image.size == 0:
            raise ValueError("🚫 输入图像为空")

        p = params or self.params

        # 🎯 转换为浮点型：进入精确的数学殿堂
        float_img = image.astype(np.float32)

        if len(image.shape) == 2:
            # 🌙 灰度图像：单色世界的对数之美
            return self._process_grayscale(float_img, p)
        elif len(image.shape) == 3:
            # 🌈 彩色图像：多彩世界的和谐统一
            return self._process_color(float_img, p)
        else:
            raise ValueError(f"🚫 不支持的图像格式，维度: {len(image.shape)}")

    def _process_grayscale(self, float_img: np.ndarray,
                          params: AdaptiveLogParams) -> np.ndarray:
        """
        🌙 处理灰度图像的私有方法

        灰度世界虽然没有色彩的绚烂，却有着纯粹的明暗之美

        Args:
            float_img: 浮点型灰度图像
            params: 自适应参数

        Returns:
            处理后的灰度图像
        """
        # 📊 计算图像统计：了解图像的性格特征
        max_val = np.max(float_img)
        mean_val = np.mean(float_img)

        if max_val <= 0:
            return np.zeros_like(float_img, dtype=np.uint8)

        # 🧮 计算自适应缩放因子：因材施教的智慧
        # 根据图像的动态范围自动调整映射强度
        scale = params.max_scale / np.log10(max_val + 1.0)

        # 🎨 智能偏置调整：根据图像平均亮度微调基调
        adaptive_bias = params.bias + 0.1 * (128 - mean_val) / 128

        # ✨ 应用对数映射：数学与艺术的邂逅
        log_img = np.log(float_img + 1.0) * scale + adaptive_bias

        return np.clip(log_img, 0, 255).astype(np.uint8)

    def _process_color(self, float_img: np.ndarray,
                      params: AdaptiveLogParams) -> np.ndarray:
        """
        🌈 处理彩色图像的私有方法

        彩色图像如同交响乐，需要协调处理每个颜色通道

        Args:
            float_img: 浮点型彩色图像
            params: 自适应参数

        Returns:
            处理后的彩色图像
        """
        if params.local_adaptation:
            # 🎭 局部自适应：精工细作的匠人精神
            return self._local_adaptive_mapping(float_img, params)
        else:
            # 🌍 全局自适应：统一和谐的世界观
            return self._global_adaptive_mapping(float_img, params)

    def _global_adaptive_mapping(self, float_img: np.ndarray,
                               params: AdaptiveLogParams) -> np.ndarray:
        """
        🌍 全局自适应映射：统一的视觉处理哲学

        采用全图统一的参数，保持整体的协调感

        Args:
            float_img: 浮点型图像
            params: 自适应参数

        Returns:
            全局自适应处理后的图像
        """
        # 📊 计算全图最大值：宏观视角的统计分析
        max_val = np.max(float_img)
        mean_val = np.mean(float_img)

        if max_val <= 0:
            return np.zeros_like(float_img, dtype=np.uint8)

        # 🧮 计算自适应缩放因子
        # 根据图像的动态范围自动调整映射强度
        scale = params.max_scale / np.log10(max_val + 1.0)

        # 🎨 智能偏置调整：根据整体亮度微调
        adaptive_bias = params.bias + 0.15 * (128 - mean_val) / 128

        # ✨ 应用映射：数学与艺术的完美融合
        log_img = np.log(float_img + 1.0) * scale + adaptive_bias

        return np.clip(log_img, 0, 255).astype(np.uint8)

    def _local_adaptive_mapping(self, float_img: np.ndarray,
                              params: AdaptiveLogParams) -> np.ndarray:
        """
        🎭 局部自适应映射：精工细作的艺术

        为每个像素根据其邻域特征进行个性化处理
        如同精心照料每株植物的园丁，体现因地制宜的智慧

        Args:
            float_img: 浮点型图像
            params: 自适应参数

        Returns:
            局部自适应处理后的图像
        """
        h, w = float_img.shape[:2]
        result = np.zeros_like(float_img)

        # 📏 计算窗口大小：观察世界的视野
        # 窗口大小决定了"局部"的定义范围
        window_size = int(min(h, w) * params.window_ratio)
        half_window = window_size // 2

        print(f"🔍 开始局部自适应处理，窗口大小: {window_size}x{window_size}")

        # 🎨 为每个像素计算局部映射
        # 这是一个计算密集的过程，体现精工细作的精神
        for y in range(h):
            if y % (h // 10) == 0:  # 显示进度
                print(f"📐 处理进度: {y/h*100:.1f}%")

            for x in range(w):
                # 🔍 定义局部区域：当前像素的"影响圈"
                y_min = max(0, y - half_window)
                y_max = min(h, y + half_window + 1)
                x_min = max(0, x - half_window)
                x_max = min(w, x + half_window + 1)

                # 📊 计算局部统计特征
                local_region = float_img[y_min:y_max, x_min:x_max]
                local_max = np.max(local_region)
                local_mean = np.mean(local_region)

                if local_max <= 0:
                    result[y, x] = 0
                    continue

                # 🧮 计算局部自适应参数
                # 每个像素都有其独特的处理参数
                local_scale = params.max_scale / np.log10(local_max + 1.0)
                local_bias = params.bias + 0.1 * (128 - local_mean) / 128

                # ✨ 应用局部映射
                current_pixel = float_img[y, x]
                mapped_pixel = np.log(current_pixel + 1.0) * local_scale + local_bias
                result[y, x] = np.clip(mapped_pixel, 0, 255)

        print("✅ 局部自适应处理完成")
        return result.astype(np.uint8)

    def enhanced_adaptive_mapping(self, image: np.ndarray,
                                params: Optional[AdaptiveLogParams] = None,
                                tone_curve: bool = True) -> np.ndarray:
        """
        ✨ 增强型自适应映射：艺术与技术的完美融合

        在基础自适应映射的基础上，添加色调曲线优化
        如同摄影师的后期调色，追求更加完美的视觉效果

        Args:
            image: 输入图像
            params: 映射参数
            tone_curve: 是否应用S型色调曲线增强

        Returns:
            增强处理后的图像，展现技术与艺术的双重美感
        """
        # 🌟 先应用基础自适应映射
        basic_result = self.adaptive_log_mapping(image, params)

        if not tone_curve:
            return basic_result

        # 🎨 应用S型色调曲线增强对比度
        # S曲线能够在保持中间调的同时增强对比度
        p = params or self.params
        enhanced = self._apply_s_curve(basic_result, p.tone_curve_strength)

        return enhanced

    def _apply_s_curve(self, image: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """
        🎨 应用S型色调曲线：优雅的对比度增强

        S型曲线是摄影后期中的经典技法
        它能够在保持自然外观的同时增强图像的层次感

        Args:
            image: 输入图像
            strength: 曲线强度，控制S型曲线的弯曲程度

        Returns:
            应用S曲线后的图像
        """
        # 创建S型查找表
        lut = np.zeros(256, dtype=np.uint8)

        for i in range(256):
            # 归一化到[0,1]范围
            x = i / 255.0

            # 应用S型函数：修改的sigmoid函数
            # 这个函数在0.5处有拐点，形成经典的S型曲线
            s_value = 1.0 / (1.0 + np.exp(-strength * 10 * (x - 0.5)))

            # 重新映射到[0,255]范围
            lut[i] = np.clip(s_value * 255, 0, 255).astype(np.uint8)

        # 应用查找表进行快速映射
        return cv2.LUT(image, lut)

    def analyze_image_statistics(self, image: np.ndarray) -> Dict[str, Any]:
        """
        📊 分析图像统计特征：了解图像的内在特质

        通过统计分析了解图像的特征，为自适应处理提供依据

        Args:
            image: 输入图像

        Returns:
            包含各种统计信息的字典
        """
        stats = {}

        # 基础统计
        stats['shape'] = image.shape
        stats['dtype'] = image.dtype
        stats['min'] = np.min(image)
        stats['max'] = np.max(image)
        stats['mean'] = np.mean(image)
        stats['std'] = np.std(image)
        stats['median'] = np.median(image)

        # 动态范围分析
        stats['dynamic_range'] = stats['max'] - stats['min']
        stats['contrast_ratio'] = stats['max'] / max(stats['min'], 1)

        # 亮度分布分析
        if len(image.shape) == 3:
            # 彩色图像：转换为灰度进行分析
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        else:
            # 灰度图像：直接计算直方图
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])

        stats['histogram'] = hist.flatten()

        # 亮度分布特征
        total_pixels = image.shape[0] * image.shape[1]
        dark_pixels = np.sum(hist[:85]) / total_pixels     # 暗部像素比例
        mid_pixels = np.sum(hist[85:170]) / total_pixels   # 中间调像素比例
        bright_pixels = np.sum(hist[170:]) / total_pixels  # 亮部像素比例

        stats['brightness_distribution'] = {
            'dark_ratio': dark_pixels,
            'mid_ratio': mid_pixels,
            'bright_ratio': bright_pixels
        }

        # 推荐参数
        stats['recommended_params'] = self._recommend_parameters(stats)

        return stats

    def _recommend_parameters(self, stats: Dict[str, Any]) -> AdaptiveLogParams:
        """
        🎯 根据图像统计特征推荐参数

        基于图像特征自动推荐最佳的处理参数
        体现算法的智能化和人性化特点

        Args:
            stats: 图像统计信息

        Returns:
            推荐的自适应对数映射参数
        """
        # 基础参数
        bias = 0.85
        max_scale = 100.0
        local_adaptation = False
        window_ratio = 0.125

        # 根据动态范围调整
        if stats['dynamic_range'] > 200:
            # 高动态范围图像：增强映射强度
            max_scale = 150.0
            bias = 0.9
        elif stats['dynamic_range'] < 100:
            # 低动态范围图像：温和处理
            max_scale = 80.0
            bias = 0.8

        # 根据亮度分布调整
        brightness_dist = stats['brightness_distribution']
        if brightness_dist['dark_ratio'] > 0.6:
            # 图像偏暗：增加偏置
            bias += 0.2
        elif brightness_dist['bright_ratio'] > 0.6:
            # 图像偏亮：减少偏置
            bias -= 0.2

        # 根据图像大小决定是否使用局部自适应
        total_pixels = stats['shape'][0] * stats['shape'][1]
        if total_pixels > 1000000:  # 大于100万像素
            # 大图像可能需要局部自适应
            local_adaptation = True
            window_ratio = 0.1

        return AdaptiveLogParams(
            bias=np.clip(bias, 0.0, 2.0),
            max_scale=max_scale,
            local_adaptation=local_adaptation,
            window_ratio=window_ratio
        )

    def performance_test(self, image: np.ndarray, iterations: int = 5) -> Dict[str, float]:
        """
        ⚡ 性能测试：评估算法的运行效率

        对不同处理方法进行性能基准测试

        Args:
            image: 测试图像
            iterations: 测试迭代次数

        Returns:
            各种方法的平均执行时间
        """
        print(f"⚡ 开始性能测试，图像尺寸: {image.shape}, 迭代次数: {iterations}")

        results = {}

        # 测试基础对数映射
        start_time = time.time()
        for _ in range(iterations):
            self.basic_log_mapping(image)
        results['basic_log_mapping'] = (time.time() - start_time) / iterations

        # 测试全局自适应映射
        params = AdaptiveLogParams(local_adaptation=False)
        start_time = time.time()
        for _ in range(iterations):
            self.adaptive_log_mapping(image, params)
        results['global_adaptive'] = (time.time() - start_time) / iterations

        # 测试增强型映射
        start_time = time.time()
        for _ in range(iterations):
            self.enhanced_adaptive_mapping(image, params)
        results['enhanced_adaptive'] = (time.time() - start_time) / iterations

        # 如果图像不太大，测试局部自适应
        if image.shape[0] * image.shape[1] < 200000:
            params_local = AdaptiveLogParams(local_adaptation=True, window_ratio=0.1)
            start_time = time.time()
            for _ in range(max(1, iterations // 3)):  # 局部自适应较慢，减少迭代次数
                self.adaptive_log_mapping(image, params_local)
            results['local_adaptive'] = (time.time() - start_time) / max(1, iterations // 3)

        # 打印结果
        print("\n📊 性能测试结果:")
        for method, time_cost in results.items():
            print(f"   {method}: {time_cost:.4f}s")

        return results

    def artistic_showcase(self, image: np.ndarray, save_path: Optional[str] = None) -> None:
        """
        🎭 自适应对数映射艺术展示：展现算法的无限可能

        创建一个视觉艺术馆，展示各种参数组合的艺术效果
        让观者直观地感受数学与美学的对话

        Args:
            image: 输入图像
            save_path: 保存路径，如果提供则保存展示图
        """
        print("🎨 开始创作自适应对数映射艺术作品...")

        # 🎨 创建不同风格的艺术作品
        effects = {
            "📷 原始图像": image,
            "🌅 基础对数映射": self.basic_log_mapping(image),
            "🌈 全局自适应": self.adaptive_log_mapping(
                image, AdaptiveLogParams(bias=0.85, max_scale=100.0, local_adaptation=False)
            ),
            "🎭 推荐参数": self.adaptive_log_mapping(
                image, self.analyze_image_statistics(image)['recommended_params']
            ),
            "✨ 增强型映射": self.enhanced_adaptive_mapping(image),
            "🌟 高对比度": self.adaptive_log_mapping(
                image, AdaptiveLogParams(bias=1.0, max_scale=150.0)
            ),
            "🌙 柔和映射": self.adaptive_log_mapping(
                image, AdaptiveLogParams(bias=0.7, max_scale=80.0)
            ),
            "🔥 强化细节": self.enhanced_adaptive_mapping(
                image, AdaptiveLogParams(bias=0.9, max_scale=200.0), tone_curve=True
            )
        }

        # 🖼️ 创造视觉艺术馆
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('🌟 自适应对数映射艺术馆：数学与美学的对话',
                    fontsize=16, fontweight='bold', y=0.98)

        for i, (title, effect_image) in enumerate(effects.items()):
            row, col = i // 4, i % 4

            # 显示图像
            if len(effect_image.shape) == 3:
                # 彩色图像：BGR转RGB显示
                display_image = cv2.cvtColor(effect_image, cv2.COLOR_BGR2RGB)
                axes[row, col].imshow(display_image)
            else:
                # 灰度图像
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
        print("🎨 艺术展示完成，感谢欣赏数学与美学的对话！")

    def interactive_demo(self, image: np.ndarray) -> None:
        """
        🎮 交互式演示：实时调整参数体验算法效果

        提供一个简化的交互界面，让用户体验参数调整的即时效果

        Args:
            image: 演示图像
        """
        print("🎮 启动交互式演示模式")
        print("💡 提示：修改下面的参数值，观察算法效果的变化")

        # 创建参数选项
        bias_options = [0.5, 0.7, 0.85, 1.0, 1.2]
        scale_options = [50.0, 80.0, 100.0, 120.0, 150.0]

        print("\n🎯 可选参数组合:")
        combinations = []
        for i, bias in enumerate(bias_options):
            for j, scale in enumerate(scale_options):
                combinations.append((bias, scale))
                if len(combinations) <= 6:  # 限制显示数量
                    print(f"   {len(combinations)}: bias={bias}, max_scale={scale}")

        # 显示几个代表性组合的效果
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('🎮 交互式参数演示：探索参数空间的艺术',
                    fontsize=14, fontweight='bold')

        selected_combinations = combinations[:6]
        for i, (bias, scale) in enumerate(selected_combinations):
            row, col = i // 3, i % 3

            params = AdaptiveLogParams(bias=bias, max_scale=scale)
            result = self.adaptive_log_mapping(image, params)

            if len(result.shape) == 3:
                display_image = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                axes[row, col].imshow(display_image)
            else:
                axes[row, col].imshow(result, cmap='gray')

            axes[row, col].set_title(f'bias={bias}, scale={scale}', fontsize=10)
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.show()

        print("🎮 交互式演示完成！你可以在代码中修改参数来探索更多效果")

def create_demo_interface():
    """
    🚀 创建演示界面：展示自适应对数映射的完整功能

    这个函数提供了算法的完整演示，包括：
    - 基础功能展示
    - 性能测试
    - 艺术效果展示
    - 参数推荐
    """
    print("🌟" + "="*60)
    print("🎨      自适应对数映射算法演示系统")
    print("🌟" + "="*60)
    print("📝 功能说明：")
    print("   1. 基础对数映射 - 发现对数美学的第一步")
    print("   2. 自适应映射 - 智慧选择的数学诗篇")
    print("   3. 增强型映射 - 艺术与技术的完美融合")
    print("   4. 性能测试 - 算法效率的量化评估")
    print("   5. 艺术展示 - 数学与美学的视觉对话")
    print("🌟" + "="*60)

def main():
    """
    🎯 主函数：自适应对数映射算法的完整演示

    提供命令行接口，支持：
    - 图像文件处理
    - 参数自定义
    - 结果保存
    - 性能测试
    """
    parser = argparse.ArgumentParser(
        description="🌟 自适应对数映射算法 - 数字世界的智慧眼镜",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎨 使用示例:
  python adaptive_logarithmic_mapping.py --input image.jpg --output result.jpg
  python adaptive_logarithmic_mapping.py --input image.jpg --bias 0.9 --scale 120
  python adaptive_logarithmic_mapping.py --input image.jpg --local --showcase
  python adaptive_logarithmic_mapping.py --demo
        """
    )

    parser.add_argument('--input', '-i', type=str,
                       help='输入图像路径')
    parser.add_argument('--output', '-o', type=str,
                       help='输出图像路径')
    parser.add_argument('--bias', type=float, default=0.85,
                       help='偏置参数 (default: 0.85)')
    parser.add_argument('--scale', type=float, default=100.0,
                       help='最大缩放因子 (default: 100.0)')
    parser.add_argument('--local', action='store_true',
                       help='启用局部自适应')
    parser.add_argument('--enhanced', action='store_true',
                       help='使用增强型映射')
    parser.add_argument('--showcase', action='store_true',
                       help='显示艺术效果展示')
    parser.add_argument('--performance', action='store_true',
                       help='运行性能测试')
    parser.add_argument('--interactive', action='store_true',
                       help='启动交互式演示')
    parser.add_argument('--demo', action='store_true',
                       help='运行完整演示（使用内置测试图像）')

    args = parser.parse_args()

    # 创建演示界面
    create_demo_interface()

    # 初始化算法艺术家
    params = AdaptiveLogParams(
        bias=args.bias,
        max_scale=args.scale,
        local_adaptation=args.local
    )
    artist = AdaptiveLogArtist(params)

    # 处理不同的运行模式
    if args.demo:
        # 演示模式：使用内置测试图像
        print("🎭 启动演示模式，使用内置测试图像...")

        # 创建测试图像
        test_image = create_test_image()

        # 分析图像
        stats = artist.analyze_image_statistics(test_image)
        print(f"\n📊 图像统计信息:")
        print(f"   尺寸: {stats['shape']}")
        print(f"   动态范围: {stats['dynamic_range']}")
        print(f"   平均亮度: {stats['mean']:.2f}")

        # 艺术展示
        if args.showcase or True:  # 演示模式默认显示
            artist.artistic_showcase(test_image)

        # 性能测试
        if args.performance or True:  # 演示模式默认测试
            artist.performance_test(test_image)

        # 交互式演示
        if args.interactive:
            artist.interactive_demo(test_image)

    elif args.input:
        # 文件处理模式
        if not os.path.exists(args.input):
            print(f"🚫 错误：输入文件 {args.input} 不存在")
            return

        # 读取图像
        image = cv2.imread(args.input)
        if image is None:
            print(f"🚫 错误：无法读取图像文件 {args.input}")
            return

        print(f"📖 成功读取图像: {args.input}, 尺寸: {image.shape}")

        # 分析图像并推荐参数
        stats = artist.analyze_image_statistics(image)
        recommended = stats['recommended_params']
        print(f"\n🎯 推荐参数: bias={recommended.bias:.2f}, scale={recommended.max_scale:.1f}")

        # 选择处理方法
        if args.enhanced:
            result = artist.enhanced_adaptive_mapping(image, params)
            print("✨ 使用增强型自适应映射")
        else:
            result = artist.adaptive_log_mapping(image, params)
            print("🌈 使用标准自适应映射")

        # 保存结果
        if args.output:
            cv2.imwrite(args.output, result)
            print(f"💾 结果已保存至: {args.output}")

        # 其他功能
        if args.showcase:
            artist.artistic_showcase(image)

        if args.performance:
            artist.performance_test(image)

        if args.interactive:
            artist.interactive_demo(image)

    else:
        print("🤔 请提供输入图像路径或使用 --demo 运行演示")
        print("💡 使用 --help 查看详细帮助信息")

def create_test_image(size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    🎨 创建测试图像：用于演示的艺术画布

    生成一个包含不同亮度区域的测试图像
    用于展示自适应对数映射的效果

    Args:
        size: 图像尺寸

    Returns:
        测试图像
    """
    h, w = size
    image = np.zeros((h, w, 3), dtype=np.uint8)

    # 创建径向渐变
    center_x, center_y = w // 2, h // 2
    max_radius = min(center_x, center_y)

    for y in range(h):
        for x in range(w):
            # 计算到中心的距离
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)

            # 创建复杂的亮度分布
            if distance < max_radius * 0.3:
                # 中心：高亮区域
                intensity = 200 + 55 * np.sin(distance * 0.1)
            elif distance < max_radius * 0.6:
                # 中间：中等亮度
                intensity = 100 + 50 * np.cos(distance * 0.05)
            elif distance < max_radius * 0.9:
                # 外环：较暗区域
                intensity = 50 + 30 * np.sin(distance * 0.02)
            else:
                # 边缘：很暗
                intensity = 20 + 10 * np.random.random()

            # 添加一些颜色变化
            image[y, x] = [
                np.clip(intensity + 20 * np.sin(x * 0.01), 0, 255),
                np.clip(intensity, 0, 255),
                np.clip(intensity + 20 * np.cos(y * 0.01), 0, 255)
            ]

    return image

if __name__ == "__main__":
    # 🌟 欢迎来到自适应对数映射的数学艺术世界
    print("🎨 欢迎使用自适应对数映射算法 - 数字世界的智慧眼镜")
    print("✨ 让数学之美与视觉艺术在像素中相遇...")

    try:
        main()
    except KeyboardInterrupt:
        print("\n🎭 用户中断，感谢体验自适应对数映射的艺术之旅！")
    except Exception as e:
        print(f"\n🚫 程序异常: {e}")
        print("💡 请检查输入参数和图像文件")

    print("�� 探索永无止境，创造无限可能！")