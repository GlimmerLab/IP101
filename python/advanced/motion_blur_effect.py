"""
🌊 运动的诗意：运动模糊算法的时间艺术Python实现

这个模块实现了完整的运动模糊算法，包含：
- 方向性运动模糊：如同疾风中的草原
- 径向运动模糊：如同涟漪向四周扩散
- 旋转运动模糊：如同舞者旋转的优美弧线
- 缩放运动模糊：如同时间隧道的视觉体验
- 交互式演示：实时体验时间的魔法
- 性能测试：评估不同方法的效率

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
class MotionBlurParams:
    """🎨 运动模糊的艺术配置参数"""
    size: int = 15              # 模糊核大小
    angle: float = 45.0         # 模糊方向角度 (0-360度)
    strength: float = 1.0       # 模糊强度 (0-1)
    motion_type: str = "linear" # 运动类型 ("linear", "radial", "rotational", "zoom")

    def __post_init__(self):
        """参数有效性检查"""
        assert self.size >= 3, "模糊核大小必须 >= 3"
        assert 0.0 <= self.angle <= 360.0, "角度必须在[0, 360]范围内"
        assert 0.0 <= self.strength <= 2.0, "强度必须在[0, 2.0]范围内"
        assert self.motion_type in ["linear", "radial", "rotational", "zoom"], "无效的运动类型"

class MotionBlurArtist:
    """🌊 运动模糊艺术家：用时间的画笔创造动感世界"""

    def __init__(self, params: Optional[MotionBlurParams] = None):
        """
        🌟 初始化我们的时间艺术家
        每个参数都是时间魔法的咒语
        """
        self.params = params or MotionBlurParams()

    def create_motion_kernel(self, size: int, angle: float) -> np.ndarray:
        """
        🖌️ 创造运动的笔触：将方向转化为数学的诗篇

        就像画家选择笔触的方向来表现风的流动

        Args:
            size: 笔触的长度，决定了运动的距离
            angle: 笔触的方向，表达着运动的意图

        Returns:
            一个蕴含时间智慧的运动核
        """
        # 🎨 确保核大小为奇数，如同艺术需要平衡
        if size % 2 == 0:
            size += 1

        kernel = np.zeros((size, size), dtype=np.float32)

        # 🌟 将角度转换为弧度，进入数学的纯净世界
        radian_angle = math.radians(angle)

        # 🎯 计算中心点，所有运动的起点
        center_x, center_y = size // 2, size // 2

        # 🧭 计算方向向量，如同指南针的指向
        dx = math.cos(radian_angle)
        dy = math.sin(radian_angle)

        # ✨ 在数学画布上绘制时间的轨迹
        norm_factor = 0.0

        for i in range(-size//2, size//2 + 1):
            x = int(round(center_x + i * dx))
            y = int(round(center_y + i * dy))

            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1.0
                norm_factor += 1.0

        # 🎭 归一化：保持能量的平衡
        if norm_factor > 0:
            kernel /= norm_factor

        return kernel

    def directional_motion_blur(self, image: np.ndarray,
                               size: int = 15,
                               angle: float = 45.0,
                               strength: float = 1.0) -> np.ndarray:
        """
        🏃‍♂️ 方向性运动模糊：如同疾风中的草原

        Args:
            image: 静止的瞬间，等待时间的魔法
            size: 运动的距离
            angle: 运动的方向
            strength: 时间的强度

        Returns:
            充满动感的艺术作品
        """
        if len(image.shape) != 3:
            raise ValueError("🚫 请提供彩色图像，就像生活需要色彩一样")

        # 🖌️ 创造运动的笔触
        kernel = self.create_motion_kernel(size, angle)

        # 🎨 用运动的笔触为图像注入时间的活力
        blurred = cv2.filter2D(image, -1, kernel)

        # 🌈 如果需要，与原图混合，创造恰到好处的动感
        if abs(strength - 1.0) > 1e-6:
            result = cv2.addWeighted(image, 1.0 - strength, blurred, strength, 0)
            return result

        return blurred

    def radial_motion_blur(self, image: np.ndarray,
                          strength: float = 0.5,
                          center: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        🌀 径向运动模糊：如同涟漪向四周扩散

        创造从中心点向外扩散的动感效果

        Args:
            image: 输入图像
            strength: 扩散的强度
            center: 涟漪的中心点

        Returns:
            带有径向动感的图像
        """
        h, w = image.shape[:2]

        # 🎯 确定扩散的中心
        if center is None:
            center_x, center_y = w // 2, h // 2
        else:
            center_x, center_y = center

        # 🌊 采样数：涟漪的圈数
        num_samples = 15
        step = strength / (num_samples - 1) if num_samples > 1 else 0

        # 🎨 创造空白的画布
        result = np.zeros_like(image, dtype=np.float64)
        total_weight = 0.0

        # 🌟 绘制每一圈涟漪
        for i in range(num_samples):
            # 💫 计算缩放系数
            scale = 1.0 - step * i
            weight = 1.0 / num_samples
            total_weight += weight

            # 🎭 创建仿射变换矩阵
            M = np.float32([
                [scale, 0, center_x * (1 - scale)],
                [0, scale, center_y * (1 - scale)]
            ])

            # 🌈 应用变换
            transformed = cv2.warpAffine(image, M, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE)

            # ✨ 累积到最终结果
            result += weight * transformed.astype(np.float64)

        # ⚖️ 确保完美的平衡
        return np.clip(result, 0, 255).astype(np.uint8)

    def rotational_motion_blur(self, image: np.ndarray,
                              strength: float = 0.5,
                              center: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        🌪️ 旋转运动模糊：如同舞者旋转的优美弧线

        Args:
            image: 输入图像
            strength: 旋转的强度
            center: 旋转的中心

        Returns:
            带有旋转动感的图像
        """
        h, w = image.shape[:2]

        # 💃 确定旋转的中心
        if center is None:
            center_point = (w // 2, h // 2)
        else:
            center_point = center

        # 🎵 采样数和角度范围
        num_samples = 15
        max_angle = 30.0 * strength
        angle_step = max_angle / (num_samples - 1) if num_samples > 1 else 0

        # 🎨 创造空白画布
        result = np.zeros_like(image, dtype=np.float64)
        total_weight = 0.0

        # 🌀 创造旋转的韵律
        for i in range(num_samples):
            # 🌟 计算旋转角度
            angle = -max_angle / 2.0 + i * angle_step
            weight = 1.0 / num_samples
            total_weight += weight

            # 🎭 创建旋转矩阵
            M = cv2.getRotationMatrix2D(center_point, angle, 1.0)

            # 💫 应用旋转
            rotated = cv2.warpAffine(image, M, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)

            # 🌈 叠加到结果
            result += weight * rotated.astype(np.float64)

        return np.clip(result, 0, 255).astype(np.uint8)

    def zoom_motion_blur(self, image: np.ndarray,
                        strength: float = 0.5,
                        center: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        🎯 缩放运动模糊：如同时间隧道的视觉体验

        Args:
            image: 输入图像
            strength: 缩放的强度
            center: 缩放的中心

        Returns:
            带有缩放动感的图像
        """
        h, w = image.shape[:2]

        # 🎯 确定缩放的焦点
        if center is None:
            center_x, center_y = w // 2, h // 2
        else:
            center_x, center_y = center

        # 🌟 采样参数
        num_samples = 15
        max_scale_delta = 0.4 * strength
        scale_step = max_scale_delta / (num_samples - 1) if num_samples > 1 else 0

        # 🎨 创造空白画布
        result = np.zeros_like(image, dtype=np.float64)
        total_weight = 0.0

        # 🎬 创造电影般的缩放效果
        for i in range(num_samples):
            # 📏 计算缩放系数
            scale = 1.0 - max_scale_delta / 2.0 + i * scale_step
            weight = 1.0 / num_samples
            total_weight += weight

            # 🎭 创建缩放变换矩阵
            M = np.float32([
                [scale, 0, center_x * (1 - scale)],
                [0, scale, center_y * (1 - scale)]
            ])

            # 🌀 应用缩放变换
            scaled = cv2.warpAffine(image, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)

            # ✨ 累积效果
            result += weight * scaled.astype(np.float64)

        return np.clip(result, 0, 255).astype(np.uint8)

    def motion_blur(self, image: np.ndarray,
                   params: Optional[MotionBlurParams] = None) -> np.ndarray:
        """
        🌊 统一的运动模糊接口：根据参数选择时间的艺术表达

        Args:
            image: 输入图像
            params: 运动模糊参数配置

        Returns:
            充满时间魔法的艺术作品
        """
        p = params or self.params

        if p.motion_type == "linear":
            return self.directional_motion_blur(image, p.size, p.angle, p.strength)
        elif p.motion_type == "radial":
            return self.radial_motion_blur(image, p.strength)
        elif p.motion_type == "rotational":
            return self.rotational_motion_blur(image, p.strength)
        elif p.motion_type == "zoom":
            return self.zoom_motion_blur(image, p.strength)
        else:
            return self.directional_motion_blur(image, p.size, p.angle, p.strength)

    def artistic_showcase(self, image: np.ndarray, save_path: Optional[str] = None) -> None:
        """
        🎭 运动模糊艺术展示：展现时间的多种诗意表达

        如同时间艺术馆的作品展览，展示运动的无限可能
        """
        print("🎨 开始创作运动模糊艺术作品...")

        # 🎨 创建不同风格的运动作品
        effects = {
            "📷 静止的瞬间": image,
            "🏃‍♂️ 线性运动": self.directional_motion_blur(image, 25, 45, 0.8),
            "🌪️ 风暴旋转": self.directional_motion_blur(image, 30, 90, 1.0),
            "🌀 径向扩散": self.radial_motion_blur(image, 0.6),
            "💫 旋转韵律": self.rotational_motion_blur(image, 0.7),
            "🎯 缩放隧道": self.zoom_motion_blur(image, 0.5),
            "⚡ 高速运动": self.directional_motion_blur(image, 40, 0, 0.9),
            "🌊 时间涟漪": self.radial_motion_blur(image, 0.8)
        }

        # 🖼️ 创造时间艺术馆
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('🌊 运动模糊艺术馆：时间的诗意表达', fontsize=16, fontweight='bold')

        for i, (title, effect_image) in enumerate(effects.items()):
            row, col = i // 4, i % 4
            axes[row, col].imshow(cv2.cvtColor(effect_image, cv2.COLOR_BGR2RGB))
            axes[row, col].set_title(title, fontsize=11)
            axes[row, col].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 运动模糊艺术展示已保存至: {save_path}")

        plt.show()

    def motion_analysis(self, image: np.ndarray) -> None:
        """
        🔍 运动模糊效果分析：展示不同参数的影响
        """
        print("🔍 分析运动参数对模糊效果的影响...")

        # 测试不同的运动方向
        angles = [0, 30, 60, 90]
        angle_results = []

        for angle in angles:
            result = self.directional_motion_blur(image, 20, angle, 0.8)
            angle_results.append((f"角度: {angle}°", result))

        # 测试不同的运动强度
        strengths = [0.3, 0.6, 0.9, 1.2]
        strength_results = []

        for strength in strengths:
            result = self.directional_motion_blur(image, 25, 45, strength)
            strength_results.append((f"强度: {strength}", result))

        # 可视化对比
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('🔍 运动参数影响分析', fontsize=14, fontweight='bold')

        # 显示角度影响
        for i, (title, img) in enumerate(angle_results):
            axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[0, i].set_title(title, fontsize=10)
            axes[0, i].axis('off')

        # 显示强度影响
        for i, (title, img) in enumerate(strength_results):
            axes[1, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[1, i].set_title(title, fontsize=10)
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.show()

    def motion_type_comparison(self, image: np.ndarray) -> None:
        """
        🎪 运动类型对比：不同运动模式的艺术表达
        """
        print("🎪 展示不同的运动模糊类型...")

        motion_types = {
            "📷 原图": image,
            "🏃‍♂️ 线性运动": self.directional_motion_blur(image, 25, 30, 0.8),
            "🌀 径向运动": self.radial_motion_blur(image, 0.6),
            "🌪️ 旋转运动": self.rotational_motion_blur(image, 0.7),
            "🎯 缩放运动": self.zoom_motion_blur(image, 0.6),
            "⚡ 快速线性": self.directional_motion_blur(image, 35, 0, 1.0),
            "💫 快速旋转": self.rotational_motion_blur(image, 0.9),
            "🌊 强径向": self.radial_motion_blur(image, 0.8)
        }

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('🎪 运动类型对比：时间的多种韵律', fontsize=16, fontweight='bold')

        for i, (title, motion_image) in enumerate(motion_types.items()):
            row, col = i // 4, i % 4
            axes[row, col].imshow(cv2.cvtColor(motion_image, cv2.COLOR_BGR2RGB))
            axes[row, col].set_title(title, fontsize=11)
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.show()

    def interactive_motion_blur(self, image: np.ndarray) -> None:
        """
        🎮 交互式运动模糊：实时调整参数体验时间魔法

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
        ax_motion_type = plt.subplot2grid((4, 4), (2, 0), rowspan=1, colspan=1)

        # 显示原图
        ax_original.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax_original.set_title('📷 原始图像', fontsize=12)
        ax_original.axis('off')

        # 初始效果
        initial_result = self.directional_motion_blur(image, 15, 45, 0.8)
        im_result = ax_result.imshow(cv2.cvtColor(initial_result, cv2.COLOR_BGR2RGB))
        ax_result.set_title('🌊 运动模糊效果', fontsize=12)
        ax_result.axis('off')

        # 运动类型选择
        motion_types = ['线性', '径向', '旋转', '缩放']
        radio = RadioButtons(ax_motion_type, motion_types, active=0)
        ax_motion_type.set_title('🎨 运动类型')

        # 创建滑块
        ax_size = plt.axes([0.15, 0.25, 0.25, 0.03])
        ax_angle = plt.axes([0.55, 0.25, 0.25, 0.03])
        ax_strength = plt.axes([0.15, 0.2, 0.25, 0.03])
        ax_center_x = plt.axes([0.55, 0.2, 0.25, 0.03])
        ax_center_y = plt.axes([0.15, 0.15, 0.25, 0.03])

        slider_size = Slider(ax_size, '核大小', 5, 50, valinit=15, valfmt='%d')
        slider_angle = Slider(ax_angle, '角度', 0, 360, valinit=45)
        slider_strength = Slider(ax_strength, '强度', 0.1, 2.0, valinit=0.8)
        slider_center_x = Slider(ax_center_x, '中心X', 0, image.shape[1], valinit=image.shape[1]//2, valfmt='%d')
        slider_center_y = Slider(ax_center_y, '中心Y', 0, image.shape[0], valinit=image.shape[0]//2, valfmt='%d')

        def update(_):
            """更新运动模糊效果"""
            motion_type = radio.value_selected
            size = int(slider_size.val)
            angle = slider_angle.val
            strength = slider_strength.val
            center_x = int(slider_center_x.val)
            center_y = int(slider_center_y.val)

            if motion_type == '线性':
                result = self.directional_motion_blur(image, size, angle, strength)
            elif motion_type == '径向':
                result = self.radial_motion_blur(image, strength, (center_x, center_y))
            elif motion_type == '旋转':
                result = self.rotational_motion_blur(image, strength, (center_x, center_y))
            elif motion_type == '缩放':
                result = self.zoom_motion_blur(image, strength, (center_x, center_y))

            im_result.set_data(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            ax_result.set_title(f'🌊 {motion_type}运动模糊')
            fig.canvas.draw()

        # 绑定事件
        slider_size.on_changed(update)
        slider_angle.on_changed(update)
        slider_strength.on_changed(update)
        slider_center_x.on_changed(update)
        slider_center_y.on_changed(update)
        radio.on_clicked(update)

        plt.tight_layout()
        plt.show()

    def performance_test(self, image_sizes: List[Tuple[int, int]] = None) -> Dict[str, float]:
        """
        ⚡ 性能测试：评估不同运动模糊方法的处理速度

        Args:
            image_sizes: 测试的图像尺寸列表

        Returns:
            性能测试结果字典
        """
        if image_sizes is None:
            image_sizes = [(256, 256), (512, 512), (1024, 1024)]

        results = {}

        print("🚀 开始运动模糊性能测试...")
        print("=" * 60)

        for width, height in image_sizes:
            # 创建测试图像
            test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

            # 测试不同方法
            methods = {
                '线性运动模糊': lambda img: self.directional_motion_blur(img, 25, 45, 0.8),
                '径向运动模糊': lambda img: self.radial_motion_blur(img, 0.6),
                '旋转运动模糊': lambda img: self.rotational_motion_blur(img, 0.7),
                '缩放运动模糊': lambda img: self.zoom_motion_blur(img, 0.5)
            }

            print(f"📊 图像尺寸: {width}x{height}")

            for method_name, method_func in methods.items():
                start_time = time.time()
                _ = method_func(test_image)
                processing_time = time.time() - start_time

                key = f"{method_name}_{width}x{height}"
                results[key] = processing_time

                print(f"  🌊 {method_name}: {processing_time:.3f}秒")

            print("-" * 40)

        print("✅ 性能测试完成")
        return results

def create_motion_blur_demo():
    """🎯 创建运动模糊演示程序"""

    def process_image_interactive():
        """交互式图像处理"""
        while True:
            print("\n" + "="*60)
            print("🌊 运动模糊艺术家 - 交互式演示")
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

            # 创建运动模糊艺术家
            artist = MotionBlurArtist()

            print("\n🎨 请选择运动模糊类型:")
            print("1. 🏃‍♂️ 线性运动模糊")
            print("2. 🌀 径向运动模糊")
            print("3. 🌪️ 旋转运动模糊")
            print("4. 🎯 缩放运动模糊")
            print("5. 🎨 艺术展示")
            print("6. 🎪 类型对比")
            print("7. 🔍 参数分析")
            print("8. 🎮 交互式调节")

            choice = input("请选择 (1-8): ").strip()

            try:
                if choice == '1':
                    size = int(input("🏃‍♂️ 核大小 [5-50, 默认25]: ") or "25")
                    angle = float(input("🏃‍♂️ 运动角度 [0-360, 默认45]: ") or "45")
                    strength = float(input("🏃‍♂️ 运动强度 [0.1-2.0, 默认0.8]: ") or "0.8")
                    result = artist.directional_motion_blur(image, size, angle, strength)
                elif choice == '2':
                    strength = float(input("🌀 扩散强度 [0.1-1.0, 默认0.6]: ") or "0.6")
                    result = artist.radial_motion_blur(image, strength)
                elif choice == '3':
                    strength = float(input("🌪️ 旋转强度 [0.1-1.0, 默认0.7]: ") or "0.7")
                    result = artist.rotational_motion_blur(image, strength)
                elif choice == '4':
                    strength = float(input("🎯 缩放强度 [0.1-1.0, 默认0.5]: ") or "0.5")
                    result = artist.zoom_motion_blur(image, strength)
                elif choice == '5':
                    artist.artistic_showcase(image)
                    continue
                elif choice == '6':
                    artist.motion_type_comparison(image)
                    continue
                elif choice == '7':
                    artist.motion_analysis(image)
                    continue
                elif choice == '8':
                    artist.interactive_motion_blur(image)
                    continue
                else:
                    print("❌ 无效选择")
                    continue

                # 显示结果
                comparison = np.hstack([image, result])
                cv2.imshow("Motion Blur (Original | Blurred)", comparison)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # 询问是否保存
                save_choice = input("\n💾 是否保存结果? (y/n): ").strip().lower()
                if save_choice == 'y':
                    output_path = input("📁 输入保存路径 (默认: motion_blur_result.jpg): ").strip() or "motion_blur_result.jpg"
                    cv2.imwrite(output_path, result)
                    print(f"✅ 结果已保存至: {output_path}")

            except ValueError:
                print("❌ 参数格式错误")
            except Exception as e:
                print(f"❌ 处理出错: {e}")

    def batch_process_demo():
        """批量处理演示"""
        print("\n" + "="*60)
        print("🚀 批量运动模糊处理演示")
        print("="*60)

        input_dir = input("📁 输入图像目录路径: ").strip()
        if not os.path.exists(input_dir):
            print("❌ 目录不存在")
            return

        output_dir = input("📁 输出目录路径: ").strip() or "motion_blur_results"
        os.makedirs(output_dir, exist_ok=True)

        # 选择运动类型
        print("\n🎨 选择运动模糊类型:")
        print("1. 线性运动")
        print("2. 径向运动")
        print("3. 旋转运动")
        print("4. 缩放运动")

        motion_choice = input("请选择 (1-4): ").strip()
        motion_map = {
            '1': ('linear', '线性运动'),
            '2': ('radial', '径向运动'),
            '3': ('rotational', '旋转运动'),
            '4': ('zoom', '缩放运动')
        }

        if motion_choice not in motion_map:
            print("❌ 无效选择")
            return

        motion_type, motion_name = motion_map[motion_choice]

        # 获取图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = [f for f in os.listdir(input_dir)
                      if Path(f).suffix.lower() in image_extensions]

        if not image_files:
            print("❌ 未找到图像文件")
            return

        print(f"📸 找到 {len(image_files)} 张图像")

        # 创建运动模糊艺术家
        artist = MotionBlurArtist()

        # 批量处理
        for i, filename in enumerate(image_files):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"motion_blur_{filename}")

            print(f"🎨 处理 ({i+1}/{len(image_files)}): {filename}")

            image = cv2.imread(input_path)
            if image is not None:
                if motion_type == 'linear':
                    result = artist.directional_motion_blur(image, 25, 45, 0.8)
                elif motion_type == 'radial':
                    result = artist.radial_motion_blur(image, 0.6)
                elif motion_type == 'rotational':
                    result = artist.rotational_motion_blur(image, 0.7)
                elif motion_type == 'zoom':
                    result = artist.zoom_motion_blur(image, 0.5)

                cv2.imwrite(output_path, result)
                print(f"✅ 已保存: {output_path}")
            else:
                print(f"❌ 无法读取: {filename}")

        print(f"\n🎉 批量处理完成！结果保存在: {output_dir}")

    # 主菜单
    while True:
        print("\n" + "="*70)
        print("🌊 运动模糊艺术家 - 时间的诗意表达")
        print("="*70)
        print("1. 📷 交互式单图处理")
        print("2. 🚀 批量图像处理")
        print("3. 🎨 艺术效果展示")
        print("4. 🎪 运动类型对比")
        print("5. 🎮 交互式参数调节")
        print("6. 📊 性能测试")
        print("7. 🔍 参数影响分析")
        print("0. 👋 退出程序")
        print("="*70)

        choice = input("请选择功能 (0-7): ").strip()

        if choice == '0':
            print("👋 感谢体验运动模糊艺术家！")
            print("愿你的世界如时间般充满流动的美感！ ✨")
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
                    artist = MotionBlurArtist()
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
                    artist = MotionBlurArtist()
                    artist.motion_type_comparison(image)
                else:
                    print("❌ 无法读取图像")
            else:
                print("❌ 文件不存在")
        elif choice == '5':
            image_path = input("📷 请输入图像路径: ").strip()
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                if image is not None:
                    artist = MotionBlurArtist()
                    artist.interactive_motion_blur(image)
                else:
                    print("❌ 无法读取图像")
            else:
                print("❌ 文件不存在")
        elif choice == '6':
            artist = MotionBlurArtist()
            artist.performance_test()
        elif choice == '7':
            image_path = input("📷 请输入图像路径: ").strip()
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                if image is not None:
                    artist = MotionBlurArtist()
                    artist.motion_analysis(image)
                else:
                    print("❌ 无法读取图像")
            else:
                print("❌ 文件不存在")
        else:
            print("❌ 无效选择，请重新输入")

def main():
    """🌟 主函数：展示运动模糊的时间魔法"""
    parser = argparse.ArgumentParser(description="🌊 运动模糊 - 时间的诗意表达")
    parser.add_argument("--input", "-i", type=str, help="输入图像路径")
    parser.add_argument("--output", "-o", type=str, help="输出图像路径")
    parser.add_argument("--type", "-t", type=str, default="linear",
                       choices=["linear", "radial", "rotational", "zoom"],
                       help="运动类型")
    parser.add_argument("--size", "-s", type=int, default=25, help="核大小 (5-50)")
    parser.add_argument("--angle", "-a", type=float, default=45.0, help="运动角度 (0-360)")
    parser.add_argument("--strength", type=float, default=0.8, help="运动强度 (0.1-2.0)")
    parser.add_argument("--center-x", type=int, default=-1, help="中心点X坐标")
    parser.add_argument("--center-y", type=int, default=-1, help="中心点Y坐标")
    parser.add_argument("--demo", action="store_true", help="启动演示模式")
    parser.add_argument("--showcase", action="store_true", help="显示艺术展示")
    parser.add_argument("--comparison", action="store_true", help="显示类型对比")
    parser.add_argument("--interactive", action="store_true", help="交互式参数调节")
    parser.add_argument("--analysis", action="store_true", help="参数影响分析")
    parser.add_argument("--performance", action="store_true", help="运行性能测试")

    args = parser.parse_args()

    if args.demo:
        create_motion_blur_demo()
        return

    if not args.input:
        print("🚫 请提供输入图像路径，或使用 --demo 启动演示模式")
        print("💡 使用示例: python motion_blur_effect.py -i image.jpg -o result.jpg")
        print("💡 演示模式: python motion_blur_effect.py --demo")
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

    # 创建运动模糊艺术家
    artist = MotionBlurArtist()

    if args.performance:
        # 性能测试
        artist.performance_test()
        return

    if args.showcase:
        # 艺术展示
        save_path = args.output.replace('.jpg', '_showcase.png') if args.output else None
        artist.artistic_showcase(image, save_path)
        return

    if args.comparison:
        # 类型对比
        artist.motion_type_comparison(image)
        return

    if args.interactive:
        # 交互式调节
        artist.interactive_motion_blur(image)
        return

    if args.analysis:
        # 参数分析
        artist.motion_analysis(image)
        return

    # 应用指定的运动模糊
    print(f"🎨 应用{args.type}运动模糊...")

    center = None
    if args.center_x >= 0 and args.center_y >= 0:
        center = (args.center_x, args.center_y)

    if args.type == "linear":
        result = artist.directional_motion_blur(image, args.size, args.angle, args.strength)
    elif args.type == "radial":
        result = artist.radial_motion_blur(image, args.strength, center)
    elif args.type == "rotational":
        result = artist.rotational_motion_blur(image, args.strength, center)
    elif args.type == "zoom":
        result = artist.zoom_motion_blur(image, args.strength, center)

    if args.output:
        cv2.imwrite(args.output, result)
        print(f"✅ 运动模糊艺术作品已保存至: {args.output}")
    else:
        # 显示对比
        comparison = np.hstack([image, result])
        cv2.imshow("Motion Blur (Original | Blurred)", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()