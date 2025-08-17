"""
🖌️ 锐化的艺术哲学：钝化蒙版算法的Python实现

这个模块实现了完整的钝化蒙版锐化算法，包含：
- 基础钝化蒙版：纯真的锐化艺术
- 阈值钝化蒙版：智慧的选择性锐化
- 自适应钝化蒙版：禅师般的觉察与慈悲
- 边缘保护分析：展示算法的智慧
- 交互式演示：实时体验锐化的魔法

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
class UnsharpMaskingParams:
    """🎨 钝化蒙版的艺术配置参数"""
    strength: float = 1.5      # 锐化强度 [0, 5]
    radius: float = 1.0        # 模糊半径
    threshold: float = 10.0    # 锐化阈值 [0, 255]
    adaptive: bool = False     # 是否自适应
    edge_protect: float = 0.5  # 边缘保护 [0, 1]

    def __post_init__(self):
        """参数有效性检查"""
        assert 0.0 <= self.strength <= 5.0, "锐化强度必须在[0.0, 5.0]范围内"
        assert 0.0 < self.radius <= 10.0, "模糊半径必须在(0.0, 10.0]范围内"
        assert 0.0 <= self.threshold <= 255.0, "锐化阈值必须在[0.0, 255.0]范围内"
        assert 0.0 <= self.edge_protect <= 1.0, "边缘保护必须在[0.0, 1.0]范围内"

class UnsharpMaskingArtist:
    """🖌️ 钝化蒙版艺术家：用逆向思维创造锐利之美"""

    def __init__(self, params: Optional[UnsharpMaskingParams] = None):
        """
        🌟 初始化我们的锐化艺术家
        每个参数都是创作工具箱中的画笔
        """
        self.params = params or UnsharpMaskingParams()

    def basic_unsharp_masking(self, image: np.ndarray,
                             strength: float = 1.5,
                             radius: float = 1.0) -> np.ndarray:
        """
        🌱 基础钝化蒙版：纯真的锐化艺术

        就像年轻画家的第一幅作品，充满热情但缺乏技巧的复杂性

        Args:
            image: 等待锐化的画布
            strength: 创作的激情强度
            radius: 模糊梦境的半径

        Returns:
            重获锐利的艺术品
        """
        if len(image.shape) != 3:
            raise ValueError("🚫 请提供彩色图像，就像画家需要调色板一样")

        # 🎨 转换为精密的浮点画布
        image_float = image.astype(np.float32)

        # 💭 用高斯的温柔创造模糊的梦境
        # 将radius转换为合适的kernel大小
        kernel_size = max(int(2 * radius * 3), 3) | 1  # 确保是奇数且至少为3
        blurred = cv2.GaussianBlur(image_float, (kernel_size, kernel_size), radius)

        # ⚡ 计算现实与梦境的差异——细节的精魂
        detail_layer = image_float - blurred

        # ✨ 将精魂注入现实，创造超越的美
        sharpened = image_float + strength * detail_layer

        # 🖼️ 裁剪到人间可见的色彩范围
        result = np.clip(sharpened, 0, 255).astype(np.uint8)

        return result

    def threshold_unsharp_masking(self, image: np.ndarray,
                                 strength: float = 1.5,
                                 radius: float = 1.0,
                                 threshold: float = 10.0) -> np.ndarray:
        """
        🎯 阈值钝化蒙版：智慧的选择性锐化

        如同成熟艺术家知道何时下重笔，何时轻描淡写

        Args:
            image: 输入图像
            strength: 锐化强度
            radius: 模糊半径
            threshold: 锐化阈值

        Returns:
            智慧锐化的图像
        """
        image_float = image.astype(np.float32)

        # 💭 创造温柔的模糊梦境
        kernel_size = max(int(2 * radius * 3), 3) | 1
        blurred = cv2.GaussianBlur(image_float, (kernel_size, kernel_size), radius)

        # ⚡ 提取细节精魂
        detail_layer = image_float - blurred

        # 🎭 创建智慧的面具：只对重要细节施展魔法
        mask = np.zeros_like(detail_layer)

        for c in range(3):  # 对每个颜色通道
            detail_abs = np.abs(detail_layer[:, :, c])

            # 🌟 对于真正的细节，全力以赴
            strong_details = detail_abs > threshold
            mask[:, :, c][strong_details] = detail_layer[:, :, c][strong_details]

            # 🌙 对于微妙的变化，温柔渐变
            weak_details = detail_abs <= threshold
            if np.any(weak_details):
                scale = np.power(detail_abs[weak_details] / threshold, 2.0)
                mask[:, :, c][weak_details] = scale * detail_layer[:, :, c][weak_details]

        # 🎨 应用智慧的锐化
        sharpened = image_float + strength * mask

        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def adaptive_unsharp_masking(self, image: np.ndarray,
                                strength: float = 1.5,
                                radius: float = 1.0,
                                edge_protect: float = 0.5) -> np.ndarray:
        """
        🧘 自适应钝化蒙版：禅师般的觉察与慈悲

        能够感知图像的内容，在需要的地方发力，在脆弱的地方保护

        Args:
            image: 输入图像
            strength: 锐化强度
            radius: 模糊半径
            edge_protect: 边缘保护强度

        Returns:
            自适应锐化的图像
        """
        image_float = image.astype(np.float32)

        # 💭 创造模糊梦境
        kernel_size = max(int(2 * radius * 3), 3) | 1
        blurred = cv2.GaussianBlur(image_float, (kernel_size, kernel_size), radius)

        # ⚡ 提取细节精魂
        detail_layer = image_float - blurred

        # 🔍 觉察边缘的存在
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)

        # 🌸 创建慈悲的权重：对边缘温柔保护
        max_edge = np.max(edges)
        edge_norm = edges / max_edge if max_edge > 0 else edges

        # 🕊️ 在强边缘处减少锐化，在平滑区域增强锐化
        protection_mask = 1.0 - edge_protect * np.power(edge_norm, 0.5)

        # 🎭 扩展为三通道
        protection_mask_3d = np.stack([protection_mask] * 3, axis=-1)

        # 🎨 应用自适应锐化
        adaptive_detail = detail_layer * protection_mask_3d
        sharpened = image_float + strength * adaptive_detail

        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def unsharp_masking(self, image: np.ndarray,
                       params: Optional[UnsharpMaskingParams] = None) -> np.ndarray:
        """
        🌟 统一的钝化蒙版接口：根据参数选择最适合的艺术表达

        Args:
            image: 输入图像
            params: 锐化参数配置

        Returns:
            锐化后的艺术作品
        """
        p = params or self.params

        if p.adaptive:
            return self.adaptive_unsharp_masking(image, p.strength, p.radius, p.edge_protect)
        elif p.threshold > 0:
            return self.threshold_unsharp_masking(image, p.strength, p.radius, p.threshold)
        else:
            return self.basic_unsharp_masking(image, p.strength, p.radius)

    def multi_scale_unsharp_masking(self, image: np.ndarray,
                                   scales: List[float] = [0.5, 1.0, 2.0],
                                   weights: List[float] = [0.3, 0.5, 0.2]) -> np.ndarray:
        """
        🌀 多尺度钝化蒙版：如同管弦乐团的和谐演奏

        在不同尺度上进行锐化，然后融合结果，就像音乐的多声部合奏

        Args:
            image: 输入图像
            scales: 不同的模糊半径尺度
            weights: 各尺度的权重

        Returns:
            多尺度融合的锐化结果
        """
        if len(scales) != len(weights):
            raise ValueError("🚫 尺度数量必须与权重数量相匹配")

        # 归一化权重
        weights = np.array(weights)
        weights = weights / np.sum(weights)

        image_float = image.astype(np.float32)
        result = np.zeros_like(image_float)

        for scale, weight in zip(scales, weights):
            # 🎵 在每个尺度上演奏锐化的旋律
            sharpened = self.basic_unsharp_masking(image, 1.5, scale)
            result += weight * sharpened.astype(np.float32)

        return np.clip(result, 0, 255).astype(np.uint8)

    def artistic_showcase(self, image: np.ndarray, save_path: Optional[str] = None) -> None:
        """
        🎭 锐化艺术展示：展现不同风格的锐化美学

        如同艺术馆中的作品展览，展示锐化的多种可能性
        """
        print("🎨 开始创作锐化艺术作品...")

        # 🎨 创建不同风格的锐化作品
        effects = {
            "📷 原始朦胧": image,
            "🌱 纯真锐化": self.basic_unsharp_masking(image, 1.5, 1.0),
            "🔥 激情锐化": self.basic_unsharp_masking(image, 3.0, 1.5),
            "🎯 智慧选择": self.threshold_unsharp_masking(image, 2.0, 1.0, 15.0),
            "🧘 禅师境界": self.adaptive_unsharp_masking(image, 2.0, 1.2, 0.7),
            "⚡ 细节强化": self.threshold_unsharp_masking(image, 2.5, 0.8, 5.0),
            "🌀 多尺度": self.multi_scale_unsharp_masking(image),
            "🎭 艺术级": self.adaptive_unsharp_masking(image, 2.5, 1.5, 0.8)
        }

        # 🖼️ 创建艺术画廊
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('🖌️ 钝化蒙版艺术馆：逆向思维的锐化美学', fontsize=16, fontweight='bold')

        for i, (title, effect_image) in enumerate(effects.items()):
            row, col = i // 4, i % 4
            axes[row, col].imshow(cv2.cvtColor(effect_image, cv2.COLOR_BGR2RGB))
            axes[row, col].set_title(title, fontsize=11)
            axes[row, col].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 锐化艺术展示已保存至: {save_path}")

        plt.show()

    def edge_preservation_analysis(self, image: np.ndarray) -> None:
        """
        🔍 边缘保护分析：展示自适应算法如何智慧地保护边缘
        """
        print("🔍 分析边缘保护机制...")

        # 转换为灰度图进行边缘分析
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 计算边缘强度
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)

        # 计算保护权重
        max_edge = np.max(edges)
        edge_norm = edges / max_edge if max_edge > 0 else edges
        protection_mask = 1.0 - 0.7 * np.power(edge_norm, 0.5)

        # 应用不同的锐化方法
        basic_sharp = self.basic_unsharp_masking(image, 2.5, 1.0)
        adaptive_sharp = self.adaptive_unsharp_masking(image, 2.5, 1.0, 0.7)

        # 可视化对比
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('🔍 边缘保护机制分析', fontsize=14, fontweight='bold')

        axes[0, 0].imshow(gray, cmap='gray')
        axes[0, 0].set_title('📷 原始图像')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(edges, cmap='hot')
        axes[0, 1].set_title('⚡ 边缘检测')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(protection_mask, cmap='viridis')
        axes[0, 2].set_title('🛡️ 保护权重')
        axes[0, 2].axis('off')

        axes[1, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('🌟 原始图像')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(cv2.cvtColor(basic_sharp, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('🌱 基础锐化')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(cv2.cvtColor(adaptive_sharp, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title('🧘 自适应锐化')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.show()

    def interactive_unsharp_masking(self, image: np.ndarray) -> None:
        """
        🎮 交互式钝化蒙版：实时调整参数体验锐化魔法

        Args:
            image: 输入图像
        """
        try:
            from matplotlib.widgets import Slider, RadioButtons
        except ImportError:
            print("❌ 需要matplotlib.widgets模块进行交互式演示")
            return

        fig = plt.figure(figsize=(16, 8))

        # 创建子图布局
        ax_original = plt.subplot2grid((3, 4), (0, 0), rowspan=2, colspan=2)
        ax_result = plt.subplot2grid((3, 4), (0, 2), rowspan=2, colspan=2)
        ax_method = plt.subplot2grid((3, 4), (2, 0))

        # 显示原图
        ax_original.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax_original.set_title('📷 原始图像', fontsize=12)
        ax_original.axis('off')

        # 初始效果
        initial_result = self.basic_unsharp_masking(image, 1.5, 1.0)
        im_result = ax_result.imshow(cv2.cvtColor(initial_result, cv2.COLOR_BGR2RGB))
        ax_result.set_title('🖌️ 锐化效果', fontsize=12)
        ax_result.axis('off')

        # 方法选择
        methods = ['基础', '阈值', '自适应']
        radio = RadioButtons(ax_method, methods, active=0)
        ax_method.set_title('🎭 锐化方法')

        # 创建滑块
        ax_strength = plt.axes([0.15, 0.1, 0.2, 0.03])
        ax_radius = plt.axes([0.45, 0.1, 0.2, 0.03])
        ax_threshold = plt.axes([0.75, 0.1, 0.2, 0.03])
        ax_edge_protect = plt.axes([0.15, 0.05, 0.2, 0.03])

        slider_strength = Slider(ax_strength, '强度', 0.1, 5.0, valinit=1.5)
        slider_radius = Slider(ax_radius, '半径', 0.1, 5.0, valinit=1.0)
        slider_threshold = Slider(ax_threshold, '阈值', 0.0, 50.0, valinit=10.0)
        slider_edge_protect = Slider(ax_edge_protect, '边缘保护', 0.0, 1.0, valinit=0.5)

        def update(_):
            """更新锐化效果"""
            method = methods[methods.index(radio.value_selected)]
            strength = slider_strength.val
            radius = slider_radius.val
            threshold = slider_threshold.val
            edge_protect = slider_edge_protect.val

            if method == '基础':
                result = self.basic_unsharp_masking(image, strength, radius)
            elif method == '阈值':
                result = self.threshold_unsharp_masking(image, strength, radius, threshold)
            else:  # 自适应
                result = self.adaptive_unsharp_masking(image, strength, radius, edge_protect)

            im_result.set_data(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            ax_result.set_title(f'🖌️ {method}锐化效果')
            fig.canvas.draw()

        # 绑定事件
        slider_strength.on_changed(update)
        slider_radius.on_changed(update)
        slider_threshold.on_changed(update)
        slider_edge_protect.on_changed(update)
        radio.on_clicked(update)

        plt.tight_layout()
        plt.show()

    def sharpness_metrics_analysis(self, image: np.ndarray) -> Dict[str, float]:
        """
        📊 锐度指标分析：量化评估不同锐化方法的效果

        Args:
            image: 输入图像

        Returns:
            不同方法的锐度指标字典
        """
        def calculate_sharpness(img: np.ndarray) -> float:
            """计算图像锐度（基于拉普拉斯算子方差）"""
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            return laplacian.var()

        methods = {
            '原始图像': image,
            '基础锐化': self.basic_unsharp_masking(image, 1.5, 1.0),
            '阈值锐化': self.threshold_unsharp_masking(image, 1.5, 1.0, 10.0),
            '自适应锐化': self.adaptive_unsharp_masking(image, 1.5, 1.0, 0.5),
            '多尺度锐化': self.multi_scale_unsharp_masking(image)
        }

        metrics = {}
        for name, img in methods.items():
            sharpness = calculate_sharpness(img)
            metrics[name] = sharpness
            print(f"📈 {name}: 锐度值 = {sharpness:.2f}")

        return metrics

    def performance_test(self, image_sizes: List[Tuple[int, int]] = None) -> Dict[str, float]:
        """
        ⚡ 性能测试：评估不同锐化方法的处理速度

        Args:
            image_sizes: 测试的图像尺寸列表

        Returns:
            性能测试结果字典
        """
        if image_sizes is None:
            image_sizes = [(256, 256), (512, 512), (1024, 1024)]

        results = {}

        print("🚀 开始锐化性能测试...")
        print("=" * 60)

        for width, height in image_sizes:
            # 创建测试图像
            test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

            # 测试不同方法
            methods = {
                '基础锐化': lambda img: self.basic_unsharp_masking(img, 1.5, 1.0),
                '阈值锐化': lambda img: self.threshold_unsharp_masking(img, 1.5, 1.0, 10.0),
                '自适应锐化': lambda img: self.adaptive_unsharp_masking(img, 1.5, 1.0, 0.5)
            }

            print(f"📊 图像尺寸: {width}x{height}")

            for method_name, method_func in methods.items():
                start_time = time.time()
                _ = method_func(test_image)
                processing_time = time.time() - start_time

                key = f"{method_name}_{width}x{height}"
                results[key] = processing_time

                print(f"  🖌️ {method_name}: {processing_time:.3f}秒")

            print("-" * 40)

        print("✅ 性能测试完成")
        return results

def create_unsharp_masking_demo():
    """🎯 创建钝化蒙版演示程序"""

    def process_image_interactive():
        """交互式图像处理"""
        while True:
            print("\n" + "="*60)
            print("🖌️ 钝化蒙版艺术家 - 交互式演示")
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

            # 创建锐化艺术家
            artist = UnsharpMaskingArtist()

            print("\n🎨 请选择锐化方法:")
            print("1. 🌱 基础锐化")
            print("2. 🎯 阈值锐化")
            print("3. 🧘 自适应锐化")
            print("4. 🌀 多尺度锐化")
            print("5. 🎭 艺术展示")
            print("6. 🔍 边缘保护分析")
            print("7. 🎮 交互式调节")
            print("8. 📊 锐度分析")

            choice = input("请选择 (1-8): ").strip()

            try:
                if choice == '1':
                    strength = float(input("🌱 锐化强度 [0.1-5.0, 默认1.5]: ") or "1.5")
                    radius = float(input("🌱 模糊半径 [0.1-5.0, 默认1.0]: ") or "1.0")
                    result = artist.basic_unsharp_masking(image, strength, radius)
                elif choice == '2':
                    strength = float(input("🎯 锐化强度 [0.1-5.0, 默认1.5]: ") or "1.5")
                    radius = float(input("🎯 模糊半径 [0.1-5.0, 默认1.0]: ") or "1.0")
                    threshold = float(input("🎯 锐化阈值 [0-50, 默认10]: ") or "10")
                    result = artist.threshold_unsharp_masking(image, strength, radius, threshold)
                elif choice == '3':
                    strength = float(input("🧘 锐化强度 [0.1-5.0, 默认1.5]: ") or "1.5")
                    radius = float(input("🧘 模糊半径 [0.1-5.0, 默认1.0]: ") or "1.0")
                    edge_protect = float(input("🧘 边缘保护 [0-1, 默认0.5]: ") or "0.5")
                    result = artist.adaptive_unsharp_masking(image, strength, radius, edge_protect)
                elif choice == '4':
                    result = artist.multi_scale_unsharp_masking(image)
                elif choice == '5':
                    artist.artistic_showcase(image)
                    continue
                elif choice == '6':
                    artist.edge_preservation_analysis(image)
                    continue
                elif choice == '7':
                    artist.interactive_unsharp_masking(image)
                    continue
                elif choice == '8':
                    artist.sharpness_metrics_analysis(image)
                    continue
                else:
                    print("❌ 无效选择")
                    continue

                # 显示结果
                comparison = np.hstack([image, result])
                cv2.imshow("Unsharp Masking (Original | Sharpened)", comparison)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # 询问是否保存
                save_choice = input("\n💾 是否保存结果? (y/n): ").strip().lower()
                if save_choice == 'y':
                    output_path = input("📁 输入保存路径 (默认: sharpened_result.jpg): ").strip() or "sharpened_result.jpg"
                    cv2.imwrite(output_path, result)
                    print(f"✅ 结果已保存至: {output_path}")

            except ValueError:
                print("❌ 参数格式错误")
            except Exception as e:
                print(f"❌ 处理出错: {e}")

    def batch_process_demo():
        """批量处理演示"""
        print("\n" + "="*60)
        print("🚀 批量钝化蒙版处理演示")
        print("="*60)

        input_dir = input("📁 输入图像目录路径: ").strip()
        if not os.path.exists(input_dir):
            print("❌ 目录不存在")
            return

        output_dir = input("📁 输出目录路径: ").strip() or "sharpened_results"
        os.makedirs(output_dir, exist_ok=True)

        # 获取图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = [f for f in os.listdir(input_dir)
                      if Path(f).suffix.lower() in image_extensions]

        if not image_files:
            print("❌ 未找到图像文件")
            return

        print(f"📸 找到 {len(image_files)} 张图像")

        # 创建锐化艺术家
        artist = UnsharpMaskingArtist()

        # 批量处理
        for i, filename in enumerate(image_files):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"sharpened_{filename}")

            print(f"🎨 处理 ({i+1}/{len(image_files)}): {filename}")

            image = cv2.imread(input_path)
            if image is not None:
                result = artist.adaptive_unsharp_masking(image, 1.8, 1.2, 0.6)
                cv2.imwrite(output_path, result)
                print(f"✅ 已保存: {output_path}")
            else:
                print(f"❌ 无法读取: {filename}")

        print(f"\n🎉 批量处理完成！结果保存在: {output_dir}")

    # 主菜单
    while True:
        print("\n" + "="*70)
        print("🖌️ 钝化蒙版艺术家 - 逆向思维的锐化美学")
        print("="*70)
        print("1. 📷 交互式单图处理")
        print("2. 🚀 批量图像处理")
        print("3. 🎭 艺术效果展示")
        print("4. 🎮 交互式参数调节")
        print("5. 📊 性能测试")
        print("6. 🔍 锐度分析")
        print("0. 👋 退出程序")
        print("="*70)

        choice = input("请选择功能 (0-6): ").strip()

        if choice == '0':
            print("👋 感谢体验钝化蒙版艺术家！")
            print("愿你的图像如锐化后般清晰美丽！ ✨")
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
                    artist = UnsharpMaskingArtist()
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
                    artist = UnsharpMaskingArtist()
                    artist.interactive_unsharp_masking(image)
                else:
                    print("❌ 无法读取图像")
            else:
                print("❌ 文件不存在")
        elif choice == '5':
            artist = UnsharpMaskingArtist()
            artist.performance_test()
        elif choice == '6':
            image_path = input("📷 请输入图像路径: ").strip()
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                if image is not None:
                    artist = UnsharpMaskingArtist()
                    artist.sharpness_metrics_analysis(image)
                else:
                    print("❌ 无法读取图像")
            else:
                print("❌ 文件不存在")
        else:
            print("❌ 无效选择，请重新输入")

def main():
    """🌟 主函数：展示钝化蒙版的艺术魅力"""
    parser = argparse.ArgumentParser(description="🖌️ 钝化蒙版 - 逆向思维的锐化艺术")
    parser.add_argument("--input", "-i", type=str, help="输入图像路径")
    parser.add_argument("--output", "-o", type=str, help="输出图像路径")
    parser.add_argument("--method", "-m", type=str, default="basic",
                       choices=["basic", "threshold", "adaptive", "multiscale"],
                       help="锐化方法")
    parser.add_argument("--strength", "-s", type=float, default=1.5, help="锐化强度 (0.1-5.0)")
    parser.add_argument("--radius", "-r", type=float, default=1.0, help="模糊半径 (0.1-5.0)")
    parser.add_argument("--threshold", "-t", type=float, default=10.0, help="锐化阈值 (0-50)")
    parser.add_argument("--edge-protect", "-e", type=float, default=0.5, help="边缘保护 (0-1)")
    parser.add_argument("--demo", action="store_true", help="启动演示模式")
    parser.add_argument("--showcase", action="store_true", help="显示艺术展示")
    parser.add_argument("--interactive", action="store_true", help="交互式参数调节")
    parser.add_argument("--analysis", action="store_true", help="锐度分析")
    parser.add_argument("--performance", action="store_true", help="运行性能测试")

    args = parser.parse_args()

    if args.demo:
        create_unsharp_masking_demo()
        return

    if not args.input:
        print("🚫 请提供输入图像路径，或使用 --demo 启动演示模式")
        print("💡 使用示例: python unsharp_masking.py -i image.jpg -o result.jpg")
        print("💡 演示模式: python unsharp_masking.py --demo")
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

    # 创建锐化艺术家
    artist = UnsharpMaskingArtist()

    if args.performance:
        # 性能测试
        artist.performance_test()
        return

    if args.showcase:
        # 艺术展示
        save_path = args.output.replace('.jpg', '_showcase.png') if args.output else None
        artist.artistic_showcase(image, save_path)
        return

    if args.interactive:
        # 交互式调节
        artist.interactive_unsharp_masking(image)
        return

    if args.analysis:
        # 锐度分析
        artist.sharpness_metrics_analysis(image)
        return

    # 应用指定的锐化方法
    print(f"🎨 应用{args.method}锐化...")

    if args.method == "basic":
        result = artist.basic_unsharp_masking(image, args.strength, args.radius)
    elif args.method == "threshold":
        result = artist.threshold_unsharp_masking(image, args.strength, args.radius, args.threshold)
    elif args.method == "adaptive":
        result = artist.adaptive_unsharp_masking(image, args.strength, args.radius, args.edge_protect)
    elif args.method == "multiscale":
        result = artist.multi_scale_unsharp_masking(image)

    if args.output:
        cv2.imwrite(args.output, result)
        print(f"✅ 锐化艺术作品已保存至: {args.output}")
    else:
        # 显示对比
        comparison = np.hstack([image, result])
        cv2.imshow("Unsharp Masking (Original | Sharpened)", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()