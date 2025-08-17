"""
🌗 光影的哲学：同态滤波算法的明暗辩证法Python实现

这个模块实现了完整的同态滤波算法，包含：
- 同态滤波器设计：智慧的选择性调节器
- 频域滤波处理：在频率海洋中精确导航
- 对数指数变换：乘性到加性的华丽转身
- 增强型同态滤波：细节与光照的完美平衡
- 交互式演示：实时体验光影的魔法
- 性能测试：评估不同参数的处理效率

作者：GlimmerLab
创建时间：2024年
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
import argparse
import os
from pathlib import Path
import time
import math
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class HomomorphicParams:
    """🌗 同态滤波的艺术配置参数"""
    gamma_low: float = 0.5      # 低频增益（光照压缩）(0.1-1.0)
    gamma_high: float = 2.0     # 高频增益（细节增强）(1.0-5.0)
    cutoff: float = 15.0        # 截止频率 (5.0-50.0)
    c: float = 1.0              # 控制参数 (0.1-3.0)

    def __post_init__(self):
        """参数有效性检查"""
        assert 0.1 <= self.gamma_low <= 1.0, "低频增益必须在[0.1, 1.0]范围内"
        assert 1.0 <= self.gamma_high <= 5.0, "高频增益必须在[1.0, 5.0]范围内"
        assert 5.0 <= self.cutoff <= 50.0, "截止频率必须在[5.0, 50.0]范围内"
        assert 0.1 <= self.c <= 3.0, "控制参数必须在[0.1, 3.0]范围内"

class HomomorphicArtist:
    """🌈 同态滤波艺术家：光影重构的数字魔法师"""

    def __init__(self, params: Optional[HomomorphicParams] = None):
        """
        🌟 初始化我们的光影魔术师
        每个参数都是调和明暗的智慧
        """
        self.params = params or HomomorphicParams()

    def create_homomorphic_filter(self, size: Tuple[int, int],
                                 gamma_low: float, gamma_high: float,
                                 cutoff: float, c: float) -> np.ndarray:
        """
        🎨 创建同态滤波器：智慧选择的数学诗篇

        就像调色师为每种颜色调配不同的亮度

        Args:
            size: 滤波器尺寸 (rows, cols)
            gamma_low: 低频增益，压缩光照变化
            gamma_high: 高频增益，增强细节对比
            cutoff: 截止频率，决定处理的边界
            c: 控制参数，影响过渡的平滑度

        Returns:
            充满智慧的同态滤波器
        """
        rows, cols = size
        filter_matrix = np.zeros((rows, cols), dtype=np.float32)

        # 🎯 计算频域中心点
        center_row, center_col = rows // 2, cols // 2

        # 📏 截止频率的平方，减少重复计算
        d0_squared = cutoff * cutoff

        # 🌟 构建同态滤波器：每个频率点都有其独特的处理方式
        for u in range(rows):
            for v in range(cols):
                # 📐 计算到中心的距离平方
                du = u - center_row
                dv = v - center_col
                d_squared = du * du + dv * dv

                # ✨ 应用同态滤波器公式：数学的艺术表达
                # H(u,v) = (γ_H - γ_L)[1 - exp(-c * D²/D₀²)] + γ_L
                h = (gamma_high - gamma_low) * \
                    (1.0 - np.exp(-c * d_squared / d0_squared)) + gamma_low

                filter_matrix[u, v] = h

        return filter_matrix

    def dft_filter(self, image: np.ndarray, filter_matrix: np.ndarray) -> np.ndarray:
        """
        🌊 频域滤波：在频率海洋中的精确导航

        Args:
            image: 对数域中的图像
            filter_matrix: 同态滤波器

        Returns:
            滤波后的频域图像
        """
        # 🌟 执行快速傅里叶变换：进入频率的世界
        f_transform = np.fft.fft2(image)
        f_shifted = np.fft.fftshift(f_transform)

        # 🎨 应用同态滤波器：在频域中进行智慧的调节
        filtered_shifted = f_shifted * filter_matrix

        # 🌈 返回空域：重回我们熟悉的图像世界
        f_ishifted = np.fft.ifftshift(filtered_shifted)
        filtered_image = np.fft.ifft2(f_ishifted)

        # 🎭 取实部：去掉数学计算带来的虚数噪音
        return np.real(filtered_image)

    def homomorphic_filter(self, image: np.ndarray,
                         params: Optional[HomomorphicParams] = None) -> np.ndarray:
        """
        🌗 同态滤波主函数：光影重构的完整艺术

        Args:
            image: 输入图像
            params: 滤波参数

        Returns:
            光影和谐的艺术作品
        """
        if image is None or image.size == 0:
            raise ValueError("🚫 输入图像为空，请提供有效的图像")

        p = params or self.params

        if len(image.shape) == 2:
            # 🌙 灰度图像处理：单色世界的光影调和
            return self._process_grayscale(image, p)
        elif len(image.shape) == 3:
            # 🌈 彩色图像处理：在保持色彩的同时调整明暗
            return self._process_color(image, p)
        else:
            raise ValueError("🚫 不支持的图像格式，请提供灰度或彩色图像")

    def _process_grayscale(self, image: np.ndarray,
                          params: HomomorphicParams) -> np.ndarray:
        """
        🌙 处理灰度图像的私有方法
        """
        # 1️⃣ 转换为浮点型并避免log(0)
        image_float = image.astype(np.float32) + 1.0

        # 2️⃣ 对数变换：进入可分离的数学世界
        log_image = np.log(image_float)

        # 3️⃣ 创建同态滤波器
        filter_matrix = self.create_homomorphic_filter(
            log_image.shape, params.gamma_low, params.gamma_high,
            params.cutoff, params.c
        )

        # 4️⃣ 频域滤波：智慧的选择性处理
        filtered_log = self.dft_filter(log_image, filter_matrix)

        # 5️⃣ 指数变换：重返现实世界
        filtered_image = np.exp(filtered_log) - 1.0

        # 6️⃣ 归一化到[0, 255]范围
        filtered_image = np.clip(filtered_image, 0, None)

        # 动态范围调整
        if filtered_image.max() > 0:
            filtered_image = (filtered_image / filtered_image.max() * 255)

        return filtered_image.astype(np.uint8)

    def _process_color(self, image: np.ndarray,
                      params: HomomorphicParams) -> np.ndarray:
        """
        🌈 处理彩色图像的私有方法
        """
        # 转换到YCrCb颜色空间：分离亮度与色彩
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        # 分离通道
        y, cr, cb = cv2.split(ycrcb)

        # 只对亮度通道应用同态滤波
        y_filtered = self._process_grayscale(y, params)

        # 重新合并通道
        ycrcb_filtered = cv2.merge([y_filtered, cr, cb])

        # 转换回BGR颜色空间
        result = cv2.cvtColor(ycrcb_filtered, cv2.COLOR_YCrCb2BGR)

        return result

    def enhanced_homomorphic_filter(self, image: np.ndarray,
                                  params: Optional[HomomorphicParams] = None,
                                  edge_enhancement: float = 0.3) -> np.ndarray:
        """
        ✨ 增强型同态滤波：在光影调和的基础上加强边缘

        Args:
            image: 输入图像
            params: 滤波参数
            edge_enhancement: 边缘增强强度 (0-1)

        Returns:
            细节更加丰富的艺术作品
        """
        # 🌟 先应用标准同态滤波
        homomorphic_result = self.homomorphic_filter(image, params)

        if edge_enhancement <= 0:
            return homomorphic_result

        # 🎨 额外的边缘增强处理
        if len(image.shape) == 2:
            # 灰度图像的边缘增强
            blurred = cv2.GaussianBlur(homomorphic_result, (0, 0), 3.0)
            enhanced = cv2.addWeighted(homomorphic_result, 1 + edge_enhancement,
                                     blurred, -edge_enhancement, 0)
        else:
            # 彩色图像的边缘增强
            blurred = cv2.GaussianBlur(homomorphic_result, (0, 0), 3.0)
            enhanced = cv2.addWeighted(homomorphic_result, 1 + edge_enhancement,
                                     blurred, -edge_enhancement, 0)

        return np.clip(enhanced, 0, 255).astype(np.uint8)

    def visualize_filter(self, size: Tuple[int, int] = (256, 256),
                        params: Optional[HomomorphicParams] = None) -> None:
        """
        🔍 可视化同态滤波器：展示频域中的智慧分布

        Args:
            size: 滤波器尺寸
            params: 滤波参数
        """
        p = params or self.params

        # 创建同态滤波器
        filter_matrix = self.create_homomorphic_filter(
            size, p.gamma_low, p.gamma_high, p.cutoff, p.c
        )

        # 可视化滤波器
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 2D视图
        im1 = axes[0].imshow(filter_matrix, cmap='jet')
        axes[0].set_title('🌗 同态滤波器 2D 视图')
        axes[0].set_xlabel('频率 u')
        axes[0].set_ylabel('频率 v')
        plt.colorbar(im1, ax=axes[0])

        # 中心横截面
        center_row = size[0] // 2
        center_line = filter_matrix[center_row, :]
        axes[1].plot(center_line)
        axes[1].set_title('🌊 滤波器中心横截面')
        axes[1].set_xlabel('频率索引')
        axes[1].set_ylabel('滤波器响应')
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f'🎨 同态滤波器可视化 (γ_L={p.gamma_low}, γ_H={p.gamma_high}, D₀={p.cutoff}, c={p.c})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def artistic_showcase(self, image: np.ndarray, save_path: Optional[str] = None) -> None:
        """
        🎭 同态滤波艺术展示：展现光影的多种表达

        如同光影艺术馆的作品展览，展示同态滤波的无限可能
        """
        print("🎨 开始创作同态滤波艺术作品...")

        # 🎨 创建不同风格的光影作品
        effects = {
            "📷 原始图像": image,
            "🌗 标准同态滤波": self.homomorphic_filter(image),
            "✨ 增强型滤波": self.enhanced_homomorphic_filter(image, edge_enhancement=0.4),
            "🌅 细节增强": self.homomorphic_filter(image, HomomorphicParams(
                gamma_low=0.3, gamma_high=3.0, cutoff=20.0, c=1.5
            )),
            "🌙 光照压缩": self.homomorphic_filter(image, HomomorphicParams(
                gamma_low=0.2, gamma_high=1.5, cutoff=10.0, c=2.0
            )),
            "⚡ 极致对比": self.homomorphic_filter(image, HomomorphicParams(
                gamma_low=0.1, gamma_high=4.0, cutoff=25.0, c=1.8
            )),
            "🌸 柔和均衡": self.homomorphic_filter(image, HomomorphicParams(
                gamma_low=0.6, gamma_high=1.8, cutoff=15.0, c=0.8
            )),
            "🔥 强烈对比": self.enhanced_homomorphic_filter(image, HomomorphicParams(
                gamma_low=0.2, gamma_high=3.5, cutoff=30.0, c=2.5
            ), edge_enhancement=0.6)
        }

        # 🖼️ 创造光影艺术馆
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('🌗 同态滤波艺术馆：光影的哲学对话', fontsize=16, fontweight='bold')

        for i, (title, effect_image) in enumerate(effects.items()):
            row, col = i // 4, i % 4
            axes[row, col].imshow(cv2.cvtColor(effect_image, cv2.COLOR_BGR2RGB))
            axes[row, col].set_title(title, fontsize=11)
            axes[row, col].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 同态滤波艺术展示已保存至: {save_path}")

        plt.show()

    def parameter_analysis(self, image: np.ndarray) -> None:
        """
        🔍 参数影响分析：展示不同参数对结果的影响
        """
        print("🔍 分析同态滤波参数对图像的影响...")

        # 测试不同的低频增益
        gamma_low_values = [0.1, 0.3, 0.5, 0.8]
        gamma_low_results = []

        for gamma_low in gamma_low_values:
            params = HomomorphicParams(gamma_low=gamma_low, gamma_high=2.0, cutoff=15.0, c=1.0)
            result = self.homomorphic_filter(image, params)
            gamma_low_results.append((f"γ_L={gamma_low}", result))

        # 测试不同的高频增益
        gamma_high_values = [1.2, 2.0, 3.0, 4.0]
        gamma_high_results = []

        for gamma_high in gamma_high_values:
            params = HomomorphicParams(gamma_low=0.5, gamma_high=gamma_high, cutoff=15.0, c=1.0)
            result = self.homomorphic_filter(image, params)
            gamma_high_results.append((f"γ_H={gamma_high}", result))

        # 可视化对比
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('🔍 同态滤波参数影响分析', fontsize=14, fontweight='bold')

        # 显示低频增益影响
        for i, (title, img) in enumerate(gamma_low_results):
            axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[0, i].set_title(f"低频增益: {title}", fontsize=10)
            axes[0, i].axis('off')

        # 显示高频增益影响
        for i, (title, img) in enumerate(gamma_high_results):
            axes[1, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[1, i].set_title(f"高频增益: {title}", fontsize=10)
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.show()

    def step_by_step_demo(self, image: np.ndarray) -> None:
        """
        🎪 分步演示：展示同态滤波的处理过程
        """
        print("🎪 展示同态滤波的分步处理过程...")

        # 转换为灰度进行演示
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()

        # 逐步处理
        steps = {}

        # 步骤1：原图
        steps["📷 原始图像"] = gray_image

        # 步骤2：对数变换
        image_float = gray_image.astype(np.float32) + 1.0
        log_image = np.log(image_float)
        log_display = ((log_image - log_image.min()) / (log_image.max() - log_image.min()) * 255).astype(np.uint8)
        steps["🌟 对数变换"] = log_display

        # 步骤3：创建滤波器
        filter_matrix = self.create_homomorphic_filter(
            log_image.shape, self.params.gamma_low, self.params.gamma_high,
            self.params.cutoff, self.params.c
        )
        filter_display = ((filter_matrix - filter_matrix.min()) / (filter_matrix.max() - filter_matrix.min()) * 255).astype(np.uint8)
        steps["🎨 同态滤波器"] = filter_display

        # 步骤4：频域滤波
        filtered_log = self.dft_filter(log_image, filter_matrix)
        filtered_log_display = ((filtered_log - filtered_log.min()) / (filtered_log.max() - filtered_log.min()) * 255).astype(np.uint8)
        steps["🌊 频域滤波"] = filtered_log_display

        # 步骤5：指数变换
        filtered_image = np.exp(filtered_log) - 1.0
        filtered_image = np.clip(filtered_image, 0, None)
        if filtered_image.max() > 0:
            filtered_image = (filtered_image / filtered_image.max() * 255)
        final_result = filtered_image.astype(np.uint8)
        steps["✨ 指数变换"] = final_result

        # 步骤6：对比结果
        steps["🌈 对比增强"] = final_result

        # 可视化处理过程
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🎪 同态滤波分步处理过程', fontsize=16, fontweight='bold')

        for i, (title, step_image) in enumerate(steps.items()):
            row, col = i // 3, i % 3
            axes[row, col].imshow(step_image, cmap='gray')
            axes[row, col].set_title(title, fontsize=12)
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.show()

    def interactive_homomorphic_filter(self, image: np.ndarray) -> None:
        """
        🎮 交互式同态滤波：实时调整参数体验光影魔法

        Args:
            image: 输入图像
        """
        try:
            from matplotlib.widgets import Slider
        except ImportError:
            print("❌ 需要matplotlib.widgets模块进行交互式演示")
            return

        fig = plt.figure(figsize=(16, 10))

        # 创建子图布局
        ax_original = plt.subplot2grid((4, 4), (0, 0), rowspan=2, colspan=2)
        ax_result = plt.subplot2grid((4, 4), (0, 2), rowspan=2, colspan=2)

        # 显示原图
        ax_original.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax_original.set_title('📷 原始图像', fontsize=12)
        ax_original.axis('off')

        # 初始效果
        initial_result = self.homomorphic_filter(image)
        im_result = ax_result.imshow(cv2.cvtColor(initial_result, cv2.COLOR_BGR2RGB))
        ax_result.set_title('🌗 同态滤波效果', fontsize=12)
        ax_result.axis('off')

        # 创建滑块控件
        ax_gamma_low = plt.axes([0.15, 0.4, 0.3, 0.03])
        ax_gamma_high = plt.axes([0.55, 0.4, 0.3, 0.03])
        ax_cutoff = plt.axes([0.15, 0.35, 0.3, 0.03])
        ax_c = plt.axes([0.55, 0.35, 0.3, 0.03])
        ax_edge = plt.axes([0.15, 0.3, 0.3, 0.03])

        slider_gamma_low = Slider(ax_gamma_low, 'γ_L (低频增益)', 0.1, 1.0, valinit=0.5)
        slider_gamma_high = Slider(ax_gamma_high, 'γ_H (高频增益)', 1.0, 5.0, valinit=2.0)
        slider_cutoff = Slider(ax_cutoff, 'D₀ (截止频率)', 5.0, 50.0, valinit=15.0)
        slider_c = Slider(ax_c, 'c (控制参数)', 0.1, 3.0, valinit=1.0)
        slider_edge = Slider(ax_edge, '边缘增强', 0.0, 1.0, valinit=0.0)

        def update(_):
            """更新同态滤波效果"""
            params = HomomorphicParams(
                gamma_low=slider_gamma_low.val,
                gamma_high=slider_gamma_high.val,
                cutoff=slider_cutoff.val,
                c=slider_c.val
            )

            if slider_edge.val > 0:
                result = self.enhanced_homomorphic_filter(image, params, slider_edge.val)
            else:
                result = self.homomorphic_filter(image, params)

            im_result.set_data(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            ax_result.set_title('🌗 同态滤波效果')
            fig.canvas.draw()

        # 绑定事件
        slider_gamma_low.on_changed(update)
        slider_gamma_high.on_changed(update)
        slider_cutoff.on_changed(update)
        slider_c.on_changed(update)
        slider_edge.on_changed(update)

        plt.tight_layout()
        plt.show()

    def performance_test(self, image_sizes: List[Tuple[int, int]] = None) -> Dict[str, float]:
        """
        ⚡ 性能测试：评估不同同态滤波方法的处理速度

        Args:
            image_sizes: 测试的图像尺寸列表

        Returns:
            性能测试结果字典
        """
        if image_sizes is None:
            image_sizes = [(256, 256), (512, 512), (1024, 1024)]

        results = {}

        print("🚀 开始同态滤波性能测试...")
        print("=" * 60)

        for width, height in image_sizes:
            # 创建测试图像
            test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

            # 测试不同方法
            methods = {
                '标准同态滤波': lambda img: self.homomorphic_filter(img),
                '增强型同态滤波': lambda img: self.enhanced_homomorphic_filter(img),
                '高精度滤波': lambda img: self.homomorphic_filter(img, HomomorphicParams(
                    gamma_low=0.3, gamma_high=3.0, cutoff=25.0, c=1.5
                )),
                '快速滤波': lambda img: self.homomorphic_filter(img, HomomorphicParams(
                    gamma_low=0.5, gamma_high=2.0, cutoff=10.0, c=1.0
                ))
            }

            print(f"📊 图像尺寸: {width}x{height}")

            for method_name, method_func in methods.items():
                start_time = time.time()
                _ = method_func(test_image)
                processing_time = time.time() - start_time

                key = f"{method_name}_{width}x{height}"
                results[key] = processing_time

                print(f"  🌗 {method_name}: {processing_time:.3f}秒")

            print("-" * 40)

        print("✅ 性能测试完成")
        return results

def create_homomorphic_demo():
    """🎯 创建同态滤波演示程序"""

    def process_image_interactive():
        """交互式图像处理"""
        while True:
            print("\n" + "="*60)
            print("🌗 同态滤波艺术家 - 交互式演示")
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

            # 创建同态滤波艺术家
            artist = HomomorphicArtist()

            print("\n🎨 请选择同态滤波选项:")
            print("1. 🌗 标准同态滤波")
            print("2. ✨ 增强型同态滤波")
            print("3. 🌅 细节增强滤波")
            print("4. 🌙 光照压缩滤波")
            print("5. 🎨 艺术展示")
            print("6. 🎪 分步演示")
            print("7. 🔍 参数分析")
            print("8. 🎮 交互式调节")
            print("9. 🔬 滤波器可视化")

            choice = input("请选择 (1-9): ").strip()

            try:
                if choice == '1':
                    gamma_low = float(input("🌗 低频增益 [0.1-1.0, 默认0.5]: ") or "0.5")
                    gamma_high = float(input("🌗 高频增益 [1.0-5.0, 默认2.0]: ") or "2.0")
                    cutoff = float(input("🌗 截止频率 [5-50, 默认15]: ") or "15")
                    c = float(input("🌗 控制参数 [0.1-3.0, 默认1.0]: ") or "1.0")

                    params = HomomorphicParams(gamma_low, gamma_high, cutoff, c)
                    result = artist.homomorphic_filter(image, params)
                elif choice == '2':
                    edge_enhancement = float(input("✨ 边缘增强强度 [0-1, 默认0.3]: ") or "0.3")
                    result = artist.enhanced_homomorphic_filter(image, edge_enhancement=edge_enhancement)
                elif choice == '3':
                    params = HomomorphicParams(gamma_low=0.3, gamma_high=3.0, cutoff=20.0, c=1.5)
                    result = artist.homomorphic_filter(image, params)
                elif choice == '4':
                    params = HomomorphicParams(gamma_low=0.2, gamma_high=1.5, cutoff=10.0, c=2.0)
                    result = artist.homomorphic_filter(image, params)
                elif choice == '5':
                    artist.artistic_showcase(image)
                    continue
                elif choice == '6':
                    artist.step_by_step_demo(image)
                    continue
                elif choice == '7':
                    artist.parameter_analysis(image)
                    continue
                elif choice == '8':
                    artist.interactive_homomorphic_filter(image)
                    continue
                elif choice == '9':
                    artist.visualize_filter()
                    continue
                else:
                    print("❌ 无效选择")
                    continue

                # 显示结果
                comparison = np.hstack([image, result])
                cv2.imshow("Homomorphic Filter (Original | Filtered)", comparison)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # 询问是否保存
                save_choice = input("\n💾 是否保存结果? (y/n): ").strip().lower()
                if save_choice == 'y':
                    output_path = input("📁 输入保存路径 (默认: homomorphic_result.jpg): ").strip() or "homomorphic_result.jpg"
                    cv2.imwrite(output_path, result)
                    print(f"✅ 结果已保存至: {output_path}")

            except ValueError:
                print("❌ 参数格式错误")
            except Exception as e:
                print(f"❌ 处理出错: {e}")

    def batch_process_demo():
        """批量处理演示"""
        print("\n" + "="*60)
        print("🚀 批量同态滤波处理演示")
        print("="*60)

        input_dir = input("📁 输入图像目录路径: ").strip()
        if not os.path.exists(input_dir):
            print("❌ 目录不存在")
            return

        output_dir = input("📁 输出目录路径: ").strip() or "homomorphic_results"
        os.makedirs(output_dir, exist_ok=True)

        # 选择滤波类型
        print("\n🎨 选择同态滤波类型:")
        print("1. 标准滤波")
        print("2. 增强型滤波")
        print("3. 细节增强")
        print("4. 光照压缩")

        filter_choice = input("请选择 (1-4): ").strip()
        filter_map = {
            '1': (HomomorphicParams(), '标准滤波'),
            '2': (HomomorphicParams(), '增强型滤波'),  # 将在处理时添加边缘增强
            '3': (HomomorphicParams(gamma_low=0.3, gamma_high=3.0, cutoff=20.0, c=1.5), '细节增强'),
            '4': (HomomorphicParams(gamma_low=0.2, gamma_high=1.5, cutoff=10.0, c=2.0), '光照压缩')
        }

        if filter_choice not in filter_map:
            print("❌ 无效选择")
            return

        params, filter_name = filter_map[filter_choice]

        # 获取图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = [f for f in os.listdir(input_dir)
                      if Path(f).suffix.lower() in image_extensions]

        if not image_files:
            print("❌ 未找到图像文件")
            return

        print(f"📸 找到 {len(image_files)} 张图像")

        # 创建同态滤波艺术家
        artist = HomomorphicArtist()

        # 批量处理
        for i, filename in enumerate(image_files):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"homomorphic_{filename}")

            print(f"🎨 处理 ({i+1}/{len(image_files)}): {filename}")

            image = cv2.imread(input_path)
            if image is not None:
                if filter_choice == '2':  # 增强型滤波
                    result = artist.enhanced_homomorphic_filter(image, params)
                else:
                    result = artist.homomorphic_filter(image, params)

                cv2.imwrite(output_path, result)
                print(f"✅ 已保存: {output_path}")
            else:
                print(f"❌ 无法读取: {filename}")

        print(f"\n🎉 批量处理完成！{filter_name}结果保存在: {output_dir}")

    # 主菜单
    while True:
        print("\n" + "="*70)
        print("🌗 同态滤波艺术家 - 光影的哲学")
        print("="*70)
        print("1. 📷 交互式单图处理")
        print("2. 🚀 批量图像处理")
        print("3. 🎨 艺术效果展示")
        print("4. 🎪 分步制作演示")
        print("5. 🎮 交互式参数调节")
        print("6. 📊 性能测试")
        print("7. 🔍 参数影响分析")
        print("8. 🔬 滤波器可视化")
        print("0. 👋 退出程序")
        print("="*70)

        choice = input("请选择功能 (0-8): ").strip()

        if choice == '0':
            print("👋 感谢体验同态滤波艺术家！")
            print("愿你的世界如光影般和谐美好！ ✨")
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
                    artist = HomomorphicArtist()
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
                    artist = HomomorphicArtist()
                    artist.step_by_step_demo(image)
                else:
                    print("❌ 无法读取图像")
            else:
                print("❌ 文件不存在")
        elif choice == '5':
            image_path = input("📷 请输入图像路径: ").strip()
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                if image is not None:
                    artist = HomomorphicArtist()
                    artist.interactive_homomorphic_filter(image)
                else:
                    print("❌ 无法读取图像")
            else:
                print("❌ 文件不存在")
        elif choice == '6':
            artist = HomomorphicArtist()
            artist.performance_test()
        elif choice == '7':
            image_path = input("📷 请输入图像路径: ").strip()
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                if image is not None:
                    artist = HomomorphicArtist()
                    artist.parameter_analysis(image)
                else:
                    print("❌ 无法读取图像")
            else:
                print("❌ 文件不存在")
        elif choice == '8':
            artist = HomomorphicArtist()
            artist.visualize_filter()
        else:
            print("❌ 无效选择，请重新输入")

def main():
    """🌟 主函数：展示同态滤波的光影魔法"""
    parser = argparse.ArgumentParser(description="🌗 同态滤波 - 光影的哲学")
    parser.add_argument("--input", "-i", type=str, help="输入图像路径")
    parser.add_argument("--output", "-o", type=str, help="输出图像路径")
    parser.add_argument("--gamma-low", type=float, default=0.5, help="低频增益 (0.1-1.0)")
    parser.add_argument("--gamma-high", type=float, default=2.0, help="高频增益 (1.0-5.0)")
    parser.add_argument("--cutoff", type=float, default=15.0, help="截止频率 (5-50)")
    parser.add_argument("--c", type=float, default=1.0, help="控制参数 (0.1-3.0)")
    parser.add_argument("--edge-enhancement", type=float, default=0.0, help="边缘增强强度 (0-1)")
    parser.add_argument("--demo", action="store_true", help="启动演示模式")
    parser.add_argument("--showcase", action="store_true", help="显示艺术展示")
    parser.add_argument("--step-by-step", action="store_true", help="显示分步演示")
    parser.add_argument("--interactive", action="store_true", help="交互式参数调节")
    parser.add_argument("--analysis", action="store_true", help="参数影响分析")
    parser.add_argument("--performance", action="store_true", help="运行性能测试")
    parser.add_argument("--visualize-filter", action="store_true", help="可视化滤波器")

    args = parser.parse_args()

    if args.demo:
        create_homomorphic_demo()
        return

    if not args.input:
        print("🚫 请提供输入图像路径，或使用 --demo 启动演示模式")
        print("💡 使用示例: python homomorphic_filter.py -i image.jpg -o filtered.jpg")
        print("💡 演示模式: python homomorphic_filter.py --demo")
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

    # 创建同态滤波艺术家
    artist = HomomorphicArtist()

    if args.performance:
        # 性能测试
        artist.performance_test()
        return

    if args.showcase:
        # 艺术展示
        save_path = args.output.replace('.jpg', '_showcase.png') if args.output else None
        artist.artistic_showcase(image, save_path)
        return

    if args.step_by_step:
        # 分步演示
        artist.step_by_step_demo(image)
        return

    if args.interactive:
        # 交互式调节
        artist.interactive_homomorphic_filter(image)
        return

    if args.analysis:
        # 参数分析
        artist.parameter_analysis(image)
        return

    if args.visualize_filter:
        # 滤波器可视化
        params = HomomorphicParams(args.gamma_low, args.gamma_high, args.cutoff, args.c)
        artist.visualize_filter(params=params)
        return

    # 应用同态滤波
    print("🎨 应用同态滤波...")

    params = HomomorphicParams(
        gamma_low=args.gamma_low,
        gamma_high=args.gamma_high,
        cutoff=args.cutoff,
        c=args.c
    )

    if args.edge_enhancement > 0:
        result = artist.enhanced_homomorphic_filter(image, params, args.edge_enhancement)
    else:
        result = artist.homomorphic_filter(image, params)

    if args.output:
        cv2.imwrite(args.output, result)
        print(f"✅ 同态滤波艺术作品已保存至: {args.output}")
    else:
        # 显示对比
        comparison = np.hstack([image, result])
        cv2.imshow("Homomorphic Filter (Original | Filtered)", comparison)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()