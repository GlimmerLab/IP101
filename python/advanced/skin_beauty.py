"""
💄 数字美妆师：磨皮美白算法的Python实现

这个模块实现了智能的磨皮美白算法，包含：
- 肌肤检测：YCrCb颜色空间的智能识别
- 智能磨皮：双边滤波与高斯滤波的完美结合
- 自然美白：LAB颜色空间的亮度优化
- 光影平衡：HSV空间的明度调节
- 细节保留：高频信息的精确保护

作者：GlimmerLab
创建时间：2024年
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import argparse
import os
from pathlib import Path

@dataclass
class SkinBeautyParams:
    """💄 数字美妆的专业配置参数"""
    smoothing_factor: float = 0.5      # 磨皮强度 [0.0, 1.0]
    whitening_factor: float = 0.2      # 美白强度 [0.0, 1.0]
    detail_factor: float = 0.3         # 细节保留 [0.0, 1.0]
    bilateral_size: int = 9            # 双边滤波窗口大小
    bilateral_color: float = 30.0      # 色彩相似性阈值
    bilateral_space: float = 7.0       # 空间相似性阈值

    def __post_init__(self):
        """参数有效性检查"""
        assert 0.0 <= self.smoothing_factor <= 1.0, "磨皮强度必须在[0.0, 1.0]范围内"
        assert 0.0 <= self.whitening_factor <= 1.0, "美白强度必须在[0.0, 1.0]范围内"
        assert 0.0 <= self.detail_factor <= 1.0, "细节保留必须在[0.0, 1.0]范围内"
        assert self.bilateral_size % 2 == 1 and self.bilateral_size > 0, "双边滤波窗口必须为正奇数"

class DigitalBeautician:
    """💎 数字美妆师：用代码雕琢自然之美"""

    def __init__(self, params: Optional[SkinBeautyParams] = None):
        """
        🌟 初始化数字美妆师

        Args:
            params: 美妆参数配置，默认使用标准配置
        """
        self.params = params or SkinBeautyParams()

    def beautify(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        💫 主美颜流程：四个温柔的步骤

        Args:
            image: 输入的彩色图像 (BGR格式)

        Returns:
            Tuple[result, intermediate_results]: 美颜后的图像和中间结果

        Raises:
            ValueError: 输入图像格式不正确
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("🚫 输入图像必须是彩色图像（3通道BGR格式）")

        intermediate_results = {}
        result = image.copy()

        print("🔍 开始肌肤检测...")
        # 🔍 第一步：识别肌肤区域
        skin_mask = self._detect_skin(image)
        intermediate_results['skin_mask'] = skin_mask

        print("🌸 开始磨皮处理...")
        # 🌸 第二步：磨皮处理
        smoothed = self._smooth_skin(image, self.params.smoothing_factor)
        intermediate_results['smoothed'] = smoothed

        # 🎯 第三步：细节保留
        if self.params.detail_factor > 0:
            print("🔍 应用细节保留...")
            smoothed = self._preserve_details(image, smoothed, self.params.detail_factor)
            intermediate_results['detail_preserved'] = smoothed

        print("🎭 智能混合中...")
        # 🎭 第四步：智能混合（只对肌肤区域应用磨皮）
        result = np.where(skin_mask[..., np.newaxis] > 128, smoothed, result)
        intermediate_results['skin_blended'] = result

        # ☀️ 第五步：美白处理
        if self.params.whitening_factor > 0:
            print("☀️ 开始美白处理...")
            result = self._whiten_skin(result, skin_mask, self.params.whitening_factor)
            intermediate_results['whitened'] = result

        # 🌟 第六步：光影优化
        print("🌟 最终光影优化...")
        result = self._improve_lighting(result, 0.3 * self.params.whitening_factor)
        intermediate_results['final_result'] = result

        print("✨ 美颜处理完成！")
        return result, intermediate_results

    def _detect_skin(self, image: np.ndarray) -> np.ndarray:
        """
        👁️ 肌肤检测：在色彩的海洋中寻找温暖的肌理

        使用YCrCb颜色空间进行肌肤检测，该空间对肌肤色调敏感

        Args:
            image: 输入图像

        Returns:
            skin_mask: 肌肤区域掩码 (0-255)
        """
        # 🌈 转换到YCrCb颜色空间
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        # 🎯 提取Cr和Cb通道
        y, cr, cb = cv2.split(ycrcb)

        # 💫 肌肤色彩的数学密码：133≤Cr≤173, 77≤Cb≤127
        # 这些范围是基于大量肌肤样本统计得出的
        skin_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        skin_condition = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)

        # 添加亮度约束，避免过暗或过亮的区域
        brightness_condition = (y >= 30) & (y <= 230)

        skin_mask[skin_condition & brightness_condition] = 255

        # 🌸 形态学操作：让掩码更加优雅和连续
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        # 闭运算：填补小孔洞
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

        # 开运算：去除小噪点
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

        # 💎 高斯模糊：让边缘如丝绸般柔滑
        skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)

        return skin_mask

    def _smooth_skin(self, image: np.ndarray, strength: float) -> np.ndarray:
        """
        🌸 磨皮魔法：用数字画笔温柔地抚平岁月痕迹

        结合双边滤波和高斯滤波，在保留边缘的同时平滑肌理

        Args:
            image: 输入图像
            strength: 磨皮强度 [0.0, 1.0]

        Returns:
            smoothed: 磨皮后的图像
        """
        if strength <= 0:
            return image.copy()

        # 🎨 根据强度自适应调整滤波参数
        d = int(7 + strength * 10)  # 窗口大小：7-17
        sigma_color = 10.0 + strength * 30.0  # 色彩相似性：10-40
        sigma_space = 5.0 + strength * 5.0   # 空间相似性：5-10

        # 🌟 双边滤波：智能的边缘保留平滑
        # 双边滤波可以在平滑图像的同时保留边缘信息
        bilateral = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

        # 🌙 高斯滤波：进一步的温柔抚慰
        gaussian = cv2.GaussianBlur(bilateral, (5, 5), 2.0)

        # 🎭 根据强度混合双边滤波和高斯滤波的结果
        result = cv2.addWeighted(bilateral, strength, gaussian, 1.0 - strength, 0)

        return result

    def _preserve_details(self, original: np.ndarray, smoothed: np.ndarray,
                         detail_factor: float) -> np.ndarray:
        """
        🔍 细节保留：守护每一份珍贵的自然纹理

        通过高通滤波提取高频细节，再融合回平滑后的图像

        Args:
            original: 原始图像
            smoothed: 平滑后的图像
            detail_factor: 细节保留强度 [0.0, 1.0]

        Returns:
            result: 保留细节后的图像
        """
        # 🌈 提取高频细节信息
        low_freq = cv2.GaussianBlur(original, (0, 0), 3.0)
        high_freq = original.astype(np.float32) - low_freq.astype(np.float32) + 128

        # 📐 计算细节保留强度（与磨皮强度成反比）
        detail_strength = 0.3 * (1.0 - detail_factor)

        # ✨ 将高频细节融合回平滑图像
        result = cv2.addWeighted(
            smoothed.astype(np.float32), 1.0,
            high_freq - 128, detail_strength,
            0
        )

        return np.clip(result, 0, 255).astype(np.uint8)

    def _whiten_skin(self, image: np.ndarray, skin_mask: np.ndarray,
                    strength: float) -> np.ndarray:
        """
        ☀️ 美白算法：如晨光洒向大地的自然提亮

        在LAB颜色空间中调整亮度通道，保持色相的自然真实

        Args:
            image: 输入图像
            skin_mask: 肌肤区域掩码
            strength: 美白强度 [0.0, 1.0]

        Returns:
            result: 美白后的图像
        """
        if strength <= 0:
            return image

        # 🌈 转换到LAB颜色空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)

        # 💫 只对肌肤区域进行美白处理
        mask_float = (skin_mask > 128).astype(np.float32)

        # 🎯 自适应美白：暗部增强更多，亮部保持自然
        adjust_factor = strength * (1.0 - l_channel / 255.0) * mask_float

        # 🌙 使用正弦曲线进行非线性增强，让中等亮度区域增强更明显
        curve_factor = np.sin((l_channel / 255.0) * np.pi) * 1.5
        l_enhanced = l_channel + adjust_factor * curve_factor * 50.0

        # 🎨 更新LAB图像的亮度通道
        lab[:, :, 0] = np.clip(l_enhanced, 0, 255).astype(np.uint8)

        # 🌟 转换回BGR颜色空间
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return result

    def _improve_lighting(self, image: np.ndarray, strength: float) -> np.ndarray:
        """
        🌟 光影平衡：最后的完美调整

        在HSV空间中优化明度分布，营造自然的光影效果

        Args:
            image: 输入图像
            strength: 调整强度 [0.0, 1.0]

        Returns:
            result: 光影优化后的图像
        """
        if strength <= 0:
            return image

        # 🌈 转换到HSV颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2].astype(np.float32)

        # 💫 伽马校正与线性增强的完美结合
        # 伽马校正可以增强中间调，线性增强提升整体亮度
        v_normalized = v_channel / 255.0
        v_gamma_corrected = np.power(v_normalized, 0.8)  # 伽马校正
        v_enhanced = v_gamma_corrected * 255.0 * (1.0 + strength * 0.3)  # 线性增强

        # 🎭 更新HSV图像的明度通道
        hsv[:, :, 2] = np.clip(v_enhanced, 0, 255).astype(np.uint8)

        # 🌟 转换回BGR颜色空间
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return result

    def analyze_skin_quality(self, image: np.ndarray) -> Dict[str, float]:
        """
        📊 肌肤质量分析：评估图像的肌肤状况

        Args:
            image: 输入图像

        Returns:
            quality_metrics: 肌肤质量指标字典
        """
        skin_mask = self._detect_skin(image)

        if np.sum(skin_mask) == 0:
            return {"skin_coverage": 0.0, "skin_smoothness": 0.0, "skin_brightness": 0.0}

        # 计算肌肤覆盖率
        skin_coverage = np.sum(skin_mask > 128) / (image.shape[0] * image.shape[1])

        # 计算肌肤平滑度（基于梯度方差）
        skin_region = cv2.bitwise_and(image, image, mask=skin_mask)
        gray_skin = cv2.cvtColor(skin_region, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray_skin, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_skin, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        skin_smoothness = 1.0 / (1.0 + np.std(gradient_magnitude[skin_mask > 128]))

        # 计算肌肤亮度
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        skin_brightness = np.mean(l_channel[skin_mask > 128]) / 255.0

        return {
            "skin_coverage": skin_coverage,
            "skin_smoothness": skin_smoothness,
            "skin_brightness": skin_brightness
        }

    def beauty_showcase(self, image: np.ndarray, save_path: Optional[str] = None) -> None:
        """
        🎭 美颜展示：展现不同强度下的美妆效果

        Args:
            image: 输入图像
            save_path: 保存路径，如果为None则只显示不保存
        """
        # 🎨 创建不同强度的美颜效果
        configs = {
            "原始自然美": (image, "原始图像"),
            "清新自然风": (SkinBeautyParams(0.3, 0.1, 0.5), "轻度美颜"),
            "优雅知性风": (SkinBeautyParams(0.5, 0.2, 0.3), "中度美颜"),
            "精致魅力风": (SkinBeautyParams(0.7, 0.4, 0.2), "强度美颜")
        }

        results = {}
        for name, config in configs.items():
            if name == "原始自然美":
                results[name] = config[0]
            else:
                beautician = DigitalBeautician(config[0])
                result, _ = beautician.beautify(image)
                results[name] = result

        # 🖼️ 创建美妆画廊
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('💄 数字美妆艺术馆', fontsize=16, fontweight='bold')

        titles = list(configs.keys())
        icons = ['📷', '🌸', '💫', '✨']

        for i, (title, icon) in enumerate(zip(titles, icons)):
            row, col = i // 2, i % 2
            axes[row, col].imshow(cv2.cvtColor(results[title], cv2.COLOR_BGR2RGB))
            axes[row, col].set_title(f'{icon} {title}', fontsize=12)
            axes[row, col].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 美颜效果对比图已保存至: {save_path}")

        plt.show()

    def process_step_by_step(self, image: np.ndarray, save_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        🔍 分步处理展示：展现美颜算法的每个处理步骤

        Args:
            image: 输入图像
            save_dir: 保存目录，如果为None则只返回结果不保存

        Returns:
            step_results: 每个步骤的处理结果
        """
        result, intermediate_results = self.beautify(image)

        # 📊 组织步骤结果
        step_results = {
            "0_原始图像": image,
            "1_肌肤掩码": cv2.applyColorMap(intermediate_results['skin_mask'], cv2.COLORMAP_JET),
            "2_磨皮效果": intermediate_results['smoothed'],
            "3_肌肤混合": intermediate_results['skin_blended'],
            "4_美白效果": intermediate_results.get('whitened', intermediate_results['skin_blended']),
            "5_最终结果": result
        }

        # 🖼️ 创建步骤展示图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🔍 数字美妆处理流程', fontsize=16, fontweight='bold')

        step_names = [
            "📷 原始图像", "🎯 肌肤检测", "🌸 磨皮处理",
            "🎭 智能混合", "☀️ 美白效果", "✨ 最终结果"
        ]

        for i, (step_key, step_name) in enumerate(zip(step_results.keys(), step_names)):
            row, col = i // 3, i % 3

            if step_key == "1_肌肤掩码":
                axes[row, col].imshow(step_results[step_key])
            else:
                axes[row, col].imshow(cv2.cvtColor(step_results[step_key], cv2.COLOR_BGR2RGB))

            axes[row, col].set_title(step_name, fontsize=11)
            axes[row, col].axis('off')

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

            # 保存步骤展示图
            plt.savefig(os.path.join(save_dir, "beauty_process_steps.png"),
                       dpi=300, bbox_inches='tight')

            # 保存各步骤的单独图像
            for step_key, step_image in step_results.items():
                if step_key != "1_肌肤掩码":  # 掩码已经是彩色映射了
                    filename = f"{step_key}.jpg"
                    cv2.imwrite(os.path.join(save_dir, filename), step_image)

            print(f"💾 处理步骤图像已保存至: {save_dir}")

        plt.show()
        return step_results

def create_beauty_demo():
    """🎯 创建美颜效果演示"""

    def process_image_interactive():
        """交互式图像处理"""
        while True:
            print("\n" + "="*50)
            print("💄 数字美妆师 - 交互式演示")
            print("="*50)

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

            # 获取美颜参数
            print("\n🎨 请设置美颜参数:")
            try:
                smoothing = float(input("🌸 磨皮强度 [0.0-1.0, 默认0.5]: ") or "0.5")
                whitening = float(input("☀️ 美白强度 [0.0-1.0, 默认0.2]: ") or "0.2")
                detail = float(input("🔍 细节保留 [0.0-1.0, 默认0.3]: ") or "0.3")

                params = SkinBeautyParams(
                    smoothing_factor=smoothing,
                    whitening_factor=whitening,
                    detail_factor=detail
                )

            except ValueError:
                print("⚠️ 参数格式错误，使用默认参数")
                params = SkinBeautyParams()

            # 创建美妆师并处理
            beautician = DigitalBeautician(params)

            print("\n🔍 开始处理...")
            result, _ = beautician.beautify(image)

            # 显示效果对比
            beautician.beauty_showcase(image)

            # 询问是否保存
            save_choice = input("\n💾 是否保存结果? (y/n): ").strip().lower()
            if save_choice == 'y':
                output_path = input("📁 输入保存路径 (默认: beauty_result.jpg): ").strip() or "beauty_result.jpg"
                cv2.imwrite(output_path, result)
                print(f"✅ 结果已保存至: {output_path}")

    def batch_process_demo():
        """批量处理演示"""
        print("\n" + "="*50)
        print("🚀 批量美颜处理演示")
        print("="*50)

        input_dir = input("📁 输入图像目录路径: ").strip()
        if not os.path.exists(input_dir):
            print("❌ 目录不存在")
            return

        output_dir = input("📁 输出目录路径: ").strip() or "beauty_results"
        os.makedirs(output_dir, exist_ok=True)

        # 获取图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = [f for f in os.listdir(input_dir)
                      if Path(f).suffix.lower() in image_extensions]

        if not image_files:
            print("❌ 未找到图像文件")
            return

        print(f"📸 找到 {len(image_files)} 张图像")

        # 创建美妆师
        beautician = DigitalBeautician(SkinBeautyParams(
            smoothing_factor=0.5,
            whitening_factor=0.3,
            detail_factor=0.2
        ))

        # 批量处理
        for i, filename in enumerate(image_files):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"beauty_{filename}")

            print(f"🎨 处理 ({i+1}/{len(image_files)}): {filename}")

            image = cv2.imread(input_path)
            if image is not None:
                result, _ = beautician.beautify(image)
                cv2.imwrite(output_path, result)
                print(f"✅ 已保存: {output_path}")
            else:
                print(f"❌ 无法读取: {filename}")

        print(f"\n🎉 批量处理完成！结果保存在: {output_dir}")

    # 主菜单
    while True:
        print("\n" + "="*60)
        print("💄 数字美妆师 - 演示系统")
        print("="*60)
        print("1. 📷 交互式单图处理")
        print("2. 🚀 批量图像处理")
        print("3. 🎭 预设效果展示")
        print("4. 📊 肌肤质量分析")
        print("5. 🔍 分步处理展示")
        print("0. 👋 退出程序")
        print("="*60)

        choice = input("请选择功能 (0-5): ").strip()

        if choice == '0':
            print("👋 感谢使用数字美妆师！")
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
                    beautician = DigitalBeautician()
                    beautician.beauty_showcase(image)
                else:
                    print("❌ 无法读取图像")
            else:
                print("❌ 文件不存在")
        elif choice == '4':
            image_path = input("📷 请输入图像路径: ").strip()
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                if image is not None:
                    beautician = DigitalBeautician()
                    quality = beautician.analyze_skin_quality(image)
                    print(f"\n📊 肌肤质量分析结果:")
                    print(f"   🎯 肌肤覆盖率: {quality['skin_coverage']:.2%}")
                    print(f"   🌸 肌肤平滑度: {quality['skin_smoothness']:.3f}")
                    print(f"   ☀️ 肌肤亮度: {quality['skin_brightness']:.2%}")
                else:
                    print("❌ 无法读取图像")
            else:
                print("❌ 文件不存在")
        elif choice == '5':
            image_path = input("📷 请输入图像路径: ").strip()
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                if image is not None:
                    beautician = DigitalBeautician()
                    save_dir = input("📁 保存目录 (可选): ").strip() or None
                    beautician.process_step_by_step(image, save_dir)
                else:
                    print("❌ 无法读取图像")
            else:
                print("❌ 文件不存在")
        else:
            print("❌ 无效选择，请重新输入")

def main():
    """🌟 主函数：展示数字美妆的魅力"""
    parser = argparse.ArgumentParser(description="💄 数字美妆师 - 磨皮美白算法")
    parser.add_argument("--input", "-i", type=str, help="输入图像路径")
    parser.add_argument("--output", "-o", type=str, help="输出图像路径")
    parser.add_argument("--smoothing", "-s", type=float, default=0.5, help="磨皮强度 (0.0-1.0)")
    parser.add_argument("--whitening", "-w", type=float, default=0.2, help="美白强度 (0.0-1.0)")
    parser.add_argument("--detail", "-d", type=float, default=0.3, help="细节保留 (0.0-1.0)")
    parser.add_argument("--demo", action="store_true", help="启动演示模式")
    parser.add_argument("--showcase", action="store_true", help="显示效果对比")
    parser.add_argument("--analyze", action="store_true", help="分析肌肤质量")

    args = parser.parse_args()

    if args.demo:
        create_beauty_demo()
        return

    if not args.input:
        print("🚫 请提供输入图像路径，或使用 --demo 启动演示模式")
        print("💡 使用示例: python skin_beauty.py -i portrait.jpg -o result.jpg")
        print("💡 演示模式: python skin_beauty.py --demo")
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

    # 创建美妆师
    params = SkinBeautyParams(
        smoothing_factor=args.smoothing,
        whitening_factor=args.whitening,
        detail_factor=args.detail
    )

    beautician = DigitalBeautician(params)

    if args.analyze:
        # 肌肤质量分析
        quality = beautician.analyze_skin_quality(image)
        print(f"\n📊 肌肤质量分析结果:")
        print(f"   🎯 肌肤覆盖率: {quality['skin_coverage']:.2%}")
        print(f"   🌸 肌肤平滑度: {quality['skin_smoothness']:.3f}")
        print(f"   ☀️ 肌肤亮度: {quality['skin_brightness']:.2%}")

    if args.showcase:
        # 效果展示
        save_path = args.output.replace('.jpg', '_showcase.png') if args.output else None
        beautician.beauty_showcase(image, save_path)
    else:
        # 单图处理
        result, intermediate_results = beautician.beautify(image)

        if args.output:
            cv2.imwrite(args.output, result)
            print(f"✅ 美颜结果已保存至: {args.output}")
        else:
            # 显示对比
            comparison = np.hstack([image, result])
            cv2.imshow("Beauty Comparison (Original | Result)", comparison)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()