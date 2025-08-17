"""
老照片特效算法 - Python实现
Vintage Effect Algorithm

基于多层次图像变换的老照片风格化处理技术，
通过模拟胶片摄影的物理特性实现怀旧视觉效果。

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
from enum import Enum

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VintageStyle(Enum):
    """老照片风格枚举"""
    CLASSIC_SEPIA = "classic_sepia"      # 经典褐色调
    WARM_TONE = "warm_tone"              # 暖色调
    COOL_VINTAGE = "cool_vintage"        # 冷色调怀旧
    HIGH_CONTRAST = "high_contrast"      # 高对比度
    FADED = "faded"                      # 褪色效果


@dataclass
class VintageParams:
    """老照片特效参数配置"""
    sepia_intensity: float = 0.8         # 褐色调强度 (0.0-1.0)
    grain_level: float = 15.0            # 颗粒噪声级别 (0-50)
    grain_size: int = 1                  # 颗粒大小 (1-5)
    vignette_strength: float = 0.3       # 暗角强度 (0.0-1.0)
    vignette_radius: float = 0.8         # 暗角半径因子 (0.1-1.0)
    scratch_count: int = 5               # 划痕数量 (0-20)
    scratch_intensity: float = 0.6       # 划痕强度 (0.0-1.0)
    dust_density: float = 0.001          # 灰尘密度 (0.0-0.01)
    contrast: float = 1.1                # 对比度调整 (0.5-2.0)
    brightness: float = -10              # 亮度调整 (-50-50)
    fade_intensity: float = 0.2          # 褪色强度 (0.0-1.0)


class VintageEffectProcessor:
    """老照片特效处理类"""

    def __init__(self):
        """初始化老照片特效处理器"""
        self.sepia_matrix = np.array([
            [0.272, 0.534, 0.131],  # Blue channel transformation
            [0.349, 0.686, 0.168],  # Green channel transformation
            [0.393, 0.769, 0.189]   # Red channel transformation
        ], dtype=np.float32)

        logger.info("老照片特效处理器初始化完成")

    def apply_vintage_effect(self, image: np.ndarray,
                           style: VintageStyle = VintageStyle.CLASSIC_SEPIA,
                           params: Optional[VintageParams] = None) -> np.ndarray:
        """
        应用老照片特效

        Args:
            image: 输入图像 (H, W, C)
            style: 特效风格
            params: 可选的参数覆盖

        Returns:
            处理后的图像

        Raises:
            ValueError: 输入参数无效时抛出
        """
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")

        if len(image.shape) not in [2, 3]:
            raise ValueError("输入图像必须是2D或3D数组")

        # 使用预设参数或自定义参数
        p = params or self._get_style_params(style)

        # 确保输入是3通道彩色图像
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # 复制输入图像避免修改原图
        result = image.copy().astype(np.float32)

        # 应用各种特效
        if p.sepia_intensity > 0:
            result = self._apply_sepia_tone(result, p.sepia_intensity)

        if p.grain_level > 0:
            result = self._add_film_grain(result, p.grain_level, p.grain_size)

        if p.vignette_strength > 0:
            result = self._add_vignette(result, p.vignette_strength, p.vignette_radius)

        if p.scratch_count > 0:
            result = self._add_scratches(result, p.scratch_count, p.scratch_intensity)

        if p.dust_density > 0:
            result = self._add_dust_spots(result, p.dust_density)

        if p.fade_intensity > 0:
            result = self._add_fade_effect(result, p.fade_intensity)

        # 调整对比度和亮度
        result = self._adjust_contrast_brightness(result, p.contrast, p.brightness)

        return np.clip(result, 0, 255).astype(np.uint8)

    def _get_style_params(self, style: VintageStyle) -> VintageParams:
        """
        获取预设风格参数

        Args:
            style: 风格类型

        Returns:
            对应的参数配置
        """
        style_configs = {
            VintageStyle.CLASSIC_SEPIA: VintageParams(
                sepia_intensity=0.8, grain_level=15.0, vignette_strength=0.3,
                scratch_count=5, contrast=1.1, brightness=-10
            ),
            VintageStyle.WARM_TONE: VintageParams(
                sepia_intensity=0.6, grain_level=10.0, vignette_strength=0.2,
                scratch_count=3, contrast=1.05, brightness=5, fade_intensity=0.1
            ),
            VintageStyle.COOL_VINTAGE: VintageParams(
                sepia_intensity=0.4, grain_level=20.0, vignette_strength=0.4,
                scratch_count=7, contrast=1.15, brightness=-15, fade_intensity=0.3
            ),
            VintageStyle.HIGH_CONTRAST: VintageParams(
                sepia_intensity=0.9, grain_level=25.0, vignette_strength=0.5,
                scratch_count=8, contrast=1.3, brightness=-5
            ),
            VintageStyle.FADED: VintageParams(
                sepia_intensity=0.5, grain_level=8.0, vignette_strength=0.15,
                scratch_count=2, contrast=0.9, brightness=10, fade_intensity=0.4
            )
        }

        return style_configs.get(style, VintageParams())

    def _apply_sepia_tone(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """
        应用褐色调变换

        Args:
            image: 输入图像 (float32, 0-255)
            intensity: 褐色调强度 (0.0-1.0)

        Returns:
            褐色调处理后的图像
        """
        # 确保图像在0-1范围内
        normalized_img = image / 255.0

        # 重塑为矩阵运算格式
        h, w, c = normalized_img.shape
        img_reshaped = normalized_img.reshape(-1, 3)

        # 应用褐色调变换矩阵
        sepia_result = np.dot(img_reshaped, self.sepia_matrix.T)

        # 限制值范围
        sepia_result = np.clip(sepia_result, 0, 1)

        # 根据强度混合原图和褐色调结果
        mixed_result = intensity * sepia_result + (1 - intensity) * img_reshaped

        # 恢复原始形状并转换回0-255范围
        result = mixed_result.reshape(h, w, c) * 255.0

        return result.astype(np.float32)

    def _add_film_grain(self, image: np.ndarray, noise_level: float,
                       grain_size: int = 1) -> np.ndarray:
        """
        添加胶片颗粒效果

        Args:
            image: 输入图像
            noise_level: 噪声强度
            grain_size: 颗粒大小

        Returns:
            添加颗粒后的图像
        """
        # 生成符合正态分布的噪声
        noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)

        # 如果需要更大的颗粒，应用轻微的高斯模糊
        if grain_size > 1:
            kernel_size = 2 * grain_size + 1
            noise = cv2.GaussianBlur(noise, (kernel_size, kernel_size), 0.5)

        # 添加噪声到图像
        grainy_image = image + noise

        # 应用中值滤波模拟胶片特性
        result = cv2.medianBlur(grainy_image.astype(np.uint8), 3).astype(np.float32)

        return result

    def _add_vignette(self, image: np.ndarray, strength: float,
                     radius_factor: float = 0.8) -> np.ndarray:
        """
        添加暗角效果

        Args:
            image: 输入图像
            strength: 暗角强度
            radius_factor: 半径因子

        Returns:
            添加暗角后的图像
        """
        h, w = image.shape[:2]

        # 计算图像中心和最大半径
        center_x, center_y = w // 2, h // 2
        max_radius = min(center_x, center_y) * radius_factor

        # 创建坐标网格
        Y, X = np.ogrid[:h, :w]

        # 计算每个像素到中心的距离
        distances = np.sqrt((X - center_x)**2 + (Y - center_y)**2)

        # 创建暗角遮罩
        vignette_mask = np.ones((h, w), dtype=np.float32)

        # 只在超出半径的区域应用暗角
        beyond_radius = distances > max_radius
        if np.any(beyond_radius):
            # 计算超出区域的衰减因子
            excess_distance = distances - max_radius
            max_excess = np.max(excess_distance)

            if max_excess > 0:
                normalized_excess = excess_distance / max_excess
                attenuation = 1.0 - strength * np.power(normalized_excess, 2)
                vignette_mask = np.where(beyond_radius,
                                       np.maximum(attenuation, 0.1),
                                       vignette_mask)

        # 应用暗角到所有通道
        if len(image.shape) == 3:
            vignette_mask = np.stack([vignette_mask] * 3, axis=2)

        result = image * vignette_mask

        return result

    def _add_scratches(self, image: np.ndarray, count: int,
                      intensity: float) -> np.ndarray:
        """
        添加划痕效果

        Args:
            image: 输入图像
            count: 划痕数量
            intensity: 划痕强度

        Returns:
            添加划痕后的图像
        """
        result = image.copy()
        h, w = image.shape[:2]

        for _ in range(count):
            # 随机生成划痕参数
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(0, h)

            # 随机长度和角度
            length = np.random.randint(h // 10, min(h, w) // 4)
            angle = np.random.uniform(0, 2 * np.pi)
            thickness = np.random.randint(1, 3)

            # 计算终点
            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))

            # 确保终点在图像范围内
            x2 = np.clip(x2, 0, w - 1)
            y2 = np.clip(y2, 0, h - 1)

            # 随机选择划痕类型（亮或暗）
            if np.random.random() > 0.5:
                color = (255, 255, 255)  # 亮划痕
            else:
                color = (0, 0, 0)        # 暗划痕

            # 绘制划痕
            cv2.line(result.astype(np.uint8), (x1, y1), (x2, y2),
                    color, thickness, cv2.LINE_AA)

        # 与原图混合
        result = result.astype(np.float32)
        blended = (1.0 - intensity) * image + intensity * result

        return blended

    def _add_dust_spots(self, image: np.ndarray, density: float) -> np.ndarray:
        """
        添加灰尘斑点

        Args:
            image: 输入图像
            density: 灰尘密度

        Returns:
            添加灰尘后的图像
        """
        result = image.copy()
        h, w = image.shape[:2]

        # 计算灰尘斑点数量
        num_spots = int(h * w * density)

        for _ in range(num_spots):
            # 随机位置
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)

            # 随机大小和颜色
            radius = np.random.randint(1, 3)
            brightness = np.random.uniform(0.3, 1.0)

            # 绘制灰尘斑点
            cv2.circle(result.astype(np.uint8), (x, y), radius,
                      (int(255 * brightness),) * 3, -1)

        return result.astype(np.float32)

    def _add_fade_effect(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """
        添加褪色效果

        Args:
            image: 输入图像
            intensity: 褪色强度

        Returns:
            褪色处理后的图像
        """
        # 创建褪色遮罩（向白色褪色）
        fade_color = np.array([240, 235, 220], dtype=np.float32)  # 略带黄色的白色

        # 根据强度混合
        result = (1.0 - intensity) * image + intensity * fade_color

        return result

    def _adjust_contrast_brightness(self, image: np.ndarray,
                                  contrast: float, brightness: float) -> np.ndarray:
        """
        调整对比度和亮度

        Args:
            image: 输入图像
            contrast: 对比度因子
            brightness: 亮度偏移

        Returns:
            调整后的图像
        """
        # 应用对比度和亮度调整
        result = contrast * image + brightness

        return result

    def create_vintage_border(self, image: np.ndarray,
                            border_width: Optional[int] = None) -> np.ndarray:
        """
        创建复古边框

        Args:
            image: 输入图像
            border_width: 边框宽度（None时自动计算）

        Returns:
            带边框的图像
        """
        h, w = image.shape[:2]

        if border_width is None:
            border_width = min(h, w) // 25

        # 创建白色边框
        bordered = cv2.copyMakeBorder(
            image, border_width, border_width, border_width, border_width,
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )

        # 添加内边框装饰
        inner_width = max(1, border_width // 3)

        # 绘制内边框
        cv2.rectangle(
            bordered,
            (border_width - inner_width, border_width - inner_width),
            (bordered.shape[1] - border_width + inner_width - 1,
             bordered.shape[0] - border_width + inner_width - 1),
            (128, 128, 128), inner_width
        )

        return bordered

    def batch_process(self, images: List[np.ndarray],
                     style: VintageStyle = VintageStyle.CLASSIC_SEPIA,
                     params: Optional[VintageParams] = None) -> List[np.ndarray]:
        """
        批量处理图像

        Args:
            images: 输入图像列表
            style: 特效风格
            params: 特效参数

        Returns:
            处理后的图像列表
        """
        results = []

        for i, image in enumerate(images):
            try:
                result = self.apply_vintage_effect(image, style, params)
                results.append(result)
                logger.info(f"处理图像 {i+1}/{len(images)} 完成")
            except Exception as e:
                logger.error(f"处理图像 {i+1} 时出错: {e}")
                results.append(image)  # 出错时返回原图

        return results

    def get_quality_metrics(self, original: np.ndarray,
                          processed: np.ndarray) -> Dict[str, float]:
        """
        计算图像质量指标

        Args:
            original: 原始图像
            processed: 处理后图像

        Returns:
            质量指标字典
        """
        # 确保图像格式一致
        if len(original.shape) != len(processed.shape):
            if len(original.shape) == 2:
                original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
            if len(processed.shape) == 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        # 转换为浮点数
        orig_float = original.astype(np.float32)
        proc_float = processed.astype(np.float32)

        # 计算PSNR
        mse = np.mean((orig_float - proc_float) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))

        # 计算结构相似性指标（简化版）
        mu1 = np.mean(orig_float)
        mu2 = np.mean(proc_float)
        sigma1 = np.std(orig_float)
        sigma2 = np.std(proc_float)
        sigma12 = np.mean((orig_float - mu1) * (proc_float - mu2))

        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2

        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))

        return {
            'psnr': psnr,
            'ssim': ssim,
            'mse': mse
        }


def demo_vintage_effects():
    """演示老照片特效功能"""
    print("=== 老照片特效演示 ===")

    processor = VintageEffectProcessor()

    # 创建测试图像
    test_image = create_test_image()
    print(f"测试图像尺寸: {test_image.shape}")

    # 测试不同风格
    styles = [
        VintageStyle.CLASSIC_SEPIA,
        VintageStyle.WARM_TONE,
        VintageStyle.COOL_VINTAGE,
        VintageStyle.HIGH_CONTRAST,
        VintageStyle.FADED
    ]

        results = {}

    for style in styles:
        print(f"应用 {style.value} 风格...")
                start_time = time.time()

        result = processor.apply_vintage_effect(test_image, style)

        elapsed_time = time.time() - start_time
        print(f"  处理时间: {elapsed_time*1000:.2f}ms")

        # 计算质量指标
        metrics = processor.get_quality_metrics(test_image, result)
        print(f"  PSNR: {metrics['psnr']:.2f}dB")
        print(f"  SSIM: {metrics['ssim']:.3f}")

        results[style.value] = result

    # 测试边框功能
    print("测试复古边框...")
    bordered = processor.create_vintage_border(test_image)
    results['bordered'] = bordered

    print("演示完成！")
        return results


def demo_custom_parameters():
    """演示自定义参数功能"""
    print("=== 自定义参数演示 ===")

    processor = VintageEffectProcessor()
    test_image = create_test_image()

    # 自定义参数配置
    custom_params = VintageParams(
        sepia_intensity=0.9,
        grain_level=30.0,
        grain_size=2,
        vignette_strength=0.6,
        scratch_count=10,
        dust_density=0.005,
        contrast=1.2,
        brightness=-20,
        fade_intensity=0.1
    )

    print("应用自定义参数...")
    result = processor.apply_vintage_effect(
        test_image,
        VintageStyle.CLASSIC_SEPIA,
        custom_params
    )

    # 计算质量指标
    metrics = processor.get_quality_metrics(test_image, result)
    print(f"PSNR: {metrics['psnr']:.2f}dB")
    print(f"SSIM: {metrics['ssim']:.3f}")

    return result


def create_test_image() -> np.ndarray:
    """创建测试图像"""
    height, width = 400, 600
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # 创建渐变背景
    for y in range(height):
        for x in range(width):
            # 创建径向渐变
            center_x, center_y = width // 2, height // 2
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)

            intensity = 1.0 - (dist / max_dist)
            image[y, x] = [
                int(intensity * 200 + 55),    # Blue
                int(intensity * 180 + 75),    # Green
                int(intensity * 160 + 95)     # Red
            ]

    # 添加几何图形
    cv2.rectangle(image, (150, 100), (250, 200), (255, 100, 100), -1)
    cv2.circle(image, (400, 150), 60, (100, 255, 100), -1)
    cv2.ellipse(image, (450, 300), (80, 50), 30, 0, 360, (100, 100, 255), -1)

    # 添加文字
    cv2.putText(image, "VINTAGE TEST", (200, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return image


def performance_benchmark():
    """性能基准测试"""
    print("=== 性能基准测试 ===")

    processor = VintageEffectProcessor()

    # 测试不同图像尺寸
    test_sizes = [(240, 320), (480, 640), (720, 960), (1080, 1440)]
    styles = [VintageStyle.CLASSIC_SEPIA, VintageStyle.HIGH_CONTRAST]

    for size in test_sizes:
        print(f"\n测试图像尺寸: {size[0]}x{size[1]}")

        # 创建测试图像
        test_image = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)

        for style in styles:
            # 预热
            _ = processor.apply_vintage_effect(test_image, style)

            # 计时测试
            start_time = time.time()
            num_iterations = 5

            for _ in range(num_iterations):
                result = processor.apply_vintage_effect(test_image, style)

            elapsed_time = time.time() - start_time
            avg_time = elapsed_time / num_iterations

            print(f"  {style.value}: {avg_time*1000:.2f}ms")


if __name__ == "__main__":
    # 运行演示
    print("启动老照片特效演示程序\n")

    # 基础功能演示
    vintage_results = demo_vintage_effects()
    print()

    # 自定义参数演示
    custom_result = demo_custom_parameters()
    print()

    # 性能基准测试
    performance_benchmark()

    print("\n所有演示完成！")