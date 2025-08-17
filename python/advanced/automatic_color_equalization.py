"""
自动色彩均衡算法 - Python实现
Automatic Color Equalization (ACE) Algorithm

基于空间色度调整的局部适应算法，
通过分析像素邻域的空间关系实现自然的色彩校正。

Author: GlimmerLab
Date: 2024
"""

import cv2
import numpy as np
import logging
import time
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ACEParams:
    """自动色彩均衡参数配置"""
    sigma_spatial: float = 50.0         # 空间权重标准差
    adjustment_strength: float = 0.8    # 调整强度
    sampling_step: int = 2              # 采样步长
    use_lab_space: bool = True          # 是否使用LAB颜色空间
    contrast_type: str = "tanh"         # 对比度函数类型 ["linear", "tanh", "sigmoid"]
    max_distance: Optional[float] = None  # 最大影响距离（自动计算）


class AutomaticColorEqualization:
    """自动色彩均衡处理类"""

    def __init__(self, params: Optional[ACEParams] = None):
        """
        初始化处理器

        Args:
            params: 算法参数配置
        """
        self.params = params or ACEParams()
        logger.info(f"初始化自动色彩均衡器，空间标准差: {self.params.sigma_spatial}, "
                   f"调整强度: {self.params.adjustment_strength}")

    def enhance_image(self, image: np.ndarray,
                     params: Optional[ACEParams] = None) -> np.ndarray:
        """
        执行自动色彩均衡

        Args:
            image: 输入图像
            params: 可选的参数覆盖

        Returns:
            均衡后的图像

        Raises:
            ValueError: 输入图像无效时抛出
        """
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("仅支持3通道彩色图像")

        p = params or self.params

        if p.use_lab_space:
            return self._process_lab_space(image, p)
        else:
            return self._process_rgb_space(image, p)

    def _process_lab_space(self, image: np.ndarray, params: ACEParams) -> np.ndarray:
        """
        在LAB颜色空间中处理图像

        Args:
            image: 输入图像
            params: 处理参数

        Returns:
            处理后的图像
        """
        # 转换到LAB颜色空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab).astype(np.float32)

        # 分离通道
        L, a, b = cv2.split(lab)

        # 对L通道进行ACE处理
        enhanced_L = self._ace_channel_processing(L, params)

        # 对a、b通道进行轻度处理
        enhanced_a = self._ace_channel_processing(a, params, strength_factor=0.3)
        enhanced_b = self._ace_channel_processing(b, params, strength_factor=0.3)

        # 合并通道
        enhanced_lab = cv2.merge([enhanced_L, enhanced_a, enhanced_b])

        # 转换回BGR
        result = cv2.cvtColor(enhanced_lab.astype(np.uint8), cv2.COLOR_Lab2BGR)
        return result

    def _process_rgb_space(self, image: np.ndarray, params: ACEParams) -> np.ndarray:
        """
        在RGB颜色空间中处理图像

        Args:
            image: 输入图像
            params: 处理参数

        Returns:
            处理后的图像
        """
        float_img = image.astype(np.float32)
        result = np.zeros_like(float_img)

        # 分别处理每个通道
        for c in range(3):
            result[:, :, c] = self._ace_channel_processing(float_img[:, :, c], params)

        return np.clip(result, 0, 255).astype(np.uint8)

    def _ace_channel_processing(self, channel: np.ndarray, params: ACEParams,
                               strength_factor: float = 1.0) -> np.ndarray:
        """
        对单个通道执行ACE处理

        Args:
            channel: 输入通道
            params: 处理参数
            strength_factor: 强度因子

        Returns:
            处理后的通道
        """
        h, w = channel.shape
        result = np.zeros_like(channel, dtype=np.float32)

        # 计算最大影响距离
        max_dist = params.max_distance or min(h, w) * 0.3

        # 有效调整强度
        effective_strength = params.adjustment_strength * strength_factor

        for y in range(0, h, 1):  # 可以调整步长以平衡速度和质量
            for x in range(0, w, 1):
                adjustment = self._compute_pixel_adjustment(
                    channel, x, y, params, max_dist, effective_strength)
                result[y, x] = np.clip(channel[y, x] + adjustment, 0, 255)

            # 进度提示
            if y % (h // 10) == 0:
                logger.debug(f"处理进度: {y}/{h}")

        return result

    def _compute_pixel_adjustment(self, channel: np.ndarray, x: int, y: int,
                                 params: ACEParams, max_dist: float,
                                 strength: float) -> float:
        """
        计算单个像素的调整值

        Args:
            channel: 输入通道
            x, y: 像素坐标
            params: 处理参数
            max_dist: 最大影响距离
            strength: 调整强度

        Returns:
            调整值
        """
        h, w = channel.shape
        target_value = channel[y, x]

        adjustment = 0.0
        total_weight = 0.0

        # 计算影响区域
        y_min = max(0, int(y - max_dist))
        y_max = min(h, int(y + max_dist) + 1)
        x_min = max(0, int(x - max_dist))
        x_max = min(w, int(x + max_dist) + 1)

        # 采样邻域像素
        for ny in range(y_min, y_max, params.sampling_step):
            for nx in range(x_min, x_max, params.sampling_step):
                if nx == x and ny == y:
                    continue

                # 计算空间距离
                spatial_dist = np.sqrt((x - nx)**2 + (y - ny)**2)
                if spatial_dist > max_dist:
                    continue

                # 计算空间权重
                spatial_weight = np.exp(-spatial_dist**2 / (2.0 * params.sigma_spatial**2))

                # 计算色彩差异
                neighbor_value = channel[ny, nx]
                color_diff = neighbor_value - target_value

                # 应用对比度函数
                enhanced_diff = self._apply_contrast_function(color_diff, params.contrast_type)

                # 累积调整
                adjustment += spatial_weight * enhanced_diff
                total_weight += spatial_weight

        # 归一化调整
        if total_weight > 0:
            adjustment = (adjustment / total_weight) * strength

        return adjustment

    def _apply_contrast_function(self, diff: float, contrast_type: str) -> float:
        """
        应用对比度函数

        Args:
            diff: 色彩差异
            contrast_type: 对比度函数类型

        Returns:
            增强后的差异
        """
        if contrast_type == "linear":
            return diff
        elif contrast_type == "tanh":
            return np.tanh(diff / 10.0) * 10.0
        elif contrast_type == "sigmoid":
            return 2.0 / (1.0 + np.exp(-diff / 10.0)) - 1.0
        else:
            return diff

    def multi_scale_enhance(self, image: np.ndarray,
                          scales: List[float] = [1.0, 0.5, 0.25],
                          strengths: List[float] = [0.8, 0.6, 0.4]) -> np.ndarray:
        """
        多尺度自动色彩均衡

        Args:
            image: 输入图像
            scales: 多个处理尺度
            strengths: 对应的处理强度

        Returns:
            多尺度处理后的图像
        """
        if len(scales) != len(strengths):
            raise ValueError("尺度和强度数量必须相等")

        h, w = image.shape[:2]
        result = image.astype(np.float32)

        for scale, strength in zip(scales, strengths):
            # 计算缩放尺寸
            new_h, new_w = int(h * scale), int(w * scale)

            # 缩放图像
            if scale != 1.0:
                scaled_img = cv2.resize(image, (new_w, new_h))
            else:
                scaled_img = image

            # 调整参数
            scale_params = ACEParams(
                sigma_spatial=self.params.sigma_spatial * scale,
                adjustment_strength=strength,
                sampling_step=max(1, int(self.params.sampling_step / scale)),
                use_lab_space=self.params.use_lab_space,
                contrast_type=self.params.contrast_type
            )

            # 处理当前尺度
            enhanced_scale = self.enhance_image(scaled_img, scale_params)

            # 恢复到原始尺寸
            if scale != 1.0:
                enhanced_scale = cv2.resize(enhanced_scale, (w, h))

            # 融合结果
            weight = 1.0 / len(scales)
            if scale == scales[0]:  # 第一个尺度
                result = enhanced_scale.astype(np.float32)
            else:
                result = cv2.addWeighted(result, 1.0 - weight,
                                       enhanced_scale.astype(np.float32), weight, 0)

        return np.clip(result, 0, 255).astype(np.uint8)

    def fast_ace(self, image: np.ndarray, params: Optional[ACEParams] = None) -> np.ndarray:
        """
        快速ACE处理（基于积分图像的优化版本）

        Args:
            image: 输入图像
            params: 处理参数

        Returns:
            快速处理后的图像
        """
        p = params or self.params

        # 转换到YUV空间进行快速处理
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV).astype(np.float32)
        y_channel = yuv[:, :, 0]

        # 计算局部均值（使用盒式滤波器）
        kernel_size = int(p.sigma_spatial // 2) * 2 + 1  # 确保奇数
        local_mean = cv2.boxFilter(y_channel, -1, (kernel_size, kernel_size))

        # 计算局部标准差
        y_squared = y_channel * y_channel
        local_mean_squared = cv2.boxFilter(y_squared, -1, (kernel_size, kernel_size))
        local_variance = local_mean_squared - local_mean * local_mean
        local_std = np.sqrt(np.maximum(local_variance, 0))

        # 自适应调整
        adjustment_factor = p.adjustment_strength * (local_std / (local_mean + 1e-6))
        enhanced_y = y_channel + adjustment_factor * (y_channel - local_mean)

        # 限制范围
        enhanced_y = np.clip(enhanced_y, 0, 255)

        # 更新Y通道
        yuv[:, :, 0] = enhanced_y

        # 转换回BGR
        result = cv2.cvtColor(yuv.astype(np.uint8), cv2.COLOR_YUV2BGR)
        return result

    def analyze_color_distribution(self, image: np.ndarray) -> Dict[str, Any]:
        """
        分析图像色彩分布特征

        Args:
            image: 输入图像

        Returns:
            色彩分析结果
        """
        # 转换到LAB空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        L, a, b = cv2.split(lab)

        # 计算各通道统计
        l_stats = {
            'mean': np.mean(L),
            'std': np.std(L),
            'min': np.min(L),
            'max': np.max(L)
        }

        a_stats = {
            'mean': np.mean(a),
            'std': np.std(a),
            'min': np.min(a),
            'max': np.max(a)
        }

        b_stats = {
            'mean': np.mean(b),
            'std': np.std(b),
            'min': np.min(b),
            'max': np.max(b)
        }

        # 计算色彩分布均匀度
        hist_L = cv2.calcHist([L], [0], None, [256], [0, 256])
        hist_a = cv2.calcHist([a], [0], None, [256], [0, 256])
        hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])

        # 计算熵值（作为分布均匀度的指标）
        def calculate_entropy(hist):
            hist = hist.flatten()
            hist = hist / (np.sum(hist) + 1e-6)
            entropy = -np.sum(hist * np.log2(hist + 1e-6))
            return entropy

        return {
            'L_channel': l_stats,
            'a_channel': a_stats,
            'b_channel': b_stats,
            'L_entropy': calculate_entropy(hist_L),
            'a_entropy': calculate_entropy(hist_a),
            'b_entropy': calculate_entropy(hist_b),
            'dynamic_range': l_stats['max'] - l_stats['min'],
            'color_cast': np.sqrt(a_stats['mean']**2 + b_stats['mean']**2)
        }

    def adaptive_parameters(self, image: np.ndarray) -> ACEParams:
        """
        根据图像特征自动调整参数

        Args:
            image: 输入图像

        Returns:
            自适应调整的参数
        """
        analysis = self.analyze_color_distribution(image)

        # 根据图像特征调整参数
        h, w = image.shape[:2]
        base_sigma = min(h, w) * 0.05  # 基础空间标准差

        # 根据动态范围调整
        dynamic_range = analysis['dynamic_range']
        if dynamic_range < 100:
            # 低动态范围图像
            strength = 1.2
            sigma = base_sigma * 1.5
        elif dynamic_range > 200:
            # 高动态范围图像
            strength = 0.6
            sigma = base_sigma * 0.8
        else:
            # 中等动态范围图像
            strength = 0.8
            sigma = base_sigma

        # 根据色偏调整
        color_cast = analysis['color_cast']
        if color_cast > 20:  # 存在明显色偏
            use_lab = True
            sampling_step = 2
        else:
            use_lab = False
            sampling_step = 3

        return ACEParams(
            sigma_spatial=sigma,
            adjustment_strength=strength,
            sampling_step=sampling_step,
            use_lab_space=use_lab,
            contrast_type="tanh"
        )

    def batch_enhance(self, images: List[np.ndarray],
                     auto_params: bool = True) -> List[np.ndarray]:
        """
        批量处理图像

        Args:
            images: 图像列表
            auto_params: 是否自动调整参数

        Returns:
            处理后的图像列表
        """
        results = []

        for i, image in enumerate(images):
            if auto_params:
                params = self.adaptive_parameters(image)
                result = self.enhance_image(image, params)
            else:
                result = self.enhance_image(image)

            results.append(result)
            logger.info(f"处理第 {i+1}/{len(images)} 张图像")

        return results


def demo_basic_enhancement():
    """演示基础色彩均衡功能"""
    print("=== 自动色彩均衡基础演示 ===")

    # 创建处理器
    ace = AutomaticColorEqualization()

    # 创建测试图像
    test_image = create_test_image()

    print(f"原始图像形状: {test_image.shape}")

    # 分析图像特征
    analysis = ace.analyze_color_distribution(test_image)
    print("图像色彩分析:")
    print(f"  亮度范围: {analysis['L_channel']['min']:.1f} - {analysis['L_channel']['max']:.1f}")
    print(f"  动态范围: {analysis['dynamic_range']:.1f}")
    print(f"  色偏程度: {analysis['color_cast']:.1f}")

    # 基础增强
    print("\n执行基础ACE增强...")
    enhanced = ace.enhance_image(test_image)

    # 快速增强
    print("执行快速ACE增强...")
    fast_enhanced = ace.fast_ace(test_image)

    # 多尺度增强
    print("执行多尺度ACE增强...")
    multi_enhanced = ace.multi_scale_enhance(test_image)

    print("演示完成！")

    return {
        'original': test_image,
        'enhanced': enhanced,
        'fast_enhanced': fast_enhanced,
        'multi_enhanced': multi_enhanced
    }


def demo_parameter_adaptation():
    """演示参数自适应功能"""
    print("=== 参数自适应演示 ===")

    ace = AutomaticColorEqualization()

    # 创建不同类型的测试图像
    test_images = {
        'low_contrast': create_low_contrast_image(),
        'high_contrast': create_high_contrast_image(),
        'color_cast': create_color_cast_image()
    }

    for name, image in test_images.items():
        print(f"\n处理 {name} 图像:")

        # 分析并获取自适应参数
        adaptive_params = ace.adaptive_parameters(image)
        print(f"  自适应参数 - 空间标准差: {adaptive_params.sigma_spatial:.1f}")
        print(f"  调整强度: {adaptive_params.adjustment_strength:.2f}")
        print(f"  使用LAB空间: {adaptive_params.use_lab_space}")

        # 处理图像
        enhanced = ace.enhance_image(image, adaptive_params)

        print(f"  处理完成")

    print("\n参数自适应演示完成！")


def create_test_image() -> np.ndarray:
    """创建测试图像"""
    height, width = 480, 640
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # 创建不同区域的色彩分布
    # 左上：低对比度蓝色区域
    image[:height//2, :width//2, 0] = 80  # B
    image[:height//2, :width//2, 1] = 90  # G
    image[:height//2, :width//2, 2] = 100 # R

    # 右上：高对比度红色区域
    mask = np.random.choice([0, 255], (height//2, width//2))
    image[:height//2, width//2:, 2] = mask
    image[:height//2, width//2:, 0] = 255 - mask

    # 左下：渐变区域
    for i in range(height//2):
        for j in range(width//2):
            value = int(255 * j / (width//2))
            image[height//2 + i, j] = [value, value, 255 - value]

    # 右下：色偏区域（偏绿）
    image[height//2:, width//2:, 1] = 180  # 强绿色
    image[height//2:, width//2:, 0] = 50   # 弱蓝色
    image[height//2:, width//2:, 2] = 50   # 弱红色

    return image


def create_low_contrast_image() -> np.ndarray:
    """创建低对比度测试图像"""
    height, width = 300, 400
    base_value = 128
    noise_range = 20

    image = np.random.normal(base_value, noise_range, (height, width, 3))
    return np.clip(image, 0, 255).astype(np.uint8)


def create_high_contrast_image() -> np.ndarray:
    """创建高对比度测试图像"""
    height, width = 300, 400
    image = np.random.choice([0, 255], (height, width, 3))
    return image.astype(np.uint8)


def create_color_cast_image() -> np.ndarray:
    """创建色偏测试图像"""
    height, width = 300, 400
    image = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)

    # 添加蓝色偏色
    image[:, :, 0] = np.clip(image[:, :, 0] + 50, 0, 255)  # 增强蓝色
    image[:, :, 1] = np.clip(image[:, :, 1] - 20, 0, 255)  # 减少绿色
    image[:, :, 2] = np.clip(image[:, :, 2] - 20, 0, 255)  # 减少红色

    return image


def performance_benchmark():
    """性能基准测试"""
    print("=== 性能基准测试 ===")

    # 测试不同参数配置
    configs = [
        ACEParams(sigma_spatial=30.0, adjustment_strength=0.8, sampling_step=3),
        ACEParams(sigma_spatial=50.0, adjustment_strength=0.8, sampling_step=2),
        ACEParams(sigma_spatial=70.0, adjustment_strength=0.8, sampling_step=1),
    ]

    # 创建测试图像
    test_sizes = [(240, 320), (480, 640), (720, 960)]

    for size in test_sizes:
        print(f"\n测试图像大小: {size[0]}x{size[1]}")
        test_image = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)

        for i, config in enumerate(configs):
            ace = AutomaticColorEqualization(config)

            # 预热
            ace.fast_ace(test_image)

            # 测试快速版本
            start_time = time.time()
            num_iterations = 5

            for _ in range(num_iterations):
                result = ace.fast_ace(test_image)

            elapsed_time = time.time() - start_time
            avg_time = elapsed_time / num_iterations

            print(f"  配置{i+1} (σ={config.sigma_spatial}, step={config.sampling_step}): "
                 f"{avg_time*1000:.2f}ms")


if __name__ == "__main__":
    # 运行演示
    print("启动自动色彩均衡演示程序\n")

    # 基础功能演示
    demo_results = demo_basic_enhancement()
    print()

    # 参数自适应演示
    demo_parameter_adaptation()
    print()

    # 性能基准测试
    performance_benchmark()

    print("\n所有演示完成！")