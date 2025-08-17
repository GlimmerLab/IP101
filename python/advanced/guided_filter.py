"""
导向滤波算法 - Python实现
Guided Filter Algorithm

基于局部线性模型的边缘保持滤波技术，
通过引导图像的结构信息实现高质量的图像滤波。

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

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class GuidedFilterParams:
    """导向滤波参数配置"""
    radius: int = 8               # 滤波半径
    eps: float = 0.01             # 正则化参数
    subsample: int = 1            # 下采样比例（快速模式）
    use_fast_mode: bool = False   # 是否使用快速模式


class GuidedFilter:
    """导向滤波处理类"""

    def __init__(self, params: Optional[GuidedFilterParams] = None):
        """
        初始化导向滤波器

        Args:
            params: 滤波参数配置
        """
        self.params = params or GuidedFilterParams()
        logger.info(f"初始化导向滤波器，半径: {self.params.radius}, "
                   f"正则化参数: {self.params.eps}")

    def guided_filter(self, input_img: np.ndarray, guide_img: np.ndarray,
                     params: Optional[GuidedFilterParams] = None) -> np.ndarray:
        """
        执行导向滤波

        Args:
            input_img: 输入图像（待滤波）
            guide_img: 引导图像
            params: 可选的参数覆盖

        Returns:
            滤波后的图像

        Raises:
            ValueError: 输入参数无效时抛出
        """
        if input_img is None or guide_img is None:
            raise ValueError("输入图像或引导图像为空")

        if input_img.shape[:2] != guide_img.shape[:2]:
            raise ValueError("输入图像和引导图像尺寸不匹配")

        p = params or self.params

        if p.use_fast_mode and p.subsample > 1:
            return self._fast_guided_filter(input_img, guide_img, p)
        else:
            return self._basic_guided_filter(input_img, guide_img, p)

    def _basic_guided_filter(self, input_img: np.ndarray, guide_img: np.ndarray,
                           params: GuidedFilterParams) -> np.ndarray:
        """
        基础导向滤波实现

        Args:
            input_img: 输入图像
            guide_img: 引导图像
            params: 滤波参数

        Returns:
            滤波后的图像
        """
        # 数据类型转换
        if len(input_img.shape) == 3:
            p = input_img.astype(np.float32) / 255.0
        else:
            p = input_img.astype(np.float32)
            if len(p.shape) == 2:
                p = p / 255.0 if p.max() > 1.0 else p

        if len(guide_img.shape) == 3:
            I = guide_img.astype(np.float32) / 255.0
        else:
            I = guide_img.astype(np.float32)
            if len(I.shape) == 2:
                I = I / 255.0 if I.max() > 1.0 else I

        # 处理多通道输入
        if len(p.shape) == 2:
            p = p[:, :, np.newaxis]

        if len(I.shape) == 2:
            I = I[:, :, np.newaxis]

        # 根据引导图像通道数选择处理方法
        if I.shape[2] == 1:
            return self._guided_filter_gray(p, I, params)
        else:
            return self._guided_filter_color(p, I, params)

    def _guided_filter_gray(self, p: np.ndarray, I: np.ndarray,
                           params: GuidedFilterParams) -> np.ndarray:
        """
        单通道引导图像的导向滤波

        Args:
            p: 输入图像（归一化后）
            I: 引导图像（归一化后）
            params: 滤波参数

        Returns:
            滤波后的图像
        """
        h, w = p.shape[:2]
        I = I[:, :, 0]  # 确保是2D

        # 计算局部均值
        mean_I = cv2.boxFilter(I, cv2.CV_32F, (2*params.radius+1, 2*params.radius+1))

        # 为每个通道处理
        results = []
        for c in range(p.shape[2]):
            p_channel = p[:, :, c]

            mean_p = cv2.boxFilter(p_channel, cv2.CV_32F, (2*params.radius+1, 2*params.radius+1))
            mean_Ip = cv2.boxFilter(I * p_channel, cv2.CV_32F, (2*params.radius+1, 2*params.radius+1))
            mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (2*params.radius+1, 2*params.radius+1))

            # 计算协方差和方差
            cov_Ip = mean_Ip - mean_I * mean_p
            var_I = mean_II - mean_I * mean_I

            # 计算线性系数
            a = cov_Ip / (var_I + params.eps)
            b = mean_p - a * mean_I

            # 对系数进行滤波
            mean_a = cv2.boxFilter(a, cv2.CV_32F, (2*params.radius+1, 2*params.radius+1))
            mean_b = cv2.boxFilter(b, cv2.CV_32F, (2*params.radius+1, 2*params.radius+1))

            # 生成输出
            q = mean_a * I + mean_b
            results.append(q)

        # 合并通道
        if len(results) == 1:
            output = results[0]
        else:
            output = np.stack(results, axis=2)

        # 确保输出格式
        if len(output.shape) == 2:
            output = output[:, :, np.newaxis]

        return np.clip(output, 0, 1)

    def _guided_filter_color(self, p: np.ndarray, I: np.ndarray,
                           params: GuidedFilterParams) -> np.ndarray:
        """
        多通道引导图像的导向滤波

        Args:
            p: 输入图像（归一化后）
            I: 引导图像（归一化后）
            params: 滤波参数

        Returns:
            滤波后的图像
        """
        h, w = p.shape[:2]

        # 分离引导图像通道
        I_r, I_g, I_b = I[:, :, 0], I[:, :, 1], I[:, :, 2]

        # 计算各通道均值
        ksize = (2*params.radius+1, 2*params.radius+1)
        mean_I_r = cv2.boxFilter(I_r, cv2.CV_32F, ksize)
        mean_I_g = cv2.boxFilter(I_g, cv2.CV_32F, ksize)
        mean_I_b = cv2.boxFilter(I_b, cv2.CV_32F, ksize)

        # 为每个输入通道处理
        results = []
        for c in range(p.shape[2]):
            p_channel = p[:, :, c]
            mean_p = cv2.boxFilter(p_channel, cv2.CV_32F, ksize)

            # 计算协方差
            mean_Ip_r = cv2.boxFilter(I_r * p_channel, cv2.CV_32F, ksize)
            mean_Ip_g = cv2.boxFilter(I_g * p_channel, cv2.CV_32F, ksize)
            mean_Ip_b = cv2.boxFilter(I_b * p_channel, cv2.CV_32F, ksize)

            cov_Ip_r = mean_Ip_r - mean_I_r * mean_p
            cov_Ip_g = mean_Ip_g - mean_I_g * mean_p
            cov_Ip_b = mean_Ip_b - mean_I_b * mean_p

            # 计算引导图像的协方差矩阵
            var_I_rr = cv2.boxFilter(I_r * I_r, cv2.CV_32F, ksize) - mean_I_r * mean_I_r
            var_I_rg = cv2.boxFilter(I_r * I_g, cv2.CV_32F, ksize) - mean_I_r * mean_I_g
            var_I_rb = cv2.boxFilter(I_r * I_b, cv2.CV_32F, ksize) - mean_I_r * mean_I_b
            var_I_gg = cv2.boxFilter(I_g * I_g, cv2.CV_32F, ksize) - mean_I_g * mean_I_g
            var_I_gb = cv2.boxFilter(I_g * I_b, cv2.CV_32F, ksize) - mean_I_g * mean_I_b
            var_I_bb = cv2.boxFilter(I_b * I_b, cv2.CV_32F, ksize) - mean_I_b * mean_I_b

            # 求解3x3线性方程组
            a_r, a_g, a_b, b = self._solve_3x3_system(
                var_I_rr, var_I_rg, var_I_rb, var_I_gg, var_I_gb, var_I_bb,
                cov_Ip_r, cov_Ip_g, cov_Ip_b, mean_I_r, mean_I_g, mean_I_b,
                mean_p, params.eps
            )

            # 对系数进行滤波
            mean_a_r = cv2.boxFilter(a_r, cv2.CV_32F, ksize)
            mean_a_g = cv2.boxFilter(a_g, cv2.CV_32F, ksize)
            mean_a_b = cv2.boxFilter(a_b, cv2.CV_32F, ksize)
            mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

            # 生成输出
            q = mean_a_r * I_r + mean_a_g * I_g + mean_a_b * I_b + mean_b
            results.append(q)

        # 合并通道
        if len(results) == 1:
            output = results[0]
        else:
            output = np.stack(results, axis=2)

        # 确保输出格式
        if len(output.shape) == 2:
            output = output[:, :, np.newaxis]

        return np.clip(output, 0, 1)

    def _solve_3x3_system(self, var_rr: np.ndarray, var_rg: np.ndarray, var_rb: np.ndarray,
                         var_gg: np.ndarray, var_gb: np.ndarray, var_bb: np.ndarray,
                         cov_r: np.ndarray, cov_g: np.ndarray, cov_b: np.ndarray,
                         mean_I_r: np.ndarray, mean_I_g: np.ndarray, mean_I_b: np.ndarray,
                         mean_p: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        求解3x3线性方程组

        Args:
            var_**: 协方差矩阵元素
            cov_*: 协方差向量元素
            mean_I_*: 引导图像均值
            mean_p: 输入图像均值
            eps: 正则化参数

        Returns:
            线性系数 (a_r, a_g, a_b, b)
        """
        h, w = var_rr.shape

        # 添加正则化项
        var_rr += eps
        var_gg += eps
        var_bb += eps

        # 逐像素求解3x3方程组
        a_r = np.zeros((h, w), dtype=np.float32)
        a_g = np.zeros((h, w), dtype=np.float32)
        a_b = np.zeros((h, w), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                # 构建3x3协方差矩阵
                sigma = np.array([
                    [var_rr[y, x], var_rg[y, x], var_rb[y, x]],
                    [var_rg[y, x], var_gg[y, x], var_gb[y, x]],
                    [var_rb[y, x], var_gb[y, x], var_bb[y, x]]
                ], dtype=np.float32)

                # 协方差向量
                cov_vec = np.array([cov_r[y, x], cov_g[y, x], cov_b[y, x]], dtype=np.float32)

                # 求解线性方程组
                try:
                    a_vec = np.linalg.solve(sigma, cov_vec)
                    a_r[y, x] = a_vec[0]
                    a_g[y, x] = a_vec[1]
                    a_b[y, x] = a_vec[2]
                except np.linalg.LinAlgError:
                    # 奇异矩阵处理
                    a_r[y, x] = a_g[y, x] = a_b[y, x] = 0.0

        # 计算偏置项
        b = mean_p - a_r * mean_I_r - a_g * mean_I_g - a_b * mean_I_b

        return a_r, a_g, a_b, b

    def _fast_guided_filter(self, input_img: np.ndarray, guide_img: np.ndarray,
                           params: GuidedFilterParams) -> np.ndarray:
        """
        快速导向滤波实现

        Args:
            input_img: 输入图像
            guide_img: 引导图像
            params: 滤波参数

        Returns:
            滤波后的图像
        """
        # 下采样
        scale = 1.0 / params.subsample
        h, w = input_img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)

        input_small = cv2.resize(input_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        guide_small = cv2.resize(guide_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 调整滤波半径
        small_params = GuidedFilterParams(
            radius=max(1, params.radius // params.subsample),
            eps=params.eps,
            subsample=1,
            use_fast_mode=False
        )

        # 在小尺寸图像上进行滤波
        output_small = self._basic_guided_filter(input_small, guide_small, small_params)

        # 上采样到原始尺寸
        output = cv2.resize(output_small.squeeze(), (w, h), interpolation=cv2.INTER_LINEAR)

        # 确保输出维度正确
        if len(input_img.shape) == 3 and len(output.shape) == 2:
            output = output[:, :, np.newaxis]

        return output

    def feather_blend(self, img1: np.ndarray, img2: np.ndarray,
                     mask: np.ndarray, radius: int = 16) -> np.ndarray:
        """
        使用导向滤波进行羽化融合

        Args:
            img1: 第一张图像
            img2: 第二张图像
            mask: 融合掩码
            radius: 羽化半径

        Returns:
            融合后的图像
        """
        # 确保掩码为浮点格式
        if mask.dtype != np.float32:
            mask = mask.astype(np.float32) / 255.0

        # 使用导向滤波平滑掩码
        smooth_params = GuidedFilterParams(radius=radius, eps=0.01)
        smooth_mask = self.guided_filter(mask, img1, smooth_params)

        # 确保掩码在[0,1]范围内
        smooth_mask = np.clip(smooth_mask, 0, 1)

        # 融合图像
        if len(smooth_mask.shape) == 2 and len(img1.shape) == 3:
            smooth_mask = smooth_mask[:, :, np.newaxis]

        blended = smooth_mask * img1 + (1 - smooth_mask) * img2

        return blended.astype(img1.dtype)

    def detail_enhancement(self, image: np.ndarray,
                          detail_strength: float = 1.5) -> np.ndarray:
        """
        使用导向滤波进行细节增强

        Args:
            image: 输入图像
            detail_strength: 细节增强强度

        Returns:
            增强后的图像
        """
        # 转换为浮点格式
        img_float = image.astype(np.float32) / 255.0

        # 使用大半径滤波获取基础层
        base_params = GuidedFilterParams(radius=16, eps=0.01)
        base_layer = self.guided_filter(img_float, img_float, base_params)

        # 计算细节层
        detail_layer = img_float - base_layer

        # 增强细节
        enhanced = base_layer + detail_strength * detail_layer

        # 转换回原始格式
        enhanced = np.clip(enhanced * 255, 0, 255).astype(image.dtype)

        return enhanced

    def edge_preserving_smoothing(self, image: np.ndarray,
                                 sigma_s: float = 50.0, sigma_r: float = 0.1) -> np.ndarray:
        """
        边缘保持平滑

        Args:
            image: 输入图像
            sigma_s: 空间权重
            sigma_r: 像素值权重

        Returns:
            平滑后的图像
        """
        # 转换为浮点格式
        img_float = image.astype(np.float32) / 255.0

        # 使用较小的eps值和适中的半径
        smooth_params = GuidedFilterParams(radius=int(sigma_s/10), eps=sigma_r)
        smoothed = self.guided_filter(img_float, img_float, smooth_params)

        # 转换回原始格式
        result = np.clip(smoothed * 255, 0, 255).astype(image.dtype)

        return result


def demo_basic_filtering():
    """演示基础导向滤波功能"""
    print("=== 导向滤波基础演示 ===")

    # 创建滤波器
    gf = GuidedFilter()

    # 创建测试图像
    test_image = create_test_image()

    print(f"原始图像形状: {test_image.shape}")

    # 基础滤波（自引导）
    print("执行基础导向滤波...")
    basic_params = GuidedFilterParams(radius=8, eps=0.01)
    filtered = gf.guided_filter(test_image, test_image, basic_params)

    # 快速滤波
    print("执行快速导向滤波...")
    fast_params = GuidedFilterParams(radius=8, eps=0.01, subsample=4, use_fast_mode=True)
    fast_filtered = gf.guided_filter(test_image, test_image, fast_params)

    # 细节增强
    print("执行细节增强...")
    enhanced = gf.detail_enhancement(test_image, detail_strength=1.5)

    print("演示完成！")

    return {
        'original': test_image,
        'basic_filtered': filtered,
        'fast_filtered': fast_filtered,
        'enhanced': enhanced
    }


def demo_advanced_applications():
    """演示高级应用功能"""
    print("=== 导向滤波高级应用演示 ===")

    gf = GuidedFilter()

    # 创建测试图像和掩码
    img1 = create_test_image()
    img2 = create_noisy_image(img1.shape)
    mask = create_test_mask(img1.shape[:2])

    print("执行羽化融合...")
    blended = gf.feather_blend(img1, img2, mask, radius=16)

    print("执行边缘保持平滑...")
    smoothed = gf.edge_preserving_smoothing(img1, sigma_s=50.0, sigma_r=0.1)

    print("高级应用演示完成！")

    return {
        'img1': img1,
        'img2': img2,
        'mask': mask,
        'blended': blended,
        'smoothed': smoothed
    }


def create_test_image() -> np.ndarray:
    """创建测试图像"""
    height, width = 400, 600
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # 创建渐变背景
    for y in range(height):
        for x in range(width):
            # 径向渐变
            center_x, center_y = width // 2, height // 2
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)

            intensity = 1.0 - (dist / max_dist)
            image[y, x] = [int(intensity * 255), int(intensity * 128), int(intensity * 64)]

    # 添加一些几何形状
    cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)
    cv2.circle(image, (400, 150), 50, (0, 255, 0), -1)
    cv2.ellipse(image, (500, 300), (80, 40), 45, 0, 360, (0, 0, 255), -1)

    # 添加噪声
    noise = np.random.normal(0, 25, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return image


def create_noisy_image(shape: Tuple[int, ...]) -> np.ndarray:
    """创建噪声图像"""
    noise_image = np.random.randint(0, 256, shape, dtype=np.uint8)
    return noise_image


def create_test_mask(shape: Tuple[int, int]) -> np.ndarray:
    """创建测试掩码"""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # 创建圆形掩码
    center_x, center_y = w // 2, h // 2
    radius = min(w, h) // 4

    y, x = np.ogrid[:h, :w]
    mask_area = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    mask[mask_area] = 255

    return mask


def performance_benchmark():
    """性能基准测试"""
    print("=== 性能基准测试 ===")

    # 测试不同参数配置
    configs = [
        GuidedFilterParams(radius=4, eps=0.01, use_fast_mode=False),
        GuidedFilterParams(radius=8, eps=0.01, use_fast_mode=False),
        GuidedFilterParams(radius=16, eps=0.01, use_fast_mode=False),
        GuidedFilterParams(radius=8, eps=0.01, subsample=2, use_fast_mode=True),
        GuidedFilterParams(radius=8, eps=0.01, subsample=4, use_fast_mode=True),
    ]

    # 创建测试图像
    test_sizes = [(240, 320), (480, 640), (720, 960)]

    for size in test_sizes:
        print(f"\n测试图像大小: {size[0]}x{size[1]}")
        test_image = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)

        for i, config in enumerate(configs):
            gf = GuidedFilter(config)

            # 预热
            _ = gf.guided_filter(test_image, test_image)

            # 计时测试
            start_time = time.time()
            num_iterations = 3

            for _ in range(num_iterations):
                result = gf.guided_filter(test_image, test_image)

            elapsed_time = time.time() - start_time
            avg_time = elapsed_time / num_iterations

            mode_str = "快速" if config.use_fast_mode else "标准"
            subsample_str = f", 下采样={config.subsample}" if config.use_fast_mode else ""
            print(f"  配置{i+1} ({mode_str}, 半径={config.radius}{subsample_str}): "
                 f"{avg_time*1000:.2f}ms")


if __name__ == "__main__":
    # 运行演示
    print("启动导向滤波演示程序\n")

    # 基础功能演示
    basic_results = demo_basic_filtering()
    print()

    # 高级应用演示
    advanced_results = demo_advanced_applications()
    print()

    # 性能基准测试
    performance_benchmark()

    print("\n所有演示完成！")