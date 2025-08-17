"""
实时自适应对比度增强算法 - Python实现
Real-time Adaptive Contrast Enhancement Algorithm

基于局部统计特征的自适应对比度增强技术，
通过分析像素邻域的均值和标准差动态调整增强强度。

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
class AdaptiveContrastParams:
    """实时自适应对比度增强参数配置"""
    window_size: int = 7                # 局部窗口大小
    clip_limit: float = 2.0             # 对比度限制阈值
    use_optimization: bool = True       # 是否启用性能优化
    border_mode: int = cv2.BORDER_REFLECT  # 边界处理模式
    epsilon: float = 1e-6               # 防除零常数


class RealTimeAdaptiveContrast:
    """实时自适应对比度增强处理类"""

    def __init__(self, params: Optional[AdaptiveContrastParams] = None):
        """
        初始化处理器

        Args:
            params: 算法参数配置
        """
        self.params = params or AdaptiveContrastParams()
        logger.info(f"初始化自适应对比度增强器，窗口大小: {self.params.window_size}, "
                   f"限制阈值: {self.params.clip_limit}")

    def enhance_image(self, image: np.ndarray,
                     params: Optional[AdaptiveContrastParams] = None) -> np.ndarray:
        """
        执行自适应对比度增强

        Args:
            image: 输入图像
            params: 可选的参数覆盖

        Returns:
            增强后的图像

        Raises:
            ValueError: 输入图像无效时抛出
        """
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")

        p = params or self.params

        # 转换为浮点型进行计算
        float_img = image.astype(np.float32)

        if len(image.shape) == 2:
            # 灰度图像处理
            return self._process_grayscale(float_img, p)
        elif len(image.shape) == 3:
            # 彩色图像处理
            return self._process_color(float_img, p)
        else:
            raise ValueError(f"不支持的图像格式，维度: {len(image.shape)}")

    def _process_grayscale(self, float_img: np.ndarray,
                          params: AdaptiveContrastParams) -> np.ndarray:
        """
        处理灰度图像

        Args:
            float_img: 浮点型灰度图像
            params: 处理参数

        Returns:
            处理后的图像
        """
        # 计算局部均值
        mean = cv2.boxFilter(float_img, -1,
                           (params.window_size, params.window_size),
                           borderType=params.border_mode)

        # 计算局部标准差
        squared_img = float_img * float_img
        mean_squared = cv2.boxFilter(squared_img, -1,
                                   (params.window_size, params.window_size),
                                   borderType=params.border_mode)

        variance = mean_squared - mean * mean
        stddev = np.sqrt(np.maximum(variance, 0))

        # 计算自适应增益
        gain = 1.0 + params.clip_limit * (stddev / (mean + params.epsilon))

        # 应用增益
        enhanced = float_img * gain

        return np.clip(enhanced, 0, 255).astype(np.uint8)

    def _process_color(self, float_img: np.ndarray,
                      params: AdaptiveContrastParams) -> np.ndarray:
        """
        处理彩色图像

        Args:
            float_img: 浮点型彩色图像
            params: 处理参数

        Returns:
            处理后的图像
        """
        if params.use_optimization:
            return self._process_color_optimized(float_img, params)
        else:
            return self._process_color_basic(float_img, params)

    def _process_color_basic(self, float_img: np.ndarray,
                           params: AdaptiveContrastParams) -> np.ndarray:
        """基础彩色图像处理"""
        result = np.zeros_like(float_img)

        # 分别处理每个通道
        for c in range(3):
            channel = float_img[:, :, c]
            result[:, :, c] = self._process_grayscale(channel, params)

        return result.astype(np.uint8)

    def _process_color_optimized(self, float_img: np.ndarray,
                               params: AdaptiveContrastParams) -> np.ndarray:
        """优化的彩色图像处理"""
        # 转换到亮度空间进行处理
        yuv = cv2.cvtColor(float_img.astype(np.uint8), cv2.COLOR_BGR2YUV)
        y_channel = yuv[:, :, 0].astype(np.float32)

        # 只对亮度通道进行增强
        enhanced_y = self._process_grayscale(y_channel, params)
        yuv[:, :, 0] = enhanced_y

        # 转换回BGR
        result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return result

    def multi_scale_enhance(self, image: np.ndarray,
                          window_sizes: List[int] = [5, 7, 9],
                          clip_limits: List[float] = [1.5, 2.0, 1.0]) -> np.ndarray:
        """
        多尺度自适应对比度增强

        Args:
            image: 输入图像
            window_sizes: 多个窗口大小
            clip_limits: 对应的限制阈值

        Returns:
            多尺度增强后的图像
        """
        if len(window_sizes) != len(clip_limits):
            raise ValueError("窗口大小和限制阈值数量必须相等")

        result = image.copy()

        for window_size, clip_limit in zip(window_sizes, clip_limits):
            params = AdaptiveContrastParams(
                window_size=window_size,
                clip_limit=clip_limit,
                use_optimization=self.params.use_optimization
            )
            result = self.enhance_image(result, params)

        return result

    def analyze_image_contrast(self, image: np.ndarray) -> Dict[str, float]:
        """
        分析图像对比度特征

        Args:
            image: 输入图像

        Returns:
            对比度分析结果
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 计算全局统计
        mean_val = np.mean(gray)
        std_val = np.std(gray)

        # 计算局部对比度变化
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        local_variance = np.var(laplacian)

        # 计算动态范围
        min_val, max_val = np.min(gray), np.max(gray)
        dynamic_range = max_val - min_val

        return {
            'global_mean': float(mean_val),
            'global_std': float(std_val),
            'local_variance': float(local_variance),
            'dynamic_range': float(dynamic_range),
            'contrast_ratio': float(std_val / (mean_val + 1e-6))
        }

    def adaptive_parameters(self, image: np.ndarray) -> AdaptiveContrastParams:
        """
        根据图像特征自动调整参数

        Args:
            image: 输入图像

        Returns:
            自适应调整的参数
        """
        analysis = self.analyze_image_contrast(image)

        # 根据图像特征调整参数
        if analysis['contrast_ratio'] < 0.3:
            # 低对比度图像
            window_size = 9
            clip_limit = 2.5
        elif analysis['contrast_ratio'] > 0.8:
            # 高对比度图像
            window_size = 5
            clip_limit = 1.5
        else:
            # 中等对比度图像
            window_size = 7
            clip_limit = 2.0

        return AdaptiveContrastParams(
            window_size=window_size,
            clip_limit=clip_limit,
            use_optimization=True
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


class RealTimeVideoProcessor:
    """实时视频处理器"""

    def __init__(self, enhancer: RealTimeAdaptiveContrast):
        """
        初始化视频处理器

        Args:
            enhancer: 对比度增强器实例
        """
        self.enhancer = enhancer
        self.frame_count = 0
        self.total_time = 0.0

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        处理单帧视频

        Args:
            frame: 输入帧

        Returns:
            处理后的帧
        """
        start_time = time.time()

        enhanced_frame = self.enhancer.enhance_image(frame)

        process_time = time.time() - start_time
        self.frame_count += 1
        self.total_time += process_time

        return enhanced_frame

    def get_fps_stats(self) -> Dict[str, float]:
        """
        获取处理性能统计

        Returns:
            性能统计信息
        """
        if self.frame_count == 0:
            return {'fps': 0.0, 'avg_time': 0.0}

        avg_time = self.total_time / self.frame_count
        fps = 1.0 / avg_time if avg_time > 0 else 0.0

        return {
            'fps': fps,
            'avg_time': avg_time,
            'total_frames': self.frame_count
        }


def demo_image_enhancement():
    """演示图像增强功能"""
    print("=== 实时自适应对比度增强演示 ===")

    # 创建增强器
    enhancer = RealTimeAdaptiveContrast()

    # 创建测试图像
    test_image = create_test_image()

    print(f"原始图像形状: {test_image.shape}")

    # 分析图像特征
    analysis = enhancer.analyze_image_contrast(test_image)
    print("图像对比度分析:")
    for key, value in analysis.items():
        print(f"  {key}: {value:.3f}")

    # 基础增强
    print("\n执行基础增强...")
    enhanced = enhancer.enhance_image(test_image)

    # 自适应参数增强
    print("执行自适应参数增强...")
    adaptive_params = enhancer.adaptive_parameters(test_image)
    adaptive_enhanced = enhancer.enhance_image(test_image, adaptive_params)

    # 多尺度增强
    print("执行多尺度增强...")
    multi_scale_enhanced = enhancer.multi_scale_enhance(test_image)

    print("演示完成！")

    return {
        'original': test_image,
        'basic_enhanced': enhanced,
        'adaptive_enhanced': adaptive_enhanced,
        'multi_scale_enhanced': multi_scale_enhanced
    }


def demo_video_processing():
    """演示视频处理功能"""
    print("=== 实时视频处理演示 ===")

    enhancer = RealTimeAdaptiveContrast(AdaptiveContrastParams(
        window_size=7,
        clip_limit=2.0,
        use_optimization=True
    ))

    processor = RealTimeVideoProcessor(enhancer)

    # 模拟视频帧处理
    num_frames = 30
    frame_width, frame_height = 640, 480

    for i in range(num_frames):
        # 创建模拟帧
        frame = np.random.randint(0, 256, (frame_height, frame_width, 3), dtype=np.uint8)

        # 处理帧
        enhanced_frame = processor.process_frame(frame)

        if (i + 1) % 10 == 0:
            stats = processor.get_fps_stats()
            print(f"已处理 {i+1} 帧, FPS: {stats['fps']:.2f}, "
                 f"平均处理时间: {stats['avg_time']*1000:.2f}ms")

    final_stats = processor.get_fps_stats()
    print(f"\n最终统计:")
    print(f"总帧数: {final_stats['total_frames']}")
    print(f"平均FPS: {final_stats['fps']:.2f}")
    print(f"平均处理时间: {final_stats['avg_time']*1000:.2f}ms")


def create_test_image() -> np.ndarray:
    """创建测试图像"""
    # 创建一个具有不同对比度区域的测试图像
    height, width = 480, 640
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # 低对比度区域
    image[:height//2, :width//2] = 128 + np.random.normal(0, 10, (height//2, width//2, 3))

    # 高对比度区域
    high_contrast = np.random.choice([50, 200], (height//2, width//2, 3))
    image[:height//2, width//2:] = high_contrast

    # 中等对比度区域
    image[height//2:, :width//2] = 100 + np.random.normal(0, 30, (height//2, width//2, 3))

    # 渐变区域
    for i in range(height//2):
        for j in range(width//2):
            value = int(255 * j / (width//2))
            image[height//2 + i, width//2 + j] = [value, value, value]

    return np.clip(image, 0, 255).astype(np.uint8)


def performance_benchmark():
    """性能基准测试"""
    print("=== 性能基准测试 ===")

    # 测试不同参数配置
    configs = [
        AdaptiveContrastParams(window_size=5, clip_limit=2.0, use_optimization=False),
        AdaptiveContrastParams(window_size=7, clip_limit=2.0, use_optimization=False),
        AdaptiveContrastParams(window_size=9, clip_limit=2.0, use_optimization=False),
        AdaptiveContrastParams(window_size=7, clip_limit=2.0, use_optimization=True),
    ]

    # 创建测试图像
    test_sizes = [(480, 640), (720, 1280), (1080, 1920)]

    for size in test_sizes:
        print(f"\n测试图像大小: {size[0]}x{size[1]}")
        test_image = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)

        for i, config in enumerate(configs):
            enhancer = RealTimeAdaptiveContrast(config)

            # 预热
            enhancer.enhance_image(test_image)

            # 计时测试
            start_time = time.time()
            num_iterations = 10

            for _ in range(num_iterations):
                enhanced = enhancer.enhance_image(test_image)

            elapsed_time = time.time() - start_time
            avg_time = elapsed_time / num_iterations
            fps = 1.0 / avg_time

            opt_str = "优化" if config.use_optimization else "基础"
            print(f"  配置{i+1} (窗口{config.window_size}, {opt_str}): "
                 f"{avg_time*1000:.2f}ms, {fps:.2f}FPS")


if __name__ == "__main__":
    # 运行演示
    print("启动实时自适应对比度增强演示程序\n")

    # 图像增强演示
    demo_results = demo_image_enhancement()
    print()

    # 视频处理演示
    demo_video_processing()
    print()

    # 性能基准测试
    performance_benchmark()

    print("\n所有演示完成！")